// ppu_hle_dispatch.h — Host-side HLE dispatcher for PPU imports.
//
// Workflow:
//   1. Caller builds a PpuHleDispatcher by scanning `.lib.stub` and
//      resolving every FNID through `ppu_hle_lookup()` (see
//      test_triangle_boot.cu for the reference construction).
//   2. Each cycle (or micro-burst) of the PPC megakernel, the host
//      reads PC and calls `dispatch(state, mem)`. If PC is at a known
//      trampoline slot, the matching handler runs on the host, updates
//      the guest state (usually just r3 = return value), and then
//      simulates a `blr` by setting PC = LR. Execution continues in the
//      real guest without needing any trap instruction patching.
//
// This is a deliberately minimal set of handlers sufficient to drive
// gs_gcm_basic_triangle.elf past CRT init; more handlers will be added
// as we crawl the boot path.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <utility>

#include "ppc_defs.h"
#include "ppu_hle_names.h"

extern "C" int megakernel_write_mem(uint64_t addr, const void* src, size_t n);

struct PpuHleDispatcher {
    struct Entry {
        uint32_t fnid;
        std::string mod;
        std::string name;
    };
    std::unordered_map<uint32_t, Entry> byPc;   // trampoline PC → entry
    std::unordered_map<std::string, uint64_t> unhandledHistogram;
    std::unordered_map<std::string, uint64_t> handledHistogram;
    uint64_t callCount = 0;
    uint64_t unknownCount = 0;

    // Cooperative globals updated by some handlers.
    uint32_t tlsBaseAddr = 0x00E00000;   // dummy TLS heap base
    uint32_t tlsNext     = 0x00E00000;
    uint64_t virtTime    = 0;

    // Libc heap scratch arena (used when we HLE-patch the ELF's
    // internal allocator; address grows upward).
    uint32_t libcHeapBase = 0x00900000;
    uint32_t libcHeapNext = 0x00900000;
    uint32_t libcHeapEnd  = 0x08000000;  // 119MB arena (0x900000..0x8000000)

    // GCM shared-memory-ish regions. Real PS3 maps these to RSX IO
    // memory; we just use guest-visible scratch addresses.
    uint32_t gcmControlRegPtr = 0x00D00000;  // CellGcmControl (put/get/ref)
    uint32_t gcmContextAddr   = 0x00D00100;  // CellGcmContextData { begin,end,current,callback }
    uint32_t gcmLabelBase     = 0x00D01000;  // Flip status labels
    bool     gcmControlInited = false;
    // The IO buffer the game passed to cellGcmInitBody. [ioBase, ioBase+ioSize)
    // is the FIFO buffer region; cellGcmContextData::current grows from
    // ioBase+0x1000 (4KB reserved at the head) toward ioBase+cmdSize.
    uint32_t gcmIoBase   = 0;
    uint32_t gcmIoSize   = 0;
    uint32_t gcmCmdSize  = 0;
    uint32_t gcmFifoBegin = 0;
    uint32_t gcmFifoEnd   = 0;

    // Number of times the guest asked to present (cellGcmSetFlipCommand).
    // Consumers (tests / raster bridge) should read+clear this after
    // each FIFO drain to drive onFlip emission.
    uint32_t gcmFlipRequests = 0;
    uint32_t gcmLastFlipBuffer = 0;   // buffer id as passed by caller

    // Extra PC-indexed handlers for non-import functions we want to
    // short-circuit (e.g. the ELF's internal malloc/free).
    enum class Builtin { Malloc, Free, Memalign, Realloc, MmapAlloc, None };
    std::unordered_map<uint32_t, Builtin> builtinByPc;

    void addBuiltinMalloc(uint32_t pc)     { builtinByPc[pc] = Builtin::Malloc;     }
    void addBuiltinFree(uint32_t pc)       { builtinByPc[pc] = Builtin::Free;       }
    void addBuiltinMemalign(uint32_t pc)   { builtinByPc[pc] = Builtin::Memalign;   }
    void addBuiltinRealloc(uint32_t pc)    { builtinByPc[pc] = Builtin::Realloc;    }
    void addBuiltinMmapAlloc(uint32_t pc)  { builtinByPc[pc] = Builtin::MmapAlloc;  }

    void add(uint32_t pc, uint32_t fnid,
             const char* mod, const char* name) {
        byPc[pc] = { fnid, mod, name };
    }

    // Called by the host loop when it observes the guest PC just
    // entered a stub trampoline.  Mutates `state` (r3, lr, pc).
    // Returns the resolved name (for logging) or nullptr if unknown.
    // `halted_out` is set if the HLE asked the VM to stop.
    const char* dispatch(ppc::PPEState& st,
                         uint8_t* mem, size_t memSize,
                         bool& halted_out) {
        halted_out = false;
        uint32_t pc = (uint32_t)st.pc;
        // Check builtin PC patches first (ELF-internal malloc/free).
        {
            auto bit = builtinByPc.find(pc);
            if (bit != builtinByPc.end()) {
                callCount++;
                if (bit->second == Builtin::Memalign) {
                    // r3 = alignment, r4 = size
                    uint64_t align = st.gpr[3];
                    uint64_t size  = st.gpr[4];
                    if (align < 16) align = 16;
                    if (size == 0) size = 0x100;
                    if (size > 0x4000000) size = 0x4000000; // cap 64MB
                    libcHeapNext = (libcHeapNext + (uint32_t)align - 1) & ~((uint32_t)align - 1);
                    uint32_t ptr = libcHeapNext;
                    if (libcHeapNext + size <= libcHeapEnd) {
                        libcHeapNext += (uint32_t)size;
                    } else {
                        ptr = 0; // oom
                    }
                    if (ptr && ptr + size <= memSize) {
                        std::memset(mem + ptr, 0, (size_t)size);
                        megakernel_write_mem(ptr, mem + ptr, (size_t)size);
                    }
                    st.gpr[3] = ptr;
                } else if (bit->second == Builtin::Malloc) {
                    // r3 = size
                    uint64_t size = st.gpr[3];
                    if (size == 0) size = 0x100;
                    if (size > 0x4000000) size = 0x4000000; // cap 64MB
                    uint32_t align = 16;
                    libcHeapNext = (libcHeapNext + align - 1) & ~(align - 1);
                    uint32_t ptr = libcHeapNext;
                    if (libcHeapNext + size <= libcHeapEnd) {
                        libcHeapNext += (uint32_t)size;
                    } else {
                        ptr = 0; // oom
                    }
                    if (ptr && ptr + size <= memSize) {
                        std::memset(mem + ptr, 0, (size_t)size);
                        megakernel_write_mem(ptr, mem + ptr, (size_t)size);
                    }
                    st.gpr[3] = ptr;
                } else if (bit->second == Builtin::Realloc) {
                    // r3 = old_ptr (ignored), r4 = new_size
                    uint64_t size = st.gpr[4];
                    if (size == 0) size = 0x100;
                    if (size > 0x4000000) size = 0x4000000;
                    uint32_t align = 16;
                    libcHeapNext = (libcHeapNext + align - 1) & ~(align - 1);
                    uint32_t ptr = libcHeapNext;
                    if (libcHeapNext + size <= libcHeapEnd) {
                        libcHeapNext += (uint32_t)size;
                    } else {
                        ptr = 0;
                    }
                    if (ptr && ptr + size <= memSize) {
                        std::memset(mem + ptr, 0, (size_t)size);
                        megakernel_write_mem(ptr, mem + ptr, (size_t)size);
                    }
                    st.gpr[3] = ptr;
                } else if (bit->second == Builtin::MmapAlloc) {
                    // r3 = size, r4 = alignment (mmap-style allocation)
                    uint64_t size  = st.gpr[3];
                    uint64_t align = st.gpr[4];
                    if (align < 16) align = 16;
                    if (size == 0) size = 0x100;
                    if (size > 0x4000000) size = 0x4000000;
                    libcHeapNext = (libcHeapNext + (uint32_t)align - 1) & ~((uint32_t)align - 1);
                    uint32_t ptr = libcHeapNext;
                    if (libcHeapNext + size <= libcHeapEnd) {
                        libcHeapNext += (uint32_t)size;
                    } else {
                        ptr = 0;
                    }
                    if (ptr && ptr + size <= memSize) {
                        std::memset(mem + ptr, 0, (size_t)size);
                        megakernel_write_mem(ptr, mem + ptr, (size_t)size);
                    }
                    st.gpr[3] = ptr;
                } else {
                    st.gpr[3] = 0;
                }
                st.pc = st.lr;
                return "libc_malloc_hle";
            }
        }
        auto it = byPc.find(pc);
        if (it == byPc.end()) return nullptr;
        const Entry& e = it->second;
        callCount++;
        handledHistogram[e.name]++;
        uint32_t retval = 0;

        // --- Minimal handler set ---
        if (e.name == "sys_initialize_tls") {
            retval = 0;
        } else if (e.name == "sys_process_exit") {
            halted_out = true;
            retval = 0;
        } else if (e.name == "_sys_process_atexitspawn" ||
                   e.name == "_sys_process_at_Exitspawn") {
            // Registration of exit handlers during CRT init — not an
            // actual exit.
            retval = 0;
        } else if (e.name == "sys_time_get_system_time") {
            virtTime += 16667;  // ~60fps frame time in µs
            retval = virtTime;
        } else if (e.name == "_sys_memset") {
            uint32_t dst = (uint32_t)st.gpr[3];
            uint32_t val = (uint32_t)st.gpr[4];
            uint32_t len = (uint32_t)st.gpr[5];
            if (dst + len <= memSize) std::memset(mem + dst, val, len);
            retval = dst;
        } else if (e.name == "_sys_memcpy") {
            uint32_t dst = (uint32_t)st.gpr[3];
            uint32_t src = (uint32_t)st.gpr[4];
            uint32_t len = (uint32_t)st.gpr[5];
            if (dst + len <= memSize && src + len <= memSize)
                std::memmove(mem + dst, mem + src, len);
            retval = dst;
        } else if (e.name == "sys_lwmutex_create" ||
                   e.name == "sys_lwmutex_lock" ||
                   e.name == "sys_lwmutex_unlock" ||
                   e.name == "sys_lwmutex_destroy") {
            retval = 0;
        } else if (e.name == "sys_ppu_thread_get_id") {
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp + 8 <= memSize) {
                // big-endian u64 1
                std::memset(mem + outp, 0, 8);
                mem[outp + 7] = 1;
            }
            retval = 0;
        } else if (e.name == "sys_prx_register_library") {
            retval = 0;
        } else if (e.name == "cellGcmAddressToOffset") {
            // r3 = effective address, r4 = out pointer to u32 offset.
            // Real HW: subtracts the RSX IO map base. We just pass the
            // address through — our "VRAM" is flat at the same EA.
            uint32_t addr = (uint32_t)st.gpr[3];
            uint32_t outp = (uint32_t)st.gpr[4];
            if (outp + 4 <= memSize) {
                uint8_t be[4] = {
                    (uint8_t)(addr >> 24), (uint8_t)(addr >> 16),
                    (uint8_t)(addr >> 8),  (uint8_t)addr
                };
                std::memcpy(mem + outp, be, 4);
                megakernel_write_mem((uint64_t)outp, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellGcmGetControlRegister") {
            // Returns pointer to CellGcmControl { u32 put, get, ref }.
            if (!gcmControlInited && gcmControlRegPtr + 12 <= memSize) {
                std::memset(mem + gcmControlRegPtr, 0, 12);
                megakernel_write_mem((uint64_t)gcmControlRegPtr,
                                     mem + gcmControlRegPtr, 12);
                gcmControlInited = true;
            }
            retval = gcmControlRegPtr;
        } else if (e.name == "cellGcmGetFlipStatus") {
            // 0 => "last flip completed". Loop waits for this before
            // submitting the next frame. Always return 0 so we don't
            // spin forever.
            retval = 0;
        } else if (e.name == "cellGcmResetFlipStatus") {
            retval = 0;
        } else if (e.name == "_cellGcmSetFlipCommand" ||
                   e.name == "cellGcmSetFlipCommand") {
            // Real HW writes a NV flip command into the FIFO. We don't
            // synthesise that packet (the FIFO replay happens separately),
            // but we do record the request so the host driver can emit an
            // onFlip barrier to the raster bridge after replay. r4 is the
            // buffer id; leave it for the consumer.
            gcmLastFlipBuffer = (uint32_t)st.gpr[4];
            gcmFlipRequests++;
            retval = 0;
        } else if (e.name == "_cellGcmInitBody" ||
                   e.name == "cellGcmInit") {
            // Per rpcs3 cellGcmSys.cpp `_cellGcmInitBody`:
            //   r3 = pptr<CellGcmContextData>   (out: *r3 := context_addr)
            //   r4 = cmdSize
            //   r5 = ioSize
            //   r6 = ioAddress (EA of guest's FIFO buffer)
            //
            // We populate a CellGcmContextData { begin, end, current,
            // callback } at `gcmContextAddr`, with begin = ioAddress +
            // 4096 (first 4KB reserved), end = ioAddress + 32*1024 - 4,
            // current = begin, callback = 0. Then write *r3 =
            // gcmContextAddr (BE u32) in both host and GPU mem.
            uint32_t pPtr    = (uint32_t)st.gpr[3];
            uint32_t cmdSize = (uint32_t)st.gpr[4];
            uint32_t ioSize  = (uint32_t)st.gpr[5];
            uint32_t ioAddr  = (uint32_t)st.gpr[6];
            std::printf("  [HLE] _cellGcmInitBody(pPtr=0x%x cmdSize=0x%x "
                        "ioSize=0x%x ioAddr=0x%x)\n",
                        pPtr, cmdSize, ioSize, ioAddr);
            // If caller did not supply an io buffer (ioAddr==0) they rely
            // on the runtime to allocate one. Provide a scratch region.
            if (ioAddr == 0) {
                ioAddr = 0x00F80000;         // scratch IO region
                if (ioSize == 0) ioSize = 0x100000;   // 1 MB scratch
                // Force a FIFO as big as the whole IO region so that the
                // single-run capture never wraps — we don't implement JUMP-
                // based ring walking, so wrap would stomp the initial
                // SURFACE/VIEWPORT setup.
                cmdSize = ioSize;
                std::printf("  [HLE] _cellGcmInitBody: synth ioAddr=0x%x ioSize=0x%x cmdSize=0x%x\n",
                            ioAddr, ioSize, cmdSize);
            }
            // Use program's cmdSize for FIFO. A 64KB FIFO captures one full
            // frame (~97 draw calls for the cube). The program's inline GCM
            // code checks current < end, so end must match expectations.
            gcmCmdSize = cmdSize ? cmdSize : 0x8000;   // 32KB default
            gcmIoSize  = ioSize;
            gcmIoBase  = ioAddr;
            gcmFifoBegin = ioAddr + 0x1000;
            gcmFifoEnd   = ioAddr + gcmCmdSize - 4;

            auto put_be32 = [&](uint32_t addr, uint32_t v) {
                if (addr + 4 > memSize) return;
                uint8_t b[4] = {
                    (uint8_t)(v >> 24), (uint8_t)(v >> 16),
                    (uint8_t)(v >> 8),  (uint8_t)v
                };
                std::memcpy(mem + addr, b, 4);
                megakernel_write_mem((uint64_t)addr, b, 4);
            };
            // CellGcmContextData layout (16 bytes): begin, end, current, callback.
            put_be32(gcmContextAddr + 0,  gcmFifoBegin);
            put_be32(gcmContextAddr + 4,  gcmFifoEnd);
            put_be32(gcmContextAddr + 8,  gcmFifoBegin);
            put_be32(gcmContextAddr + 12, 0);
            // *r3 = gcmContextAddr
            if (pPtr) put_be32(pPtr, gcmContextAddr);
            // Seed CellGcmControl { put, get, ref } = { 0, 0, 0 }.
            if (!gcmControlInited && gcmControlRegPtr + 12 <= memSize) {
                std::memset(mem + gcmControlRegPtr, 0, 12);
                megakernel_write_mem((uint64_t)gcmControlRegPtr,
                                     mem + gcmControlRegPtr, 12);
                gcmControlInited = true;
            }
            retval = 0;
        } else if (e.name == "cellGcmSetDisplayBuffer" ||
                   e.name == "cellGcmSetFlipMode" ||
                   e.name == "cellGcmSetDebugOutputLevel" ||
                   e.name == "cellGcmGetTiledPitchSize" ||
                   e.name == "cellGcmBindTile" ||
                   e.name == "cellGcmUnbindTile" ||
                   e.name == "cellGcmSetTileInfo" ||
                   e.name == "cellGcmBindZcull" ||
                   e.name == "cellGcmUnbindZcull") {
            retval = 0;
        } else if (e.name == "cellGcmGetDefaultCommandWordSize") {
            // Returns default command buffer size in words (0x400 = 1024 words = 4KB)
            retval = 0x400;
        } else if (e.name == "cellGcmGetDefaultSegmentWordSize") {
            // Returns default segment size in words (0x100 = 256 words = 1KB)
            retval = 0x100;
        } else if (e.name == "cellGcmSetDefaultFifoSize") {
            // r3 = cmdSize, r4 = ioSize — informational, we ignore
            retval = 0;
        } else if (e.name == "_cellGcmFunc15") {
            // Internal init helper (sets up context default state). No-op.
            retval = 0;
        } else if (e.name == "cellGcmSetWaitFlip") {
            // Inserts a wait-for-flip-complete into FIFO. Since we process
            // FIFO synchronously, this is a no-op.
            retval = 0;
        } else if (e.name == "cellGcmSetPrepareFlip") {
            // Same as SetFlipCommand — queue a flip request.
            gcmLastFlipBuffer = (uint32_t)st.gpr[4];
            gcmFlipRequests++;
            retval = 0;
        } else if (e.name == "cellGcmSetFlipHandler") {
            // r3 = callback function pointer. We don't invoke it; store for future use.
            retval = 0;
        } else if (e.name == "cellGcmGetCurrentField") {
            // Returns 0 (progressive scan — no interlaced field).
            retval = 0;
        } else if (e.name == "cellGcmGetConfiguration") {
            // r3 = *CellGcmConfig { u32 localAddr, ioAddr, localSize, ioSize, memFreq, coreFreq }
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp && outp + 24 <= memSize) {
                auto put_be32 = [&](uint32_t addr, uint32_t v) {
                    if (addr + 4 > memSize) return;
                    mem[addr+0] = (uint8_t)(v >> 24);
                    mem[addr+1] = (uint8_t)(v >> 16);
                    mem[addr+2] = (uint8_t)(v >> 8);
                    mem[addr+3] = (uint8_t)v;
                    megakernel_write_mem((uint64_t)addr, mem + addr, 4);
                };
                put_be32(outp + 0,  0x00C00000);   // localAddress (VRAM base)
                put_be32(outp + 4,  gcmIoBase ? gcmIoBase : 0x00F80000); // ioAddress
                put_be32(outp + 8,  256u*1024u*1024u); // localSize (256MB)
                put_be32(outp + 12, gcmIoSize ? gcmIoSize : 0x1000000); // ioSize
                put_be32(outp + 16, 650000000u);   // memoryFrequency (650MHz)
                put_be32(outp + 20, 500000000u);   // coreFrequency (500MHz)
            }
            retval = 0;
        } else if (e.name == "cellGcmGetLabelAddress") {
            // r3 = label index. Returns pointer to a u32 "label" that
            // the RSX writes to signal flip completion, etc.
            uint32_t idx = (uint32_t)st.gpr[3];
            uint32_t addr = gcmLabelBase + idx * 16;
            if (addr + 4 <= memSize) {
                std::memset(mem + addr, 0, 4);
                megakernel_write_mem((uint64_t)addr, mem + addr, 4);
            }
            retval = addr;
        } else if (e.name == "cellGcmGetTimeStamp") {
            retval = 0;
        } else if (e.name == "cellGcmMapMainMemory") {
            // r3 = EA, r4 = size, r5 = *offset_out.
            // Map main memory into RSX-visible IO space. We use identity
            // mapping (offset == EA), same as cellGcmAddressToOffset.
            uint32_t ea   = (uint32_t)st.gpr[3];
            uint32_t outp = (uint32_t)st.gpr[5];
            if (outp + 4 <= memSize) {
                uint8_t be[4] = {
                    (uint8_t)(ea >> 24), (uint8_t)(ea >> 16),
                    (uint8_t)(ea >> 8),  (uint8_t)ea
                };
                std::memcpy(mem + outp, be, 4);
                megakernel_write_mem((uint64_t)outp, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellPadInit" ||
                   e.name == "cellPadEnd") {
            retval = 0;
        } else if (e.name == "cellPadGetInfo2") {
            // r3 = *CellPadInfo2. Zero it (no pads connected).
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp + 64 <= memSize) {
                std::memset(mem + outp, 0, 64);
                megakernel_write_mem((uint64_t)outp, mem + outp, 64);
            }
            retval = 0;
        } else if (e.name == "cellPadGetData") {
            // r3 = port, r4 = *CellPadData. Zero it (no buttons).
            uint32_t outp = (uint32_t)st.gpr[4];
            if (outp + 64 <= memSize) {
                std::memset(mem + outp, 0, 64);
                megakernel_write_mem((uint64_t)outp, mem + outp, 64);
            }
            retval = 0;
        } else if (e.name == "_sys_strlen") {
            // r3 = pointer to string, returns length.
            uint32_t ptr = (uint32_t)st.gpr[3];
            uint32_t len = 0;
            if (ptr < memSize) {
                while (ptr + len < memSize && mem[ptr + len] != 0 && len < 0x10000)
                    len++;
            }
            retval = len;
        } else if (e.name == "_sys_printf") {
            // r3 = format string pointer. Just consume it, no real output.
            retval = 0;
        } else if (e.name == "_sys_heap_create_heap" ||
                   e.name == "_sys_heap_delete_heap") {
            // Return a fake heap handle (non-zero for create).
            retval = (e.name == "_sys_heap_create_heap") ? 0x00880000 : 0;
        } else if (e.name == "_sys_heap_malloc") {
            // r3 = heap handle, r4 = size.
            uint64_t size = st.gpr[4];
            if (size == 0 || size > 0x1000000) size = 0x100;
            uint32_t align = 16;
            libcHeapNext = (libcHeapNext + align - 1) & ~(align - 1);
            uint32_t ptr = libcHeapNext;
            if (libcHeapNext + size <= libcHeapEnd) {
                libcHeapNext += (uint32_t)size;
            } else {
                ptr = 0;
            }
            if (ptr && ptr + size <= memSize) {
                std::memset(mem + ptr, 0, (size_t)size);
                megakernel_write_mem(ptr, mem + ptr, (size_t)size);
            }
            retval = ptr;
        } else if (e.name == "_sys_heap_free") {
            retval = 0;
        } else if (e.name == "sys_game_process_exitspawn2") {
            halted_out = true;
            retval = 0;
        } else if (e.name == "cellVideoOutConfigure" ||
                   e.name == "cellVideoOutGetResolution" ||
                   e.name == "cellVideoOutGetState") {
            if (e.name == "cellVideoOutGetState") {
                // r3=videoOut, r4=deviceIndex, r5=*CellVideoOutState
                uint32_t outp = (uint32_t)st.gpr[5];
                if (outp && outp + 16 <= memSize) {
                    std::memset(mem + outp, 0, 16);
                    // CellVideoOutState { u8 state, u8 colorSpace, u8[6], CellVideoOutDisplayMode }
                    // state=1 (ENABLED), colorSpace=1 (RGB)
                    mem[outp + 0] = 1;  // CELL_VIDEO_OUT_OUTPUT_STATE_ENABLED
                    mem[outp + 1] = 1;  // CELL_VIDEO_OUT_COLOR_SPACE_RGB
                    // CellVideoOutDisplayMode @ offset 8:
                    // { u8 resId, u8 scanMode, u8 conversion, u8 aspect, u8[2], be16 refreshRates }
                    mem[outp + 8]  = 2;  // CELL_VIDEO_OUT_RESOLUTION_720
                    mem[outp + 9]  = 1;  // CELL_VIDEO_OUT_SCAN_MODE_PROGRESSIVE
                    mem[outp + 10] = 0;  // CELL_VIDEO_OUT_DISPLAY_CONVERSION_NONE
                    mem[outp + 11] = 0;  // aspect=0 (AUTO)
                    mem[outp + 14] = 0;  // refreshRates BE16 = 0x0001 (59.94Hz)
                    mem[outp + 15] = 1;
                    megakernel_write_mem((uint64_t)outp, mem + outp, 16);
                }
            } else if (e.name == "cellVideoOutGetResolution") {
                // r3=resolutionId, r4=*CellVideoOutResolution { be16 width, be16 height }
                uint32_t outp = (uint32_t)st.gpr[4];
                if (outp && outp + 4 <= memSize) {
                    // Return 1280x720 (HD)
                    mem[outp + 0] = 0x05; mem[outp + 1] = 0x00; // BE16 1280
                    mem[outp + 2] = 0x02; mem[outp + 3] = 0xD0; // BE16 720
                    megakernel_write_mem((uint64_t)outp, mem + outp, 4);
                }
            } else {
                // cellVideoOutConfigure — just return success
                uint32_t outp = (uint32_t)st.gpr[5];
                if (outp && outp + 64 <= memSize) {
                    std::memset(mem + outp, 0, 64);
                    megakernel_write_mem((uint64_t)outp, mem + outp, 64);
                }
            }
            retval = 0;
        } else if (e.name == "cellSysutilRegisterCallback" ||
                   e.name == "cellSysutilUnregisterCallback" ||
                   e.name == "cellSysutilCheckCallback") {
            retval = 0;
        } else if (e.name == "cellSysutilGetSystemParamInt") {
            // r3 = paramId, r4 = *value_out
            // Key params: LANG=0x0111(English=1), ENTER_BUTTON=0x0105(Circle=0,Cross=1)
            uint32_t paramId = (uint32_t)st.gpr[3];
            uint32_t outp    = (uint32_t)st.gpr[4];
            uint32_t val = 0;
            switch (paramId) {
            case 0x0111: val = 1; break;  // CELL_SYSUTIL_SYSTEMPARAM_ID_LANG → English
            case 0x0105: val = 1; break;  // ENTER_BUTTON_ASSIGN → Cross (western)
            default: val = 0; break;
            }
            if (outp + 4 <= memSize) {
                uint8_t be[4] = {
                    (uint8_t)(val >> 24), (uint8_t)(val >> 16),
                    (uint8_t)(val >> 8),  (uint8_t)val
                };
                std::memcpy(mem + outp, be, 4);
                megakernel_write_mem((uint64_t)outp, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellVideoOutGetResolutionAvailability") {
            // r3 = videoOut, r4 = resolutionId, r5 = aspect, r6 = option
            // Return 1 (available) for 1080p (resId=2) and 720p (resId=4)
            uint32_t resId = (uint32_t)st.gpr[4];
            retval = (resId == 2 || resId == 4 || resId == 1) ? 1 : 0;
        } else if (e.name == "cellFsOpen") {
            // r3=path, r4=flags, r5=fd_out, ...
            uint32_t fdOut = (uint32_t)st.gpr[5];
            if (fdOut + 4 <= memSize) {
                uint8_t be[4] = { 0, 0, 0, 3 };  // arbitrary fd #3
                std::memcpy(mem + fdOut, be, 4);
                megakernel_write_mem((uint64_t)fdOut, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellFsWrite") {
            // r3=fd, r4=buf, r5=nbytes, r6=nwritten_out
            uint32_t nbytes   = (uint32_t)st.gpr[5];
            uint32_t nwritten = (uint32_t)st.gpr[6];
            if (nwritten && nwritten + 4 <= memSize) {
                uint8_t be[4] = {
                    (uint8_t)(nbytes >> 24), (uint8_t)(nbytes >> 16),
                    (uint8_t)(nbytes >> 8),  (uint8_t)nbytes
                };
                std::memcpy(mem + nwritten, be, 4);
                megakernel_write_mem((uint64_t)nwritten, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellFsRead") {
            uint32_t nread = (uint32_t)st.gpr[6];
            if (nread && nread + 4 <= memSize) {
                std::memset(mem + nread, 0, 4);
                megakernel_write_mem((uint64_t)nread, mem + nread, 4);
            }
            retval = 0;
        } else if (e.name == "cellFsClose" ||
                   e.name == "cellFsLseek") {
            retval = 0;
        } else {
            // No handler yet — acknowledge, log, continue with r3=0.
            unknownCount++;
            unhandledHistogram[e.name]++;
            retval = 0;
        }

        st.gpr[3] = retval;
        st.pc     = st.lr;    // blr
        // PPC64 ELFv1: real PRX stubs save caller's r2 to [r1+0x28]
        // before TOC switch, and caller issues `ld r2, 0x28(r1)` right
        // after the bl to restore. Since we skip the real stub code,
        // emulate that save so r2 survives the HLE call.
        {
            uint32_t sp = (uint32_t)st.gpr[1];
            uint8_t be[8];
            uint64_t v = st.gpr[2];
            for (int i = 0; i < 8; ++i) be[i] = (uint8_t)(v >> (56 - 8*i));
            if (sp + 0x30 <= memSize) {
                std::memcpy(mem + sp + 0x28, be, 8);
                megakernel_write_mem((uint64_t)sp + 0x28, be, 8);
            }
        }
        return e.name.c_str();
    }

    void print_summary() const {
        std::printf("  HLE dispatcher: %llu calls (%llu unhandled)\n",
                    (unsigned long long)callCount,
                    (unsigned long long)unknownCount);
        {
            std::vector<std::pair<std::string, uint64_t>> sorted(
                handledHistogram.begin(), handledHistogram.end());
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            std::printf("  top handled imports:\n");
            int printed = 0;
            for (const auto& p : sorted) {
                std::printf("    %-40s %llu\n",
                            p.first.c_str(), (unsigned long long)p.second);
                if (++printed >= 20) break;
            }
            bool gotInitBody = handledHistogram.count("_cellGcmInitBody") > 0;
            std::printf("  _cellGcmInitBody fired: %s\n", gotInitBody ? "YES" : "NO");
        }
        std::vector<std::pair<std::string, uint64_t>> sorted(
            unhandledHistogram.begin(), unhandledHistogram.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        int printed = 0;
        for (const auto& p : sorted) {
            std::printf("    unhandled: %-40s %llu\n",
                        p.first.c_str(), (unsigned long long)p.second);
            if (++printed >= 20) break;
        }
    }
};
