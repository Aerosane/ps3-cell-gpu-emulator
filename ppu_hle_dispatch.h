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
    uint32_t nextHandleId = 0x1000;      // monotonic handle allocator

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

        // ── cellSpurs — SPURS task scheduler stubs ──────────────
        // These return CELL_OK (0) to let games that initialize SPURS
        // proceed past boot. Actual SPU task dispatch is handled by
        // the SPU JIT turbo/hyper multi-instance engine.
        } else if (e.name == "cellSpursInitialize" ||
                   e.name == "cellSpursInitializeWithAttribute" ||
                   e.name == "cellSpursInitializeWithAttribute2") {
            // r3 = CellSpurs* spurs, r4 = numSpu / CellSpursAttribute*
            // Zero-fill the SPURS structure so games see initialized state
            uint32_t spursPtr = (uint32_t)st.gpr[3];
            if (spursPtr && spursPtr + 0x80 <= memSize) {
                std::memset(mem + spursPtr, 0, 0x80);
                megakernel_write_mem(spursPtr, mem + spursPtr, 0x80);
            }
            retval = 0;
        } else if (e.name == "cellSpursFinalize") {
            retval = 0;
        } else if (e.name == "cellSpursAttributeInitialize") {
            // r3 = CellSpursAttribute* attr, r4 = revision, r5 = sdkVersion,
            // r6 = nSpus, r7 = spuPriority, r8 = ppuPriority
            uint32_t attrPtr = (uint32_t)st.gpr[3];
            if (attrPtr && attrPtr + 0x80 <= memSize) {
                std::memset(mem + attrPtr, 0, 0x80);
                megakernel_write_mem(attrPtr, mem + attrPtr, 0x80);
            }
            retval = 0;
        } else if (e.name == "cellSpursAttributeSetNamePrefix" ||
                   e.name == "cellSpursAttributeSetSpuThreadGroupType" ||
                   e.name == "cellSpursAttributeEnableSpuPrintfIfAvailable" ||
                   e.name == "cellSpursSetMaxContention" ||
                   e.name == "cellSpursSetPriorities" ||
                   e.name == "cellSpursSetGlobalExceptionEventHandler") {
            retval = 0;
        } else if (e.name == "cellSpursGetNumSpuThread") {
            // r3 = CellSpurs*, r4 = uint32_t* numSpu
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t n = 6;  // report 6 SPUs (standard PS3 config)
                uint8_t be[4] = { (uint8_t)(n>>24), (uint8_t)(n>>16),
                                  (uint8_t)(n>>8),  (uint8_t)n };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellSpursGetSpuThreadGroupId") {
            // r3 = CellSpurs*, r4 = sys_spu_thread_group_t* id
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t id = 1;  // dummy group ID
                uint8_t be[4] = { (uint8_t)(id>>24), (uint8_t)(id>>16),
                                  (uint8_t)(id>>8),  (uint8_t)id };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellSpursAttachLv2EventQueue" ||
                   e.name == "cellSpursDetachLv2EventQueue") {
            retval = 0;
        } else if (e.name == "cellSpursCreateTaskset" ||
                   e.name == "cellSpursCreateTasksetWithAttribute") {
            // r3 = CellSpurs*, r4 = CellSpursTaskset*, ...
            uint32_t tsPtr = (uint32_t)st.gpr[4];
            if (tsPtr && tsPtr + 0x80 <= memSize) {
                std::memset(mem + tsPtr, 0, 0x80);
                megakernel_write_mem(tsPtr, mem + tsPtr, 0x80);
            }
            retval = 0;
        } else if (e.name == "cellSpursCreateTask") {
            retval = 0;
        } else if (e.name == "cellSpursJoinTaskset") {
            // Blocking wait — return immediately (tasks "completed")
            retval = 0;
        } else if (e.name == "cellSpursShutdownTaskset") {
            retval = 0;
        } else if (e.name == "cellSpursTasksetAttributeSetName") {
            retval = 0;
        } else if (e.name == "cellSpursEventFlagInitialize") {
            // r3 = CellSpurs*, r4 = CellSpursEventFlag*, ...
            uint32_t efPtr = (uint32_t)st.gpr[4];
            if (efPtr && efPtr + 0x40 <= memSize) {
                std::memset(mem + efPtr, 0, 0x40);
                megakernel_write_mem(efPtr, mem + efPtr, 0x40);
            }
            retval = 0;
        } else if (e.name == "cellSpursEventFlagSet" ||
                   e.name == "cellSpursEventFlagClear") {
            retval = 0;
        } else if (e.name == "cellSpursEventFlagWait") {
            // Blocking wait — return immediately (flag "set")
            retval = 0;
        } else if (e.name == "cellSpursGetInfo") {
            // r3 = CellSpurs*, r4 = CellSpursInfo*
            uint32_t infoPtr = (uint32_t)st.gpr[4];
            if (infoPtr && infoPtr + 0x40 <= memSize) {
                std::memset(mem + infoPtr, 0, 0x40);
                // Write nSpus = 6 at offset 0
                uint32_t n = 6;
                uint8_t be[4] = { (uint8_t)(n>>24), (uint8_t)(n>>16),
                                  (uint8_t)(n>>8),  (uint8_t)n };
                std::memcpy(mem + infoPtr, be, 4);
                megakernel_write_mem(infoPtr, mem + infoPtr, 0x40);
            }
            retval = 0;

        // ── cellGame — game boot/content management ─────────────
        } else if (e.name == "cellGameBootCheck") {
            // r3 = CellGameContentSize* size (out), r4 = dirName (out)
            // Return CELL_GAME_RET_OK (=0), set type=CELL_GAME_GAMETYPE_DISC(1)
            retval = 0;
        } else if (e.name == "cellGameContentPermit") {
            // r3 = char* contentInfoPath (out), r4 = char* usrdirPath (out)
            // Write dummy paths
            uint32_t p1 = (uint32_t)st.gpr[3];
            uint32_t p2 = (uint32_t)st.gpr[4];
            const char* ci = "/dev_hdd0/game/TESTGAME";
            const char* ud = "/dev_hdd0/game/TESTGAME/USRDIR";
            if (p1 && p1 + 64 <= memSize) {
                std::memcpy(mem + p1, ci, strlen(ci) + 1);
                megakernel_write_mem(p1, mem + p1, strlen(ci) + 1);
            }
            if (p2 && p2 + 64 <= memSize) {
                std::memcpy(mem + p2, ud, strlen(ud) + 1);
                megakernel_write_mem(p2, mem + p2, strlen(ud) + 1);
            }
            retval = 0;
        } else if (e.name == "cellGamePatchCheck" ||
                   e.name == "cellGameDataCheck" ||
                   e.name == "cellGameCreateGameData" ||
                   e.name == "cellGameDeleteGameData" ||
                   e.name == "cellGameSetParamString") {
            retval = 0;
        } else if (e.name == "cellGameGetParamInt") {
            // r3 = int id, r4 = int* value (out)
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t val = 0;  // default value
                uint8_t be[4] = { (uint8_t)(val>>24), (uint8_t)(val>>16),
                                  (uint8_t)(val>>8),  (uint8_t)val };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellGameGetParamString") {
            // r3 = int id, r4 = char* buf, r5 = int bufsize
            uint32_t bufPtr = (uint32_t)st.gpr[4];
            uint32_t bufSz = (uint32_t)st.gpr[5];
            if (bufPtr && bufSz > 0 && bufPtr + bufSz <= memSize) {
                std::memset(mem + bufPtr, 0, bufSz);
                const char* s = "TESTGAME";
                size_t len = strlen(s);
                if (len >= bufSz) len = bufSz - 1;
                std::memcpy(mem + bufPtr, s, len);
                megakernel_write_mem(bufPtr, mem + bufPtr, bufSz);
            }
            retval = 0;
        } else if (e.name == "cellGameGetSizeKB") {
            // r3 = CellGameContentSize* size (out)
            uint32_t outPtr = (uint32_t)st.gpr[3];
            if (outPtr && outPtr + 12 <= memSize) {
                std::memset(mem + outPtr, 0, 12);
                megakernel_write_mem(outPtr, mem + outPtr, 12);
            }
            retval = 0;

        // ── cellSaveData — save/load stubs ──────────────────────
        } else if (e.name == "cellSaveDataAutoSave2" ||
                   e.name == "cellSaveDataAutoLoad2" ||
                   e.name == "cellSaveDataListSave2" ||
                   e.name == "cellSaveDataListLoad2" ||
                   e.name == "cellSaveDataFixedSave2" ||
                   e.name == "cellSaveDataFixedLoad2" ||
                   e.name == "cellSaveDataDelete2") {
            // Return CELL_SAVEDATA_RET_OK (0) — pretend save succeeded
            retval = 0;

        // ── cellAudio — audio port stubs ────────────────────────
        } else if (e.name == "cellAudioInit" ||
                   e.name == "cellAudioQuit") {
            retval = 0;
        } else if (e.name == "cellAudioPortOpen") {
            // r3 = CellAudioPortParam*, r4 = uint32_t* portNum (out)
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 4 <= memSize) {
                uint8_t be[4] = { 0, 0, 0, 0 };  // port 0
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellAudioPortClose" ||
                   e.name == "cellAudioPortStart" ||
                   e.name == "cellAudioPortStop" ||
                   e.name == "cellAudioSetNotifyEventQueue" ||
                   e.name == "cellAudioRemoveNotifyEventQueue") {
            retval = 0;
        } else if (e.name == "cellAudioGetPortConfig") {
            // r3 = portNum, r4 = CellAudioPortConfig* config (out)
            // Struct: { uint64_t readIndexAddr; uint32_t status; uint64_t nChannel;
            //           uint64_t nBlock; uint32_t portSize; uint64_t portAddr; }
            uint32_t cfgPtr = (uint32_t)st.gpr[4];
            if (cfgPtr && cfgPtr + 48 <= memSize) {
                std::memset(mem + cfgPtr, 0, 48);
                // status = CELL_AUDIO_STATUS_READY (1), nChannel = 2, nBlock = 8
                mem[cfgPtr + 11] = 1;  // status (offset 8, BE u32)
                mem[cfgPtr + 19] = 2;  // nChannel (offset 12, BE u64 low byte)
                mem[cfgPtr + 27] = 8;  // nBlock (offset 20, BE u64 low byte)
                megakernel_write_mem(cfgPtr, mem + cfgPtr, 48);
            }
            retval = 0;
        } else if (e.name == "cellAudioGetPortTimestamp") {
            // r3 = portNum, r4 = tag, r5 = uint64_t* stamp (out)
            uint32_t outPtr = (uint32_t)st.gpr[5];
            if (outPtr && outPtr + 8 <= memSize) {
                std::memset(mem + outPtr, 0, 8);
                megakernel_write_mem(outPtr, mem + outPtr, 8);
            }
            retval = 0;

        // ── cellFont — font rendering stubs ─────────────────────
        } else if (e.name == "cellFontInit" ||
                   e.name == "cellFontEnd" ||
                   e.name == "cellFontOpenFontFile" ||
                   e.name == "cellFontOpenFontMemory" ||
                   e.name == "cellFontCreateRenderer" ||
                   e.name == "cellFontSetupRenderScalePixel" ||
                   e.name == "cellFontSetScalePixel" ||
                   e.name == "cellFontBindRenderer" ||
                   e.name == "cellFontGetFontIdCode") {
            retval = 0;
        } else if (e.name == "cellFontRenderCharGlyphImage") {
            // Zero-fill the output glyph image (blank glyph)
            retval = 0;
        } else if (e.name == "cellFontGetHorizontalLayout") {
            // r3 = font*, r4 = CellFontHorizontalLayout* (out)
            // Struct has baselineY, lineHeight, effectHeight as floats
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 12 <= memSize) {
                std::memset(mem + outPtr, 0, 12);
                // Default: 16px lineHeight
                float lh = 16.0f;
                uint32_t f;
                std::memcpy(&f, &lh, 4);
                f = __builtin_bswap32(f);
                std::memcpy(mem + outPtr + 4, &f, 4);  // lineHeight at offset 4
                megakernel_write_mem(outPtr, mem + outPtr, 12);
            }
            retval = 0;
        } else if (e.name == "cellFontGetRenderCharGlyphMetrics") {
            // r3 = font*, r4 = code, r5 = CellFontGlyphMetrics* (out)
            uint32_t outPtr = (uint32_t)st.gpr[5];
            if (outPtr && outPtr + 32 <= memSize) {
                std::memset(mem + outPtr, 0, 32);
                megakernel_write_mem(outPtr, mem + outPtr, 32);
            }
            retval = 0;

        // ── sceNpTrophy — trophy system stubs ───────────────────
        } else if (e.name == "sceNpTrophyInit" ||
                   e.name == "sceNpTrophyTerm" ||
                   e.name == "sceNpTrophyDestroyContext" ||
                   e.name == "sceNpTrophyDestroyHandle") {
            retval = 0;
        } else if (e.name == "sceNpTrophyCreateContext") {
            // r3 = SceNpTrophyContext* ctx (out), ...
            uint32_t outPtr = (uint32_t)st.gpr[3];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t ctx = 1;  // dummy context ID
                uint8_t be[4] = { (uint8_t)(ctx>>24), (uint8_t)(ctx>>16),
                                  (uint8_t)(ctx>>8),  (uint8_t)ctx };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyCreateHandle") {
            // r3 = SceNpTrophyHandle* handle (out)
            uint32_t outPtr = (uint32_t)st.gpr[3];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t h = 1;  // dummy handle
                uint8_t be[4] = { (uint8_t)(h>>24), (uint8_t)(h>>16),
                                  (uint8_t)(h>>8),  (uint8_t)h };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyRegisterContext" ||
                   e.name == "sceNpTrophyUnlockTrophy") {
            retval = 0;
        } else if (e.name == "sceNpTrophyGetGameProgress") {
            // r3 = ctx, r4 = handle, r5 = int32_t* percentage (out)
            uint32_t outPtr = (uint32_t)st.gpr[5];
            if (outPtr && outPtr + 4 <= memSize) {
                uint8_t be[4] = { 0, 0, 0, 0 };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyGetGameInfo") {
            // r3 = ctx, r4 = handle, r5 = SceNpTrophyGameDetails* (out),
            // r6 = SceNpTrophyGameData* (out)
            uint32_t detPtr = (uint32_t)st.gpr[5];
            uint32_t datPtr = (uint32_t)st.gpr[6];
            if (detPtr && detPtr + 64 <= memSize) {
                std::memset(mem + detPtr, 0, 64);
                megakernel_write_mem(detPtr, mem + detPtr, 64);
            }
            if (datPtr && datPtr + 32 <= memSize) {
                std::memset(mem + datPtr, 0, 32);
                megakernel_write_mem(datPtr, mem + datPtr, 32);
            }
            retval = 0;

        // ── cellVdec — video decode stubs ───────────────────────
        } else if (e.name == "cellVdecOpen" || e.name == "cellVdecOpenEx") {
            // r3 = CellVdecType*, r4 = CellVdecResource*, r5 = cb, r6 = handle* (out)
            uint32_t outPtr = (uint32_t)st.gpr[6];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t h = 1;  // dummy decoder handle
                uint8_t be[4] = { (uint8_t)(h>>24), (uint8_t)(h>>16),
                                  (uint8_t)(h>>8),  (uint8_t)h };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellVdecClose" ||
                   e.name == "cellVdecStartSeq" ||
                   e.name == "cellVdecEndSeq" ||
                   e.name == "cellVdecDecodeAu" ||
                   e.name == "cellVdecSetFrameRate") {
            retval = 0;
        } else if (e.name == "cellVdecQueryAttr") {
            // r3 = CellVdecType*, r4 = CellVdecAttr* (out)
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 32 <= memSize) {
                std::memset(mem + outPtr, 0, 32);
                megakernel_write_mem(outPtr, mem + outPtr, 32);
            }
            retval = 0;
        } else if (e.name == "cellVdecGetPicture" ||
                   e.name == "cellVdecGetPictureExt") {
            // Return CELL_VDEC_ERROR_EMPTY (-1) to indicate no picture ready.
            // Games check return value and skip frame display.
            retval = (uint32_t)(-1);

        // ── cellFs extras ──────────────────────────────────────────────
        } else if (e.name == "cellFsOpendir" || e.name == "cellFsClosedir" ||
                   e.name == "cellFsReaddir") {
            // Directory stubs — return ENOENT for opendir, 0 for others
            if (e.name == "cellFsOpendir") {
                retval = 0x80010006;  // CELL_FS_ENOENT
            } else {
                retval = 0;
            }
        } else if (e.name == "cellFsStat") {
            // Return ENOENT — file not found
            retval = 0x80010006;
        } else if (e.name == "cellFsMkdir" || e.name == "cellFsRmdir" ||
                   e.name == "cellFsUnlink" || e.name == "cellFsRename" ||
                   e.name == "cellFsTruncate") {
            retval = 0;  // succeed silently
        } else if (e.name == "cellFsGetFreeSize") {
            // r3 = path*, r4 = block_size*, r5 = free_blocks*
            uint32_t bsPtr = (uint32_t)st.gpr[4];
            uint32_t fbPtr = (uint32_t)st.gpr[5];
            if (bsPtr && bsPtr + 4 <= memSize) {
                uint32_t bs = 512;
                uint8_t be[4] = { (uint8_t)(bs>>24), (uint8_t)(bs>>16),
                                  (uint8_t)(bs>>8),  (uint8_t)bs };
                std::memcpy(mem + bsPtr, be, 4);
                megakernel_write_mem(bsPtr, be, 4);
            }
            if (fbPtr && fbPtr + 4 <= memSize) {
                uint32_t fb = 1024*1024;  // ~512MB free
                uint8_t be[4] = { (uint8_t)(fb>>24), (uint8_t)(fb>>16),
                                  (uint8_t)(fb>>8),  (uint8_t)fb };
                std::memcpy(mem + fbPtr, be, 4);
                megakernel_write_mem(fbPtr, be, 4);
            }
            retval = 0;

        // ── cellSysmodule ──────────────────────────────────────────────
        } else if (e.name == "cellSysmoduleLoadModule" ||
                   e.name == "cellSysmoduleUnloadModule") {
            retval = 0;  // Always succeed
        } else if (e.name == "cellSysmoduleIsLoaded") {
            retval = 0;  // CELL_SYSMODULE_LOADED
        } else if (e.name == "cellSysmoduleInitialize" ||
                   e.name == "cellSysmoduleFinalize") {
            retval = 0;

        // ── cellNetCtl ─────────────────────────────────────────────────
        } else if (e.name == "cellNetCtlInit" || e.name == "cellNetCtlTerm") {
            retval = 0;
        } else if (e.name == "cellNetCtlGetState") {
            // r3 = state* (out) — set to DISCONNECTED (0)
            uint32_t statePtr = (uint32_t)st.gpr[3];
            if (statePtr && statePtr + 4 <= memSize) {
                uint32_t state = 0;  // CELL_NET_CTL_STATE_Disconnected
                uint8_t be[4] = { 0, 0, 0, 0 };
                std::memcpy(mem + statePtr, be, 4);
                megakernel_write_mem(statePtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellNetCtlGetInfo") {
            // Return error — not connected
            retval = 0x80130106;  // CELL_NET_CTL_ERROR_NOT_CONNECTED
        } else if (e.name == "cellNetCtlAddHandler" ||
                   e.name == "cellNetCtlDelHandler") {
            retval = 0;

        // ── cellMsgDialog ──────────────────────────────────────────────
        } else if (e.name == "cellMsgDialogOpen" ||
                   e.name == "cellMsgDialogOpen2") {
            // Immediately "close" dialog — game callback gets DIALOG_CLOSE
            retval = 0;
        } else if (e.name == "cellMsgDialogClose" ||
                   e.name == "cellMsgDialogAbort" ||
                   e.name == "cellMsgDialogProgressBarInc" ||
                   e.name == "cellMsgDialogProgressBarSetMsg") {
            retval = 0;

        // ── cellOskDialog ──────────────────────────────────────────────
        } else if (e.name == "cellOskDialogLoadAsync" ||
                   e.name == "cellOskDialogUnloadAsync" ||
                   e.name == "cellOskDialogSetInitialInputDevice") {
            retval = 0;
        } else if (e.name == "cellOskDialogGetInputText") {
            // r3 = buffer*, r4 = size — write empty string
            uint32_t bufPtr = (uint32_t)st.gpr[3];
            if (bufPtr && bufPtr + 2 <= memSize) {
                mem[bufPtr] = 0; mem[bufPtr+1] = 0;
                megakernel_write_mem(bufPtr, mem + bufPtr, 2);
            }
            retval = 0;

        // ── cellResc — resolution scaling ──────────────────────────────
        } else if (e.name == "cellRescInit" || e.name == "cellRescExit" ||
                   e.name == "cellRescSetDisplayMode" ||
                   e.name == "cellRescSetConvertAndFlip" ||
                   e.name == "cellRescSetBufferAddress" ||
                   e.name == "cellRescSetSrc" || e.name == "cellRescSetDsts") {
            retval = 0;
        } else if (e.name == "cellRescGetNumColorBuffers") {
            retval = 2;  // 2 color buffers (double-buffering)
        } else if (e.name == "cellRescGetBufferSize") {
            // r3 = *size (out) — return 1280*720*4 bytes per buffer
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp && outp + 4 <= memSize) {
                uint32_t sz = 1280 * 720 * 4;
                uint8_t be[4] = { (uint8_t)(sz>>24), (uint8_t)(sz>>16),
                                  (uint8_t)(sz>>8),  (uint8_t)sz };
                std::memcpy(mem + outp, be, 4);
                megakernel_write_mem(outp, be, 4);
            }
            retval = 0;

        // ── cellSysutil BGM ────────────────────────────────────────────
        } else if (e.name == "cellSysutilEnableBgmPlayback" ||
                   e.name == "cellSysutilDisableBgmPlayback") {
            retval = 0;
        } else if (e.name == "cellSysutilGetBgmPlaybackStatus") {
            // r3 = *status (out) — set to 0 (not playing)
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp && outp + 4 <= memSize) {
                std::memset(mem + outp, 0, 4);
                megakernel_write_mem(outp, mem + outp, 4);
            }
            retval = 0;

        // ── cellKb / cellMouse / cellCamera ────────────────────────────
        } else if (e.name == "cellKbInit" || e.name == "cellKbEnd" ||
                   e.name == "cellMouseInit" || e.name == "cellMouseEnd" ||
                   e.name == "cellCameraInit" || e.name == "cellCameraEnd") {
            retval = 0;
        } else if (e.name == "cellKbGetInfo" || e.name == "cellMouseGetInfo") {
            // r3 = *info (out) — zero it (no devices connected)
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp && outp + 32 <= memSize) {
                std::memset(mem + outp, 0, 32);
                megakernel_write_mem(outp, mem + outp, 32);
            }
            retval = 0;
        } else if (e.name == "cellKbRead") {
            // r3 = port, r4 = *data (out) — zero it (no key events)
            uint32_t outp = (uint32_t)st.gpr[4];
            if (outp && outp + 16 <= memSize) {
                std::memset(mem + outp, 0, 16);
                megakernel_write_mem(outp, mem + outp, 16);
            }
            retval = 0;

        // ── cellUserInfo ───────────────────────────────────────────────
        } else if (e.name == "cellUserInfoGetStat") {
            // r3 = userId, r4 = *CellUserInfoStat (out) — zero it
            uint32_t outp = (uint32_t)st.gpr[4];
            if (outp && outp + 64 <= memSize) {
                std::memset(mem + outp, 0, 64);
                megakernel_write_mem(outp, mem + outp, 64);
            }
            retval = 0;
        } else if (e.name == "cellUserInfoGetList") {
            // r3 = *listNum (out), r4 = *listBuf (out) — zero (no users)
            uint32_t numPtr = (uint32_t)st.gpr[3];
            if (numPtr && numPtr + 4 <= memSize) {
                std::memset(mem + numPtr, 0, 4);
                megakernel_write_mem(numPtr, mem + numPtr, 4);
            }
            retval = 0;

        // ── cellSsl / cellHttp ─────────────────────────────────────────
        } else if (e.name == "cellSslInit" || e.name == "cellSslEnd" ||
                   e.name == "cellHttpInit" || e.name == "cellHttpEnd") {
            retval = 0;

        // ── cellAdec — audio decoder ───────────────────────────────────
        } else if (e.name == "cellAdecOpen") {
            // r3 = type*, r4 = resource*, r5 = cb, r6 = handle* (out)
            uint32_t outPtr = (uint32_t)st.gpr[6];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t h = 2;  // dummy audio decoder handle
                uint8_t be[4] = { (uint8_t)(h>>24), (uint8_t)(h>>16),
                                  (uint8_t)(h>>8),  (uint8_t)h };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellAdecClose" || e.name == "cellAdecStartSeq" ||
                   e.name == "cellAdecEndSeq" || e.name == "cellAdecDecodeAu") {
            retval = 0;
        } else if (e.name == "cellAdecGetPcm" || e.name == "cellAdecGetPcmItem") {
            retval = (uint32_t)(-1);  // No PCM ready
        } else if (e.name == "cellAdecQueryAttr") {
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 32 <= memSize) {
                std::memset(mem + outPtr, 0, 32);
                megakernel_write_mem(outPtr, mem + outPtr, 32);
            }
            retval = 0;

        // ── cellDmux — demuxer ─────────────────────────────────────────
        } else if (e.name == "cellDmuxOpen") {
            uint32_t outPtr = (uint32_t)st.gpr[5];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t h = 3;  // dummy demux handle
                uint8_t be[4] = { (uint8_t)(h>>24), (uint8_t)(h>>16),
                                  (uint8_t)(h>>8),  (uint8_t)h };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellDmuxClose" || e.name == "cellDmuxSetStream" ||
                   e.name == "cellDmuxResetStream" || e.name == "cellDmuxEnableEs" ||
                   e.name == "cellDmuxDisableEs") {
            retval = 0;
        } else if (e.name == "cellDmuxGetAu") {
            retval = (uint32_t)(-1);  // No AU available
        } else if (e.name == "cellDmuxQueryAttr") {
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 32 <= memSize) {
                std::memset(mem + outPtr, 0, 32);
                megakernel_write_mem(outPtr, mem + outPtr, 32);
            }
            retval = 0;

        // ── cellPamf — PAMF container ──────────────────────────────────
        } else if (e.name == "cellPamfReaderInitialize") {
            retval = 0;
        } else if (e.name == "cellPamfReaderGetNumberOfStreams") {
            retval = 0;  // 0 streams (no media)
        } else if (e.name == "cellPamfReaderGetStreamInfo" ||
                   e.name == "cellPamfReaderGetStreamTypeCoding" ||
                   e.name == "cellPamfReaderGetEsFilterId") {
            retval = 0;
        } else if (e.name == "cellPamfGetHeaderSize") {
            retval = 2048;  // Typical PAMF header size

        // ── cellJpgDec / cellPngDec ────────────────────────────────────
        } else if (e.name == "cellJpgDecCreate" || e.name == "cellPngDecCreate") {
            // r3 = *mainHandle (out) — write dummy handle
            uint32_t outPtr = (uint32_t)st.gpr[3];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t h = 4;
                uint8_t be[4] = { (uint8_t)(h>>24), (uint8_t)(h>>16),
                                  (uint8_t)(h>>8),  (uint8_t)h };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellJpgDecDestroy" || e.name == "cellPngDecDestroy" ||
                   e.name == "cellJpgDecOpen" || e.name == "cellJpgDecClose" ||
                   e.name == "cellPngDecOpen" || e.name == "cellPngDecClose") {
            retval = 0;
        } else if (e.name == "cellJpgDecReadHeader" || e.name == "cellPngDecReadHeader") {
            // Return header info — write dummy 64×64 dimensions
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 16 <= memSize) {
                std::memset(mem + outPtr, 0, 16);
                // width=64, height=64 at offset 0 and 4 (big-endian)
                uint32_t dim = 64;
                mem[outPtr+0] = 0; mem[outPtr+1] = 0; mem[outPtr+2] = 0; mem[outPtr+3] = 64;
                mem[outPtr+4] = 0; mem[outPtr+5] = 0; mem[outPtr+6] = 0; mem[outPtr+7] = 64;
                megakernel_write_mem(outPtr, mem + outPtr, 16);
            }
            retval = 0;
        } else if (e.name == "cellJpgDecDecodeData" || e.name == "cellPngDecDecodeData") {
            // Fill output buffer with white pixels (ARGB 0xFFFFFFFF)
            retval = 0;

        // ── cellL10n — string conversion ───────────────────────────────
        } else if (e.name == "UTF8stoUCS2s" || e.name == "UCS2stoUTF8s" ||
                   e.name == "UTF8toUCS2" || e.name == "UCS2toUTF8" ||
                   e.name == "UTF16stoUTF8s" || e.name == "UTF8stoUTF16s" ||
                   e.name == "L10nConvertStr") {
            // String conversion stubs — return 0 (success).
            // Real impl would convert between encodings. Games usually
            // just pass ASCII through these which works even without conversion.
            retval = 0;

        // ── cellSync — synchronization ─────────────────────────────────
        } else if (e.name == "cellSyncMutexInitialize" ||
                   e.name == "cellSyncBarrierInitialize" ||
                   e.name == "cellSyncLFQueueInitialize") {
            // r3 = *object — zero-init the sync primitive
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp && outp + 32 <= memSize) {
                std::memset(mem + outp, 0, 32);
                megakernel_write_mem(outp, mem + outp, 32);
            }
            retval = 0;
        } else if (e.name == "cellSyncMutexLock" || e.name == "cellSyncMutexTryLock" ||
                   e.name == "cellSyncMutexUnlock" ||
                   e.name == "cellSyncBarrierNotify" ||
                   e.name == "cellSyncBarrierTryNotify" ||
                   e.name == "cellSyncBarrierTryWait") {
            retval = 0;  // Always succeed (single-threaded emulation)
        } else if (e.name == "cellSyncLFQueuePush" || e.name == "cellSyncLFQueueTryPush" ||
                   e.name == "cellSyncLFQueuePop" || e.name == "cellSyncLFQueueTryPop") {
            retval = 0;
        } else if (e.name == "cellSyncLFQueueGetSize") {
            retval = 0;  // Queue is empty

        // ── cellGifDec ─────────────────────────────────────────────────
        } else if (e.name == "cellGifDecCreate") {
            uint32_t outPtr = (uint32_t)st.gpr[3];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t h = 5;
                uint8_t be[4] = { (uint8_t)(h>>24), (uint8_t)(h>>16),
                                  (uint8_t)(h>>8),  (uint8_t)h };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.name == "cellGifDecDestroy" || e.name == "cellGifDecOpen" ||
                   e.name == "cellGifDecClose") {
            retval = 0;
        } else if (e.name == "cellGifDecReadHeader") {
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 16 <= memSize) {
                std::memset(mem + outPtr, 0, 16);
                mem[outPtr+3] = 64; mem[outPtr+7] = 64;  // 64×64
                megakernel_write_mem(outPtr, mem + outPtr, 16);
            }
            retval = 0;
        } else if (e.name == "cellGifDecDecodeData") {
            retval = 0;

        // ═══ sceNpTrophy — PSN trophy system ═══
        } else if (e.name == "sceNpTrophyInit") {
            retval = 0;
        } else if (e.name == "sceNpTrophyTerm") {
            retval = 0;
        } else if (e.name == "sceNpTrophyCreateHandle") {
            // r3 = handle_ptr
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyDestroyHandle") {
            retval = 0;
        } else if (e.name == "sceNpTrophyCreateContext") {
            // r3 = context_ptr
            uint32_t cid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(cid >> 24); mem[ptr+1] = (uint8_t)(cid >> 16);
                mem[ptr+2] = (uint8_t)(cid >> 8); mem[ptr+3] = (uint8_t)cid;
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyDestroyContext") {
            retval = 0;
        } else if (e.name == "sceNpTrophyRegisterContext") {
            retval = 0;
        } else if (e.name == "sceNpTrophyGetRequiredDiskSpace") {
            // r3 = context, r4 = handle, r5 = reqspace_ptr
            uint64_t ptr = st.gpr[5];
            if (ptr && ptr + 8 <= memSize) {
                // 1MB required space
                uint64_t space = 0x100000ULL;
                for (int i = 0; i < 8; i++) mem[ptr+i] = (uint8_t)(space >> (56 - 8*i));
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyGetGameInfo") {
            // r5 = game_info_ptr — zero-fill the structure
            uint64_t ptr = st.gpr[5];
            if (ptr && ptr + 64 <= memSize) {
                std::memset(mem + ptr, 0, 64);
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyUnlockTrophy") {
            // r3 = context, r4 = handle, r5 = trophy_id, r6 = platinum_id_ptr
            uint64_t ptr = st.gpr[6];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0xFF; mem[ptr+1] = 0xFF; mem[ptr+2] = 0xFF; mem[ptr+3] = 0xFF; // -1 = no platinum
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyGetTrophyInfo") {
            // r5 = info_ptr
            uint64_t ptr = st.gpr[5];
            if (ptr && ptr + 48 <= memSize) {
                std::memset(mem + ptr, 0, 48);
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyGetTrophyUnlockState") {
            // r5 = flag_array_ptr, r6 = count_ptr
            uint64_t ptr = st.gpr[5];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16); // all locked
            }
            uint64_t cnt = st.gpr[6];
            if (cnt && cnt + 4 <= memSize) {
                mem[cnt] = 0; mem[cnt+1] = 0; mem[cnt+2] = 0; mem[cnt+3] = 0;
            }
            retval = 0;
        } else if (e.name == "sceNpTrophyGetGameProgress") {
            // r5 = progress_ptr (int32 percentage)
            uint64_t ptr = st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0; mem[ptr+1] = 0; mem[ptr+2] = 0; mem[ptr+3] = 0; // 0%
            }
            retval = 0;

        // ═══ sceNp — PSN base ═══
        } else if (e.name == "sceNpInit") {
            retval = 0;
        } else if (e.name == "sceNpTerm") {
            retval = 0;
        } else if (e.name == "sceNpGetNpId" || e.name == "sceNpManagerGetNpId") {
            // r3 = npid_ptr — fill with dummy offline NP ID
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 36 <= memSize) {
                std::memset(mem + ptr, 0, 36);
                // Write "OfflineUser" as the handle
                const char* handle = "OfflineUser";
                for (int i = 0; handle[i]; i++) mem[ptr + i] = handle[i];
            }
            retval = 0;
        } else if (e.name == "sceNpGetOnlineId" || e.name == "sceNpManagerGetOnlineId") {
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 20 <= memSize) {
                std::memset(mem + ptr, 0, 20);
                const char* oid = "OfflineUser";
                for (int i = 0; oid[i]; i++) mem[ptr + i] = oid[i];
            }
            retval = 0;
        } else if (e.name == "sceNpGetUserProfile") {
            retval = 0;
        } else if (e.name == "sceNpManagerGetStatus") {
            // Return SCE_NP_MANAGER_STATUS_OFFLINE (1) via r4
            st.gpr[4] = 1;
            retval = 0;
        } else if (e.name == "sceNpManagerGetNetworkTime") {
            // r3 = time_ptr (CellRtcTick - uint64_t)
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 8 <= memSize) {
                uint64_t tick = 0x000DC46C0D3B3600ULL; // ~2024 epoch
                for (int i = 0; i < 8; i++) mem[ptr+i] = (uint8_t)(tick >> (56 - 8*i));
            }
            retval = 0;
        } else if (e.name == "sceNpManagerRegisterCallback" || e.name == "sceNpManagerUnregisterCallback") {
            retval = 0;
        } else if (e.name == "sceNpCommerce2Init" || e.name == "sceNpCommerce2Term") {
            retval = 0;

        // ═══ cellRtc — real-time clock ═══
        } else if (e.name == "cellRtcGetCurrentTick") {
            // r3 = CellRtcTick* (uint64_t big-endian)
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 8 <= memSize) {
                // ~2024 epoch in microseconds from 0001-01-01
                uint64_t tick = 0x000DC46C0D3B3600ULL + virtTime;
                for (int i = 0; i < 8; i++) mem[ptr+i] = (uint8_t)(tick >> (56 - 8*i));
            }
            virtTime += 16667; // advance ~1 frame
            retval = 0;
        } else if (e.name == "cellRtcGetCurrentClockLocalTime" || e.name == "cellRtcGetCurrentClock") {
            // r3 = CellRtcDateTime* (16 bytes)
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                // year=2024, month=6, day=15, hour=12, min=0, sec=0
                mem[ptr+0] = 0x07; mem[ptr+1] = 0xE8; // year BE 2024
                mem[ptr+2] = 0x00; mem[ptr+3] = 6;     // month
                mem[ptr+4] = 0x00; mem[ptr+5] = 15;    // day
                mem[ptr+6] = 0x00; mem[ptr+7] = 12;    // hour
            }
            retval = 0;
        } else if (e.name == "cellRtcConvertLocalTimeToUtc" || e.name == "cellRtcConvertUtcToLocalTime") {
            // r3 = src tick*, r4 = dst tick*  — just copy (no timezone offset)
            uint64_t src = st.gpr[3], dst = st.gpr[4];
            if (src && dst && src + 8 <= memSize && dst + 8 <= memSize) {
                std::memcpy(mem + dst, mem + src, 8);
            }
            retval = 0;
        } else if (e.name == "cellRtcGetTick") {
            // r3 = CellRtcDateTime*, r4 = CellRtcTick* — fake tick output
            uint64_t ptr = st.gpr[4];
            if (ptr && ptr + 8 <= memSize) {
                uint64_t tick = 0x000DC46C0D3B3600ULL;
                for (int i = 0; i < 8; i++) mem[ptr+i] = (uint8_t)(tick >> (56 - 8*i));
            }
            retval = 0;
        } else if (e.name == "cellRtcSetTick") {
            retval = 0;
        } else if (e.name == "cellRtcTickAddSeconds" || e.name == "cellRtcTickAddMinutes") {
            // r3 = dst tick*, r4 = src tick*, r5 = delta
            uint64_t dst = st.gpr[3], src = st.gpr[4];
            if (src && dst && src + 8 <= memSize && dst + 8 <= memSize) {
                uint64_t tick = 0;
                for (int i = 0; i < 8; i++) tick = (tick << 8) | mem[src+i];
                uint64_t delta = st.gpr[5];
                if (e.name == "cellRtcTickAddMinutes") delta *= 60;
                tick += delta * 1000000ULL;
                for (int i = 0; i < 8; i++) mem[dst+i] = (uint8_t)(tick >> (56 - 8*i));
            }
            retval = 0;
        } else if (e.name == "cellRtcFormatRfc2822") {
            retval = 0; // no-op (buffer left empty)
        } else if (e.name == "cellRtcIsLeapYear") {
            uint32_t year = (uint32_t)st.gpr[3];
            st.gpr[4] = ((year % 4 == 0) && (year % 100 != 0 || year % 400 == 0)) ? 1 : 0;
            retval = 0;
        } else if (e.name == "cellRtcGetDaysInMonth") {
            uint32_t year = (uint32_t)st.gpr[3];
            uint32_t month = (uint32_t)st.gpr[4];
            static const int days[] = {31,28,31,30,31,30,31,31,30,31,30,31};
            int d = (month >= 1 && month <= 12) ? days[month-1] : 30;
            if (month == 2 && ((year%4==0) && (year%100!=0 || year%400==0))) d = 29;
            st.gpr[4] = d;
            retval = 0;
        } else if (e.name == "cellRtcGetDayOfWeek") {
            st.gpr[4] = 0; // Monday
            retval = 0;

        // ═══ cellScreenshot ═══
        } else if (e.name == "cellScreenShotEnable" || e.name == "cellScreenShotDisable" ||
                   e.name == "cellScreenShotSetParameter" || e.name == "cellScreenShotSetOverlayImage") {
            retval = 0;

        // ═══ cellMic ═══
        } else if (e.name == "cellMicInit" || e.name == "cellMicEnd" ||
                   e.name == "cellMicOpen" || e.name == "cellMicClose") {
            retval = 0;
        } else if (e.name == "cellMicGetDeviceAttr") {
            retval = (uint64_t)(int64_t)-1; // CELL_MIC_ERROR_DEVICE_NOT_FOUND

        // ═══ cellSysCache ═══
        } else if (e.name == "cellSysCacheMount") {
            // r3 = CellSysCacheParam* — write cache path
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 512 <= memSize) {
                // getCachePath offset = +4, write "/dev_hdd1/caches/"
                const char* path = "/dev_hdd1/caches/";
                for (int i = 0; path[i]; i++) mem[ptr + 4 + i] = path[i];
                mem[ptr + 4 + 17] = 0;
            }
            retval = 1; // CELL_SYSCACHE_RET_OK_CLEARED
        } else if (e.name == "cellSysCacheClear") {
            retval = 0;

        // ═══ cellUsbd / cellImeJp ═══
        } else if (e.name == "cellUsbdInit" || e.name == "cellUsbdEnd" ||
                   e.name == "cellImeJpOpen" || e.name == "cellImeJpClose") {
            retval = 0;

        // ═══ cellSysutil extras ═══
        } else if (e.name == "cellSysutilGetSystemParamString") {
            // r3 = param_id, r4 = buf_ptr, r5 = buf_size
            uint64_t buf = st.gpr[4];
            uint32_t sz = (uint32_t)st.gpr[5];
            if (buf && sz > 0 && buf + sz <= memSize) {
                mem[buf] = 0; // empty string
            }
            retval = 0;
        } else if (e.name == "cellSysutilGetBgmPlaybackStatus2") {
            // r3 = status_ptr (u32) → 0 = not playing
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0; mem[ptr+1] = 0; mem[ptr+2] = 0; mem[ptr+3] = 0;
            }
            retval = 0;
        } else if (e.name == "cellSysutilEnableBgmPlaybackEx" || e.name == "cellSysutilDisableBgmPlaybackEx") {
            retval = 0;

        // ═══ cellDiscGame ═══
        } else if (e.name == "cellDiscGameGetBootDiscInfo") {
            // r3 = CellDiscGameBootInfo* — zero-fill
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
            }
            retval = 0;
        } else if (e.name == "cellDiscGameRegisterDiscChangeCallback" ||
                   e.name == "cellDiscGameUnregisterDiscChangeCallback") {
            retval = 0;

        // ═══ cellStorage ═══
        } else if (e.name == "cellStorageDataImportMove" || e.name == "cellStorageDataExport") {
            retval = 0;

        // ═══ cellSubDisplay ═══
        } else if (e.name == "cellSubDisplayInit" || e.name == "cellSubDisplayEnd") {
            retval = 0;
        } else if (e.name == "cellSubDisplayGetRequiredMemory") {
            // r3 = size_ptr
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t sz = 0x10000; // 64KB
                mem[ptr] = (uint8_t)(sz>>24); mem[ptr+1] = (uint8_t)(sz>>16);
                mem[ptr+2] = (uint8_t)(sz>>8); mem[ptr+3] = (uint8_t)sz;
            }
            retval = 0;

        // ═══ cellSearch ═══
        } else if (e.name == "cellSearchInitialize" || e.name == "cellSearchFinalize") {
            retval = 0;
        } else if (e.name == "cellSearchStartListSearch") {
            retval = 0;
        } else if (e.name == "cellSearchGetListItems") {
            // r4 = count_ptr → 0 results
            uint64_t ptr = st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0; mem[ptr+1] = 0; mem[ptr+2] = 0; mem[ptr+3] = 0;
            }
            retval = 0;

        // ═══ cellMusic ═══
        } else if (e.name == "cellMusicInitialize" || e.name == "cellMusicFinalize") {
            retval = 0;
        } else if (e.name == "cellMusicGetPlaybackStatus") {
            // r3 = status_ptr → 0 = stopped
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0; mem[ptr+1] = 0; mem[ptr+2] = 0; mem[ptr+3] = 0;
            }
            retval = 0;

        // ═══ cellPhotoExport ═══
        } else if (e.name == "cellPhotoRegistFromFile" || e.name == "cellPhotoExportInitialize" ||
                   e.name == "cellPhotoExportFinalize") {
            retval = 0;

        // ═══ cellRemotePlay / cellBgdl / cellGameUpdate ═══
        } else if (e.name == "cellRemotePlayGetStatus") {
            st.gpr[4] = 0; // not connected
            retval = 0;
        } else if (e.name == "cellBgdlGetInfo") {
            retval = (uint64_t)(int64_t)-1; // no background downloads
        } else if (e.name == "cellGameUpdateInit" || e.name == "cellGameUpdateTerm") {
            retval = 0;
        } else if (e.name == "cellGameUpdateCheckStartAsync") {
            retval = 0; // no update available

        // ═══ cellVpost — video post-processing ═══
        } else if (e.name == "cellVpostOpen" || e.name == "cellVpostOpenEx") {
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.name == "cellVpostClose" || e.name == "cellVpostExec") {
            retval = 0;

        // ═══ cellAtrac — ATRAC3+ audio ═══
        } else if (e.name == "cellAtracCreateDecoder") {
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.name == "cellAtracDeleteDecoder") {
            retval = 0;
        } else if (e.name == "cellAtracDecode") {
            retval = (uint64_t)(int64_t)-1; // no audio data
        } else if (e.name == "cellAtracGetStreamDataInfo") {
            retval = 0;
        } else if (e.name == "cellAtracAddStreamData") {
            retval = 0;
        } else if (e.name == "cellAtracIsSecondBufferNeeded") {
            st.gpr[4] = 0; // not needed
            retval = 0;

        // ═══ cellVoice ═══
        } else if (e.name == "cellVoiceInit" || e.name == "cellVoiceEnd") {
            retval = 0;
        } else if (e.name == "cellVoiceCreatePort") {
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.name == "cellVoiceDeletePort") {
            retval = 0;

        // ═══ sceNpMatching2 — matchmaking ═══
        } else if (e.name == "sceNpMatching2Init" || e.name == "sceNpMatching2Term") {
            retval = 0;
        } else if (e.name == "sceNpMatching2CreateContext") {
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.name == "sceNpMatching2DestroyContext") {
            retval = 0;

        // ═══ sceNpScore — leaderboards ═══
        } else if (e.name == "sceNpScoreInit" || e.name == "sceNpScoreTerm") {
            retval = 0;
        } else if (e.name == "sceNpScoreCreateTitleCtx") {
            st.gpr[4] = nextHandleId++;
            retval = 0;
        } else if (e.name == "sceNpScoreDeleteTitleCtx") {
            retval = 0;

        // ═══ sceNpTus — title user storage ═══
        } else if (e.name == "sceNpTusInit" || e.name == "sceNpTusTerm") {
            retval = 0;
        } else if (e.name == "sceNpTusCreateTitleCtx") {
            st.gpr[4] = nextHandleId++;
            retval = 0;
        } else if (e.name == "sceNpTusDeleteTitleCtx") {
            retval = 0;

        // ═══ cellSaveData v2 variants ═══
        } else if (e.name == "cellSaveDataAutoLoad2" || e.name == "cellSaveDataAutoSave2" ||
                   e.name == "cellSaveDataUserListLoad" || e.name == "cellSaveDataUserListSave" ||
                   e.name == "cellSaveDataUserFixedLoad" || e.name == "cellSaveDataUserFixedSave") {
            retval = 0;

        // ═══ cellGame extras ═══
        } else if (e.name == "cellGameSetExitParam" || e.name == "cellGameThemeInstall" ||
                   e.name == "cellGameThemeInstallFromBuffer") {
            retval = 0;
        } else if (e.name == "cellGameGetLocalWebContentPath") {
            // r3 = buf, r4 = size — empty path
            uint64_t buf = st.gpr[3];
            if (buf && buf < memSize) mem[buf] = 0;
            retval = 0;

        // ═══ cellWebBrowser ═══
        } else if (e.name == "cellWebBrowserInitialize" || e.name == "cellWebBrowserShutdown") {
            retval = 0;

        // ═══ cellPad extras ═══
        } else if (e.name == "cellPadSetActDirect" || e.name == "cellPadSetPortSetting" ||
                   e.name == "cellPadSetSensorMode" || e.name == "cellPadLddRegisterController") {
            retval = 0;
        } else if (e.name == "cellPadGetCapabilityInfo") {
            // r4 = info_ptr — zero-fill
            uint64_t ptr = st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.name == "cellPadGetInfo2") {
            // r3 = CellPadInfo2* — zero-fill (0 pads connected)
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;
        } else if (e.name == "cellPadLddGetPortNo") {
            st.gpr[4] = (uint64_t)(int64_t)-1; // no port
            retval = 0;

        // ── cellPrint ────────────────────────────────────────────
        } else if (e.name == "cellPrintOpenConfig" || e.name == "cellPrintGetStatus" ||
                   e.name == "cellPrintStartJob" || e.name == "cellPrintEndJob") {
            retval = 0;

        // ── cellMusicDecode ──────────────────────────────────────
        } else if (e.name == "cellMusicDecodeInitialize" || e.name == "cellMusicDecodeInitialize2") {
            retval = 0;
        } else if (e.name == "cellMusicDecodeFinalize" || e.name == "cellMusicDecodeFinalize2") {
            retval = 0;
        } else if (e.name == "cellMusicDecodeSelectContents" || e.name == "cellMusicDecodeSetDecodeCommand") {
            retval = 0;
        } else if (e.name == "cellMusicDecodeGetDecodeStatus") {
            // Write status=0 (idle) to pointer in r4
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t zero = 0;
                std::memcpy(mem + (uint32_t)st.gpr[4], &zero, 4);
            }
            retval = 0;
        } else if (e.name == "cellMusicDecodeRead") {
            retval = (uint64_t)(int64_t)-1; // no data

        // ── sceNpFriends ─────────────────────────────────────────
        } else if (e.name == "sceNpFriendsInit") {
            retval = 0;
        } else if (e.name == "sceNpFriendsTerm") {
            retval = 0;
        } else if (e.name == "sceNpFriendsGetFriendListEntryCount") {
            // Write count=0 to pointer in r4
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t zero = 0;
                std::memcpy(mem + (uint32_t)st.gpr[4], &zero, 4);
            }
            retval = 0;
        } else if (e.name == "sceNpFriendsGetFriendListEntry" ||
                   e.name == "sceNpFriendsGetFriendPresence" ||
                   e.name == "sceNpFriendsGetFriendInfo") {
            retval = (uint64_t)(int64_t)-1; // not found

        // ── cellSail (media player) ──────────────────────────────
        } else if (e.name == "cellSailPlayerInitialize") {
            retval = 0;
        } else if (e.name == "cellSailPlayerFinalize") {
            retval = 0;
        } else if (e.name == "cellSailPlayerSetParameter" || e.name == "cellSailPlayerGetParameter") {
            retval = 0;
        } else if (e.name == "cellSailPlayerBoot") {
            retval = 0;
        } else if (e.name == "cellSailPlayerCreateDescriptor") {
            // Write handle to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[4], &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellSailPlayerDestroyDescriptor") {
            retval = 0;
        } else if (e.name == "cellSailPlayerOpenStream") {
            retval = 0;

        // ── cellRudp (reliable UDP) ──────────────────────────────
        } else if (e.name == "cellRudpInit") {
            retval = 0;
        } else if (e.name == "cellRudpEnd") {
            retval = 0;
        } else if (e.name == "cellRudpCreateContext") {
            retval = (uint64_t)nextHandleId++; // context ID
        } else if (e.name == "cellRudpBind") {
            retval = 0;
        } else if (e.name == "cellRudpSend") {
            retval = (uint64_t)(int64_t)-1; // network error
        } else if (e.name == "cellRudpReceive") {
            retval = (uint64_t)(int64_t)-1; // no data

        // ── cellHttpUtil ─────────────────────────────────────────
        } else if (e.name == "cellHttpUtilParseUri" || e.name == "cellHttpUtilBuildUri" ||
                   e.name == "cellHttpUtilEscapeUri" || e.name == "cellHttpUtilUnescapeUri") {
            retval = 0;

        // ── cellSsl ──────────────────────────────────────────────
        } else if (e.name == "cellSslInit") {
            retval = 0;
        } else if (e.name == "cellSslEnd") {
            retval = 0;
        } else if (e.name == "cellSslCertificateLoader" || e.name == "cellSslCertGetSerialNumber" ||
                   e.name == "cellSslCertGetPublicKey") {
            retval = 0;

        // ── cellHttp ─────────────────────────────────────────────
        } else if (e.name == "cellHttpInit") {
            retval = 0;
        } else if (e.name == "cellHttpEnd") {
            retval = 0;
        } else if (e.name == "cellHttpCreateClient") {
            // Write client handle to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[4], &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellHttpDestroyClient") {
            retval = 0;
        } else if (e.name == "cellHttpCreateTransaction") {
            // Write transaction handle to r5 pointer
            if (st.gpr[5] && st.gpr[5] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[5], &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellHttpDestroyTransaction") {
            retval = 0;
        } else if (e.name == "cellHttpSendRequest") {
            retval = (uint64_t)(int64_t)-1; // network error
        } else if (e.name == "cellHttpRecvResponse") {
            retval = (uint64_t)(int64_t)-1; // network error

        // ── cellNetCtl ───────────────────────────────────────────
        } else if (e.name == "cellNetCtlInit") {
            retval = 0;
        } else if (e.name == "cellNetCtlTerm") {
            retval = 0;
        } else if (e.name == "cellNetCtlGetState") {
            // Write state=0 (disconnected) to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t zero = 0;
                std::memcpy(mem + (uint32_t)st.gpr[4], &zero, 4);
            }
            retval = 0;
        } else if (e.name == "cellNetCtlGetInfo") {
            retval = (uint64_t)(int64_t)-1; // not connected
        } else if (e.name == "cellNetCtlAddHandler") {
            retval = 0;
        } else if (e.name == "cellNetCtlDelHandler") {
            retval = 0;

        // ── cellFont ─────────────────────────────────────────────
        } else if (e.name == "cellFontInit" || e.name == "cellFontEnd") {
            retval = 0;
        } else if (e.name == "cellFontNewLibrary") {
            // Write library handle to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[4], &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellFontFreeLibrary" || e.name == "cellFontCloseFont") {
            retval = 0;
        } else if (e.name == "cellFontOpenFontset" || e.name == "cellFontOpenFontFile") {
            retval = 0;
        } else if (e.name == "cellFontRenderCharGlyphImage") {
            // Render nothing — games fall back to blank glyph
            retval = 0;

        // ── cellFontFT ───────────────────────────────────────────
        } else if (e.name == "cellFontFTInit" || e.name == "cellFontFTEnd" ||
                   e.name == "cellFontFTLoadModule") {
            retval = 0;
        } else if (e.name == "cellFontFTGetInitializedRevisionFlags") {
            retval = 0; // flags=0

        // ── cellSpurs ────────────────────────────────────────────
        } else if (e.name == "cellSpursInitialize") {
            retval = 0;
        } else if (e.name == "cellSpursFinalize") {
            retval = 0;
        } else if (e.name == "cellSpursGetNumSpuThread") {
            // Write count=6 to r4 pointer (PS3 has 6 SPUs available)
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t six = __builtin_bswap32(6);
                std::memcpy(mem + (uint32_t)st.gpr[4], &six, 4);
            }
            retval = 0;
        } else if (e.name == "cellSpursGetSpuThreadGroupId") {
            // Write fake group ID to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[4], &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellSpursSetMaxContention" || e.name == "cellSpursSetPriorities" ||
                   e.name == "cellSpursAttachLv2EventQueue") {
            retval = 0;
        } else if (e.name == "cellSpursCreateTask") {
            retval = 0;
        } else if (e.name == "cellSpursCreateTaskset") {
            retval = 0;
        } else if (e.name == "cellSpursJoinTaskset") {
            retval = 0;
        } else if (e.name == "cellSpursShutdownTaskset" || e.name == "cellSpursWakeupTaskset") {
            retval = 0;

        // ── cellSpursJq ──────────────────────────────────────────
        } else if (e.name == "cellSpursJqInitialize" || e.name == "cellSpursJqFinalize") {
            retval = 0;
        } else if (e.name == "cellSpursJqAddJob") {
            retval = 0;
        } else if (e.name == "cellSpursJqGetJobCount") {
            // Write count=0 to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t zero = 0;
                std::memcpy(mem + (uint32_t)st.gpr[4], &zero, 4);
            }
            retval = 0;

        // ── sys_net (BSD sockets) ────────────────────────────────
        } else if (e.name == "sys_net_bnet_socket") {
            retval = (uint64_t)nextHandleId++; // fake socket fd
        } else if (e.name == "sys_net_bnet_close") {
            retval = 0;
        } else if (e.name == "sys_net_bnet_bind" || e.name == "sys_net_bnet_listen") {
            retval = 0;
        } else if (e.name == "sys_net_bnet_accept" || e.name == "sys_net_bnet_connect") {
            retval = (uint64_t)(int64_t)-1; // network error
        } else if (e.name == "sys_net_bnet_sendto") {
            retval = (uint64_t)(int64_t)-1;
        } else if (e.name == "sys_net_bnet_recvfrom") {
            retval = (uint64_t)(int64_t)-1;
        } else if (e.name == "sys_net_bnet_setsockopt" || e.name == "sys_net_bnet_getsockopt") {
            retval = 0;

        // ── cellUserInfo ─────────────────────────────────────────
        } else if (e.name == "cellUserInfoGetStat") {
            retval = 0;
        } else if (e.name == "cellUserInfoSelectUser_ListType") {
            retval = 0;
        } else if (e.name == "cellUserInfoEnableOverlay") {
            retval = 0;
        } else if (e.name == "cellUserInfoGetList") {
            // Write count=1 (one user) to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t one = __builtin_bswap32(1);
                std::memcpy(mem + (uint32_t)st.gpr[4], &one, 4);
            }
            retval = 0;

        // ── cellAdec (audio decoder) ─────────────────────────────
        } else if (e.name == "cellAdecOpen") {
            // Write handle to r5 pointer
            if (st.gpr[5] && st.gpr[5] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[5], &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellAdecClose") {
            retval = 0;
        } else if (e.name == "cellAdecStartSeq" || e.name == "cellAdecEndSeq") {
            retval = 0;
        } else if (e.name == "cellAdecDecodeAu") {
            retval = 0;
        } else if (e.name == "cellAdecGetPcm") {
            retval = (uint64_t)(int64_t)-1; // no audio data

        // ── cellDmux (demuxer) ───────────────────────────────────
        } else if (e.name == "cellDmuxOpen") {
            // Write handle to r5 pointer
            if (st.gpr[5] && st.gpr[5] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[5], &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellDmuxClose") {
            retval = 0;
        } else if (e.name == "cellDmuxSetStream" || e.name == "cellDmuxResetStream") {
            retval = 0;
        } else if (e.name == "cellDmuxEnableEs" || e.name == "cellDmuxDisableEs") {
            retval = 0;

        // ── cellAvconfExt — audio/video config ───────────────────────
        } else if (e.name == "cellVideoOutGetDeviceInfo") {
            // r3 = videoOut (0=PRIMARY), r4 = ptr to CellVideoOutDeviceInfo
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
                // portType=HDMI(1), colorSpace=RGB(1)
                mem[ptr] = 1;     // portType
                mem[ptr+4] = 1;   // colorSpace
            }
            retval = 0;
        } else if (e.name == "cellVideoOutGetNumberOfDevice") {
            retval = 1;  // 1 device (primary display)
        } else if (e.name == "cellVideoOutGetResolutionAvailability") {
            retval = 1;  // all resolutions "available"
        } else if (e.name == "cellVideoOutGetState") {
            // r3 = videoOut, r4 = deviceIndex, r5 = ptr to CellVideoOutState
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                // state=2 (ENABLED), colorMode=1 (32bit), resolutionId=2 (720p)
                mem[ptr]   = 2;   // state
                mem[ptr+4] = 1;   // colorMode
                mem[ptr+8] = 0; mem[ptr+9] = 2; // resolutionId=2 (720p) big-endian
            }
            retval = 0;
        } else if (e.name == "cellVideoOutConfigure") {
            retval = 0;  // success
        } else if (e.name == "cellAudioOutGetState") {
            // r3 = audioOut, r4 = deviceIndex, r5 = ptr to CellAudioOutState
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                mem[ptr] = 2;   // state = ENABLED
                mem[ptr+4] = 1; // encoder = LPCM
                mem[ptr+8] = 2; // channels = stereo
            }
            retval = 0;
        } else if (e.name == "cellAudioOutConfigure" ||
                   e.name == "cellAudioOutGetSoundAvailability") {
            retval = (e.name == "cellAudioOutGetSoundAvailability") ? 8 : 0;
        } else if (e.name == "cellAudioOutGetDeviceInfo") {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.name == "cellAudioOutGetNumberOfDevice") {
            retval = 1;  // 1 audio output

        // ── cellSync2 — enhanced sync primitives ─────────────────────
        } else if (e.name == "cellSync2MutexNew") {
            // r3 = ptr to mutex struct, zero-init it
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.name == "cellSync2MutexLock" ||
                   e.name == "cellSync2MutexUnlock" ||
                   e.name == "cellSync2MutexTryLock") {
            retval = 0;  // single-threaded: always succeeds
        } else if (e.name == "cellSync2QueueNew") {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 128 <= memSize) std::memset(mem + ptr, 0, 128);
            retval = 0;
        } else if (e.name == "cellSync2QueuePush" ||
                   e.name == "cellSync2QueueTryPush") {
            retval = 0;
        } else if (e.name == "cellSync2QueuePop" ||
                   e.name == "cellSync2QueueTryPop") {
            retval = (int32_t)0x80410101;  // CELL_SYNC2_ERROR_EMPTY
        } else if (e.name == "cellSync2QueueGetSize") {
            retval = 0;  // empty

        // ── cellVideoExport ──────────────────────────────────────────
        } else if (e.name == "cellVideoExportInitialize" ||
                   e.name == "cellVideoExportFinalize") {
            retval = 0;
        } else if (e.name == "cellVideoExportProgress") {
            retval = 100;  // 100% done (stub)

        // ── cellPhotoImport ──────────────────────────────────────────
        } else if (e.name == "cellPhotoImportInitialize" ||
                   e.name == "cellPhotoImportFinalize") {
            retval = 0;
        } else if (e.name == "cellPhotoImportSelectImage") {
            retval = (int32_t)0x80028b01;  // CELL_PHOTO_IMPORT_ERROR_CANCEL

        // ── cellNetCtlExt — extended network ─────────────────────────
        } else if (e.name == "cellNetCtlGetNatInfo") {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                mem[ptr] = 3;  // NAT type 3 (strict — offline)
            }
            retval = 0;
        } else if (e.name == "cellNetCtlGetIfAddr") {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                std::memcpy(mem + ptr, "0.0.0.0", 7);
            }
            retval = 0;
        } else if (e.name == "cellNetCtlNetStartDialogLoadAsync" ||
                   e.name == "cellNetCtlNetStartDialogAbortAsync" ||
                   e.name == "cellNetCtlNetStartDialogUnloadAsync") {
            retval = 0;

        // ── sys_io — controller I/O ──────────────────────────────────
        } else if (e.name == "sys_io_pad_get_capability") {
            retval = 0;  // no special capabilities
        } else if (e.name == "sys_io_pad_set_press_mode") {
            retval = 0;
        } else if (e.name == "sys_io_pad_clear_buf") {
            retval = 0;
        } else if (e.name == "sys_io_pad_get_data" ||
                   e.name == "sys_io_pad_get_data_extra") {
            // r3 = port, r4 = ptr to CellPadData (zero = no input)
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;

        // ── cellL10n — single-char conversion ────────────────────────
        } else if (e.name == "UTF8toUCS2") {
            // r3 = ptr to utf8 byte(s), r4 = ptr to ucs2 output
            // Simple: just copy first byte as UCS-2
            uint32_t src = (uint32_t)st.gpr[3];
            uint32_t dst = (uint32_t)st.gpr[4];
            if (src < memSize && dst + 2 <= memSize) {
                uint16_t c = (uint16_t)mem[src];
                mem[dst] = (uint8_t)(c >> 8);
                mem[dst+1] = (uint8_t)(c & 0xFF);
            }
            retval = 1;  // 1 byte consumed
        } else if (e.name == "UCS2toUTF8") {
            uint32_t src = (uint32_t)st.gpr[3];
            uint32_t dst = (uint32_t)st.gpr[4];
            if (src + 2 <= memSize && dst < memSize) {
                uint16_t c = ((uint16_t)mem[src] << 8) | mem[src+1];
                mem[dst] = (uint8_t)(c & 0x7F);
            }
            retval = 1;  // 1 byte written

        // ── cellGem (PlayStation Move) ───────────────────────────────
        } else if (e.name == "cellGemInit") {
            retval = 0;
        } else if (e.name == "cellGemEnd" || e.name == "cellGemReset" ||
                   e.name == "cellGemCalibrate" ||
                   e.name == "cellGemEnableCameraPitchAngleCorrection" ||
                   e.name == "cellGemUpdateStart" || e.name == "cellGemUpdateFinish") {
            retval = 0;
        } else if (e.name == "cellGemGetState") {
            // r3 = gem_num, r4 = ptr to state, r5 = flag
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = (int32_t)0x80121806;  // CELL_GEM_ERROR_NOT_CONNECTED
        } else if (e.name == "cellGemGetInfo") {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
                // max_connect=0 (no Move controllers)
            }
            retval = 0;
        } else if (e.name == "cellGemGetTrackerHue") {
            retval = 0;

        // ── cellMove ─────────────────────────────────────────────────
        } else if (e.name == "cellMoveInit" || e.name == "cellMoveEnd" ||
                   e.name == "cellMoveStart" || e.name == "cellMoveStop") {
            retval = 0;

        // ── cellOvis ─────────────────────────────────────────────────
        } else if (e.name == "cellOvisInitialize" || e.name == "cellOvisFinalize") {
            retval = 0;
        } else if (e.name == "cellOvisGetStatus") {
            retval = 0;

        // ── cellCamera ───────────────────────────────────────────────
        } else if (e.name == "cellCameraInit" || e.name == "cellCameraEnd") {
            retval = 0;
        } else if (e.name == "cellCameraOpen" || e.name == "cellCameraClose" ||
                   e.name == "cellCameraStart" || e.name == "cellCameraStop") {
            retval = 0;
        } else if (e.name == "cellCameraGetDeviceGUID") {
            retval = 0;
        } else if (e.name == "cellCameraGetType") {
            // r3 = devNum, r4 = ptr to type output
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t v = 0; // no camera
                std::memcpy(mem + ptr, &v, 4);
            }
            retval = 0;
        } else if (e.name == "cellCameraIsAvailable") {
            retval = 0;  // not available

        // ── cellResc (Resolution Scaler) ─────────────────────────────
        } else if (e.name == "cellRescInit") {
            retval = 0;
        } else if (e.name == "cellRescExit") {
            retval = 0;
        } else if (e.name == "cellRescSetDisplayMode" || e.name == "cellRescSetConvertAndFlip" ||
                   e.name == "cellRescSetBufferAddress" || e.name == "cellRescSetSrc" ||
                   e.name == "cellRescSetDsts") {
            retval = 0;
        } else if (e.name == "cellRescGetNumColorBuffers") {
            retval = 2;  // double buffer
        } else if (e.name == "cellRescGetFlipStatus") {
            retval = 0;  // ready
        } else if (e.name == "cellRescResetFlipStatus") {
            retval = 0;

        // ── cellPamf (PS3 media format) ──────────────────────────────
        } else if (e.name == "cellPamfReaderInitialize") {
            retval = 0;
        } else if (e.name == "cellPamfReaderGetNumberOfStreams") {
            retval = 0;  // no streams
        } else if (e.name == "cellPamfReaderGetStreamTypeCoding") {
            retval = 0;
        } else if (e.name == "cellPamfReaderSetStreamWithTypeAndIndex" ||
                   e.name == "cellPamfReaderGetStreamIndex") {
            retval = 0;
        } else if (e.name == "cellPamfReaderGetNumberOfSpecificStreams") {
            retval = 0;
        } else if (e.name == "cellPamfStreamGetInfo") {
            retval = 0;

        // ── sceNpUtil ────────────────────────────────────────────────
        } else if (e.name == "sceNpUtilBandwidthTestInitStart") {
            retval = 0;
        } else if (e.name == "sceNpUtilBandwidthTestGetStatus") {
            retval = 100;  // complete
        } else if (e.name == "sceNpUtilBandwidthTestShutdown") {
            retval = 0;

        // ── sceNpSignaling ───────────────────────────────────────────
        } else if (e.name == "sceNpSignalingCreateCtx") {
            // r3 = ptr to npId, r4 = handler, r5 = arg, r6 = ptr to ctxId
            uint32_t ptr = (uint32_t)st.gpr[6];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.name == "sceNpSignalingDestroyCtx") {
            retval = 0;
        } else if (e.name == "sceNpSignalingActivateConnection" ||
                   e.name == "sceNpSignalingDeactivateConnection") {
            retval = 0;
        } else if (e.name == "sceNpSignalingGetLocalNetInfo") {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;

        // ── sceNpClans ───────────────────────────────────────────────
        } else if (e.name == "sceNpClansInit" || e.name == "sceNpClansTerm") {
            retval = 0;
        } else if (e.name == "sceNpClansCreateRequest") {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.name == "sceNpClansDestroyRequest") {
            retval = 0;

        // ── cellSpursTrace ───────────────────────────────────────────
        } else if (e.name == "cellSpursTraceInitialize" ||
                   e.name == "cellSpursTraceFinalize" ||
                   e.name == "cellSpursTraceStart" ||
                   e.name == "cellSpursTraceStop") {
            retval = 0;

        // ── cellCrossController ──────────────────────────────────────
        } else if (e.name == "cellCrossControllerInitialize" ||
                   e.name == "cellCrossControllerFinalize") {
            retval = 0;

        // ── cellSysconf ──────────────────────────────────────────────
        } else if (e.name == "cellSysconfAbort") {
            retval = 0;
        } else if (e.name == "cellSysconfBtGetDeviceList") {
            // r3 = ptr to device list, zero-fill (no BT devices)
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;
        } else if (e.name == "cellSysconfGetIntegerVariable") {
            retval = 0;

        // ── cellMusicExport ──────────────────────────────────────────
        } else if (e.name == "cellMusicExportInitialize" ||
                   e.name == "cellMusicExportFinalize") {
            retval = 0;
        } else if (e.name == "cellMusicExportProgress") {
            retval = 100;

        // ── cellPhotoUtility ─────────────────────────────────────────
        } else if (e.name == "cellPhotoInitialize" ||
                   e.name == "cellPhotoFinalize") {
            retval = 0;
        } else if (e.name == "cellPhotoRegistFromFile") {
            retval = 0;

        // ── cellVdec (video decoder) ─────────────────────────────────
        } else if (e.name == "cellVdecOpen") {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellVdecClose" || e.name == "cellVdecEndSeq") {
            retval = 0;
        } else if (e.name == "cellVdecStartSeq") {
            retval = 0;
        } else if (e.name == "cellVdecDecodeAu") {
            retval = 0;
        } else if (e.name == "cellVdecGetPicture" || e.name == "cellVdecGetPictureExt") {
            retval = (int32_t)0x80610102;  // CELL_VDEC_ERROR_EMPTY
        } else if (e.name == "cellVdecSetFrameRate") {
            retval = 0;
        } else if (e.name == "cellVdecQueryAttr") {
            // r3 = type, r4 = ptr to CellVdecAttr (zero-fill with defaults)
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
                // memSize field at offset 4: 4MB default
                uint32_t ms = __builtin_bswap32(4 * 1024 * 1024);
                std::memcpy(mem + ptr + 4, &ms, 4);
            }
            retval = 0;

        // ── cellSysutil extras ───────────────────────────────────────
        } else if (e.name == "cellSysutilGetBgmPlaybackStatus" ||
                   e.name == "cellSysutilGetBgmPlaybackStatus2") {
            // r3 = ptr to status (0 = not playing)
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) std::memset(mem + ptr, 0, 4);
            retval = 0;
        } else if (e.name == "cellSysutilEnableBgmPlayback" ||
                   e.name == "cellSysutilDisableBgmPlayback") {
            retval = 0;
        } else if (e.name == "cellSysutilGetSystemParamFloat") {
            retval = 0;
        } else if (e.name == "cellSysutilCheckCallback") {
            retval = 0;  // no pending callbacks
        } else if (e.name == "cellSysutilRegisterCallbackDispatcher") {
            retval = 0;
        } else if (e.name == "cellSysutilGetLicenseArea") {
            retval = 1;  // area = SCEA (North America)

        // ── cellGcmSys extras ────────────────────────────────────────
        } else if (e.name == "cellGcmGetTimeStamp") {
            retval = 0;
        } else if (e.name == "cellGcmGetLastSecondVTime") {
            retval = 16667;  // ~60fps
        } else if (e.name == "cellGcmGetTiledPitchSize") {
            retval = 0;
        } else if (e.name == "cellGcmSetFlipImmediate") {
            retval = 0;
        } else if (e.name == "cellGcmSetZcull") {
            retval = 0;
        } else if (e.name == "cellGcmGetConfiguration") {
            // r3 = ptr to CellGcmConfig
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
                // localAddress at offset 0, ioAddress at offset 8
                // localSize = 256MB at offset 16
                uint32_t localSz = __builtin_bswap32(256 * 1024 * 1024);
                std::memcpy(mem + ptr + 16, &localSz, 4);
            }
            retval = 0;
        } else if (e.name == "cellGcmGetDisplayInfo") {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.name == "cellGcmSetFlipMode" ||
                   e.name == "cellGcmSetDefaultCommandBuffer") {
            retval = 0;
        } else if (e.name == "cellGcmBindTile" || e.name == "cellGcmUnbindTile") {
            retval = 0;
        } else if (e.name == "cellGcmMapMainMemory") {
            // r3 = ea, r4 = size, r5 = ptr to offset
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t off = __builtin_bswap32((uint32_t)st.gpr[3]);
                std::memcpy(mem + ptr, &off, 4);
            }
            retval = 0;
        } else if (e.name == "cellGcmUnmapEaIoAddress") {
            retval = 0;
        } else if (e.name == "cellGcmSetFlipHandler" ||
                   e.name == "cellGcmSetVBlankHandler") {
            retval = 0;

        // ── cellSpurs extras ─────────────────────────────────────────
        } else if (e.name == "cellSpursCreateTask" ||
                   e.name == "cellSpursCreateTaskset2") {
            retval = 0;
        } else if (e.name == "cellSpursJoinTask") {
            retval = 0;  // task already finished (stub)
        } else if (e.name == "cellSpursShutdownTaskset") {
            retval = 0;
        } else if (e.name == "cellSpursGetTasksetId") {
            retval = 0;
        } else if (e.name == "cellSpursTasksetAttributeSetName" ||
                   e.name == "cellSpursLookUpTasksetAddress") {
            retval = 0;
        } else if (e.name == "cellSpursEventFlagClear" ||
                   e.name == "cellSpursEventFlagSet") {
            retval = 0;
        } else if (e.name == "cellSpursEventFlagWait" ||
                   e.name == "cellSpursEventFlagTryWait") {
            retval = 0;  // single-threaded: event already signaled
        } else if (e.name == "cellSpursEventFlagAttachLv2EventQueue") {
            retval = 0;

        // ── cellGame extras ──────────────────────────────────────────
        } else if (e.name == "cellGameGetParamString") {
            // r3 = paramId, r4 = buf, r5 = bufSize
            uint32_t ptr = (uint32_t)st.gpr[4];
            uint32_t sz  = (uint32_t)st.gpr[5];
            if (ptr && sz > 0 && ptr + sz <= memSize) mem[ptr] = 0;
            retval = 0;
        } else if (e.name == "cellGameGetLocalWebContentPath") {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 64 <= memSize) {
                std::memset(mem + ptr, 0, 64);
                std::memcpy(mem + ptr, "/dev_hdd0/game/", 15);
            }
            retval = 0;
        } else if (e.name == "cellGameRegisterDiscChangeCallback") {
            retval = 0;
        } else if (e.name == "cellGameGetBgmPlaybackStatus") {
            retval = 0;

        // ── cellSaveData extras ──────────────────────────────────────
        } else if (e.name == "cellSaveDataListLoad2" ||
                   e.name == "cellSaveDataListSave2") {
            retval = 0;
        } else if (e.name == "cellSaveDataDelete2") {
            retval = 0;
        } else if (e.name == "cellSaveDataGetListItem") {
            retval = 0;

        // ── cellPad extras ───────────────────────────────────────────
        } else if (e.name == "cellPadGetRawData") {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;
        } else if (e.name == "cellPadSetActDirect") {
            retval = 0;  // rumble — no-op
        } else if (e.name == "cellPadPeriphGetInfo") {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;
        } else if (e.name == "cellPadGetDataExtra") {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;

        // ── cellKb (keyboard) ────────────────────────────────────────
        } else if (e.name == "cellKbInit") {
            retval = 0;
        } else if (e.name == "cellKbEnd") {
            retval = 0;
        } else if (e.name == "cellKbGetInfo") {
            // r3 = ptr to CellKbInfo (zero-fill = no keyboards)
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) std::memset(mem + ptr, 0, 16);
            retval = 0;
        } else if (e.name == "cellKbRead") {
            // r3 = port, r4 = ptr to CellKbData
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.name == "cellKbSetCodeType" ||
                   e.name == "cellKbSetReadMode" ||
                   e.name == "cellKbClearBuf") {
            retval = 0;

        // ── cellMouse ────────────────────────────────────────────────
        } else if (e.name == "cellMouseInit") {
            retval = 0;
        } else if (e.name == "cellMouseEnd") {
            retval = 0;
        } else if (e.name == "cellMouseGetInfo") {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) std::memset(mem + ptr, 0, 16);
            retval = 0;
        } else if (e.name == "cellMouseGetData") {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.name == "cellMouseClearBuf") {
            retval = 0;

        // ── cellJpgEnc ───────────────────────────────────────────────
        } else if (e.name == "cellJpgEncOpen") {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellJpgEncClose" || e.name == "cellJpgEncReset") {
            retval = 0;
        } else if (e.name == "cellJpgEncEncodeFully") {
            retval = 0;

        // ── cellHttp extras ──────────────────────────────────────────
        } else if (e.name == "cellHttpCreateTransaction") {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.name == "cellHttpDestroyTransaction") {
            retval = 0;
        } else if (e.name == "cellHttpSendRequest") {
            retval = (int32_t)0x80710002;  // CELL_HTTP_ERROR_NO_CONNECTION
        } else if (e.name == "cellHttpRecvResponse") {
            retval = (int32_t)0x80710002;
        } else if (e.name == "cellHttpGetResponseContentLength") {
            retval = 0;
        } else if (e.name == "cellHttpGetStatusCode") {
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
