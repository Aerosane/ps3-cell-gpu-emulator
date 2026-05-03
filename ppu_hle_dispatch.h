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
        if (e.fnid == 0x744680a2) {
            retval = 0;
        } else if (e.fnid == 0xe6f2c1e7) {
            halted_out = true;
            retval = 0;
        } else if (e.fnid == 0x2c847572 ||
                   e.fnid == 0x96328741) {
            // Registration of exit handlers during CRT init — not an
            // actual exit.
            retval = 0;
        } else if (e.fnid == 0x8461e528) {
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
        } else if (e.fnid == 0x2f85c0ef ||
                   e.fnid == 0x1573dc3f ||
                   e.fnid == 0x1bc200f4 ||
                   e.fnid == 0xc3476d0c) {
            retval = 0;
        } else if (e.fnid == 0x350d454e) {
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp + 8 <= memSize) {
                // big-endian u64 1
                std::memset(mem + outp, 0, 8);
                mem[outp + 7] = 1;
            }
            retval = 0;
        } else if (e.fnid == 0x42b23552) {
            retval = 0;
        } else if (e.fnid == 0x21ac3697) {
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
        } else if (e.fnid == 0xa547adde) {
            // Returns pointer to CellGcmControl { u32 put, get, ref }.
            if (!gcmControlInited && gcmControlRegPtr + 12 <= memSize) {
                std::memset(mem + gcmControlRegPtr, 0, 12);
                megakernel_write_mem((uint64_t)gcmControlRegPtr,
                                     mem + gcmControlRegPtr, 12);
                gcmControlInited = true;
            }
            retval = gcmControlRegPtr;
        } else if (e.fnid == 0x72a577ce) {
            // 0 => "last flip completed". Loop waits for this before
            // submitting the next frame. Always return 0 so we don't
            // spin forever.
            retval = 0;
        } else if (e.fnid == 0xb2e761d4) {
            retval = 0;
        } else if (e.fnid == 0x21397818 ||
                   e.fnid == 0xd9b7653e) {
            // Real HW writes a NV flip command into the FIFO. We don't
            // synthesise that packet (the FIFO replay happens separately),
            // but we do record the request so the host driver can emit an
            // onFlip barrier to the raster bridge after replay. r4 is the
            // buffer id; leave it for the consumer.
            gcmLastFlipBuffer = (uint32_t)st.gpr[4];
            gcmFlipRequests++;
            retval = 0;
        } else if (e.fnid == 0x15bae46b ||
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
        } else if (e.fnid == 0xa53d12ae ||
                   e.fnid == 0x4ae8d215 ||
                   e.fnid == 0x51c9d62b ||
                   e.fnid == 0x055bd74d ||
                   e.fnid == 0x4524cccd ||
                   e.fnid == 0xd9b7653e ||
                   e.fnid == 0xbd100dbc ||
                   e.fnid == 0x9dc04436 ||
                   e.fnid == 0xa75640e8) {
            retval = 0;
        } else if (e.fnid == 0x5e2ee0f0) {
            // Returns default command buffer size in words (0x400 = 1024 words = 4KB)
            retval = 0x400;
        } else if (e.fnid == 0x8cdf8c70) {
            // Returns default segment size in words (0x100 = 256 words = 1KB)
            retval = 0x100;
        } else if (e.fnid == 0x9ba451e4) {
            // r3 = cmdSize, r4 = ioSize — informational, we ignore
            retval = 0;
        } else if (e.fnid == 0x3a33c1fd) {
            // Internal init helper (sets up context default state). No-op.
            retval = 0;
        } else if (e.fnid == 0xacee8542) {
            // Inserts a wait-for-flip-complete into FIFO. Since we process
            // FIFO synchronously, this is a no-op.
            retval = 0;
        } else if (e.fnid == 0x723c5c6c) {
            // Same as SetFlipCommand — queue a flip request.
            gcmLastFlipBuffer = (uint32_t)st.gpr[4];
            gcmFlipRequests++;
            retval = 0;
        } else if (e.fnid == 0x9a4c1b5f) {
            // r3 = callback function pointer. We don't invoke it; store for future use.
            retval = 0;
        } else if (e.fnid == 0x0e6b0dae) {
            // Returns 0 (progressive scan — no interlaced field).
            retval = 0;
        } else if (e.fnid == 0xe315a0b2) {
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
        } else if (e.fnid == 0xf80196c1) {
            // r3 = label index. Returns pointer to a u32 "label" that
            // the RSX writes to signal flip completion, etc.
            uint32_t idx = (uint32_t)st.gpr[3];
            uint32_t addr = gcmLabelBase + idx * 16;
            if (addr + 4 <= memSize) {
                std::memset(mem + addr, 0, 4);
                megakernel_write_mem((uint64_t)addr, mem + addr, 4);
            }
            retval = addr;
        } else if (e.fnid == 0x5a41c10f) {
            retval = 0;
        } else if (e.fnid == 0xa114ec67) {
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
        } else if (e.fnid == 0x1cf98800 ||
                   e.fnid == 0x4d9b75d5) {
            retval = 0;
        } else if (e.fnid == 0xa703a51d) {
            // r3 = *CellPadInfo2. Zero it (no pads connected).
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp + 64 <= memSize) {
                std::memset(mem + outp, 0, 64);
                megakernel_write_mem((uint64_t)outp, mem + outp, 64);
            }
            retval = 0;
        } else if (e.fnid == 0x8b72cda1) {
            // r3 = port, r4 = *CellPadData. Zero it (no buttons).
            uint32_t outp = (uint32_t)st.gpr[4];
            if (outp + 64 <= memSize) {
                std::memset(mem + outp, 0, 64);
                megakernel_write_mem((uint64_t)outp, mem + outp, 64);
            }
            retval = 0;
        } else if (e.fnid == 0x2d36462b) {
            // r3 = pointer to string, returns length.
            uint32_t ptr = (uint32_t)st.gpr[3];
            uint32_t len = 0;
            if (ptr < memSize) {
                while (ptr + len < memSize && mem[ptr + len] != 0 && len < 0x10000)
                    len++;
            }
            retval = len;
        } else if (e.fnid == 0x9f04f7af) {
            // r3 = format string pointer. Just consume it, no real output.
            retval = 0;
        } else if (e.fnid == 0xb2fcf2c8 ||
                   e.fnid == 0xaede4b03) {
            // Return a fake heap handle (non-zero for create).
            retval = (e.fnid == 0xb2fcf2c8) ? 0x00880000 : 0;
        } else if (e.fnid == 0x35168520) {
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
        } else if (e.fnid == 0x8a561d92) {
            retval = 0;
        } else if (e.fnid == 0x67f9fedb) {
            halted_out = true;
            retval = 0;
        } else if (e.fnid == 0x0bae8772 ||
                   e.fnid == 0xe558748d ||
                   e.fnid == 0x887572d5) {
            if (e.fnid == 0x887572d5) {
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
            } else if (e.fnid == 0xe558748d) {
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
        } else if (e.fnid == 0x9d98afa0 ||
                   e.fnid == 0x02ff3c1b ||
                   e.fnid == 0x189a74da) {
            retval = 0;
        } else if (e.fnid == 0x40e895d3) {
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
        } else if (e.fnid == 0x938013a0) {
            // r3 = videoOut, r4 = resolutionId, r5 = aspect, r6 = option
            // Return 1 (available) for 1080p (resId=2) and 720p (resId=4)
            uint32_t resId = (uint32_t)st.gpr[4];
            retval = (resId == 2 || resId == 4 || resId == 1) ? 1 : 0;
        } else if (e.fnid == 0x718bf5f8) {
            // r3=path, r4=flags, r5=fd_out, ...
            uint32_t fdOut = (uint32_t)st.gpr[5];
            if (fdOut + 4 <= memSize) {
                uint8_t be[4] = { 0, 0, 0, 3 };  // arbitrary fd #3
                std::memcpy(mem + fdOut, be, 4);
                megakernel_write_mem((uint64_t)fdOut, be, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xecdcf2ab) {
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
        } else if (e.fnid == 0x4d5ff8e2) {
            uint32_t nread = (uint32_t)st.gpr[6];
            if (nread && nread + 4 <= memSize) {
                std::memset(mem + nread, 0, 4);
                megakernel_write_mem((uint64_t)nread, mem + nread, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x2cb51f0d ||
                   e.fnid == 0xa397d042) {
            retval = 0;

        // ── cellSpurs — SPURS task scheduler stubs ──────────────
        // These return CELL_OK (0) to let games that initialize SPURS
        // proceed past boot. Actual SPU task dispatch is handled by
        // the SPU JIT turbo/hyper multi-instance engine.
        } else if (e.fnid == 0x70e3d58a ||
                   e.fnid == 0xb0c0d66a ||
                   e.fnid == 0xf7234b81) {
            // r3 = CellSpurs* spurs, r4 = numSpu / CellSpursAttribute*
            // Zero-fill the SPURS structure so games see initialized state
            uint32_t spursPtr = (uint32_t)st.gpr[3];
            if (spursPtr && spursPtr + 0x80 <= memSize) {
                std::memset(mem + spursPtr, 0, 0x80);
                megakernel_write_mem(spursPtr, mem + spursPtr, 0x80);
            }
            retval = 0;
        } else if (e.fnid == 0xe48bf572) {
            retval = 0;
        } else if (e.fnid == 0x47b25489) {
            // r3 = CellSpursAttribute* attr, r4 = revision, r5 = sdkVersion,
            // r6 = nSpus, r7 = spuPriority, r8 = ppuPriority
            uint32_t attrPtr = (uint32_t)st.gpr[3];
            if (attrPtr && attrPtr + 0x80 <= memSize) {
                std::memset(mem + attrPtr, 0, 0x80);
                megakernel_write_mem(attrPtr, mem + attrPtr, 0x80);
            }
            retval = 0;
        } else if (e.fnid == 0x6c01b4fb ||
                   e.fnid == 0x3f422300 ||
                   e.fnid == 0xc23c4581 ||
                   e.fnid == 0x3942f6bf ||
                   e.fnid == 0x79b36c3e ||
                   e.fnid == 0xc41beb33) {
            retval = 0;
        } else if (e.fnid == 0x1bf8a000) {
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
        } else if (e.fnid == 0xe2df437e) {
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
        } else if (e.fnid == 0xbc7001af ||
                   e.fnid == 0xe74dc23c) {
            retval = 0;
        } else if (e.fnid == 0x3fd6c11a ||
                   e.fnid == 0xefa6c145) {
            // r3 = CellSpurs*, r4 = CellSpursTaskset*, ...
            uint32_t tsPtr = (uint32_t)st.gpr[4];
            if (tsPtr && tsPtr + 0x80 <= memSize) {
                std::memset(mem + tsPtr, 0, 0x80);
                megakernel_write_mem(tsPtr, mem + tsPtr, 0x80);
            }
            retval = 0;
        } else if (e.fnid == 0xc97cec02) {
            retval = 0;
        } else if (e.fnid == 0xe591f650) {
            // Blocking wait — return immediately (tasks "completed")
            retval = 0;
        } else if (e.fnid == 0x899934dc) {
            retval = 0;
        } else if (e.fnid == 0xe7fc14af) {
            retval = 0;
        } else if (e.fnid == 0xbc3dab02) {
            // r3 = CellSpurs*, r4 = CellSpursEventFlag*, ...
            uint32_t efPtr = (uint32_t)st.gpr[4];
            if (efPtr && efPtr + 0x40 <= memSize) {
                std::memset(mem + efPtr, 0, 0x40);
                megakernel_write_mem(efPtr, mem + efPtr, 0x40);
            }
            retval = 0;
        } else if (e.fnid == 0x36c4f4a8 ||
                   e.fnid == 0x2f0a4998) {
            retval = 0;
        } else if (e.fnid == 0xe82a263a) {
            // Blocking wait — return immediately (flag "set")
            retval = 0;
        } else if (e.fnid == 0x144d8656) {
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
        } else if (e.fnid == 0x188beb05) {
            // r3 = CellGameContentSize* size (out), r4 = dirName (out)
            // Return CELL_GAME_RET_OK (=0), set type=CELL_GAME_GAMETYPE_DISC(1)
            retval = 0;
        } else if (e.fnid == 0xd0536716) {
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
        } else if (e.fnid == 0x4cbca5b6 ||
                   e.fnid == 0x405e5cba ||
                   e.fnid == 0xfadd9ad8 ||
                   e.fnid == 0xf2364153 ||
                   e.fnid == 0xc707e826) {
            retval = 0;
        } else if (e.fnid == 0x0fa06f6d) {
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
        } else if (e.fnid == 0xde9c0881) {
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
        } else if (e.fnid == 0xd1a90b31) {
            // r3 = CellGameContentSize* size (out)
            uint32_t outPtr = (uint32_t)st.gpr[3];
            if (outPtr && outPtr + 12 <= memSize) {
                std::memset(mem + outPtr, 0, 12);
                megakernel_write_mem(outPtr, mem + outPtr, 12);
            }
            retval = 0;

        // ── cellSaveData — save/load stubs ──────────────────────
        } else if (e.fnid == 0xf0f530b7 ||
                   e.fnid == 0x590e6d0b ||
                   e.fnid == 0x3604d4f4 ||
                   e.fnid == 0xd739cc4b ||
                   e.fnid == 0x38a0f7d2 ||
                   e.fnid == 0x7b5e041a ||
                   e.fnid == 0x1c8b05e2) {
            // Return CELL_SAVEDATA_RET_OK (0) — pretend save succeeded
            retval = 0;

        // ── cellAudio — audio port stubs ────────────────────────
        } else if (e.fnid == 0x980f750b ||
                   e.fnid == 0x81df6d86) {
            retval = 0;
        } else if (e.fnid == 0x708017e5) {
            // r3 = CellAudioPortParam*, r4 = uint32_t* portNum (out)
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 4 <= memSize) {
                uint8_t be[4] = { 0, 0, 0, 0 };  // port 0
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xebb73ea7 ||
                   e.fnid == 0x8a8c0417 ||
                   e.fnid == 0x3ba2ba64 ||
                   e.fnid == 0x20e88c87 ||
                   e.fnid == 0x83afa9b3) {
            retval = 0;
        } else if (e.fnid == 0xffd2b376) {
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
        } else if (e.fnid == 0x69792b3b) {
            // r3 = portNum, r4 = tag, r5 = uint64_t* stamp (out)
            uint32_t outPtr = (uint32_t)st.gpr[5];
            if (outPtr && outPtr + 8 <= memSize) {
                std::memset(mem + outPtr, 0, 8);
                megakernel_write_mem(outPtr, mem + outPtr, 8);
            }
            retval = 0;

        // ── cellFont — font rendering stubs ─────────────────────
        } else if (e.fnid == 0x4b734c8c ||
                   e.fnid == 0x3e3712ed ||
                   e.fnid == 0x62f4f193 ||
                   e.fnid == 0x21ef94ac ||
                   e.fnid == 0x36149928 ||
                   e.fnid == 0xa19dbfba ||
                   e.fnid == 0x881e825b ||
                   e.fnid == 0x6dc15d05 ||
                   e.fnid == 0x3085a953) {
            retval = 0;
        } else if (e.fnid == 0x4b02499c) {
            // Zero-fill the output glyph image (blank glyph)
            retval = 0;
        } else if (e.fnid == 0xb61f6678) {
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
        } else if (e.fnid == 0x61717f26) {
            // r3 = font*, r4 = code, r5 = CellFontGlyphMetrics* (out)
            uint32_t outPtr = (uint32_t)st.gpr[5];
            if (outPtr && outPtr + 32 <= memSize) {
                std::memset(mem + outPtr, 0, 32);
                megakernel_write_mem(outPtr, mem + outPtr, 32);
            }
            retval = 0;

        // ── sceNpTrophy — trophy system stubs ───────────────────
        } else if (e.fnid == 0xdd74bdae ||
                   e.fnid == 0xd972691b ||
                   e.fnid == 0x0cde7100 ||
                   e.fnid == 0x7775c461) {
            retval = 0;
        } else if (e.fnid == 0x7aa2fcff) {
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
        } else if (e.fnid == 0xc10b8fce) {
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
        } else if (e.fnid == 0x287f6018 ||
                   e.fnid == 0x7a3ceb91) {
            retval = 0;
        } else if (e.fnid == 0xc868dea5) {
            // r3 = ctx, r4 = handle, r5 = int32_t* percentage (out)
            uint32_t outPtr = (uint32_t)st.gpr[5];
            if (outPtr && outPtr + 4 <= memSize) {
                uint8_t be[4] = { 0, 0, 0, 0 };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x3fbae39e) {
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
        } else if (e.fnid == 0xa83f253b || e.fnid == 0x9afd0b7d) {
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
        } else if (e.fnid == 0xb6d5cacd ||
                   e.fnid == 0x59f481ef ||
                   e.fnid == 0x9625f90a ||
                   e.fnid == 0xc1901ab8 ||
                   e.fnid == 0x5877b8ab) {
            retval = 0;
        } else if (e.fnid == 0xa1ffa426) {
            // r3 = CellVdecType*, r4 = CellVdecAttr* (out)
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 32 <= memSize) {
                std::memset(mem + outPtr, 0, 32);
                megakernel_write_mem(outPtr, mem + outPtr, 32);
            }
            retval = 0;
        } else if (e.fnid == 0xac73ada8 ||
                   e.fnid == 0x52b9a8b0) {
            // Return CELL_VDEC_ERROR_EMPTY (-1) to indicate no picture ready.
            // Games check return value and skip frame display.
            retval = (uint32_t)(-1);

        // ── cellFs extras ──────────────────────────────────────────────
        } else if (e.fnid == 0x9eaf0f23 || e.fnid == 0xe1464921 ||
                   e.fnid == 0x49e735fd) {
            // Directory stubs — return ENOENT for opendir, 0 for others
            if (e.fnid == 0x9eaf0f23) {
                retval = 0x80010006;  // CELL_FS_ENOENT
            } else {
                retval = 0;
            }
        } else if (e.fnid == 0x0a7faeba) {
            // Return ENOENT — file not found
            retval = 0x80010006;
        } else if (e.fnid == 0x05b82844 || e.fnid == 0xadff8f10 ||
                   e.fnid == 0x40ba070f || e.fnid == 0x3591cf02 ||
                   e.fnid == 0x60a6c70f) {
            retval = 0;  // succeed silently
        } else if (e.fnid == 0xe7d6ba00) {
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
        } else if (e.fnid == 0xb8a0bf48 ||
                   e.fnid == 0x2a1321c1) {
            retval = 0;  // Always succeed
        } else if (e.fnid == 0x452adbc2) {
            retval = 0;  // CELL_SYSMODULE_LOADED
        } else if (e.fnid == 0x6e0040b4 ||
                   e.fnid == 0x8b59a7b1) {
            retval = 0;

        // ── cellNetCtl ─────────────────────────────────────────────────
        } else if (e.fnid == 0xf53f04bb || e.fnid == 0x9f18ccad) {
            retval = 0;
        } else if (e.fnid == 0x5c413ca9) {
            // r3 = state* (out) — set to DISCONNECTED (0)
            uint32_t statePtr = (uint32_t)st.gpr[3];
            if (statePtr && statePtr + 4 <= memSize) {
                uint32_t state = 0;  // CELL_NET_CTL_STATE_Disconnected
                uint8_t be[4] = { 0, 0, 0, 0 };
                std::memcpy(mem + statePtr, be, 4);
                megakernel_write_mem(statePtr, be, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x08a8a347) {
            // Return error — not connected
            retval = 0x80130106;  // CELL_NET_CTL_ERROR_NOT_CONNECTED
        } else if (e.fnid == 0x2e274b74 ||
                   e.fnid == 0x438cee7b) {
            retval = 0;

        // ── cellMsgDialog ──────────────────────────────────────────────
        } else if (e.fnid == 0x503d1913 ||
                   e.fnid == 0x860634da) {
            // Immediately "close" dialog — game callback gets DIALOG_CLOSE
            retval = 0;
        } else if (e.fnid == 0x7da3e04f ||
                   e.fnid == 0xf0d11ff1 ||
                   e.fnid == 0xbb49b97e ||
                   e.fnid == 0x6df96f85) {
            retval = 0;

        // ── cellOskDialog ──────────────────────────────────────────────
        } else if (e.fnid == 0x36cff0d8 ||
                   e.fnid == 0xf5daa620 ||
                   e.fnid == 0x15cb6934) {
            retval = 0;
        } else if (e.fnid == 0xcae76f24) {
            // r3 = buffer*, r4 = size — write empty string
            uint32_t bufPtr = (uint32_t)st.gpr[3];
            if (bufPtr && bufPtr + 2 <= memSize) {
                mem[bufPtr] = 0; mem[bufPtr+1] = 0;
                megakernel_write_mem(bufPtr, mem + bufPtr, 2);
            }
            retval = 0;

        // ── cellResc — resolution scaling ──────────────────────────────
        } else if (e.fnid == 0x3711bd4f || e.fnid == 0xcb4ba814 ||
                   e.fnid == 0x089a5067 ||
                   e.fnid == 0xffd96327 ||
                   e.fnid == 0x770f5ac4 ||
                   e.fnid == 0x147dde14 || e.fnid == 0xbe0760f8) {
            retval = 0;
        } else if (e.fnid == 0x35239700) {
            retval = 2;  // 2 color buffers (double-buffering)
        } else if (e.fnid == 0xe99bbc60) {
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
        } else if (e.fnid == 0x799b5fae ||
                   e.fnid == 0xda004916) {
            retval = 0;
        } else if (e.fnid == 0xfc6fe574) {
            // r3 = *status (out) — set to 0 (not playing)
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp && outp + 4 <= memSize) {
                std::memset(mem + outp, 0, 4);
                megakernel_write_mem(outp, mem + outp, 4);
            }
            retval = 0;

        // ── cellKb / cellMouse / cellCamera ────────────────────────────
        } else if (e.fnid == 0x85656de9 || e.fnid == 0x8fe7f827 ||
                   e.fnid == 0x890c9fb8 || e.fnid == 0x67c67673 ||
                   e.fnid == 0x30eaee99 || e.fnid == 0xf1e65d8d) {
            retval = 0;
        } else if (e.fnid == 0xead11ae9 || e.fnid == 0xb4086711) {
            // r3 = *info (out) — zero it (no devices connected)
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp && outp + 32 <= memSize) {
                std::memset(mem + outp, 0, 32);
                megakernel_write_mem(outp, mem + outp, 32);
            }
            retval = 0;
        } else if (e.fnid == 0xda870fc8) {
            // r3 = port, r4 = *data (out) — zero it (no key events)
            uint32_t outp = (uint32_t)st.gpr[4];
            if (outp && outp + 16 <= memSize) {
                std::memset(mem + outp, 0, 16);
                megakernel_write_mem(outp, mem + outp, 16);
            }
            retval = 0;

        // ── cellUserInfo ───────────────────────────────────────────────
        } else if (e.fnid == 0xdcf8eb53) {
            // r3 = userId, r4 = *CellUserInfoStat (out) — zero it
            uint32_t outp = (uint32_t)st.gpr[4];
            if (outp && outp + 64 <= memSize) {
                std::memset(mem + outp, 0, 64);
                megakernel_write_mem(outp, mem + outp, 64);
            }
            retval = 0;
        } else if (e.fnid == 0x4a5b749a) {
            // r3 = *listNum (out), r4 = *listBuf (out) — zero (no users)
            uint32_t numPtr = (uint32_t)st.gpr[3];
            if (numPtr && numPtr + 4 <= memSize) {
                std::memset(mem + numPtr, 0, 4);
                megakernel_write_mem(numPtr, mem + numPtr, 4);
            }
            retval = 0;

        // ── cellSsl / cellHttp ─────────────────────────────────────────
        } else if (e.fnid == 0x8c1678ef || e.fnid == 0x2a574a25 ||
                   e.fnid == 0xf7644439 || e.fnid == 0x33457271) {
            retval = 0;

        // ── cellAdec — audio decoder ───────────────────────────────────
        } else if (e.fnid == 0x8f027e01) {
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
        } else if (e.fnid == 0xcd96044f || e.fnid == 0xcc534abe ||
                   e.fnid == 0xccfeefc2 || e.fnid == 0x4d682b55) {
            retval = 0;
        } else if (e.fnid == 0x0cfa3baf || e.fnid == 0x382b0cf4) {
            retval = (uint32_t)(-1);  // No PCM ready
        } else if (e.fnid == 0xf07a2ec7) {
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 32 <= memSize) {
                std::memset(mem + outPtr, 0, 32);
                megakernel_write_mem(outPtr, mem + outPtr, 32);
            }
            retval = 0;

        // ── cellDmux — demuxer ─────────────────────────────────────────
        } else if (e.fnid == 0x2d4ce3df) {
            uint32_t outPtr = (uint32_t)st.gpr[5];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t h = 3;  // dummy demux handle
                uint8_t be[4] = { (uint8_t)(h>>24), (uint8_t)(h>>16),
                                  (uint8_t)(h>>8),  (uint8_t)h };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x438a5a0f || e.fnid == 0xcd076abf ||
                   e.fnid == 0xabbbb443 || e.fnid == 0x181d1978 ||
                   e.fnid == 0x022dda52) {
            retval = 0;
        } else if (e.fnid == 0x7d539c24) {
            retval = (uint32_t)(-1);  // No AU available
        } else if (e.fnid == 0x143a3968) {
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 32 <= memSize) {
                std::memset(mem + outPtr, 0, 32);
                megakernel_write_mem(outPtr, mem + outPtr, 32);
            }
            retval = 0;

        // ── cellPamf — PAMF container ──────────────────────────────────
        } else if (e.fnid == 0xb13dac24) {
            retval = 0;
        } else if (e.fnid == 0x484cfb6f) {
            retval = 0;  // 0 streams (no media)
        } else if (e.fnid == 0xbf6c31af ||
                   e.fnid == 0x79df04e4 ||
                   e.fnid == 0x847d96d4) {
            retval = 0;
        } else if (e.fnid == 0xd245f601) {
            retval = 2048;  // Typical PAMF header size

        // ── cellJpgDec / cellPngDec ────────────────────────────────────
        } else if (e.fnid == 0xb4fdda24 || e.fnid == 0xf8d2000f) {
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
        } else if (e.fnid == 0xd1f838b9 || e.fnid == 0x7515cc92 ||
                   e.fnid == 0x8b18c8eb || e.fnid == 0xbbbfa42c ||
                   e.fnid == 0xebeab923 || e.fnid == 0x01c158db) {
            retval = 0;
        } else if (e.fnid == 0x5920178f || e.fnid == 0x8cdae535) {
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
        } else if (e.fnid == 0xc2a0d674 || e.fnid == 0x994e6188) {
            // Fill output buffer with white pixels (ARGB 0xFFFFFFFF)
            retval = 0;

        // ── cellL10n — string conversion ───────────────────────────────
        } else if (e.fnid == 0x49a66a63 || e.fnid == 0xc5925b36 ||
                   e.fnid == 0x9e989095 || e.fnid == 0x487b4dc1 ||
                   e.fnid == 0xa5be179f || e.fnid == 0x13e7cfc1 ||
                   e.fnid == 0x3b95766d) {
            // String conversion stubs — return 0 (success).
            // Real impl would convert between encodings. Games usually
            // just pass ASCII through these which works even without conversion.
            retval = 0;

        // ── cellSync — synchronization ─────────────────────────────────
        } else if (e.fnid == 0x2733a870 ||
                   e.fnid == 0x24f80472 ||
                   e.fnid == 0xf8ef9193) {
            // r3 = *object — zero-init the sync primitive
            uint32_t outp = (uint32_t)st.gpr[3];
            if (outp && outp + 32 <= memSize) {
                std::memset(mem + outp, 0, 32);
                megakernel_write_mem(outp, mem + outp, 32);
            }
            retval = 0;
        } else if (e.fnid == 0x91e6dcb9 || e.fnid == 0xc463ee6c ||
                   e.fnid == 0xb9fbcc28 ||
                   e.fnid == 0xcb40434b ||
                   e.fnid == 0xbb54b37f ||
                   e.fnid == 0x3c81b4da) {
            retval = 0;  // Always succeed (single-threaded emulation)
        } else if (e.fnid == 0x8c497d76 || e.fnid == 0x0db15740 ||
                   e.fnid == 0xe5c224e2 || e.fnid == 0x65f2b02f) {
            retval = 0;
        } else if (e.fnid == 0x8558da70) {
            retval = 0;  // Queue is empty

        // ── cellGifDec ─────────────────────────────────────────────────
        } else if (e.fnid == 0x66315b6d) {
            uint32_t outPtr = (uint32_t)st.gpr[3];
            if (outPtr && outPtr + 4 <= memSize) {
                uint32_t h = 5;
                uint8_t be[4] = { (uint8_t)(h>>24), (uint8_t)(h>>16),
                                  (uint8_t)(h>>8),  (uint8_t)h };
                std::memcpy(mem + outPtr, be, 4);
                megakernel_write_mem(outPtr, be, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xfe153bbf || e.fnid == 0xec45d754 ||
                   e.fnid == 0x11ae5215) {
            retval = 0;
        } else if (e.fnid == 0xc52c449b) {
            uint32_t outPtr = (uint32_t)st.gpr[4];
            if (outPtr && outPtr + 16 <= memSize) {
                std::memset(mem + outPtr, 0, 16);
                mem[outPtr+3] = 64; mem[outPtr+7] = 64;  // 64×64
                megakernel_write_mem(outPtr, mem + outPtr, 16);
            }
            retval = 0;
        } else if (e.fnid == 0x0c0c12bd) {
            retval = 0;

        // ═══ sceNpTrophy — PSN trophy system ═══
        } else if (e.fnid == 0xdd74bdae) {
            retval = 0;
        } else if (e.fnid == 0xd972691b) {
            retval = 0;
        } else if (e.fnid == 0xc10b8fce) {
            // r3 = handle_ptr
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.fnid == 0x7775c461) {
            retval = 0;
        } else if (e.fnid == 0x7aa2fcff) {
            // r3 = context_ptr
            uint32_t cid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(cid >> 24); mem[ptr+1] = (uint8_t)(cid >> 16);
                mem[ptr+2] = (uint8_t)(cid >> 8); mem[ptr+3] = (uint8_t)cid;
            }
            retval = 0;
        } else if (e.fnid == 0x0cde7100) {
            retval = 0;
        } else if (e.fnid == 0x287f6018) {
            retval = 0;
        } else if (e.fnid == 0x99d5fcdb) {
            // r3 = context, r4 = handle, r5 = reqspace_ptr
            uint64_t ptr = st.gpr[5];
            if (ptr && ptr + 8 <= memSize) {
                // 1MB required space
                uint64_t space = 0x100000ULL;
                for (int i = 0; i < 8; i++) mem[ptr+i] = (uint8_t)(space >> (56 - 8*i));
            }
            retval = 0;
        } else if (e.fnid == 0x3fbae39e) {
            // r5 = game_info_ptr — zero-fill the structure
            uint64_t ptr = st.gpr[5];
            if (ptr && ptr + 64 <= memSize) {
                std::memset(mem + ptr, 0, 64);
            }
            retval = 0;
        } else if (e.fnid == 0x7a3ceb91) {
            // r3 = context, r4 = handle, r5 = trophy_id, r6 = platinum_id_ptr
            uint64_t ptr = st.gpr[6];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0xFF; mem[ptr+1] = 0xFF; mem[ptr+2] = 0xFF; mem[ptr+3] = 0xFF; // -1 = no platinum
            }
            retval = 0;
        } else if (e.fnid == 0xdd29bb02) {
            // r5 = info_ptr
            uint64_t ptr = st.gpr[5];
            if (ptr && ptr + 48 <= memSize) {
                std::memset(mem + ptr, 0, 48);
            }
            retval = 0;
        } else if (e.fnid == 0x74b16e87) {
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
        } else if (e.fnid == 0xc868dea5) {
            // r5 = progress_ptr (int32 percentage)
            uint64_t ptr = st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0; mem[ptr+1] = 0; mem[ptr+2] = 0; mem[ptr+3] = 0; // 0%
            }
            retval = 0;

        // ═══ sceNp — PSN base ═══
        } else if (e.fnid == 0xd59da152) {
            retval = 0;
        } else if (e.fnid == 0xec72d838) {
            retval = 0;
        } else if (e.fnid == 0xd1fcb865 || e.fnid == 0xebae6c1a) {
            // r3 = npid_ptr — fill with dummy offline NP ID
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 36 <= memSize) {
                std::memset(mem + ptr, 0, 36);
                // Write "OfflineUser" as the handle
                const char* handle = "OfflineUser";
                for (int i = 0; handle[i]; i++) mem[ptr + i] = handle[i];
            }
            retval = 0;
        } else if (e.fnid == 0x259ab84f || e.fnid == 0x54bcafb8) {
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 20 <= memSize) {
                std::memset(mem + ptr, 0, 20);
                const char* oid = "OfflineUser";
                for (int i = 0; oid[i]; i++) mem[ptr + i] = oid[i];
            }
            retval = 0;
        } else if (e.fnid == 0x2aba00d1) {
            retval = 0;
        } else if (e.fnid == 0x9b98354e) {
            // Return SCE_NP_MANAGER_STATUS_OFFLINE (1) via r4
            st.gpr[4] = 1;
            retval = 0;
        } else if (e.fnid == 0x3566cd38) {
            // r3 = time_ptr (CellRtcTick - uint64_t)
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 8 <= memSize) {
                uint64_t tick = 0x000DC46C0D3B3600ULL; // ~2024 epoch
                for (int i = 0; i < 8; i++) mem[ptr+i] = (uint8_t)(tick >> (56 - 8*i));
            }
            retval = 0;
        } else if (e.fnid == 0x13079e64 || e.fnid == 0x39853605) {
            retval = 0;
        } else if (e.fnid == 0xaaeae4a9 || e.fnid == 0x66da2edd) {
            retval = 0;

        // ═══ cellRtc — real-time clock ═══
        } else if (e.fnid == 0x9148aab9) {
            // r3 = CellRtcTick* (uint64_t big-endian)
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 8 <= memSize) {
                // ~2024 epoch in microseconds from 0001-01-01
                uint64_t tick = 0x000DC46C0D3B3600ULL + virtTime;
                for (int i = 0; i < 8; i++) mem[ptr+i] = (uint8_t)(tick >> (56 - 8*i));
            }
            virtTime += 16667; // advance ~1 frame
            retval = 0;
        } else if (e.fnid == 0x27ffb3e3 || e.fnid == 0xd85c3c42) {
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
        } else if (e.fnid == 0x18466b0e || e.fnid == 0xead86714) {
            // r3 = src tick*, r4 = dst tick*  — just copy (no timezone offset)
            uint64_t src = st.gpr[3], dst = st.gpr[4];
            if (src && dst && src + 8 <= memSize && dst + 8 <= memSize) {
                std::memcpy(mem + dst, mem + src, 8);
            }
            retval = 0;
        } else if (e.fnid == 0x9c1aa28d) {
            // r3 = CellRtcDateTime*, r4 = CellRtcTick* — fake tick output
            uint64_t ptr = st.gpr[4];
            if (ptr && ptr + 8 <= memSize) {
                uint64_t tick = 0x000DC46C0D3B3600ULL;
                for (int i = 0; i < 8; i++) mem[ptr+i] = (uint8_t)(tick >> (56 - 8*i));
            }
            retval = 0;
        } else if (e.fnid == 0x7767eeb3) {
            retval = 0;
        } else if (e.fnid == 0xf8cf8759 || e.fnid == 0x17947909) {
            // r3 = dst tick*, r4 = src tick*, r5 = delta
            uint64_t dst = st.gpr[3], src = st.gpr[4];
            if (src && dst && src + 8 <= memSize && dst + 8 <= memSize) {
                uint64_t tick = 0;
                for (int i = 0; i < 8; i++) tick = (tick << 8) | mem[src+i];
                uint64_t delta = st.gpr[5];
                if (e.fnid == 0x17947909) delta *= 60;
                tick += delta * 1000000ULL;
                for (int i = 0; i < 8; i++) mem[dst+i] = (uint8_t)(tick >> (56 - 8*i));
            }
            retval = 0;
        } else if (e.fnid == 0x9ab17608) {
            retval = 0; // no-op (buffer left empty)
        } else if (e.fnid == 0x49119def) {
            uint32_t year = (uint32_t)st.gpr[3];
            st.gpr[4] = ((year % 4 == 0) && (year % 100 != 0 || year % 400 == 0)) ? 1 : 0;
            retval = 0;
        } else if (e.fnid == 0x2bb6158e) {
            uint32_t year = (uint32_t)st.gpr[3];
            uint32_t month = (uint32_t)st.gpr[4];
            static const int days[] = {31,28,31,30,31,30,31,31,30,31,30,31};
            int d = (month >= 1 && month <= 12) ? days[month-1] : 30;
            if (month == 2 && ((year%4==0) && (year%100!=0 || year%400==0))) d = 29;
            st.gpr[4] = d;
            retval = 0;
        } else if (e.fnid == 0xd68d676a) {
            st.gpr[4] = 0; // Monday
            retval = 0;

        // ═══ cellScreenshot ═══
        } else if (e.fnid == 0x3ecc4646 || e.fnid == 0xa58d3f77 ||
                   e.fnid == 0xd1bb8d91 || e.fnid == 0xdf3c9711) {
            retval = 0;

        // ═══ cellMic ═══
        } else if (e.fnid == 0x1bf34275 || e.fnid == 0xb278948c ||
                   e.fnid == 0xe57b4e17 || e.fnid == 0x287bf034) {
            retval = 0;
        } else if (e.fnid == 0x448f9a7c) {
            retval = (uint64_t)(int64_t)-1; // CELL_MIC_ERROR_DEVICE_NOT_FOUND

        // ═══ cellSysCache ═══
        } else if (e.fnid == 0x83a4afca) {
            // r3 = CellSysCacheParam* — write cache path
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 512 <= memSize) {
                // getCachePath offset = +4, write "/dev_hdd1/caches/"
                const char* path = "/dev_hdd1/caches/";
                for (int i = 0; path[i]; i++) mem[ptr + 4 + i] = path[i];
                mem[ptr + 4 + 17] = 0;
            }
            retval = 1; // CELL_SYSCACHE_RET_OK_CLEARED
        } else if (e.fnid == 0x78f6204d) {
            retval = 0;

        // ═══ cellUsbd / cellImeJp ═══
        } else if (e.fnid == 0x23658577 || e.fnid == 0x3782d73f ||
                   e.fnid == 0x4cf28923 || e.fnid == 0x2a9372c3) {
            retval = 0;

        // ═══ cellSysutil extras ═══
        } else if (e.fnid == 0xf89e3dbd) {
            // r3 = param_id, r4 = buf_ptr, r5 = buf_size
            uint64_t buf = st.gpr[4];
            uint32_t sz = (uint32_t)st.gpr[5];
            if (buf && sz > 0 && buf + sz <= memSize) {
                mem[buf] = 0; // empty string
            }
            retval = 0;
        } else if (e.fnid == 0x38ecda87) {
            // r3 = status_ptr (u32) → 0 = not playing
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0; mem[ptr+1] = 0; mem[ptr+2] = 0; mem[ptr+3] = 0;
            }
            retval = 0;
        } else if (e.fnid == 0x6ae48aa9 || e.fnid == 0x78410b3f) {
            retval = 0;

        // ═══ cellDiscGame ═══
        } else if (e.fnid == 0x312c9ac4) {
            // r3 = CellDiscGameBootInfo* — zero-fill
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
            }
            retval = 0;
        } else if (e.fnid == 0xfee36d78 ||
                   e.fnid == 0x04bb3c46) {
            retval = 0;

        // ═══ cellStorage ═══
        } else if (e.fnid == 0xe462f5e5 || e.fnid == 0x1d911b52) {
            retval = 0;

        // ═══ cellSubDisplay ═══
        } else if (e.fnid == 0xb8fea1c2 || e.fnid == 0x64c14696) {
            retval = 0;
        } else if (e.fnid == 0x278e51bf) {
            // r3 = size_ptr
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t sz = 0x10000; // 64KB
                mem[ptr] = (uint8_t)(sz>>24); mem[ptr+1] = (uint8_t)(sz>>16);
                mem[ptr+2] = (uint8_t)(sz>>8); mem[ptr+3] = (uint8_t)sz;
            }
            retval = 0;

        // ═══ cellSearch ═══
        } else if (e.fnid == 0x45e3ec64 || e.fnid == 0xb79cbc6e) {
            retval = 0;
        } else if (e.fnid == 0xa660aaea) {
            retval = 0;
        } else if (e.fnid == 0x5e331ebe) {
            // r4 = count_ptr → 0 results
            uint64_t ptr = st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0; mem[ptr+1] = 0; mem[ptr+2] = 0; mem[ptr+3] = 0;
            }
            retval = 0;

        // ═══ cellMusic ═══
        } else if (e.fnid == 0x21929812 || e.fnid == 0x74c0bb2f) {
            retval = 0;
        } else if (e.fnid == 0xa7bf1b92) {
            // r3 = status_ptr → 0 = stopped
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = 0; mem[ptr+1] = 0; mem[ptr+2] = 0; mem[ptr+3] = 0;
            }
            retval = 0;

        // ═══ cellPhotoExport ═══
        } else if (e.fnid == 0xd3b10cbd || e.fnid == 0xb29f2aed ||
                   e.fnid == 0x8d422d8b) {
            retval = 0;

        // ═══ cellRemotePlay / cellBgdl / cellGameUpdate ═══
        } else if (e.fnid == 0xed1c0428) {
            st.gpr[4] = 0; // not connected
            retval = 0;
        } else if (e.fnid == 0xf09e1d35) {
            retval = (uint64_t)(int64_t)-1; // no background downloads
        } else if (e.fnid == 0xae299ba5 || e.fnid == 0x00e38e0b) {
            retval = 0;
        } else if (e.fnid == 0x4b88c75a) {
            retval = 0; // no update available

        // ═══ cellVpost — video post-processing ═══
        } else if (e.fnid == 0xce247a9c || e.fnid == 0xd28c4359) {
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.fnid == 0x292d1115 || e.fnid == 0xf62c6004) {
            retval = 0;

        // ═══ cellAtrac — ATRAC3+ audio ═══
        } else if (e.fnid == 0xeeb4289f) {
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.fnid == 0x34e241d7) {
            retval = 0;
        } else if (e.fnid == 0xb771667c) {
            retval = (uint64_t)(int64_t)-1; // no audio data
        } else if (e.fnid == 0x1c5004f5) {
            retval = 0;
        } else if (e.fnid == 0x05d6b33e) {
            retval = 0;
        } else if (e.fnid == 0x4343cb43) {
            st.gpr[4] = 0; // not needed
            retval = 0;

        // ═══ cellVoice ═══
        } else if (e.fnid == 0x8cf0742d || e.fnid == 0x176d528e) {
            retval = 0;
        } else if (e.fnid == 0x25e2f250) {
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.fnid == 0x47f672fb) {
            retval = 0;

        // ═══ sceNpMatching2 — matchmaking ═══
        } else if (e.fnid == 0x91a66b60 || e.fnid == 0x8df1c55b) {
            retval = 0;
        } else if (e.fnid == 0x7795cbfb) {
            uint32_t hid = nextHandleId++;
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                mem[ptr] = (uint8_t)(hid >> 24); mem[ptr+1] = (uint8_t)(hid >> 16);
                mem[ptr+2] = (uint8_t)(hid >> 8); mem[ptr+3] = (uint8_t)hid;
            }
            retval = 0;
        } else if (e.fnid == 0x43e7ee1a) {
            retval = 0;

        // ═══ sceNpScore — leaderboards ═══
        } else if (e.fnid == 0x368bac16 || e.fnid == 0xe36d39ae) {
            retval = 0;
        } else if (e.fnid == 0x57efaab2) {
            st.gpr[4] = nextHandleId++;
            retval = 0;
        } else if (e.fnid == 0xec58df2b) {
            retval = 0;

        // ═══ sceNpTus — title user storage ═══
        } else if (e.fnid == 0x3a24b2ef || e.fnid == 0x63496d84) {
            retval = 0;
        } else if (e.fnid == 0x6a49f21a) {
            st.gpr[4] = nextHandleId++;
            retval = 0;
        } else if (e.fnid == 0xdb17369f) {
            retval = 0;

        // ═══ cellSaveData v2 variants ═══
        } else if (e.fnid == 0x590e6d0b || e.fnid == 0xf0f530b7 ||
                   e.fnid == 0x22126da4 || e.fnid == 0x06a7a4a8 ||
                   e.fnid == 0xf14197af || e.fnid == 0xa4726925) {
            retval = 0;

        // ═══ cellGame extras ═══
        } else if (e.fnid == 0x8e422adc || e.fnid == 0x586bdc5c ||
                   e.fnid == 0xc6bee834) {
            retval = 0;
        } else if (e.fnid == 0xbaed3165) {
            // r3 = buf, r4 = size — empty path
            uint64_t buf = st.gpr[3];
            if (buf && buf < memSize) mem[buf] = 0;
            retval = 0;

        // ═══ cellWebBrowser ═══
        } else if (e.fnid == 0xf19fd906 || e.fnid == 0x2b5f3544) {
            retval = 0;

        // ═══ cellPad extras ═══
        } else if (e.fnid == 0x23ef9b61 || e.fnid == 0x0f2c5daf ||
                   e.fnid == 0xbd8f1ead || e.fnid == 0x0353d3cf) {
            retval = 0;
        } else if (e.fnid == 0x316cdf06) {
            // r4 = info_ptr — zero-fill
            uint64_t ptr = st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0xa703a51d) {
            // r3 = CellPadInfo2* — zero-fill (0 pads connected)
            uint64_t ptr = st.gpr[3];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;
        } else if (e.fnid == 0x9589f71c) {
            st.gpr[4] = (uint64_t)(int64_t)-1; // no port
            retval = 0;

        // ── cellPrint ────────────────────────────────────────────
        } else if (e.fnid == 0xb2d72bb9 || e.fnid == 0xb75eb4f8 ||
                   e.fnid == 0xfc06712d || e.fnid == 0xd94e89bc) {
            retval = 0;

        // ── cellMusicDecode ──────────────────────────────────────
        } else if (e.fnid == 0x98fdafd5 || e.fnid == 0x7659b1e7) {
            retval = 0;
        } else if (e.fnid == 0xdda1bc5e || e.fnid == 0x3be565bf) {
            retval = 0;
        } else if (e.fnid == 0x9609ca68 || e.fnid == 0x18cfe79d) {
            retval = 0;
        } else if (e.fnid == 0xc8e261ce) {
            // Write status=0 (idle) to pointer in r4
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t zero = 0;
                std::memcpy(mem + (uint32_t)st.gpr[4], &zero, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x22953249) {
            retval = (uint64_t)(int64_t)-1; // no data

        // ── sceNpFriends ─────────────────────────────────────────
        } else if (e.fnid == 0xa72ca286) {
            retval = 0;
        } else if (e.fnid == 0xbdef1053) {
            retval = 0;
        } else if (e.fnid == 0x912e5879) {
            // Write count=0 to pointer in r4
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t zero = 0;
                std::memcpy(mem + (uint32_t)st.gpr[4], &zero, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xfdd24474 ||
                   e.fnid == 0xa98a49f1 ||
                   e.fnid == 0x8ae071cf) {
            retval = (uint64_t)(int64_t)-1; // not found

        // ── cellSail (media player) ──────────────────────────────
        } else if (e.fnid == 0x948daf24) {
            retval = 0;
        } else if (e.fnid == 0x1c674a46) {
            retval = 0;
        } else if (e.fnid == 0x928af120 || e.fnid == 0xd43f8537) {
            retval = 0;
        } else if (e.fnid == 0x9d9da3f4) {
            retval = 0;
        } else if (e.fnid == 0x254936db) {
            // Write handle to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[4], &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x2b5f20be) {
            retval = 0;
        } else if (e.fnid == 0x3a6e01b1) {
            retval = 0;

        // ── cellRudp (reliable UDP) ──────────────────────────────
        } else if (e.fnid == 0xe26b7209) {
            retval = 0;
        } else if (e.fnid == 0xb92636ea) {
            retval = 0;
        } else if (e.fnid == 0xca071aac) {
            retval = (uint64_t)nextHandleId++; // context ID
        } else if (e.fnid == 0x500a408e) {
            retval = 0;
        } else if (e.fnid == 0x23293e9b) {
            retval = (uint64_t)(int64_t)-1; // network error
        } else if (e.fnid == 0x61db1f47) {
            retval = (uint64_t)(int64_t)-1; // no data

        // ── cellHttpUtil ─────────────────────────────────────────
        } else if (e.fnid == 0x1c7430ee || e.fnid == 0x77a3d37c ||
                   e.fnid == 0xd95b0c60 || e.fnid == 0xa4c9b7dc) {
            retval = 0;

        // ── cellSsl ──────────────────────────────────────────────
        } else if (e.fnid == 0x8c1678ef) {
            retval = 0;
        } else if (e.fnid == 0x2a574a25) {
            retval = 0;
        } else if (e.fnid == 0xca220e5b || e.fnid == 0x10c1bffc ||
                   e.fnid == 0xab209809) {
            retval = 0;

        // ── cellHttp ─────────────────────────────────────────────
        } else if (e.fnid == 0xf7644439) {
            retval = 0;
        } else if (e.fnid == 0x33457271) {
            retval = 0;
        } else if (e.fnid == 0x598dd02e) {
            // Write client handle to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[4], &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x6907a5fb) {
            retval = 0;
        } else if (e.fnid == 0x21149d0a) {
            // Write transaction handle to r5 pointer
            if (st.gpr[5] && st.gpr[5] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[5], &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xc2d6836b) {
            retval = 0;
        } else if (e.fnid == 0x6180ba1a) {
            retval = (uint64_t)(int64_t)-1; // network error
        } else if (e.fnid == 0xfb40bf8e) {
            retval = (uint64_t)(int64_t)-1; // network error

        // ── cellNetCtl ───────────────────────────────────────────
        } else if (e.fnid == 0xf53f04bb) {
            retval = 0;
        } else if (e.fnid == 0x9f18ccad) {
            retval = 0;
        } else if (e.fnid == 0x5c413ca9) {
            // Write state=0 (disconnected) to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t zero = 0;
                std::memcpy(mem + (uint32_t)st.gpr[4], &zero, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x08a8a347) {
            retval = (uint64_t)(int64_t)-1; // not connected
        } else if (e.fnid == 0x2e274b74) {
            retval = 0;
        } else if (e.fnid == 0x438cee7b) {
            retval = 0;

        // ── cellFont ─────────────────────────────────────────────
        } else if (e.fnid == 0x4b734c8c || e.fnid == 0x3e3712ed) {
            retval = 0;
        } else if (e.fnid == 0xc1b20a00) {
            // Write library handle to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[4], &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x859a29d7 || e.fnid == 0xec2185a8) {
            retval = 0;
        } else if (e.fnid == 0x85cad796 || e.fnid == 0x62f4f193) {
            retval = 0;
        } else if (e.fnid == 0x4b02499c) {
            // Render nothing — games fall back to blank glyph
            retval = 0;

        // ── cellFontFT ───────────────────────────────────────────
        } else if (e.fnid == 0xe67ca1db || e.fnid == 0xecd30388 ||
                   e.fnid == 0x6741115e) {
            retval = 0;
        } else if (e.fnid == 0xa3f2dceb) {
            retval = 0; // flags=0

        // ── cellSpurs ────────────────────────────────────────────
        } else if (e.fnid == 0x70e3d58a) {
            retval = 0;
        } else if (e.fnid == 0xe48bf572) {
            retval = 0;
        } else if (e.fnid == 0x1bf8a000) {
            // Write count=6 to r4 pointer (PS3 has 6 SPUs available)
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t six = __builtin_bswap32(6);
                std::memcpy(mem + (uint32_t)st.gpr[4], &six, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xe2df437e) {
            // Write fake group ID to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[4], &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x3942f6bf || e.fnid == 0x79b36c3e ||
                   e.fnid == 0xbc7001af) {
            retval = 0;
        } else if (e.fnid == 0xc97cec02) {
            retval = 0;
        } else if (e.fnid == 0x3fd6c11a) {
            retval = 0;
        } else if (e.fnid == 0xe591f650) {
            retval = 0;
        } else if (e.fnid == 0x899934dc || e.fnid == 0x71dc1454) {
            retval = 0;

        // ── cellSpursJq ──────────────────────────────────────────
        } else if (e.fnid == 0x94965961 || e.fnid == 0x5afc1e7c) {
            retval = 0;
        } else if (e.fnid == 0xc8101522) {
            retval = 0;
        } else if (e.fnid == 0xab27b2c4) {
            // Write count=0 to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t zero = 0;
                std::memcpy(mem + (uint32_t)st.gpr[4], &zero, 4);
            }
            retval = 0;

        // ── sys_net (BSD sockets) ────────────────────────────────
        } else if (e.fnid == 0xae26eab1) {
            retval = (uint64_t)nextHandleId++; // fake socket fd
        } else if (e.fnid == 0x5a048bcb) {
            retval = 0;
        } else if (e.fnid == 0xfc5f38c2 || e.fnid == 0xdca3f6f6) {
            retval = 0;
        } else if (e.fnid == 0x3b214ba1 || e.fnid == 0x4e89a57a) {
            retval = (uint64_t)(int64_t)-1; // network error
        } else if (e.fnid == 0x8cc694ce) {
            retval = (uint64_t)(int64_t)-1;
        } else if (e.fnid == 0x75b3c344) {
            retval = (uint64_t)(int64_t)-1;
        } else if (e.fnid == 0x9a5c66e3 || e.fnid == 0xf448bbf6) {
            retval = 0;

        // ── cellUserInfo ─────────────────────────────────────────
        } else if (e.fnid == 0xdcf8eb53) {
            retval = 0;
        } else if (e.fnid == 0xadccc2b9) {
            retval = 0;
        } else if (e.fnid == 0x389eb4ec) {
            retval = 0;
        } else if (e.fnid == 0x4a5b749a) {
            // Write count=1 (one user) to r4 pointer
            if (st.gpr[4] && st.gpr[4] + 4 <= memSize) {
                uint32_t one = __builtin_bswap32(1);
                std::memcpy(mem + (uint32_t)st.gpr[4], &one, 4);
            }
            retval = 0;

        // ── cellAdec (audio decoder) ─────────────────────────────
        } else if (e.fnid == 0x8f027e01) {
            // Write handle to r5 pointer
            if (st.gpr[5] && st.gpr[5] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[5], &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xcd96044f) {
            retval = 0;
        } else if (e.fnid == 0xcc534abe || e.fnid == 0xccfeefc2) {
            retval = 0;
        } else if (e.fnid == 0x4d682b55) {
            retval = 0;
        } else if (e.fnid == 0x0cfa3baf) {
            retval = (uint64_t)(int64_t)-1; // no audio data

        // ── cellDmux (demuxer) ───────────────────────────────────
        } else if (e.fnid == 0x2d4ce3df) {
            // Write handle to r5 pointer
            if (st.gpr[5] && st.gpr[5] + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + (uint32_t)st.gpr[5], &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x438a5a0f) {
            retval = 0;
        } else if (e.fnid == 0xcd076abf || e.fnid == 0xabbbb443) {
            retval = 0;
        } else if (e.fnid == 0x181d1978 || e.fnid == 0x022dda52) {
            retval = 0;

        // ── cellAvconfExt — audio/video config ───────────────────────
        } else if (e.fnid == 0x9df98130) {
            // r3 = videoOut (0=PRIMARY), r4 = ptr to CellVideoOutDeviceInfo
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
                // portType=HDMI(1), colorSpace=RGB(1)
                mem[ptr] = 1;     // portType
                mem[ptr+4] = 1;   // colorSpace
            }
            retval = 0;
        } else if (e.fnid == 0x1e930eef) {
            retval = 1;  // 1 device (primary display)
        } else if (e.fnid == 0x938013a0) {
            retval = 1;  // all resolutions "available"
        } else if (e.fnid == 0x887572d5) {
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
        } else if (e.fnid == 0x0bae8772) {
            retval = 0;  // success
        } else if (e.fnid == 0x15b0b0cd) {
            // r3 = audioOut, r4 = deviceIndex, r5 = ptr to CellAudioOutState
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                mem[ptr] = 2;   // state = ENABLED
                mem[ptr+4] = 1; // encoder = LPCM
                mem[ptr+8] = 2; // channels = stereo
            }
            retval = 0;
        } else if (e.fnid == 0x4692ab35 ||
                   e.fnid == 0xa0e6fdf0) {
            retval = (e.fnid == 0xa0e6fdf0) ? 8 : 0;
        } else if (e.fnid == 0xa5927fc5) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0x7794d7e7) {
            retval = 1;  // 1 audio output

        // ── cellSync2 — enhanced sync primitives ─────────────────────
        } else if (e.fnid == 0x55836e73) {
            // r3 = ptr to mutex struct, zero-init it
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0x5551b4df ||
                   e.fnid == 0x5551b540 ||
                   e.fnid == 0x5551b56c) {
            retval = 0;  // single-threaded: always succeeds
        } else if (e.fnid == 0xa661b35c) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 128 <= memSize) std::memset(mem + ptr, 0, 128);
            retval = 0;
        } else if (e.fnid == 0xa661b4d0 ||
                   e.fnid == 0xa661b4e8) {
            retval = 0;
        } else if (e.fnid == 0xa661b500 ||
                   e.fnid == 0xa661b518) {
            retval = (int32_t)0x80410101;  // CELL_SYNC2_ERROR_EMPTY
        } else if (e.fnid == 0xa661b530) {
            retval = 0;  // empty

        // ── cellVideoExport ──────────────────────────────────────────
        } else if (e.fnid == 0xe7998490 ||
                   e.fnid == 0x12998e3a) {
            retval = 0;
        } else if (e.fnid == 0x3cf0b78e) {
            retval = 100;  // 100% done (stub)

        // ── cellPhotoImport ──────────────────────────────────────────
        } else if (e.fnid == 0x0783bce0 ||
                   e.fnid == 0x1c231710) {
            retval = 0;
        } else if (e.fnid == 0x59405c00) {
            retval = (int32_t)0x80028b01;  // CELL_PHOTO_IMPORT_ERROR_CANCEL

        // ── cellNetCtlExt — extended network ─────────────────────────
        } else if (e.fnid == 0xca8cd5b7) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                mem[ptr] = 3;  // NAT type 3 (strict — offline)
            }
            retval = 0;
        } else if (e.fnid == 0x3b23dbd0) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                std::memcpy(mem + ptr, "0.0.0.0", 7);
            }
            retval = 0;
        } else if (e.fnid == 0x2a72ed91 ||
                   e.fnid == 0x7e4a2c6e ||
                   e.fnid == 0x6f000e53) {
            retval = 0;

        // ── sys_io — controller I/O ──────────────────────────────────
        } else if (e.fnid == 0x3733ea3c) {
            retval = 0;  // no special capabilities
        } else if (e.fnid == 0x1cf98800) {
            retval = 0;
        } else if (e.fnid == 0x578e3c98) {
            retval = 0;
        } else if (e.fnid == 0xa703a917 ||
                   e.fnid == 0x6bc09c61) {
            // r3 = port, r4 = ptr to CellPadData (zero = no input)
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;

        // ── cellL10n — single-char conversion ────────────────────────
        } else if (e.fnid == 0x9e989095) {
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
        } else if (e.fnid == 0x487b4dc1) {
            uint32_t src = (uint32_t)st.gpr[3];
            uint32_t dst = (uint32_t)st.gpr[4];
            if (src + 2 <= memSize && dst < memSize) {
                uint16_t c = ((uint16_t)mem[src] << 8) | mem[src+1];
                mem[dst] = (uint8_t)(c & 0x7F);
            }
            retval = 1;  // 1 byte written

        // ── cellGem (PlayStation Move) ───────────────────────────────
        } else if (e.fnid == 0xabb4b268) {
            retval = 0;
        } else if (e.fnid == 0xa8bc1648 || e.fnid == 0xc7622586 ||
                   e.fnid == 0xfb5887f9 ||
                   e.fnid == 0xe1f85a80 ||
                   e.fnid == 0x6d245f02 || e.fnid == 0x3507f03b) {
            retval = 0;
        } else if (e.fnid == 0x3e24e759) {
            // r3 = gem_num, r4 = ptr to state, r5 = flag
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = (int32_t)0x80121806;  // CELL_GEM_ERROR_NOT_CONNECTED
        } else if (e.fnid == 0x13ea53e7) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
                // max_connect=0 (no Move controllers)
            }
            retval = 0;
        } else if (e.fnid == 0x6a666297) {
            retval = 0;

        // ── cellMove ─────────────────────────────────────────────────
        } else if (e.fnid == 0x9eb07a5b || e.fnid == 0xadee0a65 ||
                   e.fnid == 0x82cfb3d1 || e.fnid == 0xd37b8e36) {
            retval = 0;

        // ── cellOvis ─────────────────────────────────────────────────
        } else if (e.fnid == 0xcc78cd7b || e.fnid == 0x2e70a5a1) {
            retval = 0;
        } else if (e.fnid == 0x71894bfa) {
            retval = 0;

        // ── cellCamera ───────────────────────────────────────────────
        } else if (e.fnid == 0x30eaee99 || e.fnid == 0xf1e65d8d) {
            retval = 0;
        } else if (e.fnid == 0x5de25cd1 || e.fnid == 0x379c5dd6 ||
                   e.fnid == 0x40f6ead6 || e.fnid == 0xa6b20b8c) {
            retval = 0;
        } else if (e.fnid == 0x60237200) {
            retval = 0;
        } else if (e.fnid == 0x10697d02) {
            // r3 = devNum, r4 = ptr to type output
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t v = 0; // no camera
                std::memcpy(mem + ptr, &v, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xb0647e5a) {
            retval = 0;  // not available

        // ── cellResc (Resolution Scaler) ─────────────────────────────
        } else if (e.fnid == 0x3711bd4f) {
            retval = 0;
        } else if (e.fnid == 0xcb4ba814) {
            retval = 0;
        } else if (e.fnid == 0x089a5067 || e.fnid == 0xffd96327 ||
                   e.fnid == 0x770f5ac4 || e.fnid == 0x147dde14 ||
                   e.fnid == 0xbe0760f8) {
            retval = 0;
        } else if (e.fnid == 0x35239700) {
            retval = 2;  // double buffer
        } else if (e.fnid == 0x2ea3061e) {
            retval = 0;  // ready
        } else if (e.fnid == 0xaa8b2baa) {
            retval = 0;

        // ── cellPamf (PS3 media format) ──────────────────────────────
        } else if (e.fnid == 0xb13dac24) {
            retval = 0;
        } else if (e.fnid == 0x484cfb6f) {
            retval = 0;  // no streams
        } else if (e.fnid == 0x79df04e4) {
            retval = 0;
        } else if (e.fnid == 0x28b4e2c1 ||
                   e.fnid == 0x45d62a3b) {
            retval = 0;
        } else if (e.fnid == 0x461534a4) {
            retval = 0;
        } else if (e.fnid == 0xd6a50759) {
            retval = 0;

        // ── sceNpUtil ────────────────────────────────────────────────
        } else if (e.fnid == 0x05af7b56) {
            retval = 0;
        } else if (e.fnid == 0xaa9a4c83) {
            retval = 100;  // complete
        } else if (e.fnid == 0xfade2b8d) {
            retval = 0;

        // ── sceNpSignaling ───────────────────────────────────────────
        } else if (e.fnid == 0xa3c4ddeb) {
            // r3 = ptr to npId, r4 = handler, r5 = arg, r6 = ptr to ctxId
            uint32_t ptr = (uint32_t)st.gpr[6];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x7db3e905) {
            retval = 0;
        } else if (e.fnid == 0x55e42a79 ||
                   e.fnid == 0x7d5a4a87) {
            retval = 0;
        } else if (e.fnid == 0xa10fedd3) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;

        // ── sceNpClans ───────────────────────────────────────────────
        } else if (e.fnid == 0xaa79031d || e.fnid == 0x1f51ae44) {
            retval = 0;
        } else if (e.fnid == 0x6e24f290) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x3c67c847) {
            retval = 0;

        // ── cellSpursTrace ───────────────────────────────────────────
        } else if (e.fnid == 0x72ec1bf4 ||
                   e.fnid == 0x2db41dea ||
                   e.fnid == 0x9cae4fdc ||
                   e.fnid == 0x04a6bd22) {
            retval = 0;

        // ── cellCrossController ──────────────────────────────────────
        } else if (e.fnid == 0x174ece14 ||
                   e.fnid == 0x7adf3bab) {
            retval = 0;

        // ── cellSysconf ──────────────────────────────────────────────
        } else if (e.fnid == 0x00753e2a) {
            retval = 0;
        } else if (e.fnid == 0x0beecf67) {
            // r3 = ptr to device list, zero-fill (no BT devices)
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;
        } else if (e.fnid == 0xac410de9) {
            retval = 0;

        // ── cellMusicExport ──────────────────────────────────────────
        } else if (e.fnid == 0xe8ad3dd4 ||
                   e.fnid == 0x4ab73a73) {
            retval = 0;
        } else if (e.fnid == 0x61ead640) {
            retval = 100;

        // ── cellPhotoUtility ─────────────────────────────────────────
        } else if (e.fnid == 0x5caa19e7 ||
                   e.fnid == 0xd46fa1f7) {
            retval = 0;
        } else if (e.fnid == 0xd3b10cbd) {
            retval = 0;

        // ── cellVdec (video decoder) ─────────────────────────────────
        } else if (e.fnid == 0xa83f253b) {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xb6d5cacd || e.fnid == 0x9625f90a) {
            retval = 0;
        } else if (e.fnid == 0x59f481ef) {
            retval = 0;
        } else if (e.fnid == 0xc1901ab8) {
            retval = 0;
        } else if (e.fnid == 0xac73ada8 || e.fnid == 0x52b9a8b0) {
            retval = (int32_t)0x80610102;  // CELL_VDEC_ERROR_EMPTY
        } else if (e.fnid == 0x5877b8ab) {
            retval = 0;
        } else if (e.fnid == 0xa1ffa426) {
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
        } else if (e.fnid == 0xfc6fe574 ||
                   e.fnid == 0x38ecda87) {
            // r3 = ptr to status (0 = not playing)
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) std::memset(mem + ptr, 0, 4);
            retval = 0;
        } else if (e.fnid == 0x799b5fae ||
                   e.fnid == 0xda004916) {
            retval = 0;
        } else if (e.fnid == 0xf4e3caa0) {
            retval = 0;
        } else if (e.fnid == 0x189a74da) {
            retval = 0;  // no pending callbacks
        } else if (e.fnid == 0x9d98afa0) {
            retval = 0;
        } else if (e.fnid == 0xf1e5f2c0) {
            retval = 1;  // area = SCEA (North America)

        // ── cellGcmSys extras ────────────────────────────────────────
        } else if (e.fnid == 0x5a41c10f) {
            retval = 0;
        } else if (e.fnid == 0xd01b570d) {
            retval = 16667;  // ~60fps
        } else if (e.fnid == 0x055bd74d) {
            retval = 0;
        } else if (e.fnid == 0x06edea9e) {
            retval = 0;
        } else if (e.fnid == 0xd8f88f1a) {
            retval = 0;
        } else if (e.fnid == 0xe315a0b2) {
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
        } else if (e.fnid == 0x99d397ac) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0x4ae8d215 ||
                   e.fnid == 0xe44e78ec) {
            retval = 0;
        } else if (e.fnid == 0x4524cccd || e.fnid == 0xd9b7653e) {
            retval = 0;
        } else if (e.fnid == 0xa114ec67) {
            // r3 = ea, r4 = size, r5 = ptr to offset
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t off = __builtin_bswap32((uint32_t)st.gpr[3]);
                std::memcpy(mem + ptr, &off, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xdb23b5e3) {
            retval = 0;
        } else if (e.fnid == 0x9a4c1b5f ||
                   e.fnid == 0xd34a420d) {
            retval = 0;

        // ── cellSpurs extras ─────────────────────────────────────────
        } else if (e.fnid == 0xc97cec02 ||
                   e.fnid == 0xe7dd87e1) {
            retval = 0;
        } else if (e.fnid == 0x52cc6c82) {
            retval = 0;  // task already finished (stub)
        } else if (e.fnid == 0x899934dc) {
            retval = 0;
        } else if (e.fnid == 0x39a8e757) {
            retval = 0;
        } else if (e.fnid == 0xe7fc14af ||
                   e.fnid == 0x8b000b8a) {
            retval = 0;
        } else if (e.fnid == 0x2f0a4998 ||
                   e.fnid == 0x36c4f4a8) {
            retval = 0;
        } else if (e.fnid == 0xe82a263a ||
                   e.fnid == 0x61f97115) {
            retval = 0;  // single-threaded: event already signaled
        } else if (e.fnid == 0xb9bc6207) {
            retval = 0;

        // ── cellGame extras ──────────────────────────────────────────
        } else if (e.fnid == 0xde9c0881) {
            // r3 = paramId, r4 = buf, r5 = bufSize
            uint32_t ptr = (uint32_t)st.gpr[4];
            uint32_t sz  = (uint32_t)st.gpr[5];
            if (ptr && sz > 0 && ptr + sz <= memSize) mem[ptr] = 0;
            retval = 0;
        } else if (e.fnid == 0xbaed3165) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 64 <= memSize) {
                std::memset(mem + ptr, 0, 64);
                std::memcpy(mem + ptr, "/dev_hdd0/game/", 15);
            }
            retval = 0;
        } else if (e.fnid == 0x2a8e6b92) {
            retval = 0;
        } else if (e.fnid == 0x8ade82a6) {
            retval = 0;

        // ── cellSaveData extras ──────────────────────────────────────
        } else if (e.fnid == 0xd739cc4b ||
                   e.fnid == 0x3604d4f4) {
            retval = 0;
        } else if (e.fnid == 0x1c8b05e2) {
            retval = 0;
        } else if (e.fnid == 0x6c2975f4) {
            retval = 0;

        // ── cellPad extras ───────────────────────────────────────────
        } else if (e.fnid == 0x0e2dfaad) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;
        } else if (e.fnid == 0x23ef9b61) {
            retval = 0;  // rumble — no-op
        } else if (e.fnid == 0x4cc9b68d) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;
        } else if (e.fnid == 0xa703a917) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;

        // ── cellKb (keyboard) ────────────────────────────────────────
        } else if (e.fnid == 0x85656de9) {
            retval = 0;
        } else if (e.fnid == 0x8fe7f827) {
            retval = 0;
        } else if (e.fnid == 0xead11ae9) {
            // r3 = ptr to CellKbInfo (zero-fill = no keyboards)
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) std::memset(mem + ptr, 0, 16);
            retval = 0;
        } else if (e.fnid == 0xda870fc8) {
            // r3 = port, r4 = ptr to CellKbData
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0xa5f85cb3 ||
                   e.fnid == 0x3f72c56e ||
                   e.fnid == 0xdeefdfa7) {
            retval = 0;

        // ── cellMouse ────────────────────────────────────────────────
        } else if (e.fnid == 0x890c9fb8) {
            retval = 0;
        } else if (e.fnid == 0x67c67673) {
            retval = 0;
        } else if (e.fnid == 0xb4086711) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) std::memset(mem + ptr, 0, 16);
            retval = 0;
        } else if (e.fnid == 0xff0a21b7) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0xa328cc35) {
            retval = 0;

        // ── cellJpgEnc ───────────────────────────────────────────────
        } else if (e.fnid == 0xa9e4e0e3) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x636dc9c2 || e.fnid == 0xb81e5765) {
            retval = 0;
        } else if (e.fnid == 0x969fc5f7) {
            retval = 0;

        // ── cellHttp extras ──────────────────────────────────────────
        } else if (e.fnid == 0x21149d0a) {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xc2d6836b) {
            retval = 0;
        } else if (e.fnid == 0x6180ba1a) {
            retval = (int32_t)0x80710002;  // CELL_HTTP_ERROR_NO_CONNECTION
        } else if (e.fnid == 0xfb40bf8e) {
            retval = (int32_t)0x80710002;
        } else if (e.fnid == 0x4d29fb5c) {
            retval = 0;
        } else if (e.fnid == 0x7abb3bdf) {
            retval = 0;

        // ── cellFiber ────────────────────────────────────────────────
        } else if (e.fnid == 0x0a16d843 ||
                   e.fnid == 0x1e471a02 ||
                   e.fnid == 0x5f3afb12) {
            retval = 0;
        } else if (e.fnid == 0x3b3bdb7c) {
            retval = 0;
        } else if (e.fnid == 0x72086315) {
            retval = 0;  // fiber completed (stub)
        } else if (e.fnid == 0x4bfb9b97) {
            retval = 1;  // dummy fiber id
        } else if (e.fnid == 0x1f5e999d) {
            retval = 0;

        // ── cellSubDisplay ───────────────────────────────────────────
        } else if (e.fnid == 0xb8fea1c2 ||
                   e.fnid == 0x64c14696) {
            retval = 0;
        } else if (e.fnid == 0x278e51bf) {
            retval = 0;
        } else if (e.fnid == 0x3c189067 ||
                   e.fnid == 0xd8415c65) {
            retval = 0;

        // ── cellStorageData ──────────────────────────────────────────
        } else if (e.fnid == 0x62e0c5c1 ||
                   e.fnid == 0x1d911b52) {
            retval = 0;
        } else if (e.fnid == 0x97b4b8e5) {
            // r3 = ptr to path buf
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 64 <= memSize) {
                std::memset(mem + ptr, 0, 64);
                std::memcpy(mem + ptr, "/dev_hdd0/", 10);
            }
            retval = 0;

        // ── cellSysmodule extras ─────────────────────────────────────
        } else if (e.fnid == 0xb8a0bf48 ||
                   e.fnid == 0x2a1321c1) {
            retval = 0;  // always succeed
        } else if (e.fnid == 0x452adbc2) {
            retval = 0;  // CELL_SYSMODULE_LOADED
        } else if (e.fnid == 0x6e0040b4 ||
                   e.fnid == 0x8b59a7b1 ||
                   e.fnid == 0x5f2b3e4a) {
            retval = 0;

        // ── cellAudio extras ─────────────────────────────────────────
        } else if (e.fnid == 0x708017e5) {
            // r3 = ptr to config, r4 = ptr to portNum
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t p = __builtin_bswap32(0);  // port 0
                std::memcpy(mem + ptr, &p, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xebb73ea7 ||
                   e.fnid == 0x8a8c0417 ||
                   e.fnid == 0x3ba2ba64) {
            retval = 0;
        } else if (e.fnid == 0xffd2b376) {
            // r3 = portNum, r4 = ptr to CellAudioPortConfig
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
                // nChannel=2 at offset 4, nBlock=8 at offset 8
                uint32_t ch = __builtin_bswap32(2);
                uint32_t bl = __builtin_bswap32(8);
                std::memcpy(mem + ptr + 4, &ch, 4);
                std::memcpy(mem + ptr + 8, &bl, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xca5ac370) {
            retval = 0;
        } else if (e.fnid == 0xcd7bc431) {
            // r3 = ptr to event queue id, r4 = ptr to key
            uint32_t pq = (uint32_t)st.gpr[3];
            uint32_t pk = (uint32_t)st.gpr[4];
            if (pq && pq + 4 <= memSize) {
                uint32_t q = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + pq, &q, 4);
            }
            if (pk && pk + 8 <= memSize) std::memset(mem + pk, 0, 8);
            retval = 0;
        } else if (e.fnid == 0x20e88c87 ||
                   e.fnid == 0x83afa9b3) {
            retval = 0;

        // ── sys_timer ────────────────────────────────────────────────
        } else if (e.fnid == 0x2c4a5c4c) {
            // r3 = ptr to timer_id
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x16e39df0 ||
                   e.fnid == 0x6c8e12e0 ||
                   e.fnid == 0x94c7cbc0) {
            retval = 0;
        } else if (e.fnid == 0xba7cd4ca) {
            retval = 0;
        } else if (e.fnid == 0x9cb2daa0 ||
                   e.fnid == 0x39c34d80) {
            retval = 0;  // return immediately (no actual sleep)

        // ── sys_event ────────────────────────────────────────────────
        } else if (e.fnid == 0x5dc20964) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x33b2c0e1) {
            retval = 0;
        } else if (e.fnid == 0x44e72545) {
            // r3 = queue id, r4 = ptr to event, r5 = timeout
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0x7b991b4e) {
            retval = (int32_t)0x80010005;  // EAGAIN — no events
        } else if (e.fnid == 0x7de8a40a) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x0e2b75ba ||
                   e.fnid == 0xd8140352 ||
                   e.fnid == 0x6da044e8) {
            retval = 0;
        } else if (e.fnid == 0xaab10b22) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x3690ed06) {
            retval = 0;
        } else if (e.fnid == 0x10e41720 ||
                   e.fnid == 0x145b35a0) {
            retval = 0;  // single-threaded: flag set
        } else if (e.fnid == 0x38ae2ce7 ||
                   e.fnid == 0xfc7f7b5a) {
            retval = 0;

        // ── sys_semaphore ────────────────────────────────────────────
        } else if (e.fnid == 0x54e90bac) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x17b54a0a) {
            retval = 0;
        } else if (e.fnid == 0x4e1a8a04 ||
                   e.fnid == 0x34a72786) {
            retval = 0;  // single-threaded
        } else if (e.fnid == 0x2eb3637a) {
            retval = 0;

        // ── sys_lwcond ───────────────────────────────────────────────
        } else if (e.fnid == 0x1b80eb8b) {
            retval = 0;
        } else if (e.fnid == 0xc862cd8c) {
            retval = 0;
        } else if (e.fnid == 0x8c8fce78) {
            retval = 0;  // single-threaded
        } else if (e.fnid == 0xdf07e2b3 ||
                   e.fnid == 0x7c532280) {
            retval = 0;

        // ── sys_lwmutex ──────────────────────────────────────────────
        } else if (e.fnid == 0x2f85c0ef) {
            retval = 0;
        } else if (e.fnid == 0xc3476d0c) {
            retval = 0;
        } else if (e.fnid == 0x1573dc3f ||
                   e.fnid == 0x1a1bc780) {
            retval = 0;  // single-threaded
        } else if (e.fnid == 0x1bc200f4) {
            retval = 0;

        // ── sys_ppu_thread ───────────────────────────────────────────
        } else if (e.fnid == 0x0d70ea34) {
            // r3 = ptr to thread_id
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 8 <= memSize) {
                uint64_t tid = nextHandleId++;
                for (int i = 0; i < 8; ++i)
                    mem[ptr + i] = (uint8_t)(tid >> (56 - 8*i));
            }
            retval = 0;
        } else if (e.fnid == 0xa4102f57) {
            retval = 0;
        } else if (e.fnid == 0x4a7bb945) {
            retval = 0;  // thread already finished
        } else if (e.fnid == 0x350d454e) {
            retval = 1;  // main thread id
        } else if (e.fnid == 0x744680a2) {
            retval = 0;
        } else if (e.fnid == 0x1386cd6e ||
                   e.fnid == 0x67f9fedb) {
            retval = 0;
        } else if (e.fnid == 0x350d454e) {
            // r3 = ptr to stack info struct
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                // addr at 0, size at 8 (64KB default)
                uint64_t sz = 64 * 1024;
                for (int i = 0; i < 8; ++i)
                    mem[ptr + 8 + i] = (uint8_t)(sz >> (56 - 8*i));
            }
            retval = 0;

        // ── sys_memory extras ────────────────────────────────────────
        } else if (e.fnid == 0x4e32b8a0) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x36b2d657) {
            retval = 0;
        } else if (e.fnid == 0x2a4a3e74) {
            // r3 = container, r4 = ptr to CellSysMemInfo
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                uint32_t total = __builtin_bswap32(64 * 1024 * 1024);
                uint32_t avail = __builtin_bswap32(32 * 1024 * 1024);
                std::memcpy(mem + ptr, &total, 4);
                std::memcpy(mem + ptr + 4, &avail, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xe1ef7570) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                uint32_t total = __builtin_bswap32(256 * 1024 * 1024);
                uint32_t avail = __builtin_bswap32(128 * 1024 * 1024);
                std::memcpy(mem + ptr, &total, 4);
                std::memcpy(mem + ptr + 4, &avail, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x88a93f56) {
            retval = 0;

        // ── sys_mmapper ──────────────────────────────────────────────
        } else if (e.fnid == 0x409ad9c8) {
            retval = 0;
        } else if (e.fnid == 0xa31eaf11) {
            // r3 = key, r4 = size, r5 = flags, r6 = ptr to mem_id
            uint32_t ptr = (uint32_t)st.gpr[6];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x7215b800 ||
                   e.fnid == 0x93f3c9a5 ||
                   e.fnid == 0x59d4584b) {
            retval = 0;

        // ── sys_mutex ────────────────────────────────────────────────
        } else if (e.fnid == 0xa85b74c4) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x1b27de0c) {
            retval = 0;
        } else if (e.fnid == 0xa2a90534 || e.fnid == 0x1a1bc780) {
            retval = 0;
        } else if (e.fnid == 0xba14c54e) {
            retval = 0;

        // ── sys_cond ─────────────────────────────────────────────────
        } else if (e.fnid == 0xb429a16e) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x5e37e3b4) {
            retval = 0;
        } else if (e.fnid == 0xa95e10fa) {
            retval = 0;
        } else if (e.fnid == 0x1e82ee27 || e.fnid == 0x08148c52) {
            retval = 0;

        // ── sys_spu ──────────────────────────────────────────────────
        } else if (e.fnid == 0x9f85b88b) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x21438f23) {
            retval = 0;
        } else if (e.fnid == 0x5ef1b6dd) {
            retval = 0;
        } else if (e.fnid == 0x79da05f7) {
            retval = 0;  // group finished (stub)
        } else if (e.fnid == 0x52421b2c) {
            // r3 = ptr to thread id
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xa7a709f1 ||
                   e.fnid == 0xa4ab5fbe) {
            retval = 0;
        } else if (e.fnid == 0x1b1d5125) {
            // r3 = thread, r4 = ptr to status
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t s = 0;
                std::memcpy(mem + ptr, &s, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x90da2c0a ||
                   e.fnid == 0x2ead4547) {
            retval = 0;

        // ── sys_prx ──────────────────────────────────────────────────
        } else if (e.fnid == 0x26090058) {
            retval = nextHandleId++;  // module id
        } else if (e.fnid == 0xef68c17c ||
                   e.fnid == 0xd5b4f13d ||
                   e.fnid == 0xae9e3e74) {
            retval = 0;
        } else if (e.fnid == 0x34b15cc4) {
            // r3 = flags, r4 = ptr to list — zero-fill
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0x98ef68a9) {
            retval = 1;  // dummy module id
        } else if (e.fnid == 0x42b23552 ||
                   e.fnid == 0x7d7fc79f) {
            retval = 0;

        // ── cellMsgDialog ────────────────────────────────────────────
        } else if (e.fnid == 0x860634da ||
                   e.fnid == 0xf81abf09) {
            retval = 0;  // dialog opens (stub — no UI)
        } else if (e.fnid == 0x7da3e04f ||
                   e.fnid == 0xf0d11ff1) {
            retval = 0;
        } else if (e.fnid == 0xbb49b97e ||
                   e.fnid == 0x6df96f85) {
            retval = 0;

        // ── cellOskDialog ────────────────────────────────────────────
        } else if (e.fnid == 0x36cff0d8 ||
                   e.fnid == 0xf5daa620) {
            retval = 0;
        } else if (e.fnid == 0xb6174ee6) {
            retval = 0;
        } else if (e.fnid == 0x0cc4e3be ||
                   e.fnid == 0xa5bdef02) {
            retval = 0;

        // ── cellVideoOut extras ──────────────────────────────────────
        } else if (e.fnid == 0xe558748d) {
            // r3 = resolutionId, r4 = ptr to CellVideoOutResolution
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 8 <= memSize) {
                // 720p: 1280×720
                uint16_t w = __builtin_bswap16(1280);
                uint16_t h = __builtin_bswap16(720);
                std::memcpy(mem + ptr, &w, 2);
                std::memcpy(mem + ptr + 2, &h, 2);
            }
            retval = 0;
        } else if (e.fnid == 0x15ae62f5) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 16 <= memSize) std::memset(mem + ptr, 0, 16);
            retval = 0;
        } else if (e.fnid == 0x4f488b9a) {
            retval = 0;
        } else if (e.fnid == 0xa322db75) {
            // r3 = videoOut, r4 = ptr to float (screen size in inches)
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                float sz = 40.0f;  // 40 inch default
                std::memcpy(mem + ptr, &sz, 4);
            }
            retval = 0;

        // ── cellSysutilAvc ───────────────────────────────────────────
        } else if (e.fnid == 0xc8f3e84f ||
                   e.fnid == 0xbd52cde0 ||
                   e.fnid == 0xb5ee2c29 ||
                   e.fnid == 0xee6f2fa5 ||
                   e.fnid == 0x71cfedfa) {
            retval = 0;

        // ── cellNetAoi ───────────────────────────────────────────────
        } else if (e.fnid == 0x48b2f1a8 || e.fnid == 0xa8a2b92e) {
            retval = 0;
        } else if (e.fnid == 0xfa710304) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x7a16b85c) {
            retval = 0;

        // ── cellVpost extras ─────────────────────────────────────────
        } else if (e.fnid == 0xce247a9c || e.fnid == 0xd28c4359) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x292d1115) {
            retval = 0;
        } else if (e.fnid == 0xf62c6004) {
            retval = 0;

        // ── sceNpCommerce2 extras ────────────────────────────────────
        } else if (e.fnid == 0x62023cf5) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x7370d4b4) {
            retval = 0;
        } else if (e.fnid == 0x0e10476b) {
            retval = 0;
        } else if (e.fnid == 0x1fa1b2e8) {
            retval = (int32_t)0x80550407;  // not ready

        // ── sceNpMatching2 extras ────────────────────────────────────
        } else if (e.fnid == 0x0b282cb1 ||
                   e.fnid == 0x1a16e826 ||
                   e.fnid == 0x54f1b23f ||
                   e.fnid == 0x3b89b01b) {
            retval = 0;
        } else if (e.fnid == 0x0ea3d356 ||
                   e.fnid == 0x4e4a5bca) {
            retval = 0;
        } else if (e.fnid == 0x8b14b92b) {
            retval = 0;

        // ── sys_dbg ──────────────────────────────────────────────────
        } else if (e.fnid == 0xeb52a81c ||
                   e.fnid == 0x71fba7a9) {
            retval = 0;
        } else if (e.fnid == 0x1ab4c9f2) {
            // r3 = ptr to list, r4 = num — zero-fill
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;

        // ── sys_ss ───────────────────────────────────────────────────
        } else if (e.fnid == 0x33195de0) {
            // r3 = ptr to buf, r4 = size — fill with pseudo-random
            uint32_t ptr = (uint32_t)st.gpr[3];
            uint32_t sz  = (uint32_t)st.gpr[4];
            if (ptr && sz > 0 && ptr + sz <= memSize) {
                for (uint32_t i = 0; i < sz; ++i)
                    mem[ptr + i] = (uint8_t)(nextHandleId * 7 + i * 13);
            }
            retval = 0;
        } else if (e.fnid == 0xa1c9f3e3) {
            // r3 = ptr to 16-byte console id
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                mem[ptr] = 0x01;  // type = retail
            }
            retval = 0;
        } else if (e.fnid == 0x8a4f7605) {
            retval = 0;

        // ── cellJpgDec extras ────────────────────────────────────────
        } else if (e.fnid == 0xb4fdda24) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xd1f838b9 || e.fnid == 0xbbbfa42c) {
            retval = 0;
        } else if (e.fnid == 0x8b18c8eb) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x5920178f) {
            // r3 = mainHandle, r4 = subHandle, r5 = ptr to CellJpgDecInfo
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                // width=1, height=1, components=3 (minimal)
                uint32_t one = __builtin_bswap32(1);
                std::memcpy(mem + ptr, &one, 4);
                std::memcpy(mem + ptr + 4, &one, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xc2a0d674) {
            retval = 0;
        } else if (e.fnid == 0xaf8bb012) {
            retval = 0;

        // ── cellPngDec extras ────────────────────────────────────────
        } else if (e.fnid == 0xf8d2000f) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x7515cc92 || e.fnid == 0x01c158db) {
            retval = 0;
        } else if (e.fnid == 0xebeab923) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x8cdae535) {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                uint32_t one = __builtin_bswap32(1);
                std::memcpy(mem + ptr, &one, 4);
                std::memcpy(mem + ptr + 4, &one, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x994e6188 ||
                   e.fnid == 0x32b3b60d) {
            retval = 0;

        // ── cellGifDec extras ────────────────────────────────────────
        } else if (e.fnid == 0x66315b6d) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xfe153bbf || e.fnid == 0x11ae5215) {
            retval = 0;
        } else if (e.fnid == 0xec45d754) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xc52c449b) {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                uint32_t one = __builtin_bswap32(1);
                std::memcpy(mem + ptr, &one, 4);
                std::memcpy(mem + ptr + 4, &one, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x0c0c12bd ||
                   e.fnid == 0xb3b69ad7) {
            retval = 0;

        // ── cellSaveData extras 2 ────────────────────────────────────
        } else if (e.fnid == 0x7b5e041a ||
                   e.fnid == 0x38a0f7d2 ||
                   e.fnid == 0xf6482036) {
            retval = 0;
        } else if (e.fnid == 0x6c9c5fac ||
                   e.fnid == 0xa3f30630) {
            retval = 0;

        // ── cellGcmSys extras 2 ──────────────────────────────────────
        } else if (e.fnid == 0x723c5c6c ||
                   e.fnid == 0xacee8542 ||
                   e.fnid == 0xd9b7653e) {
            retval = 0;
        } else if (e.fnid == 0xbd100dbc) {
            retval = 0;
        } else if (e.fnid == 0x9a0159af ||
                   e.fnid == 0x3b9bd5bd) {
            retval = 0;
        } else if (e.fnid == 0xdc09357e) {
            retval = 0;
        } else if (e.fnid == 0x0e6b0dae) {
            retval = 0;
        } else if (e.fnid == 0xbb42a3ff) {
            retval = 0;
        } else if (e.fnid == 0xa114ec67 ||
                   e.fnid == 0x2922aed0) {
            retval = 0;
        } else if (e.fnid == 0xbecb3dab) {
            retval = 0;
        } else if (e.fnid == 0x9dc04436 || e.fnid == 0xa75640e8) {
            retval = 0;

        // ── cellSpurs extras 2 ───────────────────────────────────────
        } else if (e.fnid == 0xb9bc6207 ||
                   e.fnid == 0x0a27c45e) {
            retval = 0;
        } else if (e.fnid == 0x1bf8a000) {
            retval = 6;  // 6 SPU threads
        } else if (e.fnid == 0x9a92c01a ||
                   e.fnid == 0xbdeaeff1) {
            retval = 0;

        // ── sys_tty ──────────────────────────────────────────────────
        } else if (e.fnid == 0xde3a9bf4) {
            retval = 0;  // no input
        } else if (e.fnid == 0x17d6a9e1) {
            // r3 = channel, r4 = ptr, r5 = len — debug print (ignore)
            retval = 0;

        // ── sys_time ─────────────────────────────────────────────────
        } else if (e.fnid == 0x01a2f171) {
            // r3 = ptr to sec, r4 = ptr to nsec
            uint32_t ps = (uint32_t)st.gpr[3];
            uint32_t pn = (uint32_t)st.gpr[4];
            if (ps && ps + 8 <= memSize) {
                uint64_t sec = 1700000000ULL;  // fixed epoch
                for (int i = 0; i < 8; ++i)
                    mem[ps + i] = (uint8_t)(sec >> (56 - 8*i));
            }
            if (pn && pn + 8 <= memSize) std::memset(mem + pn, 0, 8);
            retval = 0;
        } else if (e.fnid == 0x35168520) {
            retval = 79800000;  // 79.8 MHz (PS3 timebase)
        } else if (e.fnid == 0x8461e528) {
            retval = 0;

        // ── cellSync extras ──────────────────────────────────────────
        } else if (e.fnid == 0x24f80472) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) std::memset(mem + ptr, 0, 16);
            retval = 0;
        } else if (e.fnid == 0xcb40434b ||
                   e.fnid == 0xbb54b37f) {
            retval = 0;
        } else if (e.fnid == 0x268edd6d ||
                   e.fnid == 0x3c81b4da) {
            retval = 0;  // single-threaded: barrier passed
        } else if (e.fnid == 0x6c272124) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0xece0e664 || e.fnid == 0xba5bee48) {
            retval = 0;
        } else if (e.fnid == 0xbb73dee0 || e.fnid == 0x8a650722) {
            retval = 0;

        // ── cellSync queue ───────────────────────────────────────────
        } else if (e.fnid == 0xa5362e73) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 64 <= memSize) std::memset(mem + ptr, 0, 64);
            retval = 0;
        } else if (e.fnid == 0x48154c9b || e.fnid == 0x68af923c) {
            retval = 0;
        } else if (e.fnid == 0x0c7cb9f7 || e.fnid == 0x1bb675c2) {
            retval = (int32_t)0x80410101;  // CELL_SYNC_ERROR_EMPTY
        } else if (e.fnid == 0x4da349b2) {
            retval = 0;
        } else if (e.fnid == 0x167b0bfe) {
            retval = 0;
        } else if (e.fnid == 0xdcc99a07 || e.fnid == 0x74c37666) {
            retval = (int32_t)0x80410101;

        // ── cellFs extras 2 ──────────────────────────────────────────
        } else if (e.fnid == 0x967a162b) {
            // r3 = fd, r4 = ptr to blockSize, r5 = ptr to ioBlock
            uint32_t p4 = (uint32_t)st.gpr[4];
            uint32_t p5 = (uint32_t)st.gpr[5];
            if (p4 && p4 + 8 <= memSize) {
                uint64_t bs = 4096;
                for (int i = 0; i < 8; ++i) mem[p4+i] = (uint8_t)(bs >> (56-8*i));
            }
            if (p5 && p5 + 8 <= memSize) {
                uint64_t io = 4096;
                for (int i = 0; i < 8; ++i) mem[p5+i] = (uint8_t)(io >> (56-8*i));
            }
            retval = 0;
        } else if (e.fnid == 0xd86d2c67) {
            // r3 = fd, r4 = ptr to entries, r5 = size, r6 = ptr to count
            uint32_t pc = (uint32_t)st.gpr[6];
            if (pc && pc + 4 <= memSize) {
                uint32_t zero = 0;
                std::memcpy(mem + pc, &zero, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x9b19b22e || e.fnid == 0x58a17470 ||
                   e.fnid == 0x1cf98800) {
            retval = 0;
        } else if (e.fnid == 0x4cef342e || e.fnid == 0x8f71c5b2) {
            retval = 0;
        } else if (e.fnid == 0x1a108ab7) {
            // r3 = path, r4 = ptr to blockSize, r5 = ptr to freeBlocks
            uint32_t p4 = (uint32_t)st.gpr[4];
            uint32_t p5 = (uint32_t)st.gpr[5];
            if (p4 && p4 + 4 <= memSize) {
                uint32_t bs = __builtin_bswap32(4096);
                std::memcpy(mem + p4, &bs, 4);
            }
            if (p5 && p5 + 4 <= memSize) {
                uint32_t fb = __builtin_bswap32(1024 * 1024);  // ~4GB free
                std::memcpy(mem + p5, &fb, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x7a7b98c4 || e.fnid == 0xc1c507e7) {
            retval = 0;

        // ── cellAtrac extras ─────────────────────────────────────────
        } else if (e.fnid == 0xeeb4289f) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x34e241d7) {
            retval = 0;
        } else if (e.fnid == 0x99fb73d1) {
            retval = 0;
        } else if (e.fnid == 0x1c5004f5) {
            retval = 0;
        } else if (e.fnid == 0xdfab73aa) {
            retval = 0;

        // ── cellVdec extras ──────────────────────────────────────────
        } else if (e.fnid == 0x9afd0b7d) {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x4e4ae73c) {
            retval = (int32_t)0x80610102;
        } else if (e.fnid == 0x807c86f4) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) {
                std::memset(mem + ptr, 0, 32);
                uint32_t ms = __builtin_bswap32(4 * 1024 * 1024);
                std::memcpy(mem + ptr + 4, &ms, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xc460cd15) {
            retval = 0;

        // ── cellAdec extras ──────────────────────────────────────────
        } else if (e.fnid == 0xf0e6e1df) {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x382b0cf4) {
            retval = (int32_t)0x80610102;
        } else if (e.fnid == 0xf07a2ec7) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                uint32_t ms = __builtin_bswap32(1 * 1024 * 1024);
                std::memcpy(mem + ptr + 4, &ms, 4);
            }
            retval = 0;
        } else if (e.fnid == 0xcc534abe || e.fnid == 0xccfeefc2) {
            retval = 0;

        // ── cellDmux extras ──────────────────────────────────────────
        } else if (e.fnid == 0xa2d4189b) {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 4 <= memSize) {
                uint32_t h = __builtin_bswap32(nextHandleId++);
                std::memcpy(mem + ptr, &h, 4);
            }
            retval = 0;
        } else if (e.fnid == 0x143a3968 || e.fnid == 0x11bc3a6c) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 16 <= memSize) std::memset(mem + ptr, 0, 16);
            retval = 0;

        // ── cellRtc extras ───────────────────────────────────────────
        } else if (e.fnid == 0x9c1aa28d || e.fnid == 0x7767eeb3) {
            retval = 0;
        } else if (e.fnid == 0x17947909 ||
                   e.fnid == 0x26f2c987 ||
                   e.fnid == 0x42b9316f ||
                   e.fnid == 0xc2d8cf95 ||
                   e.fnid == 0x5316b4a6 ||
                   e.fnid == 0xd41d3bd2) {
            retval = 0;
        } else if (e.fnid == 0x18466b0e ||
                   e.fnid == 0xead86714) {
            retval = 0;
        } else if (e.fnid == 0xa92c8e5d) {
            // r3 = ptr to buf, r4 = ptr to CellRtcDateTime
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 32 <= memSize) {
                const char* fixed = "2024-01-01T00:00:00Z";
                std::memcpy(mem + ptr, fixed, 20);
                mem[ptr + 20] = 0;
            }
            retval = 0;

        // ── sceNpTrophy extras ───────────────────────────────────────
        } else if (e.fnid == 0x99d5fcdb) {
            retval = 0;
        } else if (e.fnid == 0x5ce3c18f ||
                   e.fnid == 0x0cde7100 ||
                   e.fnid == 0x7775c461) {
            retval = 0;
        } else if (e.fnid == 0xdd29bb02) {
            uint32_t ptr = (uint32_t)st.gpr[5];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;

        // ── sceNp extras ─────────────────────────────────────────────
        } else if (e.fnid == 0xab73f551) {
            retval = (int32_t)0x80550d04;  // no events
        } else if (e.fnid == 0xbda9fdc1) {
            retval = 0;
        } else if (e.fnid == 0x16a488a8) {
            retval = 0;
        } else if (e.fnid == 0x726b3b91 ||
                   e.fnid == 0xf1cbf5b0) {
            retval = 0;

        // ── cellNetCtl extras ────────────────────────────────────────
        } else if (e.fnid == 0xbd5a59fc) {
            uint32_t ptr = (uint32_t)st.gpr[4];
            if (ptr && ptr + 32 <= memSize) std::memset(mem + ptr, 0, 32);
            retval = 0;
        } else if (e.fnid == 0x1e585b5d) {
            uint32_t ptr = (uint32_t)st.gpr[3];
            if (ptr && ptr + 16 <= memSize) {
                std::memset(mem + ptr, 0, 16);
                mem[ptr] = 3;  // NAT type 3
            }
            retval = 0;
        } else if (e.fnid == 0xf9a26fb5 ||
                   e.fnid == 0xa42c2d7e) {
            retval = 0;

        // ── cellSsl extras ───────────────────────────────────────────
        } else if (e.fnid == 0x10c1bffc ||
                   e.fnid == 0xab209809 ||
                   e.fnid == 0x7a9ccf22 ||
                   e.fnid == 0x32c61bdc ||
                   e.fnid == 0x5d9a9e24) {
            retval = 0;

        // ── cellHttps ────────────────────────────────────────────────
        } else if (e.fnid == 0x1b5b3d2e || e.fnid == 0xd17fc413) {
            retval = 0;
        } else if (e.fnid == 0x9f9702c6 ||
                   e.fnid == 0x74f97476) {
            retval = 0;

        // ── cellPadFilter ────────────────────────────────────────────
        } else if (e.fnid == 0xaa54cfae ||
                   e.fnid == 0x0fb7a6bb) {
            retval = 0;

        // ── cellImeJp extras ─────────────────────────────────────────
        } else if (e.fnid == 0x2619237f || e.fnid == 0x8c358fd3) {
            retval = 0;
        } else if (e.fnid == 0x7f4e3c55 ||
                   e.fnid == 0xea1d1638 ||
                   e.fnid == 0xa3c3ae0e) {
            retval = 0;

        // ── sys_trace ────────────────────────────────────────────────
        } else if (e.fnid == 0x6c71cc3c || e.fnid == 0x5f532aa3 ||
                   e.fnid == 0x1a83a714 || e.fnid == 0x3c0b3ba0) {
            retval = 0;

        // ── sys_interrupt ────────────────────────────────────────────
        } else if (e.fnid == 0x5c1b6e32 ||
                   e.fnid == 0xdf36ccc6 ||
                   e.fnid == 0xa1b8cc63) {
            retval = 0;

        // ── sys_overlay ──────────────────────────────────────────────
        } else if (e.fnid == 0xd7aa07b0) {
            retval = nextHandleId++;
        } else if (e.fnid == 0x6396bdd6) {
            retval = 0;

        // ── cellSpurs workload ───────────────────────────────────────
        } else if (e.fnid == 0x2e07e92f ||
                   e.fnid == 0x9fcb6beb) {
            retval = 0;
        } else if (e.fnid == 0x1656a731 ||
                   e.fnid == 0x25957ef8) {
            retval = 0;
        } else if (e.fnid == 0x1e76a16b) {
            retval = 0;

        // ── sceNpManager extras ──────────────────────────────────────
        } else if (e.fnid == 0x25b41bc1) {
            retval = 18;
        } else if (e.fnid == 0xbac7287f) {
            retval = 0;  // no restriction
        } else if (e.fnid == 0x311d4297) {
            retval = 0;
        } else if (e.fnid == 0x42281240) {
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
