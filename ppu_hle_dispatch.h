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
    uint32_t libcHeapEnd  = 0x00D00000;

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

    // Extra PC-indexed handlers for non-import functions we want to
    // short-circuit (e.g. the ELF's internal malloc/free).
    enum class Builtin { Malloc, Free, None };
    std::unordered_map<uint32_t, Builtin> builtinByPc;

    void addBuiltinMalloc(uint32_t pc) { builtinByPc[pc] = Builtin::Malloc; }
    void addBuiltinFree(uint32_t pc)   { builtinByPc[pc] = Builtin::Free;   }

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
                if (bit->second == Builtin::Malloc) {
                    // r3 = size (PS3 libc wrapper passes ptr-to-heap as
                    // r3 and size as r4; but the outer wrapper at 0x1fed4
                    // is called with r3=size, r4=alignment maybe).  We
                    // treat r3 OR r4 as the size — pick the largest
                    // non-zero value up to 16 MB.
                    uint64_t s3 = st.gpr[3], s4 = st.gpr[4];
                    uint64_t size = (s3 && s3 < 0x1000000) ? s3 :
                                    (s4 && s4 < 0x1000000) ? s4 : 0x100;
                    uint32_t align = 16;
                    libcHeapNext = (libcHeapNext + align - 1) & ~(align - 1);
                    uint32_t ptr = libcHeapNext;
                    if (libcHeapNext + size <= libcHeapEnd) {
                        libcHeapNext += (uint32_t)size;
                    } else {
                        ptr = 0; // oom
                    }
                    // zero the returned region in both host & guest mem
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
            retval = 0;
            virtTime += 1000;
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
            // Real HW writes a NV flip command into the FIFO. Skip; the
            // game's FIFO is being processed by our RSX emulator
            // separately (or will be once we wire it up).
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
                ioAddr = 0x00F80000;         // 512 KB scratch FIFO
                if (ioSize == 0) ioSize = 0x80000;
                std::printf("  [HLE] _cellGcmInitBody: synth ioAddr=0x%x ioSize=0x%x\n",
                            ioAddr, ioSize);
            }
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
                   e.name == "cellGcmGetConfiguration") {
            retval = 0;
        } else if (e.name == "cellVideoOutConfigure" ||
                   e.name == "cellVideoOutGetResolution" ||
                   e.name == "cellVideoOutGetState") {
            // Clear the output-state struct if given; return success.
            uint32_t outp = (uint32_t)st.gpr[5];
            if (outp && outp + 64 <= memSize) {
                std::memset(mem + outp, 0, 64);
                megakernel_write_mem((uint64_t)outp, mem + outp, 64);
            }
            retval = 0;
        } else if (e.name == "cellSysutilRegisterCallback" ||
                   e.name == "cellSysutilUnregisterCallback" ||
                   e.name == "cellSysutilCheckCallback") {
            retval = 0;
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
