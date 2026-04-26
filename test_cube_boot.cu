// test_cube_boot.cu — Load RPCS3's gs_gcm_cube.elf (textured 3D cube),
// boot the PPC megakernel, HLE-dispatch all 52 imports, and capture
// the RSX FIFO stream. This is the next milestone after triangle:
// cube uses textures, depth/stencil, matrix transforms, and input polling.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <utility>

#include "elf_loader.h"
#include "ppu_hle_names.h"
#include "ppu_hle_dispatch.h"
#include "ppc_defs.h"
#include "rsx_defs.h"
#include "rsx_fp_shader.h"
#include "rsx_vp_shader.h"
#include "rsx_raster.h"
#include "rsx_raster_bridge.h"

extern "C" {
    int   megakernel_init();
    int   megakernel_load(uint64_t, const void*, size_t);
    int   megakernel_set_entry(uint64_t, uint64_t, uint64_t);
    int   megakernel_set_code_range(uint64_t, uint64_t);
    float megakernel_run(uint32_t);
    int   megakernel_read_state(ppc::PPEState*);
    int   megakernel_write_state(const ppc::PPEState*);
    int   megakernel_read_mem(uint64_t, void*, size_t);
    int   megakernel_write_mem(uint64_t, const void*, size_t);
    int   megakernel_read_hle_log(uint32_t*, int);
    int   megakernel_read_poll_state(uint64_t*, uint32_t*, uint64_t*, uint32_t*);
    void  megakernel_dump_poll_state();
    void  megakernel_shutdown();
}

namespace rsx {
    int  rsx_init(RSXState* state);
    int  rsx_process_fifo(RSXState* state, const uint32_t* fifo, uint32_t fifoSize,
                          uint8_t* vram, uint32_t maxCmds);
    void rsx_shutdown(RSXState* state);
}

static uint32_t be32(const uint8_t* p) {
    return (uint32_t)p[0]<<24 | (uint32_t)p[1]<<16 | (uint32_t)p[2]<<8 | p[3];
}

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  PS3 cube sample boot attempt (gs_gcm_cube.elf)\n");
    std::printf("══════════════════════════════════════════════════════\n");

    const char* path = "/workspaces/codespace/rpcs3-src/bin/test/gs_gcm_cube.elf";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CHECK((bool)f, "open cube ELF");
    if (!f) { std::printf("ABORT: cannot open %s\n", path); return 1; }
    auto sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> file(sz);
    f.read(reinterpret_cast<char*>(file.data()), sz);

    std::vector<uint8_t> mem(128u * 1024u * 1024u, 0);
    PS3ExeInfo info{};
    int rc = ps3_load_elf(file.data(), file.size(),
                          mem.data(), mem.size(), &info);
    CHECK(rc == ELF_OK, "ps3_load_elf");
    std::printf("  e_entry=0x%lx  segments=%d  fileSize=%ld\n",
                (unsigned long)info.entry_point, info.num_segments, (long)sz);

    // ── Resolve 32-bit function descriptor at e_entry ──
    uint32_t descEntry = be32(mem.data() + info.entry_point);
    uint32_t descTOC   = be32(mem.data() + info.entry_point + 4);
    std::printf("  descriptor[0x%lx]: entry=0x%x toc=0x%x\n",
                (unsigned long)info.entry_point, descEntry, descTOC);
    CHECK(descEntry >= 0x10000 && descEntry < 0x80000, "entry in code seg");

    // ── Parse section headers: find .lib.stub + resolve imports ──
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)file.data();
    uint64_t shoff   = elf_bswap64(eh->e_shoff);
    uint16_t shnum   = elf_bswap16(eh->e_shnum);
    uint16_t shentsz = elf_bswap16(eh->e_shentsize);
    uint16_t shstrndx= elf_bswap16(eh->e_shstrndx);
    const Elf64_Shdr* strSh = (const Elf64_Shdr*)(
        file.data() + shoff + (uint64_t)shstrndx * shentsz);
    const char* strtab = (const char*)(file.data() + elf_bswap64(strSh->sh_offset));
    auto findSec = [&](const char* name) -> const Elf64_Shdr* {
        for (uint16_t i = 0; i < shnum; ++i) {
            const Elf64_Shdr* sh = (const Elf64_Shdr*)(
                file.data() + shoff + (uint64_t)i * shentsz);
            if (!std::strcmp(strtab + elf_bswap32(sh->sh_name), name)) return sh;
        }
        return nullptr;
    };

    const Elf64_Shdr* shStub     = findSec(".lib.stub");
    const Elf64_Shdr* shStubText = findSec(".sceStub.text");
    CHECK(shStub && shStubText, "stub sections present");

    // Build stub_pc → {module, name} map.
    struct StubInfo { const char* mod; const char* name; uint32_t fnid; };
    std::unordered_map<uint32_t, StubInfo> stubMap;
    uint32_t stubTextLo = (uint32_t)elf_bswap64(shStubText->sh_addr);
    uint32_t stubTextHi = stubTextLo + (uint32_t)elf_bswap64(shStubText->sh_size);

    uint64_t stubAddr = elf_bswap64(shStub->sh_addr);
    uint64_t stubSize = elf_bswap64(shStub->sh_size);
    int totalFuncs = 0, resolved = 0;
    for (uint64_t a = stubAddr; a + 0x2C <= stubAddr + stubSize; a += 0x2C) {
        const uint8_t* s = mem.data() + a;
        uint16_t num_func    = (uint16_t)((s[6] << 8) | s[7]);
        uint32_t func_fnid_a = be32(s + 0x14);
        uint32_t func_stub_a = be32(s + 0x18);
        uint32_t mod_name_a  = be32(s + 0x10);
        const char* modName = (mod_name_a && mod_name_a + 32 < mem.size())
            ? (const char*)(mem.data() + mod_name_a) : "?";
        for (uint16_t k = 0; k < num_func; ++k) {
            uint32_t fn = be32(mem.data() + func_fnid_a + k * 4);
            uint32_t pc = be32(mem.data() + func_stub_a + k * 4);
            const PpuHleEntry* e = ppu_hle_lookup(fn);
            StubInfo si { modName, e ? e->name : "<unresolved>", fn };
            stubMap[pc] = si;
            if (e) resolved++;
            totalFuncs++;
        }
    }
    std::printf("  stub map: %zu entries  (%d/%d FNIDs resolved to names)\n",
                stubMap.size(), resolved, totalFuncs);
    std::printf("  .sceStub.text range: [0x%x .. 0x%x)\n", stubTextLo, stubTextHi);
    CHECK(resolved == totalFuncs, "all 52 cube FNIDs resolved");

    // ── Boot ──
    CHECK(megakernel_init() != 0, "megakernel_init");
    uint64_t hi = 0;
    for (int i = 0; i < info.num_segments; ++i) {
        uint64_t end = info.segments[i].addr + info.segments[i].size;
        if (end > hi) hi = end;
    }
    uint64_t loadSpan = (hi + 0xFFFFu) & ~uint64_t(0xFFFFu);
    if (loadSpan > 32u * 1024u * 1024u) loadSpan = 32u * 1024u * 1024u;
    megakernel_load(0, mem.data(), loadSpan);
    std::printf("  loaded %lu bytes; entry=0x%x toc=0x%x\n",
                (unsigned long)loadSpan, descEntry, descTOC);

    megakernel_set_entry(descEntry, 0x00F00000ULL, descTOC);
    // Cube code range is larger than triangle — find .text bounds
    uint32_t codeHi = descEntry + 0x20000;  // generous default
    const Elf64_Shdr* shText = findSec(".text");
    if (shText) {
        codeHi = (uint32_t)(elf_bswap64(shText->sh_addr) + elf_bswap64(shText->sh_size));
        codeHi = (codeHi + 0xFF) & ~0xFFu;
    }
    megakernel_set_code_range(descEntry & ~0xFFFFu, codeHi);
    std::printf("  code range: [0x%x .. 0x%x)\n", descEntry & ~0xFFFFu, codeHi);

    // Build dispatcher from stubMap.
    PpuHleDispatcher disp;
    for (auto& kv : stubMap) {
        disp.add(kv.first, kv.second.fnid,
                 kv.second.mod, kv.second.name);
    }

    // HLE-patch the ELF's internal libc allocator — cube version.
    // The REAL allocators are at 0x22568 (memalign) and 0x225a0 (malloc).
    // Previous patches at 0x1e834/0x1e930/0x1e954 were WRONG — those are
    // panic/assert functions (sys_tty_write + tw trap), not allocators.
    disp.addBuiltinMemalign(0x22568);  // memalign(alignment=r3, size=r4)
    disp.addBuiltinMalloc(0x225a0);    // malloc(size=r3)
    disp.addBuiltinFree(0x225d8);      // free(ptr=r3)
    disp.addBuiltinRealloc(0x22604);   // realloc(ptr=r3, size=r4)
    disp.addBuiltinMmapAlloc(0x154dc); // mmap-style alloc(size=r3, align=r4) — 6 callers in GCM init

    // Single-step dispatch loop — run until exit, stall, or step limit.
    int hleHits = 0, steps = 0, maxSteps = 16000000;
    bool stopped = false;
    int stallCount = 0;
    uint32_t stallPc = 0;
    int lastHleStep = 0;
    std::unordered_map<uint32_t, uint64_t> pcBuckets;
    std::unordered_map<uint32_t, uint64_t> blTargets;
    uint64_t prevLR = 0;

    for (steps = 0; steps < maxSteps && !stopped; ++steps) {
        // Fast-forward pure compute loops: if PC is in the hot function
        // (0x12d34..0x12fc0) and no HLE patches are in that range, batch run.
        {
            ppc::PPEState peek{};
            megakernel_read_state(&peek);
            uint32_t ppc = (uint32_t)peek.pc;
            if (ppc >= 0x12d34 && ppc < 0x12fc0 &&
                !disp.byPc.count(ppc) && !disp.builtinByPc.count(ppc)) {
                int batch = 500;
                megakernel_run(batch);
                steps += batch - 1;  // account for batch
                megakernel_read_state(&peek);
                ppc = (uint32_t)peek.pc;
                if ((steps & 0xFF) == 0) pcBuckets[ppc & ~0x3Fu]++;
                continue;
            }
        }
        megakernel_run(1);
        ppc::PPEState st{};
        megakernel_read_state(&st);
        uint32_t pc = (uint32_t)st.pc;
        if ((steps & 0xF) == 0) pcBuckets[pc & ~0x3Fu]++;
        if (st.lr != prevLR) { blTargets[pc]++; prevLR = st.lr; }
        if ((steps % 500000) == 0) {
            std::printf("  progress: step=%d PC=0x%x LR=0x%llx\n",
                        steps, pc, (unsigned long long)st.lr);
        }
        // Trace key function entries
        if (pc == 0x121f8) std::printf("  [TRACE] step=%d ENTER 0x121f8 (display buf setup) LR=0x%llx\n", steps, (unsigned long long)st.lr);
        if (pc == 0x12420) std::printf("  [TRACE] step=%d ENTER 0x12420 (tiled pitch fn) LR=0x%llx\n", steps, (unsigned long long)st.lr);
        if (pc == 0x154dc) std::printf("  [TRACE] step=%d ENTER 0x154dc (alloc fn) LR=0x%llx r3=0x%llx r4=0x%llx\n", steps, (unsigned long long)st.lr, (unsigned long long)st.gpr[3], (unsigned long long)st.gpr[4]);
        if (pc == 0x1e930) std::printf("  [TRACE] step=%d ENTER 0x1e930 (ASSERT!) LR=0x%llx\n", steps, (unsigned long long)st.lr);
        if (pc == 0x1e940) std::printf("  [TRACE] step=%d PC=0x1e940 (tw TRAP!) LR=0x%llx\n", steps, (unsigned long long)st.lr);
        if (pc == 0x1225c) std::printf("  [TRACE] step=%d PC=0x1225c (alloc OK path) r0=0x%llx\n", steps, (unsigned long long)st.gpr[0]);
        if (pc == 0x12270) std::printf("  [TRACE] step=%d PC=0x12270 (bl cellGcmAddrToOff) r3=0x%llx r4=0x%llx\n", steps, (unsigned long long)st.gpr[3], (unsigned long long)st.gpr[4]);
        if (pc == 0x12248) std::printf("  [TRACE] step=%d PC=0x12248 (branch check) r0=0x%llx\n", steps, (unsigned long long)st.gpr[0]);
        if (st.halted) std::printf("  [TRACE] step=%d HALTED at PC=0x%x LR=0x%llx\n", steps, pc, (unsigned long long)st.lr);
        // Init sequence flow
        if (pc == 0x12bd8) std::printf("  [TRACE] step=%d ENTER 0x12bd8 LR=0x%llx\n", steps, (unsigned long long)st.lr);
        if (pc == 0x11ee4) std::printf("  [TRACE] step=%d ENTER 0x11ee4 LR=0x%llx\n", steps, (unsigned long long)st.lr);
        if (pc == 0x13000) std::printf("  [TRACE] step=%d ENTER 0x13000 LR=0x%llx\n", steps, (unsigned long long)st.lr);
        if (pc == 0x108a0) std::printf("  [TRACE] step=%d ENTER 0x108a0 (RENDER LOOP!) LR=0x%llx\n", steps, (unsigned long long)st.lr);
        if (pc == 0x12d34) std::printf("  [TRACE] step=%d ENTER 0x12d34 (hot fn) LR=0x%llx\n", steps, (unsigned long long)st.lr);
        if (pc == 0x145a0 && steps > 100) std::printf("  [TRACE] step=%d RE-ENTER 0x145a0 LR=0x%llx\n", steps, (unsigned long long)st.lr);
        // Inner loop bound trace (once)
        static bool innerTraced = false;
        if (pc == 0x12efc && !innerTraced) {
            innerTraced = true;
            std::printf("  [TRACE] step=%d INNER LOOP cmp: r0(ctr)=0x%llx r9(bound)=? cr7 gpr0=%lld gpr9=%lld\n",
                steps, (unsigned long long)st.gpr[0], (long long)st.gpr[0], (long long)st.gpr[9]);
        }
        // Outer loop bound trace (once)
        static bool outerTraced = false;
        if (pc == 0x12f44 && !outerTraced) {
            outerTraced = true;
            std::printf("  [TRACE] step=%d OUTER LOOP cmp: gpr0=%lld gpr9=%lld\n",
                steps, (long long)st.gpr[0], (long long)st.gpr[9]);
        }
        if (pc == stallPc) {
            if (++stallCount > 32768) {
                std::printf("  stalled at PC=0x%x for %d steps; stopping.\n",
                            pc, stallCount);
                break;
            }
        } else { stallPc = pc; stallCount = 0; }
        // If no HLE calls for 8M+ steps after rendering started, likely stuck
        // in flip-wait loop — break early to avoid wasting time.
        // (Must be > 6M to survive the 256×256 vertex compute at 0x12d34)
        if (hleHits > 50 && (steps - lastHleStep) > 8000000) {
            std::printf("  no HLE progress for %d steps; likely flip-wait — stopping.\n",
                        steps - lastHleStep);
            break;
        }
        // Specific flip-wait loop detector: PC 0x162b0-0x162c4
        // Instead of stopping, fake-complete the wait by writing the
        // expected reference value to CellGcmControl, then jump past
        // the loop to continue execution for more frames.
        if (pc >= 0x162b0 && pc <= 0x162c4) {
            static int flipWaitCount = 0;
            static int flipWaitSkips = 0;
            if (++flipWaitCount > 50000) {
                // r31 holds the expected reference value, r30 holds the
                // CellGcmControl pointer. Write ref = expected so cmpw passes.
                uint32_t ctrlPtr = (uint32_t)st.gpr[30];
                uint32_t expected = (uint32_t)st.gpr[31];
                if (ctrlPtr && ctrlPtr + 8 < mem.size()) {
                    // Write expected value to CellGcmControl.ref (offset +8)
                    mem[ctrlPtr+8] = (expected >> 24) & 0xFF;
                    mem[ctrlPtr+9] = (expected >> 16) & 0xFF;
                    mem[ctrlPtr+10] = (expected >> 8) & 0xFF;
                    mem[ctrlPtr+11] = expected & 0xFF;
                    megakernel_write_mem(ctrlPtr+8, &mem[ctrlPtr+8], 4);
                }
                flipWaitCount = 0;
                flipWaitSkips++;
                if (flipWaitSkips >= 100) {
                    std::printf("  flip-wait loop detected at PC=0x%x (%d skips); stopping.\n",
                                pc, flipWaitSkips);
                    break;
                }
            }
        }

        if (disp.byPc.count(pc) || disp.builtinByPc.count(pc)) {
            bool haltReq = false;
            const char* nm = disp.dispatch(st, mem.data(), mem.size(),
                                           haltReq);
            megakernel_write_state(&st);
            hleHits++;
            lastHleStep = steps;
            if (hleHits <= 200 || haltReq) {
                const char* mod = disp.byPc.count(pc) ? disp.byPc[pc].mod.c_str() : "libc-hle";
                std::printf("  HLE #%d step=%d PC=0x%x  %s::%s  LR→0x%llx\n",
                            hleHits, steps, pc, mod,
                            nm ? nm : "?",
                            (unsigned long long)st.pc);
            }
            if (haltReq) {
                std::printf("  >>> Program requested exit; stopping.\n");
                stopped = true;
            }
            continue;
        }
        if (pc == 0 || st.halted) {
            std::printf("  halted/null at step %d PC=0x%x LR=0x%llx\n",
                        steps, pc, (unsigned long long)st.lr);
            break;
        }
    }
    std::printf("  steps=%d  HLE=%d\n", steps, hleHits);

    // Poll detector state
    {
        uint64_t pt=0, at=0; uint32_t pc_c=0, ac=0;
        megakernel_read_poll_state(&pt, &pc_c, &at, &ac);
        std::printf("  poll detector: primary=0x%llx (hits=%u), alt=0x%llx (hits=%u)\n",
                    (unsigned long long)pt, pc_c,
                    (unsigned long long)at, ac);
    }
    // Final state
    {
        ppc::PPEState fs{};
        megakernel_read_state(&fs);
        std::printf("  final PC=0x%x  LR=0x%llx  r1=0x%llx  r3=0x%llx\n",
                    (uint32_t)fs.pc,
                    (unsigned long long)fs.lr,
                    (unsigned long long)fs.gpr[1],
                    (unsigned long long)fs.gpr[3]);
    }
    // Unimplemented opcode log
    {
        uint32_t log[257] = {};
        int n = megakernel_read_hle_log(log, 256);
        if (n > 0) {
            std::printf("  unknown syscall/opcode log (%d entries):", n);
            for (int i = 0; i < n && i < 16; ++i)
                std::printf(" 0x%x", log[1 + i]);
            std::printf("\n");
            std::unordered_map<uint32_t, int> opCounts;
            for (int i = 0; i < n; ++i) {
                uint32_t tag = log[1 + i];
                if ((tag & 0xFF000000u) == 0xDE000000u)
                    opCounts[tag & 0xFFFFFu]++;
            }
            if (!opCounts.empty()) {
                std::vector<std::pair<uint32_t,int>> v(opCounts.begin(), opCounts.end());
                std::sort(v.begin(), v.end(),
                          [](auto& a, auto& b){ return a.second > b.second; });
                std::printf("  unimpl opcode histogram:\n");
                int k = 0;
                for (auto& p : v) {
                    std::printf("    OPCD=%u XO=%u  (%d)\n",
                                (p.first >> 10) & 0x3F, p.first & 0x3FF, p.second);
                    if (++k >= 12) break;
                }
            }
        }
    }
    disp.print_summary();

    // Hot PC buckets
    {
        std::vector<std::pair<uint32_t, uint64_t>> sorted(pcBuckets.begin(), pcBuckets.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::printf("  hot PC buckets (64B, sampled 1:16):\n");
        int n = 0;
        for (const auto& p : sorted) {
            std::printf("    0x%08x  %llu\n", p.first, (unsigned long long)p.second);
            if (++n >= 20) break;
        }
    }
    // Top bl targets (skipped — crashes on corrupted map)

    // ── FIFO capture + RSX replay ──
    bool gcmInitRan = disp.handledHistogram.count("_cellGcmInitBody") > 0;
    if (gcmInitRan && disp.gcmIoBase) {
        std::printf("  GCM FIFO: io=[0x%x..0x%x) cmdSize=0x%x\n",
                    disp.gcmIoBase, disp.gcmIoBase + disp.gcmIoSize,
                    disp.gcmCmdSize);
        uint8_t ctxBytes[16] = {};
        megakernel_read_mem(disp.gcmContextAddr, ctxBytes, 16);
        auto rd = [&](int off) {
            return (uint32_t)ctxBytes[off]<<24 | (uint32_t)ctxBytes[off+1]<<16
                 | (uint32_t)ctxBytes[off+2]<<8 | ctxBytes[off+3];
        };
        uint32_t ctxBegin = rd(0), ctxEnd = rd(4), ctxCur = rd(8), ctxCb = rd(12);
        std::printf("  GCM ctx: begin=0x%x end=0x%x current=0x%x cb=0x%x\n",
                    ctxBegin, ctxEnd, ctxCur, ctxCb);
        if (ctxCur > ctxBegin && ctxCur - ctxBegin < 0x2000000) {
            uint32_t bytes = ctxCur - ctxBegin;
            std::vector<uint8_t> fifoBE(bytes);
            megakernel_read_mem(ctxBegin, fifoBE.data(), bytes);
            uint32_t words = bytes / 4;
            std::vector<uint32_t> fifoLE(words);
            for (uint32_t j = 0; j < words; ++j) {
                fifoLE[j] = (uint32_t)fifoBE[j*4]<<24
                           | (uint32_t)fifoBE[j*4+1]<<16
                           | (uint32_t)fifoBE[j*4+2]<<8
                           | fifoBE[j*4+3];
            }
            // Scan for NV method headers and histogram
            std::unordered_map<uint32_t, uint32_t> methodHist;
            uint32_t nonzero = 0, methods = 0;
            uint32_t i = 0;
            while (i < words) {
                uint32_t w = fifoLE[i];
                if (w) nonzero++;
                uint32_t count = (w >> 18) & 0x7FF;
                uint32_t method = w & 0xFFFC;
                bool isMethod = (count > 0 && count < 2048 &&
                                 method > 0 && method < 0x2000 &&
                                 (w & 0x00030000) == 0);
                if (isMethod) {
                    methods++;
                    methodHist[method]++;
                    i += 1 + count;
                } else {
                    i++;
                }
            }
            std::printf("  GCM FIFO bytes written: %u  (%u nonzero, ~%u method packets)\n",
                        bytes, nonzero, methods);

            auto nvMethodName = [](uint32_t m) -> const char* {
                switch (m) {
                    case 0x0200: return "SURFACE_CLIP_H";
                    case 0x0204: return "SURFACE_CLIP_V";
                    case 0x0208: return "SURFACE_FORMAT";
                    case 0x020c: return "SURFACE_PITCH_A";
                    case 0x0210: return "SURFACE_COLOR_AOFFSET";
                    case 0x0214: return "SURFACE_ZETA_OFFSET";
                    case 0x0218: return "SURFACE_COLOR_BOFFSET";
                    case 0x021c: return "SURFACE_PITCH_B";
                    case 0x0220: return "SURFACE_COLOR_TARGET";
                    case 0x0300: return "CLEAR_RECT_HORIZONTAL";
                    case 0x0304: return "CLEAR_DEPTH_STENCIL_VALUE";
                    case 0x030C: return "CLEAR_COLOR_VALUE";
                    case 0x0310: return "CLEAR_SURFACE";
                    case 0x0a00: return "VIEWPORT_H";
                    case 0x0a04: return "VIEWPORT_V";
                    case 0x0a20: return "VIEWPORT_SCALE_X";
                    case 0x0a24: return "VIEWPORT_SCALE_Y";
                    case 0x0a28: return "VIEWPORT_SCALE_Z";
                    case 0x0a2c: return "VIEWPORT_SCALE_W";
                    case 0x0a30: return "VIEWPORT_OFFSET_X";
                    case 0x0a34: return "VIEWPORT_OFFSET_Y";
                    case 0x0a38: return "VIEWPORT_OFFSET_Z";
                    case 0x0a3c: return "VIEWPORT_OFFSET_W";
                    case 0x1680: return "VERTEX_DATA_ARRAY_OFFSET[0]";
                    case 0x1740: return "VERTEX_DATA_ARRAY_FORMAT[0]";
                    case 0x1808: return "DRAW_ARRAYS";
                    case 0x1800: return "BEGIN_END";
                    case 0x1828: return "BIND_VERTEX_ATTRIB";
                    case 0x1B40: return "BLEND_ENABLE";
                    case 0x1B68: return "DEPTH_TEST_ENABLE";
                    case 0x1D78: return "DEPTH_FUNC";
                    case 0x1D94: return "CLEAR_DEPTH_ENABLE";
                    case 0x1E9C: return "VP_PROGRAM_LOAD";
                    case 0x1EA0: return "VP_PROGRAM_START";
                    case 0x1EF8: return "SHADER_PROGRAM";
                    case 0x1EFC: return "VP_CONSTANT_LOAD";
                    case 0x1F00: return "VP_CONSTANT[0]";
                    case 0x1720: return "DRAW_INDEX_ARRAY";
                    case 0x0380: return "TEXTURE_OFFSET[0]";
                    case 0x0384: return "TEXTURE_FORMAT[0]";
                    case 0x0388: return "TEXTURE_ADDRESS[0]";
                    case 0x038C: return "TEXTURE_CONTROL0[0]";
                    case 0x0394: return "TEXTURE_FILTER[0]";
                    case 0x0398: return "TEXTURE_IMAGE_RECT[0]";
                    default:     return "?";
                }
            };
            std::vector<std::pair<uint32_t,uint32_t>> sorted(
                methodHist.begin(), methodHist.end());
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b){ return a.second > b.second; });
            std::printf("  top RSX methods (offset → count → name):\n");
            int kk = 0;
            for (const auto& p : sorted) {
                std::printf("    0x%04x  %6u  %s\n",
                            p.first, p.second, nvMethodName(p.first));
                if (++kk >= 25) break;
            }

            // Replay through rsx_process_fifo + rasterizer
            constexpr uint32_t W = 1280, H = 720;
            constexpr uint32_t VRAM_BYTES = 64u * 1024u * 1024u;
            std::vector<uint8_t> vram(VRAM_BYTES, 0);
            {
                uint32_t mirror = std::min<uint32_t>(VRAM_BYTES, 32u*1024u*1024u);
                megakernel_read_mem(0, vram.data(), mirror);
            }
            rsx::CudaRasterizer raster;
            raster.init(W, H);
            rsx::RasterBridge bridge;
            bridge.attach(&raster);
            bridge.setVRAM(vram.data(), VRAM_BYTES);
            rsx::RSXState rs;
            rsx::rsx_init(&rs);
            rs.vulkanEmitter = &bridge;

            int nCmds = rsx::rsx_process_fifo(&rs, fifoLE.data(), words,
                                              vram.data(), words);
            std::printf("  RSX replay: %d commands processed\n", nCmds);
            std::printf("  bridge counters: surf=%u clears=%u draws=%u flips=%u\n",
                        bridge.counters.surfaceSetups, bridge.counters.clears,
                        bridge.counters.draws, bridge.counters.flips);
            std::printf("  RSX state: viewport=%dx%d scissor=%dx%d surfWH=%ux%u\n",
                        rs.viewportW, rs.viewportH,
                        rs.scissorW, rs.scissorH,
                        rs.surfaceWidth, rs.surfaceHeight);
            std::printf("             surfA_off=0x%x pitch=%u depth_off=0x%x\n",
                        rs.surfaceOffsetA, rs.surfacePitchA, rs.depthOffset);
            std::printf("             depthTest=%d blendEnable=%d colorTarget=0x%x\n",
                        rs.depthTestEnable, rs.blendEnable, rs.surfaceColorTarget);
            int enabledVAs = 0;
            for (int va = 0; va < 16; ++va)
                if (rs.vertexArrays[va].enabled) enabledVAs++;
            std::printf("             enabled VAs: %d  vpStart=%u fpOffset=0x%x fpCtrl=0x%x\n",
                        enabledVAs, rs.vpStart, rs.fpOffset, rs.fpControl);
            // Texture diagnostics
            for (int t = 0; t < 4; ++t) {
                if (rs.textures[t].enabled)
                    std::printf("             tex[%d]: off=0x%x fmt=0x%x %ux%u baseFmt=0x%02x\n",
                                t, rs.textures[t].offset, rs.textures[t].format,
                                rs.textures[t].width, rs.textures[t].height,
                                ((rs.textures[t].format >> 8) & 0xFF) & 0x9F);
            }
            // Dump first few texels from texture 0
            if (rs.textures[0].enabled && rs.textures[0].offset < VRAM_BYTES) {
                const uint8_t* texBase = vram.data() + rs.textures[0].offset;
                uint8_t baseFmt = ((rs.textures[0].format >> 8) & 0xFF) & 0x9F;
                std::printf("             tex0 first 8 texels (fmt=0x%02x):", baseFmt);
                for (int p = 0; p < 8 && p * 4 + 4 <= (int)(VRAM_BYTES - rs.textures[0].offset); ++p)
                    std::printf(" [%02x%02x%02x%02x]", texBase[p*4], texBase[p*4+1], texBase[p*4+2], texBase[p*4+3]);
                std::printf("\n");
                // Check how much non-zero data is in the texture area
                uint32_t texBytes = rs.textures[0].width * rs.textures[0].height * (baseFmt == 0x85 ? 4 : 1);
                uint32_t nonZero = 0;
                for (uint32_t b = 0; b < texBytes && rs.textures[0].offset + b < VRAM_BYTES; ++b)
                    if (texBase[b]) nonZero++;
                std::printf("             tex0 nonzero bytes: %u / %u\n", nonZero, texBytes);
                // Histogram of first 1000 non-zero texel values
                uint32_t hist[256] = {};
                for (uint32_t b = 0; b < texBytes && rs.textures[0].offset + b < VRAM_BYTES; ++b)
                    hist[texBase[b]]++;
                std::printf("             tex0 value histogram (top 8):");
                std::vector<std::pair<uint32_t,uint32_t>> hv;
                for (int h = 0; h < 256; ++h)
                    if (hist[h]) hv.push_back({h, hist[h]});
                std::sort(hv.begin(), hv.end(), [](auto&a,auto&b){return a.second>b.second;});
                for (size_t h = 0; h < std::min<size_t>(8,hv.size()); ++h)
                    std::printf(" 0x%02x=%u", hv[h].first, hv[h].second);
                std::printf("\n");
            }
            // Check vertex UV range from VA[2] (texture coords)
            if (rs.vertexArrays[2].enabled) {
                auto& va2 = rs.vertexArrays[2];
                uint32_t stride = (va2.format >> 8) & 0xFF;
                uint32_t size = (va2.format >> 4) & 0xF;
                std::printf("             VA[2]: off=0x%x stride=%u size=%u\n",
                            va2.offset, stride, size);
            }
            // Decode full FP program
            if (rs.fpOffset != 0) {
                uint32_t fpOff = rs.fpOffset & 0x0FFFFFFFu;
                if (fpOff < VRAM_BYTES && (VRAM_BYTES - fpOff) >= 64) {
                    const uint8_t* fpBytes = vram.data() + fpOff;
                    std::printf("             FP raw bytes at 0x%x:\n", fpOff);
                    for (int row = 0; row < 5; ++row) {
                        std::printf("               +%02x:", row * 16);
                        for (int b = 0; b < 16; ++b)
                            std::printf(" %02x", fpBytes[row * 16 + b]);
                        std::printf("\n");
                    }
                    // Now decode via our fp_decode
                    const uint32_t* fpData = reinterpret_cast<const uint32_t*>(fpBytes);
                    for (uint32_t fi = 0; fi < 5; ++fi) {
                        rsx::FPDecodedInsn ins = rsx::fp_decode(&fpData[fi * 4]);
                        std::printf("             FP[%u]: op=%u(%s) end=%d mask=%d%d%d%d dst=r%u tex=%u inAttr=%u\n",
                                    fi, ins.opcode, ins.opcode < 50 ? rsx::FP_OP_NAMES[ins.opcode] : "?",
                                    (int)ins.endFlag,
                                    (int)ins.maskX, (int)ins.maskY, (int)ins.maskZ, (int)ins.maskW,
                                    ins.dstReg, ins.texUnit, ins.inputAttr);
                        if (ins.endFlag) break;
                    }
                }
            }
            // VP output diagnostic: run VP for first vertex, dump all outputs
            if (rs.vpData[0] || rs.vpData[1]) {
                // Show all enabled VAs with their slot/format
                std::printf("  Enabled VAs:\n");
                // Dump first vertex raw bytes
                uint32_t minOff = UINT32_MAX;
                for (int va = 0; va < 16; ++va) {
                    if (!rs.vertexArrays[va].enabled) continue;
                    uint32_t off = rs.vertexArrays[va].offset;
                    uint32_t fmtRaw = rs.vertexArrays[va].format;
                    uint32_t stride = (fmtRaw >> 8) & 0xFF;
                    uint32_t sz = (fmtRaw >> 4) & 0xF;
                    uint32_t fmt = fmtRaw & 0xF;
                    std::printf("    VA[%d]: off=0x%x fmt=0x%x stride=%u size=%u type=%u\n",
                                va, off, fmtRaw, stride, sz, fmt);
                    if (off < minOff) minOff = off;
                }
                // Dump first 48 bytes from vertex buffer start
                if (minOff < VRAM_BYTES - 48) {
                    std::printf("    Raw vertex buffer at 0x%x:\n      ", minOff);
                    for (int b = 0; b < 48; ++b) {
                        std::printf("%02x ", vram[minOff + b]);
                        if (b == 23) std::printf("\n      ");
                    }
                    std::printf("\n");
                }
                // Load vertex 0 inputs (with proper BE→LE byte swap)
                rsx::VPFloat4 vpInputs[16] = {};
                for (int va = 0; va < 16; ++va) {
                    if (!rs.vertexArrays[va].enabled) continue;
                    float attr[4] = {0,0,0,1};
                    uint32_t off = rs.vertexArrays[va].offset;
                    uint32_t stride = (rs.vertexArrays[va].format >> 8) & 0xFF;
                    uint32_t sz = (rs.vertexArrays[va].format >> 4) & 0xF;
                    uint32_t fmt = rs.vertexArrays[va].format & 0xF;
                    if (off + stride > VRAM_BYTES || stride == 0) continue;
                    const uint8_t* raw = vram.data() + off;
                    if (fmt == 2) { // FLOAT — byte-swap from BE
                        for (uint32_t c = 0; c < sz && c < 4; ++c) {
                            uint32_t be;
                            std::memcpy(&be, raw + c*4, 4);
                            uint32_t le = __builtin_bswap32(be);
                            std::memcpy(&attr[c], &le, 4);
                        }
                    } else if (fmt == 4) { // UBYTE
                        if (sz == 4) {
                            attr[2] = raw[0] / 255.f;
                            attr[1] = raw[1] / 255.f;
                            attr[0] = raw[2] / 255.f;
                            attr[3] = raw[3] ? (raw[3] / 255.f) : 1.f;
                        } else {
                            for (uint32_t c = 0; c < sz && c < 4; ++c)
                                attr[c] = raw[c] / 255.f;
                        }
                    }
                    for (int c = 0; c < 4; ++c)
                        vpInputs[va].v[c] = attr[c];
                    std::printf("    VP input[%d] = (%f, %f, %f, %f)\n",
                                va, attr[0], attr[1], attr[2], attr[3]);
                }
                // Detailed VP disassembly with source info
                std::printf("  VP disassembly (start=%u):\n", rs.vpStart);
                for (uint32_t ii = rs.vpStart; ii < 512; ++ii) {
                    rsx::VPDecodedInsn ins = rsx::vp_decode(&rs.vpData[ii * 4]);
                    const char* VecN[] = {"NOP","MOV","MUL","ADD","MAD","DP3","DPH","DP4","DST","MIN","MAX","SLT","SGE","ARL","FRC","FLR"};
                    const char* ScaN[] = {"NOP","MOV","RCP","RCC","RSQ","EXP","LOG","LIT"};
                    if (ins.vecOp != 0 && ins.vecOp < 16)
                        std::printf("    [%u] VEC:%s in[%u] c[%u]",
                                    ii, VecN[ins.vecOp], ins.inputIdx, ins.constIdx);
                    if (ins.scaOp != 0 && ins.scaOp < 8)
                        std::printf("    [%u] SCA:%s", ii, ScaN[ins.scaOp]);
                    if (ins.vecOp == 0 && ins.scaOp == 0)
                        std::printf("    [%u] NOP", ii);
                    // Show sources
                    for (int s = 0; s < 3; ++s) {
                        const char* types[] = {"?","t","in","c"};
                        if (ins.src[s].regType != 0)
                            std::printf(" src%d=%s%u", s, types[ins.src[s].regType], ins.src[s].regIdx);
                    }
                    // Show dest
                    if (ins.vecWriteResult)
                        std::printf(" → o%u.%s%s%s%s", ins.vecDstOut,
                                    ins.vecMaskX?"x":"", ins.vecMaskY?"y":"",
                                    ins.vecMaskZ?"z":"", ins.vecMaskW?"w":"");
                    else if (ins.vecDstTmp != 0x3F)
                        std::printf(" → t%u.%s%s%s%s", ins.vecDstTmp,
                                    ins.vecMaskX?"x":"", ins.vecMaskY?"y":"",
                                    ins.vecMaskZ?"z":"", ins.vecMaskW?"w":"");
                    if (ins.endFlag) std::printf(" [END]");
                    std::printf("\n");
                    if (ins.endFlag) break;
                }
                // Execute VP and show outputs
                const rsx::VPFloat4* cnst = reinterpret_cast<const rsx::VPFloat4*>(rs.vpConstants);
                rsx::VPFloat4 vpOuts[16] = {};
                rsx::vp_execute(rs.vpData, 512u*4u, rs.vpStart, vpInputs, cnst, vpOuts);
                std::printf("  VP output registers (vertex 0):\n");
                for (int o = 0; o < 16; ++o) {
                    float *v = vpOuts[o].v;
                    if (v[0] != 0 || v[1] != 0 || v[2] != 0 || v[3] != 0)
                        std::printf("    o[%2d] = (%f, %f, %f, %f)\n", o, v[0], v[1], v[2], v[3]);
                }
            }

            std::printf("  raster stats: tris=%u skipped=%u clears=%u\n",
                        raster.stats.triangles, raster.stats.triangleSkipped, raster.stats.clears);

            // Framebuffer readback
            std::vector<uint32_t> fb(W * H, 0);
            raster.readbackPlane(0, fb.data());
            uint32_t nonClear = 0;
            uint32_t bgColor = fb[0];
            for (uint32_t k = 0; k < W * H; ++k)
                if (fb[k] != bgColor) nonClear++;
            std::printf("  framebuffer: %u painted (bg=0x%08x)\n",
                        nonClear, bgColor);
            std::unordered_map<uint32_t, uint32_t> uniq;
            for (uint32_t k = 0; k < W * H; ++k)
                if (fb[k] != bgColor) uniq[fb[k]]++;
            std::vector<std::pair<uint32_t,uint32_t>> ur(uniq.begin(), uniq.end());
            std::sort(ur.begin(), ur.end(),
                      [](auto&a, auto&b){ return a.second > b.second; });
            std::printf("    distinct painted colors: %zu; top:\n", ur.size());
            for (size_t ii = 0; ii < std::min<size_t>(8, ur.size()); ++ii)
                std::printf("      0x%08x  x%u\n", ur[ii].first, ur[ii].second);

            CHECK(nonClear > 10000, "cube renders visible pixels");

            rsx::rsx_shutdown(&rs);
        } else {
            std::printf("  GCM FIFO: current==begin (no commands submitted yet)\n");
        }
    } else if (gcmInitRan) {
        std::printf("  GCM FIFO: cellGcmInitBody ran but ioBase=0\n");
    } else {
        std::printf("  GCM FIFO: cellGcmInitBody never called\n");
    }

    CHECK(hleHits >= 1, "at least one HLE dispatch");
    CHECK(resolved == totalFuncs, "all imports resolved");

    megakernel_shutdown();
    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
