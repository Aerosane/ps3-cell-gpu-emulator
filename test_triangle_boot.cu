// test_triangle_boot.cu — Load RPCS3's gs_gcm_basic_triangle.elf,
// resolve the entry descriptor, boot the PPC megakernel, and report
// where execution first lands inside an HLE import stub.
//
// This is the integration smoke-test that wires together:
//   * real-ELF loader
//   * section-header-based import discovery
//   * FNID→module::name resolution
//   * PPC megakernel
// into the first end-to-end "execute a PS3 sample" pipeline.

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

static uint32_t be32(const uint8_t* p) {
    return (uint32_t)p[0]<<24 | (uint32_t)p[1]<<16 | (uint32_t)p[2]<<8 | p[3];
}

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  PS3 triangle sample boot attempt\n");
    std::printf("══════════════════════════════════════════════════════\n");

    const char* path = "/workspaces/codespace/rpcs3-src/bin/test/gs_gcm_basic_triangle.elf";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CHECK((bool)f, "open triangle ELF");
    auto sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> file(sz);
    f.read(reinterpret_cast<char*>(file.data()), sz);

    std::vector<uint8_t> mem(128u * 1024u * 1024u, 0);
    PS3ExeInfo info{};
    int rc = ps3_load_elf(file.data(), file.size(),
                          mem.data(), mem.size(), &info);
    CHECK(rc == ELF_OK, "ps3_load_elf");
    std::printf("  e_entry=0x%lx  segments=%d\n",
                (unsigned long)info.entry_point, info.num_segments);

    // ── Resolve 32-bit function descriptor at e_entry ──
    uint32_t descEntry = be32(mem.data() + info.entry_point);
    uint32_t descTOC   = be32(mem.data() + info.entry_point + 4);
    std::printf("  descriptor[0x%lx]: entry=0x%x toc=0x%x\n",
                (unsigned long)info.entry_point, descEntry, descTOC);
    CHECK(descEntry >= 0x10000 && descEntry < 0x30000, "entry in code seg");

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
            // stub_tbl points into .data.sceFStub; each u32 there is
            // the real trampoline PC in .sceStub.text.
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
    // Dump first few stubMap PCs
    std::printf("  stubMap sample: ");
    int dc = 0;
    for (auto& kv : stubMap) { if (dc++ < 6) std::printf("0x%x ", kv.first); }
    std::printf("\n");

    // ── Boot ──
    CHECK(megakernel_init() != 0, "megakernel_init");
    // Copy all of the loaded image (up through data segment end) into
    // the kernel sandbox.
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
    // DEBUG: verify critical TOC entry present in host mem and GPU mem
    {
        uint32_t hostVal = be32(mem.data() + 0x40fd8);
        uint8_t gpubuf[4] = {};
        megakernel_read_mem(0x40fd8, gpubuf, 4);
        uint32_t gpuVal = ((uint32_t)gpubuf[0]<<24)|((uint32_t)gpubuf[1]<<16)|((uint32_t)gpubuf[2]<<8)|gpubuf[3];
        std::printf("  DBG mem[0x40fd8]: host=0x%x  gpu=0x%x  (should match)\n", hostVal, gpuVal);
    }

    megakernel_set_entry(descEntry, 0x00F00000ULL, descTOC);
    megakernel_set_code_range(0x10000, 0x30330);

    uint32_t haltedInStub = 0;
    (void)haltedInStub;
    // Build dispatcher from stubMap.
    PpuHleDispatcher disp;
    for (auto& kv : stubMap) {
        disp.add(kv.first, kv.second.fnid,
                 kv.second.mod, kv.second.name);
    }
    // HLE-patch the ELF's internal libc allocator — its heap descriptor
    // never gets initialised because we don't run cellGcmInitBody, so
    // calls return NULL and downstream code (memset, strcpy) spins
    // forever. Short-circuit to a bump arena.
    disp.addBuiltinMalloc(0x1fed4);
    disp.addBuiltinMalloc(0x1ff10);
    disp.addBuiltinMalloc(0x1ff44);
    disp.addBuiltinMalloc(0x1ff70);  // realloc-style, still returns a ptr
    disp.addBuiltinMalloc(0x1ffa0);

    // Single-step dispatch — reliable capture of trampoline entries.
    int hleHits = 0, steps = 0, maxSteps = 2000000;
    bool stopped = false;
    int stallCount = 0;
    uint32_t stallPc = 0;
    uint32_t lastReportedPc = 0;
    int traceCount = 0;
    const int traceMax = 0;
    // PC histogram — coarse 64-byte bucket sampling every 16 steps so we
    // can see where the CRT spins without paying per-step map cost.
    std::unordered_map<uint32_t, uint64_t> pcBuckets;
    // Branch-and-link target tracer: when LR changes between steps, the
    // new LR holds the return address (PC+4 at the bl) and the *new* PC
    // is the callee. This gives us a call-graph histogram without
    // instrumenting the interpreter.
    std::unordered_map<uint32_t, uint64_t> blTargets;
    std::unordered_map<uint32_t, uint64_t> bogusCallSites;
    // Track callers (LR-4) of the 0x104xx hot region to find external loop driver.
    std::unordered_map<uint32_t, uint64_t> hot104Callers;
    uint64_t prevLR = 0;
    for (steps = 0; steps < maxSteps && !stopped; ++steps) {
        megakernel_run(1);
        ppc::PPEState st{};
        megakernel_read_state(&st);
        uint32_t pc = (uint32_t)st.pc;
        // Trace entry/exit boundaries of the looping CRT function
        if ((pc == 0x13598 || pc == 0x1358c || pc == 0x13588 || pc == 0x123d0 || pc == 0x12410) && traceCount < traceMax) {
            std::printf("  trace#%d step=%d PC=0x%x r3=0x%llx r4=0x%llx r5=0x%llx r30=0x%llx r31=0x%llx\n",
                        traceCount, steps, pc,
                        (unsigned long long)st.gpr[3],
                        (unsigned long long)st.gpr[4],
                        (unsigned long long)st.gpr[5],
                        (unsigned long long)st.gpr[30],
                        (unsigned long long)st.gpr[31]);
            traceCount++;
        }
        if ((steps & 0xF) == 0) {
            pcBuckets[pc & ~0x3Fu]++;
        }
        // bl detector: LR changed → previous instruction was bl/blrl/bctrl;
        // current PC is the callee entry.
        if (st.lr != prevLR) {
            blTargets[pc]++;
            // If we jumped to a bogus (out-of-code-range) PC, the callsite
            // is LR-4 on PPC64 (bl returns to insn after bl; LR holds that).
            if (pc >= 0x30330 || pc < 0x10000) {
                uint32_t site = (uint32_t)(st.lr - 4);
                bogusCallSites[site]++;
            }
            // Track callers (LR-4) of hot 0x104xx region — whoever is looping us.
            if (pc >= 0x10400 && pc < 0x10600) {
                uint32_t caller = (uint32_t)(st.lr - 4);
                hot104Callers[caller]++;
            }
            prevLR = st.lr;
        }
        if ((steps % 200000) == 0 && pc != lastReportedPc) {
            std::printf("  progress: step=%d PC=0x%x LR=0x%llx\n",
                        steps, pc, (unsigned long long)st.lr);
            lastReportedPc = pc;
        }
        if (pc == stallPc) {
            if (++stallCount > 32768) {
                std::printf("  stalled at PC=0x%x for %d steps; stopping.\n",
                            pc, stallCount);
                break;
            }
        } else {
            stallPc = pc; stallCount = 0;
        }
        if (disp.byPc.count(pc) || disp.builtinByPc.count(pc)) {
            bool haltReq = false;
            const char* nm = disp.dispatch(st, mem.data(), mem.size(),
                                           haltReq);
            megakernel_write_state(&st);
            hleHits++;
            if (hleHits <= 40 || haltReq) {
                const char* mod = disp.byPc.count(pc) ? disp.byPc[pc].mod.c_str() : "libc-hle";
                std::printf("  HLE #%d step=%d PC=0x%x  %s::%s  LR→0x%llx\n",
                            hleHits, steps, pc,
                            mod,
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
    {
        uint64_t pt=0, at=0; uint32_t pc_c=0, ac=0;
        megakernel_read_poll_state(&pt, &pc_c, &at, &ac);
        std::printf("  poll detector: primary=0x%llx (hits=%u), alt=0x%llx (hits=%u)\n",
                    (unsigned long long)pt, pc_c,
                    (unsigned long long)at, ac);
        megakernel_dump_poll_state();
    }
    {
        ppc::PPEState fs{};
        megakernel_read_state(&fs);
        std::printf("  final PC=0x%x  LR=0x%llx  r1=0x%llx  r3=0x%llx\n",
                    (uint32_t)fs.pc,
                    (unsigned long long)fs.lr,
                    (unsigned long long)fs.gpr[1],
                    (unsigned long long)fs.gpr[3]);
    }
    {
        uint32_t log[257] = {};
        int n = megakernel_read_hle_log(log, 256);
        if (n > 0) {
            std::printf("  unknown syscall log (%d entries):", n);
            int seen = 0;
            std::unordered_map<uint32_t, int> opCounts;
            for (int i = 0; i < n && seen < 16; ++i) {
                std::printf(" 0x%x", log[1 + i]);
                seen++;
            }
            std::printf("\n");
            for (int i = 0; i < n; ++i) {
                uint32_t tag = log[1 + i];
                if ((tag & 0xFF000000u) == 0xDE000000u) {
                    uint32_t key = tag & 0xFFFFFu;  // OPCD<<10 | XO
                    opCounts[key]++;
                }
            }
            if (!opCounts.empty()) {
                std::vector<std::pair<uint32_t,int>> v(opCounts.begin(), opCounts.end());
                std::sort(v.begin(), v.end(),
                          [](auto& a, auto& b){ return a.second > b.second; });
                std::printf("  unimpl opcode histogram (OPCD/XO → count):\n");
                int k = 0;
                for (auto& p : v) {
                    uint32_t opcd = (p.first >> 10) & 0x3F;
                    uint32_t xo   =  p.first        & 0x3FF;
                    std::printf("    OPCD=%u XO=%u  (%d)\n", opcd, xo, p.second);
                    if (++k >= 12) break;
                }
            }
        } else {
            std::printf("  no unknown syscalls logged\n");
        }
    }
    disp.print_summary();
    // Top-20 hot PC buckets (64-byte aligned) from 1-in-16 sampling.
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
    // Top-10 bl targets — these are the most-called functions regardless
    // of how long each call takes, complementing the PC bucket view.
    {
        std::vector<std::pair<uint32_t, uint64_t>> sorted(blTargets.begin(), blTargets.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::printf("  top bl targets (function entry → call count):\n");
        int n = 0;
        for (const auto& p : sorted) {
            std::printf("    0x%08x  %llu\n", p.first, (unsigned long long)p.second);
            if (++n >= 10) break;
        }
    }
    // Bogus-call sites: addresses that issued a bl to an out-of-range
    // target. These are the callers whose function pointer is broken.
    if (!bogusCallSites.empty()) {
        std::vector<std::pair<uint32_t, uint64_t>> sorted(
            bogusCallSites.begin(), bogusCallSites.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::printf("  bogus-call sites (LR-4 of rescued bl/bctrl):\n");
        int n = 0;
        for (const auto& p : sorted) {
            std::printf("    0x%08x  %llu\n", p.first, (unsigned long long)p.second);
            if (++n >= 10) break;
        }
    }
    if (!hot104Callers.empty()) {
        std::vector<std::pair<uint32_t, uint64_t>> sorted(
            hot104Callers.begin(), hot104Callers.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::printf("  callers of 0x104xx region (LR-4 at entry):\n");
        int n = 0;
        for (const auto& p : sorted) {
            std::printf("    0x%08x  %llu\n", p.first, (unsigned long long)p.second);
            if (++n >= 10) break;
        }
    }
    // FIFO drain diagnostic — if cellGcmInitBody ran, we have a real
    // CellGcmContextData in guest memory. Scan the region between
    // `begin` and `current` looking for NV method headers as a proxy
    // for "did the game actually submit RSX commands?"
    if (disp.gcmIoBase) {
        std::printf("  GCM FIFO: io=[0x%x..0x%x) cmdSize=0x%x\n",
                    disp.gcmIoBase, disp.gcmIoBase + disp.gcmIoSize,
                    disp.gcmCmdSize);
        // Read current cellGcmContextData.current from GPU mem.
        uint8_t ctxBytes[16] = {};
        megakernel_read_mem(disp.gcmContextAddr, ctxBytes, 16);
        auto rd = [&](int off) {
            return (uint32_t)ctxBytes[off]<<24 | (uint32_t)ctxBytes[off+1]<<16
                 | (uint32_t)ctxBytes[off+2]<<8 | ctxBytes[off+3];
        };
        uint32_t ctxBegin = rd(0), ctxEnd = rd(4), ctxCur = rd(8), ctxCb = rd(12);
        std::printf("  GCM ctx: begin=0x%x end=0x%x current=0x%x cb=0x%x\n",
                    ctxBegin, ctxEnd, ctxCur, ctxCb);
        if (ctxCur > ctxBegin && ctxCur - ctxBegin < 0x100000) {
            uint32_t bytes = ctxCur - ctxBegin;
            std::vector<uint8_t> fifoBE(bytes);
            megakernel_read_mem(ctxBegin, fifoBE.data(), bytes);
            // Scan as big-endian u32 and count non-zero headers.
            uint32_t words = bytes / 4;
            uint32_t nonzero = 0, methods = 0;
            for (uint32_t i = 0; i < words; ++i) {
                uint32_t w = (uint32_t)fifoBE[i*4]<<24
                           | (uint32_t)fifoBE[i*4+1]<<16
                           | (uint32_t)fifoBE[i*4+2]<<8
                           | fifoBE[i*4+3];
                if (w) nonzero++;
                // NV method header: bits [15:2] = method offset, bits [29:18] = count.
                uint32_t count = (w >> 18) & 0x7FF;
                uint32_t method = w & 0xFFFC;
                if (count > 0 && count < 32 && method > 0 && method < 0x2000) methods++;
            }
            std::printf("  GCM FIFO bytes written: %u  (%u nonzero words, ~%u plausible method headers)\n",
                        bytes, nonzero, methods);
        } else {
            std::printf("  GCM FIFO: current == begin (no commands submitted yet)\n");
        }
    } else {
        std::printf("  GCM FIFO: cellGcmInitBody never called\n");
    }
    CHECK(hleHits >= 1, "at least one HLE dispatch happened");

    if (haltedInStub) {
        std::printf("  ⇒ First HLE call attempted: %s::%s\n",
                    stubMap[haltedInStub].mod,
                    stubMap[haltedInStub].name);
    }

    CHECK(totalFuncs == resolved, "all imports resolved");

    megakernel_shutdown();
    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
