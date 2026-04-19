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

#include "elf_loader.h"
#include "ppu_hle_names.h"
#include "ppu_hle_dispatch.h"
#include "ppc_defs.h"

extern "C" {
    int   megakernel_init();
    int   megakernel_load(uint64_t, const void*, size_t);
    int   megakernel_set_entry(uint64_t, uint64_t, uint64_t);
    float megakernel_run(uint32_t);
    int   megakernel_read_state(ppc::PPEState*);
    int   megakernel_write_state(const ppc::PPEState*);
    int   megakernel_read_mem(uint64_t, void*, size_t);
    int   megakernel_write_mem(uint64_t, const void*, size_t);
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

    megakernel_set_entry(descEntry, 0x00F00000ULL, descTOC);

    uint32_t haltedInStub = 0;
    (void)haltedInStub;
    // Build dispatcher from stubMap.
    PpuHleDispatcher disp;
    for (auto& kv : stubMap) {
        disp.add(kv.first, kv.second.fnid,
                 kv.second.mod, kv.second.name);
    }

    // Single-step and dispatch HLE trampolines as we encounter them.
    int hleHits = 0, steps = 0, maxSteps = 100000;
    bool stopped = false;
    for (steps = 0; steps < maxSteps && !stopped; ++steps) {
        megakernel_run(1);
        ppc::PPEState st{};
        megakernel_read_state(&st);
        uint32_t pc = (uint32_t)st.pc;
        if (disp.byPc.count(pc)) {
            bool haltReq = false;
            const char* nm = disp.dispatch(st, mem.data(), mem.size(),
                                           haltReq);
            megakernel_write_state(&st);
            hleHits++;
            if (hleHits <= 12 || haltReq) {
                std::printf("  HLE #%d step=%d  %s::%s  r3→0x%llx  LR→0x%llx\n",
                            hleHits, steps,
                            disp.byPc[pc].mod.c_str(),
                            nm ? nm : "?",
                            (unsigned long long)st.gpr[3],
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
    std::printf("  total steps=%d  HLE calls dispatched=%d\n",
                steps, hleHits);
    {
        ppc::PPEState fs{};
        megakernel_read_state(&fs);
        std::printf("  final PC=0x%x  LR=0x%llx  r1=0x%llx  r3=0x%llx\n",
                    (uint32_t)fs.pc,
                    (unsigned long long)fs.lr,
                    (unsigned long long)fs.gpr[1],
                    (unsigned long long)fs.gpr[3]);
    }
    disp.print_summary();
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
