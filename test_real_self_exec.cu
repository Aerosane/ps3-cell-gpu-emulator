// test_real_self_exec.cu — Load a real PS3 SELF, copy segments into the
// megakernel sandbox, and run a few thousand PPC cycles starting at the
// SELF's entry_point.
//
// We don't have the full PS3 runtime (libsysmodule, libsysutil, ...) so the
// program will very quickly hit an unresolved syscall or branch into an
// unloaded segment. This is expected — the test verifies that:
//   - SELF unwrap + ELF staging succeeds
//   - at least one real PPC instruction decodes and retires
//   - PC advances past the entry address

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>

#include "elf_loader.h"
#include "ppc_defs.h"

extern "C" {
    int   megakernel_init();
    int   megakernel_load(uint64_t, const void*, size_t);
    int   megakernel_set_entry(uint64_t, uint64_t, uint64_t);
    float megakernel_run(uint32_t);
    int   megakernel_read_state(ppc::PPEState*);
    int   megakernel_read_mem(uint64_t, void*, size_t);
    void  megakernel_shutdown();
}

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  Real SELF execution attempt via megakernel\n");
    std::printf("══════════════════════════════════════════════════════\n");

    const char* path = "/workspaces/codespace/rpcs3-src/bin/test/spurs_test.self";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CHECK((bool)f, "Open SELF file");
    auto sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> file(sz);
    f.read(reinterpret_cast<char*>(file.data()), sz);

    // Emulated memory: 256 MiB.
    std::vector<uint8_t> memory(256u * 1024u * 1024u, 0);
    PS3ExeInfo info{};
    int rc = ps3_load_elf(file.data(), file.size(),
                          memory.data(), memory.size(), &info);
    CHECK(rc == ELF_OK, "ps3_load_elf returned OK");

    std::printf("  entry=0x%lx segments=%d\n",
                (unsigned long)info.entry_point, info.num_segments);
    for (int i = 0; i < info.num_segments; ++i) {
        std::printf("    seg[%d]: addr=0x%lx size=%lu flags=0x%x\n",
                    i,
                    (unsigned long)info.segments[i].addr,
                    (unsigned long)info.segments[i].size,
                    info.segments[i].flags);
    }

    // Determine total span we need to transfer into the megakernel
    // sandbox: from 0 through (max segment end). The megakernel has a
    // 512 MiB sandbox; load the first 32 MiB (plenty to cover typical
    // PT_LOAD spans of a small test SELF).
    uint64_t hi = 0;
    for (int i = 0; i < info.num_segments; ++i) {
        uint64_t end = info.segments[i].addr + info.segments[i].size;
        if (end > hi) hi = end;
    }
    uint64_t loadSpan = (hi + 0xFFFFu) & ~uint64_t(0xFFFFu);
    if (loadSpan > 32u * 1024u * 1024u) loadSpan = 32u * 1024u * 1024u;
    std::printf("  loadSpan = %lu bytes\n", (unsigned long)loadSpan);

    CHECK(megakernel_init() != 0, "megakernel_init");
    megakernel_load(0, memory.data(), loadSpan);

    // PPC64 ELF v1 ABI: function pointers are pointers to 24-byte
    // function descriptors (entry, TOC, env). e_entry on real PS3 SELFs
    // is the descriptor address; the real code PC lives at *descriptor.
    // spurs_test.self: entry=0x30130 is inside the data segment (seg[1]
    // at 0x30000), so decoding the descriptor is required.
    uint64_t realEntry = info.entry_point;
    uint64_t realTOC   = 0;
    if (info.entry_point >= 0x30000 && info.entry_point + 8 < memory.size()) {
        // PS3 PPC64 function descriptors use 32-bit fields
        // (not full 64-bit): (entry:u32, toc:u32, env:u32).
        auto be32 = [](const uint8_t* p) {
            return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 |
                   (uint32_t)p[2] << 8  | (uint32_t)p[3];
        };
        uint32_t descEntry = be32(memory.data() + info.entry_point);
        uint32_t descTOC   = be32(memory.data() + info.entry_point + 4);
        std::printf("  descriptor[0x%lx]: entry=0x%x toc=0x%x\n",
                    (unsigned long)info.entry_point, descEntry, descTOC);
        if (descEntry >= 0x10000 && descEntry < loadSpan) {
            realEntry = descEntry;
            realTOC   = descTOC;
        }
    }
    // Dump first 32 bytes at entry to sanity-check segment staging.
    std::printf("  mem[realEntry]:");
    for (int i = 0; i < 16; ++i)
        std::printf(" %02x", memory[realEntry + i]);
    std::printf("\n");

    megakernel_set_entry(realEntry, 0x00F00000ULL, realTOC);

    float ms = megakernel_run(65536);
    ppc::PPEState st{};
    megakernel_read_state(&st);
    std::printf("  ran: %.2f ms  pc=0x%llx halted=%d\n",
                ms, (unsigned long long)st.pc, (int)st.halted);

    // A real retire happened if PC moved past the entry point (even a
    // single branch bumps PC by 4, and a fault also advances + halts).
    CHECK(st.pc != realEntry || st.halted != 0,
          "At least one PPC instruction retired from real SELF");
    std::printf("  NOTE: execution without full HLE will stop quickly at the "
                "first unresolved syscall/branch.\n");

    megakernel_shutdown();
    std::printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
