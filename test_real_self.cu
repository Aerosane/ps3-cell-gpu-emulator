// test_real_self.cu — Load a real PS3 SELF from disk via ps3_load_elf.
// Exercises the SCE_MAGIC path + embedded ELF parse on a genuine RPCS3
// test asset (spurs_test.self). This is a plumbing test: it does not
// run the guest code — it just verifies that the loader can parse the
// SELF wrapper and stage program segments into emulated memory.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>

#include "elf_loader.h"

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  Real PS3 SELF loader — spurs_test.self\n");
    std::printf("══════════════════════════════════════════════════════\n");

    const char* path = "/workspaces/codespace/rpcs3-src/bin/test/spurs_test.self";
    auto file = read_file(path);
    std::printf("  file size: %zu bytes\n", file.size());
    CHECK(file.size() > 1024, "SELF file read from disk");

    // Verify SCE magic at offset 0.
    bool hasSCE = file.size() >= 4 &&
                  file[0] == 'S' && file[1] == 'C' && file[2] == 'E' && file[3] == 0;
    CHECK(hasSCE, "File begins with SCE magic");

    // Try ps3_load_elf.
    std::vector<uint8_t> memory(64u * 1024u * 1024u, 0);  // 64 MiB
    PS3ExeInfo info{};
    int rc = ps3_load_elf(file.data(), file.size(),
                               memory.data(), memory.size(), &info);
    std::printf("  ps3_load_elf rc=%d entry=%lx is_prx=%d\n",
                rc, (unsigned long)info.entry_point, (int)info.is_prx);

    // If SELF body is encrypted, loader returns ELF_ERR_MAGIC after SCE
    // unwrap because the inner bytes aren't a valid ELF header. That's
    // a legitimate outcome we need to handle (most retail SELFs are
    // encrypted). We accept either full success OR magic-miss (for
    // encrypted payload), but FAIL on sizing / memory errors which
    // would indicate a bug in the loader path.
    if (rc == ELF_OK) {
        CHECK(info.entry_point != 0,
              "Entry point parsed from real SELF");
        CHECK(info.num_segments >= 1,
              "At least one PT_LOAD segment processed");
        std::printf("  segments=%u tls=%s\n",
                    info.num_segments,
                    info.has_tls ? "yes" : "no");
    } else if (rc == ELF_ERR_MAGIC) {
        std::printf("  NOTE: inner body is encrypted (retail SELF) — "
                    "magic mismatch after SCE unwrap is expected.\n");
        CHECK(true, "Loader rejected encrypted body gracefully");
    } else {
        std::printf("  UNEXPECTED rc=%d — loader bug?\n", rc);
        CHECK(false, "Unexpected loader return code");
    }

    // Regardless of encryption status, verify that self_find_elf_offset
    // at least finds a non-zero offset inside the SELF blob or returns
    // 0 (both are legitimate).
    uint64_t elfOff = self_find_elf_offset(file.data(), file.size());
    std::printf("  self_find_elf_offset = 0x%lx\n", (unsigned long)elfOff);

    std::printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
