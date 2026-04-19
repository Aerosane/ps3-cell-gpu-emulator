// test_ppu_hle_resolve.cu — Load gs_gcm_basic_triangle.elf, walk its
// .lib.stub table, and resolve every imported FNID to a real PS3
// function name via the baked-in HLE name table. This proves we have
// the full HLE dispatch keyspace for this sample.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>

#include "elf_loader.h"
#include "ppu_hle_names.h"

static uint32_t be32(const uint8_t* p) {
    return (uint32_t)p[0]<<24 | (uint32_t)p[1]<<16 | (uint32_t)p[2]<<8 | p[3];
}

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  PPU HLE FNID → function-name resolution\n");
    std::printf("══════════════════════════════════════════════════════\n");

    const char* path = "/workspaces/codespace/rpcs3-src/bin/test/gs_gcm_basic_triangle.elf";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CHECK((bool)f, "open test ELF");
    auto sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> file(sz);
    f.read(reinterpret_cast<char*>(file.data()), sz);

    std::vector<uint8_t> mem(64u * 1024u * 1024u, 0);
    PS3ExeInfo info{};
    int rc = ps3_load_elf(file.data(), file.size(),
                          mem.data(), mem.size(), &info);
    CHECK(rc == ELF_OK, "ps3_load_elf");

    // Locate .lib.stub via section headers.
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)file.data();
    uint64_t shoff = elf_bswap64(eh->e_shoff);
    uint16_t shnum = elf_bswap16(eh->e_shnum);
    uint16_t shentsz = elf_bswap16(eh->e_shentsize);
    uint16_t shstrndx = elf_bswap16(eh->e_shstrndx);
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

    const Elf64_Shdr* shStub = findSec(".lib.stub");
    CHECK(shStub != nullptr, "found .lib.stub section");
    if (!shStub) return 1;

    uint64_t stubAddr = elf_bswap64(shStub->sh_addr);
    uint64_t stubSize = elf_bswap64(shStub->sh_size);

    int totalFuncs = 0, resolved = 0;
    for (uint64_t a = stubAddr; a + 0x2C <= stubAddr + stubSize; a += 0x2C) {
        const uint8_t* s = mem.data() + a;
        uint16_t num_func    = (uint16_t)((s[6] << 8) | s[7]);
        uint32_t mod_name_a  = be32(s + 0x10);
        uint32_t func_fnid_a = be32(s + 0x14);
        uint32_t func_stub_a = be32(s + 0x18);

        const char* mod_name = (mod_name_a && mod_name_a + 32 <= mem.size())
            ? (const char*)(mem.data() + mod_name_a) : "(?)";

        std::printf("\n  %s  (%u funcs, stubs @ 0x%x):\n",
                    mod_name, num_func, func_stub_a);

        for (uint16_t k = 0; k < num_func; ++k) {
            uint32_t fnid = be32(mem.data() + func_fnid_a + k * 4);
            const PpuHleEntry* e = ppu_hle_lookup(fnid);
            uint32_t stubPc = func_stub_a + k * 0x20;  // 8 insns per stub
            if (e) {
                std::printf("    0x%08x  pc=0x%x  %s::%s\n",
                            fnid, stubPc, e->module, e->name);
                resolved++;
            } else {
                std::printf("    0x%08x  pc=0x%x  <unresolved>\n",
                            fnid, stubPc);
            }
            totalFuncs++;
        }
    }

    std::printf("\n  Summary: %d / %d imports resolved to HLE names\n",
                resolved, totalFuncs);
    CHECK(resolved == totalFuncs, "every import resolved");

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
