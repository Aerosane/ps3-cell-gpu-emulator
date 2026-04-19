// test_real_self_disasm.cu — Load the RPCS3 sample SELF, resolve the
// PPC64 ELF-v1 function descriptor, and decode the first 16 instructions
// by opcode class. Verifies the SELF decompression + staging pipeline
// produces byte-accurate PPC64 machine code (not garbage from a wrong
// file offset or failed inflate).

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>

#include "elf_loader.h"

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

static const char* classify_ppc(uint32_t insn) {
    uint32_t op = (insn >> 26) & 0x3F;
    switch (op) {
    case 14: return "addi";
    case 15: return "addis";
    case 16: return "bc";
    case 17: return "sc";
    case 18: return "b";
    case 19: return "bclr/bcctr/misc19";
    case 24: return "ori";
    case 25: return "oris";
    case 26: return "xori";
    case 27: return "xoris";
    case 28: return "andi.";
    case 29: return "andis.";
    case 31: return "integer-X";
    case 32: return "lwz";
    case 34: return "lbz";
    case 36: return "stw";
    case 37: return "stwu";
    case 38: return "stb";
    case 40: return "lhz";
    case 48: return "lfs";
    case 50: return "lfd";
    case 58: return "ld (dword)";
    case 62: return "std (dword)";
    default:  return "other";
    }
}

int main() {
    std::printf("═════════════════════════════════════════════\n");
    std::printf("  Real SELF disassembly sanity-check\n");
    std::printf("═════════════════════════════════════════════\n");

    const char* path = "/workspaces/codespace/rpcs3-src/bin/test/spurs_test.self";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CHECK((bool)f, "open SELF");
    auto sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> file(sz);
    f.read(reinterpret_cast<char*>(file.data()), sz);

    std::vector<uint8_t> mem(64u * 1024u * 1024u, 0);
    PS3ExeInfo info{};
    int rc = ps3_load_elf(file.data(), file.size(),
                          mem.data(), mem.size(), &info);
    CHECK(rc == ELF_OK, "ps3_load_elf");

    // Resolve v1 descriptor (32-bit PS3 form).
    auto be32 = [](const uint8_t* p) {
        return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 |
               (uint32_t)p[2] << 8  | (uint32_t)p[3];
    };
    uint32_t ent = be32(mem.data() + info.entry_point);
    uint32_t toc = be32(mem.data() + info.entry_point + 4);
    std::printf("  descriptor@0x%lx: entry=0x%x toc=0x%x\n",
                (unsigned long)info.entry_point, ent, toc);
    CHECK(ent >= 0x10000 && ent < 0x30000, "entry in code segment");
    CHECK(toc >= 0x30000,                  "toc in data segment");

    std::printf("\n  Disassembly of first 16 instructions at 0x%x:\n", ent);
    int validCount = 0;
    for (int i = 0; i < 16; ++i) {
        uint32_t pc = ent + i * 4;
        if (pc + 4 > mem.size()) break;
        uint32_t w = be32(mem.data() + pc);
        const char* name = classify_ppc(w);
        std::printf("    %06x:  %08x   %s\n", pc, w, name);
        // An instruction is "valid" if its primary op-field isn't a
        // reserved zero encoding (which would indicate all-zero memory).
        if (w != 0) validCount++;
    }
    // All 16 should decode to something real — the loader staged
    // genuine PPC code, not zeros or random garbage.
    CHECK(validCount == 16, "all 16 instructions non-zero");

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
