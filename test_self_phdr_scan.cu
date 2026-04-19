// test_self_phdr_scan.cu — Dump every PHDR of a loaded SELF (including
// PS3-specific PT_SCE_* entries) and locate the sys_proc_prx_param /
// sys_prx_module_info blocks that list library imports + exports.
// Future HLE dispatch will hook functions listed here.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>

#include "elf_loader.h"

static uint32_t be32(const uint8_t* p) {
    return (uint32_t)p[0]<<24 | (uint32_t)p[1]<<16 | (uint32_t)p[2]<<8 | p[3];
}

// sys_proc_prx_param at PT_SCE_PROCPARAM + some offset.
// Layout per RPCS3 sys/sys_process.h:
//   u32 size, magic, version, ...
//   u32 libent_start, libent_end;
//   u32 libstub_start, libstub_end;
//   u32 ver, pad;
struct PpuPrxParam {
    uint32_t size;
    uint32_t magic;            // 0x1B434FEC
    uint32_t version;
    uint32_t pad0;
    uint32_t libent_start;
    uint32_t libent_end;
    uint32_t libstub_start;
    uint32_t libstub_end;
    uint32_t ver;
    uint32_t pad1;
};

// sys_prx_module_info (16 bytes) at libent_start:
//   u16 attrs; u8 minor, major; char name[28]; u32 toc; u32 exports[3];
// (variable trailing layout).

// sys_prx_library_info (stub descriptor) at libstub_start, 0x2C bytes
// stride per entry per RPCS3 pkg/sys_prx.h:
//   u8 struct_size; u8 unused; u16 version; u16 attributes;
//   u16 num_func; u16 num_var; u16 num_tls;
//   u8 hash_info, tls_hash_info, unused2[2];
//   u32 module_name_addr; u32 func_fnid_addr;
//   u32 func_stub_addr; u32 var_fnid_addr; u32 var_stub_addr; u32 tls_...;
struct PpuStubHeader {
    uint8_t  s_size;
    uint8_t  unk0;
    uint16_t version;
    uint16_t attrs;
    uint16_t num_func;
    uint16_t num_var;
    uint16_t num_tls;
    uint8_t  hash_info;
    uint8_t  tls_hash;
    uint8_t  unk1[2];
    uint32_t module_name_addr;
    uint32_t module_hash_addr;
    uint32_t func_fnid_addr;
    uint32_t func_stub_addr;
    uint32_t var_fnid_addr;
    uint32_t var_stub_addr;
    uint32_t tls_fnid_addr;
    uint32_t tls_stub_addr;
};

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("════════════════════════════════════════════════════\n");
    std::printf("  PS3 SELF PHDR + PRX param scanner\n");
    std::printf("════════════════════════════════════════════════════\n");

    const char* path = "/workspaces/codespace/rpcs3-src/bin/test/gs_gcm_basic_triangle.elf";
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

    // Find inner ELF.
    uint64_t elfOff = self_find_elf_offset(file.data(), file.size());
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)(file.data() + elfOff);
    uint64_t phoff   = elf_bswap64(eh->e_phoff);
    uint16_t phnum   = elf_bswap16(eh->e_phnum);
    uint16_t phentsz = elf_bswap16(eh->e_phentsize);

    std::printf("\n  PHDR table: %u entries @ ELF+0x%lx\n",
                (unsigned)phnum, (unsigned long)phoff);

    uint64_t procParamVaddr = 0;
    uint64_t procParamSize  = 0;

    for (uint16_t i = 0; i < phnum; ++i) {
        const Elf64_Phdr* ph = (const Elf64_Phdr*)(
            file.data() + elfOff + phoff + (uint64_t)i * phentsz);
        uint32_t p_type   = elf_bswap32(ph->p_type);
        uint64_t p_vaddr  = elf_bswap64(ph->p_vaddr);
        uint64_t p_filesz = elf_bswap64(ph->p_filesz);
        uint64_t p_memsz  = elf_bswap64(ph->p_memsz);

        const char* kind = "?";
        switch (p_type) {
            case 1: kind = "PT_LOAD"; break;
            case 2: kind = "PT_DYNAMIC"; break;
            case 4: kind = "PT_NOTE"; break;
            case 6: kind = "PT_PHDR"; break;
            case 7: kind = "PT_TLS"; break;
            case 0x60000001: kind = "PT_SCE_RELA"; break;
            case 0x60000002: kind = "PT_SCE_LICINFO"; break;
            case 0x60000003: kind = "PT_SCE_PROCPARAM"; break;
            case 0x60000004: kind = "PT_SCE_PPUNAME"; break;
            case 0x60000005: kind = "PT_SCE_PPURELA"; break;
            case 0x60000006: kind = "PT_SCE_COMMENT"; break;
        }
        std::printf("    [%u] %08x %-18s vaddr=0x%08lx filesz=0x%lx memsz=0x%lx\n",
                    (unsigned)i, p_type, kind,
                    (unsigned long)p_vaddr, (unsigned long)p_filesz,
                    (unsigned long)p_memsz);

        if (p_type == 0x60000001 /* PT_SCE_PROCPARAM alt */ ||
            p_type == 0x60000003) {
            procParamVaddr = p_vaddr;
            procParamSize  = p_filesz;
        }
    }

    // Parse section headers (the PPU SDK linker emits everything we
    // need as proper sections, so this is a more reliable path than
    // scanning loaded memory for magics).
    uint64_t shoff = elf_bswap64(eh->e_shoff);
    uint16_t shnum = elf_bswap16(eh->e_shnum);
    uint16_t shentsz = elf_bswap16(eh->e_shentsize);
    uint16_t shstrndx = elf_bswap16(eh->e_shstrndx);

    // Locate string table for section names.
    const Elf64_Shdr* strSh = (const Elf64_Shdr*)(
        file.data() + elfOff + shoff + (uint64_t)shstrndx * shentsz);
    uint64_t strSoff = elf_bswap64(strSh->sh_offset);
    const char* strtab = (const char*)(file.data() + elfOff + strSoff);

    auto findSec = [&](const char* name) -> const Elf64_Shdr* {
        for (uint16_t i = 0; i < shnum; ++i) {
            const Elf64_Shdr* sh = (const Elf64_Shdr*)(
                file.data() + elfOff + shoff + (uint64_t)i * shentsz);
            uint32_t nameIdx = elf_bswap32(sh->sh_name);
            if (!std::strcmp(strtab + nameIdx, name)) return sh;
        }
        return nullptr;
    };

    const Elf64_Shdr* shParam = findSec(".sys_proc_prx_param");
    const Elf64_Shdr* shStub  = findSec(".lib.stub");
    const Elf64_Shdr* shFNID  = findSec(".rodata.sceFNID");
    const Elf64_Shdr* shStubText = findSec(".sceStub.text");

    if (shParam) procParamVaddr = elf_bswap64(shParam->sh_addr);
    if (shStub && shFNID) {
        uint64_t stubAddr = elf_bswap64(shStub->sh_addr);
        uint64_t stubSize = elf_bswap64(shStub->sh_size);
        uint64_t fnidAddr = elf_bswap64(shFNID->sh_addr);
        uint64_t fnidSize = elf_bswap64(shFNID->sh_size);
        uint64_t textAddr = shStubText ? elf_bswap64(shStubText->sh_addr) : 0;
        std::printf("\n  .sys_proc_prx_param @ 0x%lx\n",
                    (unsigned long)procParamVaddr);
        std::printf("  .lib.stub           @ 0x%lx size=0x%lx (%lu entries)\n",
                    (unsigned long)stubAddr, (unsigned long)stubSize,
                    (unsigned long)(stubSize / 0x2C));
        std::printf("  .rodata.sceFNID     @ 0x%lx size=0x%lx (%lu FNIDs)\n",
                    (unsigned long)fnidAddr, (unsigned long)fnidSize,
                    (unsigned long)(fnidSize / 4));
        std::printf("  .sceStub.text       @ 0x%lx\n", (unsigned long)textAddr);

        CHECK(stubAddr + stubSize <= mem.size(), "stub section in mem");

        std::printf("\n  Imported modules:\n");
        int stubCount = 0;
        for (uint64_t a = stubAddr; a + 0x2C <= stubAddr + stubSize; a += 0x2C) {
            const uint8_t* s = mem.data() + a;
            uint16_t num_func    = (uint16_t)((s[6] << 8) | s[7]);
            uint16_t num_var     = (uint16_t)((s[8] << 8) | s[9]);
            uint32_t mod_name_a  = be32(s + 0x10);
            uint32_t func_fnid_a = be32(s + 0x14);
            uint32_t func_stub_a = be32(s + 0x18);
            const char* mod_name = "(?)";
            if (mod_name_a && mod_name_a + 32 <= mem.size())
                mod_name = (const char*)(mem.data() + mod_name_a);
            std::printf("    [%d] %-22s  %u funcs (%u vars)  fnid=0x%x stub=0x%x\n",
                        stubCount, mod_name, num_func, num_var,
                        func_fnid_a, func_stub_a);
            if (func_fnid_a && num_func &&
                func_fnid_a + num_func * 4 <= mem.size()) {
                int n = num_func < 6 ? num_func : 6;
                std::printf("        FNIDs:");
                for (int k = 0; k < n; ++k)
                    std::printf(" 0x%08x", be32(mem.data() + func_fnid_a + k * 4));
                if (num_func > 6) std::printf(" ...");
                std::printf("\n");
            }
            stubCount++;
        }
        CHECK(stubCount > 0, "found imported modules");
    } else {
        std::printf("  (no .lib.stub section — statically linked?)\n");
    }

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
