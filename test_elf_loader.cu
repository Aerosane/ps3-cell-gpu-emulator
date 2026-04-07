// test_elf_loader.cu — ELF/SELF Loader Test Suite
//
// Validates the PS3 ELF64 parser and loader against synthetic test binaries.
// Host-side tests — CUDA file extension for build-system compatibility only.

#include "elf_loader.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  ✅ %s\n", msg); tests_passed++; } \
    else { printf("  ❌ %s\n", msg); tests_failed++; } \
} while(0)

// ═══════════════════════════════════════════════════════════════
// Helper: write big-endian values into a byte buffer
// ═══════════════════════════════════════════════════════════════

static void put_be16(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)(v);
}

static void put_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >>  8);
    p[3] = (uint8_t)(v);
}

static void put_be64(uint8_t* p, uint64_t v) {
    put_be32(p,     (uint32_t)(v >> 32));
    put_be32(p + 4, (uint32_t)(v));
}

// ═══════════════════════════════════════════════════════════════
// Build a minimal valid PPC64 big-endian ELF in memory
// ═══════════════════════════════════════════════════════════════

struct TestELF {
    std::vector<uint8_t> data;
};

/// Create a minimal ELF with one PT_LOAD segment and one PT_SCE_PROCPARAM.
static TestELF make_test_elf() {
    // Layout:
    //   0x000 : ELF header  (64 bytes)
    //   0x040 : Phdr[0] — PT_LOAD          (56 bytes)
    //   0x078 : Phdr[1] — PT_SCE_PROCPARAM (56 bytes)
    //   0x0B0 : padding to 0x100
    //   0x100 : PT_LOAD payload  — PPC instructions (li r3,42; blr)
    //   0x200 : PT_SCE_PROCPARAM payload — process_param_t

    const size_t ehdr_size   = 64;
    const size_t phdr_size   = 56;
    const size_t num_phdrs   = 2;
    const size_t code_offset = 0x100;
    const size_t code_size   = 8;  // two 4-byte PPC instructions
    const size_t param_offset = 0x200;
    const size_t param_size  = 32; // process_param_t
    const size_t total       = 0x300;

    const uint64_t load_vaddr  = 0x10000;
    const uint64_t param_vaddr = 0x20000;
    const uint64_t entry_point = 0x10000;

    TestELF elf;
    elf.data.resize(total, 0);
    uint8_t* d = elf.data.data();

    // ── ELF header ──────────────────────────────────────────
    d[0] = 0x7F; d[1] = 'E'; d[2] = 'L'; d[3] = 'F'; // magic
    d[EI_CLASS]   = 2;    // 64-bit
    d[EI_DATA]    = 2;    // big-endian
    d[EI_VERSION] = 1;    // EV_CURRENT
    d[EI_OSABI]   = 0x00; // executable

    put_be16(d + 16, 0x0002);          // e_type = ET_EXEC
    put_be16(d + 18, 0x0015);          // e_machine = PPC64
    put_be32(d + 20, 1);               // e_version
    put_be64(d + 24, entry_point);     // e_entry
    put_be64(d + 32, ehdr_size);       // e_phoff (phdr table right after ehdr)
    put_be64(d + 40, 0);               // e_shoff (no sections)
    put_be32(d + 48, 0);               // e_flags
    put_be16(d + 52, (uint16_t)ehdr_size);  // e_ehsize
    put_be16(d + 54, (uint16_t)phdr_size);  // e_phentsize
    put_be16(d + 56, (uint16_t)num_phdrs);  // e_phnum
    put_be16(d + 58, (uint16_t)64);         // e_shentsize
    put_be16(d + 60, 0);                    // e_shnum
    put_be16(d + 62, 0);                    // e_shstrndx

    // ── Phdr[0]: PT_LOAD ────────────────────────────────────
    uint8_t* ph0 = d + ehdr_size;
    put_be32(ph0 +  0, PT_LOAD);       // p_type
    put_be32(ph0 +  4, 0x5);           // p_flags = PF_R|PF_X
    put_be64(ph0 +  8, code_offset);   // p_offset
    put_be64(ph0 + 16, load_vaddr);    // p_vaddr
    put_be64(ph0 + 24, load_vaddr);    // p_paddr
    put_be64(ph0 + 32, code_size);     // p_filesz
    put_be64(ph0 + 40, code_size + 16);// p_memsz (extra for BSS test)
    put_be64(ph0 + 48, 0x10);          // p_align

    // ── Phdr[1]: PT_SCE_PROCPARAM ───────────────────────────
    uint8_t* ph1 = d + ehdr_size + phdr_size;
    put_be32(ph1 +  0, PT_SCE_PROCPARAM);
    put_be32(ph1 +  4, 0x4);           // p_flags = PF_R
    put_be64(ph1 +  8, param_offset);
    put_be64(ph1 + 16, param_vaddr);
    put_be64(ph1 + 24, param_vaddr);
    put_be64(ph1 + 32, param_size);
    put_be64(ph1 + 40, param_size);
    put_be64(ph1 + 48, 0x4);

    // ── Code payload: li r3,42 (= addi r3,r0,42) ; blr ─────
    //   li r3, 42  → 0x3860002A
    //   blr        → 0x4E800020
    put_be32(d + code_offset,     0x3860002Au);
    put_be32(d + code_offset + 4, 0x4E800020u);

    // ── Process param payload ───────────────────────────────
    uint8_t* pp = d + param_offset;
    put_be32(pp +  0, param_size);      // size
    put_be32(pp +  4, PROC_PARAM_MAGIC);// magic
    put_be32(pp +  8, 1);               // version
    put_be32(pp + 12, 0x00360001);      // sdk_version (3.60)
    put_be32(pp + 16, 0x000003E9);      // primary_prio = 1001
    put_be32(pp + 20, 0x00100000);      // primary_stacksize = 1MB
    put_be32(pp + 24, 0x00100000);      // malloc_pagesize
    put_be32(pp + 28, 0);               // ppc_seg

    return elf;
}

/// Create a test ELF with multiple PT_LOAD segments.
static TestELF make_multi_segment_elf() {
    const size_t ehdr_size = 64;
    const size_t phdr_size = 56;
    const size_t num_phdrs = 3;
    const size_t total     = 0x500;

    TestELF elf;
    elf.data.resize(total, 0);
    uint8_t* d = elf.data.data();

    // ── ELF header ──────────────────────────────────────────
    d[0] = 0x7F; d[1] = 'E'; d[2] = 'L'; d[3] = 'F';
    d[EI_CLASS] = 2; d[EI_DATA] = 2; d[EI_VERSION] = 1;

    put_be16(d + 16, 0x0002);
    put_be16(d + 18, 0x0015);
    put_be32(d + 20, 1);
    put_be64(d + 24, 0x10000);
    put_be64(d + 32, ehdr_size);
    put_be64(d + 40, 0);
    put_be32(d + 48, 0);
    put_be16(d + 52, (uint16_t)ehdr_size);
    put_be16(d + 54, (uint16_t)phdr_size);
    put_be16(d + 56, (uint16_t)num_phdrs);
    put_be16(d + 58, 64); put_be16(d + 60, 0); put_be16(d + 62, 0);

    // Segment 0: code at vaddr 0x10000, file offset 0x200, 16 bytes
    uint8_t* ph0 = d + ehdr_size;
    put_be32(ph0, PT_LOAD);
    put_be32(ph0 + 4, 0x5);
    put_be64(ph0 +  8, 0x200);
    put_be64(ph0 + 16, 0x10000);
    put_be64(ph0 + 24, 0x10000);
    put_be64(ph0 + 32, 16);
    put_be64(ph0 + 40, 16);
    put_be64(ph0 + 48, 0x10);

    // Segment 1: data at vaddr 0x20000, file offset 0x300, 32 bytes
    uint8_t* ph1 = d + ehdr_size + phdr_size;
    put_be32(ph1, PT_LOAD);
    put_be32(ph1 + 4, 0x6); // PF_R|PF_W
    put_be64(ph1 +  8, 0x300);
    put_be64(ph1 + 16, 0x20000);
    put_be64(ph1 + 24, 0x20000);
    put_be64(ph1 + 32, 32);
    put_be64(ph1 + 40, 64); // memsz > filesz → BSS
    put_be64(ph1 + 48, 0x10);

    // Segment 2: rodata at vaddr 0x30000, file offset 0x400, 8 bytes
    uint8_t* ph2 = d + ehdr_size + 2 * phdr_size;
    put_be32(ph2, PT_LOAD);
    put_be32(ph2 + 4, 0x4); // PF_R
    put_be64(ph2 +  8, 0x400);
    put_be64(ph2 + 16, 0x30000);
    put_be64(ph2 + 24, 0x30000);
    put_be64(ph2 + 32, 8);
    put_be64(ph2 + 40, 8);
    put_be64(ph2 + 48, 0x8);

    // Fill segment payloads with recognizable patterns
    memset(d + 0x200, 0xAA, 16);  // seg0: 0xAA
    memset(d + 0x300, 0xBB, 32);  // seg1: 0xBB
    memset(d + 0x400, 0xCC, 8);   // seg2: 0xCC

    return elf;
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

static void test_basic_load() {
    printf("╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: Basic ELF Load                ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    TestELF elf = make_test_elf();
    const size_t mem_size = 256 * 1024;  // 256 KB test sandbox
    uint8_t* mem = (uint8_t*)calloc(1, mem_size);

    PS3ExeInfo info;
    int rc = ps3_load_elf(elf.data.data(), elf.data.size(), mem, mem_size, &info);

    CHECK(rc == ELF_OK, "Load returns ELF_OK");
    CHECK(info.entry_point == 0x10000, "Entry point = 0x10000");
    CHECK(info.is_prx == false, "Not a PRX");
    CHECK(info.num_segments >= 1, "At least 1 segment loaded");

    // Verify instructions at correct address
    // li r3,42 = 0x3860002A in big-endian
    uint32_t inst0 = read_be32(mem + 0x10000);
    uint32_t inst1 = read_be32(mem + 0x10004);
    CHECK(inst0 == 0x3860002Au, "Instruction 'li r3,42' at 0x10000");
    CHECK(inst1 == 0x4E800020u, "Instruction 'blr' at 0x10004");

    // Verify BSS zero-fill (memsz = code_size + 16, filesz = code_size)
    bool bss_zeroed = true;
    for (int i = 8; i < 24; i++) {
        if (mem[0x10000 + i] != 0) { bss_zeroed = false; break; }
    }
    CHECK(bss_zeroed, "BSS region zero-filled");

    // Verify process params
    CHECK(info.has_proc_param, "Process param parsed");
    CHECK(info.sdk_version == 0x00360001, "SDK version = 0x00360001");
    CHECK(info.primary_stacksize == 0x00100000, "Stack size = 1MB");

    ps3_print_elf_info(&info);
    free(mem);
    printf("\n");
}

static void test_self_detection() {
    printf("╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: SELF Detection                ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    // Build a fake SELF: SCE header + ext header pointing to an embedded ELF
    TestELF inner = make_test_elf();

    // Place the ELF at offset 0x200 inside a SELF wrapper
    const size_t elf_embed_offset = 0x200;
    const size_t self_size = elf_embed_offset + inner.data.size();
    std::vector<uint8_t> self_data(self_size, 0);

    // SCE header
    put_be32(self_data.data() + 0, SCE_MAGIC);       // magic
    put_be32(self_data.data() + 4, 2);                // version
    put_be16(self_data.data() + 8, 0);                // key_revision
    put_be16(self_data.data() + 10, 1);               // header_type
    put_be32(self_data.data() + 12, 0);               // metadata_offset
    put_be64(self_data.data() + 16, elf_embed_offset);// header_len
    put_be64(self_data.data() + 24, inner.data.size());// data_len

    // ExtHeader — starts at offset sizeof(SceHeader) = 32
    uint8_t* ext = self_data.data() + 32;
    // elf_offset is at byte 44 within ExtHeader (after authid(8)+vendor(8)+self_type(8)+version(8)+padding(8)+app_info_offset(4))
    put_be32(ext + 44, (uint32_t)elf_embed_offset);   // elf_offset

    // Copy inner ELF at the embed offset
    memcpy(self_data.data() + elf_embed_offset, inner.data.data(), inner.data.size());

    // Try to load it
    const size_t mem_size = 256 * 1024;
    uint8_t* mem = (uint8_t*)calloc(1, mem_size);
    PS3ExeInfo info;

    int rc = ps3_load_elf(self_data.data(), self_data.size(), mem, mem_size, &info);
    CHECK(rc == ELF_OK, "SELF-wrapped ELF loads successfully");
    CHECK(info.entry_point == 0x10000, "SELF: entry point correct");

    uint32_t inst = read_be32(mem + 0x10000);
    CHECK(inst == 0x3860002Au, "SELF: code loaded correctly");

    // Also test that bare SCE magic without valid ELF inside fails
    std::vector<uint8_t> bad_self(64, 0);
    put_be32(bad_self.data(), SCE_MAGIC);
    PS3ExeInfo bad_info;
    int rc2 = ps3_load_elf(bad_self.data(), bad_self.size(), mem, mem_size, &bad_info);
    CHECK(rc2 != ELF_OK, "Invalid SELF returns error");

    free(mem);
    printf("\n");
}

static void test_invalid_elf() {
    printf("╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: Invalid ELF Rejection         ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    const size_t mem_size = 256 * 1024;
    uint8_t* mem = (uint8_t*)calloc(1, mem_size);
    PS3ExeInfo info;

    // Bad magic
    {
        uint8_t bad[128] = {};
        bad[0] = 0x00; bad[1] = 'E'; bad[2] = 'L'; bad[3] = 'F';
        int rc = ps3_load_elf(bad, sizeof(bad), mem, mem_size, &info);
        CHECK(rc == ELF_ERR_MAGIC, "Bad magic → ELF_ERR_MAGIC");
    }

    // Wrong class (32-bit)
    {
        uint8_t bad[128] = {};
        bad[0] = 0x7F; bad[1] = 'E'; bad[2] = 'L'; bad[3] = 'F';
        bad[EI_CLASS] = 1; // 32-bit
        bad[EI_DATA]  = 2;
        int rc = ps3_load_elf(bad, sizeof(bad), mem, mem_size, &info);
        CHECK(rc == ELF_ERR_CLASS, "32-bit ELF → ELF_ERR_CLASS");
    }

    // Wrong endianness (little-endian)
    {
        uint8_t bad[128] = {};
        bad[0] = 0x7F; bad[1] = 'E'; bad[2] = 'L'; bad[3] = 'F';
        bad[EI_CLASS] = 2;
        bad[EI_DATA]  = 1; // little-endian
        int rc = ps3_load_elf(bad, sizeof(bad), mem, mem_size, &info);
        CHECK(rc == ELF_ERR_CLASS, "Little-endian → ELF_ERR_CLASS");
    }

    // Wrong machine (x86_64)
    {
        uint8_t bad[128] = {};
        bad[0] = 0x7F; bad[1] = 'E'; bad[2] = 'L'; bad[3] = 'F';
        bad[EI_CLASS] = 2;
        bad[EI_DATA]  = 2;
        put_be16(bad + 18, 0x003E); // x86_64
        int rc = ps3_load_elf(bad, sizeof(bad), mem, mem_size, &info);
        CHECK(rc == ELF_ERR_MACHINE, "x86_64 → ELF_ERR_MACHINE");
    }

    // Truncated file
    {
        uint8_t tiny[8] = {0x7F, 'E', 'L', 'F', 2, 2, 1, 0};
        int rc = ps3_load_elf(tiny, sizeof(tiny), mem, mem_size, &info);
        CHECK(rc == ELF_ERR_SIZE, "Truncated file → ELF_ERR_SIZE");
    }

    free(mem);
    printf("\n");
}

static void test_segment_loading() {
    printf("╔═══════════════════════════════════════╗\n");
    printf("║  TEST 4: Multi-Segment Loading         ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    TestELF elf = make_multi_segment_elf();
    const size_t mem_size = 256 * 1024;
    uint8_t* mem = (uint8_t*)calloc(1, mem_size);

    PS3ExeInfo info;
    int rc = ps3_load_elf(elf.data.data(), elf.data.size(), mem, mem_size, &info);

    CHECK(rc == ELF_OK, "Multi-segment load OK");
    CHECK(info.num_segments == 3, "3 segments loaded");

    // Verify segment 0 data (0xAA pattern at vaddr 0x10000)
    bool seg0_ok = true;
    for (int i = 0; i < 16; i++) {
        if (mem[0x10000 + i] != 0xAA) { seg0_ok = false; break; }
    }
    CHECK(seg0_ok, "Segment 0: 0xAA pattern at 0x10000");

    // Verify segment 1 data (0xBB pattern at vaddr 0x20000, 32 bytes)
    bool seg1_ok = true;
    for (int i = 0; i < 32; i++) {
        if (mem[0x20000 + i] != 0xBB) { seg1_ok = false; break; }
    }
    CHECK(seg1_ok, "Segment 1: 0xBB pattern at 0x20000");

    // Verify segment 1 BSS (bytes 32..63 should be zero)
    bool seg1_bss = true;
    for (int i = 32; i < 64; i++) {
        if (mem[0x20000 + i] != 0) { seg1_bss = false; break; }
    }
    CHECK(seg1_bss, "Segment 1: BSS zero-filled (0x20020..0x2003F)");

    // Verify segment 2 data (0xCC pattern at vaddr 0x30000)
    bool seg2_ok = true;
    for (int i = 0; i < 8; i++) {
        if (mem[0x30000 + i] != 0xCC) { seg2_ok = false; break; }
    }
    CHECK(seg2_ok, "Segment 2: 0xCC pattern at 0x30000");

    // Verify segment metadata
    CHECK(info.segments[0].addr == 0x10000, "Seg 0 addr = 0x10000");
    CHECK(info.segments[1].addr == 0x20000, "Seg 1 addr = 0x20000");
    CHECK(info.segments[2].addr == 0x30000, "Seg 2 addr = 0x30000");
    CHECK(info.segments[1].size == 64, "Seg 1 memsz = 64 (includes BSS)");

    ps3_print_elf_info(&info);
    free(mem);
    printf("\n");
}

static void test_tls_segment() {
    printf("╔═══════════════════════════════════════╗\n");
    printf("║  TEST 5: TLS Segment Parsing           ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    // Reuse the basic ELF but inject a PT_TLS phdr
    const size_t ehdr_size = 64;
    const size_t phdr_size = 56;
    const size_t total     = 0x300;

    std::vector<uint8_t> raw(total, 0);
    uint8_t* d = raw.data();

    // Minimal valid ELF header
    d[0] = 0x7F; d[1] = 'E'; d[2] = 'L'; d[3] = 'F';
    d[EI_CLASS] = 2; d[EI_DATA] = 2; d[EI_VERSION] = 1;
    put_be16(d + 16, 0x0002);
    put_be16(d + 18, 0x0015);
    put_be32(d + 20, 1);
    put_be64(d + 24, 0x10000);
    put_be64(d + 32, ehdr_size);
    put_be64(d + 40, 0);
    put_be32(d + 48, 0);
    put_be16(d + 52, (uint16_t)ehdr_size);
    put_be16(d + 54, (uint16_t)phdr_size);
    put_be16(d + 56, 1);  // 1 phdr: PT_TLS
    put_be16(d + 58, 64); put_be16(d + 60, 0); put_be16(d + 62, 0);

    // PT_TLS phdr
    uint8_t* ph = d + ehdr_size;
    put_be32(ph, PT_TLS);
    put_be32(ph + 4, 0x4);
    put_be64(ph +  8, 0);        // p_offset
    put_be64(ph + 16, 0x50000);  // p_vaddr
    put_be64(ph + 24, 0x50000);  // p_paddr
    put_be64(ph + 32, 0x100);    // p_filesz
    put_be64(ph + 40, 0x200);    // p_memsz
    put_be64(ph + 48, 0x10);     // p_align

    const size_t mem_size = 256 * 1024;
    uint8_t* mem = (uint8_t*)calloc(1, mem_size);
    PS3ExeInfo info;

    int rc = ps3_load_elf(raw.data(), raw.size(), mem, mem_size, &info);
    CHECK(rc == ELF_OK, "TLS ELF loads OK");
    CHECK(info.has_tls, "TLS info present");
    CHECK(info.tls.addr == 0x50000, "TLS addr = 0x50000");
    CHECK(info.tls.filesz == 0x100, "TLS filesz = 0x100");
    CHECK(info.tls.memsz == 0x200, "TLS memsz = 0x200");

    free(mem);
    printf("\n");
}

static void test_prx_detection() {
    printf("╔═══════════════════════════════════════╗\n");
    printf("║  TEST 6: PRX Module Detection          ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    // Minimal ELF with e_type = 0xFFA4 (PRX)
    std::vector<uint8_t> raw(128, 0);
    uint8_t* d = raw.data();

    d[0] = 0x7F; d[1] = 'E'; d[2] = 'L'; d[3] = 'F';
    d[EI_CLASS] = 2; d[EI_DATA] = 2; d[EI_VERSION] = 1;
    put_be16(d + 16, 0xFFA4);  // ET_PRX
    put_be16(d + 18, 0x0015);
    put_be32(d + 20, 1);
    put_be64(d + 24, 0);
    put_be64(d + 32, 0);       // no phdrs
    put_be16(d + 52, 64);
    put_be16(d + 54, 56);
    put_be16(d + 56, 0);       // 0 phdrs

    const size_t mem_size = 256 * 1024;
    uint8_t* mem = (uint8_t*)calloc(1, mem_size);
    PS3ExeInfo info;

    int rc = ps3_load_elf(raw.data(), raw.size(), mem, mem_size, &info);
    CHECK(rc == ELF_OK, "PRX loads OK");
    CHECK(info.is_prx == true, "Detected as PRX module");

    free(mem);
    printf("\n");
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  🚀 PS3 ELF/SELF Loader Test Suite       ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  PPC64 Big-Endian ELF Parser + Loader    ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    test_basic_load();
    test_self_detection();
    test_invalid_elf();
    test_segment_loading();
    test_tls_segment();
    test_prx_detection();

    printf("═══════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
