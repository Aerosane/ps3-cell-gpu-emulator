// test_elf_boot.cu — End-to-end PS3 ELF loader → PPC warp-JIT bringup
//
// Synthesizes a minimal PowerPC64 big-endian ELF executable in memory
// (one PT_LOAD segment containing a tiny PPC program), pushes it through
// the existing host-side ps3_load_elf parser, then executes the loaded
// segment via the NVRTC-cached warp JIT and checks the resulting PPEState.
//
// This is the first piece of plumbing that proves the loader → executor
// path works end-to-end.  Real PS3 SELF/ELF binaries can now be dropped
// in once HLE coverage catches up.

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "elf_loader.h"
#include "ppc_defs.h"
#include "ppc_jit.h"

using namespace ppc;

// ── Borrow PPC opcode helpers (mirror of test_cell.cu) ───────────
static uint32_t ppc_addi (int rD, int rA, int16_t imm) {
    return (14u<<26) | ((uint32_t)rD<<21) | ((uint32_t)rA<<16) | (uint16_t)imm;
}
static uint32_t ppc_addis(int rD, int rA, int16_t imm) {
    return (15u<<26) | ((uint32_t)rD<<21) | ((uint32_t)rA<<16) | (uint16_t)imm;
}
static uint32_t ppc_add  (int rD, int rA, int rB) {
    return (31u<<26) | ((uint32_t)rD<<21) | ((uint32_t)rA<<16) |
           ((uint32_t)rB<<11) | (266u<<1);
}
static uint32_t ppc_mullw(int rD, int rA, int rB) {
    return (31u<<26) | ((uint32_t)rD<<21) | ((uint32_t)rA<<16) |
           ((uint32_t)rB<<11) | (235u<<1);
}
static uint32_t ppc_stw  (int rS, int16_t d, int rA) {
    return (36u<<26) | ((uint32_t)rS<<21) | ((uint32_t)rA<<16) | (uint16_t)d;
}
static uint32_t ppc_sc()  { return (17u<<26) | (1u<<1); }

// ── Big-endian writers ───────────────────────────────────────────
static void be_u16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back((uint8_t)(v >> 8)); b.push_back((uint8_t)v);
}
static void be_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 3; i >= 0; --i) b.push_back((uint8_t)(v >> (i*8)));
}
static void be_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 7; i >= 0; --i) b.push_back((uint8_t)(v >> (i*8)));
}

// ── Build a minimal PPC64 BE ELF with one PT_LOAD segment ────────
static std::vector<uint8_t> build_test_elf(uint64_t loadAddr,
                                           const uint32_t* code,
                                           size_t codeWords)
{
    constexpr size_t EHDR = 64;     // sizeof(Elf64_Ehdr)
    constexpr size_t PHDR = 56;     // sizeof(Elf64_Phdr)
    const size_t codeBytes = codeWords * 4;
    const size_t codeOff   = EHDR + PHDR;

    std::vector<uint8_t> f;
    f.reserve(codeOff + codeBytes);

    // ── ELF identification ────────────────────────────────────
    f.push_back(0x7F); f.push_back('E'); f.push_back('L'); f.push_back('F');
    f.push_back(2);   // EI_CLASS = ELFCLASS64
    f.push_back(2);   // EI_DATA  = ELFDATA2MSB (big-endian)
    f.push_back(1);   // EI_VERSION
    f.push_back(0);   // EI_OSABI
    f.push_back(0);   // EI_ABIVERSION
    while (f.size() < 16) f.push_back(0); // padding to EI_NIDENT

    be_u16(f, 0x0002);          // e_type     = ET_EXEC
    be_u16(f, 0x0015);          // e_machine  = EM_PPC64 (21)
    be_u32(f, 1);               // e_version
    be_u64(f, loadAddr);        // e_entry
    be_u64(f, EHDR);            // e_phoff
    be_u64(f, 0);               // e_shoff
    be_u32(f, 0);               // e_flags
    be_u16(f, EHDR);            // e_ehsize
    be_u16(f, PHDR);            // e_phentsize
    be_u16(f, 1);               // e_phnum
    be_u16(f, 0);               // e_shentsize
    be_u16(f, 0);               // e_shnum
    be_u16(f, 0);               // e_shstrndx

    // ── Single PT_LOAD program header ─────────────────────────
    be_u32(f, 0x00000001);      // p_type   = PT_LOAD
    be_u32(f, 0x00000005);      // p_flags  = PF_R | PF_X
    be_u64(f, codeOff);         // p_offset
    be_u64(f, loadAddr);        // p_vaddr
    be_u64(f, loadAddr);        // p_paddr
    be_u64(f, codeBytes);       // p_filesz
    be_u64(f, codeBytes);       // p_memsz
    be_u64(f, 0x10000);         // p_align

    // ── Code payload (PPC big-endian instructions) ────────────
    for (size_t i = 0; i < codeWords; ++i) be_u32(f, code[i]);
    return f;
}

// ── PPC warp-JIT runner (mirrors test_cell.cu helper) ────────────
// Allocates a full PS3 sandbox on the device (the warp runner reads
// up to 64 MB back during block discovery) and copies the loader's
// host image into it before launching.
static bool run_loaded(const uint8_t* mem, size_t memSize,
                       uint64_t entry, uint32_t maxCycles,
                       PPEState* outSt, float* outMs)
{
    uint8_t* d_mem = nullptr;
    if (cudaMalloc(&d_mem, PS3_SANDBOX_SIZE) != cudaSuccess) return false;
    cudaMemset(d_mem, 0, PS3_SANDBOX_SIZE);
    size_t copy = (memSize < PS3_SANDBOX_SIZE) ? memSize : PS3_SANDBOX_SIZE;
    cudaMemcpy(d_mem, mem, copy, cudaMemcpyHostToDevice);

    ppc_jit::PPCJITState jit;
    ppc_jit::ppc_jit_init(&jit);

    *outSt = {};
    outSt->pc = entry;
    uint32_t cyc = 0;
    ppc_jit::ppc_jit_run_warp(&jit, outSt, d_mem, maxCycles, outMs, &cyc);
    outSt->cycles = cyc;

    ppc_jit::ppc_jit_shutdown(&jit);
    cudaFree(d_mem);
    return true;
}

int main() {
    printf("\n╔═══════════════════════════════════════════════╗\n");
    printf("║  PS3 ELF Loader → PPC Warp-JIT Bringup        ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");

    int failures = 0;
    const uint64_t kLoadAddr = 0x10000;

    // Tiny program: r3=42 + r4=58 → r5=100; r5*3 → r7=300; store at 0x1000;
    // r11=1 (SYS_PROCESS_EXIT) + sc → halt.
    uint32_t code[] = {
        ppc_addi (3, 0, 42),
        ppc_addi (4, 0, 58),
        ppc_add  (5, 3, 4),
        ppc_addi (6, 0, 3),
        ppc_mullw(7, 5, 6),
        ppc_addis(8, 0, 0),
        ppc_stw  (7, 0x1000, 8),
        ppc_addi (11, 0, 1),
        ppc_sc   (),
    };
    const size_t codeWords = sizeof(code)/4;

    // ── 1. Synthesize an in-memory ELF ────────────────────────
    std::vector<uint8_t> elf = build_test_elf(kLoadAddr, code, codeWords);
    printf("\n── Synthesized ELF: %zu bytes\n", elf.size());

    // ── 2. Load it with the production PS3 loader ─────────────
    constexpr size_t kSandboxBytes = 4 * 1024 * 1024;
    std::vector<uint8_t> mem(kSandboxBytes, 0);

    PS3ExeInfo info{};
    int rc = ps3_load_elf(elf.data(), elf.size(),
                          mem.data(), mem.size(), &info);

    printf("  ps3_load_elf      → %d %s\n", rc, rc == ELF_OK ? "(OK)" : "(FAIL)");
    printf("  entry_point       = 0x%llx\n", (unsigned long long)info.entry_point);
    printf("  num_segments      = %d\n", info.num_segments);
    if (info.num_segments > 0) {
        printf("  segment[0].addr   = 0x%llx  size=%llu  flags=0x%x\n",
               (unsigned long long)info.segments[0].addr,
               (unsigned long long)info.segments[0].size,
               info.segments[0].flags);
    }

    if (rc != ELF_OK)              { printf("  ❌ loader failed\n"); ++failures; }
    if (info.entry_point != kLoadAddr)
                                   { printf("  ❌ entry mismatch\n"); ++failures; }
    if (info.num_segments != 1)    { printf("  ❌ segment count wrong\n"); ++failures; }

    // ── 3. Verify the code landed at the expected vaddr ───────
    bool bytesOk = (memcmp(mem.data() + kLoadAddr, elf.data() + 64 + 56,
                           codeWords * 4) == 0);
    printf("  loaded bytes match ELF payload : %s\n", bytesOk ? "OK" : "FAIL");
    if (!bytesOk) ++failures;

    // ── 4. Execute the loaded segment through the warp JIT ────
    PPEState st{};
    float ms = 0;
    if (!run_loaded(mem.data(), mem.size(), info.entry_point,
                    1000, &st, &ms)) {
        printf("  ❌ run_loaded: cudaMalloc failed\n");
        ++failures;
    } else {
        printf("\n── Executed loaded code (%.3f ms, %u cyc)\n", ms, st.cycles);
        printf("  r3=%llu  r4=%llu  r5=%llu  r7=%llu  halted=%d\n",
               (unsigned long long)st.gpr[3], (unsigned long long)st.gpr[4],
               (unsigned long long)st.gpr[5], (unsigned long long)st.gpr[7],
               (int)st.halted);
        bool execOk = (st.gpr[5] == 100) && (st.gpr[7] == 300) && st.halted;
        printf("  result            : %s\n", execOk ? "OK" : "FAIL");
        if (!execOk) ++failures;
    }

    printf("\n══════════════════════════════════════════════════\n");
    if (failures == 0) printf("  ✅  ALL PASSED (ELF → loader → warp JIT)\n");
    else               printf("  ❌  %d failure(s)\n", failures);
    printf("══════════════════════════════════════════════════\n");
    return failures == 0 ? 0 : 1;
}
