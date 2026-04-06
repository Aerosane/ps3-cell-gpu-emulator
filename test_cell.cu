// test_cell.cu — PS3 Cell BE Emulator Test Harness
// Assembles tiny PPC and SPU programs in memory, runs the megakernel,
// and verifies correctness.
//
#include "ppc_defs.h"
#include "spu_defs.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace ppc;
using namespace spu;

// ═══════════════════════════════════════════════════════════════
// Extern C APIs from our modules
// ═══════════════════════════════════════════════════════════════
extern "C" {
    int  megakernel_init();
    int  megakernel_load(uint64_t offset, const void* data, size_t size);
    int  megakernel_set_entry(uint64_t pc, uint64_t sp, uint64_t toc);
    float megakernel_run(uint32_t maxCycles);
    int  megakernel_read_state(PPEState* out);
    int  megakernel_read_hle_log(uint32_t* out, int maxEntries);
    void megakernel_shutdown();

    int   spu_init();
    int   spu_load_program(int spuId, const void* data, size_t size, uint32_t entryPC);
    float spu_run(uint32_t maxCycles, uint8_t* d_mainMem);
    int   spu_read_state(int spuId, SPUState* out);
    void  spu_shutdown();

    int   cell_init();
    int   cell_load_ppe(uint64_t loadAddr, const void* data, size_t size,
                        uint64_t entryPC, uint64_t stackPtr, uint64_t toc);
    int   cell_load_spu(int spuId, const void* data, size_t size, uint32_t entryPC);
    float cell_run(uint32_t totalSlices, uint32_t cyclesPerSlice);
    int   cell_read_ppe(PPEState* out);
    int   cell_read_spu(int spuId, SPUState* out);
    void  cell_shutdown();
}

// ═══════════════════════════════════════════════════════════════
// PPC Instruction Assembler (big-endian 32-bit words)
// ═══════════════════════════════════════════════════════════════

static uint32_t bswap32(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

// addi rD, rA, SIMM  (also li rD, SIMM when rA=0)
static uint32_t ppc_addi(int rD, int rA, int16_t imm) {
    return bswap32((14u << 26) | (rD << 21) | (rA << 16) | (uint16_t)imm);
}
// addis rD, rA, SIMM  (also lis rD, SIMM when rA=0)
static uint32_t ppc_addis(int rD, int rA, int16_t imm) {
    return bswap32((15u << 26) | (rD << 21) | (rA << 16) | (uint16_t)imm);
}
// ori rA, rS, UIMM
static uint32_t ppc_ori(int rA, int rS, uint16_t imm) {
    return bswap32((24u << 26) | (rS << 21) | (rA << 16) | imm);
}
// add rD, rA, rB  (XO=266)
static uint32_t ppc_add(int rD, int rA, int rB) {
    return bswap32((31u << 26) | (rD << 21) | (rA << 16) | (rB << 11) | (266 << 1));
}
// stw rS, d(rA)
static uint32_t ppc_stw(int rS, int16_t d, int rA) {
    return bswap32((36u << 26) | (rS << 21) | (rA << 16) | (uint16_t)d);
}
// lwz rD, d(rA)
static uint32_t ppc_lwz(int rD, int16_t d, int rA) {
    return bswap32((32u << 26) | (rD << 21) | (rA << 16) | (uint16_t)d);
}
// mullw rD, rA, rB  (XO=235)
static uint32_t ppc_mullw(int rD, int rA, int rB) {
    return bswap32((31u << 26) | (rD << 21) | (rA << 16) | (rB << 11) | (235 << 1));
}
// sc (syscall, r11 = syscall number)
static uint32_t ppc_sc() {
    return bswap32((17u << 26) | 2);
}
// b disp (branch relative)
static uint32_t ppc_b(int32_t disp) {
    return bswap32((18u << 26) | (disp & 0x03FFFFFC));
}

// ═══════════════════════════════════════════════════════════════
// SPU Instruction Assembler (big-endian)
// ═══════════════════════════════════════════════════════════════

// il rT, I16 (Immediate Load Word)  — op9 = 0x081
static uint32_t spu_il(int rT, int16_t imm) {
    return bswap32((0x081u << 23) | (((uint32_t)(uint16_t)imm) << 7) | (rT & 0x7F));
}
// a rT, rA, rB (Add Word)  — op11 = 0x0C0
static uint32_t spu_a(int rT, int rA, int rB) {
    return bswap32((0x0C0u << 21) | ((rT & 0x7F) << 14) | ((rB & 0x7F) << 7) | (rA & 0x7F));
}
// fm rT, rA, rB (Float Multiply)  — op11 = 0x2C6
static uint32_t spu_fm(int rT, int rA, int rB) {
    return bswap32((0x2C6u << 21) | ((rT & 0x7F) << 14) | ((rB & 0x7F) << 7) | (rA & 0x7F));
}
// fa rT, rA, rB (Float Add)  — op11 = 0x2C4
static uint32_t spu_fa(int rT, int rA, int rB) {
    return bswap32((0x2C4u << 21) | ((rT & 0x7F) << 14) | ((rB & 0x7F) << 7) | (rA & 0x7F));
}
// sf rT, rA, rB (Subtract From)  — op11 = 0x040
static uint32_t spu_sf(int rT, int rA, int rB) {
    return bswap32((0x040u << 21) | ((rT & 0x7F) << 14) | ((rB & 0x7F) << 7) | (rA & 0x7F));
}
// stop  — op11 = 0x000
static uint32_t spu_stop() {
    return bswap32(0x00000000u);
}
// nop  — op11 = 0x201
static uint32_t spu_nop() {
    return bswap32(0x201u << 21);
}
// fma rT, rA, rB, rC  — op4 = 0xE
static uint32_t spu_fma(int rT, int rA, int rB, int rC) {
    return bswap32((0xEu << 28) | ((rT & 0x7F) << 21) | ((rB & 0x7F) << 14) | ((rA & 0x7F) << 7) | (rC & 0x7F));
}

// ═══════════════════════════════════════════════════════════════
// Test 1: PPE Integer ALU
// ═══════════════════════════════════════════════════════════════

static bool test_ppe_alu() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: PPE Integer ALU              ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    megakernel_init();

    // Assemble: compute (42 + 58) * 3 = 300, store to address 0x1000
    //   li r3, 42
    //   li r4, 58
    //   add r5, r3, r4      → r5 = 100
    //   li r6, 3
    //   mullw r7, r5, r6    → r7 = 300
    //   lis r8, 0            → r8 = 0
    //   stw r7, 0x1000(r8)  → mem[0x1000] = 300
    //   li r11, 1            → syscall number = SYS_PROCESS_EXIT
    //   sc                   → halt

    uint32_t code[] = {
        ppc_addi(3, 0, 42),        // li r3, 42
        ppc_addi(4, 0, 58),        // li r4, 58
        ppc_add(5, 3, 4),          // add r5, r3, r4
        ppc_addi(6, 0, 3),         // li r6, 3
        ppc_mullw(7, 5, 6),        // mullw r7, r5, r6
        ppc_addis(8, 0, 0),        // lis r8, 0
        ppc_stw(7, 0x1000, 8),     // stw r7, 0x1000(r8)
        ppc_addi(11, 0, 1),        // li r11, 1 (SYS_PROCESS_EXIT)
        ppc_sc(),                   // sc
    };

    uint64_t loadAddr = 0x10000;
    megakernel_load(loadAddr, code, sizeof(code));
    megakernel_set_entry(loadAddr, 0x80000, 0);

    float ms = megakernel_run(1000);

    PPEState st;
    megakernel_read_state(&st);

    printf("  Execution time: %.3f ms\n", ms);
    printf("  Cycles: %u\n", st.cycles);
    printf("  r3=%llu  r4=%llu  r5=%llu  r6=%llu  r7=%llu\n",
           (unsigned long long)st.gpr[3], (unsigned long long)st.gpr[4],
           (unsigned long long)st.gpr[5], (unsigned long long)st.gpr[6],
           (unsigned long long)st.gpr[7]);
    printf("  Halted: %s\n", st.halted ? "YES" : "NO");

    bool pass = (st.gpr[5] == 100) && (st.gpr[7] == 300) && st.halted;
    printf("  Result: %s (r5=%llu, expected 100; r7=%llu, expected 300)\n",
           pass ? "✅ PASS" : "❌ FAIL",
           (unsigned long long)st.gpr[5], (unsigned long long)st.gpr[7]);

    megakernel_shutdown();
    return pass;
}

// ═══════════════════════════════════════════════════════════════
// Test 2: PPE Branch + Loop
// ═══════════════════════════════════════════════════════════════

static bool test_ppe_loop() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: PPE Branch + Loop            ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    megakernel_init();

    // Sum 1..10 in a loop:
    //   li r3, 0        # accumulator
    //   li r4, 1        # counter
    //   li r5, 11       # limit
    // loop:
    //   add r3, r3, r4  # acc += counter
    //   addi r4, r4, 1  # counter++
    //   cmpwi cr0, r4, r5  → actually use subf+branch pattern
    //   ...
    // For simplicity, use a counted approach with addi and branch:
    //   li r3, 0        # sum = 0
    //   li r4, 10       # count = 10
    // loop:
    //   add r3, r3, r4  # sum += count
    //   addi r4, r4, -1 # count--
    //   cmpi cr0, r4, 0
    //   bne loop (bc 4, 2, -12) → branch if CR0[EQ]=0

    // cmpi cr0, rA, SIMM = opcode 11, bf=0, rA, SIMM
    auto ppc_cmpi = [](int bf, int rA, int16_t imm) -> uint32_t {
        return bswap32((11u << 26) | ((bf & 7) << 23) | (rA << 16) | (uint16_t)imm);
    };
    // bc BO, BI, disp  (opcode 16)
    // bne = BO=4 (branch if condition FALSE), BI=2 (CR0[EQ])
    auto ppc_bc = [](int bo, int bi, int16_t disp) -> uint32_t {
        return bswap32((16u << 26) | (bo << 21) | (bi << 16) | ((uint16_t)disp & 0xFFFC));
    };

    uint32_t code[] = {
        ppc_addi(3, 0, 0),        // li r3, 0
        ppc_addi(4, 0, 10),       // li r4, 10
        // loop (offset +8 from start):
        ppc_add(3, 3, 4),         // add r3, r3, r4
        ppc_addi(4, 4, -1),       // addi r4, r4, -1
        ppc_cmpi(0, 4, 0),        // cmpi cr0, r4, 0
        ppc_bc(4, 2, -12),        // bne cr0, loop (back 3 instructions = -12 bytes)
        ppc_addi(11, 0, 1),       // li r11, 1
        ppc_sc(),                  // halt
    };

    uint64_t loadAddr = 0x10000;
    megakernel_load(loadAddr, code, sizeof(code));
    megakernel_set_entry(loadAddr, 0x80000, 0);

    float ms = megakernel_run(10000);

    PPEState st;
    megakernel_read_state(&st);

    // sum(1..10) = 55
    printf("  Execution time: %.3f ms\n", ms);
    printf("  Cycles: %u\n", st.cycles);
    printf("  r3 (sum) = %llu (expected 55)\n", (unsigned long long)st.gpr[3]);
    printf("  r4 (counter) = %llu (expected 0)\n", (unsigned long long)st.gpr[4]);

    bool pass = (st.gpr[3] == 55) && (st.gpr[4] == 0) && st.halted;
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");

    megakernel_shutdown();
    return pass;
}

// ═══════════════════════════════════════════════════════════════
// Test 3: SPU SIMD Integer
// ═══════════════════════════════════════════════════════════════

static bool test_spu_simd() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: SPU 128-bit SIMD Integer     ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    spu_init();

    // Program: load 10 into all 4 words of r3,
    //          load 20 into all 4 words of r4,
    //          add r5 = r3 + r4 (should be 30 in all 4 words)
    //          stop
    uint32_t code[] = {
        spu_il(3, 10),     // il r3, 10   → r3 = {10, 10, 10, 10}
        spu_il(4, 20),     // il r4, 20   → r4 = {20, 20, 20, 20}
        spu_a(5, 3, 4),    // a r5, r3, r4 → r5 = {30, 30, 30, 30}
        spu_il(6, 7),      // il r6, 7
        spu_sf(7, 6, 5),   // sf r7, r6, r5 → r7 = r5 - r6 = {23, 23, 23, 23}
        spu_stop(),
    };

    // Need to get d_mainMem pointer — for SPU standalone test, pass NULL
    spu_load_program(0, code, sizeof(code), 0);

    float ms = spu_run(1000, nullptr);

    SPUState st;
    spu_read_state(0, &st);

    printf("  Execution time: %.3f ms\n", ms);
    printf("  Cycles: %u\n", st.cycles);
    printf("  r3 = {%u, %u, %u, %u} (expected {10, 10, 10, 10})\n",
           st.gpr[3].u32[0], st.gpr[3].u32[1], st.gpr[3].u32[2], st.gpr[3].u32[3]);
    printf("  r5 = {%u, %u, %u, %u} (expected {30, 30, 30, 30})\n",
           st.gpr[5].u32[0], st.gpr[5].u32[1], st.gpr[5].u32[2], st.gpr[5].u32[3]);
    printf("  r7 = {%u, %u, %u, %u} (expected {23, 23, 23, 23})\n",
           st.gpr[7].u32[0], st.gpr[7].u32[1], st.gpr[7].u32[2], st.gpr[7].u32[3]);

    bool pass = (st.gpr[5].u32[0] == 30 && st.gpr[5].u32[1] == 30 &&
                 st.gpr[5].u32[2] == 30 && st.gpr[5].u32[3] == 30 &&
                 st.gpr[7].u32[0] == 23 && st.gpr[7].u32[1] == 23 &&
                 st.gpr[7].u32[2] == 23 && st.gpr[7].u32[3] == 23 &&
                 st.halted);
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");

    spu_shutdown();
    return pass;
}

// ═══════════════════════════════════════════════════════════════
// Test 4: SPU FP SIMD (Havok-style physics vector math)
// ═══════════════════════════════════════════════════════════════

static bool test_spu_fp() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 4: SPU Float4 SIMD (Havok-like) ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    spu_init();

    // We need to load float constants into SPU registers.
    // SPU loads floats via: ilhu + iohl to build bit pattern, or via LS load.
    // Simplest: write float data into LS, then lqd to load.
    //
    // Program layout in LS:
    //   0x000: code
    //   0x100: vec_a = {1.0, 2.0, 3.0, 4.0}
    //   0x110: vec_b = {5.0, 6.0, 7.0, 8.0}

    // Build LS image
    uint8_t ls_image[512];
    memset(ls_image, 0, sizeof(ls_image));

    // Code at 0x000
    uint32_t code[] = {
        // lqd rT, I10(rA) — op8=0x34, ea = rA + (I10 << 4)
        // lqd r3, 0x10(r0) → loads from 0x100 (I10=0x10, 0x10<<4=0x100)
        bswap32((0x34u << 24) | (0x10u << 14) | (0u << 7) | 3),
        // lqd r4, 0x11(r0) → loads from 0x110
        bswap32((0x34u << 24) | (0x11u << 14) | (0u << 7) | 4),
        spu_fa(5, 3, 4),    // fa r5, r3, r4 → r5 = vec_a + vec_b
        spu_fm(6, 3, 4),    // fm r6, r3, r4 → r6 = vec_a * vec_b
        spu_fma(7, 3, 4, 5),// fma r7, r3, r4, r5 → r7 = a*b + (a+b)
        spu_stop(),
    };
    memcpy(ls_image, code, sizeof(code));

    // Float data at 0x100 (stored in big-endian)
    float vec_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vec_b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    for (int i = 0; i < 4; i++) {
        uint32_t a_bits, b_bits;
        memcpy(&a_bits, &vec_a[i], 4);
        memcpy(&b_bits, &vec_b[i], 4);
        a_bits = bswap32(a_bits);
        b_bits = bswap32(b_bits);
        memcpy(&ls_image[0x100 + i*4], &a_bits, 4);
        memcpy(&ls_image[0x110 + i*4], &b_bits, 4);
    }

    spu_load_program(0, ls_image, sizeof(ls_image), 0);

    float ms = spu_run(1000, nullptr);

    SPUState st;
    spu_read_state(0, &st);

    printf("  Execution time: %.3f ms\n", ms);
    printf("  Cycles: %u\n", st.cycles);
    printf("  r3 (vec_a) = {%.1f, %.1f, %.1f, %.1f}\n",
           st.gpr[3].f32[0], st.gpr[3].f32[1], st.gpr[3].f32[2], st.gpr[3].f32[3]);
    printf("  r4 (vec_b) = {%.1f, %.1f, %.1f, %.1f}\n",
           st.gpr[4].f32[0], st.gpr[4].f32[1], st.gpr[4].f32[2], st.gpr[4].f32[3]);
    printf("  r5 (a+b)   = {%.1f, %.1f, %.1f, %.1f} (expected {6,8,10,12})\n",
           st.gpr[5].f32[0], st.gpr[5].f32[1], st.gpr[5].f32[2], st.gpr[5].f32[3]);
    printf("  r6 (a*b)   = {%.1f, %.1f, %.1f, %.1f} (expected {5,12,21,32})\n",
           st.gpr[6].f32[0], st.gpr[6].f32[1], st.gpr[6].f32[2], st.gpr[6].f32[3]);
    printf("  r7 (fma)   = {%.1f, %.1f, %.1f, %.1f} (expected {11,20,31,44})\n",
           st.gpr[7].f32[0], st.gpr[7].f32[1], st.gpr[7].f32[2], st.gpr[7].f32[3]);

    auto close = [](float a, float b) { return fabsf(a - b) < 0.01f; };
    bool pass = close(st.gpr[5].f32[0], 6.0f)  && close(st.gpr[5].f32[1], 8.0f) &&
                close(st.gpr[5].f32[2], 10.0f)  && close(st.gpr[5].f32[3], 12.0f) &&
                close(st.gpr[6].f32[0], 5.0f)  && close(st.gpr[6].f32[1], 12.0f) &&
                close(st.gpr[6].f32[2], 21.0f)  && close(st.gpr[6].f32[3], 32.0f) &&
                close(st.gpr[7].f32[0], 11.0f) && close(st.gpr[7].f32[1], 20.0f) &&
                close(st.gpr[7].f32[2], 31.0f) && close(st.gpr[7].f32[3], 44.0f);
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");

    spu_shutdown();
    return pass;
}

// ═══════════════════════════════════════════════════════════════
// Test 5: Cooperative Cell Megakernel (PPE + SPU together)
// ═══════════════════════════════════════════════════════════════

static bool test_cell_cooperative() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 5: Cell Cooperative Megakernel  ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    cell_init();

    // Simple PPE program: set up registers, halt
    uint32_t ppe_code[] = {
        ppc_addi(3, 0, 0x42),    // li r3, 0x42
        ppc_addi(4, 0, 0x43),    // li r4, 0x43
        ppc_add(5, 3, 4),        // add r5, r3, r4 → 0x85 = 133
        ppc_addi(11, 0, 1),      // SYS_PROCESS_EXIT
        ppc_sc(),
    };

    cell_load_ppe(0x10000, ppe_code, sizeof(ppe_code), 0x10000, 0x80000, 0);

    // Simple SPU program: compute and stop
    uint32_t spu_code[] = {
        spu_il(10, 100),
        spu_il(11, 200),
        spu_a(12, 10, 11),   // r12 = {300, 300, 300, 300}
        spu_stop(),
    };
    cell_load_spu(0, spu_code, sizeof(spu_code), 0);

    float ms = cell_run(10, 1000);

    PPEState pst;
    SPUState sst;
    cell_read_ppe(&pst);
    cell_read_spu(0, &sst);

    printf("  Cooperative kernel time: %.3f ms\n", ms);
    printf("  PPE: r3=0x%llx r5=0x%llx halted=%d cycles=%u\n",
           (unsigned long long)pst.gpr[3], (unsigned long long)pst.gpr[5],
           pst.halted, pst.cycles);
    printf("  SPU0: r12={%u,%u,%u,%u} halted=%d cycles=%u\n",
           sst.gpr[12].u32[0], sst.gpr[12].u32[1],
           sst.gpr[12].u32[2], sst.gpr[12].u32[3],
           sst.halted, sst.cycles);

    bool pass = (pst.gpr[5] == 133) && pst.halted;
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");

    cell_shutdown();
    return pass;
}

// ═══════════════════════════════════════════════════════════════
// Performance Benchmark: PPE throughput
// ═══════════════════════════════════════════════════════════════

static void bench_ppe_throughput() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  BENCH: PPE Instruction Throughput    ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    megakernel_init();

    // Tight loop: 1M iterations of add
    //   li r3, 0
    //   lis r4, 0x000F     → r4 = 0xF0000 = 983040
    //   ori r4, r4, 0x4240 → r4 = 1000000
    // loop:
    //   addi r3, r3, 1
    //   addi r5, r5, 1     (dummy work)
    //   addi r6, r6, 1     (dummy work)
    //   subf r7 = r4 - r3 via: cmpli + bne pattern
    //   ... simplified: just run N cycles of addi

    // Simple: li r3, 0 + 999998 × addi r3, r3, 1 + halt
    // We can't assemble 1M instructions, so we use a loop

    auto ppc_cmpi_fn = [](int bf, int rA, int16_t imm) -> uint32_t {
        return bswap32((11u << 26) | ((bf & 7) << 23) | (rA << 16) | (uint16_t)imm);
    };
    auto ppc_bc_fn = [](int bo, int bi, int16_t disp) -> uint32_t {
        return bswap32((16u << 26) | (bo << 21) | (bi << 16) | ((uint16_t)disp & 0xFFFC));
    };

    uint32_t code[] = {
        ppc_addi(3, 0, 0),          // r3 = 0 (counter)
        ppc_addis(4, 0, 0x000F),    // r4 = 0xF0000
        ppc_ori(4, 4, 0x4240),      // r4 = 0xF4240 = 1000000
        // loop:
        ppc_addi(3, 3, 1),          // r3++
        ppc_cmpi_fn(0, 3, 0),       // cmpi cr0, r3, r4 — but can't compare to reg with cmpi...
        // Use subf + cmpi 0 pattern:
        // Actually let's just run a fixed 100K cycle budget
        ppc_addi(5, 5, 1),          // dummy
        ppc_addi(6, 6, 1),          // dummy
        ppc_b(-16),                  // b loop (-4 instructions = -16 bytes)
    };

    uint64_t loadAddr = 0x10000;
    megakernel_load(loadAddr, code, sizeof(code));
    megakernel_set_entry(loadAddr, 0x80000, 0);

    // Run 1M cycles
    uint32_t cycles = 1000000;
    float ms = megakernel_run(cycles);

    PPEState st;
    megakernel_read_state(&st);

    float mips = (st.cycles / (ms / 1000.0f)) / 1e6f;
    float effective_mhz = mips; // 1 instruction per cycle ideally

    printf("  Cycles executed: %u\n", st.cycles);
    printf("  Wall time: %.3f ms\n", ms);
    printf("  Throughput: %.1f MIPS\n", mips);
    printf("  Effective clock: %.1f MHz\n", effective_mhz);
    printf("  vs PS3 PPE (3200 MHz): %.1f%%\n", (effective_mhz / 3200.0f) * 100.0f);

    megakernel_shutdown();
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  🎮 Project Megakernel — Cell BE Emulator    ║\n");
    printf("║  GPU: Tesla V100 · CUDA Compute 7.0          ║\n");
    printf("║  512MB PS3 Sandbox · PPE + 6×SPU on CUDA     ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    int passed = 0, total = 0;

    total++; if (test_ppe_alu())          passed++;
    total++; if (test_ppe_loop())         passed++;
    total++; if (test_spu_simd())         passed++;
    total++; if (test_spu_fp())           passed++;
    total++; if (test_cell_cooperative()) passed++;

    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d/%d tests passed\n", passed, total);
    printf("═══════════════════════════════════════════\n");

    bench_ppe_throughput();

    return (passed == total) ? 0 : 1;
}
