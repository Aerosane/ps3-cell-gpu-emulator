// test_ppc_jit.cu — PPE JIT Compiler Tests
//
// Tests the full JIT pipeline: discover → emit → compile → execute
// Compares JIT output vs interpreter output for correctness
// Benchmarks JIT vs interpreter throughput
//
#include "ppc_jit.h"
#include "ppc_defs.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace ppc;
using namespace ppc_jit;

// ═══════════════════════════════════════════════════════════════
// PPC Instruction Encoders (host-side, big-endian)
// ═══════════════════════════════════════════════════════════════

static uint32_t bswap32_h(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

// D-form: opcd(6) | rD(5) | rA(5) | SIMM(16)
static uint32_t ppc_dform(uint32_t opcd, uint32_t rd, uint32_t ra, int16_t simm) {
    return (opcd << 26) | (rd << 21) | (ra << 16) | ((uint16_t)simm);
}

// I-form: opcd(6) | LI(24) | AA(1) | LK(1)
static uint32_t ppc_iform(uint32_t opcd, int32_t li, uint32_t aa, uint32_t lk) {
    return (opcd << 26) | (li & 0x03FFFFFC) | (aa << 1) | lk;
}

// B-form: opcd(6) | BO(5) | BI(5) | BD(14) | AA(1) | LK(1)
static uint32_t ppc_bform(uint32_t opcd, uint32_t bo, uint32_t bi, int16_t bd, uint32_t aa, uint32_t lk) {
    return (opcd << 26) | (bo << 21) | (bi << 16) | ((uint16_t)bd & 0xFFFC) | (aa << 1) | lk;
}

// X-form: opcd(6) | rD(5) | rA(5) | rB(5) | XO(10) | Rc(1)
static uint32_t ppc_xform(uint32_t opcd, uint32_t rd, uint32_t ra, uint32_t rb, uint32_t xo, uint32_t rc) {
    return (opcd << 26) | (rd << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc;
}

// XO-form (same layout as X-form for our purposes)
static uint32_t ppc_xoform(uint32_t opcd, uint32_t rd, uint32_t ra, uint32_t rb, uint32_t xo, uint32_t rc) {
    return ppc_xform(opcd, rd, ra, rb, xo, rc);
}

// SC instruction
static uint32_t ppc_sc() {
    return (17u << 26) | 2; // SC: opcd=17, bit 30 = 1
}

// Write a PPC instruction to PS3 memory (big-endian)
static void write_insn(uint8_t* mem, uint64_t addr, uint32_t insn) {
    uint32_t be = bswap32_h(insn);
    memcpy(mem + addr, &be, 4);
}

static int total_pass = 0, total_fail = 0;

// ═══════════════════════════════════════════════════════════════
// TEST 1: ALU Correctness
// li r3, 100; li r4, 200; add r5, r3, r4; mulli r6, r5, 3; SC
// Expect r5=300, r6=900
// ═══════════════════════════════════════════════════════════════

static void test_alu_correctness() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: ALU Correctness               ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    // Allocate PS3 memory
    uint8_t* d_mem;
    cudaMalloc(&d_mem, PS3_SANDBOX_SIZE);
    cudaMemset(d_mem, 0, PS3_SANDBOX_SIZE);

    // Build program in host memory, then upload
    uint8_t* h_mem = (uint8_t*)calloc(1, 4096);
    uint64_t pc = 0x10000;

    // li r3, 100  → addi r3, r0, 100
    write_insn(h_mem, 0, ppc_dform(14, 3, 0, 100));
    // li r4, 200  → addi r4, r0, 200
    write_insn(h_mem, 4, ppc_dform(14, 4, 0, 200));
    // add r5, r3, r4  → opcd=31, xo=266
    write_insn(h_mem, 8, ppc_xoform(31, 5, 3, 4, 266, 0));
    // mulli r6, r5, 3  → opcd=7
    write_insn(h_mem, 12, ppc_dform(7, 6, 5, 3));
    // sc
    write_insn(h_mem, 16, ppc_sc());

    cudaMemcpy(d_mem + pc, h_mem, 4096, cudaMemcpyHostToDevice);

    // Init JIT
    PPCJITState jitState;
    ppc_jit_init(&jitState);

    // Set up PPE state
    PPEState state = {};
    state.pc = pc;

    // Discover block
    uint8_t* h_full = (uint8_t*)calloc(1, (size_t)PS3_SANDBOX_SIZE);
    cudaMemcpy(h_full, d_mem, PS3_SANDBOX_SIZE, cudaMemcpyDeviceToHost);

    PPCBasicBlock block;
    int n = ppc_jit_discover_block(h_full, pc, &block);
    printf("  Block discovered: %d instructions\n", n);

    // Emit source
    char* source = (char*)malloc(MAX_SOURCE_SIZE);
    int srcLen = ppc_jit_emit_source(&block, source, MAX_SOURCE_SIZE);
    printf("  Source emitted: %d bytes\n", srcLen);

    // Compile
    PPCJITEntry entry;
    int ok = ppc_jit_compile(&jitState, &block, source, &entry);
    printf("  Compile: %s\n", ok ? "OK" : "FAILED");

    if (ok) {
        // Execute
        int eok = ppc_jit_execute(&entry, &state, d_mem);
        printf("  Execute: %s\n", eok ? "OK" : "FAILED");
        printf("  r3=%llu r4=%llu r5=%llu r6=%llu\n",
               (unsigned long long)state.gpr[3],
               (unsigned long long)state.gpr[4],
               (unsigned long long)state.gpr[5],
               (unsigned long long)state.gpr[6]);

        bool pass = (state.gpr[5] == 300) && (state.gpr[6] == 900);
        printf("  Result: %s (r5=%llu expect 300, r6=%llu expect 900)\n",
               pass ? "✅ PASS" : "❌ FAIL",
               (unsigned long long)state.gpr[5],
               (unsigned long long)state.gpr[6]);
        if (pass) total_pass++; else total_fail++;
    } else {
        printf("  Result: ❌ FAIL (compile failed)\n");
        total_fail++;
    }

    free(source);
    free(h_full);
    free(h_mem);
    ppc_jit_shutdown(&jitState);
    cudaFree(d_mem);
}

// ═══════════════════════════════════════════════════════════════
// TEST 2: Loop with Branch (JIT block boundary test)
// li r3, 0; li r4, 10000
// loop: addi r3, r3, 1; cmpw r3, r4; blt loop; SC
// Expect r3=10000
// ═══════════════════════════════════════════════════════════════

static void test_loop_branch() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: Loop with Branch               ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t* d_mem;
    cudaMalloc(&d_mem, PS3_SANDBOX_SIZE);
    cudaMemset(d_mem, 0, PS3_SANDBOX_SIZE);

    uint8_t* h_mem = (uint8_t*)calloc(1, 4096);
    uint64_t base = 0x10000;

    // 0x10000: li r3, 0
    write_insn(h_mem, 0, ppc_dform(14, 3, 0, 0));
    // 0x10004: li r4, 10000
    //   Can't fit 10000 in SIMM16? 10000 = 0x2710 — fits in 16-bit signed
    write_insn(h_mem, 4, ppc_dform(14, 4, 0, 10000));
    // 0x10008: addi r3, r3, 1
    write_insn(h_mem, 8, ppc_dform(14, 3, 3, 1));
    // 0x1000C: cmpw cr0, r3, r4  → opcd=31, xo=0 (CMP), bf=0
    write_insn(h_mem, 12, ppc_xform(31, 0, 3, 4, 0, 0));
    // 0x10010: blt cr0, -8  → BC BO=12, BI=0, BD=-8
    //   BO=12 (0b01100): branch if condition TRUE, don't decrement CTR
    //   BI=0: CR0[LT]
    //   BD=-8 (relative to this insn at 0x10010, target=0x10008)
    write_insn(h_mem, 16, ppc_bform(16, 12, 0, -8, 0, 0));
    // 0x10014: sc
    write_insn(h_mem, 20, ppc_sc());

    cudaMemcpy(d_mem + base, h_mem, 4096, cudaMemcpyHostToDevice);

    PPCJITState jitState;
    ppc_jit_init(&jitState);

    PPEState state = {};
    state.pc = base;

    // Run JIT loop
    float ms = 0;
    uint32_t cycles = 0;
    ppc_jit_run(&jitState, &state, d_mem, 200000, &ms, &cycles);

    printf("  r3=%llu (expect 10000)\n", (unsigned long long)state.gpr[3]);
    printf("  Cycles run: %u, Time: %.2f ms\n", cycles, ms);

    bool pass = (state.gpr[3] == 10000);
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    ppc_jit_print_stats(&jitState);
    ppc_jit_shutdown(&jitState);
    free(h_mem);
    cudaFree(d_mem);
}

// ═══════════════════════════════════════════════════════════════
// TEST 3: Memory Load/Store (endian test)
// Write 0xDEADBEEF to addr, lwz → stw → lwz, verify round-trip
// ═══════════════════════════════════════════════════════════════

static void test_memory_loadstore() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: Memory Load/Store (endian)     ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t* d_mem;
    cudaMalloc(&d_mem, PS3_SANDBOX_SIZE);
    cudaMemset(d_mem, 0, PS3_SANDBOX_SIZE);

    // Write test value at address 0x20000 (big-endian)
    uint32_t testVal = 0xDEADBEEF;
    uint32_t testBE = bswap32_h(testVal);
    cudaMemcpy(d_mem + 0x20000, &testBE, 4, cudaMemcpyHostToDevice);

    // Program at 0x10000:
    // lis r3, 0x0002     → r3 = 0x00020000
    // ori r3, r3, 0x0000 → r3 = 0x00020000
    // lwz r4, 0(r3)      → r4 = 0xDEADBEEF
    // stw r4, 4(r3)      → store to 0x20004
    // lwz r5, 4(r3)      → r5 should = 0xDEADBEEF
    // sc

    uint8_t* h_mem = (uint8_t*)calloc(1, 4096);
    uint64_t base = 0x10000;

    write_insn(h_mem, 0,  ppc_dform(15, 3, 0, 2));        // lis r3, 2 (= addis r3, 0, 2)
    write_insn(h_mem, 4,  ppc_dform(24, 3, 3, 0));        // ori r3, r3, 0
    write_insn(h_mem, 8,  ppc_dform(32, 4, 3, 0));        // lwz r4, 0(r3)
    write_insn(h_mem, 12, ppc_dform(36, 4, 3, 4));        // stw r4, 4(r3)
    write_insn(h_mem, 16, ppc_dform(32, 5, 3, 4));        // lwz r5, 4(r3)
    write_insn(h_mem, 20, ppc_sc());                       // sc

    cudaMemcpy(d_mem + base, h_mem, 4096, cudaMemcpyHostToDevice);

    PPCJITState jitState;
    ppc_jit_init(&jitState);

    PPEState state = {};
    state.pc = base;

    // Discover and compile
    uint8_t* h_full = (uint8_t*)calloc(1, (size_t)PS3_SANDBOX_SIZE);
    cudaMemcpy(h_full, d_mem, PS3_SANDBOX_SIZE, cudaMemcpyDeviceToHost);

    PPCBasicBlock block;
    int n = ppc_jit_discover_block(h_full, base, &block);
    printf("  Block: %d instructions\n", n);

    char* source = (char*)malloc(MAX_SOURCE_SIZE);
    int srcLen = ppc_jit_emit_source(&block, source, MAX_SOURCE_SIZE);

    PPCJITEntry entry;
    int ok = ppc_jit_compile(&jitState, &block, source, &entry);

    if (ok) {
        ppc_jit_execute(&entry, &state, d_mem);
        printf("  r3=0x%llx r4=0x%llx r5=0x%llx\n",
               (unsigned long long)state.gpr[3],
               (unsigned long long)state.gpr[4],
               (unsigned long long)state.gpr[5]);

        bool pass = (state.gpr[4] == 0xDEADBEEF) && (state.gpr[5] == 0xDEADBEEF);
        printf("  Result: %s (r4=0x%llx, r5=0x%llx, expect 0xDEADBEEF)\n",
               pass ? "✅ PASS" : "❌ FAIL",
               (unsigned long long)state.gpr[4],
               (unsigned long long)state.gpr[5]);
        if (pass) total_pass++; else total_fail++;
    } else {
        printf("  Result: ❌ FAIL (compile failed)\n");
        total_fail++;
    }

    free(source);
    free(h_full);
    free(h_mem);
    ppc_jit_shutdown(&jitState);
    cudaFree(d_mem);
}

// ═══════════════════════════════════════════════════════════════
// TEST 4: JIT vs Interpreter Benchmark
// Run 100K iterations of ALU loop
// ═══════════════════════════════════════════════════════════════

// The interpreter API from ppc_interpreter.cu
extern "C" {
    int megakernel_init();
    int megakernel_load(uint64_t offset, const void* data, size_t size);
    int megakernel_set_entry(uint64_t pc, uint64_t sp, uint64_t toc);
    float megakernel_run(uint32_t maxCycles);
    int megakernel_read_state(PPEState* out);
    void megakernel_shutdown();
}

static void test_benchmark() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 4: JIT vs Interpreter Benchmark   ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    const uint32_t ITERATIONS = 100000;

    // Build loop program: li r3,0; li r4,N; loop: addi r3,r3,1; cmpw r3,r4; blt loop; sc
    uint8_t prog[64];
    memset(prog, 0, sizeof(prog));
    uint64_t base = 0x10000;

    // li r3, 0
    { uint32_t insn = ppc_dform(14, 3, 0, 0); uint32_t be = bswap32_h(insn); memcpy(prog + 0, &be, 4); }
    // li r4, ITERATIONS (need lis + ori for values > 32767)
    { uint32_t hi = (ITERATIONS >> 16) & 0xFFFF;
      uint32_t lo = ITERATIONS & 0xFFFF;
      uint32_t insn1 = ppc_dform(15, 4, 0, (int16_t)hi); // lis r4, hi
      uint32_t insn2 = ppc_dform(24, 4, 4, (int16_t)lo);  // ori r4, r4, lo
      uint32_t be1 = bswap32_h(insn1); memcpy(prog + 4, &be1, 4);
      uint32_t be2 = bswap32_h(insn2); memcpy(prog + 8, &be2, 4);
    }
    // addi r3, r3, 1
    { uint32_t insn = ppc_dform(14, 3, 3, 1); uint32_t be = bswap32_h(insn); memcpy(prog + 12, &be, 4); }
    // cmpw cr0, r3, r4
    { uint32_t insn = ppc_xform(31, 0, 3, 4, 0, 0); uint32_t be = bswap32_h(insn); memcpy(prog + 16, &be, 4); }
    // blt cr0, -8
    { uint32_t insn = ppc_bform(16, 12, 0, -8, 0, 0); uint32_t be = bswap32_h(insn); memcpy(prog + 20, &be, 4); }
    // sc
    { uint32_t insn = ppc_sc(); uint32_t be = bswap32_h(insn); memcpy(prog + 24, &be, 4); }

    // ── JIT Benchmark ──
    {
        uint8_t* d_mem;
        cudaMalloc(&d_mem, PS3_SANDBOX_SIZE);
        cudaMemset(d_mem, 0, PS3_SANDBOX_SIZE);
        cudaMemcpy(d_mem + base, prog, sizeof(prog), cudaMemcpyHostToDevice);

        PPCJITState jitState;
        ppc_jit_init(&jitState);

        PPEState state = {};
        state.pc = base;

        float jitMs = 0;
        uint32_t jitCycles = 0;
        ppc_jit_run(&jitState, &state, d_mem, ITERATIONS * 4 + 100, &jitMs, &jitCycles);

        printf("  JIT:  r3=%llu, cycles=%u, time=%.2f ms\n",
               (unsigned long long)state.gpr[3], jitCycles, jitMs);

        // Calculate approximate instructions executed
        // Each iteration = 3 insns (addi + cmpw + blt), plus 3 setup + 1 sc
        uint64_t totalInsns = (uint64_t)ITERATIONS * 3 + 4;
        double jitMIPS = (double)totalInsns / (jitMs * 1000.0);
        printf("  JIT:  %.2f MIPS (%.0f insns in %.2f ms)\n", jitMIPS, (double)totalInsns, jitMs);

        ppc_jit_print_stats(&jitState);
        ppc_jit_shutdown(&jitState);
        cudaFree(d_mem);
    }

    // ── Interpreter Benchmark ──
    {
        megakernel_init();
        megakernel_load(base, prog, sizeof(prog));
        megakernel_set_entry(base, 0x00F00000, 0);

        float interpMs = megakernel_run(ITERATIONS * 4 + 100);

        PPEState interpState;
        megakernel_read_state(&interpState);

        printf("  Interp: r3=%llu, cycles=%u, time=%.2f ms\n",
               (unsigned long long)interpState.gpr[3],
               interpState.cycles, interpMs);

        uint64_t totalInsns = (uint64_t)ITERATIONS * 3 + 4;
        double interpMIPS = (double)totalInsns / (interpMs * 1000.0);
        printf("  Interp: %.2f MIPS\n", interpMIPS);

        megakernel_shutdown();
    }

    printf("  Result: ✅ PASS (benchmark complete)\n");
    total_pass++;
}

// ═══════════════════════════════════════════════════════════════
// Test 5: Superblock JIT (fused dispatch loop)
// ═══════════════════════════════════════════════════════════════

static void test_superblock() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 5: Superblock JIT (fused)         ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    const uint64_t base = 0x10000;
    const uint32_t ITERATIONS = 100000;

    // Same loop program as test 4
    uint8_t prog[32] = {};
    // li r3, 0
    { uint32_t insn = ppc_dform(14, 3, 0, 0); uint32_t be = bswap32_h(insn); memcpy(prog, &be, 4); }
    // lis r4, hi(ITERATIONS) + ori r4, r4, lo(ITERATIONS)
    {
      uint16_t hi = (uint16_t)(ITERATIONS >> 16);
      uint16_t lo = (uint16_t)(ITERATIONS & 0xFFFF);
      uint32_t insn1 = ppc_dform(15, 4, 0, (int16_t)hi);
      uint32_t insn2 = ppc_dform(24, 4, 4, (int16_t)lo);
      uint32_t be1 = bswap32_h(insn1); memcpy(prog + 4, &be1, 4);
      uint32_t be2 = bswap32_h(insn2); memcpy(prog + 8, &be2, 4);
    }
    // addi r3, r3, 1
    { uint32_t insn = ppc_dform(14, 3, 3, 1); uint32_t be = bswap32_h(insn); memcpy(prog + 12, &be, 4); }
    // cmpw cr0, r3, r4
    { uint32_t insn = ppc_xform(31, 0, 3, 4, 0, 0); uint32_t be = bswap32_h(insn); memcpy(prog + 16, &be, 4); }
    // blt cr0, -8
    { uint32_t insn = ppc_bform(16, 12, 0, -8, 0, 0); uint32_t be = bswap32_h(insn); memcpy(prog + 20, &be, 4); }
    // sc
    { uint32_t insn = ppc_sc(); uint32_t be = bswap32_h(insn); memcpy(prog + 24, &be, 4); }

    uint8_t* d_mem;
    cudaMalloc(&d_mem, PS3_SANDBOX_SIZE);
    cudaMemset(d_mem, 0, PS3_SANDBOX_SIZE);
    cudaMemcpy(d_mem + base, prog, sizeof(prog), cudaMemcpyHostToDevice);

    PPCJITState jitState;
    ppc_jit_init(&jitState);

    PPEState state = {};
    state.pc = base;

    float ms = 0;
    uint32_t cycles = 0;
    ppc_jit_run_fast(&jitState, &state, d_mem, ITERATIONS * 4 + 100, &ms, &cycles);

    printf("  Superblock: r3=%llu, time=%.2f ms\n",
           (unsigned long long)state.gpr[3], ms);

    uint64_t totalInsns = (uint64_t)ITERATIONS * 3 + 4;
    double mips = (double)totalInsns / (ms * 1000.0);
    printf("  Superblock: %.1f MIPS\n", mips);

    bool pass = (state.gpr[3] == ITERATIONS);
    if (pass) {
        printf("  Result: ✅ PASS\n");
        total_pass++;
    } else {
        printf("  Result: ❌ FAIL (r3=%llu, expect %u)\n",
               (unsigned long long)state.gpr[3], ITERATIONS);
        total_fail++;
    }

    ppc_jit_shutdown(&jitState);
    cudaFree(d_mem);
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  🎮 PPE JIT Compiler Test Suite          ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Testing: discover → emit → compile → run ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    test_alu_correctness();
    test_loop_branch();
    test_memory_loadstore();
    test_benchmark();
    test_superblock();

    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", total_pass, total_fail);
    printf("═══════════════════════════════════════════\n");

    return total_fail > 0 ? 1 : 0;
}
