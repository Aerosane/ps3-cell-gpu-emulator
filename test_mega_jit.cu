// test_mega_jit.cu — Megakernel JIT tests
//
// Tests the persistent megakernel JIT: one compile, one launch, run to halt.
// Compares correctness and throughput vs interpreter.
//
#include "spu_defs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

using namespace spu;

// Forward declarations for mega JIT
struct MegaJITResult {
    void*    cuModule;
    void*    cuFunction;
    uint32_t numBlocks;
    uint32_t totalInsns;
    float    compileTimeMs;
};

extern "C" int   mega_jit_compile(const uint8_t* h_ls, uint32_t entryPC, MegaJITResult* result);
extern "C" float mega_jit_run(const MegaJITResult* jit, SPUState* h_state,
                              uint8_t* d_ls, uint8_t* d_mainMem, uint32_t maxCycles);
extern "C" void  mega_jit_free(MegaJITResult* jit);

// Interpreter API
extern "C" int   spu_init();
extern "C" int   spu_load_program(int spuId, const void* data, size_t size, uint32_t entryPC);
extern "C" float spu_run(uint32_t maxCycles, uint8_t* d_mainMem);
extern "C" int   spu_read_state(int spuId, SPUState* out);
extern "C" void  spu_shutdown();

// Helpers
static uint32_t bswap32_h(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

static uint32_t spu_ri16(uint32_t op9, uint32_t rT, int32_t i16) {
    return (op9 << 23) | ((i16 & 0xFFFF) << 7) | (rT & 0x7F);
}
static uint32_t spu_ri10(uint32_t op8, uint32_t rT, uint32_t rA, int32_t i10) {
    return (op8 << 24) | ((i10 & 0x3FF) << 14) | ((rA & 0x7F) << 7) | (rT & 0x7F);
}
static uint32_t spu_rr(uint32_t op11, uint32_t rT, uint32_t rA, uint32_t rB) {
    return (op11 << 21) | ((rT & 0x7F) << 14) | ((rB & 0x7F) << 7) | (rA & 0x7F);
}
static uint32_t spu_rrr(uint32_t op4, uint32_t rT, uint32_t rA, uint32_t rB, uint32_t rC) {
    return (op4 << 28) | ((rT & 0x7F) << 21) | ((rB & 0x7F) << 14) | ((rA & 0x7F) << 7) | (rC & 0x7F);
}

static int total_pass = 0, total_fail = 0;

// Store BE float in LS
static void store_be_float(uint8_t* ls, uint32_t addr, float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    bits = bswap32_h(bits);
    memcpy(ls + addr, &bits, 4);
}

// ═══════════════════════════════════════════════════════════════
// TEST 1: Simple ALU (il, a, stop)
// ═══════════════════════════════════════════════════════════════

static void test_simple_alu() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: MegaJIT Simple ALU            ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE];
    memset(ls, 0, sizeof(ls));

    uint32_t pc = 0;
    auto w = [&](uint32_t inst) { uint32_t be=bswap32_h(inst); memcpy(ls+pc,&be,4); pc+=4; };

    w(spu_ri16(0x081, 3, 42));   // il r3, 42
    w(spu_ri16(0x081, 4, 58));   // il r4, 58
    w(spu_rr(0x0C0, 5, 3, 4));   // a r5, r3, r4
    w(spu_rr(0x000, 0, 0, 0));   // stop

    MegaJITResult jit;
    if (!mega_jit_compile(ls, 0, &jit)) {
        printf("  ❌ FAIL (compile failed)\n");
        total_fail++; return;
    }

    // Setup device
    uint8_t* d_ls;
    cudaMalloc(&d_ls, SPU_LS_SIZE);
    cudaMemcpy(d_ls, ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);

    SPUState h_spu;
    memset(&h_spu, 0, sizeof(h_spu));
    h_spu.pc = 0;

    float ms = mega_jit_run(&jit, &h_spu, d_ls, nullptr, 100);

    printf("  r3=%u r4=%u r5=%u halted=%u cycles=%u (%.3f ms)\n",
           h_spu.gpr[3].u32[0], h_spu.gpr[4].u32[0], h_spu.gpr[5].u32[0],
           h_spu.halted, h_spu.cycles, ms);

    bool pass = h_spu.gpr[5].u32[0] == 100 && h_spu.halted;
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    cudaFree(d_ls);
    mega_jit_free(&jit);
}

// ═══════════════════════════════════════════════════════════════
// TEST 2: Float4 SIMD (fa, fm, fma)
// ═══════════════════════════════════════════════════════════════

static void test_float4_simd() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: MegaJIT Float4 SIMD           ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE];
    memset(ls, 0, sizeof(ls));

    for (int i = 0; i < 4; i++) {
        store_be_float(ls, 0x100 + i*4, (float)(i+1));
        store_be_float(ls, 0x110 + i*4, (float)(i+5));
    }

    uint32_t pc = 0;
    auto w = [&](uint32_t inst) { uint32_t be=bswap32_h(inst); memcpy(ls+pc,&be,4); pc+=4; };

    w(spu_ri10(0x34, 3, 0, 0x10));  // lqd r3, 0x100
    w(spu_ri10(0x34, 4, 0, 0x11));  // lqd r4, 0x110
    w(spu_rr(0x2C4, 5, 3, 4));      // fa r5, r3, r4
    w(spu_rr(0x2C6, 6, 3, 4));      // fm r6, r3, r4
    w(spu_rrr(0xE, 7, 3, 4, 5));    // fma r7, r3, r4, r5
    w(spu_rr(0x000, 0, 0, 0));      // stop

    MegaJITResult jit;
    if (!mega_jit_compile(ls, 0, &jit)) {
        printf("  ❌ FAIL (compile failed)\n");
        total_fail++; return;
    }

    uint8_t* d_ls;
    cudaMalloc(&d_ls, SPU_LS_SIZE);
    cudaMemcpy(d_ls, ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);

    SPUState h_spu;
    memset(&h_spu, 0, sizeof(h_spu));

    float ms = mega_jit_run(&jit, &h_spu, d_ls, nullptr, 100);

    printf("  r5 (a+b) = {%.1f, %.1f, %.1f, %.1f} (expected {6,8,10,12})\n",
           h_spu.gpr[5].f32[0], h_spu.gpr[5].f32[1],
           h_spu.gpr[5].f32[2], h_spu.gpr[5].f32[3]);
    printf("  r6 (a*b) = {%.1f, %.1f, %.1f, %.1f} (expected {5,12,21,32})\n",
           h_spu.gpr[6].f32[0], h_spu.gpr[6].f32[1],
           h_spu.gpr[6].f32[2], h_spu.gpr[6].f32[3]);
    printf("  r7 (fma) = {%.1f, %.1f, %.1f, %.1f} (expected {11,20,31,44})\n",
           h_spu.gpr[7].f32[0], h_spu.gpr[7].f32[1],
           h_spu.gpr[7].f32[2], h_spu.gpr[7].f32[3]);

    bool pass =
        h_spu.gpr[5].f32[0] == 6.0f && h_spu.gpr[5].f32[3] == 12.0f &&
        h_spu.gpr[6].f32[0] == 5.0f && h_spu.gpr[6].f32[3] == 32.0f &&
        h_spu.gpr[7].f32[0] == 11.0f && h_spu.gpr[7].f32[3] == 44.0f;

    printf("  Execution: %.3f ms\n", ms);
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    cudaFree(d_ls);
    mega_jit_free(&jit);
}

// ═══════════════════════════════════════════════════════════════
// TEST 3: Loop (branch back, counter decrement)
// ═══════════════════════════════════════════════════════════════

static void test_loop() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: MegaJIT Loop (sum 1..10)      ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE];
    memset(ls, 0, sizeof(ls));

    uint32_t pc = 0;
    auto w = [&](uint32_t inst) { uint32_t be=bswap32_h(inst); memcpy(ls+pc,&be,4); pc+=4; };

    // r3 = sum (0), r4 = counter (10), r5 = 1
    w(spu_ri16(0x081, 3, 0));       // il r3, 0       (sum)
    w(spu_ri16(0x081, 4, 10));      // il r4, 10      (counter)
    w(spu_ri16(0x081, 5, 1));       // il r5, 1       (constant 1)
    // loop: (PC = 0x0C)
    w(spu_rr(0x0C0, 3, 3, 4));     // a r3, r3, r4   (sum += counter)
    w(spu_rr(0x040, 4, 5, 4));     // sf r4, r5, r4  (counter = counter - 1)
    w(spu_ri16(0x042, 4, -3));     // brnz r4, -3    (PC-12 → back to loop)
    w(spu_rr(0x000, 0, 0, 0));     // stop

    MegaJITResult jit;
    if (!mega_jit_compile(ls, 0, &jit)) {
        printf("  ❌ FAIL (compile failed)\n");
        total_fail++; return;
    }

    uint8_t* d_ls;
    cudaMalloc(&d_ls, SPU_LS_SIZE);
    cudaMemcpy(d_ls, ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);

    SPUState h_spu;
    memset(&h_spu, 0, sizeof(h_spu));

    float ms = mega_jit_run(&jit, &h_spu, d_ls, nullptr, 10000);

    printf("  r3 (sum) = %d (expected 55)\n", h_spu.gpr[3].s32[0]);
    printf("  r4 (ctr) = %d (expected 0)\n", h_spu.gpr[4].s32[0]);
    printf("  cycles = %u, time = %.3f ms\n", h_spu.cycles, ms);

    bool pass = h_spu.gpr[3].s32[0] == 55 && h_spu.gpr[4].s32[0] == 0 && h_spu.halted;
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    cudaFree(d_ls);
    mega_jit_free(&jit);
}

// ═══════════════════════════════════════════════════════════════
// TEST 4: Big Benchmark — JIT vs Interpreter
// ═══════════════════════════════════════════════════════════════

static void test_benchmark() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 4: MegaJIT vs Interpreter Bench  ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE];
    memset(ls, 0, sizeof(ls));

    // Store initial vectors for a Havok-like workload
    for (int i = 0; i < 4; i++) {
        store_be_float(ls, 0x200 + i*4, 1.0f);
        store_be_float(ls, 0x210 + i*4, 0.01f);
    }

    uint32_t pc = 0;
    auto w = [&](uint32_t inst) { uint32_t be=bswap32_h(inst); memcpy(ls+pc,&be,4); pc+=4; };

    // Load vectors
    w(spu_ri10(0x34, 3, 0, 0x20));  // lqd r3, 0x200 (accumulator)
    w(spu_ri10(0x34, 4, 0, 0x21));  // lqd r4, 0x210 (increment)
    w(spu_ri16(0x081, 10, 10000));   // il r10, 10000 (counter)
    w(spu_ri16(0x081, 11, 1));       // il r11, 1
    // loop: (PC = 0x10)
    w(spu_rr(0x2C4, 3, 3, 4));      // fa r3, r3, r4
    w(spu_rr(0x2C6, 5, 3, 3));      // fm r5, r3, r3 (r3 squared)
    w(spu_rrr(0xE, 6, 3, 4, 5));    // fma r6, r3, r4, r5
    w(spu_rr(0x040, 10, 11, 10));   // sf r10, r11, r10
    w(spu_ri16(0x042, 10, -4));     // brnz r10, -4 (→ loop)
    w(spu_rr(0x000, 0, 0, 0));      // stop

    uint32_t progSize = pc;

    // === Megakernel JIT ===
    MegaJITResult jit;
    if (!mega_jit_compile(ls, 0, &jit)) {
        printf("  ❌ FAIL (compile failed)\n");
        total_fail++; return;
    }

    uint8_t* d_ls;
    cudaMalloc(&d_ls, SPU_LS_SIZE);
    cudaMemcpy(d_ls, ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);

    SPUState h_spu;
    memset(&h_spu, 0, sizeof(h_spu));

    // Warm up
    mega_jit_run(&jit, &h_spu, d_ls, nullptr, 200000);

    // Benchmark: run 10K iterations of the loop (= 54 insns per iteration × 10K = 540K total)
    memset(&h_spu, 0, sizeof(h_spu));
    cudaMemcpy(d_ls, ls, SPU_LS_SIZE, cudaMemcpyHostToDevice); // reload LS

    float jitMs = mega_jit_run(&jit, &h_spu, d_ls, nullptr, 200000);
    uint32_t jitCycles = h_spu.cycles;
    float jitMips = (float)jitCycles / (jitMs * 1000.0f);

    printf("  JIT: %u cycles in %.3f ms = %.1f MIPS\n", jitCycles, jitMs, jitMips);
    printf("  JIT r3[0] = %.6f\n", h_spu.gpr[3].f32[0]);

    // === Interpreter ===
    spu_init();
    spu_load_program(0, ls, progSize, 0);

    cudaEvent_t intStart, intStop;
    cudaEventCreate(&intStart);
    cudaEventCreate(&intStop);
    cudaEventRecord(intStart);
    spu_run(200000, nullptr);
    cudaEventRecord(intStop);
    cudaEventSynchronize(intStop);

    float intMs = 0;
    cudaEventElapsedTime(&intMs, intStart, intStop);

    SPUState intState;
    spu_read_state(0, &intState);
    float intMips = (float)intState.cycles / (intMs * 1000.0f);

    printf("  Interpreter: %u cycles in %.3f ms = %.1f MIPS\n",
           intState.cycles, intMs, intMips);
    printf("  Interp r3[0] = %.6f\n", intState.gpr[3].f32[0]);

    float speedup = jitMips / (intMips > 0 ? intMips : 0.001f);

    printf("\n  ┌──────────────────────────────────────────┐\n");
    printf("  │ MegaJIT:     %8.1f MIPS  (%7.3f ms)  │\n", jitMips, jitMs);
    printf("  │ Interpreter: %8.1f MIPS  (%7.3f ms)  │\n", intMips, intMs);
    printf("  │ Speedup:     %8.1f×                    │\n", speedup);
    printf("  │ Compile:     %8.1f ms (one-time)       │\n", jit.compileTimeMs);
    printf("  └──────────────────────────────────────────┘\n");

    bool pass = h_spu.halted && speedup > 1.5f;
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    spu_shutdown();
    cudaEventDestroy(intStart);
    cudaEventDestroy(intStop);
    cudaFree(d_ls);
    mega_jit_free(&jit);
}

// ═══════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  🚀 SPU Megakernel JIT Tests                 ║\n");
    printf("║  ONE compile, ONE launch, run to halt        ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s · Compute %d.%d\n", prop.name, prop.major, prop.minor);

    cuInit(0);

    test_simple_alu();
    test_float4_simd();
    test_loop();
    test_benchmark();

    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d/%d tests passed\n",
           total_pass, total_pass + total_fail);
    printf("═══════════════════════════════════════════\n");

    return total_fail > 0 ? 1 : 0;
}
