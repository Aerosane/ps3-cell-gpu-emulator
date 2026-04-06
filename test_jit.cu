// test_jit.cu — SPU JIT Compiler Tests
//
// Tests the full JIT pipeline: discover → emit → compile → execute
// Compares JIT output vs interpreter output for correctness
// Benchmarks JIT vs interpreter throughput
//
#include "spu_jit.h"
#include "spu_defs.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace spu;
using namespace spu_jit;

// Mini-assembler helpers (same as test_cell.cu)
static uint32_t bswap32_h(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

// SPU instruction encoders
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
static uint32_t spu_ri18(uint32_t op7, uint32_t rT, uint32_t i18) {
    return (op7 << 25) | ((i18 & 0x3FFFF) << 7) | (rT & 0x7F);
}

static int total_pass = 0, total_fail = 0;

// ═══════════════════════════════════════════════════════════════
// TEST 1: Block Discovery
// ═══════════════════════════════════════════════════════════════

static void test_block_discovery() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: Basic Block Discovery         ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    // Build a simple program in host-side LS
    uint8_t ls[SPU_LS_SIZE];
    memset(ls, 0, sizeof(ls));

    // Program: il r3,42 ; il r4,58 ; a r5,r3,r4 ; stop
    uint32_t prog[] = {
        bswap32_h(spu_ri16(0x081, 3, 42)),   // il r3, 42
        bswap32_h(spu_ri16(0x081, 4, 58)),   // il r4, 58
        bswap32_h(spu_rr(0x0C0, 5, 3, 4)),   // a r5, r3, r4
        bswap32_h(spu_rr(0x000, 0, 0, 0)),   // stop
    };
    memcpy(ls, prog, sizeof(prog));

    BasicBlock block;
    int n = jit_discover_block(ls, 0, &block);

    printf("  Block @ 0x%04x: %d instructions\n", block.entryPC, n);
    printf("  End PC: 0x%04x\n", block.endPC);
    for (uint32_t i = 0; i < block.numInsns; i++) {
        printf("  [%d] pc=0x%04x fmt=%d op=0x%03x rT=%d rA=%d rB=%d",
               i, block.insns[i].pc, block.insns[i].format,
               block.insns[i].opcode, block.insns[i].rT,
               block.insns[i].rA, block.insns[i].rB);
        if (block.insns[i].isBranch) printf(" BRANCH");
        if (block.insns[i].isStop) printf(" STOP");
        printf("\n");
    }

    bool pass = (n == 4) && block.insns[3].isStop;
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;
}

// ═══════════════════════════════════════════════════════════════
// TEST 2: Source Emission
// ═══════════════════════════════════════════════════════════════

static void test_source_emission() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: CUDA Source Emission          ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE];
    memset(ls, 0, sizeof(ls));

    uint32_t prog[] = {
        bswap32_h(spu_ri16(0x081, 3, 42)),   // il r3, 42
        bswap32_h(spu_ri16(0x081, 4, 58)),   // il r4, 58
        bswap32_h(spu_rr(0x0C0, 5, 3, 4)),   // a r5, r3, r4
        bswap32_h(spu_rr(0x000, 0, 0, 0)),   // stop
    };
    memcpy(ls, prog, sizeof(prog));

    BasicBlock block;
    jit_discover_block(ls, 0, &block);

    char source[MAX_SOURCE_SIZE];
    int srcLen = jit_emit_source(&block, source, MAX_SOURCE_SIZE);

    printf("  Generated %d bytes of CUDA C++ source\n", srcLen);
    printf("  --- Source preview (first 500 chars) ---\n");
    printf("%.500s\n", source);
    printf("  --- End preview ---\n");

    bool pass = (srcLen > 100) &&
                strstr(source, "jit_block_0x0") != nullptr &&
                strstr(source, "s.gpr[5]") != nullptr;
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;
}

// ═══════════════════════════════════════════════════════════════
// TEST 3: NVRTC Compile + Execute (Integer ALU)
// ═══════════════════════════════════════════════════════════════

static void test_jit_compile_execute() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: JIT Compile + Execute (ALU)   ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    JITState jitState;
    if (!jit_init(&jitState)) {
        printf("  ❌ FAIL (JIT init failed)\n");
        total_fail++;
        return;
    }

    // Create SPU program: il r3,42 ; il r4,58 ; a r5,r3,r4 ; stop
    uint8_t h_ls[SPU_LS_SIZE];
    memset(h_ls, 0, sizeof(h_ls));

    uint32_t prog[] = {
        bswap32_h(spu_ri16(0x081, 3, 42)),
        bswap32_h(spu_ri16(0x081, 4, 58)),
        bswap32_h(spu_rr(0x0C0, 5, 3, 4)),
        bswap32_h(spu_rr(0x000, 0, 0, 0)),
    };
    memcpy(h_ls, prog, sizeof(prog));

    // Discover + emit
    BasicBlock block;
    jit_discover_block(h_ls, 0, &block);

    char source[MAX_SOURCE_SIZE];
    jit_emit_source(&block, source, MAX_SOURCE_SIZE);

    // Compile
    JITEntry entry;
    if (!jit_compile(&jitState, &block, source, &entry)) {
        printf("  ❌ FAIL (compilation failed)\n");
        total_fail++;
        jit_shutdown(&jitState);
        return;
    }

    // Setup device state
    SPUState h_spu;
    memset(&h_spu, 0, sizeof(h_spu));
    h_spu.pc = 0;

    SPUState* d_spu;
    uint8_t* d_ls;
    cudaMalloc(&d_spu, sizeof(SPUState));
    cudaMalloc(&d_ls, SPU_LS_SIZE);
    cudaMemcpy(d_spu, &h_spu, sizeof(SPUState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls, h_ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);

    // Execute JIT block
    int ok = jit_execute(&entry, d_spu, d_ls, nullptr);

    // Read back
    cudaMemcpy(&h_spu, d_spu, sizeof(SPUState), cudaMemcpyDeviceToHost);

    printf("  r3 = %u (expected 42)\n", h_spu.gpr[3].u32[0]);
    printf("  r4 = %u (expected 58)\n", h_spu.gpr[4].u32[0]);
    printf("  r5 = %u (expected 100)\n", h_spu.gpr[5].u32[0]);
    printf("  halted = %u, cycles = %u\n", h_spu.halted, h_spu.cycles);

    bool pass = ok && (h_spu.gpr[3].u32[0] == 42) &&
                (h_spu.gpr[4].u32[0] == 58) &&
                (h_spu.gpr[5].u32[0] == 100) &&
                h_spu.halted;
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    cudaFree(d_spu);
    cudaFree(d_ls);
    jit_shutdown(&jitState);
}

// ═══════════════════════════════════════════════════════════════
// TEST 4: JIT Float4 SIMD (same as interpreter test 4)
// ═══════════════════════════════════════════════════════════════

static void test_jit_float_simd() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 4: JIT Float4 SIMD               ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    JITState jitState;
    if (!jit_init(&jitState)) {
        printf("  ❌ FAIL (JIT init failed)\n");
        total_fail++;
        return;
    }

    uint8_t h_ls[SPU_LS_SIZE];
    memset(h_ls, 0, sizeof(h_ls));

    // Load vec_a = {1,2,3,4} from LS @ 0x100
    // Load vec_b = {5,6,7,8} from LS @ 0x110
    // fa r5, r3, r4  → r5 = {6,8,10,12}
    // fm r6, r3, r4  → r6 = {5,12,21,32}
    // fma r7, r3, r4, r5  → r7 = r3*r4 + r5 = {11,20,31,44}
    // stop

    // Store vectors in LS (big-endian float)
    auto store_be_float = [&](uint32_t addr, float v) {
        uint32_t bits;
        memcpy(&bits, &v, 4);
        bits = bswap32_h(bits);
        memcpy(h_ls + addr, &bits, 4);
    };

    // vec_a at 0x100
    store_be_float(0x100, 1.0f);
    store_be_float(0x104, 2.0f);
    store_be_float(0x108, 3.0f);
    store_be_float(0x10C, 4.0f);

    // vec_b at 0x110
    store_be_float(0x110, 5.0f);
    store_be_float(0x114, 6.0f);
    store_be_float(0x118, 7.0f);
    store_be_float(0x11C, 8.0f);

    uint32_t pc = 0;
    auto write_inst = [&](uint32_t inst) {
        uint32_t be = bswap32_h(inst);
        memcpy(h_ls + pc, &be, 4);
        pc += 4;
    };

    // lqd r3, 0x10(r0)  → load from 0x100 (i10=0x10, shifted <<4 = 0x100)
    write_inst(spu_ri10(0x34, 3, 0, 0x10));
    // lqd r4, 0x11(r0)  → load from 0x110
    write_inst(spu_ri10(0x34, 4, 0, 0x11));
    // fa r5, r3, r4
    write_inst(spu_rr(0x2C4, 5, 3, 4));
    // fm r6, r3, r4
    write_inst(spu_rr(0x2C6, 6, 3, 4));
    // fma r7, r3, r4, r5
    write_inst(spu_rrr(0xE, 7, 3, 4, 5));
    // stop
    write_inst(spu_rr(0x000, 0, 0, 0));

    // Discover + emit + compile
    BasicBlock block;
    jit_discover_block(h_ls, 0, &block);

    char source[MAX_SOURCE_SIZE];
    jit_emit_source(&block, source, MAX_SOURCE_SIZE);

    JITEntry entry;
    if (!jit_compile(&jitState, &block, source, &entry)) {
        printf("  ❌ FAIL (compilation failed)\n");
        total_fail++;
        jit_shutdown(&jitState);
        return;
    }

    // Setup device state
    SPUState h_spu;
    memset(&h_spu, 0, sizeof(h_spu));

    SPUState* d_spu;
    uint8_t* d_ls;
    cudaMalloc(&d_spu, sizeof(SPUState));
    cudaMalloc(&d_ls, SPU_LS_SIZE);
    cudaMemcpy(d_spu, &h_spu, sizeof(SPUState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls, h_ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);

    int ok = jit_execute(&entry, d_spu, d_ls, nullptr);
    cudaMemcpy(&h_spu, d_spu, sizeof(SPUState), cudaMemcpyDeviceToHost);

    printf("  r5 (a+b) = {%.1f, %.1f, %.1f, %.1f} (expected {6,8,10,12})\n",
           h_spu.gpr[5].f32[0], h_spu.gpr[5].f32[1],
           h_spu.gpr[5].f32[2], h_spu.gpr[5].f32[3]);
    printf("  r6 (a*b) = {%.1f, %.1f, %.1f, %.1f} (expected {5,12,21,32})\n",
           h_spu.gpr[6].f32[0], h_spu.gpr[6].f32[1],
           h_spu.gpr[6].f32[2], h_spu.gpr[6].f32[3]);
    printf("  r7 (fma) = {%.1f, %.1f, %.1f, %.1f} (expected {11,20,31,44})\n",
           h_spu.gpr[7].f32[0], h_spu.gpr[7].f32[1],
           h_spu.gpr[7].f32[2], h_spu.gpr[7].f32[3]);

    bool pass = ok &&
        h_spu.gpr[5].f32[0] == 6.0f && h_spu.gpr[5].f32[1] == 8.0f &&
        h_spu.gpr[5].f32[2] == 10.0f && h_spu.gpr[5].f32[3] == 12.0f &&
        h_spu.gpr[6].f32[0] == 5.0f && h_spu.gpr[6].f32[1] == 12.0f &&
        h_spu.gpr[6].f32[2] == 21.0f && h_spu.gpr[6].f32[3] == 32.0f &&
        h_spu.gpr[7].f32[0] == 11.0f && h_spu.gpr[7].f32[1] == 20.0f &&
        h_spu.gpr[7].f32[2] == 31.0f && h_spu.gpr[7].f32[3] == 44.0f;

    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    cudaFree(d_spu);
    cudaFree(d_ls);
    jit_shutdown(&jitState);
}

// ═══════════════════════════════════════════════════════════════
// TEST 5: JIT vs Interpreter Benchmark
// ═══════════════════════════════════════════════════════════════

// We need the interpreter's spuExecOne — declare it externally
// For this benchmark, we'll compile a tight loop and measure
// JIT block execution vs equivalent interpreter cycles

// We need the interpreter's SPU API — declared at file scope
extern "C" int spu_init();
extern "C" int spu_load_program(int spuId, const void* data, size_t size, uint32_t entryPC);
extern "C" float spu_run(uint32_t maxCycles, uint8_t* d_mainMem);
extern "C" int spu_read_state(int spuId, SPUState* out);
extern "C" void spu_shutdown();

static void test_jit_benchmark() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 5: JIT vs Interpreter Benchmark  ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    JITState jitState;
    if (!jit_init(&jitState)) {
        printf("  ❌ FAIL (JIT init failed)\n");
        total_fail++;
        return;
    }

    // Build a compute-heavy block: 50 fa instructions (float add)
    uint8_t h_ls[SPU_LS_SIZE];
    memset(h_ls, 0, sizeof(h_ls));

    // Store initial values
    auto store_be_float = [&](uint32_t addr, float v) {
        uint32_t bits;
        memcpy(&bits, &v, 4);
        bits = bswap32_h(bits);
        memcpy(h_ls + addr, &bits, 4);
    };

    // r3 = {1,1,1,1}, r4 = {1,1,1,1} loaded from LS
    for (int i = 0; i < 4; i++) {
        store_be_float(0x200 + i * 4, 1.0f);
        store_be_float(0x210 + i * 4, 1.0f);
    }

    uint32_t pc = 0;
    auto write_inst = [&](uint32_t inst) {
        uint32_t be = bswap32_h(inst);
        memcpy(h_ls + pc, &be, 4);
        pc += 4;
    };

    // Load initial vectors
    write_inst(spu_ri10(0x34, 3, 0, 0x20));  // lqd r3, 0x200
    write_inst(spu_ri10(0x34, 4, 0, 0x21));  // lqd r4, 0x210

    // 50× fa r3, r3, r4  (accumulate: r3 += r4 each iteration)
    for (int i = 0; i < 50; i++) {
        write_inst(spu_rr(0x2C4, 3, 3, 4));
    }

    // stop
    write_inst(spu_rr(0x000, 0, 0, 0));

    uint32_t totalInsns = 2 + 50 + 1; // 53 instructions

    // Discover + compile
    BasicBlock block;
    jit_discover_block(h_ls, 0, &block);
    printf("  Block: %u instructions\n", block.numInsns);

    char source[MAX_SOURCE_SIZE];
    jit_emit_source(&block, source, MAX_SOURCE_SIZE);

    JITEntry entry;
    if (!jit_compile(&jitState, &block, source, &entry)) {
        printf("  ❌ FAIL (compilation failed)\n");
        total_fail++;
        jit_shutdown(&jitState);
        return;
    }

    // Setup device
    SPUState h_spu;
    memset(&h_spu, 0, sizeof(h_spu));
    SPUState* d_spu;
    uint8_t* d_ls;
    cudaMalloc(&d_spu, sizeof(SPUState));
    cudaMalloc(&d_ls, SPU_LS_SIZE);
    cudaMemcpy(d_ls, h_ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);

    // === JIT benchmark: run compiled block N times ===
    const int RUNS = 1000;

    cudaEvent_t jitStart, jitStop;
    cudaEventCreate(&jitStart);
    cudaEventCreate(&jitStop);

    // Warm up
    cudaMemcpy(d_spu, &h_spu, sizeof(SPUState), cudaMemcpyHostToDevice);
    jit_execute(&entry, d_spu, d_ls, nullptr);

    cudaEventRecord(jitStart);
    for (int r = 0; r < RUNS; r++) {
        memset(&h_spu, 0, sizeof(h_spu));
        cudaMemcpy(d_spu, &h_spu, sizeof(SPUState), cudaMemcpyHostToDevice);
        jit_execute(&entry, d_spu, d_ls, nullptr);
    }
    cudaEventRecord(jitStop);
    cudaEventSynchronize(jitStop);

    float jitMs = 0;
    cudaEventElapsedTime(&jitMs, jitStart, jitStop);

    float jitMips = (float)totalInsns * RUNS / (jitMs * 1000.0f);

    // Verify correctness: r3 should be {51,51,51,51} (1 + 50*1)
    cudaMemcpy(&h_spu, d_spu, sizeof(SPUState), cudaMemcpyDeviceToHost);
    printf("  JIT result: r3 = {%.1f, %.1f, %.1f, %.1f} (expected {51,51,51,51})\n",
           h_spu.gpr[3].f32[0], h_spu.gpr[3].f32[1],
           h_spu.gpr[3].f32[2], h_spu.gpr[3].f32[3]);

    // === Interpreter benchmark: run same program via interpreter kernel ===
    // We'll use the external spu_run function
    spu_init();

    cudaEvent_t intStart, intStop;
    cudaEventCreate(&intStart);
    cudaEventCreate(&intStop);

    // Warm up
    spu_load_program(0, h_ls, pc, 0);
    spu_run(1000, nullptr);

    cudaEventRecord(intStart);
    for (int r = 0; r < RUNS; r++) {
        spu_load_program(0, h_ls, pc, 0);
        spu_run(1000, nullptr);
    }
    cudaEventRecord(intStop);
    cudaEventSynchronize(intStop);

    float intMs = 0;
    cudaEventElapsedTime(&intMs, intStart, intStop);

    float intMips = (float)totalInsns * RUNS / (intMs * 1000.0f);

    SPUState intState;
    spu_read_state(0, &intState);
    printf("  Interpreter result: r3 = {%.1f, %.1f, %.1f, %.1f}\n",
           intState.gpr[3].f32[0], intState.gpr[3].f32[1],
           intState.gpr[3].f32[2], intState.gpr[3].f32[3]);

    printf("\n  ┌─────────────────────────────────────┐\n");
    printf("  │ JIT:         %8.1f MIPS (%6.2f ms) │\n", jitMips, jitMs);
    printf("  │ Interpreter: %8.1f MIPS (%6.2f ms) │\n", intMips, intMs);
    printf("  │ Speedup:     %6.1f×                 │\n", jitMips / intMips);
    printf("  └─────────────────────────────────────┘\n");

    bool pass = h_spu.gpr[3].f32[0] == 51.0f;
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    spu_shutdown();
    cudaEventDestroy(jitStart);
    cudaEventDestroy(jitStop);
    cudaEventDestroy(intStart);
    cudaEventDestroy(intStop);
    cudaFree(d_spu);
    cudaFree(d_ls);
    jit_shutdown(&jitState);
}

// ═══════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  🚀 SPU JIT Compiler Tests                   ║\n");
    printf("║  NVRTC: SPU → native CUDA kernels            ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    // Check GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s · Compute %d.%d\n\n", prop.name, prop.major, prop.minor);

    test_block_discovery();
    test_source_emission();
    test_jit_compile_execute();
    test_jit_float_simd();
    test_jit_benchmark();

    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d/%d tests passed\n",
           total_pass, total_pass + total_fail);
    printf("═══════════════════════════════════════════\n");

    return total_fail > 0 ? 1 : 0;
}
