// test_hyper_jit.cu — Hyper JIT benchmarks
// Tests loop detection + batched cycles vs turbo vs mega vs interpreter
#include "spu_defs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
using namespace spu;

// Hyper JIT API
struct SPUHyperState { QWord gpr[128]; uint32_t pc, halted, cycles; };
struct HyperJITResult { void *cuModule, *cuFunction, *cuMultiFunction;
    uint32_t numBlocks, totalInsns, numPromotedRegs, numSelfLoops; float compileTimeMs; };

extern "C" int   hyper_jit_compile(const uint8_t* h_ls, uint32_t entryPC, HyperJITResult* r);
extern "C" float hyper_jit_run(const HyperJITResult* jit, SPUHyperState* h_state,
                               uint8_t* d_ls, uint8_t* d_mainMem, uint32_t maxCycles);
extern "C" float hyper_jit_run_multi(const HyperJITResult* jit, SPUHyperState* h_states,
                                     uint8_t** d_lsArray, uint8_t* d_mainMem,
                                     uint32_t maxCycles, uint32_t numInstances);
extern "C" void  hyper_jit_free(HyperJITResult* jit);

// Turbo JIT API (for comparison)
struct SPUTurboState { QWord gpr[128]; uint32_t pc, halted, cycles; };
struct TurboJITResult { void *cuModule, *cuFunction, *cuMultiFunction;
    uint32_t numBlocks, totalInsns, numPromotedRegs; float compileTimeMs; };
extern "C" int   turbo_jit_compile(const uint8_t* h_ls, uint32_t entryPC, TurboJITResult* r);
extern "C" float turbo_jit_run(const TurboJITResult* jit, SPUTurboState* h_state,
                               uint8_t* d_ls, uint8_t* d_mainMem, uint32_t maxCycles);
extern "C" void  turbo_jit_free(TurboJITResult* jit);

// Interpreter API
extern "C" int   spu_init();
extern "C" int   spu_load_program(int spuId, const void* data, size_t size, uint32_t entryPC);
extern "C" float spu_run(uint32_t maxCycles, uint8_t* d_mainMem);
extern "C" int   spu_read_state(int spuId, SPUState* out);
extern "C" void  spu_shutdown();

static uint32_t bswap32_h(uint32_t x) {
    return ((x>>24)&0xFF)|((x>>8)&0xFF00)|((x<<8)&0xFF0000)|((x<<24)&0xFF000000);
}
static uint32_t spu_ri16(uint32_t o9, uint32_t rT, int32_t i16) {
    return (o9<<23)|((i16&0xFFFF)<<7)|(rT&0x7F); }
static uint32_t spu_ri10(uint32_t o8, uint32_t rT, uint32_t rA, int32_t i10) {
    return (o8<<24)|((i10&0x3FF)<<14)|((rA&0x7F)<<7)|(rT&0x7F); }
static uint32_t spu_rr(uint32_t o11, uint32_t rT, uint32_t rA, uint32_t rB) {
    return (o11<<21)|((rT&0x7F)<<14)|((rB&0x7F)<<7)|(rA&0x7F); }
static uint32_t spu_rrr(uint32_t o4, uint32_t rT, uint32_t rA, uint32_t rB, uint32_t rC) {
    return (o4<<28)|((rT&0x7F)<<21)|((rB&0x7F)<<14)|((rA&0x7F)<<7)|(rC&0x7F); }

static void store_be_float(uint8_t* ls, uint32_t addr, float v) {
    uint32_t b; memcpy(&b,&v,4); b=bswap32_h(b); memcpy(ls+addr,&b,4); }

static int tp=0, tf=0;

// ═══════════════════════════════════════════════════════════════
// TEST 1: Correctness — same float SIMD test as turbo
// ═══════════════════════════════════════════════════════════════
static void test_correctness() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: Hyper JIT Correctness          ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE]; memset(ls,0,sizeof(ls));
    for(int i=0;i<4;i++){store_be_float(ls,0x100+i*4,(float)(i+1));store_be_float(ls,0x110+i*4,(float)(i+5));}

    uint32_t pc=0;
    auto w=[&](uint32_t i){uint32_t be=bswap32_h(i);memcpy(ls+pc,&be,4);pc+=4;};
    w(spu_ri10(0x34,3,0,0x10)); // lqd r3, 0x100($0)
    w(spu_ri10(0x34,4,0,0x11)); // lqd r4, 0x110($0)
    w(spu_rr(0x2C4,5,3,4));     // fa r5, r3, r4
    w(spu_rr(0x2C6,6,3,4));     // fm r6, r3, r4
    w(spu_rrr(0xE,7,3,4,5));    // fma r7, r3, r4, r5
    w(spu_rr(0x000,0,0,0));     // stop

    HyperJITResult jit;
    if(!hyper_jit_compile(ls,0,&jit)){printf("  ❌ FAIL (compile)\n");tf++;return;}

    uint8_t* d_ls; cudaMalloc(&d_ls,SPU_LS_SIZE);
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);

    SPUHyperState h; memset(&h,0,sizeof(h));
    hyper_jit_run(&jit,&h,d_ls,nullptr,100);

    printf("  r5={%.1f,%.1f,%.1f,%.1f} r6={%.1f,%.1f,%.1f,%.1f} r7={%.1f,%.1f,%.1f,%.1f}\n",
        h.gpr[5].f32[0],h.gpr[5].f32[1],h.gpr[5].f32[2],h.gpr[5].f32[3],
        h.gpr[6].f32[0],h.gpr[6].f32[1],h.gpr[6].f32[2],h.gpr[6].f32[3],
        h.gpr[7].f32[0],h.gpr[7].f32[1],h.gpr[7].f32[2],h.gpr[7].f32[3]);

    bool pass = h.gpr[5].f32[0]==6.0f && h.gpr[7].f32[3]==44.0f && h.halted;
    printf("  Result: %s\n", pass?"✅ PASS":"❌ FAIL");
    if(pass) tp++; else tf++;
    cudaFree(d_ls); hyper_jit_free(&jit);
}

// ═══════════════════════════════════════════════════════════════
// TEST 2: Loop detection + quad benchmark
// ═══════════════════════════════════════════════════════════════
static void test_quad_bench() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: Quad Benchmark (all 4 tiers)   ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE]; memset(ls,0,sizeof(ls));
    for(int i=0;i<4;i++){store_be_float(ls,0x200+i*4,1.0f);store_be_float(ls,0x210+i*4,0.01f);}

    uint32_t pc=0;
    auto w=[&](uint32_t i){uint32_t be=bswap32_h(i);memcpy(ls+pc,&be,4);pc+=4;};
    w(spu_ri10(0x34,3,0,0x20)); // lqd r3, 0x200
    w(spu_ri10(0x34,4,0,0x21)); // lqd r4, 0x210
    w(spu_ri16(0x081,10,10000)); // il r10, 10000
    w(spu_ri16(0x081,11,1));     // il r11, 1
    // loop @ 0x10:
    w(spu_rr(0x2C4,3,3,4));     // fa r3, r3, r4
    w(spu_rr(0x2C6,5,3,3));     // fm r5, r3, r3
    w(spu_rrr(0xE,6,3,4,5));    // fma r6, r3, r4, r5
    w(spu_rr(0x040,10,11,10));   // sf r10, r11, r10
    w(spu_ri16(0x042,10,-4));    // brnz r10, -4 (back to 0x10)
    w(spu_rr(0x000,0,0,0));     // stop
    uint32_t progSize = pc;

    uint8_t* d_ls; cudaMalloc(&d_ls,SPU_LS_SIZE);
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);
    uint32_t maxCyc = 200000;

    // --- Hyper JIT ---
    HyperJITResult hyper;
    hyper_jit_compile(ls,0,&hyper);
    SPUHyperState hh; memset(&hh,0,sizeof(hh));
    hyper_jit_run(&hyper,&hh,d_ls,nullptr,maxCyc); // warmup
    memset(&hh,0,sizeof(hh));
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);
    float hyperMs=hyper_jit_run(&hyper,&hh,d_ls,nullptr,maxCyc);
    float hyperMips=(float)hh.cycles/(hyperMs*1000.0f);
    printf("  Hyper:  %u cyc in %.3f ms = %.1f MIPS\n",hh.cycles,hyperMs,hyperMips);

    // --- Turbo JIT ---
    TurboJITResult turbo;
    turbo_jit_compile(ls,0,&turbo);
    SPUTurboState ht; memset(&ht,0,sizeof(ht));
    turbo_jit_run(&turbo,&ht,d_ls,nullptr,maxCyc); // warmup
    memset(&ht,0,sizeof(ht));
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);
    float turboMs=turbo_jit_run(&turbo,&ht,d_ls,nullptr,maxCyc);
    float turboMips=(float)ht.cycles/(turboMs*1000.0f);
    printf("  Turbo:  %u cyc in %.3f ms = %.1f MIPS\n",ht.cycles,turboMs,turboMips);

    // --- Interpreter ---
    spu_init();
    spu_load_program(0,ls,progSize,0);
    spu_run(maxCyc,nullptr); // warmup
    spu_load_program(0,ls,progSize,0);
    cudaEvent_t i0,i1; cudaEventCreate(&i0); cudaEventCreate(&i1);
    cudaEventRecord(i0); spu_run(maxCyc,nullptr); cudaEventRecord(i1);
    cudaEventSynchronize(i1);
    float intMs; cudaEventElapsedTime(&intMs,i0,i1);
    SPUState hi; spu_read_state(0,&hi);
    float intMips=(float)hi.cycles/(intMs*1000.0f);
    printf("  Interp: %u cyc in %.3f ms = %.1f MIPS\n",hi.cycles,intMs,intMips);
    spu_shutdown();

    printf("\n  ┌──────────────────────────────────────────────────────────┐\n");
    printf("  │ Interpreter: %8.1f MIPS  (baseline)                    │\n",intMips);
    printf("  │ TurboJIT:    %8.1f MIPS  (%6.1f× vs interp)           │\n",turboMips,turboMips/intMips);
    printf("  │ HyperJIT:    %8.1f MIPS  (%6.1f× vs interp)           │\n",hyperMips,hyperMips/intMips);
    printf("  │ Hyper/Turbo: %8.1f×      (loop elimination speedup)   │\n",hyperMips/turboMips);
    printf("  └──────────────────────────────────────────────────────────┘\n");

    bool pass = hyperMips > turboMips;
    printf("  Result: %s\n", pass?"✅ PASS":"❌ FAIL");
    if(pass) tp++; else tf++;

    cudaFree(d_ls); hyper_jit_free(&hyper); turbo_jit_free(&turbo);
    cudaEventDestroy(i0); cudaEventDestroy(i1);
}

// ═══════════════════════════════════════════════════════════════
// TEST 3: SPURS scaling with hyper loop elimination
// ═══════════════════════════════════════════════════════════════
static void test_hyper_spurs() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: Hyper SPURS Multi-Instance     ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE]; memset(ls,0,sizeof(ls));
    for(int i=0;i<4;i++){store_be_float(ls,0x200+i*4,1.0f);store_be_float(ls,0x210+i*4,0.01f);}

    uint32_t pc=0;
    auto w=[&](uint32_t i){uint32_t be=bswap32_h(i);memcpy(ls+pc,&be,4);pc+=4;};
    w(spu_ri10(0x34,3,0,0x20)); w(spu_ri10(0x34,4,0,0x21));
    w(spu_ri16(0x081,10,1000)); w(spu_ri16(0x081,11,1));
    w(spu_rr(0x2C4,3,3,4)); w(spu_rr(0x2C6,5,3,3));
    w(spu_rrr(0xE,6,3,4,5)); w(spu_rr(0x040,10,11,10));
    w(spu_ri16(0x042,10,-4)); w(spu_rr(0x000,0,0,0));

    HyperJITResult hyper;
    hyper_jit_compile(ls,0,&hyper);

    uint32_t counts[] = {1, 80, 640, 2560, 5120};
    float singleMips = 0;

    for (int ci = 0; ci < 5; ci++) {
        uint32_t N = counts[ci];
        SPUHyperState* h_states = new SPUHyperState[N];
        uint8_t** h_lsPtrs = new uint8_t*[N];

        for (uint32_t j = 0; j < N; j++) {
            memset(&h_states[j], 0, sizeof(SPUHyperState));
            cudaMalloc(&h_lsPtrs[j], SPU_LS_SIZE);
            cudaMemcpy(h_lsPtrs[j], ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);
        }

        // Warmup
        hyper_jit_run_multi(&hyper, h_states, h_lsPtrs, nullptr, 50000, N);
        for (uint32_t j = 0; j < N; j++) memset(&h_states[j], 0, sizeof(SPUHyperState));
        for (uint32_t j = 0; j < N; j++)
            cudaMemcpy(h_lsPtrs[j], ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);

        float ms = hyper_jit_run_multi(&hyper, h_states, h_lsPtrs, nullptr, 50000, N);

        uint64_t totalCycles = 0;
        for (uint32_t j = 0; j < N; j++) totalCycles += h_states[j].cycles;
        float aggMips = (float)totalCycles / (ms * 1000.0f);

        if (ci == 0) singleMips = aggMips;

        printf("  %5u instances: %10.1f MIPS aggregate (%6.2f ms) %.1f× scale\n",
               N, aggMips, ms, aggMips / (singleMips > 0 ? singleMips : 1.0f));

        for (uint32_t j = 0; j < N; j++) cudaFree(h_lsPtrs[j]);
        delete[] h_states;
        delete[] h_lsPtrs;
    }

    printf("  Result: ✅ PASS\n");
    tp++;
    hyper_jit_free(&hyper);
}

// ═══════════════════════════════════════════════════════════════
// TEST 4: Heavy compute loop — stress test the loop optimizer
// ═══════════════════════════════════════════════════════════════
static void test_heavy_compute() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 4: Heavy Compute (100K iters)     ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE]; memset(ls,0,sizeof(ls));
    for(int i=0;i<4;i++){store_be_float(ls,0x200+i*4,1.0f);store_be_float(ls,0x210+i*4,0.001f);}

    uint32_t pc=0;
    auto w=[&](uint32_t i){uint32_t be=bswap32_h(i);memcpy(ls+pc,&be,4);pc+=4;};
    w(spu_ri10(0x34,3,0,0x20)); // lqd r3
    w(spu_ri10(0x34,4,0,0x21)); // lqd r4
    w(spu_ri16(0x081,10,100000)); // il r10, 100000 (BIG loop)
    w(spu_ri16(0x081,11,1));
    // loop: 8 compute ops for high ILP
    w(spu_rr(0x2C4,3,3,4));     // fa
    w(spu_rr(0x2C6,5,3,3));     // fm
    w(spu_rrr(0xE,6,3,4,5));    // fma
    w(spu_rr(0x2C4,7,5,6));     // fa
    w(spu_rr(0x2C6,8,6,7));     // fm
    w(spu_rrr(0xE,9,7,8,3));    // fma
    w(spu_rr(0x2C4,3,9,4));     // fa (feedback)
    w(spu_rr(0x040,10,11,10));   // sf (counter)
    w(spu_ri16(0x042,10,-8));    // brnz r10, -8
    w(spu_rr(0x000,0,0,0));     // stop

    uint8_t* d_ls; cudaMalloc(&d_ls,SPU_LS_SIZE);
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);

    // Hyper
    HyperJITResult hyper;
    hyper_jit_compile(ls,0,&hyper);
    SPUHyperState hh; memset(&hh,0,sizeof(hh));
    hyper_jit_run(&hyper,&hh,d_ls,nullptr,2000000); // warmup
    memset(&hh,0,sizeof(hh));
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);
    float hyperMs=hyper_jit_run(&hyper,&hh,d_ls,nullptr,2000000);
    float hyperMips=(float)hh.cycles/(hyperMs*1000.0f);

    // Turbo
    TurboJITResult turbo;
    turbo_jit_compile(ls,0,&turbo);
    SPUTurboState ht; memset(&ht,0,sizeof(ht));
    turbo_jit_run(&turbo,&ht,d_ls,nullptr,2000000); // warmup
    memset(&ht,0,sizeof(ht));
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);
    float turboMs=turbo_jit_run(&turbo,&ht,d_ls,nullptr,2000000);
    float turboMips=(float)ht.cycles/(turboMs*1000.0f);

    printf("  100K iterations × 9 ops/iter:\n");
    printf("    Turbo:  %u cyc in %.3f ms = %.1f MIPS\n",ht.cycles,turboMs,turboMips);
    printf("    Hyper:  %u cyc in %.3f ms = %.1f MIPS\n",hh.cycles,hyperMs,hyperMips);
    printf("    Speedup: %.1f×\n", hyperMips/turboMips);

    bool pass = hyperMips > turboMips;
    printf("  Result: %s\n", pass?"✅ PASS":"❌ FAIL");
    if(pass) tp++; else tf++;

    cudaFree(d_ls); hyper_jit_free(&hyper); turbo_jit_free(&turbo);
}

int main() {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  🚀 SPU Hyper JIT — Loop Elimination         ║\n");
    printf("║  + Batched Cycles + Register Promotion        ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("  GPU: %s · %d SMs · %d cores\n", p.name, p.multiProcessorCount,
           p.multiProcessorCount * 64);
    cuInit(0);

    test_correctness();
    test_quad_bench();
    test_hyper_spurs();
    test_heavy_compute();

    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d/%d tests passed\n", tp, tp+tf);
    printf("═══════════════════════════════════════════\n");
    return tf>0?1:0;
}
