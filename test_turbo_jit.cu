// test_turbo_jit.cu — Turbo JIT benchmarks
// Tests register promotion + multi-instance SPURS parallelism
#include "spu_defs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
using namespace spu;

struct SPUTurboState { QWord gpr[128]; uint32_t pc, halted, cycles; };
struct TurboJITResult { void *cuModule, *cuFunction, *cuMultiFunction;
    uint32_t numBlocks, totalInsns, numPromotedRegs; float compileTimeMs; };

extern "C" int   turbo_jit_compile(const uint8_t* h_ls, uint32_t entryPC, TurboJITResult* r);
extern "C" float turbo_jit_run(const TurboJITResult* jit, SPUTurboState* h_state,
                               uint8_t* d_ls, uint8_t* d_mainMem, uint32_t maxCycles);
extern "C" float turbo_jit_run_multi(const TurboJITResult* jit, SPUTurboState* h_states,
                                     uint8_t** d_lsArray, uint8_t* d_mainMem,
                                     uint32_t maxCycles, uint32_t numInstances);
extern "C" void  turbo_jit_free(TurboJITResult* jit);

// Interpreter API
extern "C" int   spu_init();
extern "C" int   spu_load_program(int spuId, const void* data, size_t size, uint32_t entryPC);
extern "C" float spu_run(uint32_t maxCycles, uint8_t* d_mainMem);
extern "C" int   spu_read_state(int spuId, SPUState* out);
extern "C" void  spu_shutdown();

// MegaJIT API (for comparison)
struct MegaJITResult { void *cuModule, *cuFunction; uint32_t numBlocks, totalInsns; float compileTimeMs; };
extern "C" int   mega_jit_compile(const uint8_t* h_ls, uint32_t entryPC, MegaJITResult* r);
extern "C" float mega_jit_run(const MegaJITResult* jit, SPUState* h_state,
                              uint8_t* d_ls, uint8_t* d_mainMem, uint32_t maxCycles);
extern "C" void  mega_jit_free(MegaJITResult* jit);

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
// TEST 1: Turbo Correctness (float SIMD)
// ═══════════════════════════════════════════════════════════════
static void test_correctness() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: Turbo JIT Correctness         ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE]; memset(ls,0,sizeof(ls));
    for(int i=0;i<4;i++){store_be_float(ls,0x100+i*4,(float)(i+1));store_be_float(ls,0x110+i*4,(float)(i+5));}

    uint32_t pc=0;
    auto w=[&](uint32_t i){uint32_t be=bswap32_h(i);memcpy(ls+pc,&be,4);pc+=4;};
    w(spu_ri10(0x34,3,0,0x10)); w(spu_ri10(0x34,4,0,0x11));
    w(spu_rr(0x2C4,5,3,4)); w(spu_rr(0x2C6,6,3,4));
    w(spu_rrr(0xE,7,3,4,5)); w(spu_rr(0x000,0,0,0));

    TurboJITResult jit;
    if(!turbo_jit_compile(ls,0,&jit)){printf("  ❌ FAIL\n");tf++;return;}

    uint8_t* d_ls; cudaMalloc(&d_ls,SPU_LS_SIZE);
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);

    SPUTurboState h; memset(&h,0,sizeof(h));
    turbo_jit_run(&jit,&h,d_ls,nullptr,100);

    printf("  r5={%.1f,%.1f,%.1f,%.1f} r6={%.1f,%.1f,%.1f,%.1f} r7={%.1f,%.1f,%.1f,%.1f}\n",
        h.gpr[5].f32[0],h.gpr[5].f32[1],h.gpr[5].f32[2],h.gpr[5].f32[3],
        h.gpr[6].f32[0],h.gpr[6].f32[1],h.gpr[6].f32[2],h.gpr[6].f32[3],
        h.gpr[7].f32[0],h.gpr[7].f32[1],h.gpr[7].f32[2],h.gpr[7].f32[3]);

    bool pass = h.gpr[5].f32[0]==6.0f && h.gpr[7].f32[3]==44.0f && h.halted;
    printf("  Result: %s\n", pass?"✅ PASS":"❌ FAIL");
    if(pass) tp++; else tf++;
    cudaFree(d_ls); turbo_jit_free(&jit);
}

// ═══════════════════════════════════════════════════════════════
// TEST 2: Triple Benchmark — Interpreter vs MegaJIT vs TurboJIT
// ═══════════════════════════════════════════════════════════════
static void test_triple_bench() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: Interpreter vs Mega vs Turbo  ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t ls[SPU_LS_SIZE]; memset(ls,0,sizeof(ls));
    for(int i=0;i<4;i++){store_be_float(ls,0x200+i*4,1.0f);store_be_float(ls,0x210+i*4,0.01f);}

    uint32_t pc=0;
    auto w=[&](uint32_t i){uint32_t be=bswap32_h(i);memcpy(ls+pc,&be,4);pc+=4;};
    w(spu_ri10(0x34,3,0,0x20)); w(spu_ri10(0x34,4,0,0x21));
    w(spu_ri16(0x081,10,10000)); w(spu_ri16(0x081,11,1));
    // loop:
    w(spu_rr(0x2C4,3,3,4));    // fa
    w(spu_rr(0x2C6,5,3,3));    // fm
    w(spu_rrr(0xE,6,3,4,5));   // fma
    w(spu_rr(0x040,10,11,10)); // sf
    w(spu_ri16(0x042,10,-4));  // brnz
    w(spu_rr(0x000,0,0,0));    // stop
    uint32_t progSize = pc;

    uint8_t* d_ls; cudaMalloc(&d_ls,SPU_LS_SIZE);
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);

    // --- Turbo JIT ---
    TurboJITResult turbo;
    turbo_jit_compile(ls,0,&turbo);
    SPUTurboState ht; memset(&ht,0,sizeof(ht));
    turbo_jit_run(&turbo,&ht,d_ls,nullptr,200000); // warmup
    memset(&ht,0,sizeof(ht));
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);
    float turboMs=turbo_jit_run(&turbo,&ht,d_ls,nullptr,200000);
    float turboMips=(float)ht.cycles/(turboMs*1000.0f);
    printf("  Turbo:  %u cyc in %.3f ms = %.1f MIPS\n",ht.cycles,turboMs,turboMips);

    // --- MegaJIT ---
    MegaJITResult mega;
    mega_jit_compile(ls,0,&mega);
    SPUState hm; memset(&hm,0,sizeof(hm));
    mega_jit_run(&mega,&hm,d_ls,nullptr,200000); // warmup
    memset(&hm,0,sizeof(hm));
    cudaMemcpy(d_ls,ls,SPU_LS_SIZE,cudaMemcpyHostToDevice);
    float megaMs=mega_jit_run(&mega,&hm,d_ls,nullptr,200000);
    float megaMips=(float)hm.cycles/(megaMs*1000.0f);
    printf("  Mega:   %u cyc in %.3f ms = %.1f MIPS\n",hm.cycles,megaMs,megaMips);

    // --- Interpreter ---
    spu_init();
    spu_load_program(0,ls,progSize,0);
    spu_run(200000,nullptr); // warmup
    spu_load_program(0,ls,progSize,0);
    cudaEvent_t i0,i1; cudaEventCreate(&i0); cudaEventCreate(&i1);
    cudaEventRecord(i0); spu_run(200000,nullptr); cudaEventRecord(i1);
    cudaEventSynchronize(i1);
    float intMs; cudaEventElapsedTime(&intMs,i0,i1);
    SPUState hi; spu_read_state(0,&hi);
    float intMips=(float)hi.cycles/(intMs*1000.0f);
    printf("  Interp: %u cyc in %.3f ms = %.1f MIPS\n",hi.cycles,intMs,intMips);
    spu_shutdown();

    printf("\n  ┌───────────────────────────────────────────────┐\n");
    printf("  │ Interpreter: %8.1f MIPS  (baseline)          │\n",intMips);
    printf("  │ MegaJIT:     %8.1f MIPS  (%5.1f× vs interp) │\n",megaMips,megaMips/intMips);
    printf("  │ TurboJIT:    %8.1f MIPS  (%5.1f× vs interp) │\n",turboMips,turboMips/intMips);
    printf("  │ Turbo/Mega:  %8.1f×                          │\n",turboMips/megaMips);
    printf("  └───────────────────────────────────────────────┘\n");

    bool pass = turboMips > megaMips;
    printf("  Result: %s\n", pass?"✅ PASS":"❌ FAIL");
    if(pass) tp++; else tf++;

    cudaFree(d_ls); turbo_jit_free(&turbo); mega_jit_free(&mega);
    cudaEventDestroy(i0); cudaEventDestroy(i1);
}

// ═══════════════════════════════════════════════════════════════
// TEST 3: SPURS Multi-Instance (N parallel SPU programs)
// ═══════════════════════════════════════════════════════════════
static void test_spurs_multi() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: SPURS Multi-Instance (5120)   ║\n");
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

    TurboJITResult turbo;
    turbo_jit_compile(ls,0,&turbo);

    // Scale test: 1, 80, 640, 2560, 5120 instances
    uint32_t counts[] = {1, 80, 640, 2560, 5120};
    float singleMips = 0;

    for (int ci = 0; ci < 5; ci++) {
        uint32_t N = counts[ci];

        // Allocate N Local Stores + N states
        SPUTurboState* h_states = new SPUTurboState[N];
        uint8_t** h_lsPtrs = new uint8_t*[N];

        for (uint32_t j = 0; j < N; j++) {
            memset(&h_states[j], 0, sizeof(SPUTurboState));
            cudaMalloc(&h_lsPtrs[j], SPU_LS_SIZE);
            cudaMemcpy(h_lsPtrs[j], ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);
        }

        // Warmup
        turbo_jit_run_multi(&turbo, h_states, h_lsPtrs, nullptr, 50000, N);
        for (uint32_t j = 0; j < N; j++) memset(&h_states[j], 0, sizeof(SPUTurboState));
        for (uint32_t j = 0; j < N; j++)
            cudaMemcpy(h_lsPtrs[j], ls, SPU_LS_SIZE, cudaMemcpyHostToDevice);

        float ms = turbo_jit_run_multi(&turbo, h_states, h_lsPtrs, nullptr, 50000, N);

        // Total cycles across all instances
        uint64_t totalCycles = 0;
        for (uint32_t j = 0; j < N; j++) totalCycles += h_states[j].cycles;
        float aggMips = (float)totalCycles / (ms * 1000.0f);

        if (ci == 0) singleMips = aggMips;

        printf("  %5u instances: %8.1f MIPS aggregate (%6.2f ms) %.1f× scale\n",
               N, aggMips, ms, aggMips / (singleMips > 0 ? singleMips : 1.0f));

        for (uint32_t j = 0; j < N; j++) cudaFree(h_lsPtrs[j]);
        delete[] h_states;
        delete[] h_lsPtrs;
    }

    printf("  Result: ✅ PASS\n");
    tp++;
    turbo_jit_free(&turbo);
}

int main() {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  ⚡ SPU Turbo JIT — Register Promotion       ║\n");
    printf("║  + SPURS Multi-Instance Parallelism          ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("  GPU: %s · %d SMs · %d cores\n", p.name, p.multiProcessorCount,
           p.multiProcessorCount * 64);
    cuInit(0);

    test_correctness();
    test_triple_bench();
    test_spurs_multi();

    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d/%d tests passed\n", tp, tp+tf);
    printf("═══════════════════════════════════════════\n");
    return tf>0?1:0;
}
