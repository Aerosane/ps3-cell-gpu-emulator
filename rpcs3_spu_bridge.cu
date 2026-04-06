// rpcs3_spu_bridge.cu — CUDA SPU Bridge Implementation
//
// Bridges any PS3 emulator's SPU execution to GPU via our HyperJIT.
// Manages CUDA context, device memory, JIT compilation cache, and
// bidirectional LS/register transfer.
//
#include "rpcs3_spu_bridge.h"
#include "spu_defs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// SPU Local Store size
static constexpr uint32_t LS_SIZE = 256 * 1024;  // 256KB

// JIT cache entry
struct SPUCacheEntry {
    uint64_t  codeHash;     // FNV-1a hash of LS code region
    CUmodule  cuModule;
    CUfunction cuFunction;
    uint32_t  hitCount;
    bool      valid;
};

static constexpr int MAX_CACHE = 256;

// Bridge global state
struct BridgeState {
    bool          initialized;
    CUcontext     cuCtx;

    // Device memory (persistent across runs to avoid realloc)
    uint8_t*      d_ls;          // 256KB device Local Store
    uint32_t*     d_regs;        // 128 × 4 uint32 = 2KB
    uint32_t*     d_info;        // [pc, status, stopped, cycles_out]

    // JIT cache
    SPUCacheEntry cache[MAX_CACHE];
    uint32_t      numEntries;

    // Cumulative stats
    uint32_t      totalRuns;
    uint32_t      cacheHits;
    uint32_t      compiles;
    double        totalExecMs;
    double        totalCompileMs;
    uint64_t      totalInsns;
};

static BridgeState g_bridge = {};

// FNV-1a hash
static uint64_t hash_code(const uint8_t* data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// ═══════════════════════════════════════════════════════════════
// NVRTC SPU Kernel Generation
// ═══════════════════════════════════════════════════════════════

// Generate a simple SPU interpreter kernel that runs on GPU.
// This is the fallback — the real HyperJIT would emit optimized code per-program.
// For the bridge, we emit a compact interpreter loop that handles core SPU ops.
static const char* spu_kernel_template =
"typedef unsigned int uint32_t;\n"
"typedef int int32_t;\n"
"typedef unsigned short uint16_t;\n"
"typedef short int16_t;\n"
"typedef unsigned char uint8_t;\n"
"typedef unsigned long long uint64_t;\n"
"\n"
"__device__ __forceinline__ uint32_t bswap32(uint32_t x) { return __byte_perm(x, 0, 0x0123); }\n"
"\n"
"// Fetch instruction from LS (big-endian)\n"
"__device__ __forceinline__ uint32_t ls_fetch(const uint8_t* ls, uint32_t pc) {\n"
"    uint32_t raw; memcpy(&raw, ls + (pc & 0x3FFFF), 4); return bswap32(raw);\n"
"}\n"
"\n"
"// SPU register file: 128 × 4 uint32 (word[0] = preferred scalar)\n"
"extern \"C\" __global__ void spu_bridge_kernel(\n"
"    uint8_t*  __restrict__ ls,\n"
"    uint32_t* __restrict__ regs,  // [128][4] flattened\n"
"    uint32_t* __restrict__ info)  // [0]=pc, [1]=status, [2]=stopped, [3]=max_insns, [4]=cycles_out\n"
"{\n"
"    if (threadIdx.x != 0) return;\n"
"    uint32_t pc = info[0];\n"
"    uint32_t max_insns = info[3];\n"
"    uint32_t cycles = 0;\n"
"\n"
"    while (cycles < max_insns) {\n"
"        uint32_t insn = ls_fetch(ls, pc);\n"
"        uint32_t op4  = (insn >> 28) & 0xF;\n"
"        uint32_t op7  = (insn >> 25) & 0x7F;\n"
"        uint32_t op8  = (insn >> 24) & 0xFF;\n"
"        uint32_t op9  = (insn >> 23) & 0x1FF;\n"
"        uint32_t op11 = (insn >> 21) & 0x7FF;\n"
"\n"
"        // RR-type fields\n"
"        uint32_t rt = (insn >> 0) & 0x7F;\n"
"        uint32_t ra = (insn >> 7) & 0x7F;\n"
"        uint32_t rb = (insn >> 14) & 0x7F;\n"
"\n"
"        // RI16 immediate\n"
"        int32_t i16 = (int32_t)((int16_t)((insn >> 7) & 0xFFFF));\n"
"        // RI10 immediate\n"
"        int32_t i10 = (int32_t)(((int32_t)((insn >> 14) & 0x3FF) << 22) >> 22);\n"
"        // RI7 immediate\n"
"        int32_t i7 = (int32_t)(((int32_t)((insn >> 14) & 0x7F) << 25) >> 25);\n"
"\n"
"        // Register access macros (word 0 = preferred scalar for most ops)\n"
"        #define R(i) regs[(i)*4]\n"
"        #define RW(i,w) regs[(i)*4+(w)]\n"
"\n"
"        bool handled = true;\n"
"        switch (op11) {\n"
"        // RR-type integer\n"
"        case 0x0C0: // a rt,ra,rb (ADD WORD)\n"
"            for(int w=0;w<4;w++) RW(rt,w) = RW(ra,w) + RW(rb,w); break;\n"
"        case 0x040: // sf rt,ra,rb (SUBTRACT FROM)\n"
"            for(int w=0;w<4;w++) RW(rt,w) = RW(rb,w) - RW(ra,w); break;\n"
"        case 0x0C1: // and rt,ra,rb\n"
"            for(int w=0;w<4;w++) RW(rt,w) = RW(ra,w) & RW(rb,w); break;\n"
"        case 0x041: // or rt,ra,rb\n"
"            for(int w=0;w<4;w++) RW(rt,w) = RW(ra,w) | RW(rb,w); break;\n"
"        case 0x241: // xor rt,ra,rb\n"
"            for(int w=0;w<4;w++) RW(rt,w) = RW(ra,w) ^ RW(rb,w); break;\n"
"        case 0x049: // nor rt,ra,rb\n"
"            for(int w=0;w<4;w++) RW(rt,w) = ~(RW(ra,w) | RW(rb,w)); break;\n"
"        case 0x000: // stop\n"
"            info[2] = 1; goto done;\n"
"        case 0x001: // lnop\n"
"        case 0x201: // nop\n"
"            break;\n"
"        case 0x1C4: // lqx rt,ra,rb (LOAD QUADWORD X-FORM)\n"
"        {\n"
"            uint32_t addr = (RW(ra,0) + RW(rb,0)) & 0x3FFF0;\n"
"            for(int w=0;w<4;w++) { uint32_t v; memcpy(&v, ls+addr+w*4, 4); RW(rt,w)=bswap32(v); }\n"
"            break;\n"
"        }\n"
"        case 0x144: // stqx rt,ra,rb (STORE QUADWORD X-FORM)\n"
"        {\n"
"            uint32_t addr = (RW(ra,0) + RW(rb,0)) & 0x3FFF0;\n"
"            for(int w=0;w<4;w++) { uint32_t v=bswap32(RW(rt,w)); memcpy(ls+addr+w*4, &v, 4); }\n"
"            break;\n"
"        }\n"
"        case 0x3C0: // ceq rt,ra,rb (COMPARE EQUAL WORD)\n"
"            for(int w=0;w<4;w++) RW(rt,w) = (RW(ra,w)==RW(rb,w)) ? 0xFFFFFFFF : 0; break;\n"
"        case 0x240: // cgt rt,ra,rb (COMPARE GREATER THAN WORD)\n"
"            for(int w=0;w<4;w++) RW(rt,w) = ((int32_t)RW(ra,w)>(int32_t)RW(rb,w)) ? 0xFFFFFFFF : 0; break;\n"
"        default:\n"
"            handled = false;\n"
"        }\n"
"\n"
"        if (!handled) {\n"
"            handled = true;\n"
"            switch (op9) {\n"
"            case 0x081: // il rt,i16 (IMMEDIATE LOAD WORD)\n"
"                for(int w=0;w<4;w++) RW(rt,w) = (uint32_t)i16; break;\n"
"            case 0x083: // ilh rt,i16\n"
"                { uint32_t v = ((uint32_t)(uint16_t)i16) | (((uint32_t)(uint16_t)i16) << 16);\n"
"                  for(int w=0;w<4;w++) RW(rt,w) = v; break; }\n"
"            case 0x082: // ilhu rt,i16\n"
"                for(int w=0;w<4;w++) RW(rt,w) = ((uint32_t)(uint16_t)i16) << 16; break;\n"
"            case 0x0C1: // iohl rt,i16\n"
"                for(int w=0;w<4;w++) RW(rt,w) |= (uint32_t)(uint16_t)i16; break;\n"
"            case 0x040: // brz rt,i16 (BRANCH IF ZERO)\n"
"                if (R(rt) == 0) { pc = (uint32_t)(i16 << 2) & 0x3FFFC; cycles++; continue; }\n"
"                break;\n"
"            case 0x042: // brnz rt,i16 (BRANCH IF NOT ZERO)\n"
"                if (R(rt) != 0) { pc = (uint32_t)(i16 << 2) & 0x3FFFC; cycles++; continue; }\n"
"                break;\n"
"            case 0x044: // brhz rt,i16\n"
"                if ((R(rt) & 0xFFFF) == 0) { pc = (uint32_t)(i16 << 2) & 0x3FFFC; cycles++; continue; }\n"
"                break;\n"
"            case 0x046: // brhnz rt,i16\n"
"                if ((R(rt) & 0xFFFF) != 0) { pc = (uint32_t)(i16 << 2) & 0x3FFFC; cycles++; continue; }\n"
"                break;\n"
"            case 0x064: // br i16 (BRANCH RELATIVE)\n"
"                pc = (uint32_t)(i16 << 2) & 0x3FFFC; cycles++; continue;\n"
"            default:\n"
"                handled = false;\n"
"            }\n"
"        }\n"
"\n"
"        if (!handled) {\n"
"            handled = true;\n"
"            switch (op8) {\n"
"            case 0x1C: // ai rt,ra,i10 (ADD IMMEDIATE)\n"
"                for(int w=0;w<4;w++) RW(rt,w) = RW(ra,w) + (uint32_t)i10; break;\n"
"            case 0x0C: // sfi rt,ra,i10 (SUBTRACT FROM IMMEDIATE)\n"
"                for(int w=0;w<4;w++) RW(rt,w) = (uint32_t)i10 - RW(ra,w); break;\n"
"            case 0x14: // andi rt,ra,i10\n"
"                for(int w=0;w<4;w++) RW(rt,w) = RW(ra,w) & (uint32_t)i10; break;\n"
"            case 0x04: // ori rt,ra,i10\n"
"                for(int w=0;w<4;w++) RW(rt,w) = RW(ra,w) | (uint32_t)i10; break;\n"
"            case 0x44: // xori rt,ra,i10\n"
"                for(int w=0;w<4;w++) RW(rt,w) = RW(ra,w) ^ (uint32_t)i10; break;\n"
"            case 0x7C: // ceqi rt,ra,i10\n"
"                for(int w=0;w<4;w++) RW(rt,w) = (RW(ra,w)==(uint32_t)i10) ? 0xFFFFFFFF : 0; break;\n"
"            case 0x4C: // cgti rt,ra,i10\n"
"                for(int w=0;w<4;w++) RW(rt,w) = ((int32_t)RW(ra,w)>(int32_t)i10) ? 0xFFFFFFFF : 0; break;\n"
"            case 0x34: // lqd rt,i10(ra) (LOAD QUADWORD D-FORM)\n"
"            {\n"
"                uint32_t addr = ((uint32_t)((int32_t)RW(ra,0) + (i10 << 4))) & 0x3FFF0;\n"
"                for(int w=0;w<4;w++) { uint32_t v; memcpy(&v, ls+addr+w*4, 4); RW(rt,w)=bswap32(v); }\n"
"                break;\n"
"            }\n"
"            case 0x24: // stqd rt,i10(ra) (STORE QUADWORD D-FORM)\n"
"            {\n"
"                uint32_t addr = ((uint32_t)((int32_t)RW(ra,0) + (i10 << 4))) & 0x3FFF0;\n"
"                for(int w=0;w<4;w++) { uint32_t v=bswap32(RW(rt,w)); memcpy(ls+addr+w*4, &v, 4); }\n"
"                break;\n"
"            }\n"
"            default:\n"
"                handled = false;\n"
"            }\n"
"        }\n"
"\n"
"        if (!handled) {\n"
"            // ILA (op7 = 0x21, RI18 format)\n"
"            if (op7 == 0x21) {\n"
"                uint32_t imm18 = (insn >> 7) & 0x3FFFF;\n"
"                for(int w=0;w<4;w++) RW(rt,w) = imm18;\n"
"            } else {\n"
"                // Unknown instruction — halt\n"
"                info[2] = 2; // unknown op\n"
"                break;\n"
"            }\n"
"        }\n"
"\n"
"        #undef R\n"
"        #undef RW\n"
"        pc = (pc + 4) & 0x3FFFC;\n"
"        cycles++;\n"
"    }\n"
"done:\n"
"    info[0] = pc;\n"
"    info[4] = cycles;\n"
"}\n";

// ═══════════════════════════════════════════════════════════════
// Bridge API Implementation
// ═══════════════════════════════════════════════════════════════

int spu_bridge_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

int spu_bridge_init(void) {
    if (g_bridge.initialized) return 0;

    if (!spu_bridge_available()) {
        fprintf(stderr, "[SPU-BRIDGE] No CUDA GPU found\n");
        return -1;
    }

    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&g_bridge.cuCtx, 0, dev);

    // Allocate persistent device memory
    cudaMalloc(&g_bridge.d_ls, LS_SIZE);
    cudaMalloc(&g_bridge.d_regs, 128 * 4 * sizeof(uint32_t));
    cudaMalloc(&g_bridge.d_info, 5 * sizeof(uint32_t));

    memset(&g_bridge.cache, 0, sizeof(g_bridge.cache));
    g_bridge.numEntries = 0;
    g_bridge.totalRuns = 0;
    g_bridge.cacheHits = 0;
    g_bridge.compiles = 0;
    g_bridge.totalExecMs = 0;
    g_bridge.totalCompileMs = 0;
    g_bridge.totalInsns = 0;
    g_bridge.initialized = true;

    fprintf(stderr, "[SPU-BRIDGE] Initialized (256KB LS on GPU)\n");
    return 0;
}

static CUfunction bridge_get_kernel(const uint8_t* ls) {
    // Hash the code portion of LS (first 64KB is typical code region)
    uint64_t h = hash_code(ls, 64 * 1024);

    // Check cache
    for (uint32_t i = 0; i < g_bridge.numEntries; i++) {
        if (g_bridge.cache[i].valid && g_bridge.cache[i].codeHash == h) {
            g_bridge.cache[i].hitCount++;
            g_bridge.cacheHits++;
            return g_bridge.cache[i].cuFunction;
        }
    }

    // Compile fresh kernel
    cudaEvent_t cStart, cStop;
    cudaEventCreate(&cStart); cudaEventCreate(&cStop);
    cudaEventRecord(cStart);

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, spu_kernel_template, "spu_bridge.cu", 0, NULL, NULL);
    const char* opts[] = { "--gpu-architecture=sm_70", "-use_fast_math", "-w" };
    nvrtcResult res = nvrtcCompileProgram(prog, 3, opts);

    if (res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = (char*)malloc(logSize + 1);
        nvrtcGetProgramLog(prog, log); log[logSize] = 0;
        fprintf(stderr, "[SPU-BRIDGE] NVRTC compile failed:\n%s\n", log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return nullptr;
    }

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = (char*)malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUmodule cuMod;
    CUfunction cuFunc;
    cuModuleLoadData(&cuMod, ptx);
    cuModuleGetFunction(&cuFunc, cuMod, "spu_bridge_kernel");
    free(ptx);

    cudaEventRecord(cStop);
    cudaEventSynchronize(cStop);
    float compMs = 0;
    cudaEventElapsedTime(&compMs, cStart, cStop);
    cudaEventDestroy(cStart); cudaEventDestroy(cStop);

    g_bridge.totalCompileMs += compMs;
    g_bridge.compiles++;

    // Store in cache
    if (g_bridge.numEntries < MAX_CACHE) {
        SPUCacheEntry& e = g_bridge.cache[g_bridge.numEntries++];
        e.codeHash = h;
        e.cuModule = cuMod;
        e.cuFunction = cuFunc;
        e.hitCount = 0;
        e.valid = true;
    }

    return cuFunc;
}

int spu_bridge_run(uint8_t* ls, SPUBridgeState* state,
                   uint32_t max_insns, SPUBridgeStats* stats) {
    if (!g_bridge.initialized) {
        if (spu_bridge_init() != 0) return -1;
    }

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart); cudaEventCreate(&tStop);
    cudaEventRecord(tStart);

    // Get compiled kernel (cached or fresh)
    CUfunction func = bridge_get_kernel(ls);
    if (!func) return -1;

    bool cached = (g_bridge.cacheHits > 0 &&
                   g_bridge.cache[g_bridge.numEntries-1].hitCount > 0);

    // Upload LS + registers + info to GPU
    cudaMemcpy(g_bridge.d_ls, ls, LS_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(g_bridge.d_regs, state->gpr, 128 * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t info[5] = { state->pc, state->status, state->stopped, max_insns, 0 };
    cudaMemcpy(g_bridge.d_info, info, 5 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    cudaEvent_t execStart, execStop;
    cudaEventCreate(&execStart); cudaEventCreate(&execStop);
    cudaEventRecord(execStart);

    void* args[] = { &g_bridge.d_ls, &g_bridge.d_regs, &g_bridge.d_info };
    CUresult err = cuLaunchKernel(func, 1,1,1, 1,1,1, 0,0, args, NULL);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[SPU-BRIDGE] Launch failed: %d\n", err);
        cudaEventDestroy(execStart); cudaEventDestroy(execStop);
        cudaEventDestroy(tStart); cudaEventDestroy(tStop);
        return -1;
    }

    cudaEventRecord(execStop);
    cudaEventSynchronize(execStop);
    float execMs = 0;
    cudaEventElapsedTime(&execMs, execStart, execStop);
    cudaEventDestroy(execStart); cudaEventDestroy(execStop);

    // Read back results
    cudaMemcpy(ls, g_bridge.d_ls, LS_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(state->gpr, g_bridge.d_regs, 128 * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(info, g_bridge.d_info, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    state->pc = info[0];
    state->status = info[1];
    state->stopped = info[2];

    uint32_t cyclesRun = info[4];

    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);
    float totalMs = 0;
    cudaEventElapsedTime(&totalMs, tStart, tStop);
    cudaEventDestroy(tStart); cudaEventDestroy(tStop);

    // Update cumulative stats
    g_bridge.totalRuns++;
    g_bridge.totalExecMs += execMs;
    g_bridge.totalInsns += cyclesRun;

    // Fill output stats
    if (stats) {
        stats->total_ms = totalMs;
        stats->exec_ms = execMs;
        stats->cycles = cyclesRun;
        stats->cache_hit = cached ? 1 : 0;
        stats->mips = (execMs > 0) ? (double)cyclesRun / (execMs * 1000.0) : 0;
    }

    return (int)cyclesRun;
}

int spu_bridge_run_cached(uint8_t* ls, SPUBridgeState* state,
                          uint32_t max_insns, SPUBridgeStats* stats) {
    // Same as spu_bridge_run — caching is automatic
    return spu_bridge_run(ls, state, max_insns, stats);
}

void spu_bridge_print_stats(void) {
    double avgMips = (g_bridge.totalExecMs > 0) ?
        (double)g_bridge.totalInsns / (g_bridge.totalExecMs * 1000.0) : 0;

    fprintf(stderr,
        "╔═══════════════════════════════════════════╗\n"
        "║  SPU CUDA Bridge Statistics                ║\n"
        "╠═══════════════════════════════════════════╣\n"
        "║  Total runs:    %-8u                   ║\n"
        "║  Cache hits:    %-8u                   ║\n"
        "║  Compiles:      %-8u                   ║\n"
        "║  Total insns:   %-12llu               ║\n"
        "║  Exec time:     %8.2f ms               ║\n"
        "║  Compile time:  %8.2f ms               ║\n"
        "║  Avg throughput: %.1f MIPS              ║\n"
        "╚═══════════════════════════════════════════╝\n",
        g_bridge.totalRuns,
        g_bridge.cacheHits,
        g_bridge.compiles,
        (unsigned long long)g_bridge.totalInsns,
        g_bridge.totalExecMs,
        g_bridge.totalCompileMs,
        avgMips);
}

void spu_bridge_shutdown(void) {
    if (!g_bridge.initialized) return;

    // Free cached modules
    for (uint32_t i = 0; i < g_bridge.numEntries; i++) {
        if (g_bridge.cache[i].valid && g_bridge.cache[i].cuModule)
            cuModuleUnload(g_bridge.cache[i].cuModule);
    }

    cudaFree(g_bridge.d_ls);
    cudaFree(g_bridge.d_regs);
    cudaFree(g_bridge.d_info);

    cuCtxDestroy(g_bridge.cuCtx);
    g_bridge.initialized = false;
    fprintf(stderr, "[SPU-BRIDGE] Shutdown\n");
}
