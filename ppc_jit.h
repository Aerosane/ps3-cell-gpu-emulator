// ppc_jit.h — PPE Basic-Block JIT: PowerPC instructions → native CUDA kernels
//
// Strategy (mirrors SPU JIT pipeline):
//   1. Trace PPE execution to discover basic blocks (linear instruction sequences)
//   2. Translate each block's PPC instructions → CUDA C++ source with register promotion
//   3. Compile with NVRTC at runtime → cuFunction
//   4. Cache compiled blocks keyed by (entry_pc, memory content hash)
//   5. Dispatch: hot path → native kernel, cold path → interpreter fallback
//
// Why this works:
//   - PPE code in PS3 games is mostly 32-bit mode integer / branch / load-store
//   - Register promotion: 32 GPRs fit in CUDA registers (vs 128×128b for SPU)
//   - Compiled code eliminates: global mem fetch/insn, switch decode, branch mispredicts
//   - Expected speedup: 10-50× over interpreter
//
#pragma once
#include "ppc_defs.h"
#include <cstdint>
#include <cstddef>

#ifdef __CUDACC__
#define PPC_JIT_HD __host__ __device__
#else
#define PPC_JIT_HD
#endif

namespace ppc_jit {

// ═══════════════════════════════════════════════════════════════
// Basic Block Representation
// ═══════════════════════════════════════════════════════════════

static constexpr uint32_t MAX_BLOCK_INSNS  = 512;
static constexpr uint32_t MAX_BLOCKS       = 4096;
static constexpr uint32_t MAX_SOURCE_SIZE  = 256 * 1024; // 256KB source per block

// Decoded PPC instruction
struct PPCDecodedInsn {
    uint32_t raw;       // raw 32-bit instruction word
    uint64_t pc;        // address in PS3 memory
    uint32_t opcd;      // primary opcode (6-bit)
    uint32_t xo;        // extended opcode (for groups 19/31/59/63)
    uint32_t rd;        // rD / rS / fD / fS field
    uint32_t ra;        // rA field
    uint32_t rb;        // rB field
    int64_t  imm;       // sign-extended immediate
    uint64_t uimm;      // unsigned immediate
    uint32_t sh, mb, me;// rotate fields
    uint32_t bo, bi;    // branch fields
    uint32_t spr;       // SPR number
    uint32_t crm;       // CRM field for mtcrf
    uint8_t  isBranch;  // 1 = terminates block (B, BC, BCLR, BCCTR)
    uint8_t  isSyscall; // 1 = SC instruction
    uint8_t  isUnimpl;  // 1 = unhandled opcode
    uint8_t  rc;        // Rc bit
    uint8_t  lk;        // LK bit
    uint8_t  aa;        // AA bit
};

// A basic block: linear sequence ending at a branch/SC/unimplemented
struct PPCBasicBlock {
    uint64_t         entryPC;
    uint64_t         endPC;
    uint32_t         numInsns;
    PPCDecodedInsn   insns[MAX_BLOCK_INSNS];
    uint64_t         memHash;     // hash of memory [entryPC, endPC+4)
    bool             usesGPR[32]; // which GPRs are referenced
    bool             writesGPR[32];
    bool             usesFPR[32];
    bool             writesFPR[32];
    bool             usesCR;
    bool             usesLR;
    bool             usesCTR;
    bool             usesXER;
    bool             writesCR;
    bool             writesLR;
    bool             writesCTR;
    bool             writesXER;
    bool             accessesMem; // load/store instructions present
};

// ═══════════════════════════════════════════════════════════════
// JIT Cache Entry
// ═══════════════════════════════════════════════════════════════

struct PPCJITEntry {
    uint64_t  entryPC;
    uint64_t  memHash;
    void*     cuModule;    // CUmodule
    void*     cuFunction;  // CUfunction
    uint32_t  numInsns;
    uint32_t  hitCount;
    bool      valid;
};

// ═══════════════════════════════════════════════════════════════
// JIT Compiler State (host-side)
// ═══════════════════════════════════════════════════════════════

struct PPCJITState {
    PPCJITEntry  cache[MAX_BLOCKS];
    uint32_t     numEntries;
    uint32_t     compileCount;
    uint32_t     cacheHits;
    uint32_t     cacheMisses;
    uint32_t     interpreterFallbacks;
    float        totalCompileTimeMs;
    bool         ready;
};

// ═══════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════

int  ppc_jit_init(PPCJITState* state);

int  ppc_jit_discover_block(const uint8_t* mem, uint64_t entryPC,
                            PPCBasicBlock* out);

int  ppc_jit_emit_source(const PPCBasicBlock* block, char* srcBuf,
                         size_t bufSize);

int  ppc_jit_compile(PPCJITState* state, const PPCBasicBlock* block,
                     const char* source, PPCJITEntry* out);

int  ppc_jit_execute(const PPCJITEntry* entry, ppc::PPEState* h_state,
                     uint8_t* d_mem);

int  ppc_jit_run(PPCJITState* state, ppc::PPEState* h_state,
                 uint8_t* d_mem, uint32_t maxCycles,
                 float* outMs, uint32_t* outCycles);

// Superblock JIT: discover ALL reachable blocks, emit ONE kernel with
// PC-dispatch loop, single launch. Eliminates per-block kernel overhead.
int  ppc_jit_run_fast(PPCJITState* state, ppc::PPEState* h_state,
                      uint8_t* d_mem, uint32_t maxCycles,
                      float* outMs, uint32_t* outCycles);

void ppc_jit_print_stats(const PPCJITState* state);

void ppc_jit_shutdown(PPCJITState* state);

} // namespace ppc_jit
