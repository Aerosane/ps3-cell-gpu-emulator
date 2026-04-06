// spu_jit.h — SPU Basic-Block JIT: SPU instructions → native CUDA kernels
//
// Strategy:
//   1. Trace SPU execution to discover basic blocks (linear instruction sequences)
//   2. Translate each block's SPU instructions → CUDA C++ source
//   3. Compile with NVRTC at runtime → cuFunction
//   4. Cache compiled blocks keyed by (entry_pc, LS content hash)
//   5. Dispatch: hot path → native kernel, cold path → interpreter fallback
//
// Why this works:
//   - SPU SPURS jobs are small (~256-2048 instructions), self-contained
//   - 128-bit SIMD → native float4/uint4 — no emulation needed
//   - 256KB Local Store fits in L2 (6MB on V100)
//   - Compiled code eliminates: global mem fetch/insn, switch decode, branch mispredicts
//   - Expected speedup: 100-200× over interpreter
//
#pragma once
#include "spu_defs.h"
#include <cstdint>
#include <cstddef>

#ifdef __CUDACC__
#define JIT_HD __host__ __device__
#else
#define JIT_HD
#endif

namespace spu_jit {

// ═══════════════════════════════════════════════════════════════
// Basic Block Representation
// ═══════════════════════════════════════════════════════════════

static constexpr uint32_t MAX_BLOCK_INSNS = 512;  // max instructions per block
static constexpr uint32_t MAX_BLOCKS      = 4096; // max cached blocks
static constexpr uint32_t MAX_SOURCE_SIZE = 128 * 1024; // 128KB source per block

// Decoded instruction — enough info to emit CUDA C++ without re-decoding
struct DecodedInsn {
    uint32_t raw;       // raw 32-bit instruction word
    uint32_t pc;        // address in LS
    uint8_t  format;    // 0=RR(11), 1=RI16(9), 2=RI10(8), 3=RI18(7), 4=RRR(4)
    uint8_t  isBranch;  // 1 = this instruction is a branch (ends block)
    uint8_t  isCall;    // 1 = branch-and-link (BRSL, BISL, etc.)
    uint8_t  isStop;    // 1 = STOP/STOPD
    uint16_t opcode;    // format-specific opcode
    uint8_t  rT, rA, rB, rC; // register operands
    int32_t  imm;       // immediate value (sign-extended where appropriate)
    uint32_t immU;      // unsigned immediate
};

// A basic block: linear sequence of instructions ending at a branch/stop
struct BasicBlock {
    uint32_t     entryPC;                   // start address in LS
    uint32_t     endPC;                     // address of last instruction
    uint32_t     numInsns;                  // instruction count
    DecodedInsn  insns[MAX_BLOCK_INSNS];    // decoded instructions
    uint64_t     lsHash;                    // hash of LS bytes [entryPC, endPC+4)
    bool         usesRegs[128];             // which registers are referenced
    bool         writesRegs[128];           // which registers are written
    bool         readsLS;                   // does any insn read from LS?
    bool         writesLS;                  // does any insn write to LS?
    bool         usesMFC;                   // does any insn do MFC/channel ops?
    bool         hasIndirectBranch;         // BI/BISL — can't predict target
};

// ═══════════════════════════════════════════════════════════════
// JIT Cache Entry
// ═══════════════════════════════════════════════════════════════

struct JITEntry {
    uint32_t  entryPC;
    uint64_t  lsHash;
    void*     cuModule;    // CUmodule (opaque)
    void*     cuFunction;  // CUfunction (opaque)
    uint32_t  numInsns;    // instructions in block
    uint32_t  hitCount;    // execution count
    bool      valid;
};

// ═══════════════════════════════════════════════════════════════
// JIT Compiler State (host-side)
// ═══════════════════════════════════════════════════════════════

struct JITState {
    JITEntry  cache[MAX_BLOCKS];
    uint32_t  numEntries;
    uint32_t  compileCount;     // total compilations
    uint32_t  cacheHits;
    uint32_t  cacheMisses;
    uint32_t  interpreterFallbacks;
    float     totalCompileTimeMs;
    bool      ready;
};

// ═══════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════

// Initialize JIT compiler
int  jit_init(JITState* state);

// Discover and decode a basic block starting at entryPC in the given LS image
int  jit_discover_block(const uint8_t* ls, uint32_t entryPC, BasicBlock* out);

// Generate CUDA C++ source for a basic block
int  jit_emit_source(const BasicBlock* block, char* srcBuf, size_t bufSize);

// Compile source via NVRTC → cuModule + cuFunction
int  jit_compile(JITState* state, const BasicBlock* block,
                 const char* source, JITEntry* out);

// Execute a compiled block on SPU state (host-side launch)
int  jit_execute(const JITEntry* entry, spu::SPUState* d_state,
                 uint8_t* d_ls, uint8_t* d_mainMem);

// Full pipeline: discover → compile → cache → execute (with interpreter fallback)
int  jit_run_block(JITState* state, spu::SPUState* d_state,
                   uint8_t* d_ls, uint8_t* d_mainMem,
                   uint32_t entryPC);

// Run SPU with JIT for N cycles, falling back to interpreter for uncached blocks
float jit_run_spu(JITState* state, int spuId, spu::SPUState* d_states,
                  uint8_t** d_localStores, uint8_t* d_mainMem,
                  uint32_t maxCycles);

// Print cache statistics
void jit_print_stats(const JITState* state);

// Cleanup
void jit_shutdown(JITState* state);

} // namespace spu_jit
