// rpcs3_spu_bridge.h — RPCS3 ↔ CUDA SPU Bridge
//
// Drop-in accelerator for PS3 SPU emulation. Provides a C API that
// any emulator (RPCS3, our megakernel, standalone) can call to offload
// SPU execution to NVIDIA GPUs via CUDA.
//
// Integration options:
//   1. LD_PRELOAD interception (zero RPCS3 source changes)
//   2. RPCS3 source patch (add as spu_decoder = "CUDA")
//   3. Standalone benchmark (test_rpcs3_bridge.cu)
//
// Architecture:
//   Host (CPU)                          Device (GPU)
//   ┌──────────┐   cudaMemcpy    ┌──────────────────┐
//   │ RPCS3    │ ──────────────→ │ SPU Local Store  │
//   │ SPU state│   (256KB LS)    │ 128 QWord regs   │
//   └──────────┘                 │ HyperJIT kernel  │
//        ↑       cudaMemcpy      │ (NVRTC compiled) │
//        └─────────────────────← └──────────────────┘
//
#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// SPU register state (matches RPCS3's spu_thread layout)
typedef struct {
    uint32_t gpr[128][4];   // 128 × 128-bit vector registers (big-endian word order)
    uint32_t pc;            // program counter (0-0x3FFFC, 18-bit aligned)
    uint32_t status;        // SPU status register
    uint32_t ch_tag_mask;   // MFC tag mask
    uint32_t ch_event_stat; // event status
    uint32_t srr0;          // SRR0 (saved PC for interrupts)
    uint32_t stopped;       // 1 if SPU halted (STOP/STOPD)
} SPUBridgeState;

// Statistics from a GPU execution run
typedef struct {
    float    total_ms;      // total time including any JIT compile
    float    exec_ms;       // kernel execution time only
    uint32_t cycles;        // instructions executed
    uint32_t cache_hit;     // 1 if cubin was cached, 0 if fresh compile
    float    mips;          // millions of instructions per second (exec-only)
} SPUBridgeStats;

// ═══════════════════════════════════════════════════════════════
// Bridge API
// ═══════════════════════════════════════════════════════════════

// Initialize the CUDA SPU bridge. Call once at startup.
// Returns 0 on success, -1 on failure.
int spu_bridge_init(void);

// Execute SPU program on GPU.
//   ls       — 256KB Local Store contents (will be copied to/from GPU)
//   state    — SPU register state (in/out)
//   max_insns — max instructions to execute before returning
//   stats    — optional output statistics (NULL to skip)
// Returns: number of instructions executed, or -1 on error.
int spu_bridge_run(uint8_t* ls, SPUBridgeState* state,
                   uint32_t max_insns, SPUBridgeStats* stats);

// Execute SPU program using cached cubin (fast path).
// Same as spu_bridge_run but skips any recompilation if a cached
// kernel exists for this LS code region.
int spu_bridge_run_cached(uint8_t* ls, SPUBridgeState* state,
                          uint32_t max_insns, SPUBridgeStats* stats);

// Check if GPU acceleration is available.
// Returns 1 if a CUDA-capable GPU is present, 0 otherwise.
int spu_bridge_available(void);

// Print bridge statistics summary.
void spu_bridge_print_stats(void);

// Shutdown and free all GPU resources.
void spu_bridge_shutdown(void);

#ifdef __cplusplus
}
#endif
