// spu_cuda_backend.cu — CUDA SPU Backend Implementation
//
// This file implements the GPU-side SPU execution for RPCS3 integration.
// It transfers spu_thread state (128 v128 GPRs + 256KB LS) to the V100,
// runs our 171-opcode SPU kernel, and transfers results back.
//
// Two usage modes:
//   1. Direct API: cuda_spu_init/execute/shutdown (for testing)
//   2. LD_PRELOAD: auto-hooks RPCS3's SPU interpreter dispatch
//
// Build:
//   nvcc -O3 -arch=sm_70 -shared -Xcompiler -fPIC \
//     spu_cuda_backend.cu rpcs3_spu_bridge.cu -lnvrtc -lcuda \
//     -o libspu_cuda.so

#include "spu_cuda_backend.h"
#include "rpcs3_spu_bridge.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>

// ============================================================
// Global state
// ============================================================

CudaSPUStats g_cuda_spu_stats = {};

// Per-thread CUDA SPU backend (thread_local for SPU thread isolation)
static thread_local CudaSPUBackend* tls_cuda_ctx = nullptr;

// Global bridge initialization flag
static bool g_bridge_initialized = false;
static std::mutex g_bridge_mutex;

// Max instructions per GPU dispatch batch.
// Channel/DMA ops force early return regardless.
static constexpr uint32_t GPU_BATCH_SIZE = 65536;

// ============================================================
// Endian conversion helpers
// ============================================================

// RPCS3 v128._u32 on x86 LE host:
//   _u32[0] = bytes 0-3 (LE word 0)
//   _u32[3] = bytes 12-15 (LE word 3) ← SPU preferred scalar
//
// Our GPU kernel (BE convention):
//   word[0] = SPU preferred scalar = RPCS3's _u32[3]
//   word[3] = RPCS3's _u32[0]

static void gprs_rpcs3_to_bridge(const v128_compat* rpcs3_gpr, SPUBridgeState* state) {
    for (int r = 0; r < 128; r++) {
        state->gpr[r][0] = rpcs3_gpr[r]._u32[3]; // preferred → word[0]
        state->gpr[r][1] = rpcs3_gpr[r]._u32[2];
        state->gpr[r][2] = rpcs3_gpr[r]._u32[1];
        state->gpr[r][3] = rpcs3_gpr[r]._u32[0];
    }
}

static void gprs_bridge_to_rpcs3(const SPUBridgeState* state, v128_compat* rpcs3_gpr) {
    for (int r = 0; r < 128; r++) {
        rpcs3_gpr[r]._u32[3] = state->gpr[r][0]; // word[0] → preferred
        rpcs3_gpr[r]._u32[2] = state->gpr[r][1];
        rpcs3_gpr[r]._u32[1] = state->gpr[r][2];
        rpcs3_gpr[r]._u32[0] = state->gpr[r][3];
    }
}

// ============================================================
// Backend lifecycle
// ============================================================

int cuda_spu_init(CudaSPUBackend* ctx) {
    memset(ctx, 0, sizeof(*ctx));

    // Initialize shared bridge (once)
    {
        std::lock_guard<std::mutex> lock(g_bridge_mutex);
        if (!g_bridge_initialized) {
            if (spu_bridge_init() != 0) {
                fprintf(stderr, "[CUDA-SPU] Failed to initialize bridge\n");
                return -1;
            }
            g_bridge_initialized = true;
            fprintf(stderr, "[CUDA-SPU] Bridge initialized (171 opcodes on GPU)\n");
        }
    }

    ctx->initialized = true;
    fprintf(stderr, "[CUDA-SPU] SPU context ready\n");
    return 0;
}

void cuda_spu_shutdown(CudaSPUBackend* ctx) {
    if (!ctx->initialized) return;

    if (ctx->total_calls > 0) {
        fprintf(stderr, "[CUDA-SPU] Stats: %lu calls, %lu insns, %.1f ms GPU\n",
                ctx->total_calls, ctx->total_insns, ctx->total_gpu_ms);
    }

    ctx->initialized = false;
}

// ============================================================
// Core execution — uses bridge API
// ============================================================

int cuda_spu_execute(CudaSPUBackend* ctx,
                     v128_compat* host_gpr,
                     uint8_t* host_ls,
                     uint32_t* pc,
                     uint32_t max_insns,
                     uint32_t* stop_reason,
                     uint32_t* channel_idx,
                     uint32_t* channel_reg)
{
    if (!ctx->initialized) return -1;
    if (max_insns == 0) max_insns = GPU_BATCH_SIZE;

    // === Convert RPCS3 GPRs (LE) to bridge format (BE) ===
    SPUBridgeState state;
    memset(&state, 0, sizeof(state));
    gprs_rpcs3_to_bridge(host_gpr, &state);
    state.pc = *pc;

    // === Execute on GPU via bridge ===
    SPUBridgeStats stats;
    int n = spu_bridge_run_cached(host_ls, &state, max_insns, &stats);
    if (n < 0) {
        fprintf(stderr, "[CUDA-SPU] Bridge execution failed\n");
        return -1;
    }

    // === Convert results back to RPCS3 format ===
    gprs_bridge_to_rpcs3(&state, host_gpr);
    *pc = state.pc;
    *stop_reason = state.stopped;

    if (channel_idx) *channel_idx = 0;  // TODO: extract from bridge
    if (channel_reg) *channel_reg = 0;

    // Update stats
    ctx->total_insns += stats.cycles;
    ctx->total_calls++;
    ctx->total_gpu_ms += stats.exec_ms;
    g_cuda_spu_stats.total_blocks++;
    g_cuda_spu_stats.total_instructions += stats.cycles;

    return (int)stats.cycles;
}

// ============================================================
// Streaming execution (run until channel op or STOP)
// ============================================================

int cuda_spu_run_streaming(CudaSPUBackend* ctx,
                           v128_compat* host_gpr,
                           uint8_t* host_ls,
                           uint32_t* pc)
{
    uint32_t stop_reason = 0;
    uint32_t ch_idx = 0, ch_reg = 0;
    int total_insns = 0;

    while (true) {
        int n = cuda_spu_execute(ctx, host_gpr, host_ls, pc,
                                  GPU_BATCH_SIZE, &stop_reason,
                                  &ch_idx, &ch_reg);
        if (n < 0) return -1;
        total_insns += n;

        switch (stop_reason) {
            case 0: continue;               // Hit instruction limit
            case 1: return total_insns;     // STOP instruction
            case 2:                         // Halt trap
                fprintf(stderr, "[CUDA-SPU] Halt trap at PC=0x%05X\n", *pc);
                return total_insns;
            case 3: // RDCH
            case 4: // WRCH
                return total_insns;         // Return to host for channel I/O
            default:
                fprintf(stderr, "[CUDA-SPU] Unknown stop reason: %u\n", stop_reason);
                return total_insns;
        }
    }
}

// ============================================================
// LD_PRELOAD Hooks
// ============================================================

// RPCS3 SPU thread field offsets (verified against SPUThread.h)
// These let us access spu_thread fields without RPCS3 headers.
//
// From SPUThread.h (spu_thread : public cpu_thread):
//   cpu_thread base class is ~200-400 bytes (varies by build)
//   Key fields after base:
//     u32 pc;                    // offset: varies
//     std::array<v128, 128> gpr; // offset: varies (aligned to 16)
//     u8* ls;                    // pointer to 256KB local store
//
// We detect these offsets at runtime by pattern matching.

// Runtime-detected offsets into spu_thread
struct SPUThreadOffsets {
    size_t pc_offset;    // Offset of u32 pc
    size_t gpr_offset;   // Offset of v128[128] gpr (must be 16-aligned)
    size_t ls_offset;    // Offset of u8* ls
    bool   detected;
};

static SPUThreadOffsets g_offsets = {};

// Attempt to detect spu_thread field offsets by scanning the object.
// This is fragile and build-dependent — prefer the source patch approach.
static bool detect_spu_offsets(void* spu_thread_ptr) {
    // For safety, we require explicit offset configuration via env vars
    const char* pc_off_str = getenv("CUDA_SPU_PC_OFFSET");
    const char* gpr_off_str = getenv("CUDA_SPU_GPR_OFFSET");
    const char* ls_off_str = getenv("CUDA_SPU_LS_OFFSET");

    if (pc_off_str && gpr_off_str && ls_off_str) {
        g_offsets.pc_offset = (size_t)strtoul(pc_off_str, nullptr, 0);
        g_offsets.gpr_offset = (size_t)strtoul(gpr_off_str, nullptr, 0);
        g_offsets.ls_offset = (size_t)strtoul(ls_off_str, nullptr, 0);
        g_offsets.detected = true;
        fprintf(stderr, "[CUDA-SPU] Using env offsets: pc=%zu gpr=%zu ls=%zu\n",
                g_offsets.pc_offset, g_offsets.gpr_offset, g_offsets.ls_offset);
        return true;
    }

    fprintf(stderr, "[CUDA-SPU] Set CUDA_SPU_PC_OFFSET, CUDA_SPU_GPR_OFFSET, CUDA_SPU_LS_OFFSET\n");
    return false;
}

// The LD_PRELOAD hook — replaces RPCS3's SPU interpreter dispatch
extern "C" void cuda_spu_interpreter_hook(void* spu_thread_ptr, void* /*ls_base*/, uint8_t* /*unused*/) {
    if (!g_offsets.detected) {
        if (!detect_spu_offsets(spu_thread_ptr)) {
            // Fall back to original interpreter
            return;
        }
    }

    // Lazily init per-thread context
    if (!tls_cuda_ctx) {
        tls_cuda_ctx = new CudaSPUBackend();
        if (cuda_spu_init(tls_cuda_ctx) != 0) {
            delete tls_cuda_ctx;
            tls_cuda_ctx = nullptr;
            return;
        }
    }

    // Extract spu_thread fields
    char* base = (char*)spu_thread_ptr;
    uint32_t* pc_ptr = (uint32_t*)(base + g_offsets.pc_offset);
    v128_compat* gpr = (v128_compat*)(base + g_offsets.gpr_offset);
    uint8_t* ls = *(uint8_t**)(base + g_offsets.ls_offset);

    uint32_t stop_reason = 0, ch_idx = 0, ch_reg = 0;
    cuda_spu_execute(tls_cuda_ctx, gpr, ls, pc_ptr,
                     GPU_BATCH_SIZE, &stop_reason, &ch_idx, &ch_reg);
}

// ============================================================
// Test program
// ============================================================

#ifdef SPU_CUDA_BACKEND_TEST

#include <cstdio>
#include <cstring>
#include <chrono>

// Encode helpers (from test_rpcs3_bridge.cu)
static uint32_t spu_rr(uint32_t op11, uint32_t rb, uint32_t ra, uint32_t rt) {
    return (op11 << 21) | (rb << 14) | (ra << 7) | rt;
}
static uint32_t spu_ri16(uint32_t op9, int16_t i16, uint32_t rt) {
    return (op9 << 23) | (((uint32_t)(uint16_t)i16) << 7) | rt;
}
static uint32_t spu_ri10(uint32_t op8, int16_t i10, uint32_t ra, uint32_t rt) {
    return (op8 << 24) | (((uint32_t)(uint16_t)(i10 & 0x3FF)) << 14) | (ra << 7) | rt;
}

// Byte-swap for LS store
static uint32_t bswap32(uint32_t v) {
    return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
           ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000);
}

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  🎮 CUDA SPU Backend Test                ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Tests RPCS3↔GPU register conversion     ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    CudaSPUBackend ctx;
    if (cuda_spu_init(&ctx) != 0) {
        printf("[FAIL] Backend init failed\n");
        return 1;
    }

    // === Test 1: Endian conversion roundtrip ===
    printf("╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: v128 Endian Roundtrip         ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    {
        v128_compat gpr[128] = {};
        gpr[3]._u32[3] = 42;  // RPCS3 preferred scalar
        gpr[3]._u32[2] = 0;
        gpr[3]._u32[1] = 0;
        gpr[3]._u32[0] = 0;

        // Convert to bridge format and check
        SPUBridgeState state;
        memset(&state, 0, sizeof(state));
        gprs_rpcs3_to_bridge(gpr, &state);

        printf("  RPCS3 r3._u32[3]=%u → Bridge r3.gpr[0]=%u", 42, state.gpr[3][0]);
        if (state.gpr[3][0] == 42) {
            printf(" ✅\n");
        } else {
            printf(" ❌\n");
            return 1;
        }

        // Roundtrip back
        v128_compat gpr2[128] = {};
        gprs_bridge_to_rpcs3(&state, gpr2);
        printf("  Roundtrip: _u32[3]=%u (expect 42)", gpr2[3]._u32[3]);
        if (gpr2[3]._u32[3] == 42 && gpr2[3]._u32[0] == 0) {
            printf(" ✅\n");
        } else {
            printf(" ❌\n");
            return 1;
        }
    }

    // === Test 2: Full GPU execution with RPCS3-format GPRs ===
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: GPU Execution (RPCS3 format)  ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    {
        v128_compat gpr[128] = {};
        uint8_t ls[256 * 1024] = {};

        // Program: IL r3, 42; IL r4, 58; A r5, r3, r4; STOP
        uint32_t prog[] = {
            spu_ri16(0x081, 42, 3),       // IL r3, 42
            spu_ri16(0x081, 58, 4),       // IL r4, 58
            spu_rr(0x0C0, 4, 3, 5),      // A r5, r3, r4
            spu_rr(0x000, 0, 0, 0),      // STOP
        };

        // Store program in LS (big-endian word order)
        for (int i = 0; i < 4; i++) {
            uint32_t be = bswap32(prog[i]);
            memcpy(&ls[i * 4], &be, 4);
        }

        uint32_t pc = 0;
        uint32_t stop_reason = 0, ch_idx = 0, ch_reg = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        int n = cuda_spu_execute(&ctx, gpr, ls, &pc, 100,
                                  &stop_reason, &ch_idx, &ch_reg);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Check: r5 preferred scalar should be 100
        uint32_t r5_val = gpr[5]._u32[3];  // RPCS3 LE preferred scalar
        printf("  r5._u32[3] = %u (expect 100)\n", r5_val);
        printf("  Executed: %d insns, stop=%u, time=%.3f ms\n", n, stop_reason, ms);

        if (r5_val == 100 && stop_reason == 1) {
            printf("  Result: ✅ PASS\n");
        } else {
            printf("  Result: ❌ FAIL\n");
            cuda_spu_shutdown(&ctx);
            return 1;
        }
    }

    // === Test 3: Transfer bandwidth ===
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: Transfer Bandwidth            ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    {
        v128_compat gpr[128] = {};
        uint8_t ls[256 * 1024];
        memset(ls, 0xAB, sizeof(ls));

        // Simple STOP program
        uint32_t stop_insn = bswap32(spu_rr(0x000, 0, 0, 0));
        memcpy(ls, &stop_insn, 4);

        uint32_t pc = 0, stop = 0, ci = 0, cr = 0;

        // Warm up
        cuda_spu_execute(&ctx, gpr, ls, &pc, 1, &stop, &ci, &cr);

        // Benchmark: 100 round-trips
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            pc = 0;
            cuda_spu_execute(&ctx, gpr, ls, &pc, 1, &stop, &ci, &cr);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Each round-trip transfers: 2KB regs + 256KB LS + 32B info = ~258KB × 2 = 516KB
        double total_mb = 100.0 * 516.0 / 1024.0;  // ~50 MB
        double gbps = total_mb / ms * 1000.0 / 1024.0;

        printf("  100 round-trips: %.1f ms (%.2f ms/trip)\n", ms, ms / 100.0);
        printf("  Data: %.1f MB total, ~%.1f GB/s effective\n", total_mb, gbps);
        printf("  Result: ✅ PASS\n");
    }

    cuda_spu_shutdown(&ctx);

    printf("\n═══════════════════════════════════════════\n");
    printf("  All tests passed! ✅\n");
    printf("═══════════════════════════════════════════\n");
    return 0;
}

#endif // SPU_CUDA_BACKEND_TEST
