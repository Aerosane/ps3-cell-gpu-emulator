// spu_cuda_backend.h — CUDA SPU Backend for RPCS3
//
// Bridges RPCS3's spu_thread state to our GPU SPU kernel.
// Transfers v128 GPRs + 256KB LS to device, executes on V100,
// transfers results back. MFC DMA commands bounce through host memory.
//
// Usage: LD_PRELOAD=libspu_cuda.so rpcs3
//   or: patch RPCS3's init_spu_decoder() to call make_cuda_backend()

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <atomic>

// ============================================================
// RPCS3 type stubs (matching RPCS3's ABI without full headers)
// ============================================================

// v128: RPCS3 uses union v128 with masked_array_t for endian-aware access.
// On LE host (x86): _u32[0] = bytes 0-3, _u32[3] = bytes 12-15 = preferred scalar
// Our GPU kernel uses BE convention: word[0] = preferred scalar
struct alignas(16) v128_compat {
    uint32_t _u32[4];  // LE layout: [0]=bytes 0-3, [3]=bytes 12-15 (preferred)
};

// ============================================================
// CUDA SPU Backend State
// ============================================================

struct CudaSPUBackend {
    bool      initialized;
    uint64_t  total_insns;
    uint64_t  total_calls;
    double    total_gpu_ms;
};

// ============================================================
// Backend API
// ============================================================

int cuda_spu_init(CudaSPUBackend* ctx);

int cuda_spu_execute(CudaSPUBackend* ctx,
                     v128_compat* host_gpr,
                     uint8_t* host_ls,
                     uint32_t* pc,
                     uint32_t max_insns,
                     uint32_t* stop_reason,
                     uint32_t* channel_idx,
                     uint32_t* channel_reg);

int cuda_spu_run_streaming(CudaSPUBackend* ctx,
                           v128_compat* host_gpr,
                           uint8_t* host_ls,
                           uint32_t* pc);

void cuda_spu_shutdown(CudaSPUBackend* ctx);

// ============================================================
// Stats
// ============================================================

struct CudaSPUStats {
    std::atomic<uint64_t> total_blocks;
    std::atomic<uint64_t> total_instructions;
    std::atomic<uint64_t> gpu_offloaded;
    std::atomic<uint64_t> host_fallback;
};

extern CudaSPUStats g_cuda_spu_stats;
