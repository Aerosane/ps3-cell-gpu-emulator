// cell_megakernel.cu — Cooperative Cell Broadband Engine Hypervisor
// Launches PPE (1 thread) + 6 SPUs (6 threads) as a single cooperative kernel.
//
// Uses CUDA Cooperative Groups for cross-SM synchronization:
//   Block 0: PPE core (1 thread active)
//   Block 1: SPU cores (6 threads active)
//
// Shared mailbox in global memory for PPE ↔ SPU communication.
//
#include "ppc_defs.h"
#include "spu_defs.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

using namespace ppc;
using namespace spu;

// ═══════════════════════════════════════════════════════════════
// Shared Communication: PPE ↔ SPU Atomic Mailbox
// ═══════════════════════════════════════════════════════════════

struct CellMailbox {
    // PPE → SPU (one per SPU, 4-deep FIFO simplified to 1-deep)
    volatile uint32_t ppe_to_spu[6];
    volatile uint32_t ppe_to_spu_valid[6];

    // SPU → PPE (one per SPU)
    volatile uint32_t spu_to_ppe[6];
    volatile uint32_t spu_to_ppe_valid[6];

    // SPU interrupt mailbox (triggers PPE attention)
    volatile uint32_t spu_intr[6];
    volatile uint32_t spu_intr_valid[6];

    // Global sync: all-halt flag
    volatile uint32_t allHalt;

    // Per-SPU run enable (PPE controls which SPUs are active)
    volatile uint32_t spuEnabled[6];

    // Cycle counter (shared timebase)
    volatile uint64_t timebase;
};

// ═══════════════════════════════════════════════════════════════
// Extern: Single-step functions from ppc_interpreter.cu / spu_interpreter.cu
// We forward-declare them here; the actual kernels are in their own files.
// For the cooperative kernel, we inline simplified versions.
// ═══════════════════════════════════════════════════════════════

// Big-endian memory access (same as ppc_interpreter.cu)
__device__ __forceinline__ uint32_t cell_bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

__device__ __forceinline__ uint32_t cell_mem_read32(const uint8_t* mem, uint64_t addr) {
    uint32_t raw;
    memcpy(&raw, mem + (addr & (PS3_SANDBOX_SIZE - 1)), 4);
    return cell_bswap32(raw);
}

__device__ __forceinline__ void cell_mem_write32(uint8_t* mem, uint64_t addr, uint32_t val) {
    uint32_t swapped = cell_bswap32(val);
    memcpy(mem + (addr & (PS3_SANDBOX_SIZE - 1)), &swapped, 4);
}

// SPU Local Store access
__device__ __forceinline__ uint32_t cell_ls_fetch(const uint8_t* ls, uint32_t pc) {
    pc &= (SPU_LS_SIZE - 1) & ~0x3;
    uint32_t raw;
    memcpy(&raw, ls + pc, 4);
    return cell_bswap32(raw);
}

// ═══════════════════════════════════════════════════════════════
// PPE Thread — runs on Block 0, Thread 0
// ═══════════════════════════════════════════════════════════════

__device__ static void ppeThread(PPEState* state, uint8_t* mem,
                                  CellMailbox* mbox, uint32_t cyclesPerSlice) {
    PPEState& s = *state;

    for (uint32_t c = 0; c < cyclesPerSlice && !s.halted; c++) {
        // Check for SPU interrupt mailboxes
        for (int i = 0; i < 6; i++) {
            if (mbox->spu_intr_valid[i]) {
                // SPU sent interrupt — could trigger PPE interrupt handler
                // For HLE: just acknowledge it
                mbox->spu_intr_valid[i] = 0;
            }
        }

        // Check SPU → PPE mailboxes (polling, as PPE would)
        for (int i = 0; i < 6; i++) {
            if (mbox->spu_to_ppe_valid[i]) {
                // Game-specific: PPE reads SPU completion signals
                // Store in a readable location for HLE handlers
            }
        }

        // Execute one PPE instruction
        // (simplified inline — full version in ppc_interpreter.cu)
        uint32_t inst = cell_mem_read32(mem, s.pc);
        uint32_t opcd = (inst >> 26) & 0x3F;
        s.npc = s.pc + 4;

        // Handle only the most critical ops inline for the cooperative kernel
        // Full dispatch is in ppc_interpreter.cu's execOne()
        switch (opcd) {
        case 14: { // addi
            uint32_t rd = (inst >> 21) & 0x1F, ra = (inst >> 16) & 0x1F;
            int16_t imm = (int16_t)(inst & 0xFFFF);
            s.gpr[rd] = (ra == 0) ? (uint64_t)(int64_t)imm : (uint64_t)((int64_t)s.gpr[ra] + imm);
            break;
        }
        case 15: { // addis
            uint32_t rd = (inst >> 21) & 0x1F, ra = (inst >> 16) & 0x1F;
            int32_t imm = ((int16_t)(inst & 0xFFFF)) << 16;
            s.gpr[rd] = (ra == 0) ? (uint64_t)(int64_t)imm : (uint64_t)((int64_t)s.gpr[ra] + imm);
            break;
        }
        case 18: { // b/bl
            int32_t disp = (int32_t)(inst & 0x03FFFFFC);
            if (disp & 0x02000000) disp |= 0xFC000000;
            s.npc = ((inst >> 1) & 1) ? (uint64_t)disp : s.pc + (uint64_t)disp;
            if (inst & 1) s.lr = s.pc + 4;
            break;
        }
        case 17: { // sc (syscall)
            uint32_t sc_num = (uint32_t)s.gpr[11];
            // SPU management syscalls → control mailbox
            if (sc_num == SYS_SPU_THREAD_GROUP_START) {
                // Enable all SPUs
                for (int i = 0; i < 6; i++) mbox->spuEnabled[i] = 1;
                s.gpr[3] = 0;
            } else if (sc_num == SYS_SPU_THREAD_WRITE_LS) {
                // Write value to SPU local store
                uint32_t spuId = (uint32_t)s.gpr[3];
                uint32_t lsAddr = (uint32_t)s.gpr[4];
                uint32_t value = (uint32_t)s.gpr[5];
                if (spuId < 6) {
                    mbox->ppe_to_spu[spuId] = value;
                    mbox->ppe_to_spu_valid[spuId] = 1;
                }
                s.gpr[3] = 0;
            } else if (sc_num == SYS_PROCESS_EXIT) {
                s.halted = 1;
                mbox->allHalt = 1;
            } else {
                s.gpr[3] = 0; // fake success
            }
            break;
        }
        case 32: { // lwz
            uint32_t rd = (inst >> 21) & 0x1F, ra = (inst >> 16) & 0x1F;
            int16_t imm = (int16_t)(inst & 0xFFFF);
            uint64_t ea = (int64_t)imm + ((ra == 0) ? 0 : s.gpr[ra]);
            s.gpr[rd] = cell_mem_read32(mem, ea);
            break;
        }
        case 36: { // stw
            uint32_t rs = (inst >> 21) & 0x1F, ra = (inst >> 16) & 0x1F;
            int16_t imm = (int16_t)(inst & 0xFFFF);
            uint64_t ea = (int64_t)imm + ((ra == 0) ? 0 : s.gpr[ra]);
            cell_mem_write32(mem, ea, (uint32_t)s.gpr[rs]);
            break;
        }
        case 24: { // ori (nop when rs==ra==0)
            uint32_t rs = (inst >> 21) & 0x1F, ra = (inst >> 16) & 0x1F;
            s.gpr[ra] = s.gpr[rs] | (inst & 0xFFFF);
            break;
        }
        case 31: { // Group 31 — extended ALU
            uint32_t xo = (inst >> 1) & 0x3FF;
            uint32_t rd = (inst >> 21) & 0x1F, ra = (inst >> 16) & 0x1F, rb = (inst >> 11) & 0x1F;
            switch (xo) {
            case 266: // add
                s.gpr[rd] = (uint32_t)((uint32_t)s.gpr[ra] + (uint32_t)s.gpr[rb]);
                break;
            case 40: // subf
                s.gpr[rd] = (uint32_t)((uint32_t)s.gpr[rb] - (uint32_t)s.gpr[ra]);
                break;
            case 235: // mullw
                s.gpr[rd] = (uint32_t)((int32_t)(uint32_t)s.gpr[ra] * (int32_t)(uint32_t)s.gpr[rb]);
                break;
            case 444: // or (also mr)
                s.gpr[ra] = s.gpr[rd] | s.gpr[rb];
                break;
            case 339: // mfspr
            {
                uint32_t spr = ((inst >> 16) & 0x1F) << 5 | ((inst >> 11) & 0x1F);
                if (spr == 8) s.gpr[rd] = s.lr;
                else if (spr == 9) s.gpr[rd] = s.ctr;
                break;
            }
            case 467: // mtspr
            {
                uint32_t spr = ((inst >> 16) & 0x1F) << 5 | ((inst >> 11) & 0x1F);
                if (spr == 8) s.lr = s.gpr[rd];
                else if (spr == 9) s.ctr = s.gpr[rd];
                break;
            }
            default: break;
            }
            break;
        }
        default:
            break;
        }

        s.pc = s.npc;
        s.cycles++;
        s.tbl++;
    }
}

// ═══════════════════════════════════════════════════════════════
// SPU Thread — runs on Block 1, Threads 0-5
// ═══════════════════════════════════════════════════════════════

__device__ static void spuThread(SPUState* state, uint8_t* ls,
                                  uint8_t* mainMem, CellMailbox* mbox,
                                  int spuId, uint32_t cyclesPerSlice) {
    SPUState& s = *state;

    // Wait until PPE enables us
    if (!mbox->spuEnabled[spuId]) return;

    // Check inbound mailbox from PPE
    if (mbox->ppe_to_spu_valid[spuId]) {
        s.inMbox = mbox->ppe_to_spu[spuId];
        s.inMboxValid = 1;
        mbox->ppe_to_spu_valid[spuId] = 0;
    }

    for (uint32_t c = 0; c < cyclesPerSlice && !s.halted; c++) {
        if (mbox->allHalt) { s.halted = 1; break; }

        // Fetch + decode one SPU instruction (simplified inline)
        uint32_t inst = cell_ls_fetch(ls, s.pc);
        s.npc = (s.pc + 4) & (SPU_LS_SIZE - 1);

        // Check for STOP instruction
        uint32_t o11 = (inst >> 21) & 0x7FF;
        if (o11 == 0x000) { // STOP
            s.halted = 1;
            break;
        }

        // Handle outbound mailbox write → signal PPE via shared memory
        if (o11 == 0x10D) { // wrch
            uint32_t ch = inst & 0x7F;
            uint32_t rT = (inst >> 14) & 0x7F;
            if (ch == SPU_WrOutMbox) {
                mbox->spu_to_ppe[spuId] = s.gpr[rT].u32[0];
                mbox->spu_to_ppe_valid[spuId] = 1;
            } else if (ch == SPU_WrOutIntrMbox) {
                mbox->spu_intr[spuId] = s.gpr[rT].u32[0];
                mbox->spu_intr_valid[spuId] = 1;
            }
        }

        // For the cooperative kernel, we only handle the most critical
        // SPU ops inline. The full dispatch is in spu_interpreter.cu.
        // This is a proof-of-concept for the cooperative launch pattern.

        s.pc = s.npc;
        s.cycles++;
        if (s.decrementer > 0) s.decrementer--;
    }
}

// ═══════════════════════════════════════════════════════════════
// Cooperative Megakernel — THE kernel
//
// Block 0: PPE (thread 0 active)
// Block 1: 6 SPUs (threads 0-5 active)
//
// They synchronize via grid.sync() at time-slice boundaries.
// ═══════════════════════════════════════════════════════════════

__global__ void cellMegakernel(PPEState* ppeState, SPUState* spuStates,
                                uint8_t* mainMem, uint8_t** spuLocalStores,
                                CellMailbox* mbox,
                                uint32_t totalSlices, uint32_t cyclesPerSlice) {
    cg::grid_group grid = cg::this_grid();

    for (uint32_t slice = 0; slice < totalSlices; slice++) {
        // Check global halt
        if (mbox->allHalt) break;

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            // PPE executes its time slice
            ppeThread(ppeState, mainMem, mbox, cyclesPerSlice);

            // Update shared timebase
            mbox->timebase += cyclesPerSlice;
        }

        if (blockIdx.x == 1 && threadIdx.x < 6) {
            int spuId = threadIdx.x;
            // SPU executes its time slice
            spuThread(&spuStates[spuId], spuLocalStores[spuId],
                      mainMem, mbox, spuId, cyclesPerSlice);
        }

        // Barrier: all blocks sync before next slice
        // This ensures PPE writes are visible to SPUs and vice versa
        grid.sync();
    }
}

// ═══════════════════════════════════════════════════════════════
// Host API — Cell Broadband Engine Launcher
// ═══════════════════════════════════════════════════════════════

extern "C" {

struct CellContext {
    PPEState*      d_ppeState;
    SPUState*      d_spuStates;
    uint8_t*       d_mainMem;
    uint8_t*       d_spuLS[6];
    uint8_t**      d_spuLSPtrs;
    CellMailbox*   d_mailbox;
    cudaStream_t   stream;
    bool           ready;
};

static CellContext g_cell = {};

int cell_init() {
    if (g_cell.ready) return 1;

    // Check cooperative launch support
    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    if (!supportsCoopLaunch) {
        fprintf(stderr, "[Cell] ERROR: Device does not support cooperative launch\n");
        return 0;
    }

    cudaStreamCreate(&g_cell.stream);

    // PPE state
    cudaMalloc(&g_cell.d_ppeState, sizeof(PPEState));
    cudaMemset(g_cell.d_ppeState, 0, sizeof(PPEState));

    // 6 SPU states
    cudaMalloc(&g_cell.d_spuStates, 6 * sizeof(SPUState));
    cudaMemset(g_cell.d_spuStates, 0, 6 * sizeof(SPUState));

    // Main memory (512MB PS3 sandbox)
    cudaMalloc(&g_cell.d_mainMem, PS3_SANDBOX_SIZE);
    cudaMemset(g_cell.d_mainMem, 0, PS3_SANDBOX_SIZE);

    // 6 × 256KB SPU Local Stores
    uint8_t* h_ptrs[6];
    for (int i = 0; i < 6; i++) {
        cudaMalloc(&g_cell.d_spuLS[i], SPU_LS_SIZE);
        cudaMemset(g_cell.d_spuLS[i], 0, SPU_LS_SIZE);
        h_ptrs[i] = g_cell.d_spuLS[i];
    }
    cudaMalloc(&g_cell.d_spuLSPtrs, 6 * sizeof(uint8_t*));
    cudaMemcpy(g_cell.d_spuLSPtrs, h_ptrs, 6 * sizeof(uint8_t*), cudaMemcpyHostToDevice);

    // Mailbox
    cudaMalloc(&g_cell.d_mailbox, sizeof(CellMailbox));
    cudaMemset(g_cell.d_mailbox, 0, sizeof(CellMailbox));

    g_cell.ready = true;

    size_t totalVRAM = PS3_SANDBOX_SIZE + 6 * SPU_LS_SIZE + sizeof(CellMailbox)
                     + sizeof(PPEState) + 6 * sizeof(SPUState);
    fprintf(stderr, "[Cell] Hypervisor initialized\n");
    fprintf(stderr, "[Cell]   Main RAM: 256MB + VRAM: 256MB = 512MB sandbox\n");
    fprintf(stderr, "[Cell]   6 × SPU LS: 1.5MB\n");
    fprintf(stderr, "[Cell]   Total VRAM: %.1f MB\n", totalVRAM / (1024.0 * 1024.0));
    return 1;
}

// Load PPE program into main memory
int cell_load_ppe(uint64_t loadAddr, const void* data, size_t size,
                   uint64_t entryPC, uint64_t stackPtr, uint64_t toc) {
    if (!g_cell.ready) return 0;
    if (loadAddr + size > PS3_SANDBOX_SIZE) return 0;

    cudaMemcpy(g_cell.d_mainMem + loadAddr, data, size, cudaMemcpyHostToDevice);

    PPEState init = {};
    init.pc = entryPC;
    init.gpr[1] = stackPtr;
    init.gpr[2] = toc;
    init.msr = 0x8000000000000000ULL;
    cudaMemcpy(g_cell.d_ppeState, &init, sizeof(PPEState), cudaMemcpyHostToDevice);

    fprintf(stderr, "[Cell:PPE] Loaded %zu bytes @ 0x%llx, entry=0x%llx\n",
            size, (unsigned long long)loadAddr, (unsigned long long)entryPC);
    return 1;
}

// Load SPU program
int cell_load_spu(int spuId, const void* data, size_t size, uint32_t entryPC) {
    if (!g_cell.ready || spuId < 0 || spuId >= 6) return 0;
    if (size > SPU_LS_SIZE) size = SPU_LS_SIZE;

    cudaMemcpy(g_cell.d_spuLS[spuId], data, size, cudaMemcpyHostToDevice);

    SPUState init = {};
    init.pc = entryPC & (SPU_LS_SIZE - 1);
    init.spuId = spuId;
    init.gpr[1].u32[0] = SPU_LS_SIZE - 16; // SP at top of LS
    cudaMemcpy(&g_cell.d_spuStates[spuId], &init, sizeof(SPUState), cudaMemcpyHostToDevice);

    // Enable this SPU in the mailbox
    uint32_t one = 1;
    cudaMemcpy((void*)((uint8_t*)g_cell.d_mailbox + offsetof(CellMailbox, spuEnabled) + spuId * sizeof(uint32_t)),
               &one, sizeof(uint32_t), cudaMemcpyHostToDevice);

    fprintf(stderr, "[Cell:SPU%d] Loaded %zu bytes, entry=0x%x\n", spuId, size, entryPC);
    return 1;
}

// Run the cooperative Cell megakernel
float cell_run(uint32_t totalSlices, uint32_t cyclesPerSlice) {
    if (!g_cell.ready) return -1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 2 blocks: block 0 = PPE (32 threads, only t0 active),
    //           block 1 = SPUs (32 threads, t0-5 active)
    // 32 threads per block = 1 warp (minimum scheduling unit)
    dim3 grid(2);
    dim3 block(32);

    void* args[] = {
        &g_cell.d_ppeState, &g_cell.d_spuStates,
        &g_cell.d_mainMem, &g_cell.d_spuLSPtrs,
        &g_cell.d_mailbox,
        &totalSlices, &cyclesPerSlice
    };

    cudaEventRecord(start, g_cell.stream);

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)cellMegakernel, grid, block, args, 0, g_cell.stream);

    if (err != cudaSuccess) {
        fprintf(stderr, "[Cell] Cooperative launch failed: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }

    cudaEventRecord(stop, g_cell.stream);
    cudaStreamSynchronize(g_cell.stream);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Readback state for debugging
int cell_read_ppe(PPEState* out) {
    if (!g_cell.ready || !out) return 0;
    cudaMemcpy(out, g_cell.d_ppeState, sizeof(PPEState), cudaMemcpyDeviceToHost);
    return 1;
}

int cell_read_spu(int spuId, SPUState* out) {
    if (!g_cell.ready || spuId < 0 || spuId >= 6 || !out) return 0;
    cudaMemcpy(out, &g_cell.d_spuStates[spuId], sizeof(SPUState), cudaMemcpyDeviceToHost);
    return 1;
}

void cell_shutdown() {
    if (!g_cell.ready) return;
    cudaFree(g_cell.d_ppeState);
    cudaFree(g_cell.d_spuStates);
    cudaFree(g_cell.d_mainMem);
    for (int i = 0; i < 6; i++) cudaFree(g_cell.d_spuLS[i]);
    cudaFree(g_cell.d_spuLSPtrs);
    cudaFree(g_cell.d_mailbox);
    cudaStreamDestroy(g_cell.stream);
    g_cell.ready = false;
    fprintf(stderr, "[Cell] Hypervisor shutdown\n");
}

} // extern "C"
