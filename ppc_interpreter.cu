// ppc_interpreter.cu — CUDA PowerPC Interpreter Kernel
// Modeled after VK_RT/layer/rt_ir_exec.cu: per-thread state + opcode dispatch loop.
//
// One CUDA thread = one PPE hardware thread.
// The kernel runs as a persistent megakernel — launched once, interprets until halt.
//
#include "ppc_defs.h"
#include <cuda_runtime.h>
#include <cstdio>

using namespace ppc;

// ═══════════════════════════════════════════════════════════════
// Big-Endian Memory Access via __byte_perm
// PS3 is big-endian; GPU is little-endian. Every memory op swaps.
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

__device__ __forceinline__ uint16_t bswap16(uint16_t x) {
    return (uint16_t)__byte_perm((uint32_t)x, 0, 0x0001);
}

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    return ((uint64_t)bswap32(lo) << 32) | (uint64_t)bswap32(hi);
}

// Memory read/write with endian swap
// `mem` is the base of the 512MB PS3 sandbox in GPU VRAM
__device__ __forceinline__ uint32_t mem_read32(const uint8_t* mem, uint64_t addr) {
    uint32_t raw;
    memcpy(&raw, mem + (addr & (PS3_SANDBOX_SIZE - 1)), 4);
    return bswap32(raw);
}

__device__ __forceinline__ uint16_t mem_read16(const uint8_t* mem, uint64_t addr) {
    uint16_t raw;
    memcpy(&raw, mem + (addr & (PS3_SANDBOX_SIZE - 1)), 2);
    return bswap16(raw);
}

__device__ __forceinline__ uint8_t mem_read8(const uint8_t* mem, uint64_t addr) {
    return mem[addr & (PS3_SANDBOX_SIZE - 1)];
}

__device__ __forceinline__ void mem_write32(uint8_t* mem, uint64_t addr, uint32_t val) {
    uint32_t swapped = bswap32(val);
    memcpy(mem + (addr & (PS3_SANDBOX_SIZE - 1)), &swapped, 4);
}

__device__ __forceinline__ void mem_write16(uint8_t* mem, uint64_t addr, uint16_t val) {
    uint16_t swapped = bswap16(val);
    memcpy(mem + (addr & (PS3_SANDBOX_SIZE - 1)), &swapped, 2);
}

__device__ __forceinline__ void mem_write8(uint8_t* mem, uint64_t addr, uint8_t val) {
    mem[addr & (PS3_SANDBOX_SIZE - 1)] = val;
}

// Floating-point memory ops (IEEE 754 is endian-dependent)
__device__ __forceinline__ float mem_readf32(const uint8_t* mem, uint64_t addr) {
    uint32_t bits = mem_read32(mem, addr);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

__device__ __forceinline__ double mem_readf64(const uint8_t* mem, uint64_t addr) {
    uint64_t bits = bswap64(*(const uint64_t*)(mem + (addr & (PS3_SANDBOX_SIZE - 1))));
    double d;
    memcpy(&d, &bits, 8);
    return d;
}

__device__ __forceinline__ void mem_writef32(uint8_t* mem, uint64_t addr, float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    mem_write32(mem, addr, bits);
}

__device__ __forceinline__ void mem_writef64(uint8_t* mem, uint64_t addr, double d) {
    uint64_t bits;
    memcpy(&bits, &d, 8);
    uint64_t swapped = bswap64(bits);
    memcpy(mem + (addr & (PS3_SANDBOX_SIZE - 1)), &swapped, 8);
}

// ═══════════════════════════════════════════════════════════════
// HLE Syscall Handler
// ═══════════════════════════════════════════════════════════════

__device__ static void handleSyscall(PPEState& s, uint8_t* mem, uint32_t* hle_log,
                                      volatile uint32_t* hle_signal) {
    uint32_t sc_num = (uint32_t)s.gpr[11]; // r11 = syscall number on CellOS

    switch (sc_num) {
    case SYS_PROCESS_EXIT:
        s.halted = 1;
        break;

    case SYS_TICKS_GET:
        // Return a fake timebase tick (monotonic, 79.8 MHz on PS3)
        s.gpr[3] = s.cycles * 10;  // rough approximation
        break;

    case SYS_MEMORY_ALLOCATE: {
        // r3 = size, r4 = alignment
        // Bump allocator in upper RAM region
        static __device__ uint64_t heapPtr = 0x10000000ULL; // start at 256MB
        uint64_t size = s.gpr[3];
        uint64_t align = s.gpr[4] ? s.gpr[4] : 4096;
        heapPtr = (heapPtr + align - 1) & ~(align - 1);
        s.gpr[3] = 0;           // CELL_OK
        s.gpr[4] = heapPtr;     // allocated address
        heapPtr += size;
        break;
    }

    case SYS_MEMORY_FREE:
        s.gpr[3] = 0; // CELL_OK (no-op for bump allocator)
        break;

    case SYS_MEMORY_GET_PAGE_SIZE:
        s.gpr[3] = 0;      // CELL_OK
        s.gpr[4] = 4096;   // 4KB pages
        break;

    case SYS_SPU_THREAD_GROUP_CREATE:
    case SYS_SPU_THREAD_INITIALIZE:
    case SYS_SPU_THREAD_GROUP_START:
    case SYS_SPU_THREAD_GROUP_JOIN:
        // SPU management — signal host for cooperative dispatch
        if (hle_signal) atomicAdd((uint32_t*)hle_signal, 1);
        s.gpr[3] = 0; // CELL_OK
        break;

    default:
        // Log unknown syscall for debugging
        if (hle_log) {
            uint32_t idx = atomicAdd(hle_log, 1);
            if (idx < 255) {
                hle_log[1 + idx] = sc_num;
            }
        }
        s.gpr[3] = 0; // Fake success to keep game running
        break;
    }
}

// ═══════════════════════════════════════════════════════════════
// Branch Evaluation (BO field decoding)
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ bool evalBranch(PPEState& s, uint32_t bo, uint32_t bi) {
    // BO field encoding:
    // bit 0 (4): branch if condition true
    // bit 1 (3): don't test condition
    // bit 2 (2): decrement CTR
    // bit 3 (1): don't decrement CTR
    // bit 4 (0): branch hint (not used for correctness)

    bool ctr_ok = true;
    if (!(bo & 0x04)) {  // decrement CTR
        s.ctr--;
        ctr_ok = (bo & 0x02) ? (s.ctr == 0) : (s.ctr != 0);
    }
    bool cond_ok = true;
    if (!(bo & 0x10)) {  // test condition
        bool bit = getCRBit(s.cr, bi);
        cond_ok = (bo & 0x08) ? bit : !bit;
    }
    return ctr_ok && cond_ok;
}

// ═══════════════════════════════════════════════════════════════
// Rotate Mask Helper (for rlwinm / rlwimi)
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ uint32_t rotateMask32(uint32_t mb, uint32_t me) {
    uint32_t mask = 0;
    if (mb <= me) {
        for (uint32_t i = mb; i <= me; i++) mask |= (1U << (31 - i));
    } else {
        for (uint32_t i = 0; i <= me; i++)   mask |= (1U << (31 - i));
        for (uint32_t i = mb; i <= 31; i++)   mask |= (1U << (31 - i));
    }
    return mask;
}

__device__ __forceinline__ uint32_t rotl32(uint32_t v, uint32_t n) {
    n &= 31;
    return (v << n) | (v >> (32 - n));
}

// ═══════════════════════════════════════════════════════════════
// Single-Step Execute — decode and run one PPC instruction
// Returns 0 = ok, 1 = halted, 2 = unimplemented
// ═══════════════════════════════════════════════════════════════

__device__ static int execOne(PPEState& s, uint8_t* mem,
                               uint32_t* hle_log, volatile uint32_t* hle_signal) {
    if (s.halted) return 1;

    // Fetch (big-endian 32-bit instruction)
    uint32_t inst = mem_read32(mem, s.pc);
    uint32_t opcd = OPCD(inst);
    s.npc = s.pc + 4;

    switch (opcd) {

    // ─── Immediate ALU ────────────────────────────────────────

    case OP_ADDI: {  // also "li rD, SIMM" when rA=0
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        s.gpr[rd] = (ra == 0) ? (uint64_t)imm : (uint64_t)((int64_t)s.gpr[ra] + imm);
        break;
    }

    case OP_ADDIS: {  // also "lis rD, SIMM" when rA=0
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst) << 16;
        s.gpr[rd] = (ra == 0) ? (uint64_t)imm : (uint64_t)((int64_t)s.gpr[ra] + imm);
        break;
    }

    case OP_ADDIC: {
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        uint64_t a = s.gpr[ra];
        uint64_t result = a + (uint64_t)imm;
        s.gpr[rd] = result;
        setCA(s.xer, (uint32_t)result < (uint32_t)a);
        break;
    }

    case OP_ADDIC_D: {
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        uint64_t a = s.gpr[ra];
        uint64_t result = a + (uint64_t)imm;
        s.gpr[rd] = result;
        setCA(s.xer, (uint32_t)result < (uint32_t)a);
        setCRField(s.cr, 0, (int32_t)(uint32_t)result, s.xer);
        break;
    }

    case OP_SUBFIC: {
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        uint64_t result = (uint64_t)imm - s.gpr[ra];
        s.gpr[rd] = result;
        setCA(s.xer, (uint32_t)s.gpr[ra] <= (uint32_t)(uint64_t)imm);
        break;
    }

    case OP_MULLI: {
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        s.gpr[rd] = (uint64_t)((int64_t)s.gpr[ra] * imm);
        break;
    }

    // ─── Compare ──────────────────────────────────────────────

    case OP_CMPI: {
        uint32_t bf = RD(inst) >> 2;
        uint32_t ra = RA(inst);
        int64_t imm = SIMM16(inst);
        int32_t a = (int32_t)(uint32_t)s.gpr[ra];
        setCRField(s.cr, bf, (int64_t)a - imm, s.xer);
        break;
    }

    case OP_CMPLI: {
        uint32_t bf = RD(inst) >> 2;
        uint32_t ra = RA(inst);
        uint64_t imm = UIMM16(inst);
        uint32_t a = (uint32_t)s.gpr[ra];
        int64_t diff = (a > (uint32_t)imm) ? 1 : (a < (uint32_t)imm) ? -1 : 0;
        setCRField(s.cr, bf, diff, s.xer);
        break;
    }

    // ─── Logical Immediate ────────────────────────────────────

    case OP_ORI: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] | UIMM16(inst);
        break;
    }

    case OP_ORIS: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] | (UIMM16(inst) << 16);
        break;
    }

    case OP_XORI: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] ^ UIMM16(inst);
        break;
    }

    case OP_XORIS: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] ^ (UIMM16(inst) << 16);
        break;
    }

    case OP_ANDI: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] & UIMM16(inst);
        setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    case OP_ANDIS: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] & (UIMM16(inst) << 16);
        setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    // ─── Rotate / Mask ───────────────────────────────────────

    case OP_RLWINM: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint32_t sh = SH(inst), mb = MB(inst), me = ME(inst);
        uint32_t rotated = rotl32((uint32_t)s.gpr[rs], sh);
        uint32_t mask = rotateMask32(mb, me);
        s.gpr[ra] = rotated & mask;
        if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    case OP_RLWIMI: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint32_t sh = SH(inst), mb = MB(inst), me = ME(inst);
        uint32_t rotated = rotl32((uint32_t)s.gpr[rs], sh);
        uint32_t mask = rotateMask32(mb, me);
        s.gpr[ra] = (rotated & mask) | ((uint32_t)s.gpr[ra] & ~mask);
        if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    case OP_RLWNM: {
        uint32_t rs = RS(inst), ra = RA(inst), rb = RB(inst);
        uint32_t mb = MB(inst), me = ME(inst);
        uint32_t rotated = rotl32((uint32_t)s.gpr[rs], (uint32_t)s.gpr[rb] & 0x1F);
        uint32_t mask = rotateMask32(mb, me);
        s.gpr[ra] = rotated & mask;
        if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    // ─── Branches ─────────────────────────────────────────────

    case OP_B: {
        int64_t disp = LI26(inst);
        s.npc = AA(inst) ? (uint64_t)disp : s.pc + (uint64_t)disp;
        if (LK(inst)) s.lr = s.pc + 4;
        break;
    }

    case OP_BC: {
        uint32_t bo = BO(inst), bi = BI(inst);
        if (evalBranch(s, bo, bi)) {
            int64_t disp = BD16(inst);
            s.npc = AA(inst) ? (uint64_t)disp : s.pc + (uint64_t)disp;
        }
        if (LK(inst)) s.lr = s.pc + 4;
        break;
    }

    case OP_SC: {
        handleSyscall(s, mem, hle_log, hle_signal);
        break;
    }

    // ─── Group 19 (Branch to LR/CTR, CR ops) ─────────────────

    case OP_GRP19: {
        uint32_t xo = XO_10(inst);
        switch (xo) {
        case XO_BCLR: {
            uint32_t bo = BO(inst), bi = BI(inst);
            if (evalBranch(s, bo, bi)) {
                s.npc = s.lr & ~3ULL;
            }
            if (LK(inst)) s.lr = s.pc + 4;
            break;
        }
        case XO_BCCTR: {
            uint32_t bo = BO(inst), bi = BI(inst);
            if (evalBranch(s, bo, bi)) {
                s.npc = s.ctr & ~3ULL;
            }
            if (LK(inst)) s.lr = s.pc + 4;
            break;
        }
        case XO_CROR: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            bool a = getCRBit(s.cr, ba), b = getCRBit(s.cr, bb);
            if (a | b) s.cr |=  (1U << (31 - bt));
            else       s.cr &= ~(1U << (31 - bt));
            break;
        }
        case XO_CRXOR: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            bool a = getCRBit(s.cr, ba), b = getCRBit(s.cr, bb);
            if (a ^ b) s.cr |=  (1U << (31 - bt));
            else       s.cr &= ~(1U << (31 - bt));
            break;
        }
        case XO_CRAND: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            bool a = getCRBit(s.cr, ba), b = getCRBit(s.cr, bb);
            if (a & b) s.cr |=  (1U << (31 - bt));
            else       s.cr &= ~(1U << (31 - bt));
            break;
        }
        case XO_ISYNC:
            __threadfence(); // GPU memory fence as isync approximation
            break;
        default:
            return 2; // unimplemented group 19
        }
        break;
    }

    // ─── Load/Store (D-form) ──────────────────────────────────

    case OP_LWZ: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.gpr[rd] = mem_read32(mem, ea);
        break;
    }

    case OP_LWZU: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        s.gpr[rd] = mem_read32(mem, ea);
        s.gpr[ra] = ea;
        break;
    }

    case OP_LBZ: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.gpr[rd] = mem_read8(mem, ea);
        break;
    }

    case OP_LBZU: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        s.gpr[rd] = mem_read8(mem, ea);
        s.gpr[ra] = ea;
        break;
    }

    case OP_LHZ: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.gpr[rd] = mem_read16(mem, ea);
        break;
    }

    case OP_LHZU: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        s.gpr[rd] = mem_read16(mem, ea);
        s.gpr[ra] = ea;
        break;
    }

    case OP_LHA: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.gpr[rd] = (uint64_t)(int64_t)(int16_t)mem_read16(mem, ea);
        break;
    }

    case OP_STW: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_write32(mem, ea, (uint32_t)s.gpr[rs]);
        break;
    }

    case OP_STWU: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        mem_write32(mem, ea, (uint32_t)s.gpr[rs]);
        s.gpr[ra] = ea;
        break;
    }

    case OP_STB: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_write8(mem, ea, (uint8_t)s.gpr[rs]);
        break;
    }

    case OP_STBU: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        mem_write8(mem, ea, (uint8_t)s.gpr[rs]);
        s.gpr[ra] = ea;
        break;
    }

    case OP_STH: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_write16(mem, ea, (uint16_t)s.gpr[rs]);
        break;
    }

    case OP_STHU: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        mem_write16(mem, ea, (uint16_t)s.gpr[rs]);
        s.gpr[ra] = ea;
        break;
    }

    // ─── FP Load/Store ────────────────────────────────────────

    case OP_LFS: {
        uint32_t frd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.fpr[frd] = (double)mem_readf32(mem, ea);
        break;
    }

    case OP_LFD: {
        uint32_t frd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.fpr[frd] = mem_readf64(mem, ea);
        break;
    }

    case OP_STFS: {
        uint32_t frs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_writef32(mem, ea, (float)s.fpr[frs]);
        break;
    }

    case OP_STFD: {
        uint32_t frs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_writef64(mem, ea, s.fpr[frs]);
        break;
    }

    // ─── Group 31 (ALU extended, SPR, indexed load/store) ─────

    case OP_GRP31: {
        uint32_t xo = XO_10(inst);
        uint32_t rd = RD(inst), ra = RA(inst), rb = RB(inst);

        switch (xo) {

        // Arithmetic
        case XO_ADD: {
            s.gpr[rd] = (uint32_t)((uint32_t)s.gpr[ra] + (uint32_t)s.gpr[rb]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_SUBF: {
            s.gpr[rd] = (uint32_t)((uint32_t)s.gpr[rb] - (uint32_t)s.gpr[ra]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_SUBFC: {
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            uint64_t result = (uint64_t)b + (uint64_t)(~a) + 1ULL;
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_ADDC: {
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            uint64_t result = (uint64_t)a + (uint64_t)b;
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_ADDE: {
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            uint64_t result = (uint64_t)a + (uint64_t)b + (getCA(s.xer) ? 1ULL : 0ULL);
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_ADDZE: {
            uint32_t a = (uint32_t)s.gpr[ra];
            uint64_t result = (uint64_t)a + (getCA(s.xer) ? 1ULL : 0ULL);
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_NEG: {
            s.gpr[rd] = (uint32_t)(-(int32_t)(uint32_t)s.gpr[ra]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_MULLW: {
            int32_t a = (int32_t)(uint32_t)s.gpr[ra];
            int32_t b = (int32_t)(uint32_t)s.gpr[rb];
            s.gpr[rd] = (uint32_t)(uint64_t)(int64_t)((int64_t)a * (int64_t)b);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_MULHW: {
            int64_t result = (int64_t)(int32_t)(uint32_t)s.gpr[ra] *
                             (int64_t)(int32_t)(uint32_t)s.gpr[rb];
            s.gpr[rd] = (uint32_t)(result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_MULHWU: {
            uint64_t result = (uint64_t)(uint32_t)s.gpr[ra] *
                              (uint64_t)(uint32_t)s.gpr[rb];
            s.gpr[rd] = (uint32_t)(result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_DIVW: {
            int32_t a = (int32_t)(uint32_t)s.gpr[ra];
            int32_t b = (int32_t)(uint32_t)s.gpr[rb];
            s.gpr[rd] = (b != 0 && !(a == (int32_t)0x80000000 && b == -1))
                         ? (uint32_t)(a / b) : 0;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_DIVWU: {
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            s.gpr[rd] = (b != 0) ? (a / b) : 0;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }

        // Logical
        case XO_AND: {
            s.gpr[ra] = s.gpr[rd] & s.gpr[rb];  // note: rS=rD field in logical ops
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_OR: {
            s.gpr[ra] = s.gpr[rd] | s.gpr[rb];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_XOR: {
            s.gpr[ra] = s.gpr[rd] ^ s.gpr[rb];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_ANDC: {
            s.gpr[ra] = s.gpr[rd] & ~s.gpr[rb];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_ORC: {
            s.gpr[ra] = s.gpr[rd] | ~s.gpr[rb];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_NOR: {
            s.gpr[ra] = ~(s.gpr[rd] | s.gpr[rb]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_NAND: {
            s.gpr[ra] = ~(s.gpr[rd] & s.gpr[rb]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_EXTSB: {
            s.gpr[ra] = (uint64_t)(int64_t)(int8_t)(uint8_t)s.gpr[rd];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_EXTSH: {
            s.gpr[ra] = (uint64_t)(int64_t)(int16_t)(uint16_t)s.gpr[rd];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_CNTLZW: {
            uint32_t val = (uint32_t)s.gpr[rd]; // rS
            s.gpr[ra] = val ? __clz(val) : 32;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }

        // Shifts
        case XO_SLW: {
            uint32_t sh = (uint32_t)s.gpr[rb] & 0x3F;
            s.gpr[ra] = (sh < 32) ? ((uint32_t)s.gpr[rd] << sh) : 0;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_SRW: {
            uint32_t sh = (uint32_t)s.gpr[rb] & 0x3F;
            s.gpr[ra] = (sh < 32) ? ((uint32_t)s.gpr[rd] >> sh) : 0;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_SRAW: {
            uint32_t sh = (uint32_t)s.gpr[rb] & 0x3F;
            int32_t val = (int32_t)(uint32_t)s.gpr[rd];
            if (sh == 0) {
                s.gpr[ra] = (uint32_t)val;
                setCA(s.xer, false);
            } else if (sh < 32) {
                bool carry = (val < 0) && ((val & ((1 << sh) - 1)) != 0);
                s.gpr[ra] = (uint32_t)(val >> sh);
                setCA(s.xer, carry);
            } else {
                s.gpr[ra] = (val < 0) ? 0xFFFFFFFF : 0;
                setCA(s.xer, val < 0);
            }
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_SRAWI: {
            uint32_t sh = SH(inst);
            int32_t val = (int32_t)(uint32_t)s.gpr[rd];
            if (sh == 0) {
                s.gpr[ra] = (uint32_t)val;
                setCA(s.xer, false);
            } else {
                bool carry = (val < 0) && ((val & ((1 << sh) - 1)) != 0);
                s.gpr[ra] = (uint32_t)(val >> sh);
                setCA(s.xer, carry);
            }
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }

        // Compare
        case XO_CMP: {
            uint32_t bf = rd >> 2;
            int32_t a = (int32_t)(uint32_t)s.gpr[ra];
            int32_t b = (int32_t)(uint32_t)s.gpr[rb];
            setCRField(s.cr, bf, (int64_t)a - (int64_t)b, s.xer);
            break;
        }
        case XO_CMPL: {
            uint32_t bf = rd >> 2;
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            int64_t diff = (a > b) ? 1 : (a < b) ? -1 : 0;
            setCRField(s.cr, bf, diff, s.xer);
            break;
        }

        // Indexed load/store
        case XO_LWZX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = mem_read32(mem, ea);
            break;
        }
        case XO_LBZX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = mem_read8(mem, ea);
            break;
        }
        case XO_LHZX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = mem_read16(mem, ea);
            break;
        }
        case XO_STWX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_write32(mem, ea, (uint32_t)s.gpr[rd]);
            break;
        }
        case XO_STBX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_write8(mem, ea, (uint8_t)s.gpr[rd]);
            break;
        }
        case XO_STHX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_write16(mem, ea, (uint16_t)s.gpr[rd]);
            break;
        }

        // SPR access
        case XO_MFSPR: {
            uint32_t spr = SPR(inst);
            switch (spr) {
            case SPR_LR:   s.gpr[rd] = s.lr; break;
            case SPR_CTR:  s.gpr[rd] = s.ctr; break;
            case SPR_XER:  s.gpr[rd] = s.xer; break;
            case SPR_TBL:  s.gpr[rd] = s.tbl; break;
            case SPR_TBU:  s.gpr[rd] = s.tbu; break;
            case SPR_DEC:  s.gpr[rd] = s.dec; break;
            case SPR_SRR0: s.gpr[rd] = s.srr0; break;
            case SPR_SRR1: s.gpr[rd] = s.srr1; break;
            case SPR_PVR:  s.gpr[rd] = 0x00700000; break;  // Cell PPE
            default:
                if (spr >= SPR_SPRG0 && spr <= SPR_SPRG3)
                    s.gpr[rd] = s.sprg[spr - SPR_SPRG0];
                else
                    s.gpr[rd] = 0;
                break;
            }
            break;
        }

        case XO_MTSPR: {
            uint32_t spr = SPR(inst);
            switch (spr) {
            case SPR_LR:   s.lr  = s.gpr[rd]; break;
            case SPR_CTR:  s.ctr = s.gpr[rd]; break;
            case SPR_XER:  s.xer = s.gpr[rd]; break;
            case SPR_DEC:  s.dec = s.gpr[rd]; break;
            case SPR_SRR0: s.srr0 = s.gpr[rd]; break;
            case SPR_SRR1: s.srr1 = s.gpr[rd]; break;
            default:
                if (spr >= SPR_SPRG0 && spr <= SPR_SPRG3)
                    s.sprg[spr - SPR_SPRG0] = s.gpr[rd];
                break;
            }
            break;
        }

        case XO_MFCR:
            s.gpr[rd] = s.cr;
            break;

        case XO_MTCRF: {
            uint32_t crm = CRM(inst);
            uint32_t val = (uint32_t)s.gpr[rd]; // rS
            for (int i = 0; i < 8; i++) {
                if (crm & (1 << (7 - i))) {
                    int shift = (7 - i) * 4;
                    s.cr = (s.cr & ~(0xF << shift)) | (val & (0xF << shift));
                }
            }
            break;
        }

        // Synchronization (no-ops in single-threaded interpreter)
        case XO_SYNC:
        case XO_EIEIO:
            __threadfence();
            break;

        default:
            return 2; // unimplemented group 31
        }
        break;
    }

    // ─── Load/Store Multiple (used in function prologues) ─────

    case OP_LMW: {
        uint32_t rd_start = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        for (uint32_t r = rd_start; r < 32; r++, ea += 4) {
            s.gpr[r] = mem_read32(mem, ea);
        }
        break;
    }

    case OP_STMW: {
        uint32_t rs_start = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        for (uint32_t r = rs_start; r < 32; r++, ea += 4) {
            mem_write32(mem, ea, (uint32_t)s.gpr[r]);
        }
        break;
    }

    default:
        return 2; // unimplemented primary opcode
    }

    s.pc = s.npc;
    s.cycles++;
    s.tbl++;
    return 0;
}

// ═══════════════════════════════════════════════════════════════
// Persistent Megakernel — one thread = one PPE core
// ═══════════════════════════════════════════════════════════════

__global__ void ppeMegakernel(PPEState* states, uint8_t* mem,
                               uint32_t maxCycles,
                               uint32_t* hle_log,
                               volatile uint32_t* hle_signal,
                               volatile uint32_t* status) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1) return;  // Single PPE thread for now

    PPEState& s = states[tid];

    for (uint32_t cycle = 0; cycle < maxCycles && !s.halted; cycle++) {
        int result = execOne(s, mem, hle_log, hle_signal);
        if (result == 1) break; // halted
        if (result == 2) {
            // Unimplemented — log the PC and instruction, then skip
            uint32_t inst = mem_read32(mem, s.pc - 4); // already advanced
            if (hle_log) {
                uint32_t idx = atomicAdd(hle_log, 1);
                if (idx < 255) {
                    hle_log[1 + idx] = 0xDEAD0000 | OPCD(inst);
                }
            }
        }
    }

    // Signal completion
    if (status) atomicExch((uint32_t*)status, s.halted ? 2 : 1);
}

// ═══════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════

extern "C" {

struct MegakernelCtx {
    PPEState*          d_states;
    uint8_t*           d_mem;
    uint32_t*          d_hle_log;
    uint32_t*          d_hle_signal;
    uint32_t*          d_status;
    cudaStream_t       stream;
    bool               ready;
};

static MegakernelCtx g_ctx = {};

int megakernel_init() {
    if (g_ctx.ready) return 1;

    cudaStreamCreate(&g_ctx.stream);

    // Allocate PPE state (1 core for now)
    cudaMalloc(&g_ctx.d_states, sizeof(PPEState));
    cudaMemset(g_ctx.d_states, 0, sizeof(PPEState));

    // Allocate PS3 memory sandbox (512 MB)
    cudaMalloc(&g_ctx.d_mem, PS3_SANDBOX_SIZE);
    cudaMemset(g_ctx.d_mem, 0, PS3_SANDBOX_SIZE);

    // HLE logging buffer (1024 entries)
    cudaMalloc(&g_ctx.d_hle_log, 1024 * sizeof(uint32_t));
    cudaMemset(g_ctx.d_hle_log, 0, 1024 * sizeof(uint32_t));

    // Signal/status
    cudaMalloc(&g_ctx.d_hle_signal, sizeof(uint32_t));
    cudaMemset(g_ctx.d_hle_signal, 0, sizeof(uint32_t));
    cudaMalloc(&g_ctx.d_status, sizeof(uint32_t));
    cudaMemset(g_ctx.d_status, 0, sizeof(uint32_t));

    g_ctx.ready = true;
    fprintf(stderr, "[PPE] Megakernel initialized (512MB sandbox)\n");
    return 1;
}

// Load raw bytes into PS3 memory at given offset
int megakernel_load(uint64_t offset, const void* data, size_t size) {
    if (!g_ctx.ready || offset + size > PS3_SANDBOX_SIZE) return 0;
    cudaMemcpyAsync(g_ctx.d_mem + offset, data, size,
                     cudaMemcpyHostToDevice, g_ctx.stream);
    return 1;
}

// Set initial PPE state (entry point, stack pointer, etc.)
int megakernel_set_entry(uint64_t pc, uint64_t sp, uint64_t toc) {
    if (!g_ctx.ready) return 0;
    PPEState init = {};
    init.pc = pc;
    init.gpr[1] = sp;          // Stack pointer
    init.gpr[2] = toc;         // Table of Contents (PPC64 ABI)
    init.gpr[3] = 0;           // argc
    init.gpr[13] = sp - 0x7000; // Small data area base (r13)
    init.msr = 0x8000000000000000ULL; // 64-bit mode
    cudaMemcpyAsync(g_ctx.d_states, &init, sizeof(PPEState),
                     cudaMemcpyHostToDevice, g_ctx.stream);
    fprintf(stderr, "[PPE] Entry: PC=0x%llx SP=0x%llx TOC=0x%llx\n",
            (unsigned long long)pc, (unsigned long long)sp, (unsigned long long)toc);
    return 1;
}

// Run N cycles of the megakernel
float megakernel_run(uint32_t maxCycles) {
    if (!g_ctx.ready) return -1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Reset status
    cudaMemsetAsync(g_ctx.d_status, 0, sizeof(uint32_t), g_ctx.stream);

    cudaEventRecord(start, g_ctx.stream);

    ppeMegakernel<<<1, 1, 0, g_ctx.stream>>>(
        g_ctx.d_states, g_ctx.d_mem, maxCycles,
        g_ctx.d_hle_log, g_ctx.d_hle_signal, g_ctx.d_status);

    cudaEventRecord(stop, g_ctx.stream);
    cudaStreamSynchronize(g_ctx.stream);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Readback PPE state for debugging
int megakernel_read_state(PPEState* out) {
    if (!g_ctx.ready || !out) return 0;
    cudaMemcpy(out, g_ctx.d_states, sizeof(PPEState), cudaMemcpyDeviceToHost);
    return 1;
}

// Readback HLE log
int megakernel_read_hle_log(uint32_t* out, int maxEntries) {
    if (!g_ctx.ready || !out) return 0;
    cudaMemcpy(out, g_ctx.d_hle_log, (1 + maxEntries) * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    return out[0]; // count
}

void megakernel_shutdown() {
    if (!g_ctx.ready) return;
    cudaFree(g_ctx.d_states);
    cudaFree(g_ctx.d_mem);
    cudaFree(g_ctx.d_hle_log);
    cudaFree(g_ctx.d_hle_signal);
    cudaFree(g_ctx.d_status);
    cudaStreamDestroy(g_ctx.stream);
    g_ctx.ready = false;
    fprintf(stderr, "[PPE] Megakernel shutdown\n");
}

} // extern "C"
