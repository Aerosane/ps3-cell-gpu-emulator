// ppc_jit.cu — PPE Basic-Block JIT Compiler
//
// Translates PowerPC instruction sequences into native CUDA kernels at runtime
// using NVRTC (NVIDIA Runtime Compilation). Each compiled block executes
// PPE instructions at native GPU speed — no fetch, no decode, no switch.
//
// Pipeline: PS3 RAM bytes → decode → CUDA C++ source → NVRTC → cubin → cuFunction
//
#include "ppc_jit.h"
#include "ppc_defs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>

using namespace ppc;

namespace ppc_jit {

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

static uint32_t bswap32_host(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

static uint64_t hash_mem_region(const uint8_t* data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static uint32_t fetch_inst_host(const uint8_t* mem, uint64_t pc) {
    uint64_t off = pc & (PS3_SANDBOX_SIZE - 1);
    uint32_t raw;
    memcpy(&raw, mem + off, 4);
    return bswap32_host(raw);
}

// Formatted emit to source buffer
static int emit(char* buf, size_t bufSize, size_t* pos, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(buf + *pos, bufSize - *pos, fmt, args);
    va_end(args);
    if (written > 0) *pos += (size_t)written;
    return written;
}

// ═══════════════════════════════════════════════════════════════
// Instruction Decoding
// ═══════════════════════════════════════════════════════════════

static void decode_ppc_insn(uint32_t raw, uint64_t pc, PPCDecodedInsn* out) {
    memset(out, 0, sizeof(PPCDecodedInsn));
    out->raw = raw;
    out->pc  = pc;
    out->opcd = OPCD(raw);
    out->rd   = RD(raw);
    out->ra   = RA(raw);
    out->rb   = RB(raw);
    out->imm  = SIMM16(raw);
    out->uimm = UIMM16(raw);
    out->sh   = SH(raw);
    out->mb   = MB(raw);
    out->me   = ME(raw);
    out->bo   = BO(raw);
    out->bi   = BI(raw);
    out->spr  = SPR(raw);
    out->crm  = CRM(raw);
    out->rc   = RC(raw);
    out->lk   = LK(raw);
    out->aa   = AA(raw);

    switch (out->opcd) {
    case OP_B:
        out->isBranch = 1;
        out->imm = LI26(raw);
        break;
    case OP_BC:
        out->isBranch = 1;
        out->imm = BD16(raw);
        break;
    case OP_SC:
        out->isSyscall = 1;
        break;
    case OP_GRP19: {
        out->xo = XO_10(raw);
        if (out->xo == XO_BCLR || out->xo == XO_BCCTR)
            out->isBranch = 1;
        break;
    }
    case OP_GRP31:
        out->xo = XO_10(raw);
        break;
    case OP_GRP59:
    case OP_GRP63:
        out->xo = XO_10(raw);
        break;
    default:
        break;
    }
}

// ═══════════════════════════════════════════════════════════════
// Basic Block Discovery
// ═══════════════════════════════════════════════════════════════

int ppc_jit_discover_block(const uint8_t* mem, uint64_t entryPC,
                           PPCBasicBlock* out) {
    memset(out, 0, sizeof(PPCBasicBlock));
    out->entryPC = entryPC;

    uint64_t pc = entryPC;
    uint32_t n = 0;

    while (n < MAX_BLOCK_INSNS) {
        if (pc >= PS3_SANDBOX_SIZE) break;

        uint32_t raw = fetch_inst_host(mem, pc);
        PPCDecodedInsn* di = &out->insns[n];
        decode_ppc_insn(raw, pc, di);

        // Track GPR usage based on opcode
        uint32_t opcd = di->opcd;
        uint32_t rd = di->rd, ra = di->ra, rb = di->rb;

        // Mark GPR reads/writes based on instruction class
        switch (opcd) {
        case OP_ADDI: case OP_ADDIS: case OP_ADDIC: case OP_ADDIC_D:
        case OP_SUBFIC: case OP_MULLI:
            if (ra != 0 || (opcd != OP_ADDI && opcd != OP_ADDIS)) {
                if (ra < 32) out->usesGPR[ra] = true;
            }
            if (rd < 32) { out->usesGPR[rd] = true; out->writesGPR[rd] = true; }
            if (opcd == OP_ADDIC || opcd == OP_ADDIC_D || opcd == OP_SUBFIC)
                out->usesXER = out->writesXER = true;
            if (opcd == OP_ADDIC_D) out->writesCR = out->usesCR = true;
            break;
        case OP_CMPI: case OP_CMPLI:
            if (ra < 32) out->usesGPR[ra] = true;
            out->writesCR = out->usesCR = true;
            break;
        case OP_ORI: case OP_ORIS: case OP_XORI: case OP_XORIS:
            if (rd < 32) out->usesGPR[rd] = true; // rS
            if (ra < 32) { out->usesGPR[ra] = true; out->writesGPR[ra] = true; }
            break;
        case OP_ANDI: case OP_ANDIS:
            if (rd < 32) out->usesGPR[rd] = true;
            if (ra < 32) { out->usesGPR[ra] = true; out->writesGPR[ra] = true; }
            out->writesCR = out->usesCR = true;
            break;
        case OP_RLWINM: case OP_RLWIMI: case OP_RLWNM:
            if (rd < 32) out->usesGPR[rd] = true; // rS
            if (ra < 32) { out->usesGPR[ra] = true; out->writesGPR[ra] = true; }
            if (opcd == OP_RLWNM && rb < 32) out->usesGPR[rb] = true;
            if (di->rc) out->writesCR = out->usesCR = true;
            break;
        case OP_LWZ: case OP_LWZU: case OP_LBZ: case OP_LBZU:
        case OP_LHZ: case OP_LHZU: case OP_LHA:
            if (ra != 0) { if (ra < 32) out->usesGPR[ra] = true; }
            if (rd < 32) { out->usesGPR[rd] = true; out->writesGPR[rd] = true; }
            if (opcd == OP_LWZU || opcd == OP_LBZU || opcd == OP_LHZU) {
                if (ra < 32) out->writesGPR[ra] = true;
            }
            out->accessesMem = true;
            break;
        case OP_STW: case OP_STWU: case OP_STB: case OP_STBU:
        case OP_STH: case OP_STHU:
            if (rd < 32) out->usesGPR[rd] = true; // rS
            if (ra != 0) { if (ra < 32) out->usesGPR[ra] = true; }
            if (opcd == OP_STWU || opcd == OP_STBU || opcd == OP_STHU) {
                if (ra < 32) out->writesGPR[ra] = true;
            }
            out->accessesMem = true;
            break;
        case OP_LFS: case OP_LFD:
            if (ra != 0) { if (ra < 32) out->usesGPR[ra] = true; }
            if (rd < 32) { out->usesFPR[rd] = true; out->writesFPR[rd] = true; }
            out->accessesMem = true;
            break;
        case OP_STFS: case OP_STFD:
            if (rd < 32) out->usesFPR[rd] = true;
            if (ra != 0) { if (ra < 32) out->usesGPR[ra] = true; }
            out->accessesMem = true;
            break;
        case OP_LMW:
            if (ra != 0) { if (ra < 32) out->usesGPR[ra] = true; }
            for (uint32_t r = rd; r < 32; r++) {
                out->usesGPR[r] = true; out->writesGPR[r] = true;
            }
            out->accessesMem = true;
            break;
        case OP_STMW:
            if (ra != 0) { if (ra < 32) out->usesGPR[ra] = true; }
            for (uint32_t r = rd; r < 32; r++) out->usesGPR[r] = true;
            out->accessesMem = true;
            break;
        case OP_B:
            if (di->lk) out->writesLR = out->usesLR = true;
            break;
        case OP_BC:
            out->usesCR = true;
            if (di->lk) out->writesLR = out->usesLR = true;
            // BO field may use CTR
            if (!(di->bo & 0x04)) out->usesCTR = out->writesCTR = true;
            break;
        case OP_GRP19:
            if (di->xo == XO_BCLR) {
                out->usesLR = true;
                out->usesCR = true;
                if (!(di->bo & 0x04)) out->usesCTR = out->writesCTR = true;
                if (di->lk) out->writesLR = true;
            } else if (di->xo == XO_BCCTR) {
                out->usesCTR = true;
                out->usesCR = true;
                if (di->lk) out->writesLR = out->usesLR = true;
            }
            break;
        case OP_GRP31:
            // Mark usage for common group 31 ops
            switch (di->xo) {
            case XO_ADD: case XO_SUBF: case XO_SUBFC: case XO_ADDC:
            case XO_ADDE: case XO_MULLW: case XO_DIVW: case XO_DIVWU:
            case XO_MULHW: case XO_MULHWU:
                if (ra < 32) out->usesGPR[ra] = true;
                if (rb < 32) out->usesGPR[rb] = true;
                if (rd < 32) { out->usesGPR[rd] = true; out->writesGPR[rd] = true; }
                if (di->rc) out->writesCR = out->usesCR = true;
                if (di->xo == XO_SUBFC || di->xo == XO_ADDC || di->xo == XO_ADDE)
                    out->usesXER = out->writesXER = true;
                break;
            case XO_ADDZE:
                if (ra < 32) out->usesGPR[ra] = true;
                if (rd < 32) { out->usesGPR[rd] = true; out->writesGPR[rd] = true; }
                out->usesXER = out->writesXER = true;
                if (di->rc) out->writesCR = out->usesCR = true;
                break;
            case XO_NEG:
                if (ra < 32) out->usesGPR[ra] = true;
                if (rd < 32) { out->usesGPR[rd] = true; out->writesGPR[rd] = true; }
                if (di->rc) out->writesCR = out->usesCR = true;
                break;
            case XO_AND: case XO_OR: case XO_XOR: case XO_ANDC:
            case XO_ORC: case XO_NOR: case XO_NAND:
                if (rd < 32) out->usesGPR[rd] = true; // rS
                if (rb < 32) out->usesGPR[rb] = true;
                if (ra < 32) { out->usesGPR[ra] = true; out->writesGPR[ra] = true; }
                if (di->rc) out->writesCR = out->usesCR = true;
                break;
            case XO_EXTSB: case XO_EXTSH: case XO_CNTLZW:
                if (rd < 32) out->usesGPR[rd] = true; // rS
                if (ra < 32) { out->usesGPR[ra] = true; out->writesGPR[ra] = true; }
                if (di->rc) out->writesCR = out->usesCR = true;
                break;
            case XO_SLW: case XO_SRW: case XO_SRAW:
                if (rd < 32) out->usesGPR[rd] = true;
                if (rb < 32) out->usesGPR[rb] = true;
                if (ra < 32) { out->usesGPR[ra] = true; out->writesGPR[ra] = true; }
                if (di->xo == XO_SRAW) out->usesXER = out->writesXER = true;
                if (di->rc) out->writesCR = out->usesCR = true;
                break;
            case XO_SRAWI:
                if (rd < 32) out->usesGPR[rd] = true;
                if (ra < 32) { out->usesGPR[ra] = true; out->writesGPR[ra] = true; }
                out->usesXER = out->writesXER = true;
                if (di->rc) out->writesCR = out->usesCR = true;
                break;
            case XO_CMP: case XO_CMPL:
                if (ra < 32) out->usesGPR[ra] = true;
                if (rb < 32) out->usesGPR[rb] = true;
                out->writesCR = out->usesCR = true;
                break;
            case XO_LWZX: case XO_LBZX: case XO_LHZX: case XO_LHAX:
                if (ra != 0 && ra < 32) out->usesGPR[ra] = true;
                if (rb < 32) out->usesGPR[rb] = true;
                if (rd < 32) { out->usesGPR[rd] = true; out->writesGPR[rd] = true; }
                out->accessesMem = true;
                break;
            case XO_STWX: case XO_STBX: case XO_STHX:
                if (rd < 32) out->usesGPR[rd] = true;
                if (ra != 0 && ra < 32) out->usesGPR[ra] = true;
                if (rb < 32) out->usesGPR[rb] = true;
                out->accessesMem = true;
                break;
            case XO_MFSPR:
                if (rd < 32) { out->usesGPR[rd] = true; out->writesGPR[rd] = true; }
                { uint32_t sn = di->spr;
                  if (sn == SPR_LR) out->usesLR = true;
                  else if (sn == SPR_CTR) out->usesCTR = true;
                  else if (sn == SPR_XER) out->usesXER = true;
                }
                break;
            case XO_MTSPR:
                if (rd < 32) out->usesGPR[rd] = true; // rS
                { uint32_t sn = di->spr;
                  if (sn == SPR_LR) out->writesLR = out->usesLR = true;
                  else if (sn == SPR_CTR) out->writesCTR = out->usesCTR = true;
                  else if (sn == SPR_XER) out->writesXER = out->usesXER = true;
                }
                break;
            case XO_MFCR:
                if (rd < 32) { out->usesGPR[rd] = true; out->writesGPR[rd] = true; }
                out->usesCR = true;
                break;
            case XO_MTCRF:
                if (rd < 32) out->usesGPR[rd] = true;
                out->writesCR = out->usesCR = true;
                break;
            default:
                break;
            }
            break;
        default:
            break;
        }

        n++;
        out->endPC = pc;

        if (di->isBranch || di->isSyscall) break;

        // Check for truly unimplemented primary opcodes
        bool known = false;
        switch (opcd) {
        case OP_ADDI: case OP_ADDIS: case OP_ADDIC: case OP_ADDIC_D:
        case OP_SUBFIC: case OP_MULLI:
        case OP_CMPI: case OP_CMPLI:
        case OP_ORI: case OP_ORIS: case OP_XORI: case OP_XORIS:
        case OP_ANDI: case OP_ANDIS:
        case OP_RLWINM: case OP_RLWIMI: case OP_RLWNM:
        case OP_LWZ: case OP_LWZU: case OP_LBZ: case OP_LBZU:
        case OP_LHZ: case OP_LHZU: case OP_LHA:
        case OP_STW: case OP_STWU: case OP_STB: case OP_STBU:
        case OP_STH: case OP_STHU:
        case OP_LFS: case OP_LFD: case OP_STFS: case OP_STFD:
        case OP_LMW: case OP_STMW:
        case OP_GRP31: case OP_GRP19:
            known = true;
            break;
        default:
            break;
        }
        if (!known) {
            di->isUnimpl = 1;
            break;
        }

        pc += 4;
    }

    out->numInsns = n;

    // Hash the memory region for cache validation
    if (out->endPC >= out->entryPC) {
        uint64_t blockBytes = out->endPC - out->entryPC + 4;
        if (out->entryPC + blockBytes <= PS3_SANDBOX_SIZE) {
            out->memHash = hash_mem_region(mem + out->entryPC, (size_t)blockBytes);
        }
    }

    return (int)n;
}

// ═══════════════════════════════════════════════════════════════
// CUDA C++ Source Emission
// ═══════════════════════════════════════════════════════════════

int ppc_jit_emit_source(const PPCBasicBlock* block, char* buf, size_t bufSize) {
    size_t pos = 0;

    // Header: typedefs, helpers (no system includes for NVRTC)
    emit(buf, bufSize, &pos,
        "// Auto-generated PPE JIT block: PC=0x%llx, %u instructions\n"
        "typedef unsigned int uint32_t;\n"
        "typedef int int32_t;\n"
        "typedef unsigned short uint16_t;\n"
        "typedef short int16_t;\n"
        "typedef unsigned char uint8_t;\n"
        "typedef signed char int8_t;\n"
        "typedef unsigned long long uint64_t;\n"
        "typedef long long int64_t;\n\n",
        (unsigned long long)block->entryPC, block->numInsns);

    // Byte-swap helpers
    emit(buf, bufSize, &pos,
        "__device__ __forceinline__ uint32_t bswap32(uint32_t x) {\n"
        "    return __byte_perm(x, 0, 0x0123);\n"
        "}\n"
        "__device__ __forceinline__ uint16_t bswap16(uint16_t x) {\n"
        "    return (uint16_t)__byte_perm((uint32_t)x, 0, 0x0001);\n"
        "}\n"
        "__device__ __forceinline__ uint64_t bswap64(uint64_t x) {\n"
        "    uint32_t lo = (uint32_t)x;\n"
        "    uint32_t hi = (uint32_t)(x >> 32);\n"
        "    return ((uint64_t)bswap32(lo) << 32) | (uint64_t)bswap32(hi);\n"
        "}\n\n");

    // Memory access helpers
    emit(buf, bufSize, &pos,
        "static const uint64_t SANDBOX = 0x%llxULL;\n"
        "__device__ __forceinline__ uint32_t mem_rd32(const uint8_t* m, uint64_t a) {\n"
        "    uint32_t r; memcpy(&r, m + (a & (SANDBOX-1)), 4); return bswap32(r);\n"
        "}\n"
        "__device__ __forceinline__ uint16_t mem_rd16(const uint8_t* m, uint64_t a) {\n"
        "    uint16_t r; memcpy(&r, m + (a & (SANDBOX-1)), 2); return bswap16(r);\n"
        "}\n"
        "__device__ __forceinline__ uint8_t mem_rd8(const uint8_t* m, uint64_t a) {\n"
        "    return m[a & (SANDBOX-1)];\n"
        "}\n"
        "__device__ __forceinline__ void mem_wr32(uint8_t* m, uint64_t a, uint32_t v) {\n"
        "    uint32_t s = bswap32(v); memcpy(m + (a & (SANDBOX-1)), &s, 4);\n"
        "}\n"
        "__device__ __forceinline__ void mem_wr16(uint8_t* m, uint64_t a, uint16_t v) {\n"
        "    uint16_t s = bswap16(v); memcpy(m + (a & (SANDBOX-1)), &s, 2);\n"
        "}\n"
        "__device__ __forceinline__ void mem_wr8(uint8_t* m, uint64_t a, uint8_t v) {\n"
        "    m[a & (SANDBOX-1)] = v;\n"
        "}\n",
        (unsigned long long)PS3_SANDBOX_SIZE);

    // FP memory helpers
    emit(buf, bufSize, &pos,
        "__device__ __forceinline__ float mem_rdf32(const uint8_t* m, uint64_t a) {\n"
        "    uint32_t b = mem_rd32(m, a); float f; memcpy(&f, &b, 4); return f;\n"
        "}\n"
        "__device__ __forceinline__ double mem_rdf64(const uint8_t* m, uint64_t a) {\n"
        "    uint64_t b = bswap64(*(const uint64_t*)(m + (a & (SANDBOX-1))));\n"
        "    double d; memcpy(&d, &b, 8); return d;\n"
        "}\n"
        "__device__ __forceinline__ void mem_wrf32(uint8_t* m, uint64_t a, float f) {\n"
        "    uint32_t b; memcpy(&b, &f, 4); mem_wr32(m, a, b);\n"
        "}\n"
        "__device__ __forceinline__ void mem_wrf64(uint8_t* m, uint64_t a, double d) {\n"
        "    uint64_t b; memcpy(&b, &d, 8); uint64_t s = bswap64(b);\n"
        "    memcpy(m + (a & (SANDBOX-1)), &s, 8);\n"
        "}\n\n");

    // CR helpers
    emit(buf, bufSize, &pos,
        "__device__ __forceinline__ void setCR(uint32_t& cr, int field, int64_t result, uint64_t xer) {\n"
        "    uint32_t val = 0;\n"
        "    if (result < 0) val = 0x8;\n"
        "    else if (result > 0) val = 0x4;\n"
        "    else val = 0x2;\n"
        "    if (xer & (1ULL << 31)) val |= 0x1;\n"
        "    int shift = (7 - field) * 4;\n"
        "    cr = (cr & ~(0xFu << shift)) | (val << shift);\n"
        "}\n"
        "__device__ __forceinline__ bool getCRBit(uint32_t cr, int bit) {\n"
        "    return (cr >> (31 - bit)) & 1;\n"
        "}\n"
        "__device__ __forceinline__ bool getCA(uint64_t xer) { return (xer >> 29) & 1; }\n"
        "__device__ __forceinline__ void setCA(uint64_t& xer, bool ca) {\n"
        "    xer = ca ? (xer | (1ULL << 29)) : (xer & ~(1ULL << 29));\n"
        "}\n\n");

    // Rotate helpers
    emit(buf, bufSize, &pos,
        "__device__ __forceinline__ uint32_t rotl32(uint32_t v, uint32_t n) {\n"
        "    n &= 31; return (v << n) | (v >> (32 - n));\n"
        "}\n"
        "__device__ __forceinline__ uint32_t rotateMask32(uint32_t mb, uint32_t me) {\n"
        "    uint32_t mask = 0;\n"
        "    if (mb <= me) { for (uint32_t i = mb; i <= me; i++) mask |= (1u << (31 - i)); }\n"
        "    else { for (uint32_t i = 0; i <= me; i++) mask |= (1u << (31 - i));\n"
        "           for (uint32_t i = mb; i <= 31; i++) mask |= (1u << (31 - i)); }\n"
        "    return mask;\n"
        "}\n\n");

    // Kernel signature
    // spr layout: [0]=LR, [1]=CTR, [2]=XER, [3]=PC, [4]=halted
    emit(buf, bufSize, &pos,
        "extern \"C\" __global__ void ppc_jit_block_0x%llx(\n"
        "    uint64_t* __restrict__ gpr,\n"
        "    double*   __restrict__ fpr,\n"
        "    uint64_t* __restrict__ spr,\n"
        "    uint32_t* __restrict__ cr_ptr,\n"
        "    uint8_t*  __restrict__ mem)\n"
        "{\n"
        "    if (threadIdx.x != 0) return;\n\n",
        (unsigned long long)block->entryPC);

    // Register promotion: load used GPRs into locals
    emit(buf, bufSize, &pos, "    // Register promotion: load GPRs\n");
    for (int i = 0; i < 32; i++) {
        if (block->usesGPR[i]) {
            emit(buf, bufSize, &pos, "    uint64_t r%d = gpr[%d];\n", i, i);
        }
    }
    // FPR promotion
    for (int i = 0; i < 32; i++) {
        if (block->usesFPR[i]) {
            emit(buf, bufSize, &pos, "    double f%d = fpr[%d];\n", i, i);
        }
    }
    // SPR locals
    if (block->usesLR || block->writesLR)
        emit(buf, bufSize, &pos, "    uint64_t lr = spr[0];\n");
    if (block->usesCTR || block->writesCTR)
        emit(buf, bufSize, &pos, "    uint64_t ctr = spr[1];\n");
    // XER is always needed when CR is used (setCR reads XER.SO)
    if (block->usesXER || block->writesXER || block->usesCR || block->writesCR)
        emit(buf, bufSize, &pos, "    uint64_t xer = spr[2];\n");
    if (block->usesCR || block->writesCR)
        emit(buf, bufSize, &pos, "    uint32_t cr = *cr_ptr;\n");
    emit(buf, bufSize, &pos, "    uint64_t nextPC = 0x%llxULL;\n\n",
         (unsigned long long)(block->endPC + 4));

    // Emit each instruction
    for (uint32_t idx = 0; idx < block->numInsns; idx++) {
        const PPCDecodedInsn& di = block->insns[idx];
        emit(buf, bufSize, &pos, "    // [0x%llx] opcd=%u",
             (unsigned long long)di.pc, di.opcd);

        if (di.opcd == OP_GRP31 || di.opcd == OP_GRP19)
            emit(buf, bufSize, &pos, " xo=%u", di.xo);
        emit(buf, bufSize, &pos, "\n");

        uint32_t rd = di.rd, ra = di.ra, rb = di.rb;

        switch (di.opcd) {

        // ─── Immediate ALU ────────────────────────────────
        case OP_ADDI: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)((int64_t)%lld);\n", rd, (long long)imm);
            else
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)((int64_t)r%u + (int64_t)%lld);\n", rd, ra, (long long)imm);
            break;
        }
        case OP_ADDIS: {
            int64_t imm = di.imm << 16;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)((int64_t)%lld);\n", rd, (long long)imm);
            else
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)((int64_t)r%u + (int64_t)%lld);\n", rd, ra, (long long)imm);
            break;
        }
        case OP_ADDIC: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    { uint64_t _a = r%u; uint64_t _r = _a + (uint64_t)((int64_t)%lld);\n"
                "      r%u = _r; setCA(xer, (uint32_t)_r < (uint32_t)_a); }\n",
                ra, (long long)imm, rd);
            break;
        }
        case OP_ADDIC_D: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    { uint64_t _a = r%u; uint64_t _r = _a + (uint64_t)((int64_t)%lld);\n"
                "      r%u = _r; setCA(xer, (uint32_t)_r < (uint32_t)_a);\n"
                "      setCR(cr, 0, (int32_t)(uint32_t)_r, xer); }\n",
                ra, (long long)imm, rd);
            break;
        }
        case OP_SUBFIC: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    { uint64_t _r = (uint64_t)((int64_t)%lld) - r%u;\n"
                "      r%u = _r; setCA(xer, (uint32_t)r%u <= (uint32_t)(uint64_t)((int64_t)%lld)); }\n",
                (long long)imm, ra, rd, ra, (long long)imm);
            break;
        }
        case OP_MULLI: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    r%u = (uint64_t)((int64_t)r%u * (int64_t)%lld);\n",
                rd, ra, (long long)imm);
            break;
        }

        // ─── Compare ──────────────────────────────────────
        case OP_CMPI: {
            uint32_t bf = rd >> 2;
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    setCR(cr, %u, (int64_t)(int32_t)(uint32_t)r%u - (int64_t)%lld, xer);\n",
                bf, ra, (long long)imm);
            break;
        }
        case OP_CMPLI: {
            uint32_t bf = rd >> 2;
            uint64_t imm = di.uimm;
            emit(buf, bufSize, &pos,
                "    { uint32_t _a = (uint32_t)r%u; int64_t _d = (_a > (uint32_t)%lluULL) ? 1 : (_a < (uint32_t)%lluULL) ? -1 : 0;\n"
                "      setCR(cr, %u, _d, xer); }\n",
                ra, (unsigned long long)imm, (unsigned long long)imm, bf);
            break;
        }

        // ─── Logical Immediate ────────────────────────────
        case OP_ORI:
            emit(buf, bufSize, &pos, "    r%u = r%u | %lluULL;\n", ra, rd, (unsigned long long)di.uimm);
            break;
        case OP_ORIS:
            emit(buf, bufSize, &pos, "    r%u = r%u | (%lluULL << 16);\n", ra, rd, (unsigned long long)di.uimm);
            break;
        case OP_XORI:
            emit(buf, bufSize, &pos, "    r%u = r%u ^ %lluULL;\n", ra, rd, (unsigned long long)di.uimm);
            break;
        case OP_XORIS:
            emit(buf, bufSize, &pos, "    r%u = r%u ^ (%lluULL << 16);\n", ra, rd, (unsigned long long)di.uimm);
            break;
        case OP_ANDI:
            emit(buf, bufSize, &pos, "    r%u = r%u & %lluULL;\n", ra, rd, (unsigned long long)di.uimm);
            emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
            break;
        case OP_ANDIS:
            emit(buf, bufSize, &pos, "    r%u = r%u & (%lluULL << 16);\n", ra, rd, (unsigned long long)di.uimm);
            emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
            break;

        // ─── Rotate / Mask ───────────────────────────────
        case OP_RLWINM: {
            emit(buf, bufSize, &pos,
                "    { uint32_t _rot = rotl32((uint32_t)r%u, %uu);\n"
                "      uint32_t _mask = rotateMask32(%uu, %uu);\n"
                "      r%u = _rot & _mask; }\n",
                rd, di.sh, di.mb, di.me, ra);
            if (di.rc)
                emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
            break;
        }
        case OP_RLWIMI: {
            emit(buf, bufSize, &pos,
                "    { uint32_t _rot = rotl32((uint32_t)r%u, %uu);\n"
                "      uint32_t _mask = rotateMask32(%uu, %uu);\n"
                "      r%u = (_rot & _mask) | ((uint32_t)r%u & ~_mask); }\n",
                rd, di.sh, di.mb, di.me, ra, ra);
            if (di.rc)
                emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
            break;
        }
        case OP_RLWNM: {
            emit(buf, bufSize, &pos,
                "    { uint32_t _rot = rotl32((uint32_t)r%u, (uint32_t)r%u & 0x1Fu);\n"
                "      uint32_t _mask = rotateMask32(%uu, %uu);\n"
                "      r%u = _rot & _mask; }\n",
                rd, rb, di.mb, di.me, ra);
            if (di.rc)
                emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
            break;
        }

        // ─── Branches ─────────────────────────────────────
        case OP_B: {
            int64_t disp = LI26(di.raw);
            if (di.lk)
                emit(buf, bufSize, &pos, "    lr = 0x%llxULL;\n",
                     (unsigned long long)(di.pc + 4));
            if (di.aa)
                emit(buf, bufSize, &pos, "    nextPC = (uint64_t)((int64_t)%lld);\n", (long long)disp);
            else
                emit(buf, bufSize, &pos, "    nextPC = 0x%llxULL + (uint64_t)((int64_t)%lld);\n",
                     (unsigned long long)di.pc, (long long)disp);
            break;
        }
        case OP_BC: {
            int64_t disp = BD16(di.raw);
            uint32_t bo_val = di.bo, bi_val = di.bi;
            // Emit branch condition evaluation inline
            emit(buf, bufSize, &pos, "    {\n");
            emit(buf, bufSize, &pos, "        bool _ctr_ok = true, _cond_ok = true;\n");
            if (!(bo_val & 0x04)) {
                emit(buf, bufSize, &pos, "        ctr--;\n");
                if (bo_val & 0x02)
                    emit(buf, bufSize, &pos, "        _ctr_ok = (ctr == 0);\n");
                else
                    emit(buf, bufSize, &pos, "        _ctr_ok = (ctr != 0);\n");
            }
            if (!(bo_val & 0x10)) {
                emit(buf, bufSize, &pos, "        bool _bit = getCRBit(cr, %u);\n", bi_val);
                if (bo_val & 0x08)
                    emit(buf, bufSize, &pos, "        _cond_ok = _bit;\n");
                else
                    emit(buf, bufSize, &pos, "        _cond_ok = !_bit;\n");
            }
            emit(buf, bufSize, &pos, "        if (_ctr_ok && _cond_ok) {\n");
            if (di.aa)
                emit(buf, bufSize, &pos, "            nextPC = (uint64_t)((int64_t)%lld);\n", (long long)disp);
            else
                emit(buf, bufSize, &pos, "            nextPC = 0x%llxULL + (uint64_t)((int64_t)%lld);\n",
                     (unsigned long long)di.pc, (long long)disp);
            emit(buf, bufSize, &pos, "        } else {\n");
            emit(buf, bufSize, &pos, "            nextPC = 0x%llxULL;\n",
                 (unsigned long long)(di.pc + 4));
            emit(buf, bufSize, &pos, "        }\n");
            if (di.lk)
                emit(buf, bufSize, &pos, "        lr = 0x%llxULL;\n",
                     (unsigned long long)(di.pc + 4));
            emit(buf, bufSize, &pos, "    }\n");
            break;
        }
        case OP_SC:
            emit(buf, bufSize, &pos, "    nextPC = 0x%llxULL;\n",
                 (unsigned long long)(di.pc + 4));
            emit(buf, bufSize, &pos, "    spr[4] = 1; // halt for SC\n");
            break;

        // ─── Group 19 ────────────────────────────────────
        case OP_GRP19: {
            switch (di.xo) {
            case XO_BCLR: {
                uint32_t bo_val = di.bo, bi_val = di.bi;
                emit(buf, bufSize, &pos, "    {\n");
                emit(buf, bufSize, &pos, "        bool _ctr_ok = true, _cond_ok = true;\n");
                if (!(bo_val & 0x04)) {
                    emit(buf, bufSize, &pos, "        ctr--;\n");
                    if (bo_val & 0x02)
                        emit(buf, bufSize, &pos, "        _ctr_ok = (ctr == 0);\n");
                    else
                        emit(buf, bufSize, &pos, "        _ctr_ok = (ctr != 0);\n");
                }
                if (!(bo_val & 0x10)) {
                    emit(buf, bufSize, &pos, "        bool _bit = getCRBit(cr, %u);\n", bi_val);
                    if (bo_val & 0x08)
                        emit(buf, bufSize, &pos, "        _cond_ok = _bit;\n");
                    else
                        emit(buf, bufSize, &pos, "        _cond_ok = !_bit;\n");
                }
                emit(buf, bufSize, &pos, "        if (_ctr_ok && _cond_ok) nextPC = lr & ~3ULL;\n");
                emit(buf, bufSize, &pos, "        else nextPC = 0x%llxULL;\n",
                     (unsigned long long)(di.pc + 4));
                if (di.lk)
                    emit(buf, bufSize, &pos, "        lr = 0x%llxULL;\n",
                         (unsigned long long)(di.pc + 4));
                emit(buf, bufSize, &pos, "    }\n");
                break;
            }
            case XO_BCCTR: {
                uint32_t bo_val = di.bo, bi_val = di.bi;
                emit(buf, bufSize, &pos, "    {\n");
                emit(buf, bufSize, &pos, "        bool _cond_ok = true;\n");
                if (!(bo_val & 0x10)) {
                    emit(buf, bufSize, &pos, "        bool _bit = getCRBit(cr, %u);\n", bi_val);
                    if (bo_val & 0x08)
                        emit(buf, bufSize, &pos, "        _cond_ok = _bit;\n");
                    else
                        emit(buf, bufSize, &pos, "        _cond_ok = !_bit;\n");
                }
                emit(buf, bufSize, &pos, "        if (_cond_ok) nextPC = ctr & ~3ULL;\n");
                emit(buf, bufSize, &pos, "        else nextPC = 0x%llxULL;\n",
                     (unsigned long long)(di.pc + 4));
                if (di.lk)
                    emit(buf, bufSize, &pos, "        lr = 0x%llxULL;\n",
                         (unsigned long long)(di.pc + 4));
                emit(buf, bufSize, &pos, "    }\n");
                break;
            }
            default:
                emit(buf, bufSize, &pos, "    // unhandled grp19 xo=%u → fallback\n", di.xo);
                emit(buf, bufSize, &pos, "    nextPC = 0x%llxULL; spr[4] = 2;\n",
                     (unsigned long long)di.pc);
                break;
            }
            break;
        }

        // ─── Load/Store D-form ────────────────────────────
        case OP_LWZ: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd32(mem, (uint64_t)((int64_t)%lld));\n", rd, (long long)imm);
            else
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd32(mem, r%u + (uint64_t)((int64_t)%lld));\n", rd, ra, (long long)imm);
            break;
        }
        case OP_LWZU: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    { uint64_t _ea = r%u + (uint64_t)((int64_t)%lld);\n"
                "      r%u = (uint64_t)mem_rd32(mem, _ea); r%u = _ea; }\n",
                ra, (long long)imm, rd, ra);
            break;
        }
        case OP_LBZ: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd8(mem, (uint64_t)((int64_t)%lld));\n", rd, (long long)imm);
            else
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd8(mem, r%u + (uint64_t)((int64_t)%lld));\n", rd, ra, (long long)imm);
            break;
        }
        case OP_LBZU: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    { uint64_t _ea = r%u + (uint64_t)((int64_t)%lld);\n"
                "      r%u = (uint64_t)mem_rd8(mem, _ea); r%u = _ea; }\n",
                ra, (long long)imm, rd, ra);
            break;
        }
        case OP_LHZ: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd16(mem, (uint64_t)((int64_t)%lld));\n", rd, (long long)imm);
            else
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd16(mem, r%u + (uint64_t)((int64_t)%lld));\n", rd, ra, (long long)imm);
            break;
        }
        case OP_LHZU: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    { uint64_t _ea = r%u + (uint64_t)((int64_t)%lld);\n"
                "      r%u = (uint64_t)mem_rd16(mem, _ea); r%u = _ea; }\n",
                ra, (long long)imm, rd, ra);
            break;
        }
        case OP_LHA: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)(int64_t)(int16_t)mem_rd16(mem, (uint64_t)((int64_t)%lld));\n", rd, (long long)imm);
            else
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)(int64_t)(int16_t)mem_rd16(mem, r%u + (uint64_t)((int64_t)%lld));\n", rd, ra, (long long)imm);
            break;
        }
        case OP_STW: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    mem_wr32(mem, (uint64_t)((int64_t)%lld), (uint32_t)r%u);\n", (long long)imm, rd);
            else
                emit(buf, bufSize, &pos, "    mem_wr32(mem, r%u + (uint64_t)((int64_t)%lld), (uint32_t)r%u);\n", ra, (long long)imm, rd);
            break;
        }
        case OP_STWU: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    { uint64_t _ea = r%u + (uint64_t)((int64_t)%lld);\n"
                "      mem_wr32(mem, _ea, (uint32_t)r%u); r%u = _ea; }\n",
                ra, (long long)imm, rd, ra);
            break;
        }
        case OP_STB: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    mem_wr8(mem, (uint64_t)((int64_t)%lld), (uint8_t)r%u);\n", (long long)imm, rd);
            else
                emit(buf, bufSize, &pos, "    mem_wr8(mem, r%u + (uint64_t)((int64_t)%lld), (uint8_t)r%u);\n", ra, (long long)imm, rd);
            break;
        }
        case OP_STBU: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    { uint64_t _ea = r%u + (uint64_t)((int64_t)%lld);\n"
                "      mem_wr8(mem, _ea, (uint8_t)r%u); r%u = _ea; }\n",
                ra, (long long)imm, rd, ra);
            break;
        }
        case OP_STH: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    mem_wr16(mem, (uint64_t)((int64_t)%lld), (uint16_t)r%u);\n", (long long)imm, rd);
            else
                emit(buf, bufSize, &pos, "    mem_wr16(mem, r%u + (uint64_t)((int64_t)%lld), (uint16_t)r%u);\n", ra, (long long)imm, rd);
            break;
        }
        case OP_STHU: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos,
                "    { uint64_t _ea = r%u + (uint64_t)((int64_t)%lld);\n"
                "      mem_wr16(mem, _ea, (uint16_t)r%u); r%u = _ea; }\n",
                ra, (long long)imm, rd, ra);
            break;
        }

        // ─── FP Load/Store ────────────────────────────────
        case OP_LFS: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    f%u = (double)mem_rdf32(mem, (uint64_t)((int64_t)%lld));\n", rd, (long long)imm);
            else
                emit(buf, bufSize, &pos, "    f%u = (double)mem_rdf32(mem, r%u + (uint64_t)((int64_t)%lld));\n", rd, ra, (long long)imm);
            break;
        }
        case OP_LFD: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    f%u = mem_rdf64(mem, (uint64_t)((int64_t)%lld));\n", rd, (long long)imm);
            else
                emit(buf, bufSize, &pos, "    f%u = mem_rdf64(mem, r%u + (uint64_t)((int64_t)%lld));\n", rd, ra, (long long)imm);
            break;
        }
        case OP_STFS: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    mem_wrf32(mem, (uint64_t)((int64_t)%lld), (float)f%u);\n", (long long)imm, rd);
            else
                emit(buf, bufSize, &pos, "    mem_wrf32(mem, r%u + (uint64_t)((int64_t)%lld), (float)f%u);\n", ra, (long long)imm, rd);
            break;
        }
        case OP_STFD: {
            int64_t imm = di.imm;
            if (ra == 0)
                emit(buf, bufSize, &pos, "    mem_wrf64(mem, (uint64_t)((int64_t)%lld), f%u);\n", (long long)imm, rd);
            else
                emit(buf, bufSize, &pos, "    mem_wrf64(mem, r%u + (uint64_t)((int64_t)%lld), f%u);\n", ra, (long long)imm, rd);
            break;
        }

        // ─── Load/Store Multiple ──────────────────────────
        case OP_LMW: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos, "    {\n");
            if (ra == 0)
                emit(buf, bufSize, &pos, "        uint64_t _ea = (uint64_t)((int64_t)%lld);\n", (long long)imm);
            else
                emit(buf, bufSize, &pos, "        uint64_t _ea = r%u + (uint64_t)((int64_t)%lld);\n", ra, (long long)imm);
            for (uint32_t r = rd; r < 32; r++) {
                emit(buf, bufSize, &pos, "        r%u = (uint64_t)mem_rd32(mem, _ea); _ea += 4;\n", r);
            }
            emit(buf, bufSize, &pos, "    }\n");
            break;
        }
        case OP_STMW: {
            int64_t imm = di.imm;
            emit(buf, bufSize, &pos, "    {\n");
            if (ra == 0)
                emit(buf, bufSize, &pos, "        uint64_t _ea = (uint64_t)((int64_t)%lld);\n", (long long)imm);
            else
                emit(buf, bufSize, &pos, "        uint64_t _ea = r%u + (uint64_t)((int64_t)%lld);\n", ra, (long long)imm);
            for (uint32_t r = rd; r < 32; r++) {
                emit(buf, bufSize, &pos, "        mem_wr32(mem, _ea, (uint32_t)r%u); _ea += 4;\n", r);
            }
            emit(buf, bufSize, &pos, "    }\n");
            break;
        }

        // ─── Group 31 ────────────────────────────────────
        case OP_GRP31: {
            switch (di.xo) {

            // ALU
            case XO_ADD:
                emit(buf, bufSize, &pos, "    r%u = (uint32_t)((uint32_t)r%u + (uint32_t)r%u);\n", rd, ra, rb);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_SUBF:
                emit(buf, bufSize, &pos, "    r%u = (uint32_t)((uint32_t)r%u - (uint32_t)r%u);\n", rd, rb, ra);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_SUBFC:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _a = (uint32_t)r%u, _b = (uint32_t)r%u;\n"
                    "      uint64_t _r = (uint64_t)_b + (uint64_t)(~_a) + 1ULL;\n"
                    "      r%u = (uint32_t)_r; setCA(xer, _r >> 32); }\n",
                    ra, rb, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_ADDC:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _a = (uint32_t)r%u, _b = (uint32_t)r%u;\n"
                    "      uint64_t _r = (uint64_t)_a + (uint64_t)_b;\n"
                    "      r%u = (uint32_t)_r; setCA(xer, _r >> 32); }\n",
                    ra, rb, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_ADDE:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _a = (uint32_t)r%u, _b = (uint32_t)r%u;\n"
                    "      uint64_t _r = (uint64_t)_a + (uint64_t)_b + (getCA(xer) ? 1ULL : 0ULL);\n"
                    "      r%u = (uint32_t)_r; setCA(xer, _r >> 32); }\n",
                    ra, rb, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_ADDZE:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _a = (uint32_t)r%u;\n"
                    "      uint64_t _r = (uint64_t)_a + (getCA(xer) ? 1ULL : 0ULL);\n"
                    "      r%u = (uint32_t)_r; setCA(xer, _r >> 32); }\n",
                    ra, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_NEG:
                emit(buf, bufSize, &pos, "    r%u = (uint32_t)(-(int32_t)(uint32_t)r%u);\n", rd, ra);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_MULLW:
                emit(buf, bufSize, &pos,
                    "    r%u = (uint32_t)(uint64_t)(int64_t)((int64_t)(int32_t)(uint32_t)r%u * (int64_t)(int32_t)(uint32_t)r%u);\n",
                    rd, ra, rb);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_MULHW:
                emit(buf, bufSize, &pos,
                    "    { int64_t _r = (int64_t)(int32_t)(uint32_t)r%u * (int64_t)(int32_t)(uint32_t)r%u;\n"
                    "      r%u = (uint32_t)(_r >> 32); }\n",
                    ra, rb, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_MULHWU:
                emit(buf, bufSize, &pos,
                    "    { uint64_t _r = (uint64_t)(uint32_t)r%u * (uint64_t)(uint32_t)r%u;\n"
                    "      r%u = (uint32_t)(_r >> 32); }\n",
                    ra, rb, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_DIVW:
                emit(buf, bufSize, &pos,
                    "    { int32_t _a = (int32_t)(uint32_t)r%u, _b = (int32_t)(uint32_t)r%u;\n"
                    "      r%u = (_b != 0 && !(_a == (int32_t)0x80000000 && _b == -1))\n"
                    "             ? (uint32_t)(_a / _b) : 0u; }\n",
                    ra, rb, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;
            case XO_DIVWU:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _a = (uint32_t)r%u, _b = (uint32_t)r%u;\n"
                    "      r%u = (_b != 0) ? (_a / _b) : 0u; }\n",
                    ra, rb, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", rd);
                break;

            // Logical
            case XO_AND:
                emit(buf, bufSize, &pos, "    r%u = r%u & r%u;\n", ra, rd, rb);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_OR:
                emit(buf, bufSize, &pos, "    r%u = r%u | r%u;\n", ra, rd, rb);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_XOR:
                emit(buf, bufSize, &pos, "    r%u = r%u ^ r%u;\n", ra, rd, rb);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_NOR:
                emit(buf, bufSize, &pos, "    r%u = ~(r%u | r%u);\n", ra, rd, rb);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_ANDC:
                emit(buf, bufSize, &pos, "    r%u = r%u & ~r%u;\n", ra, rd, rb);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_ORC:
                emit(buf, bufSize, &pos, "    r%u = r%u | ~r%u;\n", ra, rd, rb);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_NAND:
                emit(buf, bufSize, &pos, "    r%u = ~(r%u & r%u);\n", ra, rd, rb);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_EXTSB:
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)(int64_t)(int8_t)(uint8_t)r%u;\n", ra, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_EXTSH:
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)(int64_t)(int16_t)(uint16_t)r%u;\n", ra, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_CNTLZW:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _v = (uint32_t)r%u; r%u = _v ? __clz(_v) : 32; }\n", rd, ra);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;

            // Shifts
            case XO_SLW:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _sh = (uint32_t)r%u & 0x3Fu; r%u = (_sh < 32) ? ((uint32_t)r%u << _sh) : 0u; }\n",
                    rb, ra, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_SRW:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _sh = (uint32_t)r%u & 0x3Fu; r%u = (_sh < 32) ? ((uint32_t)r%u >> _sh) : 0u; }\n",
                    rb, ra, rd);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_SRAW:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _sh = (uint32_t)r%u & 0x3Fu; int32_t _v = (int32_t)(uint32_t)r%u;\n"
                    "      if (_sh == 0) { r%u = (uint32_t)_v; setCA(xer, false); }\n"
                    "      else if (_sh < 32) { bool _c = (_v < 0) && ((_v & ((1 << _sh) - 1)) != 0);\n"
                    "        r%u = (uint32_t)(_v >> _sh); setCA(xer, _c); }\n"
                    "      else { r%u = (_v < 0) ? 0xFFFFFFFFu : 0u; setCA(xer, _v < 0); } }\n",
                    rb, rd, ra, ra, ra);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;
            case XO_SRAWI:
                emit(buf, bufSize, &pos,
                    "    { uint32_t _sh = %uu; int32_t _v = (int32_t)(uint32_t)r%u;\n"
                    "      if (_sh == 0) { r%u = (uint32_t)_v; setCA(xer, false); }\n"
                    "      else { bool _c = (_v < 0) && ((_v & ((1 << _sh) - 1)) != 0);\n"
                    "        r%u = (uint32_t)(_v >> _sh); setCA(xer, _c); } }\n",
                    di.sh, rd, ra, ra);
                if (di.rc) emit(buf, bufSize, &pos, "    setCR(cr, 0, (int32_t)(uint32_t)r%u, xer);\n", ra);
                break;

            // Compare
            case XO_CMP: {
                uint32_t bf = rd >> 2;
                emit(buf, bufSize, &pos,
                    "    setCR(cr, %u, (int64_t)(int32_t)(uint32_t)r%u - (int64_t)(int32_t)(uint32_t)r%u, xer);\n",
                    bf, ra, rb);
                break;
            }
            case XO_CMPL: {
                uint32_t bf = rd >> 2;
                emit(buf, bufSize, &pos,
                    "    { uint32_t _a = (uint32_t)r%u, _b = (uint32_t)r%u;\n"
                    "      int64_t _d = (_a > _b) ? 1 : (_a < _b) ? -1 : 0;\n"
                    "      setCR(cr, %u, _d, xer); }\n",
                    ra, rb, bf);
                break;
            }

            // Indexed load/store
            case XO_LWZX:
                if (ra == 0)
                    emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd32(mem, r%u);\n", rd, rb);
                else
                    emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd32(mem, r%u + r%u);\n", rd, ra, rb);
                break;
            case XO_LBZX:
                if (ra == 0)
                    emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd8(mem, r%u);\n", rd, rb);
                else
                    emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd8(mem, r%u + r%u);\n", rd, ra, rb);
                break;
            case XO_LHZX:
                if (ra == 0)
                    emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd16(mem, r%u);\n", rd, rb);
                else
                    emit(buf, bufSize, &pos, "    r%u = (uint64_t)mem_rd16(mem, r%u + r%u);\n", rd, ra, rb);
                break;
            case XO_LHAX:
                if (ra == 0)
                    emit(buf, bufSize, &pos, "    r%u = (uint64_t)(int64_t)(int16_t)mem_rd16(mem, r%u);\n", rd, rb);
                else
                    emit(buf, bufSize, &pos, "    r%u = (uint64_t)(int64_t)(int16_t)mem_rd16(mem, r%u + r%u);\n", rd, ra, rb);
                break;
            case XO_STWX:
                if (ra == 0)
                    emit(buf, bufSize, &pos, "    mem_wr32(mem, r%u, (uint32_t)r%u);\n", rb, rd);
                else
                    emit(buf, bufSize, &pos, "    mem_wr32(mem, r%u + r%u, (uint32_t)r%u);\n", ra, rb, rd);
                break;
            case XO_STBX:
                if (ra == 0)
                    emit(buf, bufSize, &pos, "    mem_wr8(mem, r%u, (uint8_t)r%u);\n", rb, rd);
                else
                    emit(buf, bufSize, &pos, "    mem_wr8(mem, r%u + r%u, (uint8_t)r%u);\n", ra, rb, rd);
                break;
            case XO_STHX:
                if (ra == 0)
                    emit(buf, bufSize, &pos, "    mem_wr16(mem, r%u, (uint16_t)r%u);\n", rb, rd);
                else
                    emit(buf, bufSize, &pos, "    mem_wr16(mem, r%u + r%u, (uint16_t)r%u);\n", ra, rb, rd);
                break;

            // SPR access
            case XO_MFSPR: {
                uint32_t sn = di.spr;
                switch (sn) {
                case SPR_LR:  emit(buf, bufSize, &pos, "    r%u = lr;\n", rd); break;
                case SPR_CTR: emit(buf, bufSize, &pos, "    r%u = ctr;\n", rd); break;
                case SPR_XER: emit(buf, bufSize, &pos, "    r%u = xer;\n", rd); break;
                default:      emit(buf, bufSize, &pos, "    r%u = 0;\n", rd); break;
                }
                break;
            }
            case XO_MTSPR: {
                uint32_t sn = di.spr;
                switch (sn) {
                case SPR_LR:  emit(buf, bufSize, &pos, "    lr = r%u;\n", rd); break;
                case SPR_CTR: emit(buf, bufSize, &pos, "    ctr = r%u;\n", rd); break;
                case SPR_XER: emit(buf, bufSize, &pos, "    xer = r%u;\n", rd); break;
                default: break;
                }
                break;
            }
            case XO_MFCR:
                emit(buf, bufSize, &pos, "    r%u = (uint64_t)cr;\n", rd);
                break;
            case XO_MTCRF: {
                uint32_t crm = di.crm;
                emit(buf, bufSize, &pos, "    { uint32_t _v = (uint32_t)r%u;\n", rd);
                for (int i = 0; i < 8; i++) {
                    if (crm & (1 << (7 - i))) {
                        int shift = (7 - i) * 4;
                        emit(buf, bufSize, &pos,
                            "      cr = (cr & ~(0xFu << %d)) | (_v & (0xFu << %d));\n",
                            shift, shift);
                    }
                }
                emit(buf, bufSize, &pos, "    }\n");
                break;
            }

            // Sync (no-ops)
            case XO_SYNC:
            case XO_EIEIO:
                emit(buf, bufSize, &pos, "    // sync/eieio (no-op)\n");
                break;

            default:
                emit(buf, bufSize, &pos, "    // unhandled grp31 xo=%u → fallback\n", di.xo);
                emit(buf, bufSize, &pos, "    nextPC = 0x%llxULL; spr[4] = 2;\n",
                     (unsigned long long)di.pc);
                break;
            }
            break;
        }

        default:
            emit(buf, bufSize, &pos, "    // unhandled opcd=%u → fallback\n", di.opcd);
            emit(buf, bufSize, &pos, "    nextPC = 0x%llxULL; spr[4] = 2;\n",
                 (unsigned long long)di.pc);
            break;
        }
    }

    // Epilogue: write back promoted registers
    emit(buf, bufSize, &pos, "\n    // Epilogue: write back modified registers\n");
    for (int i = 0; i < 32; i++) {
        if (block->writesGPR[i]) {
            emit(buf, bufSize, &pos, "    gpr[%d] = r%d;\n", i, i);
        }
    }
    for (int i = 0; i < 32; i++) {
        if (block->writesFPR[i]) {
            emit(buf, bufSize, &pos, "    fpr[%d] = f%d;\n", i, i);
        }
    }
    if (block->writesLR)  emit(buf, bufSize, &pos, "    spr[0] = lr;\n");
    if (block->writesCTR) emit(buf, bufSize, &pos, "    spr[1] = ctr;\n");
    if (block->writesXER) emit(buf, bufSize, &pos, "    spr[2] = xer;\n");
    emit(buf, bufSize, &pos, "    spr[3] = nextPC;\n");
    if (block->writesCR)  emit(buf, bufSize, &pos, "    *cr_ptr = cr;\n");

    emit(buf, bufSize, &pos, "}\n");

    return (int)pos;
}

// ═══════════════════════════════════════════════════════════════
// NVRTC Compilation
// ═══════════════════════════════════════════════════════════════

int ppc_jit_init(PPCJITState* state) {
    memset(state, 0, sizeof(PPCJITState));
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[PPC-JIT] cuInit failed: %d\n", err);
        return 0;
    }
    state->ready = true;
    fprintf(stderr, "[PPC-JIT] PPE JIT compiler initialized (NVRTC backend)\n");
    return 1;
}

int ppc_jit_compile(PPCJITState* state, const PPCBasicBlock* block,
                    const char* source, PPCJITEntry* out) {
    if (!state->ready) return 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, source, "ppc_jit_block.cu", 0, NULL, NULL);
    if (res != NVRTC_SUCCESS) {
        fprintf(stderr, "[PPC-JIT] nvrtcCreateProgram failed: %s\n", nvrtcGetErrorString(res));
        return 0;
    }

    const char* opts[] = {
        "--gpu-architecture=sm_70",
        "--use_fast_math",
        "--std=c++17",
        "--extra-device-vectorization",
    };
    res = nvrtcCompileProgram(prog, 4, opts);

    if (res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = (char*)malloc(logSize);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "[PPC-JIT] Compile failed for block 0x%llx:\n%s\n",
                (unsigned long long)block->entryPC, log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return 0;
    }

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = (char*)malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUmodule cuMod;
    CUresult cuRes = cuModuleLoadDataEx(&cuMod, ptx, 0, NULL, NULL);
    free(ptx);

    if (cuRes != CUDA_SUCCESS) {
        fprintf(stderr, "[PPC-JIT] cuModuleLoadData failed: %d\n", cuRes);
        return 0;
    }

    char funcName[128];
    snprintf(funcName, sizeof(funcName), "ppc_jit_block_0x%llx",
             (unsigned long long)block->entryPC);

    CUfunction cuFunc;
    cuRes = cuModuleGetFunction(&cuFunc, cuMod, funcName);
    if (cuRes != CUDA_SUCCESS) {
        fprintf(stderr, "[PPC-JIT] cuModuleGetFunction('%s') failed: %d\n", funcName, cuRes);
        cuModuleUnload(cuMod);
        return 0;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float compileMs = 0;
    cudaEventElapsedTime(&compileMs, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    state->totalCompileTimeMs += compileMs;
    state->compileCount++;

    out->entryPC    = block->entryPC;
    out->memHash    = block->memHash;
    out->cuModule   = (void*)cuMod;
    out->cuFunction = (void*)cuFunc;
    out->numInsns   = block->numInsns;
    out->hitCount   = 0;
    out->valid      = true;

    fprintf(stderr, "[PPC-JIT] Compiled block 0x%llx: %u insns → CUDA kernel (%.1f ms)\n",
            (unsigned long long)block->entryPC, block->numInsns, compileMs);
    return 1;
}

// ═══════════════════════════════════════════════════════════════
// JIT Execution via Driver API
// ═══════════════════════════════════════════════════════════════

int ppc_jit_execute(const PPCJITEntry* entry, PPEState* h_state,
                    uint8_t* d_mem) {
    if (!entry || !entry->valid) return 0;

    // Allocate device-side arrays for GPR, FPR, SPR, CR
    uint64_t* d_gpr;
    double*   d_fpr;
    uint64_t* d_spr;  // [0]=LR, [1]=CTR, [2]=XER, [3]=PC, [4]=halted
    uint32_t* d_cr;

    cudaMalloc(&d_gpr, 32 * sizeof(uint64_t));
    cudaMalloc(&d_fpr, 32 * sizeof(double));
    cudaMalloc(&d_spr, 5 * sizeof(uint64_t));
    cudaMalloc(&d_cr,  sizeof(uint32_t));

    // Upload state
    cudaMemcpy(d_gpr, h_state->gpr, 32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fpr, h_state->fpr, 32 * sizeof(double), cudaMemcpyHostToDevice);

    uint64_t spr_host[5];
    spr_host[0] = h_state->lr;
    spr_host[1] = h_state->ctr;
    spr_host[2] = h_state->xer;
    spr_host[3] = h_state->pc;
    spr_host[4] = 0; // halted flag
    cudaMemcpy(d_spr, spr_host, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cr, &h_state->cr, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    void* args[] = { &d_gpr, &d_fpr, &d_spr, &d_cr, &d_mem };
    CUresult err = cuLaunchKernel(
        (CUfunction)entry->cuFunction,
        1, 1, 1,    // grid
        1, 1, 1,    // block
        0, 0,       // shared mem, stream
        args, NULL);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[PPC-JIT] cuLaunchKernel failed: %d\n", err);
        cudaFree(d_gpr); cudaFree(d_fpr); cudaFree(d_spr); cudaFree(d_cr);
        return 0;
    }

    cudaDeviceSynchronize();

    // Read back state
    cudaMemcpy(h_state->gpr, d_gpr, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state->fpr, d_fpr, 32 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(spr_host, d_spr, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_state->cr, d_cr, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    h_state->lr  = spr_host[0];
    h_state->ctr = spr_host[1];
    h_state->xer = spr_host[2];
    h_state->pc  = spr_host[3];
    if (spr_host[4] != 0) h_state->halted = (uint32_t)spr_host[4];

    cudaFree(d_gpr);
    cudaFree(d_fpr);
    cudaFree(d_spr);
    cudaFree(d_cr);

    return 1;
}

// ═══════════════════════════════════════════════════════════════
// Cache Management
// ═══════════════════════════════════════════════════════════════

static PPCJITEntry* cache_lookup(PPCJITState* state, uint64_t pc, uint64_t memHash) {
    for (uint32_t i = 0; i < state->numEntries; i++) {
        if (state->cache[i].valid &&
            state->cache[i].entryPC == pc &&
            state->cache[i].memHash == memHash) {
            state->cacheHits++;
            state->cache[i].hitCount++;
            return &state->cache[i];
        }
    }
    state->cacheMisses++;
    return nullptr;
}

static PPCJITEntry* cache_insert(PPCJITState* state) {
    if (state->numEntries < MAX_BLOCKS) {
        return &state->cache[state->numEntries++];
    }
    uint32_t minHits = UINT32_MAX;
    uint32_t minIdx = 0;
    for (uint32_t i = 0; i < state->numEntries; i++) {
        if (state->cache[i].hitCount < minHits) {
            minHits = state->cache[i].hitCount;
            minIdx = i;
        }
    }
    if (state->cache[minIdx].cuModule)
        cuModuleUnload((CUmodule)state->cache[minIdx].cuModule);
    state->cache[minIdx].valid = false;
    return &state->cache[minIdx];
}

// ═══════════════════════════════════════════════════════════════
// Run Loop
// ═══════════════════════════════════════════════════════════════

int ppc_jit_run(PPCJITState* state, PPEState* h_state,
                uint8_t* d_mem, uint32_t maxCycles,
                float* outMs, uint32_t* outCycles) {
    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);
    cudaEventRecord(tStart);

    // We need a host-side copy of PS3 memory for block discovery
    uint8_t* h_mem = (uint8_t*)malloc((size_t)PS3_SANDBOX_SIZE < (64*1024*1024) ?
                                       (size_t)PS3_SANDBOX_SIZE : 64*1024*1024);
    size_t copySize = (size_t)PS3_SANDBOX_SIZE < (64*1024*1024) ?
                      (size_t)PS3_SANDBOX_SIZE : 64*1024*1024;
    cudaMemcpy(h_mem, d_mem, copySize, cudaMemcpyDeviceToHost);

    // Pre-allocate device-side state arrays (reused across iterations)
    uint64_t* d_gpr;
    double*   d_fpr;
    uint64_t* d_spr;
    uint32_t* d_cr;
    cudaMalloc(&d_gpr, 32 * sizeof(uint64_t));
    cudaMalloc(&d_fpr, 32 * sizeof(double));
    cudaMalloc(&d_spr, 5 * sizeof(uint64_t));
    cudaMalloc(&d_cr,  sizeof(uint32_t));

    uint32_t cyclesRun = 0;
    uint64_t spr_host[5];

    while (cyclesRun < maxCycles && !h_state->halted) {
        uint64_t pc = h_state->pc;

        // Discover block
        PPCBasicBlock block;
        int numInsns = ppc_jit_discover_block(h_mem, pc, &block);
        if (numInsns <= 0) {
            state->interpreterFallbacks++;
            break;
        }

        // Cache lookup
        PPCJITEntry* cached = cache_lookup(state, pc, block.memHash);
        if (!cached) {
            // Emit source
            char* source = (char*)malloc(MAX_SOURCE_SIZE);
            int srcLen = ppc_jit_emit_source(&block, source, MAX_SOURCE_SIZE);
            if (srcLen <= 0) {
                free(source);
                state->interpreterFallbacks++;
                break;
            }

            PPCJITEntry* slot = cache_insert(state);
            int ok = ppc_jit_compile(state, &block, source, slot);
            free(source);

            if (!ok) {
                state->interpreterFallbacks++;
                break;
            }
            cached = slot;
        }

        // Upload state to device
        cudaMemcpy(d_gpr, h_state->gpr, 32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fpr, h_state->fpr, 32 * sizeof(double), cudaMemcpyHostToDevice);
        spr_host[0] = h_state->lr;
        spr_host[1] = h_state->ctr;
        spr_host[2] = h_state->xer;
        spr_host[3] = h_state->pc;
        spr_host[4] = 0;
        cudaMemcpy(d_spr, spr_host, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cr, &h_state->cr, sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Launch kernel
        void* args[] = { &d_gpr, &d_fpr, &d_spr, &d_cr, &d_mem };
        CUresult err = cuLaunchKernel(
            (CUfunction)cached->cuFunction,
            1, 1, 1, 1, 1, 1,
            0, 0, args, NULL);

        if (err != CUDA_SUCCESS) {
            fprintf(stderr, "[PPC-JIT] cuLaunchKernel failed: %d\n", err);
            break;
        }
        cudaDeviceSynchronize();

        // Read back state
        cudaMemcpy(h_state->gpr, d_gpr, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_state->fpr, d_fpr, 32 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(spr_host, d_spr, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_state->cr, d_cr, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        h_state->lr  = spr_host[0];
        h_state->ctr = spr_host[1];
        h_state->xer = spr_host[2];
        h_state->pc  = spr_host[3];
        if (spr_host[4] != 0) h_state->halted = (uint32_t)spr_host[4];

        cyclesRun += cached->numInsns;
    }

    cudaFree(d_gpr);
    cudaFree(d_fpr);
    cudaFree(d_spr);
    cudaFree(d_cr);
    free(h_mem);

    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);
    float ms = 0;
    cudaEventElapsedTime(&ms, tStart, tStop);
    cudaEventDestroy(tStart);
    cudaEventDestroy(tStop);

    if (outMs) *outMs = ms;
    if (outCycles) *outCycles = cyclesRun;
    return 1;
}

// ═══════════════════════════════════════════════════════════════
// Superblock JIT: single kernel with PC-dispatch loop
// Discovers all reachable blocks, fuses into one kernel.
// ═══════════════════════════════════════════════════════════════

static constexpr int MAX_SUPERBLOCKS = 64;

// Forward-declare: emit instruction body for one instruction into buffer
// Returns bytes written. Uses same switch logic as ppc_jit_emit_source.
static int emit_insn(char* buf, size_t bufSize, size_t* pos,
                     const PPCDecodedInsn& di, const char* indent);

int ppc_jit_run_fast(PPCJITState* state, ppc::PPEState* h_state,
                     uint8_t* d_mem, uint32_t maxCycles,
                     float* outMs, uint32_t* outCycles) {
    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);
    cudaEventRecord(tStart);

    // Copy PS3 memory for block discovery
    size_t copySize = 64 * 1024 * 1024; // 64MB is enough for code
    if (copySize > PS3_SANDBOX_SIZE) copySize = PS3_SANDBOX_SIZE;
    uint8_t* h_mem = (uint8_t*)malloc(copySize);
    cudaMemcpy(h_mem, d_mem, copySize, cudaMemcpyDeviceToHost);

    // Phase 1: Discover all reachable blocks (BFS from entryPC)
    PPCBasicBlock blocks[MAX_SUPERBLOCKS];
    uint64_t blockPCs[MAX_SUPERBLOCKS];
    int numBlocks = 0;

    // Work queue for block discovery
    uint64_t queue[MAX_SUPERBLOCKS * 2];
    int qHead = 0, qTail = 0;
    queue[qTail++] = h_state->pc;

    while (qHead < qTail && numBlocks < MAX_SUPERBLOCKS) {
        uint64_t pc = queue[qHead++];

        // Skip if already discovered
        bool found = false;
        for (int i = 0; i < numBlocks; i++) {
            if (blocks[i].entryPC == pc) { found = true; break; }
        }
        if (found) continue;
        if (pc >= copySize) continue;

        PPCBasicBlock& blk = blocks[numBlocks];
        int n = ppc_jit_discover_block(h_mem, pc, &blk);
        if (n <= 0) continue;

        blockPCs[numBlocks] = pc;
        numBlocks++;

        // Follow branch targets to discover more blocks
        const PPCDecodedInsn& last = blk.insns[blk.numInsns - 1];
        if (last.isBranch) {
            // Unconditional B: follow target
            if (last.opcd == OP_B) {
                int64_t disp = LI26(last.raw);
                uint64_t target = last.aa ? (uint64_t)disp : last.pc + (uint64_t)disp;
                if (target < copySize && qTail < MAX_SUPERBLOCKS * 2)
                    queue[qTail++] = target;
            }
            // Conditional BC: follow both target and fallthrough
            else if (last.opcd == OP_BC) {
                int64_t disp = BD16(last.raw);
                uint64_t target = last.aa ? (uint64_t)disp : last.pc + (uint64_t)disp;
                uint64_t fall = last.pc + 4;
                if (target < copySize && qTail < MAX_SUPERBLOCKS * 2)
                    queue[qTail++] = target;
                if (fall < copySize && qTail < MAX_SUPERBLOCKS * 2)
                    queue[qTail++] = fall;
            }
            // Fallthrough for conditional branches that don't take
            uint64_t fall = last.pc + 4;
            if (fall < copySize && qTail < MAX_SUPERBLOCKS * 2)
                queue[qTail++] = fall;
        }
    }

    if (numBlocks == 0) {
        free(h_mem);
        cudaEventDestroy(tStart);
        cudaEventDestroy(tStop);
        return 0;
    }

    // Phase 2: Merge register usage across all blocks
    bool allGPR[32] = {}, allFPR[32] = {};
    bool needCR = false, needLR = false, needCTR = false, needXER = false;
    uint32_t totalInsns = 0;
    for (int b = 0; b < numBlocks; b++) {
        for (int i = 0; i < 32; i++) {
            if (blocks[b].usesGPR[i] || blocks[b].writesGPR[i]) allGPR[i] = true;
            if (blocks[b].usesFPR[i] || blocks[b].writesFPR[i]) allFPR[i] = true;
        }
        if (blocks[b].usesCR || blocks[b].writesCR) needCR = true;
        if (blocks[b].usesLR || blocks[b].writesLR) needLR = true;
        if (blocks[b].usesCTR || blocks[b].writesCTR) needCTR = true;
        if (blocks[b].usesXER || blocks[b].writesXER) needXER = true;
        totalInsns += blocks[b].numInsns;
    }
    // CR helpers use XER.SO
    if (needCR) needXER = true;

    // Phase 3: Emit superblock source
    size_t srcSize = MAX_SOURCE_SIZE;
    char* src = (char*)malloc(srcSize);
    size_t pos = 0;

    // Header + helpers (same as single-block emit)
    emit(src, srcSize, &pos,
        "// PPE Superblock: %d blocks, %u instructions, maxCycles=%u\n"
        "typedef unsigned int uint32_t;\ntypedef int int32_t;\n"
        "typedef unsigned short uint16_t;\ntypedef short int16_t;\n"
        "typedef unsigned char uint8_t;\ntypedef signed char int8_t;\n"
        "typedef unsigned long long uint64_t;\ntypedef long long int64_t;\n\n",
        numBlocks, totalInsns, maxCycles);

    // Byte-swap, memory, CR, rotate helpers (identical to single-block)
    emit(src, srcSize, &pos,
        "__device__ __forceinline__ uint32_t bswap32(uint32_t x) { return __byte_perm(x, 0, 0x0123); }\n"
        "__device__ __forceinline__ uint16_t bswap16(uint16_t x) { return (uint16_t)__byte_perm((uint32_t)x, 0, 0x0001); }\n"
        "__device__ __forceinline__ uint64_t bswap64(uint64_t x) {\n"
        "    uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32);\n"
        "    return ((uint64_t)bswap32(lo) << 32) | (uint64_t)bswap32(hi);\n"
        "}\n");
    emit(src, srcSize, &pos,
        "static const uint64_t SANDBOX = 0x%llxULL;\n"
        "__device__ __forceinline__ uint32_t mem_rd32(const uint8_t* m, uint64_t a) { uint32_t r; memcpy(&r, m+(a&(SANDBOX-1)), 4); return bswap32(r); }\n"
        "__device__ __forceinline__ uint16_t mem_rd16(const uint8_t* m, uint64_t a) { uint16_t r; memcpy(&r, m+(a&(SANDBOX-1)), 2); return bswap16(r); }\n"
        "__device__ __forceinline__ uint8_t mem_rd8(const uint8_t* m, uint64_t a) { return m[a&(SANDBOX-1)]; }\n"
        "__device__ __forceinline__ void mem_wr32(uint8_t* m, uint64_t a, uint32_t v) { uint32_t s=bswap32(v); memcpy(m+(a&(SANDBOX-1)),&s,4); }\n"
        "__device__ __forceinline__ void mem_wr16(uint8_t* m, uint64_t a, uint16_t v) { uint16_t s=bswap16(v); memcpy(m+(a&(SANDBOX-1)),&s,2); }\n"
        "__device__ __forceinline__ void mem_wr8(uint8_t* m, uint64_t a, uint8_t v) { m[a&(SANDBOX-1)]=v; }\n",
        (unsigned long long)PS3_SANDBOX_SIZE);
    emit(src, srcSize, &pos,
        "__device__ __forceinline__ float mem_rdf32(const uint8_t* m, uint64_t a) { uint32_t b=mem_rd32(m,a); float f; memcpy(&f,&b,4); return f; }\n"
        "__device__ __forceinline__ double mem_rdf64(const uint8_t* m, uint64_t a) { uint64_t b=bswap64(*(const uint64_t*)(m+(a&(SANDBOX-1)))); double d; memcpy(&d,&b,8); return d; }\n"
        "__device__ __forceinline__ void mem_wrf32(uint8_t* m, uint64_t a, float f) { uint32_t b; memcpy(&b,&f,4); mem_wr32(m,a,b); }\n"
        "__device__ __forceinline__ void mem_wrf64(uint8_t* m, uint64_t a, double d) { uint64_t b; memcpy(&b,&d,8); uint64_t s=bswap64(b); memcpy(m+(a&(SANDBOX-1)),&s,8); }\n\n");
    emit(src, srcSize, &pos,
        "__device__ __forceinline__ void setCR(uint32_t& cr, int field, int64_t result, uint64_t xer) {\n"
        "    uint32_t val=0; if(result<0) val=0x8; else if(result>0) val=0x4; else val=0x2;\n"
        "    if(xer&(1ULL<<31)) val|=0x1; int shift=(7-field)*4; cr=(cr&~(0xFu<<shift))|(val<<shift);\n"
        "}\n"
        "__device__ __forceinline__ bool getCRBit(uint32_t cr, int bit) { return (cr>>(31-bit))&1; }\n"
        "__device__ __forceinline__ bool getCA(uint64_t xer) { return (xer>>29)&1; }\n"
        "__device__ __forceinline__ void setCA(uint64_t& xer, bool ca) { xer=ca?(xer|(1ULL<<29)):(xer&~(1ULL<<29)); }\n"
        "__device__ __forceinline__ uint32_t rotl32(uint32_t v, uint32_t n) { n&=31; return (v<<n)|(v>>(32-n)); }\n"
        "__device__ __forceinline__ uint32_t rotateMask32(uint32_t mb, uint32_t me) {\n"
        "    uint32_t mask=0;\n"
        "    if(mb<=me) { for(uint32_t i=mb;i<=me;i++) mask|=(1u<<(31-i)); }\n"
        "    else { for(uint32_t i=0;i<=me;i++) mask|=(1u<<(31-i)); for(uint32_t i=mb;i<=31;i++) mask|=(1u<<(31-i)); }\n"
        "    return mask;\n"
        "}\n\n");

    // Kernel signature: same interface as single-block
    emit(src, srcSize, &pos,
        "extern \"C\" __global__ void ppc_superblock(\n"
        "    uint64_t* __restrict__ gpr,\n"
        "    double*   __restrict__ fpr,\n"
        "    uint64_t* __restrict__ spr,\n"
        "    uint32_t* __restrict__ cr_ptr,\n"
        "    uint8_t*  __restrict__ mem)\n"
        "{\n"
        "    if (threadIdx.x != 0) return;\n\n");

    // Register promotion: load ALL used GPRs/FPRs/SPRs
    for (int i = 0; i < 32; i++)
        if (allGPR[i]) emit(src, srcSize, &pos, "    uint64_t r%d = gpr[%d];\n", i, i);
    for (int i = 0; i < 32; i++)
        if (allFPR[i]) emit(src, srcSize, &pos, "    double f%d = fpr[%d];\n", i, i);
    if (needLR)  emit(src, srcSize, &pos, "    uint64_t lr = spr[0];\n");
    if (needCTR) emit(src, srcSize, &pos, "    uint64_t ctr = spr[1];\n");
    if (needXER) emit(src, srcSize, &pos, "    uint64_t xer = spr[2];\n");
    if (needCR)  emit(src, srcSize, &pos, "    uint32_t cr = *cr_ptr;\n");
    emit(src, srcSize, &pos,
        "    uint64_t pc = spr[3];\n"
        "    uint32_t cycles = 0;\n"
        "    const uint32_t MAX_CYC = %uu;\n\n", maxCycles);

    // Dispatch loop
    // Phase 3b: Detect self-loops and tight 2-block loops
    // A self-loop: block's last insn is BC back to its own entryPC
    // A tight loop: block A's BC → block B, block B's B → block A
    bool isLoopBlock[MAX_SUPERBLOCKS] = {};
    uint64_t loopTarget[MAX_SUPERBLOCKS] = {};  // where the back-edge goes
    uint64_t loopFall[MAX_SUPERBLOCKS] = {};    // where the loop exits to

    for (int b = 0; b < numBlocks; b++) {
        const PPCBasicBlock& blk = blocks[b];
        const PPCDecodedInsn& last = blk.insns[blk.numInsns - 1];
        if (last.opcd == OP_BC) {
            int64_t disp = BD16(last.raw);
            uint64_t target = last.aa ? (uint64_t)disp : last.pc + (uint64_t)disp;
            // Self-loop: BC branches back to this block's entry
            if (target == blk.entryPC) {
                isLoopBlock[b] = true;
                loopTarget[b] = target;
                loopFall[b] = last.pc + 4;
            }
        }
    }

    emit(src, srcSize, &pos,
        "    while (cycles < MAX_CYC) {\n"
        "        uint64_t nextPC = pc + 4;\n"
        "        switch (pc) {\n");

    // Emit each block as a case
    for (int b = 0; b < numBlocks; b++) {
        const PPCBasicBlock& blk = blocks[b];
        emit(src, srcSize, &pos, "        case 0x%llxULL: {\n",
             (unsigned long long)blk.entryPC);

        if (isLoopBlock[b]) {
            // Emit as a tight do-while loop — no switch re-entry!
            emit(src, srcSize, &pos, "            // FUSED LOOP (self-loop detected)\n");
            emit(src, srcSize, &pos, "            do {\n");

            // Emit all instructions EXCEPT the last BC (we handle it as the while condition)
            for (uint32_t idx = 0; idx + 1 < blk.numInsns; idx++) {
                emit_insn(src, srcSize, &pos, blk.insns[idx], "                ");
            }
            emit(src, srcSize, &pos, "                cycles += %u;\n", blk.numInsns);

            // Emit the BC condition as the while() test
            const PPCDecodedInsn& bc = blk.insns[blk.numInsns - 1];
            uint32_t bo_val = bc.bo, bi_val = bc.bi;
            bool useCTR = !(bo_val & 0x04);
            bool useCond = !(bo_val & 0x10);

            // Build the loop condition expression
            if (useCTR) {
                emit(src, srcSize, &pos, "                ctr--;\n");
            }
            // Construct while condition
            emit(src, srcSize, &pos, "            } while (");
            bool needAnd = false;
            if (useCTR) {
                emit(src, srcSize, &pos, "(ctr %s 0)", (bo_val & 0x02) ? "==" : "!=");
                needAnd = true;
            }
            if (useCond) {
                if (needAnd) emit(src, srcSize, &pos, " && ");
                if (bo_val & 0x08)
                    emit(src, srcSize, &pos, "getCRBit(cr,%u)", bi_val);
                else
                    emit(src, srcSize, &pos, "!getCRBit(cr,%u)", bi_val);
                needAnd = true;
            }
            if (!useCTR && !useCond) {
                emit(src, srcSize, &pos, "true"); // unconditional loop (shouldn't happen for BC)
            }
            emit(src, srcSize, &pos, " && cycles < MAX_CYC);\n");
            // After loop exits: pc = fallthrough
            emit(src, srcSize, &pos, "            pc = 0x%llxULL;\n", (unsigned long long)loopFall[b]);
        } else {
            // Normal block: inline all instructions
            for (uint32_t idx = 0; idx < blk.numInsns; idx++) {
                emit_insn(src, srcSize, &pos, blk.insns[idx], "            ");
            }
            emit(src, srcSize, &pos, "            pc = nextPC;\n");
            emit(src, srcSize, &pos, "            cycles += %u;\n", blk.numInsns);
        }

        emit(src, srcSize, &pos, "            break;\n        }\n");
    }

    // Default: unknown PC → exit
    emit(src, srcSize, &pos,
        "        default:\n"
        "            goto done;\n"
        "        }\n"
        "    }\n"
        "done:\n");

    // Epilogue: write back all promoted registers
    emit(src, srcSize, &pos, "    // Write back\n");
    for (int i = 0; i < 32; i++)
        if (allGPR[i]) emit(src, srcSize, &pos, "    gpr[%d] = r%d;\n", i, i);
    for (int i = 0; i < 32; i++)
        if (allFPR[i]) emit(src, srcSize, &pos, "    fpr[%d] = f%d;\n", i, i);
    if (needLR)  emit(src, srcSize, &pos, "    spr[0] = lr;\n");
    if (needCTR) emit(src, srcSize, &pos, "    spr[1] = ctr;\n");
    if (needXER) emit(src, srcSize, &pos, "    spr[2] = xer;\n");
    emit(src, srcSize, &pos, "    spr[3] = pc;\n");
    if (needCR) emit(src, srcSize, &pos, "    *cr_ptr = cr;\n");
    emit(src, srcSize, &pos, "    spr[4] = (cycles >= MAX_CYC) ? 0 : 1;\n");
    emit(src, srcSize, &pos, "}\n");

    // Phase 4: Compile via NVRTC
    fprintf(stderr, "[PPC-JIT] Superblock: %d blocks, %u insns, %zu bytes source\n",
            numBlocks, totalInsns, pos);

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, src, "ppc_superblock.cu", 0, NULL, NULL);
    const char* opts[] = { "--gpu-architecture=sm_70", "-use_fast_math",
                           "--extra-device-vectorization", "-w" };
    nvrtcResult compRes = nvrtcCompileProgram(prog, 4, opts);

    if (compRes != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = (char*)malloc(logSize + 1);
        nvrtcGetProgramLog(prog, log);
        log[logSize] = 0;
        fprintf(stderr, "[PPC-JIT] Superblock compile FAILED:\n%s\n", log);
        free(log);
        nvrtcDestroyProgram(&prog);
        free(src);
        free(h_mem);
        cudaEventDestroy(tStart);
        cudaEventDestroy(tStop);
        return 0;
    }

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = (char*)malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);
    free(src);

    CUmodule cuMod;
    CUfunction cuFunc;
    cuModuleLoadData(&cuMod, ptx);
    cuModuleGetFunction(&cuFunc, cuMod, "ppc_superblock");
    free(ptx);

    fprintf(stderr, "[PPC-JIT] Superblock compiled OK\n");

    // Phase 5: Execute — single launch!
    cudaEvent_t execStart, execStop;
    cudaEventCreate(&execStart);
    cudaEventCreate(&execStop);

    uint64_t* d_gpr;  double* d_fpr;  uint64_t* d_spr;  uint32_t* d_cr;
    cudaMalloc(&d_gpr, 32 * sizeof(uint64_t));
    cudaMalloc(&d_fpr, 32 * sizeof(double));
    cudaMalloc(&d_spr, 5 * sizeof(uint64_t));
    cudaMalloc(&d_cr,  sizeof(uint32_t));

    cudaMemcpy(d_gpr, h_state->gpr, 32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fpr, h_state->fpr, 32 * sizeof(double), cudaMemcpyHostToDevice);
    uint64_t spr_host[5] = { h_state->lr, h_state->ctr, h_state->xer, h_state->pc, 0 };
    cudaMemcpy(d_spr, spr_host, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cr, &h_state->cr, sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaEventRecord(execStart);
    void* args[] = { &d_gpr, &d_fpr, &d_spr, &d_cr, &d_mem };
    CUresult err = cuLaunchKernel(cuFunc, 1,1,1, 1,1,1, 0,0, args, NULL);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[PPC-JIT] Superblock launch failed: %d\n", err);
    }
    cudaEventRecord(execStop);
    cudaEventSynchronize(execStop);
    float execMs = 0;
    cudaEventElapsedTime(&execMs, execStart, execStop);
    fprintf(stderr, "[PPC-JIT] Superblock exec: %.3f ms (kernel only)\n", execMs);
    cudaEventDestroy(execStart);
    cudaEventDestroy(execStop);

    // Read back
    cudaMemcpy(h_state->gpr, d_gpr, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state->fpr, d_fpr, 32 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(spr_host, d_spr, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_state->cr, d_cr, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    h_state->lr  = spr_host[0];
    h_state->ctr = spr_host[1];
    h_state->xer = spr_host[2];
    h_state->pc  = spr_host[3];
    if (spr_host[4] != 0) h_state->halted = (uint32_t)spr_host[4];

    uint32_t cyclesRun = maxCycles; // approximate
    // Read actual cycles from the kernel's final pc state
    // The kernel ran until maxCycles or unknown PC

    cudaFree(d_gpr); cudaFree(d_fpr); cudaFree(d_spr); cudaFree(d_cr);
    cuModuleUnload(cuMod);
    free(h_mem);

    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);
    float ms = 0;
    cudaEventElapsedTime(&ms, tStart, tStop);
    cudaEventDestroy(tStart);
    cudaEventDestroy(tStop);

    if (outMs) *outMs = ms;
    if (outCycles) *outCycles = cyclesRun;
    return 1;
}

// emit_insn: emit one instruction's body into buffer (shared by single-block and superblock)
static int emit_insn(char* buf, size_t bufSize, size_t* pos,
                     const PPCDecodedInsn& di, const char* I) {
    size_t start = *pos;
    uint32_t rd = di.rd, ra = di.ra, rb = di.rb;

    switch (di.opcd) {
    case OP_ADDI: {
        int64_t imm = di.imm;
        if (ra == 0) emit(buf, bufSize, pos, "%sr%u = (uint64_t)((int64_t)%lldLL);\n", I, rd, (long long)imm);
        else emit(buf, bufSize, pos, "%sr%u = (uint64_t)((int64_t)r%u + (int64_t)%lldLL);\n", I, rd, ra, (long long)imm);
        break;
    }
    case OP_ADDIS: {
        int64_t imm = (int64_t)di.imm << 16;
        if (ra == 0) emit(buf, bufSize, pos, "%sr%u = (uint64_t)((int64_t)%lldLL);\n", I, rd, (long long)imm);
        else emit(buf, bufSize, pos, "%sr%u = (uint64_t)((int64_t)r%u + (int64_t)%lldLL);\n", I, rd, ra, (long long)imm);
        break;
    }
    case OP_ADDIC: {
        int64_t imm = di.imm;
        emit(buf, bufSize, pos, "%s{ uint64_t _a=r%u; uint64_t _r=_a+(uint64_t)((int64_t)%lldLL); r%u=_r; setCA(xer,(uint32_t)_r<(uint32_t)_a); }\n", I, ra, (long long)imm, rd);
        break;
    }
    case OP_ADDIC_D: {
        int64_t imm = di.imm;
        emit(buf, bufSize, pos, "%s{ uint64_t _a=r%u; uint64_t _r=_a+(uint64_t)((int64_t)%lldLL); r%u=_r; setCA(xer,(uint32_t)_r<(uint32_t)_a); setCR(cr,0,(int32_t)(uint32_t)_r,xer); }\n", I, ra, (long long)imm, rd);
        break;
    }
    case OP_SUBFIC: {
        int64_t imm = di.imm;
        emit(buf, bufSize, pos, "%s{ uint64_t _r=(uint64_t)((int64_t)%lldLL)-r%u; r%u=_r; setCA(xer,(uint32_t)r%u<=(uint32_t)(uint64_t)((int64_t)%lldLL)); }\n", I, (long long)imm, ra, rd, ra, (long long)imm);
        break;
    }
    case OP_MULLI:
        emit(buf, bufSize, pos, "%sr%u = (uint64_t)((int64_t)r%u * (int64_t)%lldLL);\n", I, rd, ra, (long long)di.imm);
        break;
    case OP_CMPI: {
        uint32_t bf = rd >> 2;
        emit(buf, bufSize, pos, "%ssetCR(cr,%u,(int64_t)(int32_t)(uint32_t)r%u-(int64_t)%lldLL,xer);\n", I, bf, ra, (long long)di.imm);
        break;
    }
    case OP_CMPLI: {
        uint32_t bf = rd >> 2;
        emit(buf, bufSize, pos, "%s{ uint32_t _a=(uint32_t)r%u; int64_t _d=(_a>(uint32_t)%lluULL)?1:(_a<(uint32_t)%lluULL)?-1:0; setCR(cr,%u,_d,xer); }\n", I, ra, (unsigned long long)di.uimm, (unsigned long long)di.uimm, bf);
        break;
    }
    case OP_ORI:
        emit(buf, bufSize, pos, "%sr%u = r%u | %lluULL;\n", I, ra, rd, (unsigned long long)di.uimm);
        break;
    case OP_ORIS:
        emit(buf, bufSize, pos, "%sr%u = r%u | (%lluULL << 16);\n", I, ra, rd, (unsigned long long)di.uimm);
        break;
    case OP_XORI:
        emit(buf, bufSize, pos, "%sr%u = r%u ^ %lluULL;\n", I, ra, rd, (unsigned long long)di.uimm);
        break;
    case OP_XORIS:
        emit(buf, bufSize, pos, "%sr%u = r%u ^ (%lluULL << 16);\n", I, ra, rd, (unsigned long long)di.uimm);
        break;
    case OP_ANDI:
        emit(buf, bufSize, pos, "%sr%u = r%u & %lluULL; setCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra, rd, (unsigned long long)di.uimm, ra);
        break;
    case OP_ANDIS:
        emit(buf, bufSize, pos, "%sr%u = r%u & (%lluULL<<16); setCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra, rd, (unsigned long long)di.uimm, ra);
        break;
    case OP_RLWINM:
        emit(buf, bufSize, pos, "%s{ uint32_t _rot=rotl32((uint32_t)r%u,%uu); uint32_t _mask=rotateMask32(%uu,%uu); r%u=_rot&_mask; }\n", I, rd, di.sh, di.mb, di.me, ra);
        if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
        break;
    case OP_RLWIMI:
        emit(buf, bufSize, pos, "%s{ uint32_t _rot=rotl32((uint32_t)r%u,%uu); uint32_t _mask=rotateMask32(%uu,%uu); r%u=(_rot&_mask)|((uint32_t)r%u&~_mask); }\n", I, rd, di.sh, di.mb, di.me, ra, ra);
        if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
        break;
    case OP_RLWNM:
        emit(buf, bufSize, pos, "%s{ uint32_t _rot=rotl32((uint32_t)r%u,(uint32_t)r%u&0x1Fu); uint32_t _mask=rotateMask32(%uu,%uu); r%u=_rot&_mask; }\n", I, rd, rb, di.mb, di.me, ra);
        if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
        break;

    // Branches: set nextPC
    case OP_B: {
        int64_t disp = LI26(di.raw);
        if (di.lk) emit(buf, bufSize, pos, "%slr = 0x%llxULL;\n", I, (unsigned long long)(di.pc + 4));
        if (di.aa) emit(buf, bufSize, pos, "%snextPC = (uint64_t)((int64_t)%lldLL);\n", I, (long long)disp);
        else emit(buf, bufSize, pos, "%snextPC = 0x%llxULL;\n", I, (unsigned long long)(di.pc + (uint64_t)disp));
        break;
    }
    case OP_BC: {
        int64_t disp = BD16(di.raw);
        uint32_t bo_val = di.bo, bi_val = di.bi;
        emit(buf, bufSize, pos, "%s{\n", I);
        emit(buf, bufSize, pos, "%s    bool _ctr_ok=true, _cond_ok=true;\n", I);
        if (!(bo_val & 0x04)) {
            emit(buf, bufSize, pos, "%s    ctr--;\n", I);
            emit(buf, bufSize, pos, "%s    _ctr_ok = (ctr %s 0);\n", I, (bo_val & 0x02) ? "==" : "!=");
        }
        if (!(bo_val & 0x10)) {
            emit(buf, bufSize, pos, "%s    bool _bit = getCRBit(cr,%u);\n", I, bi_val);
            emit(buf, bufSize, pos, "%s    _cond_ok = %s_bit;\n", I, (bo_val & 0x08) ? "" : "!");
        }
        uint64_t target = di.aa ? (uint64_t)disp : di.pc + (uint64_t)disp;
        emit(buf, bufSize, pos, "%s    if(_ctr_ok && _cond_ok) nextPC = 0x%llxULL;\n", I, (unsigned long long)target);
        emit(buf, bufSize, pos, "%s    else nextPC = 0x%llxULL;\n", I, (unsigned long long)(di.pc + 4));
        if (di.lk) emit(buf, bufSize, pos, "%s    lr = 0x%llxULL;\n", I, (unsigned long long)(di.pc + 4));
        emit(buf, bufSize, pos, "%s}\n", I);
        break;
    }
    case OP_SC:
        emit(buf, bufSize, pos, "%snextPC = 0x%llxULL; goto done;\n", I, (unsigned long long)(di.pc + 4));
        break;

    case OP_GRP19: {
        if (di.xo == XO_BCLR) {
            uint32_t bo_val = di.bo, bi_val = di.bi;
            emit(buf, bufSize, pos, "%s{\n", I);
            emit(buf, bufSize, pos, "%s    bool _ctr_ok=true, _cond_ok=true;\n", I);
            if (!(bo_val & 0x04)) {
                emit(buf, bufSize, pos, "%s    ctr--;\n", I);
                emit(buf, bufSize, pos, "%s    _ctr_ok = (ctr %s 0);\n", I, (bo_val & 0x02) ? "==" : "!=");
            }
            if (!(bo_val & 0x10)) {
                emit(buf, bufSize, pos, "%s    bool _bit = getCRBit(cr,%u);\n", I, bi_val);
                emit(buf, bufSize, pos, "%s    _cond_ok = %s_bit;\n", I, (bo_val & 0x08) ? "" : "!");
            }
            emit(buf, bufSize, pos, "%s    if(_ctr_ok && _cond_ok) nextPC = lr & ~3ULL;\n", I);
            emit(buf, bufSize, pos, "%s    else nextPC = 0x%llxULL;\n", I, (unsigned long long)(di.pc + 4));
            if (di.lk) emit(buf, bufSize, pos, "%s    lr = 0x%llxULL;\n", I, (unsigned long long)(di.pc + 4));
            emit(buf, bufSize, pos, "%s}\n", I);
        } else if (di.xo == XO_BCCTR) {
            uint32_t bo_val = di.bo, bi_val = di.bi;
            emit(buf, bufSize, pos, "%s{\n", I);
            emit(buf, bufSize, pos, "%s    bool _cond_ok=true;\n", I);
            if (!(bo_val & 0x10)) {
                emit(buf, bufSize, pos, "%s    bool _bit = getCRBit(cr,%u);\n", I, bi_val);
                emit(buf, bufSize, pos, "%s    _cond_ok = %s_bit;\n", I, (bo_val & 0x08) ? "" : "!");
            }
            emit(buf, bufSize, pos, "%s    if(_cond_ok) nextPC = ctr & ~3ULL;\n", I);
            emit(buf, bufSize, pos, "%s    else nextPC = 0x%llxULL;\n", I, (unsigned long long)(di.pc + 4));
            if (di.lk) emit(buf, bufSize, pos, "%s    lr = 0x%llxULL;\n", I, (unsigned long long)(di.pc + 4));
            emit(buf, bufSize, pos, "%s}\n", I);
        } else {
            emit(buf, bufSize, pos, "%s// unhandled grp19 xo=%u → exit\n%sgoto done;\n", I, di.xo, I);
        }
        break;
    }

    // Load/Store D-form
    case OP_LWZ:
        if (ra == 0) emit(buf, bufSize, pos, "%sr%u = (uint64_t)mem_rd32(mem,(uint64_t)((int64_t)%lldLL));\n", I, rd, (long long)di.imm);
        else emit(buf, bufSize, pos, "%sr%u = (uint64_t)mem_rd32(mem,r%u+(uint64_t)((int64_t)%lldLL));\n", I, rd, ra, (long long)di.imm);
        break;
    case OP_LWZU:
        emit(buf, bufSize, pos, "%s{ uint64_t _ea=r%u+(uint64_t)((int64_t)%lldLL); r%u=(uint64_t)mem_rd32(mem,_ea); r%u=_ea; }\n", I, ra, (long long)di.imm, rd, ra);
        break;
    case OP_LBZ:
        if (ra == 0) emit(buf, bufSize, pos, "%sr%u = (uint64_t)mem_rd8(mem,(uint64_t)((int64_t)%lldLL));\n", I, rd, (long long)di.imm);
        else emit(buf, bufSize, pos, "%sr%u = (uint64_t)mem_rd8(mem,r%u+(uint64_t)((int64_t)%lldLL));\n", I, rd, ra, (long long)di.imm);
        break;
    case OP_LBZU:
        emit(buf, bufSize, pos, "%s{ uint64_t _ea=r%u+(uint64_t)((int64_t)%lldLL); r%u=(uint64_t)mem_rd8(mem,_ea); r%u=_ea; }\n", I, ra, (long long)di.imm, rd, ra);
        break;
    case OP_LHZ:
        if (ra == 0) emit(buf, bufSize, pos, "%sr%u = (uint64_t)mem_rd16(mem,(uint64_t)((int64_t)%lldLL));\n", I, rd, (long long)di.imm);
        else emit(buf, bufSize, pos, "%sr%u = (uint64_t)mem_rd16(mem,r%u+(uint64_t)((int64_t)%lldLL));\n", I, rd, ra, (long long)di.imm);
        break;
    case OP_LHZU:
        emit(buf, bufSize, pos, "%s{ uint64_t _ea=r%u+(uint64_t)((int64_t)%lldLL); r%u=(uint64_t)mem_rd16(mem,_ea); r%u=_ea; }\n", I, ra, (long long)di.imm, rd, ra);
        break;
    case OP_LHA:
        if (ra == 0) emit(buf, bufSize, pos, "%sr%u = (uint64_t)(int64_t)(int16_t)mem_rd16(mem,(uint64_t)((int64_t)%lldLL));\n", I, rd, (long long)di.imm);
        else emit(buf, bufSize, pos, "%sr%u = (uint64_t)(int64_t)(int16_t)mem_rd16(mem,r%u+(uint64_t)((int64_t)%lldLL));\n", I, rd, ra, (long long)di.imm);
        break;
    case OP_STW:
        if (ra == 0) emit(buf, bufSize, pos, "%smem_wr32(mem,(uint64_t)((int64_t)%lldLL),(uint32_t)r%u);\n", I, (long long)di.imm, rd);
        else emit(buf, bufSize, pos, "%smem_wr32(mem,r%u+(uint64_t)((int64_t)%lldLL),(uint32_t)r%u);\n", I, ra, (long long)di.imm, rd);
        break;
    case OP_STWU:
        emit(buf, bufSize, pos, "%s{ uint64_t _ea=r%u+(uint64_t)((int64_t)%lldLL); mem_wr32(mem,_ea,(uint32_t)r%u); r%u=_ea; }\n", I, ra, (long long)di.imm, rd, ra);
        break;
    case OP_STB:
        if (ra == 0) emit(buf, bufSize, pos, "%smem_wr8(mem,(uint64_t)((int64_t)%lldLL),(uint8_t)r%u);\n", I, (long long)di.imm, rd);
        else emit(buf, bufSize, pos, "%smem_wr8(mem,r%u+(uint64_t)((int64_t)%lldLL),(uint8_t)r%u);\n", I, ra, (long long)di.imm, rd);
        break;
    case OP_STBU:
        emit(buf, bufSize, pos, "%s{ uint64_t _ea=r%u+(uint64_t)((int64_t)%lldLL); mem_wr8(mem,_ea,(uint8_t)r%u); r%u=_ea; }\n", I, ra, (long long)di.imm, rd, ra);
        break;
    case OP_STH:
        if (ra == 0) emit(buf, bufSize, pos, "%smem_wr16(mem,(uint64_t)((int64_t)%lldLL),(uint16_t)r%u);\n", I, (long long)di.imm, rd);
        else emit(buf, bufSize, pos, "%smem_wr16(mem,r%u+(uint64_t)((int64_t)%lldLL),(uint16_t)r%u);\n", I, ra, (long long)di.imm, rd);
        break;
    case OP_STHU:
        emit(buf, bufSize, pos, "%s{ uint64_t _ea=r%u+(uint64_t)((int64_t)%lldLL); mem_wr16(mem,_ea,(uint16_t)r%u); r%u=_ea; }\n", I, ra, (long long)di.imm, rd, ra);
        break;

    // FP Load/Store
    case OP_LFS:
        if (ra == 0) emit(buf, bufSize, pos, "%sf%u = (double)mem_rdf32(mem,(uint64_t)((int64_t)%lldLL));\n", I, rd, (long long)di.imm);
        else emit(buf, bufSize, pos, "%sf%u = (double)mem_rdf32(mem,r%u+(uint64_t)((int64_t)%lldLL));\n", I, rd, ra, (long long)di.imm);
        break;
    case OP_LFD:
        if (ra == 0) emit(buf, bufSize, pos, "%sf%u = mem_rdf64(mem,(uint64_t)((int64_t)%lldLL));\n", I, rd, (long long)di.imm);
        else emit(buf, bufSize, pos, "%sf%u = mem_rdf64(mem,r%u+(uint64_t)((int64_t)%lldLL));\n", I, rd, ra, (long long)di.imm);
        break;
    case OP_STFS:
        if (ra == 0) emit(buf, bufSize, pos, "%smem_wrf32(mem,(uint64_t)((int64_t)%lldLL),(float)f%u);\n", I, (long long)di.imm, rd);
        else emit(buf, bufSize, pos, "%smem_wrf32(mem,r%u+(uint64_t)((int64_t)%lldLL),(float)f%u);\n", I, ra, (long long)di.imm, rd);
        break;
    case OP_STFD:
        if (ra == 0) emit(buf, bufSize, pos, "%smem_wrf64(mem,(uint64_t)((int64_t)%lldLL),f%u);\n", I, (long long)di.imm, rd);
        else emit(buf, bufSize, pos, "%smem_wrf64(mem,r%u+(uint64_t)((int64_t)%lldLL),f%u);\n", I, ra, (long long)di.imm, rd);
        break;

    // Load/Store Multiple
    case OP_LMW: {
        if (ra == 0) emit(buf, bufSize, pos, "%s{ uint64_t _ea=(uint64_t)((int64_t)%lldLL);\n", I, (long long)di.imm);
        else emit(buf, bufSize, pos, "%s{ uint64_t _ea=r%u+(uint64_t)((int64_t)%lldLL);\n", I, ra, (long long)di.imm);
        for (uint32_t r = rd; r < 32; r++)
            emit(buf, bufSize, pos, "%s  r%u=(uint64_t)mem_rd32(mem,_ea+%u);\n", I, r, (r - rd) * 4);
        emit(buf, bufSize, pos, "%s}\n", I);
        break;
    }
    case OP_STMW: {
        if (ra == 0) emit(buf, bufSize, pos, "%s{ uint64_t _ea=(uint64_t)((int64_t)%lldLL);\n", I, (long long)di.imm);
        else emit(buf, bufSize, pos, "%s{ uint64_t _ea=r%u+(uint64_t)((int64_t)%lldLL);\n", I, ra, (long long)di.imm);
        for (uint32_t r = rd; r < 32; r++)
            emit(buf, bufSize, pos, "%s  mem_wr32(mem,_ea+%u,(uint32_t)r%u);\n", I, (r - rd) * 4, r);
        emit(buf, bufSize, pos, "%s}\n", I);
        break;
    }

    // Group 31 — ALU/logical/shift/compare/SPR/indexed
    case OP_GRP31: {
        switch (di.xo) {
        case XO_ADD:
            emit(buf, bufSize, pos, "%sr%u = (uint32_t)((uint32_t)r%u + (uint32_t)r%u);\n", I, rd, ra, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_SUBF:
            emit(buf, bufSize, pos, "%sr%u = (uint32_t)((uint32_t)r%u - (uint32_t)r%u);\n", I, rd, rb, ra);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_SUBFC:
            emit(buf, bufSize, pos, "%s{ uint32_t _a=(uint32_t)r%u,_b=(uint32_t)r%u; uint64_t _r=(uint64_t)_b+(uint64_t)(~_a)+1ULL; r%u=(uint32_t)_r; setCA(xer,_r>>32); }\n", I, ra, rb, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_ADDC:
            emit(buf, bufSize, pos, "%s{ uint32_t _a=(uint32_t)r%u,_b=(uint32_t)r%u; uint64_t _r=(uint64_t)_a+(uint64_t)_b; r%u=(uint32_t)_r; setCA(xer,_r>>32); }\n", I, ra, rb, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_ADDE:
            emit(buf, bufSize, pos, "%s{ uint32_t _a=(uint32_t)r%u,_b=(uint32_t)r%u; uint64_t _r=(uint64_t)_a+(uint64_t)_b+(getCA(xer)?1ULL:0ULL); r%u=(uint32_t)_r; setCA(xer,_r>>32); }\n", I, ra, rb, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_ADDZE:
            emit(buf, bufSize, pos, "%s{ uint32_t _a=(uint32_t)r%u; uint64_t _r=(uint64_t)_a+(getCA(xer)?1ULL:0ULL); r%u=(uint32_t)_r; setCA(xer,_r>>32); }\n", I, ra, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_NEG:
            emit(buf, bufSize, pos, "%sr%u = (uint32_t)(-(int32_t)(uint32_t)r%u);\n", I, rd, ra);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_MULLW:
            emit(buf, bufSize, pos, "%sr%u = (uint32_t)(uint64_t)((int64_t)(int32_t)(uint32_t)r%u * (int64_t)(int32_t)(uint32_t)r%u);\n", I, rd, ra, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_MULHW:
            emit(buf, bufSize, pos, "%sr%u = (uint32_t)((int64_t)(int32_t)(uint32_t)r%u * (int64_t)(int32_t)(uint32_t)r%u >> 32);\n", I, rd, ra, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_MULHWU:
            emit(buf, bufSize, pos, "%sr%u = (uint32_t)((uint64_t)(uint32_t)r%u * (uint64_t)(uint32_t)r%u >> 32);\n", I, rd, ra, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_DIVW:
            emit(buf, bufSize, pos, "%s{ int32_t _a=(int32_t)(uint32_t)r%u,_b=(int32_t)(uint32_t)r%u; r%u=(_b!=0&&!(_a==(int32_t)0x80000000&&_b==-1))?(uint32_t)(_a/_b):0; }\n", I, ra, rb, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        case XO_DIVWU:
            emit(buf, bufSize, pos, "%s{ uint32_t _a=(uint32_t)r%u,_b=(uint32_t)r%u; r%u=(_b!=0)?(_a/_b):0; }\n", I, ra, rb, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, rd);
            break;
        // Logical (note: rS=rd field, result→ra)
        case XO_AND:
            emit(buf, bufSize, pos, "%sr%u = r%u & r%u;\n", I, ra, rd, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_OR:
            emit(buf, bufSize, pos, "%sr%u = r%u | r%u;\n", I, ra, rd, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_XOR:
            emit(buf, bufSize, pos, "%sr%u = r%u ^ r%u;\n", I, ra, rd, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_NOR:
            emit(buf, bufSize, pos, "%sr%u = ~(r%u | r%u);\n", I, ra, rd, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_ANDC:
            emit(buf, bufSize, pos, "%sr%u = r%u & ~r%u;\n", I, ra, rd, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_ORC:
            emit(buf, bufSize, pos, "%sr%u = r%u | ~r%u;\n", I, ra, rd, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_NAND:
            emit(buf, bufSize, pos, "%sr%u = ~(r%u & r%u);\n", I, ra, rd, rb);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_EXTSB:
            emit(buf, bufSize, pos, "%sr%u = (uint64_t)(int64_t)(int8_t)(uint8_t)r%u;\n", I, ra, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_EXTSH:
            emit(buf, bufSize, pos, "%sr%u = (uint64_t)(int64_t)(int16_t)(uint16_t)r%u;\n", I, ra, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_CNTLZW:
            emit(buf, bufSize, pos, "%s{ uint32_t _v=(uint32_t)r%u; r%u=_v?__clz(_v):32; }\n", I, rd, ra);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_SLW:
            emit(buf, bufSize, pos, "%s{ uint32_t _sh=(uint32_t)r%u&0x3F; r%u=(_sh<32)?((uint32_t)r%u<<_sh):0; }\n", I, rb, ra, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_SRW:
            emit(buf, bufSize, pos, "%s{ uint32_t _sh=(uint32_t)r%u&0x3F; r%u=(_sh<32)?((uint32_t)r%u>>_sh):0; }\n", I, rb, ra, rd);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_SRAW:
            emit(buf, bufSize, pos,
                "%s{ uint32_t _sh=(uint32_t)r%u&0x3F; int32_t _v=(int32_t)(uint32_t)r%u;\n"
                "%s  if(_sh==0){r%u=(uint32_t)_v;setCA(xer,false);}\n"
                "%s  else if(_sh<32){bool _c=(_v<0)&&((_v&((1<<_sh)-1))!=0);r%u=(uint32_t)(_v>>_sh);setCA(xer,_c);}\n"
                "%s  else{r%u=(_v<0)?0xFFFFFFFF:0;setCA(xer,_v<0);} }\n",
                I, rb, rd, I, ra, I, ra, I, ra);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_SRAWI:
            emit(buf, bufSize, pos,
                "%s{ int32_t _v=(int32_t)(uint32_t)r%u; uint32_t _sh=%u;\n"
                "%s  if(_sh==0){r%u=(uint32_t)_v;setCA(xer,false);}\n"
                "%s  else{bool _c=(_v<0)&&((_v&((1<<_sh)-1))!=0);r%u=(uint32_t)(_v>>_sh);setCA(xer,_c);} }\n",
                I, rd, di.sh, I, ra, I, ra);
            if (di.rc) emit(buf, bufSize, pos, "%ssetCR(cr,0,(int32_t)(uint32_t)r%u,xer);\n", I, ra);
            break;
        case XO_CMP: {
            uint32_t bf = rd >> 2;
            emit(buf, bufSize, pos, "%ssetCR(cr,%u,(int64_t)(int32_t)(uint32_t)r%u-(int64_t)(int32_t)(uint32_t)r%u,xer);\n", I, bf, ra, rb);
            break;
        }
        case XO_CMPL: {
            uint32_t bf = rd >> 2;
            emit(buf, bufSize, pos, "%s{ uint32_t _a=(uint32_t)r%u,_b=(uint32_t)r%u; int64_t _d=(_a>_b)?1:(_a<_b)?-1:0; setCR(cr,%u,_d,xer); }\n", I, ra, rb, bf);
            break;
        }
        // Indexed load/store
        case XO_LWZX:
            emit(buf, bufSize, pos, "%sr%u = (uint64_t)mem_rd32(mem,%sr%u + r%u);\n", I, rd, (ra==0)?"(uint64_t)0 + ":"", (ra==0)?rb:ra, rb);
            if (ra == 0) {} // handled by ternary above
            break;
        case XO_LBZX:
            emit(buf, bufSize, pos, "%sr%u = (uint64_t)mem_rd8(mem,%sr%u + r%u);\n", I, rd, (ra==0)?"(uint64_t)0 + ":"", (ra==0)?rb:ra, rb);
            break;
        case XO_LHZX:
            emit(buf, bufSize, pos, "%sr%u = (uint64_t)mem_rd16(mem,%sr%u + r%u);\n", I, rd, (ra==0)?"(uint64_t)0 + ":"", (ra==0)?rb:ra, rb);
            break;
        case XO_LHAX:
            emit(buf, bufSize, pos, "%sr%u = (uint64_t)(int64_t)(int16_t)mem_rd16(mem,%sr%u + r%u);\n", I, rd, (ra==0)?"(uint64_t)0 + ":"", (ra==0)?rb:ra, rb);
            break;
        case XO_STWX:
            emit(buf, bufSize, pos, "%smem_wr32(mem,%sr%u + r%u,(uint32_t)r%u);\n", I, (ra==0)?"(uint64_t)0 + ":"", (ra==0)?rb:ra, rb, rd);
            break;
        case XO_STBX:
            emit(buf, bufSize, pos, "%smem_wr8(mem,%sr%u + r%u,(uint8_t)r%u);\n", I, (ra==0)?"(uint64_t)0 + ":"", (ra==0)?rb:ra, rb, rd);
            break;
        case XO_STHX:
            emit(buf, bufSize, pos, "%smem_wr16(mem,%sr%u + r%u,(uint16_t)r%u);\n", I, (ra==0)?"(uint64_t)0 + ":"", (ra==0)?rb:ra, rb, rd);
            break;
        // SPR access
        case XO_MFSPR: {
            uint32_t sn = di.spr;
            if (sn == SPR_LR) emit(buf, bufSize, pos, "%sr%u = lr;\n", I, rd);
            else if (sn == SPR_CTR) emit(buf, bufSize, pos, "%sr%u = ctr;\n", I, rd);
            else if (sn == SPR_XER) emit(buf, bufSize, pos, "%sr%u = xer;\n", I, rd);
            else if (sn == SPR_PVR) emit(buf, bufSize, pos, "%sr%u = 0x00700000ULL;\n", I, rd);
            else emit(buf, bufSize, pos, "%sr%u = 0;\n", I, rd);
            break;
        }
        case XO_MTSPR: {
            uint32_t sn = di.spr;
            if (sn == SPR_LR) emit(buf, bufSize, pos, "%slr = r%u;\n", I, rd);
            else if (sn == SPR_CTR) emit(buf, bufSize, pos, "%sctr = r%u;\n", I, rd);
            else if (sn == SPR_XER) emit(buf, bufSize, pos, "%sxer = r%u;\n", I, rd);
            break;
        }
        case XO_MFCR:
            emit(buf, bufSize, pos, "%sr%u = (uint64_t)cr;\n", I, rd);
            break;
        case XO_MTCRF: {
            uint32_t crm = di.crm;
            emit(buf, bufSize, pos, "%s{ uint32_t _v=(uint32_t)r%u;\n", I, rd);
            for (int i = 0; i < 8; i++) {
                if (crm & (1 << (7 - i))) {
                    int shift = (7 - i) * 4;
                    emit(buf, bufSize, pos, "%s  cr=(cr&~(0xF<<%d))|(_v&(0xF<<%d));\n", I, shift, shift);
                }
            }
            emit(buf, bufSize, pos, "%s}\n", I);
            break;
        }
        case XO_SYNC: case XO_EIEIO:
            emit(buf, bufSize, pos, "%s__threadfence();\n", I);
            break;
        default:
            emit(buf, bufSize, pos, "%s// unhandled grp31 xo=%u → exit\n%sgoto done;\n", I, di.xo, I);
            break;
        }
        break;
    }

    default:
        emit(buf, bufSize, pos, "%s// unhandled opcd=%u → exit\n%sgoto done;\n", I, di.opcd, I);
        break;
    }

    return (int)(*pos - start);
}

void ppc_jit_print_stats(const PPCJITState* state) {
    fprintf(stderr,
        "╔═══════════════════════════════════════╗\n"
        "║  PPE JIT Statistics                    ║\n"
        "╠═══════════════════════════════════════╣\n"
        "║  Compiled blocks: %-6u               ║\n"
        "║  Cache hits:      %-6u               ║\n"
        "║  Cache misses:    %-6u               ║\n"
        "║  Interp fallback: %-6u               ║\n"
        "║  Compile time:    %8.1f ms          ║\n"
        "╚═══════════════════════════════════════╝\n",
        state->compileCount,
        state->cacheHits,
        state->cacheMisses,
        state->interpreterFallbacks,
        state->totalCompileTimeMs);
}

void ppc_jit_shutdown(PPCJITState* state) {
    for (uint32_t i = 0; i < state->numEntries; i++) {
        if (state->cache[i].valid && state->cache[i].cuModule)
            cuModuleUnload((CUmodule)state->cache[i].cuModule);
    }
    memset(state, 0, sizeof(PPCJITState));
    fprintf(stderr, "[PPC-JIT] Shutdown\n");
}

} // namespace ppc_jit
