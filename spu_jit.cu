// spu_jit.cu — SPU Basic-Block JIT Compiler
//
// Translates SPU instruction sequences into native CUDA kernels at runtime
// using NVRTC (NVIDIA Runtime Compilation). Each compiled block executes
// SPU instructions at native GPU speed — no fetch, no decode, no switch.
//
// Pipeline: LS bytes → decode → CUDA C++ source → NVRTC → cubin → cuFunction
//
#include "spu_jit.h"
#include "spu_defs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>

using namespace spu;

// Implementation is in spu_jit namespace to match header declarations.
namespace spu_jit {

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

static uint32_t bswap32_host(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

// FNV-1a hash of LS region
static uint64_t hash_ls_region(const uint8_t* data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Fetch instruction from host-side LS copy (big-endian → native)
static uint32_t fetch_inst_host(const uint8_t* ls, uint32_t pc) {
    pc &= (SPU_LS_SIZE - 1) & ~0x3;
    uint32_t raw;
    memcpy(&raw, ls + pc, 4);
    return bswap32_host(raw);
}

// ═══════════════════════════════════════════════════════════════
// Basic Block Discovery
// ═══════════════════════════════════════════════════════════════

// Decode one instruction into DecodedInsn
static void decode_insn(uint32_t raw, uint32_t pc, DecodedInsn* out) {
    memset(out, 0, sizeof(DecodedInsn));
    out->raw = raw;
    out->pc = pc;

    // Try widest opcode first (same priority as interpreter)
    uint32_t o11 = spu_op11(raw);

    // Check RR (11-bit) first
    switch (o11) {
    case op11::A: case op11::AH: case op11::SF: case op11::SFH:
    case op11::CG: case op11::BG: case op11::ADDX: case op11::SFX:
    case op11::MPY: case op11::MPYU: case op11::MPYH: case op11::MPYS:
    case op11::CLZ: case op11::CNTB: case op11::AVGB: case op11::ABSDB:
    case op11::AND: case op11::OR: case op11::XOR: case op11::ANDC:
    case op11::ORC: case op11::NAND: case op11::NOR: case op11::EQV:
    case op11::SHL: case op11::SHLH: case op11::ROT: case op11::ROTH:
    case op11::ROTM: case op11::ROTHM: case op11::ROTMA: case op11::ROTMAH:
    case op11::ROTQBY: case op11::ROTQBI: case op11::SHLQBY: case op11::SHLQBI:
    case op11::ROTQMBY: case op11::ROTQMBI:
    case op11::CEQ: case op11::CEQH: case op11::CEQB:
    case op11::CGT: case op11::CGTH: case op11::CGTB:
    case op11::CLGT: case op11::CLGTH: case op11::CLGTB:
    case op11::FA: case op11::FS: case op11::FM:
    case op11::FCEQ: case op11::FCGT: case op11::FCMEQ: case op11::FCMGT:
    case op11::DFA: case op11::DFS: case op11::DFM:
    case op11::RDCH: case op11::RCHCNT: case op11::WRCH:
    case op11::NOP: case op11::LNOP: case op11::SYNC: case op11::DSYNC:
        out->format = 0;
        out->opcode = (uint16_t)o11;
        out->rT = spu_rT_rr(raw);
        out->rA = spu_rA_rr(raw);
        out->rB = spu_rB_rr(raw);
        return;
    case op11::BI:
        out->format = 0; out->opcode = (uint16_t)o11;
        out->rT = spu_rT_rr(raw); out->rA = spu_rA_rr(raw);
        out->isBranch = 1;
        return;
    case op11::BISL:
        out->format = 0; out->opcode = (uint16_t)o11;
        out->rT = spu_rT_rr(raw); out->rA = spu_rA_rr(raw);
        out->isBranch = 1; out->isCall = 1;
        return;
    case op11::BIZ: case op11::BINZ: case op11::BIHZ: case op11::BIHNZ:
        out->format = 0; out->opcode = (uint16_t)o11;
        out->rT = spu_rT_rr(raw); out->rA = spu_rA_rr(raw);
        out->isBranch = 1;
        return;
    case op11::STOP: case op11::STOPD:
        out->format = 0; out->opcode = (uint16_t)o11;
        out->isStop = 1;
        return;
    default:
        break;
    }

    // RI16 (9-bit)
    uint32_t o9 = spu_op9(raw);
    switch (o9) {
    case op9::IL: case op9::ILH: case op9::ILHU: case op9::IOHL: case op9::FSMBI:
        out->format = 1; out->opcode = (uint16_t)o9;
        out->rT = spu_rT_ri16(raw);
        out->imm = spu_I16(raw);
        out->immU = spu_I16u(raw);
        return;
    case op9::BR: case op9::BRA:
        out->format = 1; out->opcode = (uint16_t)o9;
        out->rT = spu_rT_ri16(raw);
        out->imm = spu_I16(raw);
        out->isBranch = 1;
        return;
    case op9::BRSL: case op9::BRASL:
        out->format = 1; out->opcode = (uint16_t)o9;
        out->rT = spu_rT_ri16(raw);
        out->imm = spu_I16(raw);
        out->isBranch = 1; out->isCall = 1;
        return;
    case op9::BRZ: case op9::BRNZ: case op9::BRHZ: case op9::BRHNZ:
        out->format = 1; out->opcode = (uint16_t)o9;
        out->rT = spu_rT_ri16(raw);
        out->imm = spu_I16(raw);
        out->isBranch = 1;
        return;
    case op9::LQA: case op9::STQA: case op9::LQR: case op9::STQR:
        out->format = 1; out->opcode = (uint16_t)o9;
        out->rT = spu_rT_ri16(raw);
        out->imm = spu_I16(raw);
        out->immU = spu_I16u(raw);
        return;
    default:
        break;
    }

    // RI10 (8-bit)
    uint32_t o8 = spu_op8(raw);
    switch (o8) {
    case op8::AI: case op8::AHI: case op8::SFI: case op8::SFHI:
    case op8::ANDI: case op8::ORI: case op8::XORI:
    case op8::CEQI: case op8::CGTI: case op8::CLGTI:
    case op8::CEQHI: case op8::CGTHI: case op8::CLGTHI:
    case op8::CEQBI: case op8::CGTBI: case op8::CLGTBI:
    case op8::LQD: case op8::STQD:
    case op8::MPYI: case op8::MPYUI:
        out->format = 2; out->opcode = (uint16_t)o8;
        out->rT = spu_rT_ri10(raw);
        out->rA = spu_rA_ri10(raw);
        out->imm = spu_I10(raw);
        return;
    default:
        break;
    }

    // RI18 (7-bit)
    uint32_t o7 = spu_op7(raw);
    if (o7 == op7::ILA) {
        out->format = 3; out->opcode = (uint16_t)o7;
        out->rT = spu_rT_ri18(raw);
        out->immU = spu_I18(raw);
        return;
    }

    // RRR (4-bit) — last
    uint32_t o4 = spu_op4(raw);
    switch (o4) {
    case op4::SELB: case op4::SHUFB:
    case op4::FMA: case op4::FMS: case op4::FNMS:
    case op4::MPYA:
    case op4::DFMA: case op4::DFMS: case op4::DFNMS:
        out->format = 4; out->opcode = (uint16_t)o4;
        out->rT = spu_rT_rrr(raw);
        out->rA = spu_rA_rrr(raw);
        out->rB = spu_rB_rrr(raw);
        out->rC = spu_rC_rrr(raw);
        return;
    default:
        break;
    }

    // Unknown — will cause interpreter fallback
    out->format = 0xFF;
}

int jit_discover_block(const uint8_t* ls, uint32_t entryPC, BasicBlock* out) {
    memset(out, 0, sizeof(BasicBlock));
    out->entryPC = entryPC & (SPU_LS_SIZE - 1);

    uint32_t pc = out->entryPC;
    uint32_t n = 0;

    while (n < MAX_BLOCK_INSNS) {
        uint32_t raw = fetch_inst_host(ls, pc);
        DecodedInsn* di = &out->insns[n];
        decode_insn(raw, pc, di);

        // Track register usage
        if (di->rT < 128) { out->usesRegs[di->rT] = true; out->writesRegs[di->rT] = true; }
        if (di->rA < 128) out->usesRegs[di->rA] = true;
        if (di->rB < 128) out->usesRegs[di->rB] = true;
        if (di->rC < 128) out->usesRegs[di->rC] = true;

        // Track LS/MFC access
        uint16_t op = di->opcode;
        if (di->format == 2 && (op == op8::LQD || op == op8::STQD)) out->readsLS = out->writesLS = true;
        if (di->format == 1 && (op == op9::LQA || op == op9::LQR)) out->readsLS = true;
        if (di->format == 1 && (op == op9::STQA || op == op9::STQR)) out->writesLS = true;
        if (di->format == 0 && (op == op11::RDCH || op == op11::WRCH || op == op11::RCHCNT))
            out->usesMFC = true;

        // Check if this is an indirect branch (BI, BISL, BIZ, etc.)
        if (di->isBranch && di->format == 0 &&
            (op == op11::BI || op == op11::BISL || op == op11::BIZ ||
             op == op11::BINZ || op == op11::BIHZ || op == op11::BIHNZ))
            out->hasIndirectBranch = true;

        n++;
        out->endPC = pc;

        // Block terminators
        if (di->isBranch || di->isStop) break;
        if (di->format == 0xFF) break; // unknown instruction

        pc = (pc + 4) & (SPU_LS_SIZE - 1);

        // Safety: don't wrap around LS
        if (pc == out->entryPC) break;
    }

    out->numInsns = n;

    // Hash the LS region for cache validation
    uint32_t blockBytes = (out->endPC - out->entryPC + 4);
    if (out->endPC >= out->entryPC) {
        out->lsHash = hash_ls_region(ls + out->entryPC, blockBytes);
    } else {
        out->lsHash = hash_ls_region(ls + out->entryPC, SPU_LS_SIZE - out->entryPC);
    }

    return (int)n;
}

// ═══════════════════════════════════════════════════════════════
// CUDA C++ Source Emission
// ═══════════════════════════════════════════════════════════════

// Append formatted string to buffer
static int emit(char* buf, size_t bufSize, size_t* pos, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(buf + *pos, bufSize - *pos, fmt, args);
    va_end(args);
    if (written > 0) *pos += written;
    return written;
}

// Emit the CUDA C++ source for a basic block
int jit_emit_source(const BasicBlock* block, char* buf, size_t bufSize) {
    size_t pos = 0;

    // Header: QWord union, bswap, LS accessors — no system includes (NVRTC)
    emit(buf, bufSize, &pos,
        "// Auto-generated SPU JIT block: PC=0x%x, %u instructions\n"
        "typedef unsigned int uint32_t;\n"
        "typedef int int32_t;\n"
        "typedef unsigned short uint16_t;\n"
        "typedef short int16_t;\n"
        "typedef unsigned char uint8_t;\n"
        "typedef signed char int8_t;\n"
        "typedef unsigned long long uint64_t;\n\n"
        "union QWord {\n"
        "    uint32_t u32[4]; int32_t s32[4]; float f32[4];\n"
        "    uint16_t u16[8]; int16_t s16[8];\n"
        "    uint8_t u8[16]; int8_t s8[16];\n"
        "    uint64_t u64[2]; double f64[2];\n"
        "};\n\n"
        "struct SPUJITState {\n"
        "    QWord gpr[128];\n"
        "    uint32_t pc;\n"
        "    uint32_t npc;\n"
        "    uint32_t halted;\n"
        "    uint32_t cycles;\n"
        "};\n\n"
        "__device__ __forceinline__ uint32_t bswap32(uint32_t x) {\n"
        "    return __byte_perm(x, 0, 0x0123);\n"
        "}\n\n"
        "__device__ __forceinline__ QWord ls_read_qw(const uint8_t* ls, uint32_t addr) {\n"
        "    addr &= 0x3FFF0u;\n"
        "    QWord q;\n"
        "    const uint32_t* p = (const uint32_t*)(ls + addr);\n"
        "    q.u32[0]=bswap32(p[0]); q.u32[1]=bswap32(p[1]);\n"
        "    q.u32[2]=bswap32(p[2]); q.u32[3]=bswap32(p[3]);\n"
        "    return q;\n"
        "}\n\n"
        "__device__ __forceinline__ void ls_write_qw(uint8_t* ls, uint32_t addr, const QWord& q) {\n"
        "    addr &= 0x3FFF0u;\n"
        "    uint32_t* p = (uint32_t*)(ls + addr);\n"
        "    p[0]=bswap32(q.u32[0]); p[1]=bswap32(q.u32[1]);\n"
        "    p[2]=bswap32(q.u32[2]); p[3]=bswap32(q.u32[3]);\n"
        "}\n\n",
        block->entryPC, block->numInsns);

    // Kernel signature
    emit(buf, bufSize, &pos,
        "extern \"C\" __global__ void jit_block_0x%x(\n"
        "    SPUJITState* __restrict__ state,\n"
        "    uint8_t* __restrict__ ls,\n"
        "    uint8_t* __restrict__ mainMem)\n"
        "{\n"
        "    if (threadIdx.x != 0) return;\n"
        "    SPUJITState& s = *state;\n\n",
        block->entryPC);

    // Emit register aliases for used registers (helps compiler optimize)
    // We use direct s.gpr[N] access — compiler can register-allocate
    // the QWord locals if it wants.

    // Emit each instruction
    for (uint32_t i = 0; i < block->numInsns; i++) {
        const DecodedInsn& di = block->insns[i];
        emit(buf, bufSize, &pos, "    // [0x%04x] ", di.pc);

        // Instruction emission by format + opcode
        if (di.format == 0) { // RR (11-bit)
            uint8_t rT = di.rT, rA = di.rA, rB = di.rB;
            switch (di.opcode) {
            // Integer arithmetic
            case op11::A:
                emit(buf, bufSize, &pos, "a r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=s.gpr[%d].u32[i]+s.gpr[%d].u32[i];\n", rT, rA, rB);
                break;
            case op11::AH:
                emit(buf, bufSize, &pos, "ah r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<8;i++) s.gpr[%d].u16[i]=s.gpr[%d].u16[i]+s.gpr[%d].u16[i];\n", rT, rA, rB);
                break;
            case op11::SF:
                emit(buf, bufSize, &pos, "sf r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=s.gpr[%d].u32[i]-s.gpr[%d].u32[i];\n", rT, rB, rA);
                break;
            case op11::SFH:
                emit(buf, bufSize, &pos, "sfh r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<8;i++) s.gpr[%d].u16[i]=s.gpr[%d].u16[i]-s.gpr[%d].u16[i];\n", rT, rB, rA);
                break;
            case op11::AND:
                emit(buf, bufSize, &pos, "and r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=s.gpr[%d].u32[i]&s.gpr[%d].u32[i];\n", rT, rA, rB);
                break;
            case op11::OR:
                emit(buf, bufSize, &pos, "or r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=s.gpr[%d].u32[i]|s.gpr[%d].u32[i];\n", rT, rA, rB);
                break;
            case op11::XOR:
                emit(buf, bufSize, &pos, "xor r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=s.gpr[%d].u32[i]^s.gpr[%d].u32[i];\n", rT, rA, rB);
                break;
            case op11::ANDC:
                emit(buf, bufSize, &pos, "andc r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=s.gpr[%d].u32[i]&~s.gpr[%d].u32[i];\n", rT, rA, rB);
                break;
            case op11::ORC:
                emit(buf, bufSize, &pos, "orc r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=s.gpr[%d].u32[i]|~s.gpr[%d].u32[i];\n", rT, rA, rB);
                break;
            case op11::NAND:
                emit(buf, bufSize, &pos, "nand r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=~(s.gpr[%d].u32[i]&s.gpr[%d].u32[i]);\n", rT, rA, rB);
                break;
            case op11::NOR:
                emit(buf, bufSize, &pos, "nor r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=~(s.gpr[%d].u32[i]|s.gpr[%d].u32[i]);\n", rT, rA, rB);
                break;

            // Shifts and rotates
            case op11::SHL:
                emit(buf, bufSize, &pos, "shl r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++){uint32_t sh=s.gpr[%d].u32[i]&0x3F;s.gpr[%d].u32[i]=(sh<32)?(s.gpr[%d].u32[i]<<sh):0;}\n", rB, rT, rA);
                break;
            case op11::ROT:
                emit(buf, bufSize, &pos, "rot r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++){uint32_t sh=s.gpr[%d].u32[i]&0x1F;uint32_t v=s.gpr[%d].u32[i];s.gpr[%d].u32[i]=(v<<sh)|(v>>(32-sh));}\n", rB, rA, rT);
                break;
            case op11::ROTM:
                emit(buf, bufSize, &pos, "rotm r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++){uint32_t sh=(-(int32_t)s.gpr[%d].u32[i])&0x3F;s.gpr[%d].u32[i]=(sh<32)?(s.gpr[%d].u32[i]>>sh):0;}\n", rB, rT, rA);
                break;
            case op11::ROTMA:
                emit(buf, bufSize, &pos, "rotma r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++){uint32_t sh=(-(int32_t)s.gpr[%d].u32[i])&0x3F;int32_t v=s.gpr[%d].s32[i];s.gpr[%d].s32[i]=(sh<32)?(v>>sh):(v>>31);}\n", rB, rA, rT);
                break;

            // Multiply
            case op11::MPY:
                emit(buf, bufSize, &pos, "mpy r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++){int16_t a=(int16_t)(s.gpr[%d].u32[i]&0xFFFF),b=(int16_t)(s.gpr[%d].u32[i]&0xFFFF);s.gpr[%d].s32[i]=(int32_t)a*(int32_t)b;}\n", rA, rB, rT);
                break;
            case op11::MPYU:
                emit(buf, bufSize, &pos, "mpyu r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++){uint16_t a=(uint16_t)(s.gpr[%d].u32[i]&0xFFFF),b=(uint16_t)(s.gpr[%d].u32[i]&0xFFFF);s.gpr[%d].u32[i]=(uint32_t)a*(uint32_t)b;}\n", rA, rB, rT);
                break;
            case op11::MPYH:
                emit(buf, bufSize, &pos, "mpyh r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++){uint16_t a=(uint16_t)(s.gpr[%d].u32[i]>>16),b=(uint16_t)(s.gpr[%d].u32[i]&0xFFFF);s.gpr[%d].u32[i]=((uint32_t)a*(uint32_t)b)<<16;}\n", rA, rB, rT);
                break;

            // Compare
            case op11::CEQ:
                emit(buf, bufSize, &pos, "ceq r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=(s.gpr[%d].u32[i]==s.gpr[%d].u32[i])?0xFFFFFFFFu:0;\n", rT, rA, rB);
                break;
            case op11::CGT:
                emit(buf, bufSize, &pos, "cgt r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=(s.gpr[%d].s32[i]>s.gpr[%d].s32[i])?0xFFFFFFFFu:0;\n", rT, rA, rB);
                break;
            case op11::CLGT:
                emit(buf, bufSize, &pos, "clgt r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=(s.gpr[%d].u32[i]>s.gpr[%d].u32[i])?0xFFFFFFFFu:0;\n", rT, rA, rB);
                break;

            // Float arithmetic
            case op11::FA:
                emit(buf, bufSize, &pos, "fa r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].f32[i]=s.gpr[%d].f32[i]+s.gpr[%d].f32[i];\n", rT, rA, rB);
                break;
            case op11::FS:
                emit(buf, bufSize, &pos, "fs r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].f32[i]=s.gpr[%d].f32[i]-s.gpr[%d].f32[i];\n", rT, rA, rB);
                break;
            case op11::FM:
                emit(buf, bufSize, &pos, "fm r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].f32[i]=s.gpr[%d].f32[i]*s.gpr[%d].f32[i];\n", rT, rA, rB);
                break;
            case op11::FCEQ:
                emit(buf, bufSize, &pos, "fceq r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=(s.gpr[%d].f32[i]==s.gpr[%d].f32[i])?0xFFFFFFFFu:0;\n", rT, rA, rB);
                break;
            case op11::FCGT:
                emit(buf, bufSize, &pos, "fcgt r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=(s.gpr[%d].f32[i]>s.gpr[%d].f32[i])?0xFFFFFFFFu:0;\n", rT, rA, rB);
                break;

            // Double precision
            case op11::DFA:
                emit(buf, bufSize, &pos, "dfa r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<2;i++) s.gpr[%d].f64[i]=s.gpr[%d].f64[i]+s.gpr[%d].f64[i];\n", rT, rA, rB);
                break;
            case op11::DFS:
                emit(buf, bufSize, &pos, "dfs r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<2;i++) s.gpr[%d].f64[i]=s.gpr[%d].f64[i]-s.gpr[%d].f64[i];\n", rT, rA, rB);
                break;
            case op11::DFM:
                emit(buf, bufSize, &pos, "dfm r%d, r%d, r%d\n", rT, rA, rB);
                emit(buf, bufSize, &pos, "    for(int i=0;i<2;i++) s.gpr[%d].f64[i]=s.gpr[%d].f64[i]*s.gpr[%d].f64[i];\n", rT, rA, rB);
                break;

            // Quadword ops
            case op11::CLZ:
                emit(buf, bufSize, &pos, "clz r%d, r%d\n", rT, rA);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=s.gpr[%d].u32[i]?__clz(s.gpr[%d].u32[i]):32;\n", rT, rA, rA);
                break;

            // Branch indirect
            case op11::BI:
                emit(buf, bufSize, &pos, "bi r%d\n", rA);
                emit(buf, bufSize, &pos, "    s.npc=s.gpr[%d].u32[0]&0x3FFFFu&~0x3u;\n", rA);
                break;
            case op11::BISL:
                emit(buf, bufSize, &pos, "bisl r%d, r%d\n", rT, rA);
                emit(buf, bufSize, &pos, "    s.gpr[%d].u32[0]=s.pc+4;s.gpr[%d].u32[1]=s.gpr[%d].u32[2]=s.gpr[%d].u32[3]=0;\n", rT, rT, rT, rT);
                emit(buf, bufSize, &pos, "    s.npc=s.gpr[%d].u32[0]&0x3FFFFu&~0x3u;\n", rA);
                break;
            case op11::BIZ:
                emit(buf, bufSize, &pos, "biz r%d, r%d\n", rT, rA);
                emit(buf, bufSize, &pos, "    if(s.gpr[%d].u32[0]==0) s.npc=s.gpr[%d].u32[0]&0x3FFFFu&~0x3u;\n", rT, rA);
                break;
            case op11::BINZ:
                emit(buf, bufSize, &pos, "binz r%d, r%d\n", rT, rA);
                emit(buf, bufSize, &pos, "    if(s.gpr[%d].u32[0]!=0) s.npc=s.gpr[%d].u32[0]&0x3FFFFu&~0x3u;\n", rT, rA);
                break;

            // Channel
            case op11::RDCH:
                emit(buf, bufSize, &pos, "rdch r%d, ch%d\n", rT, rA);
                emit(buf, bufSize, &pos, "    // channel read — fallback to interpreter\n");
                emit(buf, bufSize, &pos, "    s.gpr[%d].u32[0]=0;s.gpr[%d].u32[1]=s.gpr[%d].u32[2]=s.gpr[%d].u32[3]=0;\n", rT, rT, rT, rT);
                break;
            case op11::WRCH:
                emit(buf, bufSize, &pos, "wrch ch%d, r%d\n", rA, rT);
                emit(buf, bufSize, &pos, "    // channel write — simplified in JIT\n");
                break;
            case op11::RCHCNT:
                emit(buf, bufSize, &pos, "rchcnt r%d, ch%d\n", rT, rA);
                emit(buf, bufSize, &pos, "    s.gpr[%d].u32[0]=1;s.gpr[%d].u32[1]=s.gpr[%d].u32[2]=s.gpr[%d].u32[3]=0;\n", rT, rT, rT, rT);
                break;

            // NOP
            case op11::NOP: case op11::LNOP: case op11::SYNC: case op11::DSYNC:
                emit(buf, bufSize, &pos, "nop\n");
                break;

            // Stop
            case op11::STOP: case op11::STOPD:
                emit(buf, bufSize, &pos, "stop\n");
                emit(buf, bufSize, &pos, "    s.halted=1;\n");
                break;

            default:
                emit(buf, bufSize, &pos, "unknown_rr op=0x%03x\n", di.opcode);
                emit(buf, bufSize, &pos, "    // UNHANDLED — interpreter fallback needed\n");
                break;
            }
        }
        else if (di.format == 1) { // RI16 (9-bit)
            uint8_t rT = di.rT;
            int32_t imm = di.imm;
            uint32_t immU = di.immU;

            switch (di.opcode) {
            case op9::IL:
                emit(buf, bufSize, &pos, "il r%d, %d\n", rT, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].s32[i]=%d;\n", rT, imm);
                break;
            case op9::ILH:
                emit(buf, bufSize, &pos, "ilh r%d, 0x%x\n", rT, immU);
                emit(buf, bufSize, &pos, "    for(int i=0;i<8;i++) s.gpr[%d].u16[i]=(uint16_t)%uu;\n", rT, immU);
                break;
            case op9::ILHU:
                emit(buf, bufSize, &pos, "ilhu r%d, 0x%x\n", rT, immU);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=%uu<<16;\n", rT, immU);
                break;
            case op9::IOHL:
                emit(buf, bufSize, &pos, "iohl r%d, 0x%x\n", rT, immU);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]|=%uu;\n", rT, immU);
                break;
            case op9::FSMBI:
                emit(buf, bufSize, &pos, "fsmbi r%d, 0x%x\n", rT, immU);
                emit(buf, bufSize, &pos, "    for(int i=0;i<16;i++) s.gpr[%d].u8[i]=(%uu&(1<<(15-i)))?0xFF:0x00;\n", rT, immU);
                break;
            case op9::BR:
                emit(buf, bufSize, &pos, "br 0x%x\n", (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1));
                emit(buf, bufSize, &pos, "    s.npc=0x%xu;\n", (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1));
                break;
            case op9::BRA:
                emit(buf, bufSize, &pos, "bra 0x%x\n", ((uint32_t)(imm << 2)) & (SPU_LS_SIZE - 1));
                emit(buf, bufSize, &pos, "    s.npc=0x%xu;\n", ((uint32_t)(imm << 2)) & (SPU_LS_SIZE - 1));
                break;
            case op9::BRSL:
                emit(buf, bufSize, &pos, "brsl r%d, 0x%x\n", rT, (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1));
                emit(buf, bufSize, &pos, "    s.gpr[%d].u32[0]=0x%xu;s.gpr[%d].u32[1]=s.gpr[%d].u32[2]=s.gpr[%d].u32[3]=0;\n",
                     rT, block->insns[i].pc + 4, rT, rT, rT);
                emit(buf, bufSize, &pos, "    s.npc=0x%xu;\n", (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1));
                break;
            case op9::BRZ:
                emit(buf, bufSize, &pos, "brz r%d, 0x%x\n", rT, (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1));
                emit(buf, bufSize, &pos, "    if(s.gpr[%d].u32[0]==0) s.npc=0x%xu;\n", rT, (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1));
                break;
            case op9::BRNZ:
                emit(buf, bufSize, &pos, "brnz r%d, 0x%x\n", rT, (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1));
                emit(buf, bufSize, &pos, "    if(s.gpr[%d].u32[0]!=0) s.npc=0x%xu;\n", rT, (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1));
                break;
            case op9::LQA: {
                uint32_t ea = ((uint32_t)(imm << 2)) & (SPU_LS_SIZE - 1) & ~0xF;
                emit(buf, bufSize, &pos, "lqa r%d, 0x%x\n", rT, ea);
                emit(buf, bufSize, &pos, "    s.gpr[%d]=ls_read_qw(ls,0x%xu);\n", rT, ea);
                break;
            }
            case op9::STQA: {
                uint32_t ea = ((uint32_t)(imm << 2)) & (SPU_LS_SIZE - 1) & ~0xF;
                emit(buf, bufSize, &pos, "stqa r%d, 0x%x\n", rT, ea);
                emit(buf, bufSize, &pos, "    ls_write_qw(ls,0x%xu,s.gpr[%d]);\n", ea, rT);
                break;
            }
            case op9::LQR: {
                emit(buf, bufSize, &pos, "lqr r%d, 0x%x\n", rT, (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1) & ~0xF);
                emit(buf, bufSize, &pos, "    s.gpr[%d]=ls_read_qw(ls,0x%xu);\n", rT, (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1) & ~0xF);
                break;
            }
            case op9::STQR: {
                emit(buf, bufSize, &pos, "stqr r%d, 0x%x\n", rT, (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1) & ~0xF);
                emit(buf, bufSize, &pos, "    ls_write_qw(ls,0x%xu,s.gpr[%d]);\n", (block->insns[i].pc + (imm << 2)) & (SPU_LS_SIZE - 1) & ~0xF, rT);
                break;
            }
            default:
                emit(buf, bufSize, &pos, "unknown_ri16 op=0x%03x\n", di.opcode);
                break;
            }
        }
        else if (di.format == 2) { // RI10 (8-bit)
            uint8_t rT = di.rT, rA = di.rA;
            int32_t imm = di.imm;

            switch (di.opcode) {
            case op8::AI:
                emit(buf, bufSize, &pos, "ai r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].s32[i]=s.gpr[%d].s32[i]+%d;\n", rT, rA, imm);
                break;
            case op8::AHI:
                emit(buf, bufSize, &pos, "ahi r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    {int16_t imm=(int16_t)(%d&0xFFFF);for(int i=0;i<8;i++) s.gpr[%d].s16[i]=s.gpr[%d].s16[i]+imm;}\n", imm, rT, rA);
                break;
            case op8::SFI:
                emit(buf, bufSize, &pos, "sfi r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].s32[i]=%d-s.gpr[%d].s32[i];\n", rT, imm, rA);
                break;
            case op8::ANDI:
                emit(buf, bufSize, &pos, "andi r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].s32[i]=s.gpr[%d].s32[i]&%d;\n", rT, rA, imm);
                break;
            case op8::ORI:
                emit(buf, bufSize, &pos, "ori r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].s32[i]=s.gpr[%d].s32[i]|%d;\n", rT, rA, imm);
                break;
            case op8::XORI:
                emit(buf, bufSize, &pos, "xori r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].s32[i]=s.gpr[%d].s32[i]^%d;\n", rT, rA, imm);
                break;
            case op8::CEQI:
                emit(buf, bufSize, &pos, "ceqi r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=(s.gpr[%d].s32[i]==%d)?0xFFFFFFFFu:0;\n", rT, rA, imm);
                break;
            case op8::CGTI:
                emit(buf, bufSize, &pos, "cgti r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=(s.gpr[%d].s32[i]>%d)?0xFFFFFFFFu:0;\n", rT, rA, imm);
                break;
            case op8::CLGTI:
                emit(buf, bufSize, &pos, "clgti r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=(s.gpr[%d].u32[i]>(uint32_t)%d)?0xFFFFFFFFu:0;\n", rT, rA, imm);
                break;
            case op8::LQD:
                emit(buf, bufSize, &pos, "lqd r%d, %d(r%d)\n", rT, imm << 4, rA);
                emit(buf, bufSize, &pos, "    s.gpr[%d]=ls_read_qw(ls,(s.gpr[%d].u32[0]+%d)&0x3FFFFu);\n", rT, rA, imm << 4);
                break;
            case op8::STQD:
                emit(buf, bufSize, &pos, "stqd r%d, %d(r%d)\n", rT, imm << 4, rA);
                emit(buf, bufSize, &pos, "    ls_write_qw(ls,(s.gpr[%d].u32[0]+%d)&0x3FFFFu,s.gpr[%d]);\n", rA, imm << 4, rT);
                break;
            case op8::MPYI:
                emit(buf, bufSize, &pos, "mpyi r%d, r%d, %d\n", rT, rA, imm);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++){int16_t a=(int16_t)(s.gpr[%d].u32[i]&0xFFFF);s.gpr[%d].s32[i]=(int32_t)a*%d;}\n", rA, rT, imm);
                break;
            default:
                emit(buf, bufSize, &pos, "unknown_ri10 op=0x%02x\n", di.opcode);
                break;
            }
        }
        else if (di.format == 3) { // RI18 (7-bit)
            emit(buf, bufSize, &pos, "ila r%d, 0x%x\n", di.rT, di.immU);
            emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=%uu;\n", di.rT, di.immU);
        }
        else if (di.format == 4) { // RRR (4-bit)
            uint8_t rT = di.rT, rA = di.rA, rB = di.rB, rC = di.rC;

            switch (di.opcode) {
            case op4::SELB:
                emit(buf, bufSize, &pos, "selb r%d, r%d, r%d, r%d\n", rT, rA, rB, rC);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].u32[i]=(s.gpr[%d].u32[i]&s.gpr[%d].u32[i])|(s.gpr[%d].u32[i]&~s.gpr[%d].u32[i]);\n", rT, rB, rC, rA, rC);
                break;
            case op4::SHUFB:
                emit(buf, bufSize, &pos, "shufb r%d, r%d, r%d, r%d\n", rT, rA, rB, rC);
                // Emit inline SHUFB (can't call device function from NVRTC without providing it)
                emit(buf, bufSize, &pos,
                    "    {\n"
                    "        QWord _a=s.gpr[%d],_b=s.gpr[%d],_c=s.gpr[%d],_r;\n"
                    "        for(int _i=0;_i<16;_i++){\n"
                    "            uint8_t sel=_c.u8[_i];\n"
                    "            if(sel&0x80){if((sel&0xE0)==0xC0)_r.u8[_i]=0;else if((sel&0xE0)==0xE0)_r.u8[_i]=0xFF;else _r.u8[_i]=0x80;}\n"
                    "            else{uint8_t idx=sel&0x1F;_r.u8[_i]=(idx<16)?_a.u8[idx]:_b.u8[idx-16];}\n"
                    "        }\n"
                    "        s.gpr[%d]=_r;\n"
                    "    }\n", rA, rB, rC, rT);
                break;
            case op4::FMA:
                emit(buf, bufSize, &pos, "fma r%d, r%d, r%d, r%d\n", rT, rA, rB, rC);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].f32[i]=s.gpr[%d].f32[i]*s.gpr[%d].f32[i]+s.gpr[%d].f32[i];\n", rT, rA, rB, rC);
                break;
            case op4::FMS:
                emit(buf, bufSize, &pos, "fms r%d, r%d, r%d, r%d\n", rT, rA, rB, rC);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].f32[i]=s.gpr[%d].f32[i]*s.gpr[%d].f32[i]-s.gpr[%d].f32[i];\n", rT, rA, rB, rC);
                break;
            case op4::FNMS:
                emit(buf, bufSize, &pos, "fnms r%d, r%d, r%d, r%d\n", rT, rA, rB, rC);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++) s.gpr[%d].f32[i]=s.gpr[%d].f32[i]-s.gpr[%d].f32[i]*s.gpr[%d].f32[i];\n", rT, rC, rA, rB);
                break;
            case op4::MPYA:
                emit(buf, bufSize, &pos, "mpya r%d, r%d, r%d, r%d\n", rT, rA, rB, rC);
                emit(buf, bufSize, &pos, "    for(int i=0;i<4;i++){int16_t a=(int16_t)(s.gpr[%d].u32[i]&0xFFFF),b=(int16_t)(s.gpr[%d].u32[i]&0xFFFF);s.gpr[%d].s32[i]=(int32_t)a*(int32_t)b+s.gpr[%d].s32[i];}\n", rA, rB, rT, rC);
                break;
            case op4::DFMA:
                emit(buf, bufSize, &pos, "dfma r%d, r%d, r%d, r%d\n", rT, rA, rB, rC);
                emit(buf, bufSize, &pos, "    for(int i=0;i<2;i++) s.gpr[%d].f64[i]=s.gpr[%d].f64[i]*s.gpr[%d].f64[i]+s.gpr[%d].f64[i];\n", rT, rA, rB, rC);
                break;
            case op4::DFMS:
                emit(buf, bufSize, &pos, "dfms r%d, r%d, r%d, r%d\n", rT, rA, rB, rC);
                emit(buf, bufSize, &pos, "    for(int i=0;i<2;i++) s.gpr[%d].f64[i]=s.gpr[%d].f64[i]*s.gpr[%d].f64[i]-s.gpr[%d].f64[i];\n", rT, rA, rB, rC);
                break;
            case op4::DFNMS:
                emit(buf, bufSize, &pos, "dfnms r%d, r%d, r%d, r%d\n", rT, rA, rB, rC);
                emit(buf, bufSize, &pos, "    for(int i=0;i<2;i++) s.gpr[%d].f64[i]=s.gpr[%d].f64[i]-s.gpr[%d].f64[i]*s.gpr[%d].f64[i];\n", rT, rC, rA, rB);
                break;
            default:
                emit(buf, bufSize, &pos, "unknown_rrr op=0x%x\n", di.opcode);
                break;
            }
        }
        else {
            emit(buf, bufSize, &pos, "unknown_format\n");
        }

        // Update PC/cycles after each instruction
        emit(buf, bufSize, &pos, "    s.pc=0x%xu; s.cycles++;\n",
             (di.pc + 4) & (SPU_LS_SIZE - 1));
    }

    // Set final npc
    emit(buf, bufSize, &pos,
        "\n    // Block exit\n"
        "    s.pc = s.npc;\n"
        "}\n");

    return (int)pos;
}

// ═══════════════════════════════════════════════════════════════
// NVRTC Compilation
// ═══════════════════════════════════════════════════════════════

int jit_init(JITState* state) {
    memset(state, 0, sizeof(JITState));
    // Initialize CUDA driver API
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[JIT] cuInit failed: %d\n", err);
        return 0;
    }
    state->ready = true;
    fprintf(stderr, "[JIT] SPU JIT compiler initialized (NVRTC backend)\n");
    return 1;
}

int jit_compile(JITState* state, const BasicBlock* block,
                const char* source, JITEntry* out) {
    if (!state->ready) return 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Create NVRTC program
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, source, "spu_jit_block.cu", 0, NULL, NULL);
    if (res != NVRTC_SUCCESS) {
        fprintf(stderr, "[JIT] nvrtcCreateProgram failed: %s\n", nvrtcGetErrorString(res));
        return 0;
    }

    // Compile for sm_70 (V100)
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
        fprintf(stderr, "[JIT] Compile failed for block 0x%x:\n%s\n",
                block->entryPC, log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return 0;
    }

    // Get PTX
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = (char*)malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    // Load PTX via driver API
    CUmodule cuMod;
    CUresult cuRes = cuModuleLoadDataEx(&cuMod, ptx, 0, NULL, NULL);
    free(ptx);

    if (cuRes != CUDA_SUCCESS) {
        fprintf(stderr, "[JIT] cuModuleLoadData failed: %d\n", cuRes);
        return 0;
    }

    // Get kernel function
    char funcName[64];
    snprintf(funcName, sizeof(funcName), "jit_block_0x%x", block->entryPC);

    CUfunction cuFunc;
    cuRes = cuModuleGetFunction(&cuFunc, cuMod, funcName);
    if (cuRes != CUDA_SUCCESS) {
        fprintf(stderr, "[JIT] cuModuleGetFunction('%s') failed: %d\n", funcName, cuRes);
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

    // Fill output
    out->entryPC = block->entryPC;
    out->lsHash = block->lsHash;
    out->cuModule = (void*)cuMod;
    out->cuFunction = (void*)cuFunc;
    out->numInsns = block->numInsns;
    out->hitCount = 0;
    out->valid = true;

    fprintf(stderr, "[JIT] Compiled block 0x%04x: %u insns → CUDA kernel (%.1f ms)\n",
            block->entryPC, block->numInsns, compileMs);
    return 1;
}

// ═══════════════════════════════════════════════════════════════
// JIT Execution via Driver API
// ═══════════════════════════════════════════════════════════════

// SPUJITState layout must match the NVRTC-generated struct
struct SPUJITState {
    QWord gpr[128];
    uint32_t pc;
    uint32_t npc;
    uint32_t halted;
    uint32_t cycles;
};

int jit_execute(const JITEntry* entry, SPUState* d_state,
                uint8_t* d_ls, uint8_t* d_mainMem) {
    if (!entry || !entry->valid) return 0;

    // We need to copy SPUState → SPUJITState (subset) on device
    // For simplicity, allocate a temporary JIT state, copy from SPU state, run, copy back
    SPUJITState h_jit;
    SPUState h_spu;
    cudaMemcpy(&h_spu, d_state, sizeof(SPUState), cudaMemcpyDeviceToHost);

    // Copy registers and PC
    memcpy(h_jit.gpr, h_spu.gpr, sizeof(h_jit.gpr));
    h_jit.pc = h_spu.pc;
    h_jit.npc = h_spu.npc;
    h_jit.halted = h_spu.halted;
    h_jit.cycles = h_spu.cycles;

    // Upload JIT state
    SPUJITState* d_jit;
    cudaMalloc(&d_jit, sizeof(SPUJITState));
    cudaMemcpy(d_jit, &h_jit, sizeof(SPUJITState), cudaMemcpyHostToDevice);

    // Launch via driver API
    void* args[] = { &d_jit, &d_ls, &d_mainMem };
    CUresult err = cuLaunchKernel(
        (CUfunction)entry->cuFunction,
        1, 1, 1,    // grid
        1, 1, 1,    // block (single thread — we're executing one SPU)
        0, 0,       // shared mem, stream
        args, NULL);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[JIT] cuLaunchKernel failed: %d\n", err);
        cudaFree(d_jit);
        return 0;
    }

    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(&h_jit, d_jit, sizeof(SPUJITState), cudaMemcpyDeviceToHost);
    cudaFree(d_jit);

    // Write back to SPU state
    memcpy(h_spu.gpr, h_jit.gpr, sizeof(h_spu.gpr));
    h_spu.pc = h_jit.pc;
    h_spu.npc = h_jit.npc;
    h_spu.halted = h_jit.halted;
    h_spu.cycles = h_jit.cycles;
    cudaMemcpy(d_state, &h_spu, sizeof(SPUState), cudaMemcpyHostToDevice);

    return 1;
}

// ═══════════════════════════════════════════════════════════════
// Cache Management
// ═══════════════════════════════════════════════════════════════

static JITEntry* cache_lookup(JITState* state, uint32_t pc, uint64_t lsHash) {
    for (uint32_t i = 0; i < state->numEntries; i++) {
        if (state->cache[i].valid &&
            state->cache[i].entryPC == pc &&
            state->cache[i].lsHash == lsHash) {
            state->cacheHits++;
            state->cache[i].hitCount++;
            return &state->cache[i];
        }
    }
    state->cacheMisses++;
    return nullptr;
}

static JITEntry* cache_insert(JITState* state) {
    if (state->numEntries < MAX_BLOCKS) {
        return &state->cache[state->numEntries++];
    }
    // Evict LRU (lowest hit count)
    uint32_t minHits = UINT32_MAX;
    uint32_t minIdx = 0;
    for (uint32_t i = 0; i < state->numEntries; i++) {
        if (state->cache[i].hitCount < minHits) {
            minHits = state->cache[i].hitCount;
            minIdx = i;
        }
    }
    // Unload old module
    if (state->cache[minIdx].cuModule) {
        cuModuleUnload((CUmodule)state->cache[minIdx].cuModule);
    }
    state->cache[minIdx].valid = false;
    return &state->cache[minIdx];
}

// ═══════════════════════════════════════════════════════════════
// Full Pipeline
// ═══════════════════════════════════════════════════════════════

int jit_run_block(JITState* state, SPUState* d_state,
                  uint8_t* d_ls, uint8_t* d_mainMem,
                  uint32_t entryPC) {
    // Read LS from device to discover block
    uint8_t* h_ls = (uint8_t*)malloc(SPU_LS_SIZE);
    cudaMemcpy(h_ls, d_ls, SPU_LS_SIZE, cudaMemcpyDeviceToHost);

    // Discover block
    BasicBlock block;
    int numInsns = jit_discover_block(h_ls, entryPC, &block);
    if (numInsns <= 0) {
        free(h_ls);
        state->interpreterFallbacks++;
        return 0;
    }

    // Cache lookup
    JITEntry* cached = cache_lookup(state, entryPC, block.lsHash);
    if (cached) {
        free(h_ls);
        return jit_execute(cached, d_state, d_ls, d_mainMem);
    }

    // Blocks with MFC/channel ops fall back to interpreter
    if (block.usesMFC) {
        free(h_ls);
        state->interpreterFallbacks++;
        return 0;
    }

    // Emit source
    char* source = (char*)malloc(MAX_SOURCE_SIZE);
    int srcLen = jit_emit_source(&block, source, MAX_SOURCE_SIZE);
    if (srcLen <= 0) {
        free(h_ls);
        free(source);
        state->interpreterFallbacks++;
        return 0;
    }

    // Compile
    JITEntry* slot = cache_insert(state);
    int ok = jit_compile(state, &block, source, slot);
    free(h_ls);
    free(source);

    if (!ok) {
        state->interpreterFallbacks++;
        return 0;
    }

    // Execute
    return jit_execute(slot, d_state, d_ls, d_mainMem);
}

// Run SPU with JIT dispatch loop
float jit_run_spu(JITState* state, int spuId, SPUState* d_states,
                  uint8_t** d_localStores, uint8_t* d_mainMem,
                  uint32_t maxCycles) {
    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);
    cudaEventRecord(tStart);

    SPUState h_spu;
    uint32_t cyclesRun = 0;

    while (cyclesRun < maxCycles) {
        cudaMemcpy(&h_spu, &d_states[spuId], sizeof(SPUState), cudaMemcpyDeviceToHost);
        if (h_spu.halted) break;

        int ok = jit_run_block(state, &d_states[spuId],
                               d_localStores[spuId], d_mainMem,
                               h_spu.pc);

        if (ok) {
            // Read back to check progress
            cudaMemcpy(&h_spu, &d_states[spuId], sizeof(SPUState), cudaMemcpyDeviceToHost);
            cyclesRun = h_spu.cycles;
        } else {
            // Interpreter fallback — run one block's worth on interpreter kernel
            // For now, just break
            break;
        }
    }

    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);
    float ms = 0;
    cudaEventElapsedTime(&ms, tStart, tStop);
    cudaEventDestroy(tStart);
    cudaEventDestroy(tStop);
    return ms;
}

void jit_print_stats(const JITState* state) {
    fprintf(stderr,
        "╔═══════════════════════════════════════╗\n"
        "║  SPU JIT Statistics                    ║\n"
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

void jit_shutdown(JITState* state) {
    for (uint32_t i = 0; i < state->numEntries; i++) {
        if (state->cache[i].valid && state->cache[i].cuModule) {
            cuModuleUnload((CUmodule)state->cache[i].cuModule);
        }
    }
    memset(state, 0, sizeof(JITState));
    fprintf(stderr, "[JIT] Shutdown\n");
}

} // namespace spu_jit
