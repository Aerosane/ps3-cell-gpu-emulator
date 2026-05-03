// spu_interpreter.cu — CUDA SPU Interpreter Kernel
// Each CUDA thread = one SPU core, interpreting from 256KB Local Store.
//
// Key insight: SPU 128-bit SIMD → native CUDA float4/uint4 ops
// SPU has NO cache — LS is flat SRAM. We keep LS in GPU global memory.
//
#include "spu_defs.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

using namespace spu;

// ═══════════════════════════════════════════════════════════════
// Big-Endian Helpers
// SPU is big-endian; GPU is little-endian. LS is stored in BE order.
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

// Read a quadword from Local Store (16 bytes, BE → LE for each word)
__device__ __forceinline__ QWord ls_read_qw(const uint8_t* ls, uint32_t addr) {
    addr &= (SPU_LS_SIZE - 1) & ~0xF;  // 16-byte aligned, wrap
    QWord q;
    const uint32_t* p = (const uint32_t*)(ls + addr);
    q.u32[0] = bswap32(p[0]);
    q.u32[1] = bswap32(p[1]);
    q.u32[2] = bswap32(p[2]);
    q.u32[3] = bswap32(p[3]);
    return q;
}

// Write a quadword to Local Store
__device__ __forceinline__ void ls_write_qw(uint8_t* ls, uint32_t addr, const QWord& q) {
    addr &= (SPU_LS_SIZE - 1) & ~0xF;
    uint32_t* p = (uint32_t*)(ls + addr);
    p[0] = bswap32(q.u32[0]);
    p[1] = bswap32(q.u32[1]);
    p[2] = bswap32(q.u32[2]);
    p[3] = bswap32(q.u32[3]);
}

// Fetch instruction (32-bit BE)
__device__ __forceinline__ uint32_t ls_fetch_inst(const uint8_t* ls, uint32_t pc) {
    pc &= (SPU_LS_SIZE - 1) & ~0x3;
    uint32_t raw;
    memcpy(&raw, ls + pc, 4);
    return bswap32(raw);
}

// ═══════════════════════════════════════════════════════════════
// MFC DMA Handler
// Copies between SPU Local Store and main memory (PS3 RAM)
// ═══════════════════════════════════════════════════════════════

__device__ static void mfc_execute(SPUState& s, uint8_t* ls, uint8_t* mainMem) {
    for (uint32_t i = 0; i < s.mfcQueueLen; i++) {
        MFCCommand& cmd = s.mfcQueue[i];
        if (!cmd.active) continue;

        uint64_t ea = ((uint64_t)s.mfc_eah << 32) | cmd.ea;
        uint32_t lsa = cmd.lsa & (SPU_LS_SIZE - 1);
        uint32_t size = cmd.size;

        // Clamp to sandbox
        if (ea + size > 512ULL * 1024 * 1024) { cmd.active = 0; continue; }
        if (lsa + size > SPU_LS_SIZE) { cmd.active = 0; continue; }

        switch (cmd.cmd) {
        // GET family: main memory → LS  (B=barrier, F=fence, S=start; all same for sync DMA)
        case 0x40: // MFC_GET
        case 0x41: // MFC_GETB
        case 0x42: // MFC_GETF
        case 0x48: // MFC_GETS
            memcpy(ls + lsa, mainMem + ea, size);
            break;
        // PUT family: LS → main memory
        case 0x20: // MFC_PUT
        case 0x21: // MFC_PUTB
        case 0x22: // MFC_PUTF
        case 0x28: // MFC_PUTS
            memcpy(mainMem + ea, ls + lsa, size);
            break;
        // Atomic 128-byte cache-line ops (treat as plain copy under single-thread sync DMA)
        case 0xD0: // MFC_GETLLAR  — Get Lock Line And Reserve
            memcpy(ls + lsa, mainMem + ea, 128);
            break;
        case 0xB4: // MFC_PUTLLC   — Put Lock Line Conditional
        case 0xB0: // MFC_PUTLLUC  — Put Lock Line Unconditional
        case 0xB8: // MFC_PUTQLLUC — Put Queued Lock Line Unconditional
            memcpy(mainMem + ea, ls + lsa, 128);
            break;
        // Sync ops: instant completion under sync DMA model
        case 0xC0: // MFC_BARRIER
        case 0xC8: // MFC_EIEIO
        case 0xCC: // MFC_SYNC
        case 0xA0: // MFC_SNDSIG (signal; no-op without subscribers)
            break;
        default:
            break;
        }
        cmd.active = 0;
    }
    s.mfcQueueLen = 0;
}

// Enqueue an MFC command
__device__ static void mfc_enqueue(SPUState& s) {
    if (s.mfcQueueLen >= SPU_MFC_QUEUE) return;
    MFCCommand& cmd = s.mfcQueue[s.mfcQueueLen++];
    cmd.lsa = s.mfc_lsa;
    cmd.ea = s.mfc_eal;
    cmd.size = s.mfc_size;
    cmd.tag = s.mfc_tag;
    cmd.cmd = 0;  // set by wrch(MFC_Cmd)
    cmd.active = 1;
}

// ═══════════════════════════════════════════════════════════════
// Channel Read/Write (SPU ↔ outside world)
// ═══════════════════════════════════════════════════════════════

__device__ static uint32_t channel_read(SPUState& s, uint32_t ch) {
    switch (ch) {
    case SPU_RdInMbox:
        if (s.inMboxValid) { s.inMboxValid = 0; return s.inMbox; }
        return 0;
    case SPU_RdSRR0:       return s.srr0;
    case SPU_RdDec:        return s.decrementer;
    case SPU_RdEventStat:  return s.eventStat & s.eventMask;
    case SPU_RdMachStat:   return 1; // running in single-thread mode
    case MFC_RdTagStat:    return s.mfc_tagMask; // all tags complete (instant DMA)
    case MFC_RdTagMask:    return s.mfc_tagMask;
    default:               return 0;
    }
}

__device__ static void channel_write(SPUState& s, uint32_t ch, uint32_t val,
                                      uint8_t* ls, uint8_t* mainMem,
                                      volatile uint32_t* mailboxOut) {
    switch (ch) {
    case MFC_LSA:          s.mfc_lsa = val; break;
    case MFC_EAH:          s.mfc_eah = val; break;
    case MFC_EAL:          s.mfc_eal = val; break;
    case MFC_Size:         s.mfc_size = val; break;
    case MFC_TagID:        s.mfc_tag = val; break;
    case MFC_Cmd:
        // Issue DMA command
        if (s.mfcQueueLen < SPU_MFC_QUEUE) {
            MFCCommand& cmd = s.mfcQueue[s.mfcQueueLen++];
            cmd.lsa = s.mfc_lsa;
            cmd.ea = s.mfc_eal;
            cmd.size = s.mfc_size;
            cmd.tag = s.mfc_tag;
            cmd.cmd = val;
            cmd.active = 1;
        }
        // Execute DMA immediately (synchronous for now)
        mfc_execute(s, ls, mainMem);
        break;
    case MFC_WrTagMask:    s.mfc_tagMask = val; break;
    case MFC_WrTagUpdate:  break; // instant completion
    case SPU_WrOutMbox:
        s.outMbox = val;
        s.outMboxValid = 1;
        if (mailboxOut) atomicExch((uint32_t*)mailboxOut, val);
        break;
    case SPU_WrOutIntrMbox:
        s.outIntrMbox = val;
        s.outIntrMboxValid = 1;
        break;
    case SPU_WrEventMask:  s.eventMask = val; break;
    case SPU_WrEventAck:   s.eventStat &= ~val; break;
    case SPU_WrDec:        s.decrementer = val; break;
    case SPU_WrSRR0:       s.srr0 = val; break;
    default: break;
    }
}

// ═══════════════════════════════════════════════════════════════
// SHUFB — the most important SPU instruction
// 16-byte shuffle controlled by a pattern in rC
// ═══════════════════════════════════════════════════════════════

__device__ static QWord do_shufb(const QWord& a, const QWord& b, const QWord& c) {
    QWord result;
    for (int i = 0; i < 16; i++) {
        uint8_t sel = c.u8[i];
        if (sel & 0x80) {
            // Special patterns
            if ((sel & 0xE0) == 0xC0)      result.u8[i] = 0x00;
            else if ((sel & 0xE0) == 0xE0) result.u8[i] = 0xFF;
            else                            result.u8[i] = 0x80;
        } else {
            uint8_t idx = sel & 0x1F;
            if (idx < 16) result.u8[i] = a.u8[idx];
            else          result.u8[i] = b.u8[idx - 16];
        }
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════
// Single-Step Execute — decode and run one SPU instruction
// ═══════════════════════════════════════════════════════════════

__device__ static int spuExecOne(SPUState& s, uint8_t* ls, uint8_t* mainMem,
                                  volatile uint32_t* mailboxOut) {
    if (s.halted) return 1;
    uint32_t inst = ls_fetch_inst(ls, s.pc);
    s.npc = (s.pc + 4) & (SPU_LS_SIZE - 1);

    // Decode: SPU ISA priority order — narrower opcodes first
    // RI8 (10-bit) must be checked BEFORE RR (11-bit) to avoid collision
    // between CFLTS/CSFLT/CUFLT/CFLTU and ROTQBI/ROTQBY
    uint32_t o11 = spu_op11(inst);
    { // RI8 (10-bit opcode) — float<->int conversions
        uint32_t o10 = (inst >> 22) & 0x3FF;
        uint32_t rT_  = inst & 0x7F;
        uint32_t rA_  = (inst >> 7) & 0x7F;
        uint32_t i8   = (inst >> 14) & 0xFF;
        switch (o10) {
        case 0xED: { // CSFLT: convert signed int to float
            float div = (i8 < 32) ? (float)(1u << i8) : 4294967296.0f;
            for (int i = 0; i < 4; i++) s.gpr[rT_].f32[i] = (float)s.gpr[rA_].s32[i] / div;
            goto done;
        }
        case 0xEC: { // CFLTS: convert float to signed int
            float mul = (i8 < 32) ? (float)(1u << i8) : 4294967296.0f;
            for (int i = 0; i < 4; i++) {
                float v = s.gpr[rA_].f32[i] * mul;
                if (v >= 2147483648.0f) s.gpr[rT_].s32[i] = 0x7FFFFFFF;
                else if (v < -2147483648.0f) s.gpr[rT_].s32[i] = (int32_t)0x80000000;
                else s.gpr[rT_].s32[i] = (int32_t)v;
            }
            goto done;
        }
        case 0xEE: { // CUFLT: convert unsigned int to float
            float div = (i8 < 32) ? (float)(1u << i8) : 4294967296.0f;
            for (int i = 0; i < 4; i++) s.gpr[rT_].f32[i] = (float)s.gpr[rA_].u32[i] / div;
            goto done;
        }
        case 0xEF: { // CFLTU: convert float to unsigned int
            float mul = (i8 < 32) ? (float)(1u << i8) : 4294967296.0f;
            for (int i = 0; i < 4; i++) {
                float v = s.gpr[rA_].f32[i] * mul;
                if (v >= 4294967296.0f) s.gpr[rT_].u32[i] = 0xFFFFFFFF;
                else if (v < 0.0f) s.gpr[rT_].u32[i] = 0;
                else s.gpr[rT_].u32[i] = (uint32_t)v;
            }
            goto done;
        }
        default: break;
        }
    }
    { // RR (11-bit)
        uint32_t rT=spu_rT_rr(inst),rA=spu_rA_rr(inst),rB=spu_rB_rr(inst);
        switch(o11){
        case op11::A:    { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rA].u32[i]+s.gpr[rB].u32[i]; goto done; }
        case op11::AH:   { for(int i=0;i<8;i++) s.gpr[rT].u16[i]=s.gpr[rA].u16[i]+s.gpr[rB].u16[i]; goto done; }
        case op11::SF:   { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rB].u32[i]-s.gpr[rA].u32[i]; goto done; }
        case op11::SFH:  { for(int i=0;i<8;i++) s.gpr[rT].u16[i]=s.gpr[rB].u16[i]-s.gpr[rA].u16[i]; goto done; }
        case op11::CG:   { for(int i=0;i<4;i++){uint64_t sm=(uint64_t)s.gpr[rA].u32[i]+(uint64_t)s.gpr[rB].u32[i];s.gpr[rT].u32[i]=(uint32_t)(sm>>32);} goto done; }
        case op11::BG:   { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rA].u32[i]<=s.gpr[rB].u32[i])?1:0; goto done; }
        case op11::ADDX: { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rA].u32[i]+s.gpr[rB].u32[i]+(s.gpr[rT].u32[i]&1); goto done; }
        case op11::SFX:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rB].u32[i]-s.gpr[rA].u32[i]-(1-(s.gpr[rT].u32[i]&1)); goto done; }
        case op11::MPY:  { for(int i=0;i<4;i++){int16_t a=(int16_t)(s.gpr[rA].u32[i]&0xFFFF),b=(int16_t)(s.gpr[rB].u32[i]&0xFFFF);s.gpr[rT].s32[i]=(int32_t)a*(int32_t)b;} goto done; }
        case op11::MPYU: { for(int i=0;i<4;i++){uint16_t a=(uint16_t)(s.gpr[rA].u32[i]&0xFFFF),b=(uint16_t)(s.gpr[rB].u32[i]&0xFFFF);s.gpr[rT].u32[i]=(uint32_t)a*(uint32_t)b;} goto done; }
        case op11::MPYH: { for(int i=0;i<4;i++){uint16_t a=(uint16_t)(s.gpr[rA].u32[i]>>16),b=(uint16_t)(s.gpr[rB].u32[i]&0xFFFF);s.gpr[rT].u32[i]=((uint32_t)a*(uint32_t)b)<<16;} goto done; }
        case op11::MPYS: { for(int i=0;i<4;i++){int16_t a=(int16_t)(s.gpr[rA].u32[i]&0xFFFF),b=(int16_t)(s.gpr[rB].u32[i]&0xFFFF);s.gpr[rT].s32[i]=((int32_t)a*(int32_t)b)>>16;} goto done; }
        case op11::CLZ:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rA].u32[i]?__clz(s.gpr[rA].u32[i]):32; goto done; }
        case op11::CNTB: { for(int i=0;i<16;i++) s.gpr[rT].u8[i]=__popc(s.gpr[rA].u8[i]); goto done; }
        case op11::AVGB: { for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(uint8_t)(((uint32_t)s.gpr[rA].u8[i]+s.gpr[rB].u8[i]+1)>>1); goto done; }
        case op11::ABSDB:{ for(int i=0;i<16;i++){int d=(int)s.gpr[rA].u8[i]-(int)s.gpr[rB].u8[i];s.gpr[rT].u8[i]=(uint8_t)(d<0?-d:d);} goto done; }
        case op11::AND:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rA].u32[i]&s.gpr[rB].u32[i]; goto done; }
        case op11::OR:   { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rA].u32[i]|s.gpr[rB].u32[i]; goto done; }
        case op11::XOR:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rA].u32[i]^s.gpr[rB].u32[i]; goto done; }
        case op11::ANDC: { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rA].u32[i]&~s.gpr[rB].u32[i]; goto done; }
        case op11::ORC:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=s.gpr[rA].u32[i]|~s.gpr[rB].u32[i]; goto done; }
        case op11::NAND: { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=~(s.gpr[rA].u32[i]&s.gpr[rB].u32[i]); goto done; }
        case op11::NOR:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=~(s.gpr[rA].u32[i]|s.gpr[rB].u32[i]); goto done; }
        case op11::EQV:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=~(s.gpr[rA].u32[i]^s.gpr[rB].u32[i]); goto done; }
        case op11::SHL:  { for(int i=0;i<4;i++){uint32_t sh=s.gpr[rB].u32[i]&0x3F;s.gpr[rT].u32[i]=(sh<32)?(s.gpr[rA].u32[i]<<sh):0;} goto done; }
        case op11::SHLH: { for(int i=0;i<8;i++){uint32_t sh=s.gpr[rB].u16[i]&0x1F;s.gpr[rT].u16[i]=(sh<16)?(s.gpr[rA].u16[i]<<sh):0;} goto done; }
        case op11::ROT:  { for(int i=0;i<4;i++){uint32_t sh=s.gpr[rB].u32[i]&0x1F;uint32_t v=s.gpr[rA].u32[i];s.gpr[rT].u32[i]=(v<<sh)|(v>>(32-sh));} goto done; }
        case op11::ROTH: { for(int i=0;i<8;i++){uint32_t sh=s.gpr[rB].u16[i]&0xF;uint16_t v=s.gpr[rA].u16[i];s.gpr[rT].u16[i]=(v<<sh)|(v>>(16-sh));} goto done; }
        case op11::ROTM: { for(int i=0;i<4;i++){uint32_t sh=(-(int32_t)s.gpr[rB].u32[i])&0x3F;s.gpr[rT].u32[i]=(sh<32)?(s.gpr[rA].u32[i]>>sh):0;} goto done; }
        case op11::ROTHM:{ for(int i=0;i<8;i++){uint32_t sh=(-(int32_t)(int16_t)s.gpr[rB].u16[i])&0x1F;s.gpr[rT].u16[i]=(sh<16)?(s.gpr[rA].u16[i]>>sh):0;} goto done; }
        case op11::ROTMA:{ for(int i=0;i<4;i++){uint32_t sh=(-(int32_t)s.gpr[rB].u32[i])&0x3F;int32_t v=s.gpr[rA].s32[i];s.gpr[rT].s32[i]=(sh<32)?(v>>sh):(v>>31);} goto done; }
        case op11::ROTMAH:{ for(int i=0;i<8;i++){uint32_t sh=(-(int32_t)(int16_t)s.gpr[rB].u16[i])&0x1F;int16_t v=s.gpr[rA].s16[i];s.gpr[rT].s16[i]=(sh<16)?(v>>sh):(v>>15);} goto done; }
        case op11::ROTQBY:{ uint32_t sh=s.gpr[rB].u32[0]&0xF;QWord tmp=s.gpr[rA];for(int i=0;i<16;i++) s.gpr[rT].u8[i]=tmp.u8[(i+sh)&0xF]; goto done; }
        case op11::ROTQBI:{ uint32_t sh=s.gpr[rB].u32[0]&0x7;if(!sh){s.gpr[rT]=s.gpr[rA];goto done;}QWord a=s.gpr[rA];s.gpr[rT].u32[0]=(a.u32[0]<<sh)|(a.u32[1]>>(32-sh));s.gpr[rT].u32[1]=(a.u32[1]<<sh)|(a.u32[2]>>(32-sh));s.gpr[rT].u32[2]=(a.u32[2]<<sh)|(a.u32[3]>>(32-sh));s.gpr[rT].u32[3]=(a.u32[3]<<sh)|(a.u32[0]>>(32-sh)); goto done; }
        case op11::SHLQBY:{ uint32_t sh=s.gpr[rB].u32[0]&0x1F;for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(i+sh<16)?s.gpr[rA].u8[i+sh]:0; goto done; }
        case op11::SHLQBI:{ uint32_t sh=s.gpr[rB].u32[0]&0x7;if(!sh){s.gpr[rT]=s.gpr[rA];goto done;}QWord a=s.gpr[rA];s.gpr[rT].u32[0]=(a.u32[0]<<sh)|(a.u32[1]>>(32-sh));s.gpr[rT].u32[1]=(a.u32[1]<<sh)|(a.u32[2]>>(32-sh));s.gpr[rT].u32[2]=(a.u32[2]<<sh)|(a.u32[3]>>(32-sh));s.gpr[rT].u32[3]=(a.u32[3]<<sh); goto done; }
        case op11::ROTQMBY:{ uint32_t sh=(-(int32_t)s.gpr[rB].u32[0])&0x1F;for(int i=0;i<16;i++){int src=i-(int)sh;s.gpr[rT].u8[i]=(src>=0)?s.gpr[rA].u8[src]:0;} goto done; }
        case op11::ROTQMBI:{ uint32_t sh=(-(int32_t)s.gpr[rB].u32[0])&0x7;if(!sh){s.gpr[rT]=s.gpr[rA];goto done;}QWord a=s.gpr[rA];s.gpr[rT].u32[0]=(a.u32[0]>>sh);s.gpr[rT].u32[1]=(a.u32[1]>>sh)|(a.u32[0]<<(32-sh));s.gpr[rT].u32[2]=(a.u32[2]>>sh)|(a.u32[1]<<(32-sh));s.gpr[rT].u32[3]=(a.u32[3]>>sh)|(a.u32[2]<<(32-sh)); goto done; }
        case op11::CEQ:   { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rA].u32[i]==s.gpr[rB].u32[i])?0xFFFFFFFF:0; goto done; }
        case op11::CEQH:  { for(int i=0;i<8;i++) s.gpr[rT].u16[i]=(s.gpr[rA].u16[i]==s.gpr[rB].u16[i])?0xFFFF:0; goto done; }
        case op11::CEQB:  { for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(s.gpr[rA].u8[i]==s.gpr[rB].u8[i])?0xFF:0; goto done; }
        case op11::CGT:   { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rA].s32[i]>s.gpr[rB].s32[i])?0xFFFFFFFF:0; goto done; }
        case op11::CGTH:  { for(int i=0;i<8;i++) s.gpr[rT].u16[i]=(s.gpr[rA].s16[i]>s.gpr[rB].s16[i])?0xFFFF:0; goto done; }
        case op11::CGTB:  { for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(s.gpr[rA].s8[i]>s.gpr[rB].s8[i])?0xFF:0; goto done; }
        case op11::CLGT:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rA].u32[i]>s.gpr[rB].u32[i])?0xFFFFFFFF:0; goto done; }
        case op11::CLGTH: { for(int i=0;i<8;i++) s.gpr[rT].u16[i]=(s.gpr[rA].u16[i]>s.gpr[rB].u16[i])?0xFFFF:0; goto done; }
        case op11::CLGTB: { for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(s.gpr[rA].u8[i]>s.gpr[rB].u8[i])?0xFF:0; goto done; }
        case op11::FA:  { for(int i=0;i<4;i++) s.gpr[rT].f32[i]=s.gpr[rA].f32[i]+s.gpr[rB].f32[i]; goto done; }
        case op11::FS:  { for(int i=0;i<4;i++) s.gpr[rT].f32[i]=s.gpr[rA].f32[i]-s.gpr[rB].f32[i]; goto done; }
        case op11::FM:  { for(int i=0;i<4;i++) s.gpr[rT].f32[i]=s.gpr[rA].f32[i]*s.gpr[rB].f32[i]; goto done; }
        case op11::FCEQ: { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rA].f32[i]==s.gpr[rB].f32[i])?0xFFFFFFFF:0; goto done; }
        case op11::FCGT: { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rA].f32[i]>s.gpr[rB].f32[i])?0xFFFFFFFF:0; goto done; }
        case op11::FCMEQ:{ for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(fabsf(s.gpr[rA].f32[i])==fabsf(s.gpr[rB].f32[i]))?0xFFFFFFFF:0; goto done; }
        case op11::FCMGT:{ for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(fabsf(s.gpr[rA].f32[i])>fabsf(s.gpr[rB].f32[i]))?0xFFFFFFFF:0; goto done; }
        case op11::DFA: { for(int i=0;i<2;i++) s.gpr[rT].f64[i]=s.gpr[rA].f64[i]+s.gpr[rB].f64[i]; goto done; }
        case op11::DFS: { for(int i=0;i<2;i++) s.gpr[rT].f64[i]=s.gpr[rA].f64[i]-s.gpr[rB].f64[i]; goto done; }
        case op11::DFM: { for(int i=0;i<2;i++) s.gpr[rT].f64[i]=s.gpr[rA].f64[i]*s.gpr[rB].f64[i]; goto done; }
        case op11::BI:   { s.npc=s.gpr[rA].u32[0]&(SPU_LS_SIZE-1)&~0x3; goto done; }
        case op11::BISL: { s.gpr[rT].u32[0]=s.pc+4;s.gpr[rT].u32[1]=s.gpr[rT].u32[2]=s.gpr[rT].u32[3]=0;s.npc=s.gpr[rA].u32[0]&(SPU_LS_SIZE-1)&~0x3; goto done; }
        case op11::BIZ:  { if(s.gpr[rT].u32[0]==0) s.npc=s.gpr[rA].u32[0]&(SPU_LS_SIZE-1)&~0x3; goto done; }
        case op11::BINZ: { if(s.gpr[rT].u32[0]!=0) s.npc=s.gpr[rA].u32[0]&(SPU_LS_SIZE-1)&~0x3; goto done; }
        case op11::RDCH:  { s.gpr[rT].u32[0]=channel_read(s,rA);s.gpr[rT].u32[1]=s.gpr[rT].u32[2]=s.gpr[rT].u32[3]=0; goto done; }
        case op11::RCHCNT:{ uint32_t cnt=1;if(rA==SPU_RdInMbox)cnt=s.inMboxValid?1:0;else if(rA==SPU_WrOutMbox)cnt=s.outMboxValid?0:1;s.gpr[rT].u32[0]=cnt;s.gpr[rT].u32[1]=s.gpr[rT].u32[2]=s.gpr[rT].u32[3]=0; goto done; }
        case op11::WRCH:  { channel_write(s,rA,s.gpr[rT].u32[0],ls,mainMem,mailboxOut); goto done; }
        // Sign extension
        case op11::XSBH:  { for(int i=0;i<16;i+=2){int8_t b=s.gpr[rA].u8[i+1];s.gpr[rT].u8[i]=(b<0)?0xFF:0x00;s.gpr[rT].u8[i+1]=b;} goto done; }
        case op11::XSHW:  { for(int i=0;i<4;i++){int16_t h=s.gpr[rA].s16[i*2+1];s.gpr[rT].s32[i]=(int32_t)h;} goto done; }
        case op11::XSWD:  { s.gpr[rT].s32[0]=(s.gpr[rA].s32[0]<0)?-1:0;s.gpr[rT].s32[1]=s.gpr[rA].s32[0];s.gpr[rT].s32[2]=(s.gpr[rA].s32[2]<0)?-1:0;s.gpr[rT].s32[3]=s.gpr[rA].s32[2]; goto done; }
        // Load/Store indexed
        case op11::LQX:   { uint32_t ea=(s.gpr[rA].u32[0]+s.gpr[rB].u32[0])&(SPU_LS_SIZE-1)&~0xF;s.gpr[rT]=ls_read_qw(ls,ea); goto done; }
        case op11::STQX:  { uint32_t ea=(s.gpr[rA].u32[0]+s.gpr[rB].u32[0])&(SPU_LS_SIZE-1)&~0xF;ls_write_qw(ls,ea,s.gpr[rT]); goto done; }
        // Floating interpolate (rare, treat as NOP — result in rT stays)
        case op11::FI:    { goto done; }
        // Multiply high-high
        case op11::MPYHH:  { for(int i=0;i<4;i++){int16_t a=(int16_t)(s.gpr[rA].u32[i]>>16),b=(int16_t)(s.gpr[rB].u32[i]>>16);s.gpr[rT].s32[i]=(int32_t)a*(int32_t)b;} goto done; }
        case op11::MPYHHU: { for(int i=0;i<4;i++){uint16_t a=(uint16_t)(s.gpr[rA].u32[i]>>16),b=(uint16_t)(s.gpr[rB].u32[i]>>16);s.gpr[rT].u32[i]=(uint32_t)a*(uint32_t)b;} goto done; }
        // Extended carry/borrow
        case op11::CGX:  { for(int i=0;i<4;i++){uint64_t sm=(uint64_t)s.gpr[rA].u32[i]+(uint64_t)s.gpr[rB].u32[i]+(uint64_t)(s.gpr[rT].u32[i]&1);s.gpr[rT].u32[i]=(uint32_t)(sm>>32);} goto done; }
        case op11::BGX:  { for(int i=0;i<4;i++){int64_t d=(int64_t)(uint64_t)s.gpr[rB].u32[i]-(int64_t)(uint64_t)s.gpr[rA].u32[i]-(int64_t)(1-(s.gpr[rT].u32[i]&1));s.gpr[rT].u32[i]=(d>=0)?1:0;} goto done; }
        // Branch indirect halfword variants
        case op11::BIHZ:  { if((s.gpr[rT].u32[0]&0xFFFF)==0) s.npc=s.gpr[rA].u32[0]&(SPU_LS_SIZE-1)&~0x3; goto done; }
        case op11::BIHNZ: { if((s.gpr[rT].u32[0]&0xFFFF)!=0) s.npc=s.gpr[rA].u32[0]&(SPU_LS_SIZE-1)&~0x3; goto done; }
        // Quadword shift/rotate with byte+bit count
        case op11::SHLQBYBI:{ uint32_t sh=(s.gpr[rB].u32[0]>>3)&0x1F;for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(i+sh<16)?s.gpr[rA].u8[i+sh]:0; goto done; }
        case op11::ROTQBYBI:{ uint32_t sh=(s.gpr[rB].u32[0]>>3)&0xF;QWord tmp=s.gpr[rA];for(int i=0;i<16;i++) s.gpr[rT].u8[i]=tmp.u8[(i+sh)&0xF]; goto done; }
        case op11::ROTQMBYBI:{ uint32_t sh=((-(int32_t)(s.gpr[rB].u32[0]>>3)))&0x1F;for(int i=0;i<16;i++){int src=i-(int)sh;s.gpr[rT].u8[i]=(src>=0)?s.gpr[rA].u8[src]:0;} goto done; }
        case op11::NOP: case op11::LNOP: case op11::SYNC: case op11::DSYNC: goto done;
        case op11::STOP: case op11::STOPD: s.halted=1; return 1;
        default: break;
        }
    }
    { // RI16 (9-bit)
        uint32_t o9=spu_op9(inst),rT=spu_rT_ri16(inst);int32_t i16=spu_I16(inst);uint32_t i16u=spu_I16u(inst);
        switch(o9){
        case op9::IL:    { for(int i=0;i<4;i++) s.gpr[rT].s32[i]=i16; goto done; }
        case op9::ILH:   { for(int i=0;i<8;i++) s.gpr[rT].u16[i]=(uint16_t)i16u; goto done; }
        case op9::ILHU:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=i16u<<16; goto done; }
        case op9::IOHL:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]|=i16u; goto done; }
        case op9::FSMBI: { for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(i16u&(1<<(15-i)))?0xFF:0x00; goto done; }
        case op9::BR:    { s.npc=(s.pc+(i16<<2))&(SPU_LS_SIZE-1); goto done; }
        case op9::BRA:   { s.npc=((uint32_t)(i16<<2))&(SPU_LS_SIZE-1); goto done; }
        case op9::BRSL:  { s.gpr[rT].u32[0]=s.pc+4;s.gpr[rT].u32[1]=s.gpr[rT].u32[2]=s.gpr[rT].u32[3]=0;s.npc=(s.pc+(i16<<2))&(SPU_LS_SIZE-1); goto done; }
        case op9::BRASL: { s.gpr[rT].u32[0]=s.pc+4;s.gpr[rT].u32[1]=s.gpr[rT].u32[2]=s.gpr[rT].u32[3]=0;s.npc=((uint32_t)(i16<<2))&(SPU_LS_SIZE-1); goto done; }
        case op9::BRZ:   { if(s.gpr[rT].u32[0]==0) s.npc=(s.pc+(i16<<2))&(SPU_LS_SIZE-1); goto done; }
        case op9::BRNZ:  { if(s.gpr[rT].u32[0]!=0) s.npc=(s.pc+(i16<<2))&(SPU_LS_SIZE-1); goto done; }
        case op9::BRHZ:  { if((s.gpr[rT].u32[0]&0xFFFF)==0) s.npc=(s.pc+(i16<<2))&(SPU_LS_SIZE-1); goto done; }
        case op9::BRHNZ: { if((s.gpr[rT].u32[0]&0xFFFF)!=0) s.npc=(s.pc+(i16<<2))&(SPU_LS_SIZE-1); goto done; }
        case op9::LQA:   { uint32_t ea=((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)&~0xF;s.gpr[rT]=ls_read_qw(ls,ea); goto done; }
        case op9::STQA:  { uint32_t ea=((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)&~0xF;ls_write_qw(ls,ea,s.gpr[rT]); goto done; }
        case op9::LQR:   { uint32_t ea=(s.pc+(i16<<2))&(SPU_LS_SIZE-1)&~0xF;s.gpr[rT]=ls_read_qw(ls,ea); goto done; }
        case op9::STQR:  { uint32_t ea=(s.pc+(i16<<2))&(SPU_LS_SIZE-1)&~0xF;ls_write_qw(ls,ea,s.gpr[rT]); goto done; }
        default: break;
        }
    }
    { // RI10 (8-bit)
        uint32_t o8=spu_op8(inst),rT=spu_rT_ri10(inst),rA=spu_rA_ri10(inst);int32_t i10=spu_I10(inst);
        switch(o8){
        case op8::AI:    { for(int i=0;i<4;i++) s.gpr[rT].s32[i]=s.gpr[rA].s32[i]+i10; goto done; }
        case op8::AHI:   { int16_t imm=(int16_t)(i10&0xFFFF);for(int i=0;i<8;i++) s.gpr[rT].s16[i]=s.gpr[rA].s16[i]+imm; goto done; }
        case op8::SFI:   { for(int i=0;i<4;i++) s.gpr[rT].s32[i]=i10-s.gpr[rA].s32[i]; goto done; }
        case op8::SFHI:  { int16_t imm=(int16_t)(i10&0xFFFF);for(int i=0;i<8;i++) s.gpr[rT].s16[i]=imm-s.gpr[rA].s16[i]; goto done; }
        case op8::ANDI:  { for(int i=0;i<4;i++) s.gpr[rT].s32[i]=s.gpr[rA].s32[i]&i10; goto done; }
        case op8::ORI:   { for(int i=0;i<4;i++) s.gpr[rT].s32[i]=s.gpr[rA].s32[i]|i10; goto done; }
        case op8::XORI:  { for(int i=0;i<4;i++) s.gpr[rT].s32[i]=s.gpr[rA].s32[i]^i10; goto done; }
        case op8::CEQI:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rA].s32[i]==i10)?0xFFFFFFFF:0; goto done; }
        case op8::CEQHI: { int16_t imm=(int16_t)(i10&0xFFFF);for(int i=0;i<8;i++) s.gpr[rT].u16[i]=(s.gpr[rA].s16[i]==imm)?0xFFFF:0; goto done; }
        case op8::CEQBI: { int8_t imm=(int8_t)(i10&0xFF);for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(s.gpr[rA].s8[i]==imm)?0xFF:0; goto done; }
        case op8::CGTI:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rA].s32[i]>i10)?0xFFFFFFFF:0; goto done; }
        case op8::CGTHI: { int16_t imm=(int16_t)(i10&0xFFFF);for(int i=0;i<8;i++) s.gpr[rT].u16[i]=(s.gpr[rA].s16[i]>imm)?0xFFFF:0; goto done; }
        case op8::CGTBI: { int8_t imm=(int8_t)(i10&0xFF);for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(s.gpr[rA].s8[i]>imm)?0xFF:0; goto done; }
        case op8::CLGTI: { uint32_t uimm=(uint32_t)i10;for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rA].u32[i]>uimm)?0xFFFFFFFF:0; goto done; }
        case op8::CLGTHI:{ uint16_t uimm=(uint16_t)(i10&0xFFFF);for(int i=0;i<8;i++) s.gpr[rT].u16[i]=(s.gpr[rA].u16[i]>uimm)?0xFFFF:0; goto done; }
        case op8::CLGTBI:{ uint8_t uimm=(uint8_t)(i10&0xFF);for(int i=0;i<16;i++) s.gpr[rT].u8[i]=(s.gpr[rA].u8[i]>uimm)?0xFF:0; goto done; }
        case op8::LQD:   { uint32_t ea=((s.gpr[rA].u32[0]+(i10<<4))&(SPU_LS_SIZE-1))&~0xF;s.gpr[rT]=ls_read_qw(ls,ea); goto done; }
        case op8::STQD:  { uint32_t ea=((s.gpr[rA].u32[0]+(i10<<4))&(SPU_LS_SIZE-1))&~0xF;ls_write_qw(ls,ea,s.gpr[rT]); goto done; }
        case op8::MPYI:  { for(int i=0;i<4;i++){int16_t a=(int16_t)(s.gpr[rA].u32[i]&0xFFFF);s.gpr[rT].s32[i]=(int32_t)a*i10;} goto done; }
        case op8::MPYUI: { uint32_t uimm=(uint32_t)(i10&0xFFFF);for(int i=0;i<4;i++){uint16_t a=(uint16_t)(s.gpr[rA].u32[i]&0xFFFF);s.gpr[rT].u32[i]=(uint32_t)a*uimm;} goto done; }
        default: break;
        }
    }
    { // RI18 (7-bit)
        uint32_t o7=spu_op7(inst);
        if(o7==op7::ILA){ uint32_t rT=spu_rT_ri18(inst),imm=spu_I18(inst);for(int i=0;i<4;i++) s.gpr[rT].u32[i]=imm; goto done; }
    }
    { // RRR (4-bit) — LAST
        uint32_t o4=spu_op4(inst),rT=spu_rT_rrr(inst),rA=spu_rA_rrr(inst),rB=spu_rB_rrr(inst),rC=spu_rC_rrr(inst);
        switch(o4){
        case op4::SELB:  { for(int i=0;i<4;i++) s.gpr[rT].u32[i]=(s.gpr[rB].u32[i]&s.gpr[rC].u32[i])|(s.gpr[rA].u32[i]&~s.gpr[rC].u32[i]); goto done; }
        case op4::SHUFB: { s.gpr[rT]=do_shufb(s.gpr[rA],s.gpr[rB],s.gpr[rC]); goto done; }
        case op4::FMA:   { for(int i=0;i<4;i++) s.gpr[rT].f32[i]=s.gpr[rA].f32[i]*s.gpr[rB].f32[i]+s.gpr[rC].f32[i]; goto done; }
        case op4::FMS:   { for(int i=0;i<4;i++) s.gpr[rT].f32[i]=s.gpr[rA].f32[i]*s.gpr[rB].f32[i]-s.gpr[rC].f32[i]; goto done; }
        case op4::FNMS:  { for(int i=0;i<4;i++) s.gpr[rT].f32[i]=s.gpr[rC].f32[i]-s.gpr[rA].f32[i]*s.gpr[rB].f32[i]; goto done; }
        case op4::MPYA:  { for(int i=0;i<4;i++){int16_t a=(int16_t)(s.gpr[rA].u32[i]&0xFFFF),b=(int16_t)(s.gpr[rB].u32[i]&0xFFFF);s.gpr[rT].s32[i]=(int32_t)a*(int32_t)b+s.gpr[rC].s32[i];} goto done; }
        case op4::DFMA:  { for(int i=0;i<2;i++) s.gpr[rT].f64[i]=s.gpr[rA].f64[i]*s.gpr[rB].f64[i]+s.gpr[rC].f64[i]; goto done; }
        case op4::DFMS:  { for(int i=0;i<2;i++) s.gpr[rT].f64[i]=s.gpr[rA].f64[i]*s.gpr[rB].f64[i]-s.gpr[rC].f64[i]; goto done; }
        case op4::DFNMS: { for(int i=0;i<2;i++) s.gpr[rT].f64[i]=s.gpr[rC].f64[i]-s.gpr[rA].f64[i]*s.gpr[rB].f64[i]; goto done; }
        default: break;
        }
    }
    return 2; // unimplemented
done:
    s.pc = s.npc; s.cycles++; if(s.decrementer>0) s.decrementer--;
    return 0;
}


// ═══════════════════════════════════════════════════════════════
// SPU Megakernel — one thread per SPU (6 threads total)
// ═══════════════════════════════════════════════════════════════

__global__ void spuMegakernel(SPUState* states, uint8_t** localStores,
                               uint8_t* mainMem, uint32_t maxCycles,
                               volatile uint32_t* mailboxes,
                               volatile uint32_t* status) {
    int spuId = blockIdx.x * blockDim.x + threadIdx.x;
    if (spuId >= 6) return;

    SPUState& s = states[spuId];
    uint8_t* ls = localStores[spuId];

    if (s.halted) {
        if (status) atomicExch((uint32_t*)&status[spuId], 2);
        return;
    }

    volatile uint32_t* myMailbox = mailboxes ? &mailboxes[spuId] : nullptr;

    for (uint32_t cycle = 0; cycle < maxCycles && !s.halted; cycle++) {
        int result = spuExecOne(s, ls, mainMem, myMailbox);
        if (result == 1) break;
        if (result == 2) {
            // Unknown instruction — skip it
            s.pc = (s.pc + 4) & (SPU_LS_SIZE - 1);
        }
    }

    if (status) atomicExch((uint32_t*)&status[spuId], s.halted ? 2 : 1);
}

// ═══════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════

extern "C" {

struct SPUContext {
    SPUState*      d_states;         // 6 SPU states on GPU
    uint8_t*       d_localStores[6]; // 6 × 256KB Local Stores
    uint8_t**      d_lsPtrs;         // Device-side pointer array
    volatile uint32_t* d_mailboxes;  // 6 mailbox slots
    volatile uint32_t* d_status;     // 6 status words
    cudaStream_t   stream;
    bool           ready;
};

static SPUContext g_spu = {};

int spu_init() {
    if (g_spu.ready) return 1;

    cudaStreamCreate(&g_spu.stream);

    // Allocate 6 SPU states
    cudaMalloc(&g_spu.d_states, 6 * sizeof(SPUState));
    cudaMemset(g_spu.d_states, 0, 6 * sizeof(SPUState));

    // Allocate 6 × 256KB Local Stores
    uint8_t* h_ptrs[6];
    for (int i = 0; i < 6; i++) {
        cudaMalloc(&g_spu.d_localStores[i], SPU_LS_SIZE);
        cudaMemset(g_spu.d_localStores[i], 0, SPU_LS_SIZE);
        h_ptrs[i] = g_spu.d_localStores[i];
    }

    // Upload pointer array to device
    cudaMalloc(&g_spu.d_lsPtrs, 6 * sizeof(uint8_t*));
    cudaMemcpy(g_spu.d_lsPtrs, h_ptrs, 6 * sizeof(uint8_t*), cudaMemcpyHostToDevice);

    // Mailboxes and status
    cudaMalloc((void**)&g_spu.d_mailboxes, 6 * sizeof(uint32_t));
    cudaMemset((void*)g_spu.d_mailboxes, 0, 6 * sizeof(uint32_t));
    cudaMalloc((void**)&g_spu.d_status, 6 * sizeof(uint32_t));
    cudaMemset((void*)g_spu.d_status, 0, 6 * sizeof(uint32_t));

    g_spu.ready = true;
    fprintf(stderr, "[SPU] 6 SPU cores initialized (6 × 256KB LS = 1.5MB)\n");
    return 1;
}

// Load SPU program into a specific SPU's Local Store
int spu_load_program(int spuId, const void* data, size_t size, uint32_t entryPC) {
    if (!g_spu.ready || spuId < 0 || spuId >= 6) return 0;
    if (size > SPU_LS_SIZE) size = SPU_LS_SIZE;

    cudaMemcpy(g_spu.d_localStores[spuId], data, size, cudaMemcpyHostToDevice);

    // Set entry point
    SPUState init = {};
    init.pc = entryPC & (SPU_LS_SIZE - 1);
    init.spuId = spuId;
    init.gpr[1].u32[0] = SPU_LS_SIZE - 16;  // stack pointer at top of LS
    cudaMemcpy(&g_spu.d_states[spuId], &init, sizeof(SPUState), cudaMemcpyHostToDevice);

    fprintf(stderr, "[SPU%d] Loaded %zu bytes, entry=0x%x\n", spuId, size, entryPC);
    return 1;
}

// Run all 6 SPUs for N cycles
float spu_run(uint32_t maxCycles, uint8_t* d_mainMem) {
    if (!g_spu.ready) return -1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemsetAsync((void*)g_spu.d_status, 0, 6 * sizeof(uint32_t), g_spu.stream);

    cudaEventRecord(start, g_spu.stream);

    // Launch 6 threads (one per SPU) in a single block
    spuMegakernel<<<1, 6, 0, g_spu.stream>>>(
        g_spu.d_states, g_spu.d_lsPtrs, d_mainMem, maxCycles,
        g_spu.d_mailboxes, g_spu.d_status);

    cudaEventRecord(stop, g_spu.stream);
    cudaStreamSynchronize(g_spu.stream);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Read SPU state for debugging
int spu_read_state(int spuId, SPUState* out) {
    if (!g_spu.ready || spuId < 0 || spuId >= 6 || !out) return 0;
    cudaMemcpy(out, &g_spu.d_states[spuId], sizeof(SPUState), cudaMemcpyDeviceToHost);
    return 1;
}

// Write to SPU inbound mailbox (PPE → SPU)
int spu_write_mailbox(int spuId, uint32_t value) {
    if (!g_spu.ready || spuId < 0 || spuId >= 6) return 0;
    SPUState patch;
    cudaMemcpy(&patch, &g_spu.d_states[spuId], sizeof(SPUState), cudaMemcpyDeviceToHost);
    patch.inMbox = value;
    patch.inMboxValid = 1;
    cudaMemcpy(&g_spu.d_states[spuId], &patch, sizeof(SPUState), cudaMemcpyHostToDevice);
    return 1;
}

// Read SPU outbound mailbox
uint32_t spu_read_mailbox(int spuId) {
    if (!g_spu.ready || spuId < 0 || spuId >= 6) return 0;
    uint32_t val = 0;
    cudaMemcpy(&val, (void*)&g_spu.d_mailboxes[spuId], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return val;
}

void spu_shutdown() {
    if (!g_spu.ready) return;
    cudaFree(g_spu.d_states);
    for (int i = 0; i < 6; i++) cudaFree(g_spu.d_localStores[i]);
    cudaFree(g_spu.d_lsPtrs);
    cudaFree((void*)g_spu.d_mailboxes);
    cudaFree((void*)g_spu.d_status);
    cudaStreamDestroy(g_spu.stream);
    g_spu.ready = false;
    fprintf(stderr, "[SPU] Shutdown\n");
}

} // extern "C"
