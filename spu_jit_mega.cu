// spu_jit_mega.cu — Persistent SPU JIT Megakernel
//
// The correct JIT architecture: scan entire LS, compile ALL basic blocks
// into a single CUDA kernel with a dispatch loop. Launch ONCE, run until halt.
//
// Previous approach (spu_jit.cu) launched one kernel per basic block with
// host↔device memcpy round-trips — defeating the whole purpose of GPU execution.
//
// This version:
//   1. Scans LS to discover every basic block (by following all reachable PCs)
//   2. Emits each block as a __device__ inline function
//   3. Emits a dispatch loop: while(!halted) { switch(pc) { case 0x0: ... } }
//   4. Compiles everything into ONE cuFunction via NVRTC
//   5. Launches ONCE on GPU — stays resident until SPU halts
//
// Expected speedup: 50-100× over interpreter (no fetch, no decode, no host trips)
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
#include <vector>
#include <set>
#include <queue>

using namespace spu;

// ═══════════════════════════════════════════════════════════════
// Helpers (duplicated from spu_jit.cu to keep this self-contained)
// ═══════════════════════════════════════════════════════════════

static uint32_t bswap32_host(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

static uint32_t fetch_inst_host(const uint8_t* ls, uint32_t pc) {
    pc &= (SPU_LS_SIZE - 1) & ~0x3;
    uint32_t raw;
    memcpy(&raw, ls + pc, 4);
    return bswap32_host(raw);
}

static int emitf(char* buf, size_t bufSize, size_t* pos, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(buf + *pos, bufSize - *pos, fmt, args);
    va_end(args);
    if (written > 0 && (size_t)written < bufSize - *pos) *pos += written;
    return written;
}

// ═══════════════════════════════════════════════════════════════
// Whole-Program Block Discovery
// ═══════════════════════════════════════════════════════════════
//
// BFS from entry point, following all branch targets to discover
// every reachable basic block. Each block is a (startPC, endPC) pair.

struct MegaBlock {
    uint32_t startPC;
    uint32_t endPC;      // PC of last instruction
    uint32_t numInsns;
    bool     endsWithStop;
    bool     endsWithBranch;
    bool     hasMFC;     // contains channel/MFC ops — emit interpreter fallback
};

static void discover_all_blocks(const uint8_t* ls, uint32_t entryPC,
                                 std::vector<MegaBlock>& blocks,
                                 std::set<uint32_t>& blockStarts) {
    std::queue<uint32_t> worklist;
    std::set<uint32_t> visited;
    worklist.push(entryPC & (SPU_LS_SIZE - 1));

    // Phase 1: BFS to discover all block-start PCs
    while (!worklist.empty()) {
        uint32_t pc = worklist.front();
        worklist.pop();

        if (visited.count(pc)) continue;
        visited.insert(pc);
        blockStarts.insert(pc);

        // Scan forward to find branch targets
        uint32_t scanPC = pc;
        for (uint32_t n = 0; n < 512; n++) {
            uint32_t inst = fetch_inst_host(ls, scanPC);
            uint32_t o11 = spu_op11(inst);
            uint32_t o9  = spu_op9(inst);

            bool isBranch = false, isStop = false;

            switch (o11) {
            case op11::BI: case op11::BISL:
            case op11::BIZ: case op11::BINZ:
            case op11::BIHZ: case op11::BIHNZ:
                isBranch = true; break;
            case op11::STOP: case op11::STOPD:
                isStop = true; break;
            default: break;
            }
            if (!isBranch && !isStop) {
                switch (o9) {
                case op9::BR: case op9::BRA: case op9::BRSL: case op9::BRASL:
                case op9::BRZ: case op9::BRNZ: case op9::BRHZ: case op9::BRHNZ:
                    isBranch = true; break;
                default: break;
                }
            }

            if (isStop) break;
            if (isBranch) {
                int32_t i16 = spu_I16(inst);
                switch (o9) {
                case op9::BR: case op9::BRSL:
                    worklist.push((scanPC + (i16 << 2)) & (SPU_LS_SIZE - 1)); break;
                case op9::BRA: case op9::BRASL:
                    worklist.push(((uint32_t)(i16 << 2)) & (SPU_LS_SIZE - 1)); break;
                case op9::BRZ: case op9::BRNZ: case op9::BRHZ: case op9::BRHNZ:
                    worklist.push((scanPC + (i16 << 2)) & (SPU_LS_SIZE - 1));
                    worklist.push((scanPC + 4) & (SPU_LS_SIZE - 1));
                    break;
                default: break;
                }
                if (o11 == op11::BIZ || o11 == op11::BINZ ||
                    o11 == op11::BIHZ || o11 == op11::BIHNZ)
                    worklist.push((scanPC + 4) & (SPU_LS_SIZE - 1));
                // Unconditional branches: enqueue fall-through as potential block too
                if (o9 == op9::BR || o9 == op9::BRA || o11 == op11::BI)
                    {} // no fall-through
                else
                    worklist.push((scanPC + 4) & (SPU_LS_SIZE - 1));
                break;
            }
            scanPC = (scanPC + 4) & (SPU_LS_SIZE - 1);
        }
    }

    // Phase 2: Build blocks — each starts at a blockStart, ends at branch/stop
    // or the instruction BEFORE the next block start
    for (uint32_t startPC : blockStarts) {
        MegaBlock blk = {};
        blk.startPC = startPC;
        uint32_t pc = startPC;
        uint32_t n = 0;

        while (n < 512) {
            uint32_t inst = fetch_inst_host(ls, pc);
            uint32_t o11 = spu_op11(inst);
            uint32_t o9  = spu_op9(inst);

            bool isBranch = false, isStop = false, isMFC = false;
            switch (o11) {
            case op11::BI: case op11::BISL:
            case op11::BIZ: case op11::BINZ:
            case op11::BIHZ: case op11::BIHNZ: isBranch = true; break;
            case op11::STOP: case op11::STOPD: isStop = true; break;
            case op11::RDCH: case op11::WRCH: case op11::RCHCNT: isMFC = true; break;
            default: break;
            }
            if (!isBranch && !isStop) {
                switch (o9) {
                case op9::BR: case op9::BRA: case op9::BRSL: case op9::BRASL:
                case op9::BRZ: case op9::BRNZ: case op9::BRHZ: case op9::BRHNZ:
                    isBranch = true; break;
                default: break;
                }
            }
            if (isMFC) blk.hasMFC = true;
            n++;
            blk.endPC = pc;

            if (isStop) { blk.endsWithStop = true; break; }
            if (isBranch) { blk.endsWithBranch = true; break; }

            uint32_t nextPC = (pc + 4) & (SPU_LS_SIZE - 1);
            // Stop before next block start (don't overlap)
            if (n > 0 && blockStarts.count(nextPC)) break;
            pc = nextPC;
        }
        blk.numInsns = n;
        blocks.push_back(blk);
    }
}

// ═══════════════════════════════════════════════════════════════
// Emit one instruction as CUDA C++ (inline in block function)
// ═══════════════════════════════════════════════════════════════

static void emit_one_insn(const uint8_t* ls, uint32_t pc,
                           char* buf, size_t bufSize, size_t* pos) {
    uint32_t inst = fetch_inst_host(ls, pc);
    uint32_t o11 = spu_op11(inst);

    // RR (11-bit)
    {
        uint32_t rT = spu_rT_rr(inst), rA = spu_rA_rr(inst), rB = spu_rB_rr(inst);
        switch (o11) {
        // Integer ALU
        case op11::A:    emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]+R[%d].u32[i];\n",rT,rA,rB); return;
        case op11::AH:   emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++) R[%d].u16[i]=R[%d].u16[i]+R[%d].u16[i];\n",rT,rA,rB); return;
        case op11::SF:   emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]-R[%d].u32[i];\n",rT,rB,rA); return;
        case op11::SFH:  emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++) R[%d].u16[i]=R[%d].u16[i]-R[%d].u16[i];\n",rT,rB,rA); return;
        case op11::CG:   emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){unsigned long long s=(unsigned long long)R[%d].u32[i]+(unsigned long long)R[%d].u32[i];R[%d].u32[i]=(uint32_t)(s>>32);}\n",rA,rB,rT); return;
        case op11::BG:   emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].u32[i]<=R[%d].u32[i])?1u:0u;\n",rT,rA,rB); return;
        case op11::ADDX: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]+R[%d].u32[i]+(R[%d].u32[i]&1u);\n",rT,rA,rB,rT); return;
        case op11::SFX:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]-R[%d].u32[i]-(1u-(R[%d].u32[i]&1u));\n",rT,rB,rA,rT); return;
        case op11::MPY:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){int16_t _a=(int16_t)(R[%d].u32[i]&0xFFFFu),_b=(int16_t)(R[%d].u32[i]&0xFFFFu);R[%d].s32[i]=(int32_t)_a*(int32_t)_b;}\n",rA,rB,rT); return;
        case op11::MPYU: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){uint16_t _a=(uint16_t)(R[%d].u32[i]&0xFFFFu),_b=(uint16_t)(R[%d].u32[i]&0xFFFFu);R[%d].u32[i]=(uint32_t)_a*(uint32_t)_b;}\n",rA,rB,rT); return;
        case op11::MPYH: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){uint16_t _a=(uint16_t)(R[%d].u32[i]>>16),_b=(uint16_t)(R[%d].u32[i]&0xFFFFu);R[%d].u32[i]=((uint32_t)_a*(uint32_t)_b)<<16;}\n",rA,rB,rT); return;
        case op11::MPYS: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){int16_t _a=(int16_t)(R[%d].u32[i]&0xFFFFu),_b=(int16_t)(R[%d].u32[i]&0xFFFFu);R[%d].s32[i]=((int32_t)_a*(int32_t)_b)>>16;}\n",rA,rB,rT); return;
        case op11::CLZ:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]?__clz(R[%d].u32[i]):32u;\n",rT,rA,rA); return;
        case op11::CNTB: emitf(buf,bufSize,pos,"  for(int i=0;i<16;i++) R[%d].u8[i]=__popc(R[%d].u8[i]);\n",rT,rA); return;
        case op11::AVGB: emitf(buf,bufSize,pos,"  for(int i=0;i<16;i++) R[%d].u8[i]=(uint8_t)(((uint32_t)R[%d].u8[i]+R[%d].u8[i]+1u)>>1);\n",rT,rA,rB); return;
        case op11::ABSDB:emitf(buf,bufSize,pos,"  for(int i=0;i<16;i++){int _d=(int)R[%d].u8[i]-(int)R[%d].u8[i];R[%d].u8[i]=(uint8_t)(_d<0?-_d:_d);}\n",rA,rB,rT); return;

        // Logical
        case op11::AND:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]&R[%d].u32[i];\n",rT,rA,rB); return;
        case op11::OR:   emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]|R[%d].u32[i];\n",rT,rA,rB); return;
        case op11::XOR:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]^R[%d].u32[i];\n",rT,rA,rB); return;
        case op11::ANDC: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]&~R[%d].u32[i];\n",rT,rA,rB); return;
        case op11::ORC:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=R[%d].u32[i]|~R[%d].u32[i];\n",rT,rA,rB); return;
        case op11::NAND: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=~(R[%d].u32[i]&R[%d].u32[i]);\n",rT,rA,rB); return;
        case op11::NOR:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=~(R[%d].u32[i]|R[%d].u32[i]);\n",rT,rA,rB); return;
        case op11::EQV:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=~(R[%d].u32[i]^R[%d].u32[i]);\n",rT,rA,rB); return;

        // Shifts
        case op11::SHL:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){uint32_t sh=R[%d].u32[i]&0x3Fu;R[%d].u32[i]=(sh<32u)?(R[%d].u32[i]<<sh):0u;}\n",rB,rT,rA); return;
        case op11::SHLH: emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++){uint32_t sh=R[%d].u16[i]&0x1Fu;R[%d].u16[i]=(sh<16u)?(R[%d].u16[i]<<sh):0u;}\n",rB,rT,rA); return;
        case op11::ROT:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){uint32_t sh=R[%d].u32[i]&0x1Fu;uint32_t v=R[%d].u32[i];R[%d].u32[i]=(v<<sh)|(v>>(32u-sh));}\n",rB,rA,rT); return;
        case op11::ROTH: emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++){uint32_t sh=R[%d].u16[i]&0xFu;uint16_t v=R[%d].u16[i];R[%d].u16[i]=(v<<sh)|(v>>(16u-sh));}\n",rB,rA,rT); return;
        case op11::ROTM: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){uint32_t sh=(-(int32_t)R[%d].u32[i])&0x3Fu;R[%d].u32[i]=(sh<32u)?(R[%d].u32[i]>>sh):0u;}\n",rB,rT,rA); return;
        case op11::ROTHM:emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++){uint32_t sh=(-(int32_t)(int16_t)R[%d].u16[i])&0x1Fu;R[%d].u16[i]=(sh<16u)?(R[%d].u16[i]>>sh):0u;}\n",rB,rT,rA); return;
        case op11::ROTMA:emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){uint32_t sh=(-(int32_t)R[%d].u32[i])&0x3Fu;int32_t v=R[%d].s32[i];R[%d].s32[i]=(sh<32u)?(v>>sh):(v>>31);}\n",rB,rA,rT); return;
        case op11::ROTMAH:emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++){uint32_t sh=(-(int32_t)(int16_t)R[%d].u16[i])&0x1Fu;int16_t v=R[%d].s16[i];R[%d].s16[i]=(sh<16u)?(v>>sh):(v>>15);}\n",rB,rA,rT); return;

        // Quadword shifts/rotates
        case op11::ROTQBY: emitf(buf,bufSize,pos,"  {uint32_t sh=R[%d].u32[0]&0xFu;QW tmp=R[%d];for(int i=0;i<16;i++) R[%d].u8[i]=tmp.u8[(i+sh)&0xF];}\n",rB,rA,rT); return;
        case op11::ROTQBI: emitf(buf,bufSize,pos,"  {uint32_t sh=R[%d].u32[0]&0x7u;if(sh){QW a=R[%d];R[%d].u32[0]=(a.u32[0]<<sh)|(a.u32[1]>>(32u-sh));R[%d].u32[1]=(a.u32[1]<<sh)|(a.u32[2]>>(32u-sh));R[%d].u32[2]=(a.u32[2]<<sh)|(a.u32[3]>>(32u-sh));R[%d].u32[3]=(a.u32[3]<<sh)|(a.u32[0]>>(32u-sh));}else{R[%d]=R[%d];}}\n",rB,rA,rT,rT,rT,rT,rT,rA); return;
        case op11::SHLQBY: emitf(buf,bufSize,pos,"  {uint32_t sh=R[%d].u32[0]&0x1Fu;for(int i=0;i<16;i++) R[%d].u8[i]=(i+sh<16u)?R[%d].u8[i+sh]:0;}\n",rB,rT,rA); return;
        case op11::SHLQBI: emitf(buf,bufSize,pos,"  {uint32_t sh=R[%d].u32[0]&0x7u;if(sh){QW a=R[%d];R[%d].u32[0]=(a.u32[0]<<sh)|(a.u32[1]>>(32u-sh));R[%d].u32[1]=(a.u32[1]<<sh)|(a.u32[2]>>(32u-sh));R[%d].u32[2]=(a.u32[2]<<sh)|(a.u32[3]>>(32u-sh));R[%d].u32[3]=(a.u32[3]<<sh);}else{R[%d]=R[%d];}}\n",rB,rA,rT,rT,rT,rT,rT,rA); return;
        case op11::ROTQMBY: emitf(buf,bufSize,pos,"  {uint32_t sh=(-(int32_t)R[%d].u32[0])&0x1Fu;for(int i=0;i<16;i++){int src=i-(int)sh;R[%d].u8[i]=(src>=0)?R[%d].u8[src]:0;}}\n",rB,rT,rA); return;
        case op11::ROTQMBI: emitf(buf,bufSize,pos,"  {uint32_t sh=(-(int32_t)R[%d].u32[0])&0x7u;if(sh){QW a=R[%d];R[%d].u32[0]=(a.u32[0]>>sh);R[%d].u32[1]=(a.u32[1]>>sh)|(a.u32[0]<<(32u-sh));R[%d].u32[2]=(a.u32[2]>>sh)|(a.u32[1]<<(32u-sh));R[%d].u32[3]=(a.u32[3]>>sh)|(a.u32[2]<<(32u-sh));}else{R[%d]=R[%d];}}\n",rB,rA,rT,rT,rT,rT,rT,rA); return;

        // Compare
        case op11::CEQ:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].u32[i]==R[%d].u32[i])?0xFFFFFFFFu:0u;\n",rT,rA,rB); return;
        case op11::CEQH: emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++) R[%d].u16[i]=(R[%d].u16[i]==R[%d].u16[i])?0xFFFFu:0u;\n",rT,rA,rB); return;
        case op11::CEQB: emitf(buf,bufSize,pos,"  for(int i=0;i<16;i++) R[%d].u8[i]=(R[%d].u8[i]==R[%d].u8[i])?0xFFu:0u;\n",rT,rA,rB); return;
        case op11::CGT:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].s32[i]>R[%d].s32[i])?0xFFFFFFFFu:0u;\n",rT,rA,rB); return;
        case op11::CGTH: emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++) R[%d].u16[i]=(R[%d].s16[i]>R[%d].s16[i])?0xFFFFu:0u;\n",rT,rA,rB); return;
        case op11::CGTB: emitf(buf,bufSize,pos,"  for(int i=0;i<16;i++) R[%d].u8[i]=(R[%d].s8[i]>R[%d].s8[i])?0xFFu:0u;\n",rT,rA,rB); return;
        case op11::CLGT: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].u32[i]>R[%d].u32[i])?0xFFFFFFFFu:0u;\n",rT,rA,rB); return;
        case op11::CLGTH:emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++) R[%d].u16[i]=(R[%d].u16[i]>R[%d].u16[i])?0xFFFFu:0u;\n",rT,rA,rB); return;
        case op11::CLGTB:emitf(buf,bufSize,pos,"  for(int i=0;i<16;i++) R[%d].u8[i]=(R[%d].u8[i]>R[%d].u8[i])?0xFFu:0u;\n",rT,rA,rB); return;

        // Float
        case op11::FA:   emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].f32[i]=R[%d].f32[i]+R[%d].f32[i];\n",rT,rA,rB); return;
        case op11::FS:   emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].f32[i]=R[%d].f32[i]-R[%d].f32[i];\n",rT,rA,rB); return;
        case op11::FM:   emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].f32[i]=R[%d].f32[i]*R[%d].f32[i];\n",rT,rA,rB); return;
        case op11::FCEQ: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].f32[i]==R[%d].f32[i])?0xFFFFFFFFu:0u;\n",rT,rA,rB); return;
        case op11::FCGT: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].f32[i]>R[%d].f32[i])?0xFFFFFFFFu:0u;\n",rT,rA,rB); return;
        case op11::FCMEQ:emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(fabsf(R[%d].f32[i])==fabsf(R[%d].f32[i]))?0xFFFFFFFFu:0u;\n",rT,rA,rB); return;
        case op11::FCMGT:emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(fabsf(R[%d].f32[i])>fabsf(R[%d].f32[i]))?0xFFFFFFFFu:0u;\n",rT,rA,rB); return;
        case op11::DFA:  emitf(buf,bufSize,pos,"  for(int i=0;i<2;i++) R[%d].f64[i]=R[%d].f64[i]+R[%d].f64[i];\n",rT,rA,rB); return;
        case op11::DFS:  emitf(buf,bufSize,pos,"  for(int i=0;i<2;i++) R[%d].f64[i]=R[%d].f64[i]-R[%d].f64[i];\n",rT,rA,rB); return;
        case op11::DFM:  emitf(buf,bufSize,pos,"  for(int i=0;i<2;i++) R[%d].f64[i]=R[%d].f64[i]*R[%d].f64[i];\n",rT,rA,rB); return;

        // Branches (indirect)
        case op11::BI:   emitf(buf,bufSize,pos,"  pc=R[%d].u32[0]&0x3FFFCu; continue;\n",rA); return;
        case op11::BISL: emitf(buf,bufSize,pos,"  R[%d].u32[0]=0x%xu;R[%d].u32[1]=R[%d].u32[2]=R[%d].u32[3]=0;pc=R[%d].u32[0]&0x3FFFCu;continue;\n",rT,pc+4,rT,rT,rT,rA); return;
        case op11::BIZ:  emitf(buf,bufSize,pos,"  if(R[%d].u32[0]==0u){pc=R[%d].u32[0]&0x3FFFCu;continue;}\n",rT,rA); return;
        case op11::BINZ: emitf(buf,bufSize,pos,"  if(R[%d].u32[0]!=0u){pc=R[%d].u32[0]&0x3FFFCu;continue;}\n",rT,rA); return;
        case op11::BIHZ: emitf(buf,bufSize,pos,"  if((R[%d].u32[0]&0xFFFFu)==0u){pc=R[%d].u32[0]&0x3FFFCu;continue;}\n",rT,rA); return;
        case op11::BIHNZ:emitf(buf,bufSize,pos,"  if((R[%d].u32[0]&0xFFFFu)!=0u){pc=R[%d].u32[0]&0x3FFFCu;continue;}\n",rT,rA); return;

        // Channel (simplified — just mailbox basics for now)
        case op11::RDCH: emitf(buf,bufSize,pos,"  R[%d].u32[0]=0u;R[%d].u32[1]=R[%d].u32[2]=R[%d].u32[3]=0u;\n",rT,rT,rT,rT); return;
        case op11::WRCH: emitf(buf,bufSize,pos,"  /* wrch ch%d */;\n",rA); return;
        case op11::RCHCNT:emitf(buf,bufSize,pos,"  R[%d].u32[0]=1u;R[%d].u32[1]=R[%d].u32[2]=R[%d].u32[3]=0u;\n",rT,rT,rT,rT); return;

        // NOP
        case op11::NOP: case op11::LNOP: case op11::SYNC: case op11::DSYNC:
            return; // nothing to emit

        // Stop
        case op11::STOP: case op11::STOPD:
            emitf(buf,bufSize,pos,"  halted=1u; goto done;\n"); return;

        default: break;
        }
    }

    // RI16 (9-bit)
    uint32_t o9 = spu_op9(inst);
    {
        uint32_t rT = spu_rT_ri16(inst);
        int32_t i16 = spu_I16(inst);
        uint32_t i16u = spu_I16u(inst);

        switch (o9) {
        case op9::IL:   emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].s32[i]=%d;\n",rT,i16); return;
        case op9::ILH:  emitf(buf,bufSize,pos,"  for(int i=0;i<8;i++) R[%d].u16[i]=(uint16_t)%uu;\n",rT,i16u); return;
        case op9::ILHU: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=%uu<<16;\n",rT,i16u); return;
        case op9::IOHL: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]|=%uu;\n",rT,i16u); return;
        case op9::FSMBI:emitf(buf,bufSize,pos,"  for(int i=0;i<16;i++) R[%d].u8[i]=(%uu&(1u<<(15-i)))?0xFFu:0x00u;\n",rT,i16u); return;
        case op9::BR:   emitf(buf,bufSize,pos,"  pc=0x%xu; continue;\n",(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRA:  emitf(buf,bufSize,pos,"  pc=0x%xu; continue;\n",((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRSL: emitf(buf,bufSize,pos,"  R[%d].u32[0]=0x%xu;R[%d].u32[1]=R[%d].u32[2]=R[%d].u32[3]=0;pc=0x%xu;continue;\n",rT,pc+4,rT,rT,rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRASL:emitf(buf,bufSize,pos,"  R[%d].u32[0]=0x%xu;R[%d].u32[1]=R[%d].u32[2]=R[%d].u32[3]=0;pc=0x%xu;continue;\n",rT,pc+4,rT,rT,rT,((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRZ:  emitf(buf,bufSize,pos,"  if(R[%d].u32[0]==0u){pc=0x%xu;continue;}\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRNZ: emitf(buf,bufSize,pos,"  if(R[%d].u32[0]!=0u){pc=0x%xu;continue;}\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRHZ: emitf(buf,bufSize,pos,"  if((R[%d].u32[0]&0xFFFFu)==0u){pc=0x%xu;continue;}\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRHNZ:emitf(buf,bufSize,pos,"  if((R[%d].u32[0]&0xFFFFu)!=0u){pc=0x%xu;continue;}\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::LQA:  emitf(buf,bufSize,pos,"  R[%d]=ld(ls,0x%xu);\n",rT,((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)&~0xF); return;
        case op9::STQA: emitf(buf,bufSize,pos,"  st(ls,0x%xu,R[%d]);\n",((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)&~0xF,rT); return;
        case op9::LQR:  emitf(buf,bufSize,pos,"  R[%d]=ld(ls,0x%xu);\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)&~0xF); return;
        case op9::STQR: emitf(buf,bufSize,pos,"  st(ls,0x%xu,R[%d]);\n",(pc+(i16<<2))&(SPU_LS_SIZE-1)&~0xF,rT); return;
        default: break;
        }
    }

    // RI10 (8-bit)
    uint32_t o8 = spu_op8(inst);
    {
        uint32_t rT = spu_rT_ri10(inst), rA = spu_rA_ri10(inst);
        int32_t i10 = spu_I10(inst);

        switch (o8) {
        case op8::AI:   emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].s32[i]=R[%d].s32[i]+%d;\n",rT,rA,i10); return;
        case op8::AHI:  emitf(buf,bufSize,pos,"  {int16_t imm=(int16_t)(%d&0xFFFF);for(int i=0;i<8;i++) R[%d].s16[i]=R[%d].s16[i]+imm;}\n",i10,rT,rA); return;
        case op8::SFI:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].s32[i]=%d-R[%d].s32[i];\n",rT,i10,rA); return;
        case op8::SFHI: emitf(buf,bufSize,pos,"  {int16_t imm=(int16_t)(%d&0xFFFF);for(int i=0;i<8;i++) R[%d].s16[i]=imm-R[%d].s16[i];}\n",i10,rT,rA); return;
        case op8::ANDI: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].s32[i]=R[%d].s32[i]&%d;\n",rT,rA,i10); return;
        case op8::ORI:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].s32[i]=R[%d].s32[i]|%d;\n",rT,rA,i10); return;
        case op8::XORI: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].s32[i]=R[%d].s32[i]^%d;\n",rT,rA,i10); return;
        case op8::CEQI: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].s32[i]==%d)?0xFFFFFFFFu:0u;\n",rT,rA,i10); return;
        case op8::CGTI: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].s32[i]>%d)?0xFFFFFFFFu:0u;\n",rT,rA,i10); return;
        case op8::CLGTI: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].u32[i]>(uint32_t)%d)?0xFFFFFFFFu:0u;\n",rT,rA,i10); return;
        case op8::LQD:  emitf(buf,bufSize,pos,"  R[%d]=ld(ls,(R[%d].u32[0]+%du)&0x3FFFFu);\n",rT,rA,(uint32_t)(i10<<4)); return;
        case op8::STQD: emitf(buf,bufSize,pos,"  st(ls,(R[%d].u32[0]+%du)&0x3FFFFu,R[%d]);\n",rA,(uint32_t)(i10<<4),rT); return;
        case op8::MPYI: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){int16_t _a=(int16_t)(R[%d].u32[i]&0xFFFFu);R[%d].s32[i]=(int32_t)_a*%d;}\n",rA,rT,i10); return;
        case op8::MPYUI:emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){uint16_t _a=(uint16_t)(R[%d].u32[i]&0xFFFFu);R[%d].u32[i]=(uint32_t)_a*%uu;}\n",rA,rT,(uint32_t)(i10&0xFFFF)); return;
        default: break;
        }
    }

    // RI18 (7-bit)
    uint32_t o7 = spu_op7(inst);
    if (o7 == op7::ILA) {
        uint32_t rT = spu_rT_ri18(inst), imm = spu_I18(inst);
        emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=%uu;\n",rT,imm);
        return;
    }

    // RRR (4-bit)
    uint32_t o4 = spu_op4(inst);
    {
        uint32_t rT = spu_rT_rrr(inst), rA = spu_rA_rrr(inst);
        uint32_t rB = spu_rB_rrr(inst), rC = spu_rC_rrr(inst);

        switch (o4) {
        case op4::SELB: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].u32[i]=(R[%d].u32[i]&R[%d].u32[i])|(R[%d].u32[i]&~R[%d].u32[i]);\n",rT,rB,rC,rA,rC); return;
        case op4::SHUFB:
            emitf(buf,bufSize,pos,
                "  {QW _a=R[%d],_b=R[%d],_c=R[%d],_r;for(int _i=0;_i<16;_i++){uint8_t sel=_c.u8[_i];"
                "if(sel&0x80u){if((sel&0xE0u)==0xC0u)_r.u8[_i]=0;else if((sel&0xE0u)==0xE0u)_r.u8[_i]=0xFFu;else _r.u8[_i]=0x80u;}"
                "else{uint8_t idx=sel&0x1Fu;_r.u8[_i]=(idx<16u)?_a.u8[idx]:_b.u8[idx-16u];}}R[%d]=_r;}\n",
                rA, rB, rC, rT);
            return;
        case op4::FMA:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].f32[i]=R[%d].f32[i]*R[%d].f32[i]+R[%d].f32[i];\n",rT,rA,rB,rC); return;
        case op4::FMS:  emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].f32[i]=R[%d].f32[i]*R[%d].f32[i]-R[%d].f32[i];\n",rT,rA,rB,rC); return;
        case op4::FNMS: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++) R[%d].f32[i]=R[%d].f32[i]-R[%d].f32[i]*R[%d].f32[i];\n",rT,rC,rA,rB); return;
        case op4::MPYA: emitf(buf,bufSize,pos,"  for(int i=0;i<4;i++){int16_t _a=(int16_t)(R[%d].u32[i]&0xFFFFu),_b=(int16_t)(R[%d].u32[i]&0xFFFFu);R[%d].s32[i]=(int32_t)_a*(int32_t)_b+R[%d].s32[i];}\n",rA,rB,rT,rC); return;
        case op4::DFMA: emitf(buf,bufSize,pos,"  for(int i=0;i<2;i++) R[%d].f64[i]=R[%d].f64[i]*R[%d].f64[i]+R[%d].f64[i];\n",rT,rA,rB,rC); return;
        case op4::DFMS: emitf(buf,bufSize,pos,"  for(int i=0;i<2;i++) R[%d].f64[i]=R[%d].f64[i]*R[%d].f64[i]-R[%d].f64[i];\n",rT,rA,rB,rC); return;
        case op4::DFNMS:emitf(buf,bufSize,pos,"  for(int i=0;i<2;i++) R[%d].f64[i]=R[%d].f64[i]-R[%d].f64[i]*R[%d].f64[i];\n",rT,rC,rA,rB); return;
        default: break;
        }
    }

    // Unknown instruction — emit halt for safety
    emitf(buf,bufSize,pos,"  /* unknown 0x%08x */ halted=1u; goto done;\n", inst);
}

// ═══════════════════════════════════════════════════════════════
// Emit Full Megakernel Source
// ═══════════════════════════════════════════════════════════════

struct MegaJITResult {
    CUmodule  cuModule;
    CUfunction cuFunction;
    uint32_t  numBlocks;
    uint32_t  totalInsns;
    float     compileTimeMs;
};

static int emit_megakernel(const uint8_t* ls, uint32_t entryPC,
                            const std::vector<MegaBlock>& blocks,
                            char* buf, size_t bufSize) {
    size_t pos = 0;

    // NVRTC-compatible header (no system includes)
    emitf(buf,bufSize,&pos,
        "// SPU JIT Megakernel — %u blocks, entry=0x%x\n"
        "// ONE kernel launch, runs until halt. No host round-trips.\n\n"
        "typedef unsigned int uint32_t;\n"
        "typedef int int32_t;\n"
        "typedef unsigned short uint16_t;\n"
        "typedef short int16_t;\n"
        "typedef unsigned char uint8_t;\n"
        "typedef signed char int8_t;\n"
        "typedef unsigned long long uint64_t;\n\n"
        "union QW {\n"
        "  uint32_t u32[4]; int32_t s32[4]; float f32[4];\n"
        "  uint16_t u16[8]; int16_t s16[8];\n"
        "  uint8_t u8[16]; int8_t s8[16];\n"
        "  uint64_t u64[2]; double f64[2];\n"
        "};\n\n"
        "struct SPUMegaState {\n"
        "  QW gpr[128];\n"
        "  uint32_t pc;\n"
        "  uint32_t halted;\n"
        "  uint32_t cycles;\n"
        "};\n\n"
        "__device__ __forceinline__ uint32_t bswap32(uint32_t x) {\n"
        "  return __byte_perm(x, 0, 0x0123);\n"
        "}\n"
        "__device__ __forceinline__ QW ld(const uint8_t* ls, uint32_t a) {\n"
        "  a &= 0x3FFF0u; QW q;\n"
        "  const uint32_t* p=(const uint32_t*)(ls+a);\n"
        "  q.u32[0]=bswap32(p[0]);q.u32[1]=bswap32(p[1]);\n"
        "  q.u32[2]=bswap32(p[2]);q.u32[3]=bswap32(p[3]);\n"
        "  return q;\n"
        "}\n"
        "__device__ __forceinline__ void st(uint8_t* ls, uint32_t a, const QW& q) {\n"
        "  a &= 0x3FFF0u;\n"
        "  uint32_t* p=(uint32_t*)(ls+a);\n"
        "  p[0]=bswap32(q.u32[0]);p[1]=bswap32(q.u32[1]);\n"
        "  p[2]=bswap32(q.u32[2]);p[3]=bswap32(q.u32[3]);\n"
        "}\n"
        "__device__ float fabsf(float x) { return x < 0.0f ? -x : x; }\n\n",
        (uint32_t)blocks.size(), entryPC);

    // Kernel: single-thread persistent dispatch loop
    emitf(buf,bufSize,&pos,
        "extern \"C\" __global__ void spu_mega(\n"
        "    SPUMegaState* __restrict__ state,\n"
        "    uint8_t* __restrict__ ls,\n"
        "    uint8_t* __restrict__ mainMem,\n"
        "    uint32_t maxCycles)\n"
        "{\n"
        "  if(threadIdx.x!=0) return;\n"
        "  QW* R = state->gpr;\n"
        "  uint32_t pc = state->pc;\n"
        "  uint32_t halted = 0;\n"
        "  uint32_t cycles = state->cycles;\n"
        "  uint32_t limit = cycles + maxCycles;\n\n"
        "  while(!halted && cycles < limit) {\n"
        "    switch(pc) {\n");

    // Emit each block as a single case
    for (const auto& blk : blocks) {
        emitf(buf,bufSize,&pos, "\n    // --- block 0x%04x (%u insns) ---\n",
               blk.startPC, blk.numInsns);
        emitf(buf,bufSize,&pos, "    case 0x%xu:\n", blk.startPC);

        // Emit each instruction in the block (sequential, no mid-block labels)
        for (uint32_t i = 0; i < blk.numInsns; i++) {
            uint32_t ipc = (blk.startPC + i * 4) & (SPU_LS_SIZE - 1);
            emit_one_insn(ls, ipc, buf, bufSize, &pos);
            emitf(buf,bufSize,&pos, "      cycles++;\n");
        }

        // Fall-through to next sequential PC (if block didn't end with branch/stop)
        if (!blk.endsWithStop) {
            uint32_t nextPC = (blk.endPC + 4) & (SPU_LS_SIZE - 1);
            emitf(buf,bufSize,&pos, "      pc=0x%xu; continue;\n", nextPC);
        }
    }

    // Default case: unknown PC — halt
    emitf(buf,bufSize,&pos,
        "\n    default:\n"
        "      halted=1u;\n"
        "      break;\n"
        "    } // switch\n"
        "  } // while\n\n"
        "done:\n"
        "  state->pc = pc;\n"
        "  state->halted = halted;\n"
        "  state->cycles = cycles;\n"
        "}\n");

    return (int)pos;
}

// ═══════════════════════════════════════════════════════════════
// Compile + Launch API
// ═══════════════════════════════════════════════════════════════

extern "C" {

int mega_jit_compile(const uint8_t* h_ls, uint32_t entryPC,
                     MegaJITResult* result) {
    memset(result, 0, sizeof(MegaJITResult));

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);
    cudaEventRecord(tStart);

    // 1. Discover all reachable blocks
    std::vector<MegaBlock> blocks;
    std::set<uint32_t> blockStarts;
    discover_all_blocks(h_ls, entryPC, blocks, blockStarts);

    result->numBlocks = (uint32_t)blocks.size();
    uint32_t totalInsns = 0;
    for (auto& b : blocks) totalInsns += b.numInsns;
    result->totalInsns = totalInsns;

    fprintf(stderr, "[MegaJIT] Discovered %u blocks, %u instructions from entry 0x%x\n",
            result->numBlocks, totalInsns, entryPC);

    // 2. Emit megakernel source
    size_t srcBufSize = 512 * 1024; // 512KB should be plenty
    char* source = (char*)malloc(srcBufSize);
    int srcLen = emit_megakernel(h_ls, entryPC, blocks, source, srcBufSize);
    fprintf(stderr, "[MegaJIT] Generated %d bytes of CUDA source\n", srcLen);

    // 3. NVRTC compile
    nvrtcProgram prog;
    nvrtcResult nres = nvrtcCreateProgram(&prog, source, "spu_mega.cu", 0, NULL, NULL);
    if (nres != NVRTC_SUCCESS) {
        fprintf(stderr, "[MegaJIT] nvrtcCreateProgram failed\n");
        free(source);
        return 0;
    }

    const char* opts[] = {
        "--gpu-architecture=sm_70",
        "--use_fast_math",
        "--std=c++17",
    };
    nres = nvrtcCompileProgram(prog, 3, opts);

    if (nres != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = (char*)malloc(logSize);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "[MegaJIT] Compile FAILED:\n%s\n", log);
        free(log);
        nvrtcDestroyProgram(&prog);
        free(source);
        return 0;
    }

    free(source);

    // 4. Get PTX and load
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = (char*)malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    fprintf(stderr, "[MegaJIT] PTX: %zu bytes\n", ptxSize);

    CUmodule cuMod;
    CUresult cuRes = cuModuleLoadDataEx(&cuMod, ptx, 0, NULL, NULL);
    free(ptx);
    if (cuRes != CUDA_SUCCESS) {
        fprintf(stderr, "[MegaJIT] cuModuleLoadData failed: %d\n", cuRes);
        return 0;
    }

    CUfunction cuFunc;
    cuRes = cuModuleGetFunction(&cuFunc, cuMod, "spu_mega");
    if (cuRes != CUDA_SUCCESS) {
        fprintf(stderr, "[MegaJIT] cuModuleGetFunction failed: %d\n", cuRes);
        cuModuleUnload(cuMod);
        return 0;
    }

    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);
    cudaEventElapsedTime(&result->compileTimeMs, tStart, tStop);
    cudaEventDestroy(tStart);
    cudaEventDestroy(tStop);

    result->cuModule = cuMod;
    result->cuFunction = cuFunc;

    fprintf(stderr, "[MegaJIT] ✅ Compiled: %u blocks, %u insns → CUDA kernel (%.1f ms)\n",
            result->numBlocks, result->totalInsns, result->compileTimeMs);
    return 1;
}

// SPUMegaState layout must match NVRTC-generated struct
struct SPUMegaState {
    QWord gpr[128];
    uint32_t pc;
    uint32_t halted;
    uint32_t cycles;
};

float mega_jit_run(const MegaJITResult* jit,
                   SPUState* h_state, uint8_t* d_ls, uint8_t* d_mainMem,
                   uint32_t maxCycles) {
    // Convert SPUState → SPUMegaState
    SPUMegaState h_mega;
    memcpy(h_mega.gpr, h_state->gpr, sizeof(h_mega.gpr));
    h_mega.pc = h_state->pc;
    h_mega.halted = h_state->halted;
    h_mega.cycles = h_state->cycles;

    SPUMegaState* d_mega;
    cudaMalloc(&d_mega, sizeof(SPUMegaState));
    cudaMemcpy(d_mega, &h_mega, sizeof(SPUMegaState), cudaMemcpyHostToDevice);

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);

    // ONE launch — runs until halt or maxCycles
    void* args[] = { &d_mega, &d_ls, &d_mainMem, &maxCycles };
    cudaEventRecord(tStart);
    CUresult err = cuLaunchKernel(
        (CUfunction)jit->cuFunction,
        1, 1, 1,   // 1 block
        1, 1, 1,   // 1 thread
        0, 0,      // shared mem, stream
        args, NULL);
    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);

    float ms = 0;
    if (err == CUDA_SUCCESS) {
        cudaEventElapsedTime(&ms, tStart, tStop);
    } else {
        fprintf(stderr, "[MegaJIT] Launch failed: %d\n", err);
        ms = -1.0f;
    }

    // Read back
    cudaMemcpy(&h_mega, d_mega, sizeof(SPUMegaState), cudaMemcpyDeviceToHost);
    cudaFree(d_mega);
    cudaEventDestroy(tStart);
    cudaEventDestroy(tStop);

    // Write back to SPUState
    memcpy(h_state->gpr, h_mega.gpr, sizeof(h_state->gpr));
    h_state->pc = h_mega.pc;
    h_state->halted = h_mega.halted;
    h_state->cycles = h_mega.cycles;

    return ms;
}

void mega_jit_free(MegaJITResult* jit) {
    if (jit->cuModule) cuModuleUnload((CUmodule)jit->cuModule);
    memset(jit, 0, sizeof(MegaJITResult));
}

} // extern "C"
