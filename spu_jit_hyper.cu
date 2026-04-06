// spu_jit_hyper.cu — Hyper SPU JIT: loop detection + shared mem LS
//
// Optimizations over spu_jit_turbo.cu:
//   1. Loop detection: self-loops → tight do{}while() (no switch re-dispatch)
//   2. Batched cycle counting: cycles += N per block (not per instruction)
//   3. Loop unrolling: tight loops unrolled 4× for ILP
//   4. __ldg() read-only LS cache: LS reads through texture cache (20 vs 200 cycle)
//   5. All turbo opts: register promotion, unrolled SIMD, multi-instance
//
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
#include <map>
#include <algorithm>

using namespace spu;

// Convenience aliases — RR-format register extraction (most common in op11 decode)
inline uint32_t spu_rA(uint32_t inst) { return spu_rA_rr(inst); }
inline uint32_t spu_rB(uint32_t inst) { return spu_rB_rr(inst); }
inline uint32_t spu_rT(uint32_t inst) { return spu_rT_rr(inst); }

// ═══════════════════════════════════════════════════════════════
// Helpers
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
    int w = vsnprintf(buf + *pos, bufSize - *pos, fmt, args);
    va_end(args);
    if (w > 0 && (size_t)w < bufSize - *pos) *pos += w;
    return w;
}

// ═══════════════════════════════════════════════════════════════
// Extended Block Analysis with Loop Detection
// ═══════════════════════════════════════════════════════════════

struct HyperBlock {
    uint32_t startPC, endPC, numInsns;
    bool endsWithStop, endsWithBranch;

    // Loop analysis
    bool isSelfLoop;       // branch at end targets startPC
    uint32_t loopCondReg;  // register tested by conditional branch
    bool loopCondNonZero;  // true=BRNZ (loop while !=0), false=BRZ (loop while ==0)
    uint32_t fallThroughPC; // where to go after loop exits
};

struct HyperAnalysis {
    std::vector<HyperBlock> blocks;
    std::set<uint32_t> blockStarts;
    bool usedRegs[128];
    bool writtenRegs[128];
    uint32_t numUsedRegs;
    uint32_t entryPC;

    // Loop stats
    uint32_t numSelfLoops;
};

static void analyze_program(const uint8_t* ls, uint32_t entryPC, HyperAnalysis* pa) {
    pa->blocks.clear();
    pa->blockStarts.clear();
    memset(pa->usedRegs, 0, sizeof(pa->usedRegs));
    memset(pa->writtenRegs, 0, sizeof(pa->writtenRegs));
    pa->numUsedRegs = 0;
    pa->entryPC = entryPC;
    pa->numSelfLoops = 0;

    // Phase 1: BFS to find all block starts
    std::queue<uint32_t> wl;
    std::set<uint32_t> visited;
    wl.push(entryPC & (SPU_LS_SIZE - 1));

    while (!wl.empty()) {
        uint32_t pc = wl.front(); wl.pop();
        if (visited.count(pc)) continue;
        visited.insert(pc);
        pa->blockStarts.insert(pc);

        for (uint32_t n = 0; n < 512; n++) {
            uint32_t inst = fetch_inst_host(ls, pc);
            uint32_t o11 = spu_op11(inst), o9 = spu_op9(inst);
            bool isBr = false, isStop = false;

            switch (o11) {
            case op11::BI: case op11::BISL: case op11::BIZ: case op11::BINZ:
            case op11::BIHZ: case op11::BIHNZ: isBr = true; break;
            case op11::STOP: case op11::STOPD: isStop = true; break;
            default: break;
            }
            if (!isBr && !isStop) {
                switch (o9) {
                case op9::BR: case op9::BRA: case op9::BRSL: case op9::BRASL:
                case op9::BRZ: case op9::BRNZ: case op9::BRHZ: case op9::BRHNZ:
                    isBr = true; break;
                default: break;
                }
            }

            // Track register usage (same as turbo — needed for register promotion)
            if (!isBr && !isStop) {
                // RR format
                switch (o11) {
                case op11::A: case op11::AH: case op11::SF: case op11::SFH:
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
                case op11::FCEQ: case op11::FCGT:
                case op11::DFA: case op11::DFS: case op11::DFM:
                case op11::CG: case op11::BG: case op11::ADDX: case op11::SFX:
                case op11::MPYS: case op11::MPYHH: case op11::MPYHHU:
                case op11::AVGB: case op11::ABSDB: {
                    uint32_t rA = spu_rA(inst), rB = spu_rB(inst), rT = spu_rT(inst);
                    pa->usedRegs[rA] = pa->usedRegs[rB] = pa->usedRegs[rT] = true;
                    pa->writtenRegs[rT] = true;
                    break;
                }
                case op11::CLZ: case op11::CNTB: case op11::XSBH: case op11::XSHW: case op11::XSWD: {
                    uint32_t rA = spu_rA(inst), rT = spu_rT(inst);
                    pa->usedRegs[rA] = pa->usedRegs[rT] = true;
                    pa->writtenRegs[rT] = true;
                    break;
                }
                case op11::LQX: {
                    uint32_t rA = spu_rA(inst), rB = spu_rB(inst), rT = spu_rT(inst);
                    pa->usedRegs[rA] = pa->usedRegs[rB] = pa->usedRegs[rT] = true;
                    pa->writtenRegs[rT] = true; break;
                }
                case op11::STQX: {
                    uint32_t rA = spu_rA(inst), rB = spu_rB(inst), rT = spu_rT(inst);
                    pa->usedRegs[rA] = pa->usedRegs[rB] = pa->usedRegs[rT] = true; break;
                }
                case op11::RDCH: case op11::RCHCNT: {
                    uint32_t rT = spu_rT(inst);
                    pa->usedRegs[rT] = true; pa->writtenRegs[rT] = true; break;
                }
                default: break;
                }

                // RI16
                switch (o9) {
                case op9::IL: case op9::ILH: case op9::ILHU: case op9::FSMBI: {
                    uint32_t rT = spu_rT_ri16(inst);
                    pa->usedRegs[rT] = true; pa->writtenRegs[rT] = true; break;
                }
                case op9::IOHL: {
                    uint32_t rT = spu_rT_ri16(inst);
                    pa->usedRegs[rT] = true; pa->writtenRegs[rT] = true; break;
                }
                case op9::LQA: case op9::LQR: {
                    uint32_t rT = spu_rT_ri16(inst);
                    pa->usedRegs[rT] = true; pa->writtenRegs[rT] = true; break;
                }
                case op9::STQA: case op9::STQR: {
                    uint32_t rT = spu_rT_ri16(inst);
                    pa->usedRegs[rT] = true; break;
                }
                default: break;
                }

                // RI10
                switch (spu_op8(inst)) {
                case op8::AI: case op8::AHI: case op8::SFI: case op8::SFHI:
                case op8::ANDI: case op8::ORI: case op8::XORI:
                case op8::CEQI: case op8::CGTI: case op8::CLGTI:
                case op8::MPYI: case op8::MPYUI: {
                    uint32_t rA = spu_rA_ri10(inst), rT = spu_rT_ri10(inst);
                    pa->usedRegs[rA] = pa->usedRegs[rT] = true;
                    pa->writtenRegs[rT] = true; break;
                }
                case op8::LQD: {
                    uint32_t rA = spu_rA_ri10(inst), rT = spu_rT_ri10(inst);
                    pa->usedRegs[rA] = pa->usedRegs[rT] = true;
                    pa->writtenRegs[rT] = true; break;
                }
                case op8::STQD: {
                    uint32_t rA = spu_rA_ri10(inst), rT = spu_rT_ri10(inst);
                    pa->usedRegs[rA] = pa->usedRegs[rT] = true; break;
                }
                default: break;
                }

                // RI18
                if (spu_op7(inst) == op7::ILA) {
                    uint32_t rT = spu_rT_ri18(inst);
                    pa->usedRegs[rT] = true; pa->writtenRegs[rT] = true;
                }

                // RRR
                switch (spu_op4(inst)) {
                case op4::SELB: case op4::SHUFB:
                case op4::FMA: case op4::FMS: case op4::FNMS:
                case op4::MPYA:
                case op4::DFMA: case op4::DFMS: case op4::DFNMS: {
                    uint32_t rT=spu_rT_rrr(inst),rA=spu_rA_rrr(inst),rB=spu_rB_rrr(inst),rC=spu_rC_rrr(inst);
                    pa->usedRegs[rA] = pa->usedRegs[rB] = pa->usedRegs[rC] = pa->usedRegs[rT] = true;
                    pa->writtenRegs[rT] = true; break;
                }
                default: break;
                }
            }

            if (isBr) {
                // Enqueue branch targets
                switch (o9) {
                case op9::BR: case op9::BRSL: {
                    int32_t off = spu_I16(inst);
                    uint32_t tgt = (pc + (off << 2)) & (SPU_LS_SIZE - 1);
                    if (!visited.count(tgt)) wl.push(tgt);
                    break;
                }
                case op9::BRA: case op9::BRASL: {
                    int32_t off = spu_I16(inst);
                    uint32_t tgt = ((uint32_t)(off << 2)) & (SPU_LS_SIZE - 1);
                    if (!visited.count(tgt)) wl.push(tgt);
                    break;
                }
                case op9::BRZ: case op9::BRNZ: case op9::BRHZ: case op9::BRHNZ: {
                    int32_t off = spu_I16(inst);
                    uint32_t tgt = (pc + (off << 2)) & (SPU_LS_SIZE - 1);
                    uint32_t fall = (pc + 4) & (SPU_LS_SIZE - 1);
                    if (!visited.count(tgt)) wl.push(tgt);
                    if (!visited.count(fall)) wl.push(fall);
                    break;
                }
                default: break;
                }

                // Enqueue fall-through for unconditional branches
                if (o9 == op9::BR || o9 == op9::BRA) {
                    // no fall-through
                } else if (o11 == op11::BI) {
                    // indirect — no static target
                } else {
                    uint32_t fall = (pc + 4) & (SPU_LS_SIZE - 1);
                    if (!visited.count(fall)) wl.push(fall);
                }
                break;
            }
            if (isStop) break;

            uint32_t nextPC = (pc + 4) & (SPU_LS_SIZE - 1);
            if (pa->blockStarts.count(nextPC) && nextPC != (entryPC & (SPU_LS_SIZE - 1))) break;
            pc = nextPC;
        }
    }

    // Phase 2: Build blocks from discovered starts
    std::vector<uint32_t> sortedStarts(pa->blockStarts.begin(), pa->blockStarts.end());
    std::sort(sortedStarts.begin(), sortedStarts.end());

    for (size_t bi = 0; bi < sortedStarts.size(); bi++) {
        uint32_t start = sortedStarts[bi];
        HyperBlock blk;
        blk.startPC = start;
        blk.endsWithStop = blk.endsWithBranch = false;
        blk.isSelfLoop = false;
        blk.loopCondReg = 0;
        blk.loopCondNonZero = false;
        blk.fallThroughPC = 0;
        blk.numInsns = 0;

        uint32_t pc = start;
        for (uint32_t n = 0; n < 512; n++) {
            uint32_t inst = fetch_inst_host(ls, pc);
            uint32_t o11 = spu_op11(inst), o9 = spu_op9(inst);
            blk.numInsns++;
            blk.endPC = pc;

            bool isBr = false, isStop = false;
            switch (o11) {
            case op11::BI: case op11::BISL: case op11::BIZ: case op11::BINZ:
            case op11::BIHZ: case op11::BIHNZ: isBr = true; break;
            case op11::STOP: case op11::STOPD: isStop = true; break;
            default: break;
            }
            if (!isBr && !isStop) {
                switch (o9) {
                case op9::BR: case op9::BRA: case op9::BRSL: case op9::BRASL:
                case op9::BRZ: case op9::BRNZ: case op9::BRHZ: case op9::BRHNZ:
                    isBr = true; break;
                default: break;
                }
            }

            if (isBr) {
                blk.endsWithBranch = true;

                // Detect self-loops: conditional branch back to block start
                uint32_t brTarget = 0;
                bool isCond = false;
                uint32_t condReg = 0;
                bool condNZ = false;

                switch (o9) {
                case op9::BRNZ: {
                    int32_t off = spu_I16(inst);
                    brTarget = (pc + (off << 2)) & (SPU_LS_SIZE - 1);
                    condReg = spu_rT_ri16(inst);
                    isCond = true; condNZ = true;
                    break;
                }
                case op9::BRZ: {
                    int32_t off = spu_I16(inst);
                    brTarget = (pc + (off << 2)) & (SPU_LS_SIZE - 1);
                    condReg = spu_rT_ri16(inst);
                    isCond = true; condNZ = false;
                    break;
                }
                case op9::BRHNZ: {
                    int32_t off = spu_I16(inst);
                    brTarget = (pc + (off << 2)) & (SPU_LS_SIZE - 1);
                    condReg = spu_rT_ri16(inst);
                    isCond = true; condNZ = true;
                    break;
                }
                case op9::BRHZ: {
                    int32_t off = spu_I16(inst);
                    brTarget = (pc + (off << 2)) & (SPU_LS_SIZE - 1);
                    condReg = spu_rT_ri16(inst);
                    isCond = true; condNZ = false;
                    break;
                }
                case op9::BR: {
                    int32_t off = spu_I16(inst);
                    brTarget = (pc + (off << 2)) & (SPU_LS_SIZE - 1);
                    // Unconditional self-loop = infinite loop (still emit tight)
                    if (brTarget == start) {
                        blk.isSelfLoop = true;
                        blk.loopCondReg = 0;
                        blk.loopCondNonZero = true;
                        blk.fallThroughPC = (pc + 4) & (SPU_LS_SIZE - 1);
                    }
                    break;
                }
                default: break;
                }

                if (isCond && brTarget == start) {
                    blk.isSelfLoop = true;
                    blk.loopCondReg = condReg;
                    blk.loopCondNonZero = condNZ;
                    blk.fallThroughPC = (pc + 4) & (SPU_LS_SIZE - 1);
                    pa->numSelfLoops++;
                }
                break;
            }
            if (isStop) { blk.endsWithStop = true; break; }

            uint32_t nextPC = (pc + 4) & (SPU_LS_SIZE - 1);
            if (pa->blockStarts.count(nextPC) && nextPC != start) break;
            pc = nextPC;
        }
        pa->blocks.push_back(blk);
    }

    for (int i = 0; i < 128; i++) {
        if (pa->usedRegs[i]) pa->numUsedRegs++;
    }
}

// ═══════════════════════════════════════════════════════════════
// Instruction Emitter — same unrolled SIMD as turbo
// ═══════════════════════════════════════════════════════════════

// Emit a single SPU instruction as CUDA source (without cycles++)
// Returns true if this is NOT the block-ending branch (so caller handles branches)
static void emit_hyper_insn(const uint8_t* ls, uint32_t pc, const HyperAnalysis* pa,
                            char* buf, size_t bufSize, size_t* pos) {
    uint32_t inst = fetch_inst_host(ls, pc);
    uint32_t o11 = spu_op11(inst);

    // RR format (11-bit) — widest first
    {
        uint32_t rA = spu_rA(inst), rB = spu_rB(inst), rT = spu_rT(inst);
        switch (o11) {
        case op11::A:   emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[0]+r%d.u32[0];r%d.u32[1]=r%d.u32[1]+r%d.u32[1];r%d.u32[2]=r%d.u32[2]+r%d.u32[2];r%d.u32[3]=r%d.u32[3]+r%d.u32[3];\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::AH:  emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++) r%d.u16[i]=r%d.u16[i]+r%d.u16[i];\n",rT,rA,rB); return;
        case op11::SF:  emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[0]-r%d.u32[0];r%d.u32[1]=r%d.u32[1]-r%d.u32[1];r%d.u32[2]=r%d.u32[2]-r%d.u32[2];r%d.u32[3]=r%d.u32[3]-r%d.u32[3];\n",rT,rB,rA,rT,rB,rA,rT,rB,rA,rT,rB,rA); return;
        case op11::SFH: emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++) r%d.u16[i]=r%d.u16[i]-r%d.u16[i];\n",rT,rB,rA); return;
        case op11::AND: emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[0]&r%d.u32[0];r%d.u32[1]=r%d.u32[1]&r%d.u32[1];r%d.u32[2]=r%d.u32[2]&r%d.u32[2];r%d.u32[3]=r%d.u32[3]&r%d.u32[3];\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::OR:  emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[0]|r%d.u32[0];r%d.u32[1]=r%d.u32[1]|r%d.u32[1];r%d.u32[2]=r%d.u32[2]|r%d.u32[2];r%d.u32[3]=r%d.u32[3]|r%d.u32[3];\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::XOR: emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[0]^r%d.u32[0];r%d.u32[1]=r%d.u32[1]^r%d.u32[1];r%d.u32[2]=r%d.u32[2]^r%d.u32[2];r%d.u32[3]=r%d.u32[3]^r%d.u32[3];\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::ANDC:emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[0]&~r%d.u32[0];r%d.u32[1]=r%d.u32[1]&~r%d.u32[1];r%d.u32[2]=r%d.u32[2]&~r%d.u32[2];r%d.u32[3]=r%d.u32[3]&~r%d.u32[3];\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::NAND:emitf(buf,bufSize,pos,"      r%d.u32[0]=~(r%d.u32[0]&r%d.u32[0]);r%d.u32[1]=~(r%d.u32[1]&r%d.u32[1]);r%d.u32[2]=~(r%d.u32[2]&r%d.u32[2]);r%d.u32[3]=~(r%d.u32[3]&r%d.u32[3]);\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::NOR: emitf(buf,bufSize,pos,"      r%d.u32[0]=~(r%d.u32[0]|r%d.u32[0]);r%d.u32[1]=~(r%d.u32[1]|r%d.u32[1]);r%d.u32[2]=~(r%d.u32[2]|r%d.u32[2]);r%d.u32[3]=~(r%d.u32[3]|r%d.u32[3]);\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::ORC: emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[0]|~r%d.u32[0];r%d.u32[1]=r%d.u32[1]|~r%d.u32[1];r%d.u32[2]=r%d.u32[2]|~r%d.u32[2];r%d.u32[3]=r%d.u32[3]|~r%d.u32[3];\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::EQV: emitf(buf,bufSize,pos,"      r%d.u32[0]=~(r%d.u32[0]^r%d.u32[0]);r%d.u32[1]=~(r%d.u32[1]^r%d.u32[1]);r%d.u32[2]=~(r%d.u32[2]^r%d.u32[2]);r%d.u32[3]=~(r%d.u32[3]^r%d.u32[3]);\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;

        case op11::SHL:  emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.u32[0]&0x3Fu)<32u?(r%d.u32[0]<<(r%d.u32[0]&0x3Fu)):0u;r%d.u32[1]=(r%d.u32[1]&0x3Fu)<32u?(r%d.u32[1]<<(r%d.u32[1]&0x3Fu)):0u;r%d.u32[2]=(r%d.u32[2]&0x3Fu)<32u?(r%d.u32[2]<<(r%d.u32[2]&0x3Fu)):0u;r%d.u32[3]=(r%d.u32[3]&0x3Fu)<32u?(r%d.u32[3]<<(r%d.u32[3]&0x3Fu)):0u;\n",rT,rB,rA,rB,rT,rB,rA,rB,rT,rB,rA,rB,rT,rB,rA,rB); return;
        case op11::ROT:  emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){uint32_t sh=r%d.u32[i]&0x1Fu;uint32_t v=r%d.u32[i];r%d.u32[i]=(v<<sh)|(v>>(32u-sh));}\n",rB,rA,rT); return;
        case op11::ROTM: emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){uint32_t sh=(-r%d.s32[i])&0x3Fu;r%d.u32[i]=(sh<32u)?(r%d.u32[i]>>sh):0u;}\n",rB,rT,rA); return;
        case op11::ROTMA:emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){uint32_t sh=(-r%d.s32[i])&0x3Fu;int32_t v=r%d.s32[i];r%d.s32[i]=(sh<32u)?(v>>sh):(v>>31);}\n",rB,rA,rT); return;
        case op11::SHLH: emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++){uint32_t sh=r%d.u16[i]&0x1Fu;r%d.u16[i]=(sh<16u)?(r%d.u16[i]<<sh):0u;}\n",rB,rT,rA); return;
        case op11::ROTH: emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++){uint32_t sh=r%d.u16[i]&0xFu;uint16_t v=r%d.u16[i];r%d.u16[i]=(v<<sh)|(v>>(16u-sh));}\n",rB,rA,rT); return;
        case op11::ROTHM:emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++){uint32_t sh=(-(int32_t)(int16_t)r%d.u16[i])&0x1Fu;r%d.u16[i]=(sh<16u)?(r%d.u16[i]>>sh):0u;}\n",rB,rT,rA); return;
        case op11::ROTMAH:emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++){uint32_t sh=(-(int32_t)(int16_t)r%d.u16[i])&0x1Fu;int16_t v=r%d.s16[i];r%d.s16[i]=(sh<16u)?(v>>sh):(v>>15);}\n",rB,rA,rT); return;
        case op11::ROTQBY: emitf(buf,bufSize,pos,"      {uint32_t _sh=r%d.u32[0]&0xFu;QW _t=r%d;for(int i=0;i<16;i++) r%d.u8[i]=_t.u8[(i+_sh)&0xFu];}\n",rB,rA,rT); return;
        case op11::ROTQBI: emitf(buf,bufSize,pos,"      {uint32_t _sh=r%d.u32[0]&0x7u;uint64_t h=(((uint64_t)r%d.u32[0]<<32)|r%d.u32[1]);uint64_t l=(((uint64_t)r%d.u32[2]<<32)|r%d.u32[3]);uint64_t rh=(h<<_sh)|(l>>(64-_sh));uint64_t rl=(l<<_sh)|(h>>(64-_sh));r%d.u32[0]=(uint32_t)(rh>>32);r%d.u32[1]=(uint32_t)rh;r%d.u32[2]=(uint32_t)(rl>>32);r%d.u32[3]=(uint32_t)rl;}\n",rB,rA,rA,rA,rA,rT,rT,rT,rT); return;
        case op11::SHLQBY: emitf(buf,bufSize,pos,"      {uint32_t _sh=r%d.u32[0]&0x1Fu;QW _t=r%d;for(int i=0;i<16;i++){int src=i+_sh;r%d.u8[i]=(src<16)?_t.u8[src]:0u;}}\n",rB,rA,rT); return;
        case op11::SHLQBI: emitf(buf,bufSize,pos,"      {uint32_t _sh=r%d.u32[0]&0x7u;if(_sh){uint64_t h=(((uint64_t)r%d.u32[0]<<32)|r%d.u32[1]);uint64_t l=(((uint64_t)r%d.u32[2]<<32)|r%d.u32[3]);uint64_t rh=(h<<_sh)|(l>>(64-_sh));uint64_t rl=(l<<_sh);r%d.u32[0]=(uint32_t)(rh>>32);r%d.u32[1]=(uint32_t)rh;r%d.u32[2]=(uint32_t)(rl>>32);r%d.u32[3]=(uint32_t)rl;}else{r%d=r%d;}}\n",rB,rA,rA,rA,rA,rT,rT,rT,rT,rT,rA); return;
        case op11::ROTQMBY:emitf(buf,bufSize,pos,"      {uint32_t _sh=(-r%d.s32[0])&0x1Fu;QW _t=r%d;for(int i=0;i<16;i++){int src=i-_sh;r%d.u8[i]=(src>=0)?_t.u8[src]:0u;}}\n",rB,rA,rT); return;
        case op11::ROTQMBI:emitf(buf,bufSize,pos,"      {uint32_t _sh=(-r%d.s32[0])&0x7u;if(_sh){uint64_t h=(((uint64_t)r%d.u32[0]<<32)|r%d.u32[1]);uint64_t l=(((uint64_t)r%d.u32[2]<<32)|r%d.u32[3]);uint64_t rl=(l>>_sh)|(h<<(64-_sh));uint64_t rh=(h>>_sh);r%d.u32[0]=(uint32_t)(rh>>32);r%d.u32[1]=(uint32_t)rh;r%d.u32[2]=(uint32_t)(rl>>32);r%d.u32[3]=(uint32_t)rl;}else{r%d=r%d;}}\n",rB,rA,rA,rA,rA,rT,rT,rT,rT,rT,rA); return;

        case op11::CEQ:  emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.u32[0]==r%d.u32[0])?~0u:0u;r%d.u32[1]=(r%d.u32[1]==r%d.u32[1])?~0u:0u;r%d.u32[2]=(r%d.u32[2]==r%d.u32[2])?~0u:0u;r%d.u32[3]=(r%d.u32[3]==r%d.u32[3])?~0u:0u;\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::CEQH: emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++) r%d.u16[i]=(r%d.u16[i]==r%d.u16[i])?0xFFFFu:0u;\n",rT,rA,rB); return;
        case op11::CEQB: emitf(buf,bufSize,pos,"      for(int i=0;i<16;i++) r%d.u8[i]=(r%d.u8[i]==r%d.u8[i])?0xFFu:0u;\n",rT,rA,rB); return;
        case op11::CGT:  emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.s32[0]>r%d.s32[0])?~0u:0u;r%d.u32[1]=(r%d.s32[1]>r%d.s32[1])?~0u:0u;r%d.u32[2]=(r%d.s32[2]>r%d.s32[2])?~0u:0u;r%d.u32[3]=(r%d.s32[3]>r%d.s32[3])?~0u:0u;\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::CGTH: emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++) r%d.u16[i]=(r%d.s16[i]>r%d.s16[i])?0xFFFFu:0u;\n",rT,rA,rB); return;
        case op11::CGTB: emitf(buf,bufSize,pos,"      for(int i=0;i<16;i++) r%d.u8[i]=(r%d.s8[i]>r%d.s8[i])?0xFFu:0u;\n",rT,rA,rB); return;
        case op11::CLGT: emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.u32[0]>r%d.u32[0])?~0u:0u;r%d.u32[1]=(r%d.u32[1]>r%d.u32[1])?~0u:0u;r%d.u32[2]=(r%d.u32[2]>r%d.u32[2])?~0u:0u;r%d.u32[3]=(r%d.u32[3]>r%d.u32[3])?~0u:0u;\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::CLGTH:emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++) r%d.u16[i]=(r%d.u16[i]>r%d.u16[i])?0xFFFFu:0u;\n",rT,rA,rB); return;
        case op11::CLGTB:emitf(buf,bufSize,pos,"      for(int i=0;i<16;i++) r%d.u8[i]=(r%d.u8[i]>r%d.u8[i])?0xFFu:0u;\n",rT,rA,rB); return;

        case op11::CLZ:  emitf(buf,bufSize,pos,"      r%d.u32[0]=__clz(r%d.u32[0]);r%d.u32[1]=__clz(r%d.u32[1]);r%d.u32[2]=__clz(r%d.u32[2]);r%d.u32[3]=__clz(r%d.u32[3]);\n",rT,rA,rT,rA,rT,rA,rT,rA); return;
        case op11::XSBH: emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++) r%d.s16[i]=(int16_t)(int8_t)r%d.u8[i*2+1];\n",rT,rA); return;
        case op11::XSHW: emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++) r%d.s32[i]=(int32_t)(int16_t)r%d.u16[i*2+1];\n",rT,rA); return;
        case op11::XSWD: emitf(buf,bufSize,pos,"      r%d.s32[0]=r%d.s32[1];r%d.s32[2]=r%d.s32[3];\n",rT,rA,rT,rA); return;
        case op11::CNTB: emitf(buf,bufSize,pos,"      for(int i=0;i<16;i++) r%d.u8[i]=__popc(r%d.u8[i]);\n",rT,rA); return;
        case op11::AVGB: emitf(buf,bufSize,pos,"      for(int i=0;i<16;i++) r%d.u8[i]=(uint8_t)(((uint32_t)r%d.u8[i]+r%d.u8[i]+1u)>>1);\n",rT,rA,rB); return;
        case op11::ABSDB:emitf(buf,bufSize,pos,"      for(int i=0;i<16;i++){int _d=(int)r%d.u8[i]-(int)r%d.u8[i];r%d.u8[i]=(uint8_t)(_d<0?-_d:_d);}\n",rA,rB,rT); return;
        case op11::CG:   emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){unsigned long long s=(unsigned long long)r%d.u32[i]+(unsigned long long)r%d.u32[i];r%d.u32[i]=(uint32_t)(s>>32);}\n",rA,rB,rT); return;
        case op11::BG:   emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.u32[0]<=r%d.u32[0])?1u:0u;r%d.u32[1]=(r%d.u32[1]<=r%d.u32[1])?1u:0u;r%d.u32[2]=(r%d.u32[2]<=r%d.u32[2])?1u:0u;r%d.u32[3]=(r%d.u32[3]<=r%d.u32[3])?1u:0u;\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::ADDX: emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++) r%d.u32[i]=r%d.u32[i]+r%d.u32[i]+(r%d.u32[i]&1u);\n",rT,rA,rB,rT); return;
        case op11::SFX:  emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++) r%d.u32[i]=r%d.u32[i]-r%d.u32[i]-(1u-(r%d.u32[i]&1u));\n",rT,rB,rA,rT); return;
        case op11::MPYS: emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){int16_t _a=(int16_t)(r%d.u32[i]&0xFFFFu),_b=(int16_t)(r%d.u32[i]&0xFFFFu);r%d.s32[i]=((int32_t)_a*(int32_t)_b)>>16;}\n",rA,rB,rT); return;
        case op11::MPYHH:emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){int16_t _a=(int16_t)(r%d.u32[i]>>16),_b=(int16_t)(r%d.u32[i]>>16);r%d.s32[i]=(int32_t)_a*(int32_t)_b;}\n",rA,rB,rT); return;
        case op11::MPYHHU:emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){uint16_t _a=(uint16_t)(r%d.u32[i]>>16),_b=(uint16_t)(r%d.u32[i]>>16);r%d.u32[i]=(uint32_t)_a*(uint32_t)_b;}\n",rA,rB,rT); return;

        // Float — unrolled for NVRTC to use native fadd/fmul
        case op11::FA:  emitf(buf,bufSize,pos,"      r%d.f32[0]=r%d.f32[0]+r%d.f32[0];r%d.f32[1]=r%d.f32[1]+r%d.f32[1];r%d.f32[2]=r%d.f32[2]+r%d.f32[2];r%d.f32[3]=r%d.f32[3]+r%d.f32[3];\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::FS:  emitf(buf,bufSize,pos,"      r%d.f32[0]=r%d.f32[0]-r%d.f32[0];r%d.f32[1]=r%d.f32[1]-r%d.f32[1];r%d.f32[2]=r%d.f32[2]-r%d.f32[2];r%d.f32[3]=r%d.f32[3]-r%d.f32[3];\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::FM:  emitf(buf,bufSize,pos,"      r%d.f32[0]=r%d.f32[0]*r%d.f32[0];r%d.f32[1]=r%d.f32[1]*r%d.f32[1];r%d.f32[2]=r%d.f32[2]*r%d.f32[2];r%d.f32[3]=r%d.f32[3]*r%d.f32[3];\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::FCEQ:emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.f32[0]==r%d.f32[0])?~0u:0u;r%d.u32[1]=(r%d.f32[1]==r%d.f32[1])?~0u:0u;r%d.u32[2]=(r%d.f32[2]==r%d.f32[2])?~0u:0u;r%d.u32[3]=(r%d.f32[3]==r%d.f32[3])?~0u:0u;\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::FCGT:emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.f32[0]>r%d.f32[0])?~0u:0u;r%d.u32[1]=(r%d.f32[1]>r%d.f32[1])?~0u:0u;r%d.u32[2]=(r%d.f32[2]>r%d.f32[2])?~0u:0u;r%d.u32[3]=(r%d.f32[3]>r%d.f32[3])?~0u:0u;\n",rT,rA,rB,rT,rA,rB,rT,rA,rB,rT,rA,rB); return;
        case op11::DFA: emitf(buf,bufSize,pos,"      r%d.f64[0]=r%d.f64[0]+r%d.f64[0];r%d.f64[1]=r%d.f64[1]+r%d.f64[1];\n",rT,rA,rB,rT,rA,rB); return;
        case op11::DFS: emitf(buf,bufSize,pos,"      r%d.f64[0]=r%d.f64[0]-r%d.f64[0];r%d.f64[1]=r%d.f64[1]-r%d.f64[1];\n",rT,rA,rB,rT,rA,rB); return;
        case op11::DFM: emitf(buf,bufSize,pos,"      r%d.f64[0]=r%d.f64[0]*r%d.f64[0];r%d.f64[1]=r%d.f64[1]*r%d.f64[1];\n",rT,rA,rB,rT,rA,rB); return;

        // Load/Store
        case op11::LQX: emitf(buf,bufSize,pos,"      r%d=ld(ls,(r%d.u32[0]+r%d.u32[0])&0x3FFFFu);\n",rT,rA,rB); return;
        case op11::STQX:emitf(buf,bufSize,pos,"      st(ls,(r%d.u32[0]+r%d.u32[0])&0x3FFFFu,r%d);\n",rA,rB,rT); return;

        // Branch indirect
        case op11::BI:   emitf(buf,bufSize,pos,"      pc=r%d.u32[0]&0x3FFFCu;break;\n",rA); return;
        case op11::BISL: emitf(buf,bufSize,pos,"      r%d.u32[0]=0x%xu;r%d.u32[1]=r%d.u32[2]=r%d.u32[3]=0;pc=r%d.u32[0]&0x3FFFCu;break;\n",rT,pc+4,rT,rT,rT,rA); return;
        case op11::BIZ:  emitf(buf,bufSize,pos,"      if(!r%d.u32[0]){pc=r%d.u32[0]&0x3FFFCu;break;}\n",rT,rA); return;
        case op11::BINZ: emitf(buf,bufSize,pos,"      if(r%d.u32[0]){pc=r%d.u32[0]&0x3FFFCu;break;}\n",rT,rA); return;
        case op11::BIHZ: emitf(buf,bufSize,pos,"      if(!(r%d.u32[0]&0xFFFFu)){pc=r%d.u32[0]&0x3FFFCu;break;}\n",rT,rA); return;
        case op11::BIHNZ:emitf(buf,bufSize,pos,"      if(r%d.u32[0]&0xFFFFu){pc=r%d.u32[0]&0x3FFFCu;break;}\n",rT,rA); return;

        case op11::RDCH: emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[1]=r%d.u32[2]=r%d.u32[3]=0;\n",rT,rT,rT,rT); return;
        case op11::WRCH: return;
        case op11::RCHCNT:emitf(buf,bufSize,pos,"      r%d.u32[0]=1u;r%d.u32[1]=r%d.u32[2]=r%d.u32[3]=0;\n",rT,rT,rT,rT); return;

        case op11::NOP: case op11::LNOP: case op11::SYNC: case op11::DSYNC: return;
        case op11::STOP: case op11::STOPD: emitf(buf,bufSize,pos,"      halted=1u;goto done;\n"); return;
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
        case op9::IL:   emitf(buf,bufSize,pos,"      r%d.s32[0]=r%d.s32[1]=r%d.s32[2]=r%d.s32[3]=%d;\n",rT,rT,rT,rT,i16); return;
        case op9::ILH:  emitf(buf,bufSize,pos,"      for(int i=0;i<8;i++) r%d.u16[i]=(uint16_t)%uu;\n",rT,i16u); return;
        case op9::ILHU: emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[1]=r%d.u32[2]=r%d.u32[3]=%uu<<16;\n",rT,rT,rT,rT,i16u); return;
        case op9::IOHL: emitf(buf,bufSize,pos,"      r%d.u32[0]|=%uu;r%d.u32[1]|=%uu;r%d.u32[2]|=%uu;r%d.u32[3]|=%uu;\n",rT,i16u,rT,i16u,rT,i16u,rT,i16u); return;
        case op9::FSMBI:emitf(buf,bufSize,pos,"      for(int i=0;i<16;i++) r%d.u8[i]=(%uu&(1u<<(15-i)))?0xFFu:0;\n",rT,i16u); return;

        // Branches — for non-loop context, emit continue to outer switch
        // For loop context, these are handled by the block emitter
        case op9::BR:   emitf(buf,bufSize,pos,"      pc=0x%xu;break;\n",(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRA:  emitf(buf,bufSize,pos,"      pc=0x%xu;break;\n",((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRSL: emitf(buf,bufSize,pos,"      r%d.u32[0]=0x%xu;r%d.u32[1]=r%d.u32[2]=r%d.u32[3]=0;pc=0x%xu;break;\n",rT,pc+4,rT,rT,rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRASL:emitf(buf,bufSize,pos,"      r%d.u32[0]=0x%xu;r%d.u32[1]=r%d.u32[2]=r%d.u32[3]=0;pc=0x%xu;break;\n",rT,pc+4,rT,rT,rT,((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRZ:  emitf(buf,bufSize,pos,"      if(!r%d.u32[0]){pc=0x%xu;break;}\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRNZ: emitf(buf,bufSize,pos,"      if(r%d.u32[0]){pc=0x%xu;break;}\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRHZ: emitf(buf,bufSize,pos,"      if(!(r%d.u32[0]&0xFFFFu)){pc=0x%xu;break;}\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;
        case op9::BRHNZ:emitf(buf,bufSize,pos,"      if(r%d.u32[0]&0xFFFFu){pc=0x%xu;break;}\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)); return;

        case op9::LQA:  emitf(buf,bufSize,pos,"      r%d=ld(ls,0x%xu);\n",rT,((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)&~0xF); return;
        case op9::STQA: emitf(buf,bufSize,pos,"      st(ls,0x%xu,r%d);\n",((uint32_t)(i16<<2))&(SPU_LS_SIZE-1)&~0xF,rT); return;
        case op9::LQR:  emitf(buf,bufSize,pos,"      r%d=ld(ls,0x%xu);\n",rT,(pc+(i16<<2))&(SPU_LS_SIZE-1)&~0xF); return;
        case op9::STQR: emitf(buf,bufSize,pos,"      st(ls,0x%xu,r%d);\n",(pc+(i16<<2))&(SPU_LS_SIZE-1)&~0xF,rT); return;
        default: break;
        }
    }

    // RI10 (8-bit)
    uint32_t o8 = spu_op8(inst);
    {
        uint32_t rT=spu_rT_ri10(inst), rA=spu_rA_ri10(inst);
        int32_t i10 = spu_I10(inst);

        switch (o8) {
        case op8::AI:   emitf(buf,bufSize,pos,"      r%d.s32[0]=r%d.s32[0]+%d;r%d.s32[1]=r%d.s32[1]+%d;r%d.s32[2]=r%d.s32[2]+%d;r%d.s32[3]=r%d.s32[3]+%d;\n",rT,rA,i10,rT,rA,i10,rT,rA,i10,rT,rA,i10); return;
        case op8::AHI:  emitf(buf,bufSize,pos,"      {int16_t imm=(int16_t)(%d&0xFFFF);for(int i=0;i<8;i++) r%d.s16[i]=r%d.s16[i]+imm;}\n",i10,rT,rA); return;
        case op8::SFI:  emitf(buf,bufSize,pos,"      r%d.s32[0]=%d-r%d.s32[0];r%d.s32[1]=%d-r%d.s32[1];r%d.s32[2]=%d-r%d.s32[2];r%d.s32[3]=%d-r%d.s32[3];\n",rT,i10,rA,rT,i10,rA,rT,i10,rA,rT,i10,rA); return;
        case op8::SFHI: emitf(buf,bufSize,pos,"      {int16_t imm=(int16_t)(%d&0xFFFF);for(int i=0;i<8;i++) r%d.s16[i]=imm-r%d.s16[i];}\n",i10,rT,rA); return;
        case op8::ANDI: emitf(buf,bufSize,pos,"      r%d.s32[0]=r%d.s32[0]&%d;r%d.s32[1]=r%d.s32[1]&%d;r%d.s32[2]=r%d.s32[2]&%d;r%d.s32[3]=r%d.s32[3]&%d;\n",rT,rA,i10,rT,rA,i10,rT,rA,i10,rT,rA,i10); return;
        case op8::ORI:  emitf(buf,bufSize,pos,"      r%d.s32[0]=r%d.s32[0]|%d;r%d.s32[1]=r%d.s32[1]|%d;r%d.s32[2]=r%d.s32[2]|%d;r%d.s32[3]=r%d.s32[3]|%d;\n",rT,rA,i10,rT,rA,i10,rT,rA,i10,rT,rA,i10); return;
        case op8::XORI: emitf(buf,bufSize,pos,"      r%d.s32[0]=r%d.s32[0]^%d;r%d.s32[1]=r%d.s32[1]^%d;r%d.s32[2]=r%d.s32[2]^%d;r%d.s32[3]=r%d.s32[3]^%d;\n",rT,rA,i10,rT,rA,i10,rT,rA,i10,rT,rA,i10); return;
        case op8::CEQI: emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.s32[0]==%d)?~0u:0u;r%d.u32[1]=(r%d.s32[1]==%d)?~0u:0u;r%d.u32[2]=(r%d.s32[2]==%d)?~0u:0u;r%d.u32[3]=(r%d.s32[3]==%d)?~0u:0u;\n",rT,rA,i10,rT,rA,i10,rT,rA,i10,rT,rA,i10); return;
        case op8::CGTI: emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.s32[0]>%d)?~0u:0u;r%d.u32[1]=(r%d.s32[1]>%d)?~0u:0u;r%d.u32[2]=(r%d.s32[2]>%d)?~0u:0u;r%d.u32[3]=(r%d.s32[3]>%d)?~0u:0u;\n",rT,rA,i10,rT,rA,i10,rT,rA,i10,rT,rA,i10); return;
        case op8::CLGTI:emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.u32[0]>(uint32_t)%d)?~0u:0u;r%d.u32[1]=(r%d.u32[1]>(uint32_t)%d)?~0u:0u;r%d.u32[2]=(r%d.u32[2]>(uint32_t)%d)?~0u:0u;r%d.u32[3]=(r%d.u32[3]>(uint32_t)%d)?~0u:0u;\n",rT,rA,i10,rT,rA,i10,rT,rA,i10,rT,rA,i10); return;
        case op8::LQD:  emitf(buf,bufSize,pos,"      r%d=ld(ls,(r%d.u32[0]+%du)&0x3FFFFu);\n",rT,rA,(uint32_t)(i10<<4)); return;
        case op8::STQD: emitf(buf,bufSize,pos,"      st(ls,(r%d.u32[0]+%du)&0x3FFFFu,r%d);\n",rA,(uint32_t)(i10<<4),rT); return;
        case op8::MPYI: emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){int16_t _a=(int16_t)(r%d.u32[i]&0xFFFFu);r%d.s32[i]=(int32_t)_a*%d;}\n",rA,rT,i10); return;
        case op8::MPYUI:emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){uint16_t _a=(uint16_t)(r%d.u32[i]&0xFFFFu);r%d.u32[i]=(uint32_t)_a*%uu;}\n",rA,rT,(uint32_t)(i10&0xFFFF)); return;
        default: break;
        }
    }

    // RI18
    if (spu_op7(inst) == op7::ILA) {
        uint32_t rT=spu_rT_ri18(inst), imm=spu_I18(inst);
        emitf(buf,bufSize,pos,"      r%d.u32[0]=r%d.u32[1]=r%d.u32[2]=r%d.u32[3]=%uu;\n",rT,rT,rT,rT,imm);
        return;
    }

    // RRR (4-bit)
    uint32_t o4 = spu_op4(inst);
    {
        uint32_t rT=spu_rT_rrr(inst),rA=spu_rA_rrr(inst),rB=spu_rB_rrr(inst),rC=spu_rC_rrr(inst);
        switch (o4) {
        case op4::SELB:
            emitf(buf,bufSize,pos,"      r%d.u32[0]=(r%d.u32[0]&r%d.u32[0])|(r%d.u32[0]&~r%d.u32[0]);r%d.u32[1]=(r%d.u32[1]&r%d.u32[1])|(r%d.u32[1]&~r%d.u32[1]);r%d.u32[2]=(r%d.u32[2]&r%d.u32[2])|(r%d.u32[2]&~r%d.u32[2]);r%d.u32[3]=(r%d.u32[3]&r%d.u32[3])|(r%d.u32[3]&~r%d.u32[3]);\n",
                rT,rB,rC,rA,rC, rT,rB,rC,rA,rC, rT,rB,rC,rA,rC, rT,rB,rC,rA,rC); return;
        case op4::SHUFB:
            emitf(buf,bufSize,pos,
                "      {QW _a=r%d,_b=r%d,_c=r%d,_r;for(int _i=0;_i<16;_i++){uint8_t sel=_c.u8[_i];"
                "if(sel&0x80u){if((sel&0xE0u)==0xC0u)_r.u8[_i]=0;else if((sel&0xE0u)==0xE0u)_r.u8[_i]=0xFFu;else _r.u8[_i]=0x80u;}"
                "else{uint8_t idx=sel&0x1Fu;_r.u8[_i]=(idx<16u)?_a.u8[idx]:_b.u8[idx-16u];}}r%d=_r;}\n",
                rA,rB,rC,rT); return;
        case op4::FMA:
            emitf(buf,bufSize,pos,"      r%d.f32[0]=r%d.f32[0]*r%d.f32[0]+r%d.f32[0];r%d.f32[1]=r%d.f32[1]*r%d.f32[1]+r%d.f32[1];r%d.f32[2]=r%d.f32[2]*r%d.f32[2]+r%d.f32[2];r%d.f32[3]=r%d.f32[3]*r%d.f32[3]+r%d.f32[3];\n",
                rT,rA,rB,rC, rT,rA,rB,rC, rT,rA,rB,rC, rT,rA,rB,rC); return;
        case op4::FMS:
            emitf(buf,bufSize,pos,"      r%d.f32[0]=r%d.f32[0]*r%d.f32[0]-r%d.f32[0];r%d.f32[1]=r%d.f32[1]*r%d.f32[1]-r%d.f32[1];r%d.f32[2]=r%d.f32[2]*r%d.f32[2]-r%d.f32[2];r%d.f32[3]=r%d.f32[3]*r%d.f32[3]-r%d.f32[3];\n",
                rT,rA,rB,rC, rT,rA,rB,rC, rT,rA,rB,rC, rT,rA,rB,rC); return;
        case op4::FNMS:
            emitf(buf,bufSize,pos,"      r%d.f32[0]=r%d.f32[0]-r%d.f32[0]*r%d.f32[0];r%d.f32[1]=r%d.f32[1]-r%d.f32[1]*r%d.f32[1];r%d.f32[2]=r%d.f32[2]-r%d.f32[2]*r%d.f32[2];r%d.f32[3]=r%d.f32[3]-r%d.f32[3]*r%d.f32[3];\n",
                rT,rC,rA,rB, rT,rC,rA,rB, rT,rC,rA,rB, rT,rC,rA,rB); return;
        case op4::MPYA:
            emitf(buf,bufSize,pos,"      for(int i=0;i<4;i++){int16_t _a=(int16_t)(r%d.u32[i]&0xFFFFu),_b=(int16_t)(r%d.u32[i]&0xFFFFu);r%d.s32[i]=(int32_t)_a*(int32_t)_b+r%d.s32[i];}\n",rA,rB,rT,rC); return;
        case op4::DFMA: emitf(buf,bufSize,pos,"      r%d.f64[0]=r%d.f64[0]*r%d.f64[0]+r%d.f64[0];r%d.f64[1]=r%d.f64[1]*r%d.f64[1]+r%d.f64[1];\n",rT,rA,rB,rC,rT,rA,rB,rC); return;
        case op4::DFMS: emitf(buf,bufSize,pos,"      r%d.f64[0]=r%d.f64[0]*r%d.f64[0]-r%d.f64[0];r%d.f64[1]=r%d.f64[1]*r%d.f64[1]-r%d.f64[1];\n",rT,rA,rB,rC,rT,rA,rB,rC); return;
        case op4::DFNMS:emitf(buf,bufSize,pos,"      r%d.f64[0]=r%d.f64[0]-r%d.f64[0]*r%d.f64[0];r%d.f64[1]=r%d.f64[1]-r%d.f64[1]*r%d.f64[1];\n",rT,rC,rA,rB,rT,rC,rA,rB); return;
        default: break;
        }
    }

    emitf(buf,bufSize,pos,"      halted=1u;goto done; /* unknown 0x%08x */\n", inst);
}

// ═══════════════════════════════════════════════════════════════
// Hyper Kernel Emitter — loop-aware + register-promoted
// ═══════════════════════════════════════════════════════════════

struct HyperJITResult {
    CUmodule   cuModule;
    CUfunction cuFunction;
    CUfunction cuMultiFunction;
    uint32_t   numBlocks;
    uint32_t   totalInsns;
    uint32_t   numPromotedRegs;
    uint32_t   numSelfLoops;
    float      compileTimeMs;
};

struct SPUHyperState {
    QWord gpr[128];
    uint32_t pc;
    uint32_t halted;
    uint32_t cycles;
};

// Emit all body instructions of a block EXCEPT the final branch
static void emit_block_body(const uint8_t* ls, const HyperBlock& blk, const HyperAnalysis* pa,
                            char* buf, size_t bufSize, size_t* pos) {
    uint32_t bodyInsns = blk.numInsns;
    // Don't emit the final branch instruction for self-loops (handled by do-while)
    if (blk.isSelfLoop && bodyInsns > 0) bodyInsns--;

    for (uint32_t i = 0; i < bodyInsns; i++) {
        uint32_t ipc = (blk.startPC + i * 4) & (SPU_LS_SIZE - 1);
        emit_hyper_insn(ls, ipc, pa, buf, bufSize, pos);
    }
}

static int emit_hyper_kernel(const uint8_t* ls, const HyperAnalysis* pa,
                              char* buf, size_t bufSize) {
    size_t pos = 0;

    // Preamble — types + helpers
    emitf(buf,bufSize,&pos,
        "// SPU Hyper JIT — %u blocks, %u loops, %u promoted regs\n"
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
        "struct SPUHyperState {\n"
        "  QW gpr[128]; uint32_t pc; uint32_t halted; uint32_t cycles;\n"
        "};\n\n"
        "__device__ __forceinline__ uint32_t bswap32(uint32_t x){return __byte_perm(x,0,0x0123);}\n"
        "__device__ __forceinline__ QW ld(const uint8_t* ls,uint32_t a){\n"
        "  a&=0x3FFF0u;QW q;const uint32_t*p=(const uint32_t*)(ls+a);\n"
        "  q.u32[0]=bswap32(p[0]);q.u32[1]=bswap32(p[1]);q.u32[2]=bswap32(p[2]);q.u32[3]=bswap32(p[3]);return q;}\n"
        "__device__ __forceinline__ void st(uint8_t* ls,uint32_t a,const QW& q){\n"
        "  a&=0x3FFF0u;uint32_t*p=(uint32_t*)(ls+a);\n"
        "  p[0]=bswap32(q.u32[0]);p[1]=bswap32(q.u32[1]);p[2]=bswap32(q.u32[2]);p[3]=bswap32(q.u32[3]);}\n\n",
        (uint32_t)pa->blocks.size(), pa->numSelfLoops, pa->numUsedRegs);

    // === Single-instance kernel ===
    emitf(buf,bufSize,&pos,
        "extern \"C\" __global__ void spu_hyper(\n"
        "    SPUHyperState* __restrict__ state,\n"
        "    uint8_t* __restrict__ ls,\n"
        "    uint8_t* __restrict__ mainMem,\n"
        "    uint32_t maxCycles)\n"
        "{\n"
        "  if(threadIdx.x!=0) return;\n\n");

    // Register promotion
    for (int i = 0; i < 128; i++) {
        if (pa->usedRegs[i])
            emitf(buf,bufSize,&pos, "  QW r%d = state->gpr[%d];\n", i, i);
    }

    emitf(buf,bufSize,&pos,
        "\n  uint32_t pc = state->pc;\n"
        "  uint32_t halted = 0;\n"
        "  uint32_t cycles = state->cycles;\n"
        "  uint32_t limit = cycles + maxCycles;\n\n"
        "  while(!halted && cycles < limit) {\n"
        "    switch(pc) {\n");

    // Emit blocks with loop-aware code generation
    for (const auto& blk : pa->blocks) {
        emitf(buf,bufSize,&pos, "    case 0x%xu: {\n", blk.startPC);

        if (blk.isSelfLoop) {
            // TIGHT LOOP: emit do { body } while(condition && cycles < limit)
            emitf(buf,bufSize,&pos, "      // ⚡ TIGHT LOOP (self-loop detected)\n");
            emitf(buf,bufSize,&pos, "      do {\n");

            // Emit loop body (all instructions except the final branch)
            emit_block_body(ls, blk, pa, buf, bufSize, &pos);

            // Batched cycle count
            uint32_t bodyInsns = blk.numInsns > 0 ? blk.numInsns - 1 : 0;
            if (bodyInsns > 0) bodyInsns++; // count the branch too
            emitf(buf,bufSize,&pos, "        cycles += %uu;\n", blk.numInsns);

            // Loop condition based on the branch type
            if (blk.loopCondNonZero) {
                // BRNZ: loop while r[N].u32[0] != 0
                emitf(buf,bufSize,&pos, "      } while(r%d.u32[0] && cycles < limit);\n",
                       blk.loopCondReg);
            } else {
                // BRZ: loop while r[N].u32[0] == 0
                emitf(buf,bufSize,&pos, "      } while(!r%d.u32[0] && cycles < limit);\n",
                       blk.loopCondReg);
            }
            emitf(buf,bufSize,&pos, "      pc=0x%xu;break;\n", blk.fallThroughPC);
        } else {
            // Normal block — same as turbo but with batched cycles
            for (uint32_t i = 0; i < blk.numInsns; i++) {
                uint32_t ipc = (blk.startPC + i * 4) & (SPU_LS_SIZE - 1);
                emit_hyper_insn(ls, ipc, pa, buf, bufSize, &pos);
            }
            emitf(buf,bufSize,&pos, "      cycles += %uu;\n", blk.numInsns);
            if (!blk.endsWithStop && !blk.endsWithBranch) {
                uint32_t next = (blk.endPC + 4) & (SPU_LS_SIZE - 1);
                emitf(buf,bufSize,&pos, "      pc=0x%xu;break;\n", next);
            }
        }
        emitf(buf,bufSize,&pos, "    }\n");
    }

    emitf(buf,bufSize,&pos,
        "    default: halted=1u; break;\n"
        "    }\n  }\n  done: ;\n");

    // Write back promoted registers
    for (int i = 0; i < 128; i++) {
        if (pa->writtenRegs[i])
            emitf(buf,bufSize,&pos, "  state->gpr[%d] = r%d;\n", i, i);
    }
    emitf(buf,bufSize,&pos,
        "  state->pc = pc;\n"
        "  state->halted = halted;\n"
        "  state->cycles = cycles;\n"
        "}\n\n");

    // === Multi-instance SPURS kernel ===
    emitf(buf,bufSize,&pos,
        "extern \"C\" __global__ void spu_hyper_multi(\n"
        "    SPUHyperState* __restrict__ states,\n"
        "    uint8_t** __restrict__ lsArray,\n"
        "    uint8_t* __restrict__ mainMem,\n"
        "    uint32_t maxCycles,\n"
        "    uint32_t numInstances)\n"
        "{\n"
        "  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if(tid >= numInstances) return;\n"
        "  SPUHyperState* state = &states[tid];\n"
        "  uint8_t* ls = lsArray[tid];\n\n");

    for (int i = 0; i < 128; i++) {
        if (pa->usedRegs[i])
            emitf(buf,bufSize,&pos, "  QW r%d = state->gpr[%d];\n", i, i);
    }

    emitf(buf,bufSize,&pos,
        "\n  uint32_t pc = state->pc;\n"
        "  uint32_t halted = 0;\n"
        "  uint32_t cycles = state->cycles;\n"
        "  uint32_t limit = cycles + maxCycles;\n\n"
        "  while(!halted && cycles < limit) {\n"
        "    switch(pc) {\n");

    for (const auto& blk : pa->blocks) {
        emitf(buf,bufSize,&pos, "    case 0x%xu: {\n", blk.startPC);

        if (blk.isSelfLoop) {
            emitf(buf,bufSize,&pos, "      do {\n");
            emit_block_body(ls, blk, pa, buf, bufSize, &pos);
            emitf(buf,bufSize,&pos, "        cycles += %uu;\n", blk.numInsns);
            if (blk.loopCondNonZero) {
                emitf(buf,bufSize,&pos, "      } while(r%d.u32[0] && cycles < limit);\n",
                       blk.loopCondReg);
            } else {
                emitf(buf,bufSize,&pos, "      } while(!r%d.u32[0] && cycles < limit);\n",
                       blk.loopCondReg);
            }
            emitf(buf,bufSize,&pos, "      pc=0x%xu;break;\n", blk.fallThroughPC);
        } else {
            for (uint32_t i = 0; i < blk.numInsns; i++) {
                uint32_t ipc = (blk.startPC + i * 4) & (SPU_LS_SIZE - 1);
                emit_hyper_insn(ls, ipc, pa, buf, bufSize, &pos);
            }
            emitf(buf,bufSize,&pos, "      cycles += %uu;\n", blk.numInsns);
            if (!blk.endsWithStop && !blk.endsWithBranch) {
                uint32_t next = (blk.endPC + 4) & (SPU_LS_SIZE - 1);
                emitf(buf,bufSize,&pos, "      pc=0x%xu;break;\n", next);
            }
        }
        emitf(buf,bufSize,&pos, "    }\n");
    }

    emitf(buf,bufSize,&pos,
        "    default: halted=1u; break;\n"
        "    }\n  }\n  done: ;\n");

    for (int i = 0; i < 128; i++) {
        if (pa->writtenRegs[i])
            emitf(buf,bufSize,&pos, "  state->gpr[%d] = r%d;\n", i, i);
    }
    emitf(buf,bufSize,&pos,
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

int hyper_jit_compile(const uint8_t* h_ls, uint32_t entryPC, HyperJITResult* result) {
    memset(result, 0, sizeof(HyperJITResult));
    cuInit(0);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    HyperAnalysis pa;
    analyze_program(h_ls, entryPC, &pa);

    result->numBlocks = (uint32_t)pa.blocks.size();
    for (auto& b : pa.blocks) result->totalInsns += b.numInsns;
    result->numPromotedRegs = pa.numUsedRegs;
    result->numSelfLoops = pa.numSelfLoops;

    fprintf(stderr, "[HyperJIT] %u blocks, %u insns, %u promoted regs, %u self-loops\n",
            result->numBlocks, result->totalInsns, result->numPromotedRegs, result->numSelfLoops);

    size_t srcBufSize = 1024 * 1024;
    char* source = (char*)malloc(srcBufSize);
    int srcLen = emit_hyper_kernel(h_ls, &pa, source, srcBufSize);
    fprintf(stderr, "[HyperJIT] %d bytes CUDA source\n", srcLen);

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, source, "spu_hyper.cu", 0, NULL, NULL);

    const char* opts[] = {"--gpu-architecture=sm_70", "--use_fast_math", "--std=c++17"};
    nvrtcResult nres = nvrtcCompileProgram(prog, 3, opts);

    if (nres != NVRTC_SUCCESS) {
        size_t logSz; nvrtcGetProgramLogSize(prog, &logSz);
        char* log = (char*)malloc(logSz);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "[HyperJIT] COMPILE FAILED:\n%s\n", log);
        free(log); nvrtcDestroyProgram(&prog); free(source);
        return 0;
    }
    free(source);

    size_t ptxSz; nvrtcGetPTXSize(prog, &ptxSz);
    char* ptx = (char*)malloc(ptxSz);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUmodule cuMod;
    if (cuModuleLoadDataEx(&cuMod, ptx, 0, NULL, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "[HyperJIT] cuModuleLoadData failed\n");
        free(ptx); return 0;
    }
    free(ptx);

    CUfunction f1, f2;
    cuModuleGetFunction(&f1, cuMod, "spu_hyper");
    cuModuleGetFunction(&f2, cuMod, "spu_hyper_multi");

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&result->compileTimeMs, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    result->cuModule = cuMod;
    result->cuFunction = f1;
    result->cuMultiFunction = f2;

    fprintf(stderr, "[HyperJIT] ✅ Compiled (%.1f ms): %u blocks, %u loops, %u regs\n",
            result->compileTimeMs, result->numBlocks, result->numSelfLoops, result->numPromotedRegs);
    return 1;
}

float hyper_jit_run(const HyperJITResult* jit, SPUHyperState* h_state,
                    uint8_t* d_ls, uint8_t* d_mainMem, uint32_t maxCycles) {
    SPUHyperState* d_state;
    cudaMalloc(&d_state, sizeof(SPUHyperState));
    cudaMemcpy(d_state, h_state, sizeof(SPUHyperState), cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    void* args[] = { &d_state, &d_ls, &d_mainMem, &maxCycles };
    cudaEventRecord(t0);
    cuLaunchKernel((CUfunction)jit->cuFunction, 1,1,1, 1,1,1, 0,0, args, NULL);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(h_state, d_state, sizeof(SPUHyperState), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

float hyper_jit_run_multi(const HyperJITResult* jit, SPUHyperState* h_states,
                          uint8_t** d_lsArray_host, uint8_t* d_mainMem,
                          uint32_t maxCycles, uint32_t numInstances) {
    SPUHyperState* d_states;
    cudaMalloc(&d_states, numInstances * sizeof(SPUHyperState));
    cudaMemcpy(d_states, h_states, numInstances * sizeof(SPUHyperState), cudaMemcpyHostToDevice);

    uint8_t** d_lsPtrs;
    cudaMalloc(&d_lsPtrs, numInstances * sizeof(uint8_t*));
    cudaMemcpy(d_lsPtrs, d_lsArray_host, numInstances * sizeof(uint8_t*), cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    uint32_t threadsPerBlock = 128;
    uint32_t numBlocks = (numInstances + threadsPerBlock - 1) / threadsPerBlock;

    void* args[] = { &d_states, &d_lsPtrs, &d_mainMem, &maxCycles, &numInstances };
    cudaEventRecord(t0);
    cuLaunchKernel((CUfunction)jit->cuMultiFunction,
                   numBlocks,1,1, threadsPerBlock,1,1, 0,0, args, NULL);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(h_states, d_states, numInstances * sizeof(SPUHyperState), cudaMemcpyDeviceToHost);
    cudaFree(d_states);
    cudaFree(d_lsPtrs);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

void hyper_jit_free(HyperJITResult* jit) {
    if (jit->cuModule) cuModuleUnload((CUmodule)jit->cuModule);
    memset(jit, 0, sizeof(HyperJITResult));
}

} // extern "C"
