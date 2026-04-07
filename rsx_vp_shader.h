#pragma once
// rsx_vp_shader.h — RSX Vertex Program → CUDA Kernel Translator
//
// Decodes NV40 VP microcode (128-bit instructions) and emits CUDA C source
// for NVRTC compilation. Each VP becomes a __global__ kernel that transforms
// vertices in parallel on V100 CUDA cores.
//
// Part of Project Megakernel — hybrid RSX: VP on CUDA, rasterize via Vulkan.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace rsx {

// ═══════════════════════════════════════════════════════════════════
// VP Instruction Word Layouts (from RPCS3 RSXVertexProgram.h)
// ═══════════════════════════════════════════════════════════════════

// Register types
enum VPRegType : uint32_t {
    VP_REG_TEMP     = 1,
    VP_REG_INPUT    = 2,
    VP_REG_CONSTANT = 3,
};

// Vector opcodes (5-bit, in D1.vec_opcode)
enum VPVecOp : uint32_t {
    VP_VEC_NOP = 0x00, VP_VEC_MOV = 0x01, VP_VEC_MUL = 0x02,
    VP_VEC_ADD = 0x03, VP_VEC_MAD = 0x04, VP_VEC_DP3 = 0x05,
    VP_VEC_DPH = 0x06, VP_VEC_DP4 = 0x07, VP_VEC_DST = 0x08,
    VP_VEC_MIN = 0x09, VP_VEC_MAX = 0x0A, VP_VEC_SLT = 0x0B,
    VP_VEC_SGE = 0x0C, VP_VEC_ARL = 0x0D, VP_VEC_FRC = 0x0E,
    VP_VEC_FLR = 0x0F, VP_VEC_SEQ = 0x10, VP_VEC_SFL = 0x11,
    VP_VEC_SGT = 0x12, VP_VEC_SLE = 0x13, VP_VEC_SNE = 0x14,
    VP_VEC_STR = 0x15, VP_VEC_SSG = 0x16, VP_VEC_TXL = 0x19,
};

// Scalar opcodes (5-bit, in D1.sca_opcode)
enum VPScaOp : uint32_t {
    VP_SCA_NOP = 0x00, VP_SCA_MOV = 0x01, VP_SCA_RCP = 0x02,
    VP_SCA_RCC = 0x03, VP_SCA_RSQ = 0x04, VP_SCA_EXP = 0x05,
    VP_SCA_LOG = 0x06, VP_SCA_LIT = 0x07, VP_SCA_BRA = 0x08,
    VP_SCA_BRI = 0x09, VP_SCA_CAL = 0x0A, VP_SCA_CLI = 0x0B,
    VP_SCA_RET = 0x0C, VP_SCA_LG2 = 0x0D, VP_SCA_EX2 = 0x0E,
    VP_SCA_SIN = 0x0F, VP_SCA_COS = 0x10, VP_SCA_BRB = 0x11,
    VP_SCA_CLB = 0x12, VP_SCA_PSH = 0x13, VP_SCA_POP = 0x14,
};

// Swizzle component indices
static const char* SWZ_COMP[] = {"x", "y", "z", "w"};

// ═══════════════════════════════════════════════════════════════════
// VP Instruction Decoder
// ═══════════════════════════════════════════════════════════════════

struct VPDecodedSrc {
    uint32_t regType;   // VP_REG_TEMP/INPUT/CONSTANT
    uint32_t regIdx;    // register index
    uint32_t swzX, swzY, swzZ, swzW;  // 0-3 each
    bool neg;
    bool abs;
};

struct VPDecodedInsn {
    uint32_t vecOp;
    uint32_t scaOp;

    // Sources (3 for vec, src2 scalar input uses scalar channel)
    VPDecodedSrc src[3];

    // Vector destination
    uint32_t vecDstTmp;     // temp register index (D0.dst_tmp)
    uint32_t vecDstOut;     // output register index (D3.dst)
    bool vecWriteResult;    // D0.vec_result — write to output vs temp
    bool vecMaskX, vecMaskY, vecMaskZ, vecMaskW;

    // Scalar destination
    uint32_t scaDstTmp;     // scalar temp dest (D3.sca_dst_tmp)
    bool scaMaskX, scaMaskY, scaMaskZ, scaMaskW;

    // Modifiers
    bool saturate;
    bool endFlag;

    // Constants
    uint32_t constIdx;      // D1.const_src
    uint32_t inputIdx;      // D1.input_src
};

// Decode a single 128-bit VP instruction (4 × u32)
inline VPDecodedInsn vp_decode(const uint32_t d[4]) {
    VPDecodedInsn insn = {};

    // D0 fields
    uint32_t d0 = d[0];
    insn.vecMaskX     = !((d0 >> 8) & 0x3); // mask_x: 0=write, non-0=skip (inverted)
    insn.vecMaskY     = !((d0 >> 6) & 0x3);
    insn.vecMaskZ     = !((d0 >> 4) & 0x3);
    insn.vecMaskW     = !((d0 >> 2) & 0x3);
    insn.vecDstTmp    = (d0 >> 14) & 0x3F;   // dst_tmp: 6 bits
    bool src0Abs      = (d0 >> 21) & 1;
    bool src1Abs      = (d0 >> 22) & 1;
    bool src2Abs      = (d0 >> 23) & 1;
    insn.saturate     = (d0 >> 25) & 1;       // staturate (sic)
    insn.vecWriteResult = (d0 >> 29) & 1;     // vec_result

    // D1 fields
    uint32_t d1 = d[1];
    uint32_t src0h    = d1 & 0xFF;            // src0h: 8 bits
    insn.inputIdx     = (d1 >> 8) & 0xF;      // input_src: 4 bits
    insn.constIdx     = (d1 >> 12) & 0x3FF;   // const_src: 10 bits
    insn.vecOp        = (d1 >> 22) & 0x1F;    // vec_opcode: 5 bits
    insn.scaOp        = (d1 >> 27) & 0x1F;    // sca_opcode: 5 bits

    // D2 fields
    uint32_t d2 = d[2];
    uint32_t src2h    = d2 & 0x3F;            // src2h: 6 bits
    uint32_t src1raw  = (d2 >> 6) & 0x1FFFF;  // src1: 17 bits
    uint32_t src0l    = (d2 >> 23) & 0x1FF;   // src0l: 9 bits

    // D3 fields
    uint32_t d3 = d[3];
    insn.endFlag       = d3 & 1;              // end flag
    insn.vecDstOut     = (d3 >> 2) & 0x1F;    // dst: 5 bits (output reg)
    insn.scaDstTmp     = (d3 >> 7) & 0x3F;    // sca_dst_tmp: 6 bits
    // Write masks in D3 (1-bit each, 1=enabled)
    // Note: D3 masks are the actual per-channel enables
    insn.vecMaskW      = (d3 >> 13) & 1;
    insn.vecMaskZ      = (d3 >> 14) & 1;
    insn.vecMaskY      = (d3 >> 15) & 1;
    insn.vecMaskX      = (d3 >> 16) & 1;
    insn.scaMaskW      = (d3 >> 17) & 1;
    insn.scaMaskZ      = (d3 >> 18) & 1;
    insn.scaMaskY      = (d3 >> 19) & 1;
    insn.scaMaskX      = (d3 >> 20) & 1;

    // Reconstruct source operands
    // SRC0: { src0l[8:0] | src0h[7:0] } = 17 bits
    uint32_t src0raw = (src0h << 9) | src0l;

    // Decode each SRC (17-bit packed: regType[1:0], tmpSrc[7:2], swzW[9:8], swzZ[11:10], swzY[13:12], swzX[15:14], neg[16])
    auto decodeSrc = [](uint32_t raw) -> VPDecodedSrc {
        VPDecodedSrc s = {};
        s.regType = raw & 0x3;
        s.regIdx  = (raw >> 2) & 0x3F;
        s.swzW    = (raw >> 8) & 0x3;
        s.swzZ    = (raw >> 10) & 0x3;
        s.swzY    = (raw >> 12) & 0x3;
        s.swzX    = (raw >> 14) & 0x3;
        s.neg     = (raw >> 16) & 1;
        return s;
    };

    insn.src[0] = decodeSrc(src0raw);
    insn.src[1] = decodeSrc(src1raw);
    // SRC2: { src2h[5:0] | src2l[10:0] } = 17 bits
    uint32_t src2l = (d3 >> 21) & 0x7FF;
    uint32_t src2raw = (src2h << 11) | src2l;
    insn.src[2] = decodeSrc(src2raw);

    insn.src[0].abs = src0Abs;
    insn.src[1].abs = src1Abs;
    insn.src[2].abs = src2Abs;

    return insn;
}

// ═══════════════════════════════════════════════════════════════════
// CUDA Source Code Emitter
// ═══════════════════════════════════════════════════════════════════

struct VPEmitter {
    std::string code;
    int tempRegsUsed;    // track highest temp reg
    int outputsUsed;     // bitmask of outputs written
    int instrCount;

    VPEmitter() : tempRegsUsed(0), outputsUsed(0), instrCount(0) {}

    // Emit source register reference with swizzle
    std::string emitSrc(const VPDecodedInsn& insn, int srcIdx, uint32_t inputIdx, uint32_t constIdx) {
        const VPDecodedSrc& s = insn.src[srcIdx];
        std::string reg;

        switch (s.regType) {
        case VP_REG_TEMP:
            reg = "t" + std::to_string(s.regIdx);
            if ((int)s.regIdx >= tempRegsUsed) tempRegsUsed = s.regIdx + 1;
            break;
        case VP_REG_INPUT:
            reg = "in_attr[" + std::to_string(inputIdx) + "]";
            break;
        case VP_REG_CONSTANT:
            reg = "c[" + std::to_string(constIdx) + "]";
            break;
        default:
            reg = "float4(0,0,0,0)";
            break;
        }

        // Apply swizzle
        bool identitySwz = (s.swzX == 0 && s.swzY == 1 && s.swzZ == 2 && s.swzW == 3);
        if (!identitySwz) {
            reg = "make_float4(" + reg + "." + SWZ_COMP[s.swzX] + ","
                + reg + "." + SWZ_COMP[s.swzY] + ","
                + reg + "." + SWZ_COMP[s.swzZ] + ","
                + reg + "." + SWZ_COMP[s.swzW] + ")";
        }

        // Apply abs
        if (s.abs) {
            reg = "f4abs(" + reg + ")";
        }

        // Apply negate
        if (s.neg) {
            reg = "f4neg(" + reg + ")";
        }

        return reg;
    }

    // Generate write mask string
    std::string writeMask(bool x, bool y, bool z, bool w) {
        std::string m;
        if (x) m += "x";
        if (y) m += "y";
        if (z) m += "z";
        if (w) m += "w";
        return m;
    }

    // Emit masked assignment: dst.mask = expr
    void emitMaskedWrite(const std::string& dst, bool mx, bool my, bool mz, bool mw,
                         const std::string& expr, bool sat) {
        std::string val = sat ? ("f4sat(" + expr + ")") : expr;
        if (mx && my && mz && mw) {
            code += "  " + dst + " = " + val + ";\n";
        } else {
            if (mx) code += "  " + dst + ".x = (" + val + ").x;\n";
            if (my) code += "  " + dst + ".y = (" + val + ").y;\n";
            if (mz) code += "  " + dst + ".z = (" + val + ").z;\n";
            if (mw) code += "  " + dst + ".w = (" + val + ").w;\n";
        }
    }

    // Emit vector operation
    void emitVecOp(const VPDecodedInsn& insn) {
        if (insn.vecOp == VP_VEC_NOP) return;
        if (!insn.vecMaskX && !insn.vecMaskY && !insn.vecMaskZ && !insn.vecMaskW) return;

        std::string s0 = emitSrc(insn, 0, insn.inputIdx, insn.constIdx);
        std::string s1 = emitSrc(insn, 1, insn.inputIdx, insn.constIdx);
        std::string s2 = emitSrc(insn, 2, insn.inputIdx, insn.constIdx);

        // Determine destination
        std::string dst;
        if (insn.vecWriteResult) {
            dst = "out_attr[" + std::to_string(insn.vecDstOut) + "]";
            outputsUsed |= (1 << insn.vecDstOut);
        } else {
            dst = "t" + std::to_string(insn.vecDstTmp);
            if ((int)insn.vecDstTmp >= tempRegsUsed) tempRegsUsed = insn.vecDstTmp + 1;
        }

        std::string expr;
        switch (insn.vecOp) {
        case VP_VEC_MOV: expr = s0; break;
        case VP_VEC_MUL: expr = "f4mul(" + s0 + "," + s1 + ")"; break;
        case VP_VEC_ADD: expr = "f4add(" + s0 + "," + s2 + ")"; break;
        case VP_VEC_MAD: expr = "f4add(f4mul(" + s0 + "," + s1 + ")," + s2 + ")"; break;
        case VP_VEC_DP3: expr = "f4splat(dot3(" + s0 + "," + s1 + "))"; break;
        case VP_VEC_DPH: expr = "f4splat(dot3(" + s0 + "," + s1 + ")+" + s1 + ".w)"; break;
        case VP_VEC_DP4: expr = "f4splat(dot4(" + s0 + "," + s1 + "))"; break;
        case VP_VEC_DST:
            expr = "make_float4(1.0f," + s0 + ".y*" + s1 + ".y," + s0 + ".z," + s1 + ".w)";
            break;
        case VP_VEC_MIN: expr = "f4min(" + s0 + "," + s1 + ")"; break;
        case VP_VEC_MAX: expr = "f4max(" + s0 + "," + s1 + ")"; break;
        case VP_VEC_SLT: expr = "f4slt(" + s0 + "," + s1 + ")"; break;
        case VP_VEC_SGE: expr = "f4sge(" + s0 + "," + s1 + ")"; break;
        case VP_VEC_SEQ: expr = "f4seq(" + s0 + "," + s1 + ")"; break;
        case VP_VEC_SGT: expr = "f4sgt(" + s0 + "," + s1 + ")"; break;
        case VP_VEC_SLE: expr = "f4sle(" + s0 + "," + s1 + ")"; break;
        case VP_VEC_SNE: expr = "f4sne(" + s0 + "," + s1 + ")"; break;
        case VP_VEC_SFL: expr = "make_float4(0,0,0,0)"; break;
        case VP_VEC_STR: expr = "make_float4(1,1,1,1)"; break;
        case VP_VEC_FRC: expr = "f4frc(" + s0 + ")"; break;
        case VP_VEC_FLR: expr = "f4flr(" + s0 + ")"; break;
        case VP_VEC_SSG: expr = "f4ssg(" + s0 + ")"; break;
        case VP_VEC_ARL:
            code += "  arl = (int)floorf((" + s0 + ").x);\n";
            return;
        default:
            code += "  // UNHANDLED vec op " + std::to_string(insn.vecOp) + "\n";
            return;
        }

        emitMaskedWrite(dst, insn.vecMaskX, insn.vecMaskY, insn.vecMaskZ, insn.vecMaskW,
                        expr, insn.saturate);
    }

    // Emit scalar operation
    void emitScaOp(const VPDecodedInsn& insn) {
        if (insn.scaOp == VP_SCA_NOP) return;
        if (!insn.scaMaskX && !insn.scaMaskY && !insn.scaMaskZ && !insn.scaMaskW) return;

        // Scalar ops use src2 as input, take .x component
        std::string s = emitSrc(insn, 2, insn.inputIdx, insn.constIdx);
        std::string sx = "(" + s + ").x";

        std::string dst = "t" + std::to_string(insn.scaDstTmp);
        if ((int)insn.scaDstTmp >= tempRegsUsed) tempRegsUsed = insn.scaDstTmp + 1;

        std::string expr;
        switch (insn.scaOp) {
        case VP_SCA_MOV: expr = "f4splat(" + sx + ")"; break;
        case VP_SCA_RCP: expr = "f4splat(1.0f/(" + sx + "))"; break;
        case VP_SCA_RCC: expr = "f4splat(fminf(fmaxf(1.0f/(" + sx + "),5.42101e-20f),1.884467e+19f))"; break;
        case VP_SCA_RSQ: expr = "f4splat(rsqrtf(fabsf(" + sx + ")))"; break;
        case VP_SCA_EXP: expr = "f4splat(exp2f(" + sx + "))"; break;
        case VP_SCA_LOG: expr = "f4splat(log2f(fabsf(" + sx + ")))"; break;
        case VP_SCA_LG2: expr = "f4splat(log2f(fabsf(" + sx + ")))"; break;
        case VP_SCA_EX2: expr = "f4splat(exp2f(" + sx + "))"; break;
        case VP_SCA_SIN: expr = "f4splat(sinf(" + sx + "))"; break;
        case VP_SCA_COS: expr = "f4splat(cosf(" + sx + "))"; break;
        case VP_SCA_LIT: {
            // LIT: dst = (1.0, max(src.x,0), src.x>0 ? pow(max(src.y,0), clamp(src.w,-128,128)) : 0, 1.0)
            std::string sy = "(" + s + ").y";
            std::string sw = "(" + s + ").w";
            expr = "make_float4(1.0f, fmaxf(" + sx + ",0.0f), "
                   + sx + ">0.0f ? powf(fmaxf(" + sy + ",0.0f), fminf(fmaxf(" + sw + ",-128.0f),128.0f)) : 0.0f, 1.0f)";
            break;
        }
        // Flow control — emit as comments for now
        case VP_SCA_BRA: case VP_SCA_BRI: case VP_SCA_CAL: case VP_SCA_CLI:
        case VP_SCA_RET: case VP_SCA_BRB: case VP_SCA_CLB: case VP_SCA_PSH:
        case VP_SCA_POP:
            code += "  // FLOW: sca_op=" + std::to_string(insn.scaOp) + "\n";
            return;
        default:
            code += "  // UNHANDLED sca op " + std::to_string(insn.scaOp) + "\n";
            return;
        }

        emitMaskedWrite(dst, insn.scaMaskX, insn.scaMaskY, insn.scaMaskZ, insn.scaMaskW,
                        expr, insn.saturate);
    }
};

// ═══════════════════════════════════════════════════════════════════
// VP → CUDA Kernel Translator
// ═══════════════════════════════════════════════════════════════════

// Output attribute indices (matches NV40/RSX output register mapping)
static const char* VP_OUTPUT_NAMES[] = {
    "HPOS",     // 0  — clip position
    "COL0",     // 1  — front diffuse
    "COL1",     // 2  — front specular
    "BFC0",     // 3  — back diffuse
    "BFC1",     // 4  — back specular
    "FOGC",     // 5  — fog coordinate
    "PSZ",      // 6  — point size
    "TEX0",     // 7
    "TEX1",     // 8
    "TEX2",     // 9
    "TEX3",     // 10
    "TEX4",     // 11
    "TEX5",     // 12
    "TEX6",     // 13
    "TEX7",     // 14
    "TEX8",     // 15
    "TEX9",     // 16
};

// Translate a VP program to a CUDA kernel string
// vpData: array of uint32_t words (4 words per instruction)
// vpLen:  number of words (must be multiple of 4)
// returns: CUDA source string ready for NVRTC
inline std::string vp_translate_to_cuda(const uint32_t* vpData, uint32_t vpLen,
                                         uint32_t vpStart = 0) {
    VPEmitter emit;
    uint32_t numInsns = vpLen / 4;

    // Decode and emit all instructions
    for (uint32_t i = vpStart; i < numInsns; i++) {
        const uint32_t* d = &vpData[i * 4];
        VPDecodedInsn insn = vp_decode(d);

        emit.code += "  // insn " + std::to_string(i) + ": vec=" +
                     std::to_string(insn.vecOp) + " sca=" + std::to_string(insn.scaOp) + "\n";
        emit.emitVecOp(insn);
        emit.emitScaOp(insn);
        emit.instrCount++;

        if (insn.endFlag) break;
    }

    // Build complete CUDA kernel source
    std::string src;
    src += "// Auto-generated RSX VP kernel (" + std::to_string(emit.instrCount) + " instructions)\n\n";

    // Helper functions
    src += R"(
__device__ float4 f4splat(float v) { return make_float4(v,v,v,v); }
__device__ float4 f4neg(float4 a) { return make_float4(-a.x,-a.y,-a.z,-a.w); }
__device__ float4 f4abs(float4 a) { return make_float4(fabsf(a.x),fabsf(a.y),fabsf(a.z),fabsf(a.w)); }
__device__ float4 f4mul(float4 a, float4 b) { return make_float4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w); }
__device__ float4 f4add(float4 a, float4 b) { return make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w); }
__device__ float4 f4min(float4 a, float4 b) { return make_float4(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z),fminf(a.w,b.w)); }
__device__ float4 f4max(float4 a, float4 b) { return make_float4(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z),fmaxf(a.w,b.w)); }
__device__ float4 f4sat(float4 a) { return make_float4(fminf(fmaxf(a.x,0.f),1.f),fminf(fmaxf(a.y,0.f),1.f),fminf(fmaxf(a.z,0.f),1.f),fminf(fmaxf(a.w,0.f),1.f)); }
__device__ float4 f4frc(float4 a) { return make_float4(a.x-floorf(a.x),a.y-floorf(a.y),a.z-floorf(a.z),a.w-floorf(a.w)); }
__device__ float4 f4flr(float4 a) { return make_float4(floorf(a.x),floorf(a.y),floorf(a.z),floorf(a.w)); }
__device__ float4 f4ssg(float4 a) { return make_float4(a.x>0?1.f:(a.x<0?-1.f:0.f),a.y>0?1.f:(a.y<0?-1.f:0.f),a.z>0?1.f:(a.z<0?-1.f:0.f),a.w>0?1.f:(a.w<0?-1.f:0.f)); }
__device__ float dot3(float4 a, float4 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ float dot4(float4 a, float4 b) { return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w; }
__device__ float4 f4slt(float4 a, float4 b) { return make_float4(a.x<b.x?1.f:0.f,a.y<b.y?1.f:0.f,a.z<b.z?1.f:0.f,a.w<b.w?1.f:0.f); }
__device__ float4 f4sge(float4 a, float4 b) { return make_float4(a.x>=b.x?1.f:0.f,a.y>=b.y?1.f:0.f,a.z>=b.z?1.f:0.f,a.w>=b.w?1.f:0.f); }
__device__ float4 f4seq(float4 a, float4 b) { return make_float4(a.x==b.x?1.f:0.f,a.y==b.y?1.f:0.f,a.z==b.z?1.f:0.f,a.w==b.w?1.f:0.f); }
__device__ float4 f4sgt(float4 a, float4 b) { return make_float4(a.x>b.x?1.f:0.f,a.y>b.y?1.f:0.f,a.z>b.z?1.f:0.f,a.w>b.w?1.f:0.f); }
__device__ float4 f4sle(float4 a, float4 b) { return make_float4(a.x<=b.x?1.f:0.f,a.y<=b.y?1.f:0.f,a.z<=b.z?1.f:0.f,a.w<=b.w?1.f:0.f); }
__device__ float4 f4sne(float4 a, float4 b) { return make_float4(a.x!=b.x?1.f:0.f,a.y!=b.y?1.f:0.f,a.z!=b.z?1.f:0.f,a.w!=b.w?1.f:0.f); }
)";

    // Kernel signature
    src += "\nextern \"C\" __global__\n";
    src += "void rsx_vp_kernel(\n";
    src += "    const float4* __restrict__ vertices,  // input: N vertices × 16 attributes\n";
    src += "    float4* __restrict__ output,           // output: N vertices × 17 output slots\n";
    src += "    const float4* __restrict__ c,          // VP constants (256 vec4s)\n";
    src += "    int numVertices,\n";
    src += "    int numInputAttribs                    // active input attributes per vertex\n";
    src += ") {\n";
    src += "  int vid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    src += "  if (vid >= numVertices) return;\n\n";

    // Load input attributes
    src += "  // Load vertex input attributes\n";
    src += "  float4 in_attr[16];\n";
    src += "  for (int i = 0; i < 16; i++) {\n";
    src += "    in_attr[i] = (i < numInputAttribs) ? vertices[vid * numInputAttribs + i] : make_float4(0,0,0,1);\n";
    src += "  }\n\n";

    // Declare temp registers
    src += "  // Temp registers\n";
    int numTemps = (emit.tempRegsUsed > 0) ? emit.tempRegsUsed : 1;
    if (numTemps > 64) numTemps = 64;
    for (int i = 0; i < numTemps; i++) {
        src += "  float4 t" + std::to_string(i) + " = make_float4(0,0,0,0);\n";
    }

    // Declare output attributes
    src += "\n  // Output attributes\n";
    src += "  float4 out_attr[17];\n";
    src += "  for (int i = 0; i < 17; i++) out_attr[i] = make_float4(0,0,0,1);\n";

    // Address register
    src += "  int arl = 0;\n\n";

    // Emit translated VP body
    src += "  // === VP microcode ===\n";
    src += emit.code;
    src += "\n";

    // Write outputs
    src += "  // Store output attributes\n";
    src += "  for (int i = 0; i < 17; i++) {\n";
    src += "    output[vid * 17 + i] = out_attr[i];\n";
    src += "  }\n";
    src += "}\n";

    return src;
}

// ═══════════════════════════════════════════════════════════════════
// VP Disassembler (debug)
// ═══════════════════════════════════════════════════════════════════

static const char* VP_VEC_NAMES[] = {
    "NOP","MOV","MUL","ADD","MAD","DP3","DPH","DP4",
    "DST","MIN","MAX","SLT","SGE","ARL","FRC","FLR",
    "SEQ","SFL","SGT","SLE","SNE","STR","SSG","?23","?24","TXL"
};

static const char* VP_SCA_NAMES[] = {
    "NOP","MOV","RCP","RCC","RSQ","EXP","LOG","LIT",
    "BRA","BRI","CAL","CLI","RET","LG2","EX2","SIN",
    "COS","BRB","CLB","PSH","POP"
};

inline void vp_disassemble(const uint32_t* vpData, uint32_t vpLen) {
    uint32_t numInsns = vpLen / 4;
    for (uint32_t i = 0; i < numInsns; i++) {
        VPDecodedInsn insn = vp_decode(&vpData[i * 4]);

        printf("[%3u] ", i);
        if (insn.vecOp != VP_VEC_NOP && insn.vecOp < 26)
            printf("VEC:%-3s ", VP_VEC_NAMES[insn.vecOp]);
        if (insn.scaOp != VP_SCA_NOP && insn.scaOp < 21)
            printf("SCA:%-3s ", VP_SCA_NAMES[insn.scaOp]);

        if (insn.vecOp == VP_VEC_NOP && insn.scaOp == VP_SCA_NOP)
            printf("NOP");

        printf(" → ");
        if (insn.vecWriteResult)
            printf("o%u", insn.vecDstOut);
        else
            printf("t%u", insn.vecDstTmp);
        printf(".%s%s%s%s",
               insn.vecMaskX?"x":"", insn.vecMaskY?"y":"",
               insn.vecMaskZ?"z":"", insn.vecMaskW?"w":"");

        if (insn.endFlag) printf(" [END]");
        printf("\n");

        if (insn.endFlag) break;
    }
}

} // namespace rsx
