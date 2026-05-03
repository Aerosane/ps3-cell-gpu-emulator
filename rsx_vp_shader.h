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
    uint32_t vecDstTmp;     // temp register index (D0.dst_tmp, 0x3F = no temp write)
    uint32_t vecDstOut;     // output register index (D3.dst, 0x1F = no output write)
    bool vecWriteResult;    // D0.vec_result — 1: VEC writes to o[], 0: SCA writes to o[]
    bool vecMaskX, vecMaskY, vecMaskZ, vecMaskW;

    // Scalar destination (same output reg — vecDstOut — shared field D3.dst)
    uint32_t scaDstTmp;     // scalar temp dest (D3.sca_dst_tmp, 0x3F = no temp write)
    bool scaMaskX, scaMaskY, scaMaskZ, scaMaskW;

    // Modifiers
    bool saturate;
    bool endFlag;

    // Constants
    uint32_t constIdx;      // D1.const_src
    uint32_t inputIdx;      // D1.input_src
    uint32_t texUnit;       // texture unit for TXL (D1 bits [19:17])
};

// Decode a single 128-bit VP instruction (4 × u32)
inline VPDecodedInsn vp_decode(const uint32_t d[4]) {
    VPDecodedInsn insn = {};

    // D0 fields (NV47 vertex program instruction word 0; bit layout per
    // rpcs3 RSXVertexProgram.h D0 union)
    uint32_t d0 = d[0];
    //   bits  0-1 addr_swz
    //   bits  2-3 mask_w  (ignored — real masks live in D3)
    //   bits  4-5 mask_z
    //   bits  6-7 mask_y
    //   bits  8-9 mask_x
    //   bits 10-12 cond
    //   bit  13   cond_test_enable
    //   bit  14   cond_update_enable_0
    //   bits 15-20 dst_tmp              (6 bits)
    //   bits 21-23 src0_abs..src2_abs
    //   bits 24-25 addr_reg_sel_1, cond_reg_sel_1
    //   bit  26   saturate
    //   bit  27   index_input
    //   bit  29   cond_update_enable_1
    //   bit  30   vec_result
    insn.vecDstTmp    = (d0 >> 15) & 0x3F;   // bits 15-20
    bool src0Abs      = (d0 >> 21) & 1;
    bool src1Abs      = (d0 >> 22) & 1;
    bool src2Abs      = (d0 >> 23) & 1;
    insn.saturate     = (d0 >> 26) & 1;       // bit 26
    insn.vecWriteResult = (d0 >> 30) & 1;     // bit 30 vec_result

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
    insn.texUnit      = d2 & 0x3;             // tex_num: bits [1:0] for VP TXL
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
    bool usesVtxTex;     // true if VP uses TXL (vertex texture fetch)

    VPEmitter() : tempRegsUsed(0), outputsUsed(0), instrCount(0), usesVtxTex(false) {}

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
        case VP_VEC_TXL: {
            // Vertex texture fetch: sample tex[unit] at s0.xy with LOD = s0.w
            std::string unit = std::to_string(insn.texUnit);
            expr = "vp_tex_fetch(vtex_data, vtex_w, vtex_h, " + unit + ", " + s0 + ")";
            usesVtxTex = true;
            break;
        }
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

    // Add vertex texture fetch helper if needed
    if (emit.usesVtxTex) {
        src += R"(
// Vertex texture fetch — simple bilinear sample from RGBA32 texture
__device__ float4 vp_tex_fetch(const unsigned int* const* tex_data,
                               const unsigned int* tex_w, const unsigned int* tex_h,
                               int unit, float4 coord) {
    if (!tex_data[unit] || tex_w[unit] == 0 || tex_h[unit] == 0)
        return make_float4(0,0,0,0);
    unsigned int w = tex_w[unit], h = tex_h[unit];
    float u = coord.x - floorf(coord.x);  // wrap REPEAT
    float v = coord.y - floorf(coord.y);
    int ix = (int)(u * (float)w) % (int)w;
    int iy = (int)(v * (float)h) % (int)h;
    if (ix < 0) ix += w; if (iy < 0) iy += h;
    unsigned int c = tex_data[unit][iy * w + ix];
    return make_float4(((c>>16)&0xFF)/255.0f, ((c>>8)&0xFF)/255.0f,
                       (c&0xFF)/255.0f, ((c>>24)&0xFF)/255.0f);
}
)";
    }

    // Kernel signature
    src += "\nextern \"C\" __global__\n";
    src += "void rsx_vp_kernel(\n";
    src += "    const float4* __restrict__ vertices,  // input: N vertices × 16 attributes\n";
    src += "    float4* __restrict__ output,           // output: N vertices × 17 output slots\n";
    src += "    const float4* __restrict__ c,          // VP constants (256 vec4s)\n";
    src += "    int numVertices,\n";
    src += "    int numInputAttribs";
    if (emit.usesVtxTex) {
        src += ",\n";
        src += "    const unsigned int* const* vtex_data,  // vertex textures (4 units)\n";
        src += "    const unsigned int* vtex_w,\n";
        src += "    const unsigned int* vtex_h\n";
    } else {
        src += "                    // active input attributes per vertex\n";
    }
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

// ═══════════════════════════════════════════════════════════════════
// Host-side VP Interpreter
//
// Executes a decoded NV40 vertex program on the CPU. One vertex at a
// time. Used by the CUDA raster bridge to transform decoded vertex
// streams when a vertex program is uploaded through the FIFO.
//
// Conventions:
//   inputs[16][4]    — vertex attributes (in_attr). 0=pos, 3=color, 8=uv.
//   constants[256][4]— vpConstants (c[]).
//   temps[48][4]     — scratch registers, zero-initialised.
//   outputs[16][4]   — output registers (o[]). 0=pos (HPOS), 1=diffuse
//                      (COL0), 4..7=texcoord0..3 in common PS3 shaders.
//                      RSX HPOS is expected in clip space {x,y,z,w}.
//
// Implements the common subset of the NV40 VP ISA: vector MOV, MUL,
// ADD, MAD, DP3, DP4, MIN, MAX; scalar MOV, RCP, RSQ. End flag.
// ═══════════════════════════════════════════════════════════════════

struct VPFloat4 { float v[4]; };

inline void vp_read_src(const VPDecodedSrc& s,
                        uint32_t inputIdx, uint32_t constIdx,
                        const VPFloat4 inputs[16],
                        const VPFloat4 constants[256],
                        const VPFloat4 temps[48],
                        float out[4]) {
    const float* base = nullptr;
    float zeros[4] = {0, 0, 0, 0};
    switch (s.regType) {
    case VP_REG_TEMP:
        base = temps[s.regIdx & 0x3F].v;
        break;
    case VP_REG_INPUT:
        base = inputs[inputIdx & 0xF].v;
        break;
    case VP_REG_CONSTANT:
        base = constants[constIdx & 0x1FF].v;
        break;
    default:
        base = zeros;
        break;
    }
    uint32_t sx = s.swzX, sy = s.swzY, sz = s.swzZ, sw = s.swzW;
    out[0] = base[sx]; out[1] = base[sy];
    out[2] = base[sz]; out[3] = base[sw];
    if (s.abs) {
        for (int i = 0; i < 4; ++i) out[i] = out[i] < 0 ? -out[i] : out[i];
    }
    if (s.neg) {
        for (int i = 0; i < 4; ++i) out[i] = -out[i];
    }
}

inline void vp_write_dst(float value[4],
                         bool mx, bool my, bool mz, bool mw,
                         bool saturate,
                         float dst[4]) {
    if (saturate) {
        for (int i = 0; i < 4; ++i) {
            if (value[i] < 0) value[i] = 0;
            if (value[i] > 1) value[i] = 1;
        }
    }
    if (mx) dst[0] = value[0];
    if (my) dst[1] = value[1];
    if (mz) dst[2] = value[2];
    if (mw) dst[3] = value[3];
}

// Texture bank for VP TXL (vertex texture fetch)
struct VPTexInfo {
    const uint32_t* data;  // RGBA8 texels (host memory)
    uint32_t w, h;
};

inline void vp_sample_tex(const VPTexInfo& tex, float u, float v, float out[4]) {
    if (!tex.data || tex.w == 0 || tex.h == 0) {
        out[0] = out[1] = out[2] = out[3] = 1.0f;
        return;
    }
    // Clamp UV to [0,1], sample nearest
    u = u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u);
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    uint32_t px = (uint32_t)(u * (tex.w - 1) + 0.5f);
    uint32_t py = (uint32_t)(v * (tex.h - 1) + 0.5f);
    if (px >= tex.w) px = tex.w - 1;
    if (py >= tex.h) py = tex.h - 1;
    uint32_t rgba = tex.data[py * tex.w + px];
    out[0] = ((rgba >> 16) & 0xFF) / 255.0f;  // R
    out[1] = ((rgba >>  8) & 0xFF) / 255.0f;  // G
    out[2] = ((rgba)       & 0xFF) / 255.0f;  // B
    out[3] = ((rgba >> 24) & 0xFF) / 255.0f;  // A
}

inline void vp_execute(const uint32_t* vpData, uint32_t vpLen, uint32_t vpStart,
                       const VPFloat4 inputs[16],
                       const VPFloat4 constants[256],
                       VPFloat4 outputs[16],
                       const VPTexInfo* vpTex = nullptr,
                       uint32_t vpTexCount = 0) {
    VPFloat4 temps[48] = {};
    int32_t addrReg[4] = {0, 0, 0, 0};
    uint32_t numInsns = vpLen / 4;
    uint32_t callStack[4] = {};
    int callDepth = 0;
    uint32_t maxIter = numInsns * 16;

    for (uint32_t i = vpStart; i < numInsns && maxIter > 0; ++i, --maxIter) {
        VPDecodedInsn insn = vp_decode(&vpData[i * 4]);

        uint32_t effectiveConstIdx = insn.constIdx;
        bool indexInput = (vpData[i*4] >> 27) & 1;
        if (indexInput) {
            uint32_t addrSwz = vpData[i*4] & 3;
            effectiveConstIdx = (uint32_t)((int32_t)insn.constIdx + addrReg[addrSwz]);
            if (effectiveConstIdx > 511) effectiveConstIdx = 0;
        }

        // Read sources
        float src0[4], src1[4], src2[4];
        vp_read_src(insn.src[0], insn.inputIdx, effectiveConstIdx,
                    inputs, constants, temps, src0);
        vp_read_src(insn.src[1], insn.inputIdx, effectiveConstIdx,
                    inputs, constants, temps, src1);
        vp_read_src(insn.src[2], insn.inputIdx, effectiveConstIdx,
                    inputs, constants, temps, src2);

        // Vector op
        if (insn.vecOp != VP_VEC_NOP) {
            float r[4] = {0, 0, 0, 0};
            switch (insn.vecOp) {
            case VP_VEC_MOV:
                for (int k = 0; k < 4; ++k) r[k] = src0[k];
                break;
            case VP_VEC_MUL:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] * src1[k];
                break;
            case VP_VEC_ADD:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] + src2[k];
                break;
            case VP_VEC_MAD:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] * src1[k] + src2[k];
                break;
            case VP_VEC_DP3: {
                float d = src0[0]*src1[0] + src0[1]*src1[1] + src0[2]*src1[2];
                r[0] = r[1] = r[2] = r[3] = d;
                break;
            }
            case VP_VEC_DP4:
            case VP_VEC_DPH: {
                float d;
                if (insn.vecOp == VP_VEC_DPH)
                    d = src0[0]*src1[0] + src0[1]*src1[1] + src0[2]*src1[2] + src1[3];
                else
                    d = src0[0]*src1[0] + src0[1]*src1[1] + src0[2]*src1[2] + src0[3]*src1[3];
                r[0] = r[1] = r[2] = r[3] = d;
                break;
            }
            case VP_VEC_MIN:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] < src1[k] ? src0[k] : src1[k];
                break;
            case VP_VEC_MAX:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] > src1[k] ? src0[k] : src1[k];
                break;
            case VP_VEC_FRC:
                for (int k = 0; k < 4; ++k) {
                    float f = src0[k];
                    r[k] = f - (int)f;
                    if (r[k] < 0) r[k] += 1.0f;
                }
                break;
            case VP_VEC_FLR:
                for (int k = 0; k < 4; ++k) {
                    float f = src0[k];
                    r[k] = (float)((int)f - (f < 0 && f != (int)f ? 1 : 0));
                }
                break;
            case VP_VEC_SLT:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] <  src1[k] ? 1.0f : 0.0f;
                break;
            case VP_VEC_SGE:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] >= src1[k] ? 1.0f : 0.0f;
                break;
            case VP_VEC_DST: {
                // DST: distance vector — r = {1, s0.y*s1.y, s0.z, s1.w}
                r[0] = 1.0f;
                r[1] = src0[1] * src1[1];
                r[2] = src0[2];
                r[3] = src1[3];
                break;
            }
            case VP_VEC_SEQ:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] == src1[k] ? 1.0f : 0.0f;
                break;
            case VP_VEC_SGT:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] >  src1[k] ? 1.0f : 0.0f;
                break;
            case VP_VEC_SLE:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] <= src1[k] ? 1.0f : 0.0f;
                break;
            case VP_VEC_SNE:
                for (int k = 0; k < 4; ++k) r[k] = src0[k] != src1[k] ? 1.0f : 0.0f;
                break;
            case VP_VEC_SFL:
                for (int k = 0; k < 4; ++k) r[k] = 0.0f;  // always false
                break;
            case VP_VEC_STR:
                for (int k = 0; k < 4; ++k) r[k] = 1.0f;  // always true
                break;
            case VP_VEC_SSG:
                for (int k = 0; k < 4; ++k) {
                    r[k] = src0[k] > 0.0f ? 1.0f : (src0[k] < 0.0f ? -1.0f : 0.0f);
                }
                break;
            case VP_VEC_TXL: {
                // Vertex texture fetch: sample texture unit from insn at src0.xy
                uint32_t texUnit = insn.inputIdx & 0x3;  // texture unit 0-3
                if (vpTex && texUnit < vpTexCount && vpTex[texUnit].data) {
                    vp_sample_tex(vpTex[texUnit], src0[0], src0[1], r);
                } else {
                    for (int k = 0; k < 4; ++k) r[k] = 1.0f;
                }
                break;
            }
            case VP_VEC_ARL:
                for (int k = 0; k < 4; ++k) {
                    addrReg[k] = (int32_t)__builtin_floorf(src0[k]);
                }
                break;
            default:
                for (int k = 0; k < 4; ++k) r[k] = src0[k];
                break;
            }

            // Write destinations per NV47 semantics:
            //   VEC op writes to OUTPUT reg (o[d3.dst]) iff d0.vec_result == 1 AND d3.dst != 0x1F
            //   VEC op writes to TEMP reg (r[d0.dst_tmp]) iff d0.dst_tmp != 0x3F
            //   Both writes can happen in the same instruction.
            if (insn.vecWriteResult && insn.vecDstOut != 0x1F) {
                float* dstOut = outputs[insn.vecDstOut & 0xF].v;
                vp_write_dst(r,
                             insn.vecMaskX, insn.vecMaskY, insn.vecMaskZ, insn.vecMaskW,
                             insn.saturate, dstOut);
            }
            if (insn.vecDstTmp != 0x3F) {
                float* dstTmp = temps[insn.vecDstTmp & 0x3F].v;
                vp_write_dst(r,
                             insn.vecMaskX, insn.vecMaskY, insn.vecMaskZ, insn.vecMaskW,
                             insn.saturate, dstTmp);
            }
        }

        // Scalar op (uses src2, result broadcast to vector dst channels)
        if (insn.scaOp != VP_SCA_NOP) {
            float s = src2[0];
            float r[4] = {0, 0, 0, 0};
            switch (insn.scaOp) {
            case VP_SCA_MOV: r[0] = r[1] = r[2] = r[3] = s; break;
            case VP_SCA_RCP: r[0] = r[1] = r[2] = r[3] = (s != 0 ? 1.0f / s : 0); break;
            case VP_SCA_RCC: {
                // RCP clamped: clamp result to [5.42101e-36, 1.884467e+19]
                float v = (s != 0) ? 1.0f / s : 0.0f;
                if (v > 0.0f) v = v < 5.42101e-36f ? 5.42101e-36f : (v > 1.884467e+19f ? 1.884467e+19f : v);
                else v = v > -5.42101e-36f ? -5.42101e-36f : (v < -1.884467e+19f ? -1.884467e+19f : v);
                r[0] = r[1] = r[2] = r[3] = v;
                break;
            }
            case VP_SCA_RSQ: {
                float a = s < 0 ? -s : s;
                float inv = (a != 0 ? 1.0f / __builtin_sqrtf(a) : 0);
                r[0] = r[1] = r[2] = r[3] = inv;
                break;
            }
            case VP_SCA_EXP: {
                // Partial floor-based EXP: {2^floor(s), frac(s), 2^s, 1}
                float fl = __builtin_floorf(s);
                float fr = s - fl;
                float pw = __builtin_exp2f(s);
                r[0] = __builtin_exp2f(fl); r[1] = fr; r[2] = pw; r[3] = 1.0f;
                break;
            }
            case VP_SCA_LOG: {
                // Partial floor-based LOG: {floor(log2|s|), |s|/2^floor, log2|s|, 1}
                float a = s < 0 ? -s : s;
                if (a == 0.0f) { r[0] = r[1] = r[2] = -__builtin_inff(); r[3] = 1.0f; }
                else {
                    float lg = __builtin_log2f(a);
                    float fl = __builtin_floorf(lg);
                    r[0] = fl; r[1] = a / __builtin_exp2f(fl); r[2] = lg; r[3] = 1.0f;
                }
                break;
            }
            case VP_SCA_LIT: {
                // LIT: lighting helper — s = src2, but we need all 4 components
                // LIT uses src2.xyzw: r = {1, max(s.x,0), s.x>0 ? pow(max(s.y,0), clamp(s.w,-128,128)) : 0, 1}
                float sx = src2[0], sy = src2[1], sw = src2[3];
                r[0] = 1.0f;
                r[1] = sx > 0.0f ? sx : 0.0f;
                if (sx > 0.0f) {
                    float base = sy > 0.0f ? sy : 0.0f;
                    float exp = sw < -128.0f ? -128.0f : (sw > 128.0f ? 128.0f : sw);
                    r[2] = __builtin_powf(base, exp);
                } else {
                    r[2] = 0.0f;
                }
                r[3] = 1.0f;
                break;
            }
            case VP_SCA_EX2:
                r[0] = r[1] = r[2] = r[3] = __builtin_exp2f(s);
                break;
            case VP_SCA_LG2: {
                float a = s < 0 ? -s : s;
                r[0] = r[1] = r[2] = r[3] = (a != 0 ? __builtin_log2f(a) : -__builtin_inff());
                break;
            }
            case VP_SCA_SIN:
                r[0] = r[1] = r[2] = r[3] = __builtin_sinf(s);
                break;
            case VP_SCA_COS:
                r[0] = r[1] = r[2] = r[3] = __builtin_cosf(s);
                break;
            case VP_SCA_BRA: case VP_SCA_BRI: case VP_SCA_BRB: {
                uint32_t target = insn.constIdx;
                if (target < numInsns) i = target - 1;
                break;
            }
            case VP_SCA_CAL: case VP_SCA_CLI: case VP_SCA_CLB: {
                uint32_t target = insn.constIdx;
                if (callDepth < 4 && target < numInsns) {
                    callStack[callDepth++] = i;
                    i = target - 1;
                }
                break;
            }
            case VP_SCA_RET:
                if (callDepth > 0) i = callStack[--callDepth];
                break;
            case VP_SCA_PSH: case VP_SCA_POP:
                break;  // CC stack stubs
            default:
                r[0] = r[1] = r[2] = r[3] = s;
                break;
            }
            // Scalar writes: OUTPUT iff !d0.vec_result AND d3.dst != 0x1F,
            // TEMP iff d3.sca_dst_tmp != 0x3F.
            if (!insn.vecWriteResult && insn.vecDstOut != 0x1F) {
                float* dstOut = outputs[insn.vecDstOut & 0xF].v;
                vp_write_dst(r,
                             insn.scaMaskX, insn.scaMaskY, insn.scaMaskZ, insn.scaMaskW,
                             insn.saturate, dstOut);
            }
            if (insn.scaDstTmp != 0x3F) {
                float* dstTmp = temps[insn.scaDstTmp & 0x3F].v;
                vp_write_dst(r,
                             insn.scaMaskX, insn.scaMaskY, insn.scaMaskZ, insn.scaMaskW,
                             insn.saturate, dstTmp);
            }
        }

        if (insn.endFlag) break;
    }
}

} // namespace rsx
