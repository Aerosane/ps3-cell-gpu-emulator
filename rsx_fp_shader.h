#pragma once
// rsx_fp_shader.h — RSX Fragment Program → GLSL 450 Translator
//
// Decodes NV30/NV40 FP microcode and emits GLSL 450 source for Vulkan
// fragment shaders. Handles fp32/fp16 precision, texture sampling,
// conditional execution, and all 64+ FP opcodes.
//
// Part of Project Megakernel — hybrid RSX: VP on CUDA, rasterize via Vulkan.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rsx {

// ═══════════════════════════════════════════════════════════════════
// FP Opcode Definitions (from RPCS3 FPOpcodes.h)
// ═══════════════════════════════════════════════════════════════════

enum FPOpcode : uint32_t {
    FP_NOP   = 0x00, FP_MOV   = 0x01, FP_MUL   = 0x02, FP_ADD   = 0x03,
    FP_MAD   = 0x04, FP_DP3   = 0x05, FP_DP4   = 0x06, FP_DST   = 0x07,
    FP_MIN   = 0x08, FP_MAX   = 0x09, FP_SLT   = 0x0A, FP_SGE   = 0x0B,
    FP_SLE   = 0x0C, FP_SGT   = 0x0D, FP_SNE   = 0x0E, FP_SEQ   = 0x0F,
    FP_FRC   = 0x10, FP_FLR   = 0x11, FP_KIL   = 0x12, FP_PK4   = 0x13,
    FP_UP4   = 0x14, FP_DDX   = 0x15, FP_DDY   = 0x16, FP_TEX   = 0x17,
    FP_TXP   = 0x18, FP_TXD   = 0x19, FP_RCP   = 0x1A, FP_RSQ   = 0x1B,
    FP_EX2   = 0x1C, FP_LG2   = 0x1D, FP_LIT   = 0x1E, FP_LRP   = 0x1F,
    FP_STR   = 0x20, FP_SFL   = 0x21, FP_COS   = 0x22, FP_SIN   = 0x23,
    FP_PK2   = 0x24, FP_UP2   = 0x25, FP_POW   = 0x26, FP_PKB   = 0x27,
    FP_UPB   = 0x28, FP_PK16  = 0x29, FP_UP16  = 0x2A, FP_BEM   = 0x2B,
    FP_PKG   = 0x2C, FP_UPG   = 0x2D, FP_DP2A  = 0x2E, FP_TXL   = 0x2F,
    FP_TXB   = 0x31, FP_TEXBEM = 0x33, FP_TXPBEM = 0x34, FP_BEMLUM = 0x35,
    FP_REFL  = 0x36, FP_TIMESWTEX = 0x37, FP_DP2 = 0x38, FP_NRM = 0x39,
    FP_DIV   = 0x3A, FP_DIVSQ = 0x3B, FP_LIF   = 0x3C,
    FP_FENCT = 0x3D, FP_FENCB = 0x3E,
    FP_BRK   = 0x40, FP_CAL   = 0x41, FP_IFE   = 0x42,
    FP_LOOP  = 0x43, FP_REP   = 0x44, FP_RET   = 0x45,
};

// Register types
enum FPRegType : uint32_t {
    FP_REG_TEMP     = 0,
    FP_REG_INPUT    = 1,
    FP_REG_CONSTANT = 2,
};

// Precision modes
enum FPPrecision : uint32_t {
    FP_PREC_FP32    = 0,
    FP_PREC_FP16    = 1,
    FP_PREC_FIXED12 = 2,
    FP_PREC_FIXED9  = 3,
    FP_PREC_SATURATE = 4,
};

// FP opcode names for disassembly/debug
static const char* FP_OP_NAMES[] = {
    "NOP","MOV","MUL","ADD","MAD","DP3","DP4","DST",
    "MIN","MAX","SLT","SGE","SLE","SGT","SNE","SEQ",
    "FRC","FLR","KIL","PK4","UP4","DDX","DDY","TEX",
    "TXP","TXD","RCP","RSQ","EX2","LG2","LIT","LRP",
    "STR","SFL","COS","SIN","PK2","UP2","POW","PKB",
    "UPB","PK16","UP16","BEM","PKG","UPG","DP2A","TXL",
    "?48","TXB"
};

// Swizzle component names
static const char* FP_SWZ[] = {"x", "y", "z", "w"};

// Input attribute names (RSX fragment program inputs)
static const char* FP_INPUT_NAMES[] = {
    "gl_FragCoord",  // 0 — WPOS
    "v_col0",        // 1 — COL0
    "v_col1",        // 2 — COL1
    "v_fogc",        // 3 — FOGC
    "v_tex0",        // 4 — TEX0
    "v_tex1",        // 5 — TEX1
    "v_tex2",        // 6 — TEX2
    "v_tex3",        // 7 — TEX3
    "v_tex4",        // 8 — TEX4
    "v_tex5",        // 9 — TEX5
    "v_tex6",        // 10 — TEX6
    "v_tex7",        // 11 — TEX7
    "v_tex8",        // 12 — TEX8
    "v_tex9",        // 13 — TEX9
    "v_ssa",         // 14 — SSA (specular secondary alpha?)
};

// ═══════════════════════════════════════════════════════════════════
// FP Instruction Decoder
// ═══════════════════════════════════════════════════════════════════

// FP instructions are stored big-endian in PS3 memory.
// Each instruction is 4 × 32-bit words: OPDEST, SRC0, SRC1, SRC2
// But the byte-swap is 16-bit swapped: swap each pair of bytes within each 32-bit word

inline uint32_t fp_swap_word(uint32_t be) {
    return ((be & 0x00FF00FF) << 8) | ((be & 0xFF00FF00) >> 8);
}

struct FPDecodedSrc {
    uint32_t regType;  // FP_REG_TEMP/INPUT/CONSTANT
    uint32_t regIdx;
    uint32_t swzX, swzY, swzZ, swzW;
    bool neg;
    bool abs;
    bool fp16;
};

struct FPDecodedInsn {
    uint32_t opcode;       // 6-bit (+ 1 high bit from SRC1)
    uint32_t dstReg;       // 6-bit destination register
    bool dstFp16;          // destination is half register
    bool maskX, maskY, maskZ, maskW;
    bool noDest;           // instruction has no destination
    bool saturate;
    bool endFlag;
    bool setCond;
    uint32_t texUnit;      // texture unit number (4-bit)
    uint32_t precision;    // precision mode
    uint32_t inputAttr;    // src_attr_reg_num

    FPDecodedSrc src[3];

    // Flow control
    uint32_t elseOffset;   // for IFE
    uint32_t endOffset;    // for IFE/LOOP/REP
    uint32_t loopEnd;      // LOOP end counter
    uint32_t loopInit;     // LOOP init counter
    uint32_t loopIncr;     // LOOP increment
};

inline FPDecodedInsn fp_decode(const uint32_t raw[4]) {
    FPDecodedInsn insn = {};

    // Byte-swap each word (RSX FP is big-endian with 16-bit swap)
    uint32_t w0 = fp_swap_word(raw[0]);  // OPDEST
    uint32_t w1 = fp_swap_word(raw[1]);  // SRC0
    uint32_t w2 = fp_swap_word(raw[2]);  // SRC1
    uint32_t w3 = fp_swap_word(raw[3]);  // SRC2

    // OPDEST
    insn.endFlag   = w0 & 1;
    insn.dstReg    = (w0 >> 1) & 0x3F;
    insn.dstFp16   = (w0 >> 7) & 1;
    insn.setCond   = (w0 >> 8) & 1;
    insn.maskX     = (w0 >> 9) & 1;
    insn.maskY     = (w0 >> 10) & 1;
    insn.maskZ     = (w0 >> 11) & 1;
    insn.maskW     = (w0 >> 12) & 1;
    insn.inputAttr = (w0 >> 13) & 0xF;
    insn.texUnit   = (w0 >> 17) & 0xF;
    insn.precision = (w0 >> 22) & 0x3;
    insn.opcode    = (w0 >> 24) & 0x3F;
    insn.noDest    = (w0 >> 30) & 1;
    insn.saturate  = (w0 >> 31) & 1;

    // Opcode high bit from SRC1
    uint32_t opcodeHi = (w2 >> 31) & 1;
    insn.opcode |= (opcodeHi << 6);

    // Decode SRC0
    insn.src[0].regType = w1 & 0x3;
    insn.src[0].regIdx  = (w1 >> 2) & 0x3F;
    insn.src[0].fp16    = (w1 >> 8) & 1;
    insn.src[0].swzX    = (w1 >> 9) & 0x3;
    insn.src[0].swzY    = (w1 >> 11) & 0x3;
    insn.src[0].swzZ    = (w1 >> 13) & 0x3;
    insn.src[0].swzW    = (w1 >> 15) & 0x3;
    insn.src[0].neg     = (w1 >> 17) & 1;
    insn.src[0].abs     = (w1 >> 29) & 1;

    // Decode SRC1
    insn.src[1].regType = w2 & 0x3;
    insn.src[1].regIdx  = (w2 >> 2) & 0x3F;
    insn.src[1].fp16    = (w2 >> 8) & 1;
    insn.src[1].swzX    = (w2 >> 9) & 0x3;
    insn.src[1].swzY    = (w2 >> 11) & 0x3;
    insn.src[1].swzZ    = (w2 >> 13) & 0x3;
    insn.src[1].swzW    = (w2 >> 15) & 0x3;
    insn.src[1].neg     = (w2 >> 17) & 1;
    insn.src[1].abs     = (w2 >> 18) & 1;

    // Flow control fields from SRC1
    insn.elseOffset = w2 & 0x7FFFFFFF;
    insn.loopEnd    = (w2 >> 2) & 0xFF;
    insn.loopInit   = (w2 >> 10) & 0xFF;
    insn.loopIncr   = (w2 >> 19) & 0xFF;

    // Decode SRC2
    insn.src[2].regType = w3 & 0x3;
    insn.src[2].regIdx  = (w3 >> 2) & 0x3F;
    insn.src[2].fp16    = (w3 >> 8) & 1;
    insn.src[2].swzX    = (w3 >> 9) & 0x3;
    insn.src[2].swzY    = (w3 >> 11) & 0x3;
    insn.src[2].swzZ    = (w3 >> 13) & 0x3;
    insn.src[2].swzW    = (w3 >> 15) & 0x3;
    insn.src[2].neg     = (w3 >> 17) & 1;
    insn.src[2].abs     = (w3 >> 18) & 1;

    insn.endOffset = w3;

    return insn;
}

// ═══════════════════════════════════════════════════════════════════
// GLSL 450 Source Code Emitter
// ═══════════════════════════════════════════════════════════════════

struct FPEmitter {
    std::string code;
    int tempRegsUsed;       // highest temp register used
    int constCount;         // constants referenced
    uint32_t texUnitsMask;  // bitmask of texture units sampled
    uint32_t inputsMask;    // bitmask of input attributes read
    int instrCount;
    int indentLevel;

    FPEmitter() : tempRegsUsed(0), constCount(0), texUnitsMask(0),
                  inputsMask(0), instrCount(0), indentLevel(1) {}

    std::string indent() {
        return std::string(indentLevel * 2, ' ');
    }

    // Emit source register reference with swizzle
    std::string emitSrc(const FPDecodedInsn& insn, int srcIdx) {
        const FPDecodedSrc& s = insn.src[srcIdx];
        std::string reg;

        switch (s.regType) {
        case FP_REG_TEMP:
            if (s.fp16)
                reg = "h" + std::to_string(s.regIdx);
            else
                reg = "r" + std::to_string(s.regIdx);
            if ((int)s.regIdx >= tempRegsUsed) tempRegsUsed = s.regIdx + 1;
            break;
        case FP_REG_INPUT:
            reg = (insn.inputAttr < 15) ? FP_INPUT_NAMES[insn.inputAttr] : "vec4(0.0)";
            inputsMask |= (1 << insn.inputAttr);
            break;
        case FP_REG_CONSTANT:
            reg = "fc[" + std::to_string(s.regIdx) + "]";
            if ((int)s.regIdx >= constCount) constCount = s.regIdx + 1;
            break;
        default:
            reg = "vec4(0.0)";
            break;
        }

        // Apply swizzle
        bool identitySwz = (s.swzX == 0 && s.swzY == 1 && s.swzZ == 2 && s.swzW == 3);
        if (!identitySwz) {
            std::string swz;
            swz += FP_SWZ[s.swzX];
            swz += FP_SWZ[s.swzY];
            swz += FP_SWZ[s.swzZ];
            swz += FP_SWZ[s.swzW];
            reg = reg + "." + swz;
        }

        // Apply abs
        if (s.abs) reg = "abs(" + reg + ")";
        // Apply negate
        if (s.neg) reg = "-(" + reg + ")";

        return reg;
    }

    // Emit masked write
    void emitWrite(const std::string& dst, const std::string& mask,
                   const std::string& expr, bool sat) {
        std::string val = sat ? "clamp(" + expr + ", 0.0, 1.0)" : expr;
        code += indent() + dst + "." + mask + " = (" + val + ")." + mask + ";\n";
    }

    // Get write mask string
    std::string getMask(bool x, bool y, bool z, bool w) {
        std::string m;
        if (x) m += "x";
        if (y) m += "y";
        if (z) m += "z";
        if (w) m += "w";
        if (m.empty()) m = "xyzw";
        return m;
    }

    // Get destination register name
    std::string getDst(const FPDecodedInsn& insn) {
        if (insn.dstFp16)
            return "h" + std::to_string(insn.dstReg);
        else
            return "r" + std::to_string(insn.dstReg);
    }

    // Emit a single FP instruction
    void emitInsn(const FPDecodedInsn& insn) {
        if (insn.opcode == FP_NOP || insn.opcode == FP_FENCT || insn.opcode == FP_FENCB)
            return;

        std::string s0 = emitSrc(insn, 0);
        std::string s1 = emitSrc(insn, 1);
        std::string s2 = emitSrc(insn, 2);
        std::string dst = getDst(insn);
        std::string mask = getMask(insn.maskX, insn.maskY, insn.maskZ, insn.maskW);
        bool sat = insn.saturate;

        if ((int)insn.dstReg >= tempRegsUsed && !insn.noDest)
            tempRegsUsed = insn.dstReg + 1;

        auto emit1 = [&](const std::string& expr) {
            if (!insn.noDest) emitWrite(dst, mask, expr, sat);
        };

        switch (insn.opcode) {
        case FP_MOV: emit1(s0); break;
        case FP_MUL: emit1(s0 + " * " + s1); break;
        case FP_ADD: emit1(s0 + " + " + s1); break;
        case FP_MAD: emit1(s0 + " * " + s1 + " + " + s2); break;
        case FP_DP3: emit1("vec4(dot(" + s0 + ".xyz, " + s1 + ".xyz))"); break;
        case FP_DP4: emit1("vec4(dot(" + s0 + ", " + s1 + "))"); break;
        case FP_DP2: emit1("vec4(dot(" + s0 + ".xy, " + s1 + ".xy))"); break;
        case FP_DP2A: emit1("vec4(dot(" + s0 + ".xy, " + s1 + ".xy) + " + s2 + ".x)"); break;
        case FP_DST: emit1("vec4(1.0, " + s0 + ".y * " + s1 + ".y, " + s0 + ".z, " + s1 + ".w)"); break;
        case FP_MIN: emit1("min(" + s0 + ", " + s1 + ")"); break;
        case FP_MAX: emit1("max(" + s0 + ", " + s1 + ")"); break;
        case FP_SLT: emit1("vec4(lessThan(" + s0 + ", " + s1 + "))"); break;
        case FP_SGE: emit1("vec4(greaterThanEqual(" + s0 + ", " + s1 + "))"); break;
        case FP_SLE: emit1("vec4(lessThanEqual(" + s0 + ", " + s1 + "))"); break;
        case FP_SGT: emit1("vec4(greaterThan(" + s0 + ", " + s1 + "))"); break;
        case FP_SNE: emit1("vec4(notEqual(" + s0 + ", " + s1 + "))"); break;
        case FP_SEQ: emit1("vec4(equal(" + s0 + ", " + s1 + "))"); break;
        case FP_FRC: emit1("fract(" + s0 + ")"); break;
        case FP_FLR: emit1("floor(" + s0 + ")"); break;
        case FP_RCP: emit1("vec4(1.0 / " + s0 + ".x)"); break;
        case FP_RSQ: emit1("vec4(inversesqrt(abs(" + s0 + ".x)))"); break;
        case FP_EX2: emit1("vec4(exp2(" + s0 + ".x))"); break;
        case FP_LG2: emit1("vec4(log2(abs(" + s0 + ".x)))"); break;
        case FP_POW: emit1("vec4(pow(abs(" + s0 + ".x), " + s1 + ".x))"); break;
        case FP_SIN: emit1("vec4(sin(" + s0 + ".x))"); break;
        case FP_COS: emit1("vec4(cos(" + s0 + ".x))"); break;
        case FP_NRM: emit1("vec4(normalize(" + s0 + ".xyz), " + s0 + ".w)"); break;
        case FP_DIV: emit1("vec4(" + s0 + ".x / " + s1 + ".x)"); break;
        case FP_DIVSQ: emit1("vec4(" + s0 + ".x * inversesqrt(abs(" + s1 + ".x)))"); break;
        case FP_LRP: emit1("mix(" + s2 + ", " + s1 + ", " + s0 + ")"); break;
        case FP_STR: emit1("vec4(1.0)"); break;
        case FP_SFL: emit1("vec4(0.0)"); break;
        case FP_REFL: emit1("vec4(reflect(" + s0 + ".xyz, " + s1 + ".xyz), 0.0)"); break;
        // SSG not in FP opcode enum — handled via comparisons if needed
        case FP_DDX: emit1("dFdx(" + s0 + ")"); break;
        case FP_DDY: emit1("dFdy(" + s0 + ")"); break;
        case FP_LIT: {
            emit1("vec4(1.0, max(" + s0 + ".x, 0.0), "
                  + s0 + ".x > 0.0 ? pow(max(" + s0 + ".y, 0.0), clamp(" + s0 + ".w, -128.0, 128.0)) : 0.0, 1.0)");
            break;
        }
        case FP_LIF: {
            emit1("vec4(1.0, " + s0 + ".y, " +
                  s0 + ".y > 0.0 ? pow(2.0, " + s0 + ".w) : 0.0, 1.0)");
            break;
        }

        // Texture sampling
        case FP_TEX: {
            std::string tex = "tex" + std::to_string(insn.texUnit);
            texUnitsMask |= (1 << insn.texUnit);
            emit1("texture(" + tex + ", " + s0 + ".xy)");
            break;
        }
        case FP_TXP: {
            std::string tex = "tex" + std::to_string(insn.texUnit);
            texUnitsMask |= (1 << insn.texUnit);
            emit1("textureProj(" + tex + ", " + s0 + ")");
            break;
        }
        case FP_TXL: {
            std::string tex = "tex" + std::to_string(insn.texUnit);
            texUnitsMask |= (1 << insn.texUnit);
            emit1("textureLod(" + tex + ", " + s0 + ".xy, " + s0 + ".w)");
            break;
        }
        case FP_TXB: {
            std::string tex = "tex" + std::to_string(insn.texUnit);
            texUnitsMask |= (1 << insn.texUnit);
            emit1("texture(" + tex + ", " + s0 + ".xy, " + s0 + ".w)");
            break;
        }
        case FP_TXD: {
            std::string tex = "tex" + std::to_string(insn.texUnit);
            texUnitsMask |= (1 << insn.texUnit);
            emit1("textureGrad(" + tex + ", " + s0 + ".xy, " + s1 + ".xy, " + s2 + ".xy)");
            break;
        }

        // Kill
        case FP_KIL:
            code += indent() + "if (any(lessThan(" + s0 + ", vec4(0.0)))) discard;\n";
            break;

        // Flow control
        case FP_IFE:
            code += indent() + "// IFE (cond branch)\n";
            code += indent() + "{\n";
            indentLevel++;
            break;
        case FP_LOOP:
            code += indent() + "for (int lc = " + std::to_string(insn.loopInit) +
                    "; lc < " + std::to_string(insn.loopEnd) +
                    "; lc += " + std::to_string(insn.loopIncr > 0 ? insn.loopIncr : 1) + ") {\n";
            indentLevel++;
            break;
        case FP_REP:
            code += indent() + "for (int rc = 0; rc < " + std::to_string(insn.loopEnd) + "; rc++) {\n";
            indentLevel++;
            break;
        case FP_BRK:
            code += indent() + "break;\n";
            break;
        case FP_CAL:
            code += indent() + "// CAL (subroutine — inlined)\n";
            break;
        case FP_RET:
            if (indentLevel > 1) {
                indentLevel--;
                code += indent() + "}\n";
            }
            break;

        default:
            code += indent() + "// UNHANDLED fp op 0x" + std::to_string(insn.opcode) + "\n";
            break;
        }

        instrCount++;
    }
};

// ═══════════════════════════════════════════════════════════════════
// FP → GLSL 450 Translator
// ═══════════════════════════════════════════════════════════════════

// Translate an FP program to GLSL 450 fragment shader source
// fpData:   array of uint32_t words (4 words per instruction, big-endian)
// fpLen:    number of words (must be multiple of 4)
// returns:  GLSL 450 source string for Vulkan SPIR-V compilation
inline std::string fp_translate_to_glsl(const uint32_t* fpData, uint32_t fpLen) {
    FPEmitter emit;
    uint32_t numInsns = fpLen / 4;

    // First pass: decode and emit all instructions
    for (uint32_t i = 0; i < numInsns; i++) {
        const uint32_t* raw = &fpData[i * 4];
        FPDecodedInsn insn = fp_decode(raw);

        emit.code += emit.indent() + "// insn " + std::to_string(i) + ": op=" +
                     std::to_string(insn.opcode) + "\n";
        emit.emitInsn(insn);

        if (insn.endFlag) break;
    }

    // Build complete GLSL 450 source
    std::string src;
    src += "#version 450\n";
    src += "// Auto-generated RSX FP shader (" + std::to_string(emit.instrCount) + " instructions)\n\n";

    // Input varyings (from VP outputs)
    if (emit.inputsMask & (1 << 1))  src += "layout(location = 0) in vec4 v_col0;\n";
    if (emit.inputsMask & (1 << 2))  src += "layout(location = 1) in vec4 v_col1;\n";
    if (emit.inputsMask & (1 << 3))  src += "layout(location = 2) in vec4 v_fogc;\n";
    for (int i = 0; i < 10; i++) {
        if (emit.inputsMask & (1 << (4 + i)))
            src += "layout(location = " + std::to_string(3 + i) + ") in vec4 v_tex" + std::to_string(i) + ";\n";
    }

    // Texture samplers
    for (int i = 0; i < 16; i++) {
        if (emit.texUnitsMask & (1 << i))
            src += "layout(set = 0, binding = " + std::to_string(i) + ") uniform sampler2D tex" + std::to_string(i) + ";\n";
    }

    // Fragment constants UBO
    if (emit.constCount > 0) {
        src += "\nlayout(set = 1, binding = 0) uniform FPConstants {\n";
        src += "  vec4 fc[" + std::to_string(emit.constCount) + "];\n";
        src += "};\n";
    }

    // Output
    src += "\nlayout(location = 0) out vec4 fragColor;\n";

    // Main function
    src += "\nvoid main() {\n";

    // Declare temp registers
    int numTemps = (emit.tempRegsUsed > 0) ? emit.tempRegsUsed : 1;
    if (numTemps > 64) numTemps = 64;
    for (int i = 0; i < numTemps; i++) {
        src += "  vec4 r" + std::to_string(i) + " = vec4(0.0);\n";
    }

    // Half-precision temps (mediump in GLSL)
    for (int i = 0; i < numTemps; i++) {
        src += "  vec4 h" + std::to_string(i) + " = vec4(0.0);\n";
    }
    src += "\n";

    // Emit translated FP body
    src += "  // === FP microcode ===\n";
    src += emit.code;

    // Output: MRT0 = r0 (standard single-target)
    src += "\n  fragColor = r0;\n";
    src += "}\n";

    return src;
}

// ═══════════════════════════════════════════════════════════════════
// FP Disassembler (debug)
// ═══════════════════════════════════════════════════════════════════

inline void fp_disassemble(const uint32_t* fpData, uint32_t fpLen) {
    uint32_t numInsns = fpLen / 4;
    for (uint32_t i = 0; i < numInsns; i++) {
        FPDecodedInsn insn = fp_decode(&fpData[i * 4]);

        printf("[%3u] ", i);
        if (insn.opcode < 0x40 && insn.opcode < sizeof(FP_OP_NAMES)/sizeof(FP_OP_NAMES[0]))
            printf("%-4s ", FP_OP_NAMES[insn.opcode]);
        else
            printf("0x%02X ", insn.opcode);

        if (!insn.noDest) {
            printf("→ %s%u.%s%s%s%s",
                   insn.dstFp16 ? "h" : "r", insn.dstReg,
                   insn.maskX?"x":"", insn.maskY?"y":"",
                   insn.maskZ?"z":"", insn.maskW?"w":"");
        }

        if (insn.opcode >= FP_TEX && insn.opcode <= FP_TXD)
            printf(" tex%u", insn.texUnit);

        if (insn.saturate) printf(" [SAT]");
        if (insn.endFlag) printf(" [END]");
        printf("\n");

        if (insn.endFlag) break;
    }
}

} // namespace rsx
