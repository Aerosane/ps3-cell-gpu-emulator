// rsx_raster.cu — CUDA RSX rasterizer implementation.

#include "rsx_raster.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace rsx {

#define CU_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "cuda: %s = %s\n", #x, cudaGetErrorString(_e)); \
    return -1; } } while(0)

// ═══════════════════════════════════════════════════════════════
// Kernels
// ═══════════════════════════════════════════════════════════════

__global__ void k_clear(uint32_t* __restrict__ dst,
                        uint32_t width, uint32_t height,
                        uint32_t value) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    dst[y * width + x] = value;
}

__global__ void k_clearDepth(float* __restrict__ dst,
                             uint32_t width, uint32_t height,
                             float value) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    dst[y * width + x] = value;
}

// ═══════════════════════════════════════════════════════════════
// Per-pixel Fragment Program Interpreter
// ═══════════════════════════════════════════════════════════════
//
// Packed instruction format (6 × uint32_t per instruction):
//   [0] opcode(7) | maskX(1) | maskY(1) | maskZ(1) | maskW(1) |
//       texUnit(4) | inputAttr(4) | saturate(1) | endFlag(1) | pad(11)
//   [1] dstReg(8) | src0_type(2) | src0_idx(8) | src0_swz(8) | src0_neg(1) | pad(5)
//   [2] src1_type(2) | src1_idx(8) | src1_swz(8) | src1_neg(1) | pad(13)
//   [3] src2_type(2) | src2_idx(8) | src2_swz(8) | src2_neg(1) | pad(13)
//   [4-5] reserved

// Forward-declare texture helpers (defined later in this file)
__device__ int texWrap(int coord, int dim, uint8_t mode);
__device__ uint32_t texFetch(const uint32_t* tex,
    uint32_t tw, uint32_t th, int tx, int ty,
    uint8_t wrapS, uint8_t wrapT);
__device__ void sampleTex(const uint32_t* tex,
    uint32_t tw, uint32_t th,
    float u, float v, int bilinear,
    float& outR, float& outG, float& outB, float& outA,
    uint8_t wrapS, uint8_t wrapT);

#define FP_PACK_W0(op, mx, my, mz, mw, tu, ia, sat, end) \
    (((op)&0x7F) | (((mx)&1)<<7) | (((my)&1)<<8) | (((mz)&1)<<9) | \
     (((mw)&1)<<10) | (((tu)&0xF)<<11) | (((ia)&0xF)<<15) | \
     (((sat)&1)<<19) | (((end)&1)<<20))

#define FP_PACK_W1(dst, stype, sidx, sx, sy, sz, sw, sneg) \
    (((dst)&0xFF) | (((stype)&3)<<8) | (((sidx)&0xFF)<<10) | \
     (((sx)&3)<<18) | (((sy)&3)<<20) | (((sz)&3)<<22) | (((sw)&3)<<24) | \
     (((sneg)&1)<<26))

#define FP_PACK_SRC(stype, sidx, sx, sy, sz, sw, sneg) \
    (((stype)&3) | (((sidx)&0xFF)<<2) | \
     (((sx)&3)<<10) | (((sy)&3)<<12) | (((sz)&3)<<14) | (((sw)&3)<<16) | \
     (((sneg)&1)<<18))

// FP opcodes (matching rsx_fp_shader.h)
#define FP_OP_NOP  0x00
#define FP_OP_MOV  0x01
#define FP_OP_MUL  0x02
#define FP_OP_ADD  0x03
#define FP_OP_MAD  0x04
#define FP_OP_DP3  0x05
#define FP_OP_DP4  0x06
#define FP_OP_MIN  0x08
#define FP_OP_MAX  0x09
#define FP_OP_SLT  0x0A
#define FP_OP_SGE  0x0B
#define FP_OP_FRC  0x10
#define FP_OP_FLR  0x11
#define FP_OP_TEX  0x17
#define FP_OP_TXP  0x18
#define FP_OP_RCP  0x1A
#define FP_OP_RSQ  0x1B
#define FP_OP_EX2  0x1C
#define FP_OP_LG2  0x1D
#define FP_OP_LRP  0x1F
#define FP_OP_COS  0x22
#define FP_OP_SIN  0x23
#define FP_OP_POW  0x26
#define FP_OP_KIL  0x28
#define FP_OP_DST  0x07
#define FP_OP_DP2  0x0C
#define FP_OP_DP2A 0x0D
#define FP_OP_SEQ  0x0E
#define FP_OP_SFL  0x0F
#define FP_OP_SGT  0x12
#define FP_OP_SLE  0x13
#define FP_OP_SNE  0x14
#define FP_OP_STR  0x15
#define FP_OP_LIT  0x16
#define FP_OP_TXB  0x19
#define FP_OP_DIV  0x1E
#define FP_OP_DDX  0x20
#define FP_OP_DDY  0x21
#define FP_OP_NRM  0x24
#define FP_OP_FENCB 0x3E
#define FP_OP_FENCT 0x3D

// Source register types
#define FP_SRC_TEMP  0
#define FP_SRC_INPUT 1
#define FP_SRC_CONST 2

struct FPVec4 { float x, y, z, w; };

// Texture bank passed to kernel — 4 units
struct TexBank {
    const uint32_t* tex[4];
    uint32_t w[4], h[4];
    uint8_t wrapS[4], wrapT[4];  // RSX wrap: 1=REPEAT, 2=MIRROR, 3=CLAMP_EDGE
    uint8_t magFilter[4];        // 0=NEAREST, 1=LINEAR (per unit)
};

// Read a source operand with swizzle and negate
__device__ __forceinline__ FPVec4 fpReadSrc(
    uint32_t packed,  // packed source word
    const FPVec4* temps, uint32_t nTemps,
    const FPVec4* fpInputs, uint32_t nInputs,
    const float* consts, uint32_t nConsts,
    uint32_t inputAttr)
{
    uint32_t stype = packed & 3;
    uint32_t sidx  = (packed >> 2) & 0xFF;
    uint32_t sx    = (packed >> 10) & 3;
    uint32_t sy    = (packed >> 12) & 3;
    uint32_t sz    = (packed >> 14) & 3;
    uint32_t sw    = (packed >> 16) & 3;
    bool     neg   = (packed >> 18) & 1;

    FPVec4 base = {0,0,0,0};
    if (stype == FP_SRC_TEMP) {
        if (sidx < nTemps) base = temps[sidx];
    } else if (stype == FP_SRC_INPUT) {
        if (inputAttr < nInputs) base = fpInputs[inputAttr];
    } else if (stype == FP_SRC_CONST) {
        if (sidx * 4 + 3 < nConsts * 4) {
            base.x = consts[sidx*4+0]; base.y = consts[sidx*4+1];
            base.z = consts[sidx*4+2]; base.w = consts[sidx*4+3];
        }
    }

    // Apply swizzle
    float arr[4] = {base.x, base.y, base.z, base.w};
    FPVec4 result = {arr[sx], arr[sy], arr[sz], arr[sw]};

    if (neg) { result.x = -result.x; result.y = -result.y;
               result.z = -result.z; result.w = -result.w; }
    return result;
}

// Write result to destination with mask and optional saturate
__device__ __forceinline__ void fpWriteDst(
    FPVec4* temps, uint32_t dst, const FPVec4& val,
    bool mx, bool my, bool mz, bool mw, bool sat, uint32_t nTemps)
{
    if (dst >= nTemps) return;
    FPVec4& d = temps[dst];
    auto clmp = [](float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
    if (mx) d.x = sat ? clmp(val.x) : val.x;
    if (my) d.y = sat ? clmp(val.y) : val.y;
    if (mz) d.z = sat ? clmp(val.z) : val.z;
    if (mw) d.w = sat ? clmp(val.w) : val.w;
}

// Execute fragment program. Returns false if pixel was KIL'd (discard).
__device__ bool fpExecute(
    const uint32_t* __restrict__ insns, uint32_t insnCount,
    const float* __restrict__ consts, uint32_t constCount,
    const FPVec4* fpInputs, uint32_t nInputs,
    const TexBank& texBank,
    float& outR, float& outG, float& outB, float& outA,
    FPVec4* mrtOut)  // mrtOut[0..3] for MRT planes B/C/D (mrtOut may be null)
{
    FPVec4 temps[8] = {};  // r0..r7 (most FPs use few regs)
    const uint32_t nTemps = 8;

    for (uint32_t pc = 0; pc < insnCount; ++pc) {
        const uint32_t* iw = insns + pc * 6;
        uint32_t w0 = iw[0];
        uint32_t opcode = w0 & 0x7F;
        bool mx  = (w0 >> 7) & 1;
        bool my  = (w0 >> 8) & 1;
        bool mz  = (w0 >> 9) & 1;
        bool mw  = (w0 >> 10) & 1;
        uint32_t texUnit  = (w0 >> 11) & 0xF;
        uint32_t inAttr   = (w0 >> 15) & 0xF;
        bool sat = (w0 >> 19) & 1;
        bool end = (w0 >> 20) & 1;

        uint32_t dst = iw[1] & 0xFF;
        uint32_t src0packed = (iw[1] >> 8);
        uint32_t src1packed = iw[2];
        uint32_t src2packed = iw[3];

        FPVec4 s0 = fpReadSrc(src0packed, temps, nTemps, fpInputs, nInputs, consts, constCount, inAttr);
        FPVec4 s1 = fpReadSrc(src1packed, temps, nTemps, fpInputs, nInputs, consts, constCount, inAttr);
        FPVec4 s2 = fpReadSrc(src2packed, temps, nTemps, fpInputs, nInputs, consts, constCount, inAttr);

        FPVec4 result = {0,0,0,0};

        switch (opcode) {
        case FP_OP_NOP:
        case FP_OP_FENCT:
        case FP_OP_FENCB:
            break;
        case FP_OP_MOV:
            result = s0;
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_MUL:
            result = {s0.x*s1.x, s0.y*s1.y, s0.z*s1.z, s0.w*s1.w};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_ADD:
            result = {s0.x+s1.x, s0.y+s1.y, s0.z+s1.z, s0.w+s1.w};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_MAD:
            result = {s0.x*s1.x+s2.x, s0.y*s1.y+s2.y, s0.z*s1.z+s2.z, s0.w*s1.w+s2.w};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_DP3: {
            float d = s0.x*s1.x + s0.y*s1.y + s0.z*s1.z;
            result = {d, d, d, d};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        case FP_OP_DP4: {
            float d = s0.x*s1.x + s0.y*s1.y + s0.z*s1.z + s0.w*s1.w;
            result = {d, d, d, d};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        case FP_OP_MIN:
            result = {fminf(s0.x,s1.x), fminf(s0.y,s1.y), fminf(s0.z,s1.z), fminf(s0.w,s1.w)};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_MAX:
            result = {fmaxf(s0.x,s1.x), fmaxf(s0.y,s1.y), fmaxf(s0.z,s1.z), fmaxf(s0.w,s1.w)};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_RCP:
            result = {1.0f/s0.x, 1.0f/s0.x, 1.0f/s0.x, 1.0f/s0.x};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_RSQ:
            result = {rsqrtf(fabsf(s0.x)), rsqrtf(fabsf(s0.x)), rsqrtf(fabsf(s0.x)), rsqrtf(fabsf(s0.x))};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_TEX: {
            float tr, tg, tb, ta;
            if (texUnit < 4 && texBank.tex[texUnit] &&
                texBank.w[texUnit] > 0 && texBank.h[texUnit] > 0) {
                sampleTex(texBank.tex[texUnit], texBank.w[texUnit],
                          texBank.h[texUnit], s0.x, s0.y, (int)texBank.magFilter[texUnit],
                          tr, tg, tb, ta,
                          texBank.wrapS[texUnit], texBank.wrapT[texUnit]);
            } else {
                tr = tg = tb = ta = 1.0f;
            }
            result = {tr, tg, tb, ta};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        case FP_OP_LRP: // lerp: s0*(s1-s2)+s2 = mix(s2,s1,s0)
            result = {s0.x*(s1.x-s2.x)+s2.x, s0.y*(s1.y-s2.y)+s2.y,
                      s0.z*(s1.z-s2.z)+s2.z, s0.w*(s1.w-s2.w)+s2.w};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_FRC:
            result = {s0.x - floorf(s0.x), s0.y - floorf(s0.y),
                      s0.z - floorf(s0.z), s0.w - floorf(s0.w)};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_FLR:
            result = {floorf(s0.x), floorf(s0.y), floorf(s0.z), floorf(s0.w)};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_SLT:
            result = {s0.x<s1.x?1.f:0.f, s0.y<s1.y?1.f:0.f,
                      s0.z<s1.z?1.f:0.f, s0.w<s1.w?1.f:0.f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_SGE:
            result = {s0.x>=s1.x?1.f:0.f, s0.y>=s1.y?1.f:0.f,
                      s0.z>=s1.z?1.f:0.f, s0.w>=s1.w?1.f:0.f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_EX2:
            result = {exp2f(s0.x), exp2f(s0.x), exp2f(s0.x), exp2f(s0.x)};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_LG2:
            result = {log2f(fabsf(s0.x)), log2f(fabsf(s0.x)), log2f(fabsf(s0.x)), log2f(fabsf(s0.x))};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_COS:
            result = {cosf(s0.x), cosf(s0.x), cosf(s0.x), cosf(s0.x)};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_SIN:
            result = {sinf(s0.x), sinf(s0.x), sinf(s0.x), sinf(s0.x)};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_POW:
            result = {powf(fabsf(s0.x), s1.x), powf(fabsf(s0.x), s1.x),
                      powf(fabsf(s0.x), s1.x), powf(fabsf(s0.x), s1.x)};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_KIL:
            // Discard pixel if any component of src0 < 0
            if (s0.x < 0.0f || s0.y < 0.0f || s0.z < 0.0f || s0.w < 0.0f)
                return false;
            break;
        case FP_OP_DST: {
            // Distance: dst = (1, s0.y*s1.y, s0.z, s1.w)
            result = {1.0f, s0.y * s1.y, s0.z, s1.w};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        case FP_OP_DP2: {
            float d = s0.x*s1.x + s0.y*s1.y;
            result = {d, d, d, d};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        case FP_OP_DP2A: {
            float d = s0.x*s1.x + s0.y*s1.y + s2.x;
            result = {d, d, d, d};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        case FP_OP_SEQ:
            result = {s0.x==s1.x?1.f:0.f, s0.y==s1.y?1.f:0.f,
                      s0.z==s1.z?1.f:0.f, s0.w==s1.w?1.f:0.f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_SFL:
            result = {0.f, 0.f, 0.f, 0.f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_SGT:
            result = {s0.x>s1.x?1.f:0.f, s0.y>s1.y?1.f:0.f,
                      s0.z>s1.z?1.f:0.f, s0.w>s1.w?1.f:0.f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_SLE:
            result = {s0.x<=s1.x?1.f:0.f, s0.y<=s1.y?1.f:0.f,
                      s0.z<=s1.z?1.f:0.f, s0.w<=s1.w?1.f:0.f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_SNE:
            result = {s0.x!=s1.x?1.f:0.f, s0.y!=s1.y?1.f:0.f,
                      s0.z!=s1.z?1.f:0.f, s0.w!=s1.w?1.f:0.f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_STR:
            result = {1.f, 1.f, 1.f, 1.f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_LIT: {
            // Lighting: dst = (1, max(s0.x,0), s0.y>0?pow(s0.y, clamp(s0.w,-128,128)):0, 1)
            float diffuse = fmaxf(s0.x, 0.0f);
            float specExp = fminf(fmaxf(s0.w, -128.f), 128.f);
            float specular = s0.y > 0.0f ? powf(fmaxf(s0.y, 0.0f), specExp) : 0.0f;
            result = {1.0f, diffuse, specular, 1.0f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        case FP_OP_TXP: {
            // Projective texture: divide by w then sample
            float w = (s0.w != 0.0f) ? s0.w : 1.0f;
            float pu = s0.x / w, pv = s0.y / w;
            float tr, tg, tb, ta;
            if (texUnit < 4 && texBank.tex[texUnit] &&
                texBank.w[texUnit] > 0 && texBank.h[texUnit] > 0) {
                sampleTex(texBank.tex[texUnit], texBank.w[texUnit],
                          texBank.h[texUnit], pu, pv, (int)texBank.magFilter[texUnit],
                          tr, tg, tb, ta,
                          texBank.wrapS[texUnit], texBank.wrapT[texUnit]);
            } else {
                tr = tg = tb = ta = 1.0f;
            }
            result = {tr, tg, tb, ta};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        case FP_OP_TXB: {
            // Biased texture: LOD bias in s0.w; we ignore bias (no mipmaps yet)
            float tr, tg, tb, ta;
            if (texUnit < 4 && texBank.tex[texUnit] &&
                texBank.w[texUnit] > 0 && texBank.h[texUnit] > 0) {
                sampleTex(texBank.tex[texUnit], texBank.w[texUnit],
                          texBank.h[texUnit], s0.x, s0.y, (int)texBank.magFilter[texUnit],
                          tr, tg, tb, ta,
                          texBank.wrapS[texUnit], texBank.wrapT[texUnit]);
            } else {
                tr = tg = tb = ta = 1.0f;
            }
            result = {tr, tg, tb, ta};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        case FP_OP_DIV:
            result = {s0.x/s1.x, s0.y/s1.y, s0.z/s1.z, s0.w/s1.w};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_DDX:
        case FP_OP_DDY:
            // Screen-space derivatives: approximate as zero (no quad-level differencing)
            result = {0.f, 0.f, 0.f, 0.f};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        case FP_OP_NRM: {
            float len = sqrtf(s0.x*s0.x + s0.y*s0.y + s0.z*s0.z);
            float inv = (len > 0.0f) ? (1.0f / len) : 0.0f;
            result = {s0.x*inv, s0.y*inv, s0.z*inv, s0.w*inv};
            fpWriteDst(temps, dst, result, mx, my, mz, mw, sat, nTemps);
            break;
        }
        default:
            break;
        }

        if (end) break;
    }

    // Output is r0
    outR = temps[0].x; outG = temps[0].y;
    outB = temps[0].z; outA = temps[0].w;
    // MRT outputs: temps[1]→B, temps[2]→C, temps[3]→D
    if (mrtOut) {
        mrtOut[0] = temps[1];
        mrtOut[1] = temps[2];
        mrtOut[2] = temps[3];
    }
    return true;  // pixel not discarded
}

__device__ __forceinline__ bool depthCompare(uint32_t func, float src, float dst) {
    switch (func) {
    case 0: return false;            // Never
    case 1: return src <  dst;       // Less
    case 2: return src == dst;       // Equal
    case 3: return src <= dst;       // LEqual
    case 4: return src >  dst;       // Greater
    case 5: return src != dst;       // NotEqual
    case 6: return src >= dst;       // GEqual
    default: return true;            // Always
    }
}

__device__ __forceinline__ float blendFactor(uint32_t f,
                                             float sc, float dc,
                                             float sa, float da,
                                             float cc, float ca,
                                             int channelIsAlpha) {
    // sc/dc/sa/da: source/dest color channel (r or g or b) and alpha.
    // cc: constant color channel.  ca: constant alpha.
    switch (f) {
    case 0:  return 0.0f;                    // Zero
    case 1:  return 1.0f;                    // One
    case 2:  return sc;                      // SrcColor
    case 3:  return 1.0f - sc;               // OneMinusSrcColor
    case 4:  return dc;                      // DstColor
    case 5:  return 1.0f - dc;               // OneMinusDstColor
    case 6:  return sa;                      // SrcAlpha
    case 7:  return 1.0f - sa;               // OneMinusSrcAlpha
    case 8:  return da;                      // DstAlpha
    case 9:  return 1.0f - da;               // OneMinusDstAlpha
    case 10: return cc;                      // ConstColor
    case 11: return 1.0f - cc;               // OneMinusConstColor
    case 12: return ca;                      // ConstAlpha
    case 13: return 1.0f - ca;               // OneMinusConstAlpha
    case 14: {                               // SrcAlphaSaturate
        if (channelIsAlpha) return 1.0f;
        float f1 = 1.0f - da;
        return sa < f1 ? sa : f1;
    }
    default: return 1.0f;
    }
}

__device__ __forceinline__ float blendEquation(uint32_t eq,
                                               float src, float dst) {
    switch (eq) {
    case 0: return src + dst;                // Add
    case 1: return src - dst;                // Subtract
    case 2: return dst - src;                // RevSubtract
    case 3: return src < dst ? src : dst;    // Min
    case 4: return src > dst ? src : dst;    // Max
    default: return src + dst;
    }
}

__device__ __forceinline__ bool stencilCompare(uint32_t func, uint8_t a, uint8_t b) {
    switch (func) {
    case 0: return false;
    case 1: return a <  b;
    case 2: return a == b;
    case 3: return a <= b;
    case 4: return a >  b;
    case 5: return a != b;
    case 6: return a >= b;
    default: return true;
    }
}

__device__ __forceinline__ uint8_t stencilApply(uint32_t op, uint8_t cur, uint8_t ref) {
    switch (op) {
    case 0: return cur;                                  // Keep
    case 1: return 0;                                    // Zero
    case 2: return ref;                                  // Replace
    case 3: return cur == 255 ? 255 : cur + 1;           // IncrSat
    case 4: return cur == 0   ? 0   : cur - 1;           // DecrSat
    case 5: return (uint8_t)(~cur);                      // Invert
    case 6: return (uint8_t)(cur + 1);                   // IncrWrap
    case 7: return (uint8_t)(cur - 1);                   // DecrWrap
    default: return cur;
    }
}

__device__ __forceinline__ int texWrap(int coord, int dim, uint8_t mode) {
    // 1=REPEAT, 2=MIRROR_REPEAT, 3=CLAMP_TO_EDGE, 4=BORDER, 5=CLAMP
    switch (mode) {
    case 2: { // MIRROR_REPEAT
        int period = coord / dim;
        coord = coord % dim;
        if (coord < 0) { coord += dim; period--; }
        if (period & 1) coord = dim - 1 - coord;
        return coord;
    }
    case 3: // CLAMP_TO_EDGE
    case 5: // CLAMP (same for us — no border color)
    case 4: // BORDER (treat as clamp)
        return (coord < 0) ? 0 : (coord >= dim) ? dim - 1 : coord;
    default: // 1 = REPEAT (and fallback)
        coord = coord % dim;
        if (coord < 0) coord += dim;
        return coord;
    }
}

__device__ __forceinline__ uint32_t texFetch(const uint32_t* tex,
                                             uint32_t tw, uint32_t th,
                                             int tx, int ty,
                                             uint8_t wrapS = 1, uint8_t wrapT = 1) {
    tx = texWrap(tx, (int)tw, wrapS);
    ty = texWrap(ty, (int)th, wrapT);
    return tex[ty * tw + tx];
}

__device__ __forceinline__ void sampleTex(const uint32_t* tex,
                                          uint32_t tw, uint32_t th,
                                          float u, float v, int bilinear,
                                          float& r, float& g, float& b, float& a,
                                          uint8_t wrapS = 1, uint8_t wrapT = 1) {
    float fx = u * (float)tw - 0.5f;
    float fy = v * (float)th - 0.5f;
    int ix = (int)floorf(fx);
    int iy = (int)floorf(fy);
    if (!bilinear) {
        uint32_t c = texFetch(tex, tw, th, ix, iy, wrapS, wrapT);
        r = ((c >> 16) & 0xFF) / 255.0f;
        g = ((c >>  8) & 0xFF) / 255.0f;
        b = ((c >>  0) & 0xFF) / 255.0f;
        a = ((c >> 24) & 0xFF) / 255.0f;
        return;
    }
    float sx = fx - ix, sy = fy - iy;
    uint32_t c00 = texFetch(tex, tw, th, ix,   iy,   wrapS, wrapT);
    uint32_t c10 = texFetch(tex, tw, th, ix+1, iy,   wrapS, wrapT);
    uint32_t c01 = texFetch(tex, tw, th, ix,   iy+1, wrapS, wrapT);
    uint32_t c11 = texFetch(tex, tw, th, ix+1, iy+1, wrapS, wrapT);
    auto lerp = [] __device__ (float a, float b, float t) { return a + (b - a) * t; };
    auto ch = [&](int shift) {
        float v00 = ((c00>>shift)&0xFF)/255.0f;
        float v10 = ((c10>>shift)&0xFF)/255.0f;
        float v01 = ((c01>>shift)&0xFF)/255.0f;
        float v11 = ((c11>>shift)&0xFF)/255.0f;
        return lerp(lerp(v00, v10, sx), lerp(v01, v11, sx), sy);
    };
    r = ch(16); g = ch(8); b = ch(0); a = ch(24);
}

// One thread per pixel per triangle batch. For each pixel we iterate the
// triangle list and keep the last one that covers it (painter order).
// That matches RSX "no depth test" behaviour when depth is disabled —
// and depth will be added as an early-Z stage later.
//
// Edge function: E(p) = (p.x - v0.x)*(v1.y - v0.y) - (p.y - v0.y)*(v1.x - v0.x)
// For a CCW triangle all three edge functions are >= 0 inside.
// Top-left fill rule handled via half-open interval (>= 0 for top/left
// edges, > 0 for others) — we approximate with >= 0 which is acceptable
// for colored triangles; textured paths will tighten this.

__global__ void k_rasterTriangles(uint32_t* __restrict__ dst,
                                  float*    __restrict__ depth,
                                  uint8_t*  __restrict__ stencil,
                                  uint32_t width, uint32_t height,
                                  const RasterVertex* __restrict__ verts,
                                  uint32_t triangleCount,
                                  int blendEnable,
                                  int depthTest,
                                  int depthWrite,
                                  uint32_t depthFunc,
                                  TexBank texBank,
                                  int scX, int scY, uint32_t scW, uint32_t scH,
                                  int alphaTest, uint32_t alphaRef,
                                  uint32_t alphaFunc,
                                  uint32_t colorMask,
                                  float polyOffsetFactor, float polyOffsetUnits,
                                  int   depthClip,
                                  int   stencilTest,
                                  uint32_t stencilFunc,
                                  uint32_t stencilRef,
                                  uint32_t stencilMask,
                                  uint32_t stencilWriteMask,
                                  uint32_t opSFail,
                                  uint32_t opZFail,
                                  uint32_t opZPass,
                                  uint32_t bfSrcRGB, uint32_t bfDstRGB,
                                  uint32_t bfSrcA,   uint32_t bfDstA,
                                  uint32_t beRGB,    uint32_t beA,
                                  float    ccR, float ccG, float ccB, float ccA,
                                  const uint32_t* __restrict__ fpInsns,
                                  uint32_t fpInsnCount,
                                  const float* __restrict__ fpConsts,
                                  uint32_t fpConstCount,
                                  uint32_t* __restrict__ mrtB,
                                  uint32_t* __restrict__ mrtC,
                                  uint32_t* __restrict__ mrtD) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Scissor test: when scW>0 and scH>0, reject pixels outside rect.
    if (scW > 0 && scH > 0) {
        if ((int)x < scX || (int)x >= scX + (int)scW ||
            (int)y < scY || (int)y >= scY + (int)scH) return;
    }

    float px = (float)x + 0.5f;
    float py = (float)y + 0.5f;

    uint32_t base = y * width + x;
    uint32_t dstPx = dst[base];
    float    dstZ  = depth ? depth[base] : 1.0f;
    uint8_t  dstS  = stencil ? stencil[base] : 0;
    uint32_t mrtPxB = mrtB ? mrtB[base] : 0;
    uint32_t mrtPxC = mrtC ? mrtC[base] : 0;
    uint32_t mrtPxD = mrtD ? mrtD[base] : 0;
    FPVec4   mrtAccum[3] = {};  // accumulated MRT values from FP

    for (uint32_t t = 0; t < triangleCount; ++t) {
        const RasterVertex v0 = verts[t*3 + 0];
        const RasterVertex v1 = verts[t*3 + 1];
        const RasterVertex v2 = verts[t*3 + 2];

        float area = (v2.x - v0.x) * (v1.y - v0.y)
                   - (v2.y - v0.y) * (v1.x - v0.x);
        if (area == 0.0f) continue;
        float invArea = 1.0f / area;

        float e0 = (px - v1.x) * (v2.y - v1.y) - (py - v1.y) * (v2.x - v1.x);
        float e1 = (px - v2.x) * (v0.y - v2.y) - (py - v2.y) * (v0.x - v2.x);
        float e2 = (px - v0.x) * (v1.y - v0.y) - (py - v0.y) * (v1.x - v0.x);

        bool inside = area > 0.0f ? (e0 >= 0 && e1 >= 0 && e2 >= 0)
                                  : (e0 <= 0 && e1 <= 0 && e2 <= 0);
        if (!inside) continue;

        float w0 = e0 * invArea;
        float w1 = e1 * invArea;
        float w2 = 1.0f - w0 - w1;

        float z = w0 * v0.z + w1 * v1.z + w2 * v2.z;

        // Polygon offset (depth bias) for decals and shadow maps
        if (polyOffsetFactor != 0.0f || polyOffsetUnits != 0.0f) {
            // Approximate dz/dx and dz/dy from triangle vertices
            float dzdx = (v1.z - v0.z) * (v2.y - v0.y) - (v2.z - v0.z) * (v1.y - v0.y);
            float dzdy = (v2.z - v0.z) * (v1.x - v0.x) - (v1.z - v0.z) * (v2.x - v0.x);
            float maxSlope = fmaxf(fabsf(dzdx), fabsf(dzdy)) * invArea;
            float r_unit = 1.0f / 16777216.0f;  // 1/2^24 for 24-bit depth
            z += polyOffsetFactor * maxSlope + polyOffsetUnits * r_unit;
        }

        // Near/far plane clip (NDC-space depth after /w is mapped to
        // [0,1]; values outside mean the pixel is past the near/far
        // plane). Matches default RSX/GL behaviour — games can keep
        // relying on depth clamping by not enabling NV4097 depth-clamp.
        if (depthClip && (z < 0.0f || z > 1.0f)) continue;

        // Stencil test happens before depth test (matches RSX/GL order).
        bool sPass = true;
        if (stencilTest && stencil) {
            uint8_t refM = (uint8_t)(stencilRef & stencilMask);
            uint8_t curM = (uint8_t)(dstS & stencilMask);
            sPass = stencilCompare(stencilFunc, refM, curM);
        }

        bool zPass = true;
        if (depthTest && depth) {
            zPass = depthCompare(depthFunc, z, dstZ);
        }

        // Stencil-op selection + masked write.
        if (stencilTest && stencil) {
            uint32_t op = sPass ? (zPass ? opZPass : opZFail) : opSFail;
            uint8_t newS = stencilApply(op, dstS, (uint8_t)stencilRef);
            dstS = (uint8_t)((dstS & ~stencilWriteMask) | (newS & stencilWriteMask));
        }

        if (!sPass) continue;
        if (!zPass) continue;

        float r = w0 * v0.r + w1 * v1.r + w2 * v2.r;
        float g = w0 * v0.g + w1 * v1.g + w2 * v2.g;
        float b = w0 * v0.b + w1 * v1.b + w2 * v2.b;
        float a = w0 * v0.a + w1 * v1.a + w2 * v2.a;

        float u = w0 * v0.u + w1 * v1.u + w2 * v2.u;
        float vv = w0 * v0.v + w1 * v1.v + w2 * v2.v;

        if (fpInsns && fpInsnCount > 0) {
            // Build FP input array from interpolated vertex data
            // 0=WPOS, 1=COL0, 2=COL1, 3=FOGC, 4=TEX0, 5=TEX1, 6=TEX2, 7=TEX3
            FPVec4 fpIn[8];
            fpIn[0] = {(float)x + 0.5f, (float)y + 0.5f, z, 1.0f};  // WPOS
            fpIn[1] = {r, g, b, a};  // COL0
            fpIn[2] = {w0*v0.col1[0]+w1*v1.col1[0]+w2*v2.col1[0],
                       w0*v0.col1[1]+w1*v1.col1[1]+w2*v2.col1[1],
                       w0*v0.col1[2]+w1*v1.col1[2]+w2*v2.col1[2],
                       w0*v0.col1[3]+w1*v1.col1[3]+w2*v2.col1[3]};  // COL1
            fpIn[3] = {w0*v0.fog+w1*v1.fog+w2*v2.fog, 0, 0, 1};  // FOGC
            fpIn[4] = {u, vv, 0.0f, 1.0f};  // TEX0
            fpIn[5] = {w0*v0.tex1[0]+w1*v1.tex1[0]+w2*v2.tex1[0],
                       w0*v0.tex1[1]+w1*v1.tex1[1]+w2*v2.tex1[1], 0, 1};  // TEX1
            fpIn[6] = {w0*v0.tex2[0]+w1*v1.tex2[0]+w2*v2.tex2[0],
                       w0*v0.tex2[1]+w1*v1.tex2[1]+w2*v2.tex2[1], 0, 1};  // TEX2
            fpIn[7] = {w0*v0.tex3[0]+w1*v1.tex3[0]+w2*v2.tex3[0],
                       w0*v0.tex3[1]+w1*v1.tex3[1]+w2*v2.tex3[1], 0, 1};  // TEX3
            FPVec4 mrtVals[3] = {};
            bool alive = fpExecute(fpInsns, fpInsnCount, fpConsts, fpConstCount,
                      fpIn, 8, texBank,
                      r, g, b, a, mrtVals);
            if (!alive) continue;  // KIL'd — discard pixel
            mrtAccum[0] = mrtVals[0];
            mrtAccum[1] = mrtVals[1];
            mrtAccum[2] = mrtVals[2];
        } else if (texBank.tex[0]) {
            float tr, tg, tb, ta;
            sampleTex(texBank.tex[0], texBank.w[0], texBank.h[0],
                      u, vv, (int)texBank.magFilter[0], tr, tg, tb, ta,
                      texBank.wrapS[0], texBank.wrapT[0]);
            // Fallback: texture replaces vertex color when no FP
            r = tr; g = tg; b = tb; a = ta;
        }

        if (alphaTest) {
            uint32_t af = (uint32_t)(a * 255.0f + 0.5f);
            bool pass = false;
            switch (alphaFunc) {
                case 0x0200: pass = false; break;          // NEVER
                case 0x0201: pass = af <  alphaRef; break;  // LESS
                case 0x0202: pass = af == alphaRef; break;  // EQUAL
                case 0x0203: pass = af <= alphaRef; break;  // LEQUAL
                case 0x0204: pass = af >  alphaRef; break;  // GREATER
                case 0x0205: pass = af != alphaRef; break;  // NOTEQUAL
                case 0x0206: pass = af >= alphaRef; break;  // GEQUAL
                default:     pass = true; break;            // ALWAYS (0x0207)
            }
            if (!pass) continue;
        }

        if (blendEnable) {
            uint32_t dc = dstPx;
            float dr = ((dc >> 16) & 0xFF) / 255.0f;
            float dg = ((dc >>  8) & 0xFF) / 255.0f;
            float db = ((dc >>  0) & 0xFF) / 255.0f;
            float da = ((dc >> 24) & 0xFF) / 255.0f;

            // Evaluate per-channel source/dest factors.
            float fSr = blendFactor(bfSrcRGB, r, dr, a, da, ccR, ccA, 0);
            float fSg = blendFactor(bfSrcRGB, g, dg, a, da, ccG, ccA, 0);
            float fSb = blendFactor(bfSrcRGB, b, db, a, da, ccB, ccA, 0);
            float fSa = blendFactor(bfSrcA,   a, da, a, da, ccA, ccA, 1);
            float fDr = blendFactor(bfDstRGB, r, dr, a, da, ccR, ccA, 0);
            float fDg = blendFactor(bfDstRGB, g, dg, a, da, ccG, ccA, 0);
            float fDb = blendFactor(bfDstRGB, b, db, a, da, ccB, ccA, 0);
            float fDa = blendFactor(bfDstA,   a, da, a, da, ccA, ccA, 1);

            r = blendEquation(beRGB, r * fSr, dr * fDr);
            g = blendEquation(beRGB, g * fSg, dg * fDg);
            b = blendEquation(beRGB, b * fSb, db * fDb);
            a = blendEquation(beA,   a * fSa, da * fDa);
        }

        auto sat = [](float v) -> uint32_t {
            int i = (int)(v * 255.0f + 0.5f);
            if (i < 0) i = 0; else if (i > 255) i = 255;
            return (uint32_t)i;
        };
        uint32_t newPx = (sat(a) << 24) | (sat(r) << 16) | (sat(g) << 8) | sat(b);
        // Color mask: RSX bit layout — 0x01000000=R, 0x00010000=G, 0x00000100=B, 0x00000001=A
        if (colorMask != 0x01010101u) {
            uint32_t keep = 0, write = 0;
            if (colorMask & 0x01000000u) write |= 0x00FF0000u; else keep |= 0x00FF0000u;
            if (colorMask & 0x00010000u) write |= 0x0000FF00u; else keep |= 0x0000FF00u;
            if (colorMask & 0x00000100u) write |= 0x000000FFu; else keep |= 0x000000FFu;
            if (colorMask & 0x00000001u) write |= 0xFF000000u; else keep |= 0xFF000000u;
            newPx = (dstPx & keep) | (newPx & write);
        }
        dstPx = newPx;
        // MRT planes — convert FP output to ARGB8
        if (mrtB) {
            mrtPxB = (sat(mrtAccum[0].w) << 24) | (sat(mrtAccum[0].x) << 16) |
                     (sat(mrtAccum[0].y) << 8) | sat(mrtAccum[0].z);
        }
        if (mrtC) {
            mrtPxC = (sat(mrtAccum[1].w) << 24) | (sat(mrtAccum[1].x) << 16) |
                     (sat(mrtAccum[1].y) << 8) | sat(mrtAccum[1].z);
        }
        if (mrtD) {
            mrtPxD = (sat(mrtAccum[2].w) << 24) | (sat(mrtAccum[2].x) << 16) |
                     (sat(mrtAccum[2].y) << 8) | sat(mrtAccum[2].z);
        }
        if (depthWrite) dstZ = z;
    }

    dst[base] = dstPx;
    if (mrtB) mrtB[base] = mrtPxB;
    if (mrtC) mrtC[base] = mrtPxC;
    if (mrtD) mrtD[base] = mrtPxD;
    if (depth && depthWrite) depth[base] = dstZ;
    if (stencil && stencilTest) stencil[base] = dstS;
}

// ─────────────────────────────────────────────────────────────────
// Line kernel — one thread per pixel, iterates segments. Uses
// parametric-distance coverage: for each segment A→B compute the
// nearest parameter t ∈ [0,1] from pixel P, then cover if the
// perpendicular distance |P - (A + t*(B-A))| ≤ 0.5 px. Endpoints are
// extended by 0.5 px so the last pixel of each end draws.
// Color / alpha / depth are interpolated by t.
// ─────────────────────────────────────────────────────────────────
__global__ void k_rasterLines(uint32_t* __restrict__ dst,
                              float*    __restrict__ depth,
                              uint32_t width, uint32_t height,
                              const RasterVertex* __restrict__ verts,
                              uint32_t segmentCount,
                              int blendEnable,
                              int depthTest,
                              int depthWrite,
                              uint32_t depthFunc,
                              int scX, int scY, uint32_t scW, uint32_t scH,
                              int alphaTest, uint32_t alphaRef,
                              int depthClip,
                              uint32_t bfSrcRGB, uint32_t bfDstRGB,
                              uint32_t bfSrcA,   uint32_t bfDstA,
                              uint32_t beRGB,    uint32_t beA,
                              float ccR, float ccG, float ccB, float ccA) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (scW > 0 && scH > 0) {
        if ((int)x < scX || (int)x >= scX + (int)scW ||
            (int)y < scY || (int)y >= scY + (int)scH) return;
    }

    float px = (float)x + 0.5f;
    float py = (float)y + 0.5f;

    uint32_t base = y * width + x;
    uint32_t dstPx = dst[base];
    float    dstZ  = depth ? depth[base] : 1.0f;

    for (uint32_t s = 0; s < segmentCount; ++s) {
        const RasterVertex v0 = verts[s*2 + 0];
        const RasterVertex v1 = verts[s*2 + 1];
        float dx = v1.x - v0.x;
        float dy = v1.y - v0.y;
        float len2 = dx*dx + dy*dy;
        if (len2 < 1e-6f) continue;
        float t = ((px - v0.x) * dx + (py - v0.y) * dy) / len2;
        // Extend by 0.5 / length at each end so the endpoint pixels are
        // inside the covered segment.
        float extend = 0.5f / sqrtf(len2);
        if (t < -extend || t > 1.0f + extend) continue;
        float tc = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        float qx = v0.x + tc * dx;
        float qy = v0.y + tc * dy;
        float ex = px - qx, ey = py - qy;
        if (ex*ex + ey*ey > 0.25f) continue;  // > 0.5 px perpendicular

        float z = v0.z + tc * (v1.z - v0.z);
        if (depthClip && (z < 0.0f || z > 1.0f)) continue;
        if (depthTest && depth) {
            if (!depthCompare(depthFunc, z, dstZ)) continue;
        }

        float r = v0.r + tc * (v1.r - v0.r);
        float g = v0.g + tc * (v1.g - v0.g);
        float b = v0.b + tc * (v1.b - v0.b);
        float a = v0.a + tc * (v1.a - v0.a);

        if (alphaTest) {
            uint32_t af = (uint32_t)(a * 255.0f + 0.5f);
            if (af <= alphaRef) continue;
        }

        if (blendEnable) {
            uint32_t dc = dstPx;
            float dr = ((dc >> 16) & 0xFF) / 255.0f;
            float dg = ((dc >>  8) & 0xFF) / 255.0f;
            float db = ((dc >>  0) & 0xFF) / 255.0f;
            float da = ((dc >> 24) & 0xFF) / 255.0f;
            float fSr = blendFactor(bfSrcRGB, r, dr, a, da, ccR, ccA, 0);
            float fSg = blendFactor(bfSrcRGB, g, dg, a, da, ccG, ccA, 0);
            float fSb = blendFactor(bfSrcRGB, b, db, a, da, ccB, ccA, 0);
            float fSa = blendFactor(bfSrcA,   a, da, a, da, ccA, ccA, 1);
            float fDr = blendFactor(bfDstRGB, r, dr, a, da, ccR, ccA, 0);
            float fDg = blendFactor(bfDstRGB, g, dg, a, da, ccG, ccA, 0);
            float fDb = blendFactor(bfDstRGB, b, db, a, da, ccB, ccA, 0);
            float fDa = blendFactor(bfDstA,   a, da, a, da, ccA, ccA, 1);
            r = blendEquation(beRGB, r * fSr, dr * fDr);
            g = blendEquation(beRGB, g * fSg, dg * fDg);
            b = blendEquation(beRGB, b * fSb, db * fDb);
            a = blendEquation(beA,   a * fSa, da * fDa);
        }

        auto sat = [](float v) -> uint32_t {
            int i = (int)(v * 255.0f + 0.5f);
            if (i < 0) i = 0; else if (i > 255) i = 255;
            return (uint32_t)i;
        };
        dstPx = (sat(a) << 24) | (sat(r) << 16) | (sat(g) << 8) | sat(b);
        if (depthWrite) dstZ = z;
    }

    dst[base] = dstPx;
    if (depth && depthWrite) depth[base] = dstZ;
}

// Point kernel — one thread per point, directly writes the rounded
// pixel. Respects scissor + depth + alpha + blend, skips stencil/tex.
__global__ void k_rasterPoints(uint32_t* __restrict__ dst,
                               float*    __restrict__ depth,
                               uint32_t width, uint32_t height,
                               const RasterVertex* __restrict__ verts,
                               uint32_t pointCount,
                               int blendEnable,
                               int depthTest,
                               int depthWrite,
                               uint32_t depthFunc,
                               int scX, int scY, uint32_t scW, uint32_t scH,
                               int alphaTest, uint32_t alphaRef,
                               int depthClip,
                               uint32_t bfSrcRGB, uint32_t bfDstRGB,
                               uint32_t bfSrcA,   uint32_t bfDstA,
                               uint32_t beRGB,    uint32_t beA,
                               float ccR, float ccG, float ccB, float ccA) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pointCount) return;
    const RasterVertex v = verts[i];
    int px = (int)(v.x);  // truncation — ok for "1 px dot at floor(x)".
    int py = (int)(v.y);
    if (px < 0 || py < 0 || px >= (int)width || py >= (int)height) return;
    if (scW > 0 && scH > 0) {
        if (px < scX || px >= scX + (int)scW ||
            py < scY || py >= scY + (int)scH) return;
    }
    uint32_t base = py * width + px;
    float z = v.z;
    if (depthClip && (z < 0.0f || z > 1.0f)) return;
    if (depthTest && depth) {
        if (!depthCompare(depthFunc, z, depth[base])) return;
    }
    float r = v.r, g = v.g, b = v.b, a = v.a;
    if (alphaTest) {
        uint32_t af = (uint32_t)(a * 255.0f + 0.5f);
        if (af <= alphaRef) return;
    }
    if (blendEnable) {
        uint32_t dc = dst[base];
        float dr = ((dc >> 16) & 0xFF) / 255.0f;
        float dg = ((dc >>  8) & 0xFF) / 255.0f;
        float db = ((dc >>  0) & 0xFF) / 255.0f;
        float da = ((dc >> 24) & 0xFF) / 255.0f;
        float fSr = blendFactor(bfSrcRGB, r, dr, a, da, ccR, ccA, 0);
        float fSg = blendFactor(bfSrcRGB, g, dg, a, da, ccG, ccA, 0);
        float fSb = blendFactor(bfSrcRGB, b, db, a, da, ccB, ccA, 0);
        float fSa = blendFactor(bfSrcA,   a, da, a, da, ccA, ccA, 1);
        float fDr = blendFactor(bfDstRGB, r, dr, a, da, ccR, ccA, 0);
        float fDg = blendFactor(bfDstRGB, g, dg, a, da, ccG, ccA, 0);
        float fDb = blendFactor(bfDstRGB, b, db, a, da, ccB, ccA, 0);
        float fDa = blendFactor(bfDstA,   a, da, a, da, ccA, ccA, 1);
        r = blendEquation(beRGB, r * fSr, dr * fDr);
        g = blendEquation(beRGB, g * fSg, dg * fDg);
        b = blendEquation(beRGB, b * fSb, db * fDb);
        a = blendEquation(beA,   a * fSa, da * fDa);
    }
    auto sat = [](float v) -> uint32_t {
        int i = (int)(v * 255.0f + 0.5f);
        if (i < 0) i = 0; else if (i > 255) i = 255;
        return (uint32_t)i;
    };
    dst[base] = (sat(a) << 24) | (sat(r) << 16) | (sat(g) << 8) | sat(b);
    if (depth && depthWrite) depth[base] = z;
}

// ═══════════════════════════════════════════════════════════════
// Host class
// ═══════════════════════════════════════════════════════════════

CudaRasterizer::CudaRasterizer() = default;
CudaRasterizer::~CudaRasterizer() { shutdown(); }

int CudaRasterizer::init(uint32_t width, uint32_t height) {
    shutdown();
    fb_.width = width;
    fb_.height = height;
    size_t pixels = size_t(width) * size_t(height);
    CU_CHECK(cudaMalloc(&fb_.d_color, pixels * sizeof(uint32_t)));
    CU_CHECK(cudaMalloc(&fb_.d_depth, pixels * sizeof(float)));
    CU_CHECK(cudaMalloc(&fb_.d_stencil, pixels * sizeof(uint8_t)));
    CU_CHECK(cudaMemset(fb_.d_color, 0, pixels * sizeof(uint32_t)));
    CU_CHECK(cudaMemset(fb_.d_stencil, 0, pixels * sizeof(uint8_t)));
    // Clear depth to 1.0 (far plane) — use the kernel so we get the right value.
    dim3 bs(16,16);
    dim3 gs((width+bs.x-1)/bs.x, (height+bs.y-1)/bs.y);
    k_clearDepth<<<gs,bs>>>(fb_.d_depth, width, height, 1.0f);
    cudaDeviceSynchronize();
    return 0;
}

void CudaRasterizer::shutdown() {
    if (fb_.d_color) { cudaFree(fb_.d_color); fb_.d_color = nullptr; }
    if (fb_.d_depth) { cudaFree(fb_.d_depth); fb_.d_depth = nullptr; }
    if (fb_.d_stencil) { cudaFree(fb_.d_stencil); fb_.d_stencil = nullptr; }
    if (d_colorB_) { cudaFree(d_colorB_); d_colorB_ = nullptr; }
    if (d_colorC_) { cudaFree(d_colorC_); d_colorC_ = nullptr; }
    if (d_colorD_) { cudaFree(d_colorD_); d_colorD_ = nullptr; }
    mrtCount_ = 1;
    if (d_tex_[0]) { cudaFree(d_tex_[0]); d_tex_[0] = nullptr; }
    if (d_tex_[1]) { cudaFree(d_tex_[1]); d_tex_[1] = nullptr; }
    if (d_tex_[2]) { cudaFree(d_tex_[2]); d_tex_[2] = nullptr; }
    if (d_tex_[3]) { cudaFree(d_tex_[3]); d_tex_[3] = nullptr; }
    texW_[0] = texH_[0] = texW_[1] = texH_[1] = 0;
    texW_[2] = texH_[2] = texW_[3] = texH_[3] = 0;
    if (d_fpInsns_) { cudaFree(d_fpInsns_); d_fpInsns_ = nullptr; }
    if (d_fpConsts_) { cudaFree(d_fpConsts_); d_fpConsts_ = nullptr; }
    fpInsnCount_ = 0; fpConstCount_ = 0;
    fb_.width = fb_.height = 0;
}

int CudaRasterizer::setMRTCount(uint32_t count) {
    if (count < 1) count = 1;
    if (count > 4) count = 4;
    if (!fb_.d_color) return -1;
    size_t bytes = size_t(fb_.width) * size_t(fb_.height) * sizeof(uint32_t);
    auto ensure = [&](uint32_t** slot) -> int {
        if (*slot) return 0;
        CU_CHECK(cudaMalloc(slot, bytes));
        cudaMemset(*slot, 0, bytes);
        return 0;
    };
    auto drop = [](uint32_t** slot) {
        if (*slot) { cudaFree(*slot); *slot = nullptr; }
    };
    if (count >= 2) { int r = ensure(&d_colorB_); if (r) return r; } else drop(&d_colorB_);
    if (count >= 3) { int r = ensure(&d_colorC_); if (r) return r; } else drop(&d_colorC_);
    if (count >= 4) { int r = ensure(&d_colorD_); if (r) return r; } else drop(&d_colorD_);
    mrtCount_ = count;
    return 0;
}

void CudaRasterizer::clearPlane(uint32_t n, uint32_t rgba) {
    uint32_t* buf = nullptr;
    switch (n) {
    case 0: buf = fb_.d_color; break;
    case 1: buf = d_colorB_;   break;
    case 2: buf = d_colorC_;   break;
    case 3: buf = d_colorD_;   break;
    default: return;
    }
    if (!buf) return;
    dim3 bs(16, 16);
    dim3 gs((fb_.width + bs.x - 1) / bs.x,
            (fb_.height + bs.y - 1) / bs.y);
    k_clear<<<gs, bs>>>(buf, fb_.width, fb_.height, rgba);
    cudaDeviceSynchronize();
    stats.clears++;
}

void CudaRasterizer::readbackPlane(uint32_t n, uint32_t* out) const {
    const uint32_t* buf = nullptr;
    switch (n) {
    case 0: buf = fb_.d_color; break;
    case 1: buf = d_colorB_;   break;
    case 2: buf = d_colorC_;   break;
    case 3: buf = d_colorD_;   break;
    default: return;
    }
    if (!buf || !out) return;
    size_t bytes = size_t(fb_.width) * size_t(fb_.height) * sizeof(uint32_t);
    cudaMemcpy(out, buf, bytes, cudaMemcpyDeviceToHost);
}

int CudaRasterizer::setTexture2D(const uint32_t* data, uint32_t w, uint32_t h,
                                 uint32_t unit) {
    if (unit >= MAX_TEX_UNITS) return -1;
    if (d_tex_[unit]) { cudaFree(d_tex_[unit]); d_tex_[unit] = nullptr; texW_[unit] = texH_[unit] = 0; }
    if (!data || w == 0 || h == 0) return 0;
    size_t bytes = size_t(w) * size_t(h) * sizeof(uint32_t);
    CU_CHECK(cudaMalloc(&d_tex_[unit], bytes));
    cudaMemcpy(d_tex_[unit], data, bytes, cudaMemcpyHostToDevice);
    texW_[unit] = w;
    texH_[unit] = h;
    return 0;
}

int CudaRasterizer::setFragmentProgram(const uint32_t* packedInsns, uint32_t insnCount,
                                       const float* constants, uint32_t constCount) {
    if (d_fpInsns_) { cudaFree(d_fpInsns_); d_fpInsns_ = nullptr; }
    if (d_fpConsts_) { cudaFree(d_fpConsts_); d_fpConsts_ = nullptr; }
    fpInsnCount_ = 0;
    fpConstCount_ = 0;
    if (!packedInsns || insnCount == 0) return 0;

    size_t iBytes = size_t(insnCount) * 6 * sizeof(uint32_t);
    CU_CHECK(cudaMalloc(&d_fpInsns_, iBytes));
    cudaMemcpy(d_fpInsns_, packedInsns, iBytes, cudaMemcpyHostToDevice);
    fpInsnCount_ = insnCount;

    if (constants && constCount > 0) {
        size_t cBytes = size_t(constCount) * 4 * sizeof(float);
        CU_CHECK(cudaMalloc(&d_fpConsts_, cBytes));
        cudaMemcpy(d_fpConsts_, constants, cBytes, cudaMemcpyHostToDevice);
        fpConstCount_ = constCount;
    }
    return 0;
}

void CudaRasterizer::clear(uint32_t rgba) {
    if (!fb_.d_color) return;
    dim3 bs(16, 16);
    dim3 gs((fb_.width + bs.x - 1) / bs.x,
            (fb_.height + bs.y - 1) / bs.y);
    k_clear<<<gs, bs>>>(fb_.d_color, fb_.width, fb_.height, rgba);
    cudaDeviceSynchronize();
    stats.clears++;
}

void CudaRasterizer::clearDepth(float value) {
    if (!fb_.d_depth) return;
    dim3 bs(16, 16);
    dim3 gs((fb_.width + bs.x - 1) / bs.x,
            (fb_.height + bs.y - 1) / bs.y);
    k_clearDepth<<<gs, bs>>>(fb_.d_depth, fb_.width, fb_.height, value);
    cudaDeviceSynchronize();
}

void CudaRasterizer::clearStencil(uint8_t value) {
    if (!fb_.d_stencil) return;
    size_t pixels = size_t(fb_.width) * size_t(fb_.height);
    cudaMemset(fb_.d_stencil, value, pixels * sizeof(uint8_t));
}

uint32_t CudaRasterizer::drawIndexed(const RasterVertex* verts,
                                     uint32_t vertexCount,
                                     const void* indices,
                                     uint32_t indexCount,
                                     bool indexIs32) {
    if (!verts || !indices || indexCount < 3) return 0;
    std::vector<RasterVertex> expanded;
    expanded.reserve(indexCount);
    for (uint32_t i = 0; i < indexCount; ++i) {
        uint32_t idx = indexIs32
            ? static_cast<const uint32_t*>(indices)[i]
            : static_cast<const uint16_t*>(indices)[i];
        if (idx >= vertexCount) return 0;
        expanded.push_back(verts[idx]);
    }
    return drawTriangles(expanded.data(), (uint32_t)expanded.size());
}

uint32_t CudaRasterizer::drawTriangles(const RasterVertex* verts,
                                       uint32_t count) {
    if (!fb_.d_color || count < 3) return 0;
    uint32_t tris = count / 3;

    // Transform to pixel/NDC-z space on host. For small vertex counts
    // this is well under GPU dispatch overhead; a dedicated transform
    // kernel is a later optimization once we're pushing 100k+ verts.
    std::vector<RasterVertex> transformed;
    const RasterVertex* d_src = verts;
    if (useMVP_) {
        transformed.resize(count);
        float vpX = (vpW_ > 0) ? vpX_ : 0.0f;
        float vpY = (vpH_ > 0) ? vpY_ : 0.0f;
        float vpW = (vpW_ > 0) ? vpW_ : (float)fb_.width;
        float vpH = (vpH_ > 0) ? vpH_ : (float)fb_.height;
        for (uint32_t i = 0; i < count; ++i) {
            const auto& v = verts[i];
            // clip = M * (x,y,z,1) using column-major convention.
            float cx = mvp_.m[0][0]*v.x + mvp_.m[1][0]*v.y + mvp_.m[2][0]*v.z + mvp_.m[3][0];
            float cy = mvp_.m[0][1]*v.x + mvp_.m[1][1]*v.y + mvp_.m[2][1]*v.z + mvp_.m[3][1];
            float cz = mvp_.m[0][2]*v.x + mvp_.m[1][2]*v.y + mvp_.m[2][2]*v.z + mvp_.m[3][2];
            float cw = mvp_.m[0][3]*v.x + mvp_.m[1][3]*v.y + mvp_.m[2][3]*v.z + mvp_.m[3][3];
            if (cw == 0.0f) cw = 1e-30f;
            float nx = cx / cw, ny = cy / cw, nz = cz / cw;
            // NDC [-1,1] -> viewport pixels. Y flip to screen-down.
            transformed[i] = v;
            transformed[i].x = vpX + (nx * 0.5f + 0.5f) * vpW;
            transformed[i].y = vpY + (1.0f - (ny * 0.5f + 0.5f)) * vpH;
            transformed[i].z = nz * 0.5f + 0.5f;  // NDC z [-1,1] -> [0,1]
        }
        d_src = transformed.data();
    }

    size_t bytes = size_t(tris) * 3 * sizeof(RasterVertex);

    // Back-face cull: compute screen-space signed area; drop triangles
    // whose facing matches cullMode_ given frontFace_. Runs on host
    // after transform — for high triangle counts a GPU pass would be
    // faster but this keeps the code self-contained.
    std::vector<RasterVertex> culled;
    if (cullMode_ != CullMode::None) {
        culled.reserve(count);
        for (uint32_t t = 0; t < tris; ++t) {
            const auto& a = d_src[t*3 + 0];
            const auto& b = d_src[t*3 + 1];
            const auto& c = d_src[t*3 + 2];
            float area = (c.x - a.x) * (b.y - a.y)
                       - (c.y - a.y) * (b.x - a.x);
            // area > 0 = CCW in our kernel convention; area < 0 = CW.
            bool isFront = (frontFace_ == FrontFace::CCW) ? (area > 0)
                                                          : (area < 0);
            bool drop = false;
            if (cullMode_ == CullMode::Front) drop = isFront;
            else if (cullMode_ == CullMode::Back) drop = !isFront;
            else if (cullMode_ == CullMode::FrontAndBack) drop = true;
            if (drop) { stats.triangleSkipped++; continue; }
            culled.push_back(a);
            culled.push_back(b);
            culled.push_back(c);
        }
        d_src = culled.data();
        tris = (uint32_t)(culled.size() / 3);
        bytes = culled.size() * sizeof(RasterVertex);
        if (tris == 0) return 0;
    }

    RasterVertex* d_v = nullptr;
    if (cudaMalloc(&d_v, bytes) != cudaSuccess) return 0;
    cudaMemcpy(d_v, d_src, bytes, cudaMemcpyHostToDevice);

    dim3 bs(16, 16);
    dim3 gs((fb_.width + bs.x - 1) / bs.x,
            (fb_.height + bs.y - 1) / bs.y);
    TexBank tb;
    for (int i = 0; i < 4; ++i) {
        tb.tex[i] = d_tex_[i]; tb.w[i] = texW_[i]; tb.h[i] = texH_[i];
        tb.wrapS[i] = wrapS_[i]; tb.wrapT[i] = wrapT_[i];
        tb.magFilter[i] = magFilter_[i];
    }
    k_rasterTriangles<<<gs, bs>>>(fb_.d_color, fb_.d_depth, fb_.d_stencil,
                                  fb_.width, fb_.height,
                                  d_v, tris,
                                  blendEnable_ ? 1 : 0,
                                  depthTest_ ? 1 : 0,
                                  depthWrite_ ? 1 : 0,
                                  uint32_t(depthFunc_),
                                  tb,
                                  scX_, scY_, scW_, scH_,
                                  alphaTestEnable_ ? 1 : 0,
                                  uint32_t(alphaRef_),
                                  uint32_t(alphaFunc_),
                                  colorMask_,
                                  polyOffsetFactor_, polyOffsetUnits_,
                                  depthClip_ ? 1 : 0,
                                  stencilTest_ ? 1 : 0,
                                  uint32_t(stencilFunc_),
                                  uint32_t(stencilRef_),
                                  uint32_t(stencilMask_),
                                  uint32_t(stencilWriteMask_),
                                  uint32_t(stencilSFail_),
                                  uint32_t(stencilZFail_),
                                  uint32_t(stencilZPass_),
                                  uint32_t(bfSrcRGB_), uint32_t(bfDstRGB_),
                                  uint32_t(bfSrcA_),   uint32_t(bfDstA_),
                                  uint32_t(beRGB_),    uint32_t(beA_),
                                  blendConstR_, blendConstG_,
                                  blendConstB_, blendConstA_,
                                  d_fpInsns_, fpInsnCount_,
                                  d_fpConsts_, fpConstCount_,
                                  d_colorB_, d_colorC_, d_colorD_);
    cudaDeviceSynchronize();
    cudaFree(d_v);
    stats.triangles += tris;
    return tris;
}

void CudaRasterizer::readback(uint32_t* out) const {
    if (!fb_.d_color || !out) return;
    size_t bytes = size_t(fb_.width) * size_t(fb_.height) * sizeof(uint32_t);
    cudaMemcpy(out, fb_.d_color, bytes, cudaMemcpyDeviceToHost);
}

void CudaRasterizer::readbackDepth(float* out) const {
    if (!fb_.d_depth || !out) return;
    size_t bytes = size_t(fb_.width) * size_t(fb_.height) * sizeof(float);
    cudaMemcpy(out, fb_.d_depth, bytes, cudaMemcpyDeviceToHost);
}

void CudaRasterizer::readbackStencil(uint8_t* out) const {
    if (!fb_.d_stencil || !out) return;
    size_t bytes = size_t(fb_.width) * size_t(fb_.height) * sizeof(uint8_t);
    cudaMemcpy(out, fb_.d_stencil, bytes, cudaMemcpyDeviceToHost);
}

// Helper: apply current MVP + viewport to a single vertex. Returns a
// copy in screen/NDC-z space when useMVP_ is set; otherwise passthrough.
static inline RasterVertex xformOne(const RasterVertex& v,
                                    bool useMVP, const RasterMat4& mvp,
                                    float vpX, float vpY,
                                    float vpW, float vpH) {
    if (!useMVP) return v;
    float cx = mvp.m[0][0]*v.x + mvp.m[1][0]*v.y + mvp.m[2][0]*v.z + mvp.m[3][0];
    float cy = mvp.m[0][1]*v.x + mvp.m[1][1]*v.y + mvp.m[2][1]*v.z + mvp.m[3][1];
    float cz = mvp.m[0][2]*v.x + mvp.m[1][2]*v.y + mvp.m[2][2]*v.z + mvp.m[3][2];
    float cw = mvp.m[0][3]*v.x + mvp.m[1][3]*v.y + mvp.m[2][3]*v.z + mvp.m[3][3];
    if (cw == 0.0f) cw = 1e-30f;
    float nx = cx / cw, ny = cy / cw, nz = cz / cw;
    RasterVertex o = v;
    o.x = vpX + (nx * 0.5f + 0.5f) * vpW;
    o.y = vpY + (1.0f - (ny * 0.5f + 0.5f)) * vpH;
    o.z = nz * 0.5f + 0.5f;
    return o;
}

uint32_t CudaRasterizer::drawLines(const RasterVertex* verts, uint32_t count) {
    if (!fb_.d_color || count < 2 || !verts) return 0;
    uint32_t segs = count / 2;
    if (segs == 0) return 0;

    float vpX = (vpW_ > 0) ? vpX_ : 0.0f;
    float vpY = (vpH_ > 0) ? vpY_ : 0.0f;
    float vpW = (vpW_ > 0) ? vpW_ : (float)fb_.width;
    float vpH = (vpH_ > 0) ? vpH_ : (float)fb_.height;

    std::vector<RasterVertex> xf(segs * 2);
    for (uint32_t i = 0; i < segs * 2; ++i)
        xf[i] = xformOne(verts[i], useMVP_, mvp_, vpX, vpY, vpW, vpH);

    size_t bytes = xf.size() * sizeof(RasterVertex);
    RasterVertex* d_v = nullptr;
    if (cudaMalloc(&d_v, bytes) != cudaSuccess) return 0;
    cudaMemcpy(d_v, xf.data(), bytes, cudaMemcpyHostToDevice);

    dim3 bs(16, 16);
    dim3 gs((fb_.width + bs.x - 1) / bs.x,
            (fb_.height + bs.y - 1) / bs.y);
    k_rasterLines<<<gs, bs>>>(fb_.d_color, fb_.d_depth,
                              fb_.width, fb_.height,
                              d_v, segs,
                              blendEnable_ ? 1 : 0,
                              depthTest_ ? 1 : 0,
                              depthWrite_ ? 1 : 0,
                              uint32_t(depthFunc_),
                              scX_, scY_, scW_, scH_,
                              alphaTestEnable_ ? 1 : 0,
                              uint32_t(alphaRef_),
                              depthClip_ ? 1 : 0,
                              uint32_t(bfSrcRGB_), uint32_t(bfDstRGB_),
                              uint32_t(bfSrcA_),   uint32_t(bfDstA_),
                              uint32_t(beRGB_),    uint32_t(beA_),
                              blendConstR_, blendConstG_,
                              blendConstB_, blendConstA_);
    cudaDeviceSynchronize();
    cudaFree(d_v);
    return segs;
}

uint32_t CudaRasterizer::drawPoints(const RasterVertex* verts, uint32_t count) {
    if (!fb_.d_color || count == 0 || !verts) return 0;

    float vpX = (vpW_ > 0) ? vpX_ : 0.0f;
    float vpY = (vpH_ > 0) ? vpY_ : 0.0f;
    float vpW = (vpW_ > 0) ? vpW_ : (float)fb_.width;
    float vpH = (vpH_ > 0) ? vpH_ : (float)fb_.height;

    std::vector<RasterVertex> xf(count);
    for (uint32_t i = 0; i < count; ++i)
        xf[i] = xformOne(verts[i], useMVP_, mvp_, vpX, vpY, vpW, vpH);

    size_t bytes = xf.size() * sizeof(RasterVertex);
    RasterVertex* d_v = nullptr;
    if (cudaMalloc(&d_v, bytes) != cudaSuccess) return 0;
    cudaMemcpy(d_v, xf.data(), bytes, cudaMemcpyHostToDevice);

    dim3 bs(256);
    dim3 gs((count + bs.x - 1) / bs.x);
    k_rasterPoints<<<gs, bs>>>(fb_.d_color, fb_.d_depth,
                               fb_.width, fb_.height,
                               d_v, count,
                               blendEnable_ ? 1 : 0,
                               depthTest_ ? 1 : 0,
                               depthWrite_ ? 1 : 0,
                               uint32_t(depthFunc_),
                               scX_, scY_, scW_, scH_,
                               alphaTestEnable_ ? 1 : 0,
                               uint32_t(alphaRef_),
                               depthClip_ ? 1 : 0,
                               uint32_t(bfSrcRGB_), uint32_t(bfDstRGB_),
                               uint32_t(bfSrcA_),   uint32_t(bfDstA_),
                               uint32_t(beRGB_),    uint32_t(beA_),
                               blendConstR_, blendConstG_,
                               blendConstB_, blendConstA_);
    cudaDeviceSynchronize();
    cudaFree(d_v);
    return count;
}

bool CudaRasterizer::savePPM(const char* path) const {
    if (!fb_.d_color) return false;
    std::vector<uint32_t> tmp(size_t(fb_.width) * size_t(fb_.height));
    readback(tmp.data());
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << fb_.width << " " << fb_.height << "\n255\n";
    for (uint32_t px : tmp) {
        uint8_t rgb[3] = { uint8_t(px >> 16), uint8_t(px >> 8), uint8_t(px) };
        f.write(reinterpret_cast<const char*>(rgb), 3);
    }
    return (bool)f;
}

} // namespace rsx
