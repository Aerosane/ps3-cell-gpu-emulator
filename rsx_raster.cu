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

__device__ __forceinline__ uint32_t texFetch(const uint32_t* tex,
                                             uint32_t tw, uint32_t th,
                                             int tx, int ty) {
    // wrap mode: repeat
    tx = tx % (int)tw; if (tx < 0) tx += tw;
    ty = ty % (int)th; if (ty < 0) ty += th;
    return tex[ty * tw + tx];
}

__device__ __forceinline__ void sampleTex(const uint32_t* tex,
                                          uint32_t tw, uint32_t th,
                                          float u, float v, int bilinear,
                                          float& r, float& g, float& b, float& a) {
    float fx = u * (float)tw - 0.5f;
    float fy = v * (float)th - 0.5f;
    int ix = (int)floorf(fx);
    int iy = (int)floorf(fy);
    if (!bilinear) {
        uint32_t c = texFetch(tex, tw, th, ix, iy);
        r = ((c >> 16) & 0xFF) / 255.0f;
        g = ((c >>  8) & 0xFF) / 255.0f;
        b = ((c >>  0) & 0xFF) / 255.0f;
        a = ((c >> 24) & 0xFF) / 255.0f;
        return;
    }
    float sx = fx - ix, sy = fy - iy;
    uint32_t c00 = texFetch(tex, tw, th, ix,   iy  );
    uint32_t c10 = texFetch(tex, tw, th, ix+1, iy  );
    uint32_t c01 = texFetch(tex, tw, th, ix,   iy+1);
    uint32_t c11 = texFetch(tex, tw, th, ix+1, iy+1);
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
                                  const uint32_t* __restrict__ tex,
                                  uint32_t texW, uint32_t texH,
                                  int texBilinear,
                                  int scX, int scY, uint32_t scW, uint32_t scH,
                                  int alphaTest, uint32_t alphaRef,
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
                                  float    ccR, float ccG, float ccB, float ccA) {
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

        if (tex) {
            float u = w0 * v0.u + w1 * v1.u + w2 * v2.u;
            float vv = w0 * v0.v + w1 * v1.v + w2 * v2.v;
            float tr, tg, tb, ta;
            sampleTex(tex, texW, texH, u, vv, texBilinear, tr, tg, tb, ta);
            // Use texture color directly — real FP controls combination;
            // modulating by per-face vertex color tints/kills channels.
            r = tr; g = tg; b = tb; a = ta;
        }

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
        dstPx = (sat(a) << 24) | (sat(r) << 16) | (sat(g) << 8) | sat(b);
        if (depthWrite) dstZ = z;
    }

    dst[base] = dstPx;
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
    if (d_tex_) { cudaFree(d_tex_); d_tex_ = nullptr; }
    texW_ = texH_ = 0;
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

int CudaRasterizer::setTexture2D(const uint32_t* data, uint32_t w, uint32_t h) {
    if (d_tex_) { cudaFree(d_tex_); d_tex_ = nullptr; texW_ = texH_ = 0; }
    if (!data || w == 0 || h == 0) return 0;
    size_t bytes = size_t(w) * size_t(h) * sizeof(uint32_t);
    CU_CHECK(cudaMalloc(&d_tex_, bytes));
    cudaMemcpy(d_tex_, data, bytes, cudaMemcpyHostToDevice);
    texW_ = w;
    texH_ = h;
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
    k_rasterTriangles<<<gs, bs>>>(fb_.d_color, fb_.d_depth, fb_.d_stencil,
                                  fb_.width, fb_.height,
                                  d_v, tris,
                                  blendEnable_ ? 1 : 0,
                                  depthTest_ ? 1 : 0,
                                  depthWrite_ ? 1 : 0,
                                  uint32_t(depthFunc_),
                                  d_tex_, texW_, texH_,
                                  texBilinear_ ? 1 : 0,
                                  scX_, scY_, scW_, scH_,
                                  alphaTestEnable_ ? 1 : 0,
                                  uint32_t(alphaRef_),
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
                                  blendConstB_, blendConstA_);
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
