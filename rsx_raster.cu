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
                                  int scX, int scY, uint32_t scW, uint32_t scH) {
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

        if (depthTest && depth) {
            if (!depthCompare(depthFunc, z, dstZ)) continue;
        }

        float r = w0 * v0.r + w1 * v1.r + w2 * v2.r;
        float g = w0 * v0.g + w1 * v1.g + w2 * v2.g;
        float b = w0 * v0.b + w1 * v1.b + w2 * v2.b;
        float a = w0 * v0.a + w1 * v1.a + w2 * v2.a;

        if (tex) {
            float u = w0 * v0.u + w1 * v1.u + w2 * v2.u;
            float vv = w0 * v0.v + w1 * v1.v + w2 * v2.v;
            float tr, tg, tb, ta;
            sampleTex(tex, texW, texH, u, vv, texBilinear, tr, tg, tb, ta);
            // Modulate vertex color with texture sample.
            r *= tr; g *= tg; b *= tb; a *= ta;
        }

        if (blendEnable) {
            uint32_t dc = dstPx;
            float dr = ((dc >> 16) & 0xFF) / 255.0f;
            float dg = ((dc >>  8) & 0xFF) / 255.0f;
            float db = ((dc >>  0) & 0xFF) / 255.0f;
            r = r * a + dr * (1.0f - a);
            g = g * a + dg * (1.0f - a);
            b = b * a + db * (1.0f - a);
            a = 1.0f;
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
    CU_CHECK(cudaMemset(fb_.d_color, 0, pixels * sizeof(uint32_t)));
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
    if (d_tex_) { cudaFree(d_tex_); d_tex_ = nullptr; }
    texW_ = texH_ = 0;
    fb_.width = fb_.height = 0;
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
    k_rasterTriangles<<<gs, bs>>>(fb_.d_color, fb_.d_depth,
                                  fb_.width, fb_.height,
                                  d_v, tris,
                                  blendEnable_ ? 1 : 0,
                                  depthTest_ ? 1 : 0,
                                  depthWrite_ ? 1 : 0,
                                  uint32_t(depthFunc_),
                                  d_tex_, texW_, texH_,
                                  texBilinear_ ? 1 : 0,
                                  scX_, scY_, scW_, scH_);
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
