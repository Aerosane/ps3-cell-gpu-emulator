#pragma once
// rsx_raster.h — CUDA compute-shader RSX rasterizer.
//
// Fixed-function RSX-semantic triangle rasterizer running directly on
// CUDA cores. No graphics driver involved: we own the pipeline, so
// NV40/RSX quirks (swizzled surfaces, fp16/fixed blend, tile order)
// can be modelled faithfully over time.
//
// Current coverage (minimum-viable):
//   * RGBA8 UNORM color target, linear-tiled (no swizzle yet)
//   * Clear (color)
//   * Triangle list rasterization with top-left fill rule
//   * Per-vertex (x,y,z,r,g,b,a) attributes, barycentric interpolation
//   * Alpha blend: src*a + dst*(1-a) when enabled
//   * No depth / no textures / no shaders yet — these are the next
//     gaps in the rasterizer, but each is additive, not a redesign.
//
// All pointers in this API are CPU pointers. The implementation copies
// to device as needed. This keeps the test harness simple; a zero-copy
// fast path is trivial to add later by accepting device pointers.

#include <cstdint>
#include <cstddef>

namespace rsx {

struct RasterVertex {
    float x, y, z;     // Screen-space for now. Clip/VP transform TBD.
    float r, g, b, a;  // Linear RGBA
};

struct RasterFramebuffer {
    uint32_t width{0};
    uint32_t height{0};
    uint32_t* d_color{nullptr};  // device pointer, width*height RGBA8 little-endian
    float*    d_depth{nullptr};  // device pointer, width*height float32 depth
};

// Depth compare function. Matches RSX/NV4097 SET_DEPTH_FUNC semantics.
enum class DepthFunc : uint32_t {
    Never    = 0,
    Less     = 1,
    Equal    = 2,
    LEqual   = 3,
    Greater  = 4,
    NotEqual = 5,
    GEqual   = 6,
    Always   = 7,
};

class CudaRasterizer {
public:
    CudaRasterizer();
    ~CudaRasterizer();

    int  init(uint32_t width, uint32_t height);
    void shutdown();

    // Clear color target to AARRGGBB (little-endian uint32).
    void clear(uint32_t rgba);

    // Clear depth target.
    void clearDepth(float value = 1.0f);

    // Enable/disable alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA).
    void setBlend(bool enable) { blendEnable_ = enable; }

    // Depth test configuration. Mirrors RSX DEPTH_TEST_ENABLE /
    // DEPTH_MASK / DEPTH_FUNC.
    void setDepthTest(bool enable) { depthTest_ = enable; }
    void setDepthWrite(bool enable) { depthWrite_ = enable; }
    void setDepthFunc(DepthFunc f) { depthFunc_ = f; }

    // Rasterize a triangle list. `verts` must have count%3 == 0.
    // Returns number of triangles that passed degeneracy test.
    uint32_t drawTriangles(const RasterVertex* verts, uint32_t count);

    // Copy color buffer to host (RGBA8, packed 0xAARRGGBB little-endian).
    // `out` must have width*height uint32_t slots.
    void readback(uint32_t* out) const;

    // Copy depth buffer to host (float32 per pixel).
    void readbackDepth(float* out) const;

    uint32_t width()  const { return fb_.width; }
    uint32_t height() const { return fb_.height; }

    // Save current buffer to binary PPM (P6). Returns false on I/O error.
    bool savePPM(const char* path) const;

    // Diagnostic counters.
    struct Stats {
        uint32_t triangles{0};
        uint32_t triangleSkipped{0};
        uint32_t clears{0};
    } stats;

private:
    RasterFramebuffer fb_{};
    bool blendEnable_{false};
    bool depthTest_{false};
    bool depthWrite_{true};
    DepthFunc depthFunc_{DepthFunc::Less};
};

} // namespace rsx
