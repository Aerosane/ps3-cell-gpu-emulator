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
    float x, y, z;     // Model/clip-space depending on pipeline state.
    float r, g, b, a;  // Linear RGBA (modulated with texture when bound).
    float u, v;        // Texture coords, [0,1] with current wrap mode.
};

// Column-major 4x4 matrix, mirroring GL/RSX convention:
// M[col][row]. m[0] is the first column, so m[0][3] = element row3col0.
struct RasterMat4 {
    float m[4][4];
    static RasterMat4 identity() {
        RasterMat4 r{};
        for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0f;
        return r;
    }
};

struct RasterFramebuffer {
    uint32_t width{0};
    uint32_t height{0};
    uint32_t* d_color{nullptr};   // device, width*height RGBA8 little-endian
    float*    d_depth{nullptr};   // device, width*height float32 depth
    uint8_t*  d_stencil{nullptr}; // device, width*height uint8 stencil
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

enum class CullMode : uint32_t { None = 0, Front = 1, Back = 2, FrontAndBack = 3 };
enum class FrontFace : uint32_t { CCW = 0, CW = 1 };

// Stencil compare. Matches RSX NV4097_SET_STENCIL_FUNC semantics.
enum class StencilFunc : uint32_t {
    Never    = 0,
    Less     = 1,
    Equal    = 2,
    LEqual   = 3,
    Greater  = 4,
    NotEqual = 5,
    GEqual   = 6,
    Always   = 7,
};

// Stencil op on test result. Matches RSX NV4097_SET_STENCIL_OP semantics.
enum class StencilOp : uint32_t {
    Keep     = 0,
    Zero     = 1,
    Replace  = 2,
    IncrSat  = 3,  // clamped to 255
    DecrSat  = 4,  // clamped to 0
    Invert   = 5,
    IncrWrap = 6,  // wraps 255 -> 0
    DecrWrap = 7,  // wraps 0 -> 255
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

    // Clear stencil target.
    void clearStencil(uint8_t value = 0);

    // Stencil test configuration. Mirrors RSX SET_STENCIL_TEST_ENABLE /
    // SET_STENCIL_FUNC / SET_STENCIL_OP / SET_STENCIL_MASK.
    //
    // The compare is: (ref & mask) <func> (stencilBuf & mask).
    // On stencil-fail: sFail op. On stencil-pass but depth-fail: zFail.
    // On both pass: zPass. Writes are masked with writeMask.
    void setStencilTest(bool enable) { stencilTest_ = enable; }
    void setStencilFunc(StencilFunc f, uint8_t ref, uint8_t mask = 0xFF) {
        stencilFunc_ = f; stencilRef_ = ref; stencilMask_ = mask;
    }
    void setStencilOp(StencilOp sFail, StencilOp zFail, StencilOp zPass) {
        stencilSFail_ = sFail; stencilZFail_ = zFail; stencilZPass_ = zPass;
    }
    void setStencilWriteMask(uint8_t mask) { stencilWriteMask_ = mask; }

    // Enable/disable alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA).
    void setBlend(bool enable) { blendEnable_ = enable; }

    // Depth test configuration. Mirrors RSX DEPTH_TEST_ENABLE /
    // DEPTH_MASK / DEPTH_FUNC.
    void setDepthTest(bool enable) { depthTest_ = enable; }
    void setDepthWrite(bool enable) { depthWrite_ = enable; }
    void setDepthFunc(DepthFunc f) { depthFunc_ = f; }

    // MVP matrix: applied as clip = M * vertex, then perspective divide
    // (x,y,z /= w), then viewport mapping to pixel coords.
    // Leave at identity to pass screen-space vertices straight through.
    void setMVP(const RasterMat4& m) { mvp_ = m; useMVP_ = true; }
    void clearMVP() { useMVP_ = false; }

    // Viewport in pixels. Used only when MVP transform is active.
    void setViewport(float x, float y, float w, float h) {
        vpX_ = x; vpY_ = y; vpW_ = w; vpH_ = h;
    }

    // Texture binding. Data is RGBA8 (0xAARRGGBB little-endian),
    // row-major, no swizzle. Bilinear sampled; UVs wrap with mod 1.
    // Call with data=nullptr to unbind.
    int  setTexture2D(const uint32_t* data, uint32_t w, uint32_t h);
    void setTextureFilter(bool bilinear) { texBilinear_ = bilinear; }

    // Culling: matches RSX NV4097_SET_CULL_FACE_ENABLE + SET_CULL_FACE +
    // SET_FRONT_FACE. Default: no culling.
    void setCullMode(CullMode m) { cullMode_ = m; }
    void setFrontFace(FrontFace f) { frontFace_ = f; }

    // Alpha test: fragments with a*255 < ref are rejected when enabled.
    // Mirrors RSX NV4097_SET_ALPHA_TEST_ENABLE + SET_ALPHA_FUNC/REF.
    // Only Greater (>) is implemented today since it's the common case;
    // other funcs can be added as games need them.
    void setAlphaTest(bool enable, uint8_t ref = 0) {
        alphaTestEnable_ = enable; alphaRef_ = ref;
    }

    // Indexed draw — vertex buffer addressed by an index array.
    // Index type uint16 or uint32 selected via indexIs32.
    uint32_t drawIndexed(const RasterVertex* verts, uint32_t vertexCount,
                         const void* indices, uint32_t indexCount,
                         bool indexIs32);

    // Scissor rect in pixel coordinates. Set w=0 or h=0 to disable.
    // Matches RSX SET_SCISSOR_HORIZONTAL / SET_SCISSOR_VERTICAL.
    void setScissor(int32_t x, int32_t y, uint32_t w, uint32_t h) {
        scX_ = x; scY_ = y; scW_ = w; scH_ = h;
    }
    void disableScissor() { scW_ = 0; scH_ = 0; }

    // Rasterize a triangle list. `verts` must have count%3 == 0.
    // Returns number of triangles that passed degeneracy test.
    uint32_t drawTriangles(const RasterVertex* verts, uint32_t count);

    // Copy color buffer to host (RGBA8, packed 0xAARRGGBB little-endian).
    // `out` must have width*height uint32_t slots.
    void readback(uint32_t* out) const;

    // Copy depth buffer to host (float32 per pixel).
    void readbackDepth(float* out) const;

    // Copy stencil buffer to host (uint8 per pixel).
    void readbackStencil(uint8_t* out) const;

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
    bool useMVP_{false};
    RasterMat4 mvp_{RasterMat4::identity()};
    float vpX_{0}, vpY_{0}, vpW_{0}, vpH_{0};

    // Texture state
    uint32_t* d_tex_{nullptr};
    uint32_t  texW_{0}, texH_{0};
    bool      texBilinear_{true};
    CullMode  cullMode_{CullMode::None};
    FrontFace frontFace_{FrontFace::CCW};
    int32_t   scX_{0}, scY_{0};
    uint32_t  scW_{0}, scH_{0};
    bool      alphaTestEnable_{false};
    uint8_t   alphaRef_{0};

    bool      stencilTest_{false};
    StencilFunc stencilFunc_{StencilFunc::Always};
    uint8_t   stencilRef_{0};
    uint8_t   stencilMask_{0xFF};
    uint8_t   stencilWriteMask_{0xFF};
    StencilOp stencilSFail_{StencilOp::Keep};
    StencilOp stencilZFail_{StencilOp::Keep};
    StencilOp stencilZPass_{StencilOp::Keep};
};

} // namespace rsx
