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
#include <cuda_runtime.h>

namespace rsx {

struct RasterVertex {
    float x, y, z;     // Screen-space position after perspective divide.
    float w;           // Clip-space W for perspective-correct interpolation.
    float r, g, b, a;  // COL0: Linear RGBA diffuse color.
    float u, v;        // TEX0: Texture coords (s,t), [0,1] with current wrap mode.
    // Extended VP outputs for per-pixel FP input
    float col1[4];     // COL1: specular color (FP input 2)
    float fog;         // FOGC: fog coordinate (FP input 3)
    float tex0q[2];    // TEX0 r,q components (s,t are u,v above)
    float tex1[4];     // TEX1 (FP input 5) - full s,t,r,q
    float tex2[4];     // TEX2 (FP input 6) - full s,t,r,q
    float tex3[4];     // TEX3 (FP input 7) - full s,t,r,q
    float tex4[4];     // TEX4 (FP input 8)
    float tex5[4];     // TEX5 (FP input 9)
    float tex6[4];     // TEX6 (FP input 10)
    float tex7[4];     // TEX7 (FP input 11)
    float pointSize;   // PSIZ: point size (VP output 6)
    float backCol0[4]; // BFC0: back-face color 0 (FP input 12)
    float backCol1[4]; // BFC1: back-face color 1 (FP input 13)
    float clipDist[6]; // User clip plane distances (VP output 5 BRB0..5)
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

// Blend factors. Mirrors RSX NV4097 SET_BLEND_FUNC_SFACTOR / DFACTOR.
// Numeric values chosen to be small and compact; the RSX hardware
// encoding is mapped externally by the FIFO decoder.
enum class BlendFactor : uint32_t {
    Zero                = 0,
    One                 = 1,
    SrcColor            = 2,
    OneMinusSrcColor    = 3,
    DstColor            = 4,
    OneMinusDstColor    = 5,
    SrcAlpha            = 6,
    OneMinusSrcAlpha    = 7,
    DstAlpha            = 8,
    OneMinusDstAlpha    = 9,
    ConstColor          = 10,
    OneMinusConstColor  = 11,
    ConstAlpha          = 12,
    OneMinusConstAlpha  = 13,
    SrcAlphaSaturate    = 14,
};

// Blend equations. Mirrors RSX NV4097 SET_BLEND_EQUATION.
enum class BlendEquation : uint32_t {
    Add         = 0,
    Subtract    = 1,  // src - dst
    RevSubtract = 2,  // dst - src
    Min         = 3,
    Max         = 4,
};

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

    void setTwoSidedStencil(bool enable) { twoSidedStencil_ = enable; }
    void setTwoSidedColor(bool enable) { twoSidedColor_ = enable; }
    void setBackStencilFunc(StencilFunc f, uint8_t ref, uint8_t mask = 0xFF) {
        backStencilFunc_ = f; backStencilRef_ = ref; backStencilMask_ = mask;
    }
    void setBackStencilOp(StencilOp sFail, StencilOp zFail, StencilOp zPass) {
        backStencilSFail_ = sFail; backStencilZFail_ = zFail; backStencilZPass_ = zPass;
    }
    void setBackStencilWriteMask(uint8_t mask) { backStencilWriteMask_ = mask; }

    // Enable/disable alpha blending. When enabled the blend factors and
    // equation set via setBlendFunc / setBlendEquation apply. Defaults
    // keep the classic SRC_ALPHA, ONE_MINUS_SRC_ALPHA over-blend so
    // callers that only flip the enable bit still get sensible output.
    void setBlend(bool enable) { blendEnable_ = enable; }

    // Full RSX/GL blend setup: separate RGB and alpha factors/equations.
    void setBlendFunc(BlendFactor srcRGB, BlendFactor dstRGB,
                      BlendFactor srcA,   BlendFactor dstA) {
        bfSrcRGB_ = srcRGB; bfDstRGB_ = dstRGB;
        bfSrcA_   = srcA;   bfDstA_   = dstA;
    }
    void setBlendEquation(BlendEquation rgb, BlendEquation a) {
        beRGB_ = rgb; beA_ = a;
    }
    void setBlendColor(float r, float g, float b, float a) {
        blendConstR_ = r; blendConstG_ = g; blendConstB_ = b; blendConstA_ = a;
    }

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
    int  setTexture2D(const uint32_t* data, uint32_t w, uint32_t h,
                      uint32_t unit = 0);
    void setTextureFilter(bool bilinear) { texBilinear_ = bilinear; }
    void setTextureMagFilter(uint32_t unit, uint8_t mag) {
        if (unit < MAX_TEX_UNITS) magFilter_[unit] = mag;
    }
    void setTextureMinFilter(uint32_t unit, uint8_t min) {
        if (unit < MAX_TEX_UNITS) minFilter_[unit] = min;
    }
    void setTextureLodBias(uint32_t unit, float bias) {
        if (unit < MAX_TEX_UNITS) texLodBias_[unit] = bias;
    }
    void setTextureWrap(uint32_t unit, uint8_t wrapS, uint8_t wrapT) {
        if (unit < MAX_TEX_UNITS) { wrapS_[unit] = wrapS; wrapT_[unit] = wrapT; }
    }
    void setTextureBorderColor(uint32_t unit, uint32_t color) {
        if (unit < MAX_TEX_UNITS) borderColor_[unit] = color;
    }
    void setTextureDimension(uint32_t unit, uint8_t dim) {
        if (unit < MAX_TEX_UNITS) texDimension_[unit] = dim;
    }
    void setTextureDepth(uint32_t unit, uint32_t d) {
        if (unit < MAX_TEX_UNITS) texDepth_[unit] = (d > 0) ? d : 1;
    }
    void setTextureWrapR(uint32_t unit, uint8_t wrapR) {
        if (unit < MAX_TEX_UNITS) wrapR_[unit] = wrapR;
    }
    void setTextureRemap(uint32_t unit, uint16_t remap) {
        if (unit < MAX_TEX_UNITS) texRemap_[unit] = remap;
    }

    // Culling: matches RSX NV4097_SET_CULL_FACE_ENABLE + SET_CULL_FACE +
    // SET_FRONT_FACE. Default: no culling.
    void setCullMode(CullMode m) { cullMode_ = m; }
    void setFrontFace(FrontFace f) { frontFace_ = f; }

    // Alpha test: fragments failing the compare function are rejected.
    // Mirrors RSX NV4097_SET_ALPHA_TEST_ENABLE + SET_ALPHA_FUNC/REF.
    // Supports all 8 GL compare functions (NEVER=0x200 .. ALWAYS=0x207).
    void setAlphaTest(bool enable, uint8_t ref = 0) {
        alphaTestEnable_ = enable; alphaRef_ = ref;
        if (enable && alphaFunc_ == 0x0207) alphaFunc_ = 0x0204; // default to GREATER
    }
    void setAlphaFunc(uint32_t func) { alphaFunc_ = func; }

    // Color mask: per-channel write enable. RSX format:
    // bit 0x01000000=R, 0x00010000=G, 0x00000100=B, 0x00000001=A
    void setColorMask(uint32_t mask) { colorMask_ = mask; }

    // Polygon offset (depth bias): prevents z-fighting for decals/shadows.
    void setPolygonOffset(bool enable, float factor = 0.0f, float units = 0.0f) {
        polyOffsetFill_ = enable;
        polyOffsetFactor_ = enable ? factor : 0.0f;
        polyOffsetUnits_ = enable ? units : 0.0f;
    }

    // Flat shading: use provoking vertex color for the whole triangle
    // instead of barycentric interpolation. RSX: 0x1D00=FLAT, 0x1D01=SMOOTH.
    void setFlatShade(bool flat) { flatShade_ = flat; }

    // Logic op: bitwise operations on framebuffer pixels post-blend.
    // GL enums 0x1500..0x150F (CLEAR, AND, AND_REVERSE, COPY, ..., SET).
    void setLogicOp(bool enable, uint32_t op = 0x1503) {
        logicOpEnable_ = enable; logicOp_ = op;
    }

    // Dither: ordered 4×4 Bayer dither for banding reduction on low-bit
    // surfaces. RSX default is enabled.
    void setDither(bool enable) { ditherEnable_ = enable; }

    // Fog: hardware fog computation from VP fog coordinate.
    // mode: 0x2601=LINEAR, 0x0800=EXP, 0x0801=EXP2
    void setFogParams(uint32_t mode, float param0, float param1) {
        fogMode_ = mode; fogParam0_ = param0; fogParam1_ = param1;
    }

    // Depth bounds test: reject fragments whose depth is outside [min,max].
    void setDepthBoundsTest(bool enable, float minZ = 0.0f, float maxZ = 1.0f) {
        depthBoundsTestEnable_ = enable;
        depthBoundsMin_ = minZ;
        depthBoundsMax_ = maxZ;
    }

    // Near/far depth clip. When enabled (default), fragments whose
    // interpolated depth falls outside [0,1] are rejected. Disable for
    // RSX depth-clamp mode (NV4097_SET_DEPTH_CLAMP_CONTROL).
    void setDepthClip(bool enable) { depthClip_ = enable; }
    void setFpDepthReplace(bool enable) { fpDepthReplace_ = enable; }
    void setShaderWindow(int origin, float height) {
        shaderWindowOrigin_ = origin;
        shaderWindowHeight_ = height;
    }

    // Point / line rendering state
    void setPointSize(float sz) { pointSize_ = (sz > 0.0f) ? sz : 1.0f; }
    void setLineWidth(float w)  { lineWidth_ = (w > 0.0f) ? w : 1.0f; }
    void setPointSpriteEnable(bool e) { pointSpriteEnable_ = e; }

    // Primitive restart: when enabled, an index matching restartIndex
    // breaks the current primitive strip, starting a new one.
    void setPrimitiveRestart(bool enable, uint32_t idx = 0xFFFFFFFF) {
        restartIndexEnable_ = enable; restartIndex_ = idx;
    }

    // User clip planes: 6 planes, each 2 bits in control register.
    // 0=disabled, 2=enable (clip when distance < 0).
    void setClipPlaneControl(uint32_t ctrl) { clipPlaneControl_ = ctrl; }

    // sRGB framebuffer: gamma encode on write (linear→sRGB).
    void setSRGBWrite(bool enable) { sRGBWrite_ = enable; }

    // Fragment program — pre-decoded micro-instructions for GPU execution.
    // Each insn is 8 uint32_t packed: [opcode|masks|texUnit|inputAttr,
    //  dstReg, src0Type|idx|swz, src1Type|idx|swz, src2Type|idx|swz,
    //  flags, pad, pad]. Constants are fp32×4 each.
    // Call with count=0 to disable FP (reverts to fixed texture-replace).
    int  setFragmentProgram(const uint32_t* packedInsns, uint32_t insnCount,
                            const float* constants, uint32_t constCount);

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

    // Rasterize a line list (2 verts per segment). `verts` must have
    // count%2 == 0. Uses parametric-distance coverage (width = 1 px,
    // ends extended by 0.5 px so endpoints render). Respects blend,
    // scissor, depth test, alpha test; does not use culling.
    uint32_t drawLines(const RasterVertex* verts, uint32_t count);

    // Rasterize a point list (1 vert per point). Points are drawn as
    // 1-pixel dots at the rounded vertex position.
    uint32_t drawPoints(const RasterVertex* verts, uint32_t count);

    // Copy color buffer to host (RGBA8, packed 0xAARRGGBB little-endian).
    // `out` must have width*height uint32_t slots.
    void readback(uint32_t* out) const;

    // Copy depth buffer to host (float32 per pixel).
    void readbackDepth(float* out) const;

    // Copy stencil buffer to host (uint8 per pixel).
    void readbackStencil(uint8_t* out) const;

    // ─── MRT (Multiple Render Targets) ──────────────────────────
    // The real NV47 can bind up to 4 color planes (A/B/C/D). We
    // manage these as independent device buffers of the FB's
    // resolution, each matching plane A in layout (RGBA8 uint32_t,
    // width*height pixels). Plane A is always fb_.d_color and is
    // allocated by init(); B/C/D are allocated on first bind.
    //
    // RSX exposes plane target counts via SURFACE_COLOR_TARGET:
    //   SURFACE_TARGET_A    (1 plane, count=1)
    //   SURFACE_TARGET_AB / MRT1  (2 planes)
    //   SURFACE_TARGET_MRT2       (3 planes: ABC)
    //   SURFACE_TARGET_MRT3       (4 planes: ABCD)
    // Draw calls write fragment shader output o[N] to plane N's
    // buffer, if that plane is active. For the current Gouraud
    // modulation path, only plane A receives shaded pixels; B/C/D
    // are allocated and cleared so downstream code (readback,
    // debug capture) can access them, and the device-side FP JIT
    // will start writing them when it lands.
    int  setMRTCount(uint32_t count);   // 1..4
    uint32_t mrtCount() const { return mrtCount_; }

    // Clear plane N (0..3) to AARRGGBB.
    void clearPlane(uint32_t n, uint32_t rgba);

    // Readback plane N.
    void readbackPlane(uint32_t n, uint32_t* out) const;

    uint32_t width()  const { return fb_.width; }
    uint32_t height() const { return fb_.height; }

    // ─── Anti-aliasing (MSAA resolve) ───────────────────────────
    // RSX SURFACE_FORMAT bits 12-15 encode the AA mode:
    //   0 = CENTER_1 (no AA)
    //   4 = DIAGONAL_CENTERED_2 (2× MSAA)
    //  12 = SQUARE_CENTERED_4 / SQUARE_ROTATED_4 (4× MSAA)
    // In our software rasterizer we don't do per-sample coverage,
    // but we track the mode so games that set it don't break. When
    // AA is active, resolveAA() does a box-filter downsample from
    // the internal (super-sampled) buffer to a display-resolution
    // output.  If AA is off, resolveAA() is a plain readback.
    void     setAntialias(uint32_t mode);  // 0, 4, or 12
    uint32_t antialias() const { return aaMode_; }

    // Resolve to host buffer. When AA > 0 and the internal buffer
    // was allocated at super-sampled size, this downsamples.
    // `out` must hold (displayW × displayH) uint32_t pixels.
    void resolveAA(uint32_t* out, uint32_t displayW, uint32_t displayH) const;

    // Save current buffer to binary PPM (P6). Returns false on I/O error.
    bool savePPM(const char* path) const;

    // Async FIFO mode: when enabled, draw kernels are enqueued on a
    // dedicated CUDA stream without blocking the CPU. Multiple draws can
    // be queued before the GPU finishes. Sync happens implicitly on
    // readback(), resolveAA(), or explicit flush().
    void setAsyncFifo(bool enable);
    bool asyncFifo() const { return asyncFifo_; }
    void flush();  // Block until all queued draws complete

    // Diagnostic counters.
    struct Stats {
        uint32_t triangles{0};
        uint32_t triangleSkipped{0};
        uint32_t clears{0};
    } stats;

private:
    RasterFramebuffer fb_{};
    // MRT planes B/C/D (plane A is fb_.d_color). Allocated lazily.
    uint32_t* d_colorB_{nullptr};
    uint32_t* d_colorC_{nullptr};
    uint32_t* d_colorD_{nullptr};
    uint32_t  mrtCount_{1};
    bool blendEnable_{false};
    bool depthTest_{false};
    bool depthWrite_{true};
    DepthFunc depthFunc_{DepthFunc::Less};
    bool useMVP_{false};
    RasterMat4 mvp_{RasterMat4::identity()};
    float vpX_{0}, vpY_{0}, vpW_{0}, vpH_{0};

    // Texture state — up to 4 units (RSX supports 16, we handle the common 4)
    static constexpr uint32_t MAX_TEX_UNITS = 8;
    uint32_t* d_tex_[MAX_TEX_UNITS]{};
    uint32_t  texW_[MAX_TEX_UNITS]{}, texH_[MAX_TEX_UNITS]{};
    uint8_t   wrapS_[MAX_TEX_UNITS]{1,1,1,1,1,1,1,1};  // default REPEAT
    uint8_t   wrapT_[MAX_TEX_UNITS]{1,1,1,1,1,1,1,1};
    uint8_t   wrapR_[MAX_TEX_UNITS]{1,1,1,1,1,1,1,1};  // for 3D textures
    uint8_t   magFilter_[MAX_TEX_UNITS]{1,1,1,1,1,1,1,1};  // default LINEAR (bilinear)
    uint8_t   minFilter_[MAX_TEX_UNITS]{1,1,1,1,1,1,1,1};  // default LINEAR
    uint32_t  borderColor_[MAX_TEX_UNITS]{};
    uint32_t  texDepth_[MAX_TEX_UNITS]{1,1,1,1,1,1,1,1};   // 3D texture depth
    uint8_t   texDimension_[MAX_TEX_UNITS]{2,2,2,2,2,2,2,2}; // default 2D
    uint16_t  texRemap_[MAX_TEX_UNITS]{};  // 0 or 0xAAE4 = identity
    float     texLodBias_[MAX_TEX_UNITS]{};  // per-unit LOD bias from TEXTURE_FILTER
    uint32_t  texMipLevels_[MAX_TEX_UNITS]{1,1,1,1,1,1,1,1}; // mip chain levels
    bool      texBilinear_{true};
    CullMode  cullMode_{CullMode::None};
    FrontFace frontFace_{FrontFace::CCW};
    int32_t   scX_{0}, scY_{0};
    uint32_t  scW_{0}, scH_{0};
    bool      alphaTestEnable_{false};
    uint8_t   alphaRef_{0};
    uint32_t  alphaFunc_{0x0207};  // GL_ALWAYS default
    uint32_t  colorMask_{0x01010101u};  // RGBA all enabled
    float     polyOffsetFactor_{0.0f};
    float     polyOffsetUnits_{0.0f};
    bool      polyOffsetFill_{false};
    bool      depthClip_{true};
    bool      flatShade_{false};
    bool      logicOpEnable_{false};
    uint32_t  logicOp_{0x1503};       // GL_COPY
    bool      ditherEnable_{true};    // RSX default on
    uint32_t  fogMode_{0x2601};       // LINEAR
    float     fogParam0_{0.0f};
    float     fogParam1_{1.0f};
    bool      depthBoundsTestEnable_{false};
    float     depthBoundsMin_{0.0f};
    float     depthBoundsMax_{1.0f};

    // Fragment program state — pre-decoded microcode uploaded to device.
    // When set (fpInsnCount_ > 0), the rasterizer executes the FP per-pixel
    // instead of the fixed vertex×texture modulation.
    uint32_t* d_fpInsns_{nullptr};  // device: packed FPMicroInsn array
    float*    d_fpConsts_{nullptr}; // device: FP constants (4 floats each)
    uint32_t  fpInsnCount_{0};
    uint32_t  fpConstCount_{0};
    bool      fpDepthReplace_{false};
    int       shaderWindowOrigin_{0};   // 0=top, 1=bottom
    float     shaderWindowHeight_{720}; // default 720p

    float     pointSize_{1.0f};
    float     lineWidth_{1.0f};
    bool      pointSpriteEnable_{false};

    bool      restartIndexEnable_{false};
    uint32_t  restartIndex_{0xFFFFFFFF};

    uint32_t  clipPlaneControl_{0};  // 6 planes × 2 bits each

    // Internal: rasterize device-resident vertices (already transformed+culled).
    // tris = triangle count. d_v is device memory, caller must free.
    uint32_t drawTrianglesDevice(RasterVertex* d_v, uint32_t tris,
                                 const RasterVertex* hostVerts, uint32_t hostCount);

    bool      sRGBWrite_{false};   // gamma encode on FB write
    uint32_t  aaMode_{0};          // 0=none, 4=2x, 12=4x

    // Async FIFO: when enabled, draw kernels launch on a dedicated CUDA
    // stream without blocking the CPU. Sync happens only on readback/flip.
    cudaStream_t stream_{nullptr};
    bool asyncFifo_{false};

    bool      stencilTest_{false};
    StencilFunc stencilFunc_{StencilFunc::Always};
    uint8_t   stencilRef_{0};
    uint8_t   stencilMask_{0xFF};
    uint8_t   stencilWriteMask_{0xFF};
    StencilOp stencilSFail_{StencilOp::Keep};
    StencilOp stencilZFail_{StencilOp::Keep};
    StencilOp stencilZPass_{StencilOp::Keep};

    bool      twoSidedStencil_{false};
    bool      twoSidedColor_{false};
    StencilFunc backStencilFunc_{StencilFunc::Always};
    uint8_t   backStencilRef_{0};
    uint8_t   backStencilMask_{0xFF};
    uint8_t   backStencilWriteMask_{0xFF};
    StencilOp backStencilSFail_{StencilOp::Keep};
    StencilOp backStencilZFail_{StencilOp::Keep};
    StencilOp backStencilZPass_{StencilOp::Keep};

    // Blend func / equation state. Defaults match legacy SRC_ALPHA over.
    BlendFactor bfSrcRGB_{BlendFactor::SrcAlpha};
    BlendFactor bfDstRGB_{BlendFactor::OneMinusSrcAlpha};
    BlendFactor bfSrcA_  {BlendFactor::SrcAlpha};
    BlendFactor bfDstA_  {BlendFactor::OneMinusSrcAlpha};
    BlendEquation beRGB_ {BlendEquation::Add};
    BlendEquation beA_   {BlendEquation::Add};
    float blendConstR_{0}, blendConstG_{0}, blendConstB_{0}, blendConstA_{0};

    // ─── GPU scratch buffer pool ────────────────────────────────
    // Persistent device allocations reused across draw calls.
    // Grow-only: reallocated only when a draw exceeds current capacity.
    struct ScratchPool {
        RasterVertex* d_verts{nullptr};
        uint32_t      vertsCap{0};       // capacity in RasterVertex count
        uint32_t*     d_indices{nullptr};
        uint32_t      indicesCap{0};     // capacity in uint32_t count
        RasterVertex* d_xformed{nullptr}; // transformed/gathered verts
        uint32_t      xformedCap{0};
        RasterVertex* d_survived{nullptr}; // post-cull output
        uint32_t      survivedCap{0};
        uint32_t*     d_triCount{nullptr}; // atomic counter (1 element)
        bool          triCountAlloced{false};

        void ensureVerts(uint32_t n) {
            if (n <= vertsCap) return;
            if (d_verts) cudaFree(d_verts);
            vertsCap = n + (n >> 2) + 256; // 25% headroom
            cudaMalloc(&d_verts, vertsCap * sizeof(RasterVertex));
        }
        void ensureIndices(uint32_t n) {
            if (n <= indicesCap) return;
            if (d_indices) cudaFree(d_indices);
            indicesCap = n + (n >> 2) + 256;
            cudaMalloc(&d_indices, indicesCap * sizeof(uint32_t));
        }
        void ensureXformed(uint32_t n) {
            if (n <= xformedCap) return;
            if (d_xformed) cudaFree(d_xformed);
            xformedCap = n + (n >> 2) + 256;
            cudaMalloc(&d_xformed, xformedCap * sizeof(RasterVertex));
        }
        void ensureSurvived(uint32_t n) {
            if (n <= survivedCap) return;
            if (d_survived) cudaFree(d_survived);
            survivedCap = n + (n >> 2) + 256;
            cudaMalloc(&d_survived, survivedCap * sizeof(RasterVertex));
        }
        void ensureTriCount() {
            if (triCountAlloced) return;
            cudaMalloc(&d_triCount, sizeof(uint32_t));
            triCountAlloced = true;
        }
        void freeAll() {
            if (d_verts)    { cudaFree(d_verts);    d_verts = nullptr;    vertsCap = 0; }
            if (d_indices)  { cudaFree(d_indices);  d_indices = nullptr;  indicesCap = 0; }
            if (d_xformed)  { cudaFree(d_xformed);  d_xformed = nullptr;  xformedCap = 0; }
            if (d_survived) { cudaFree(d_survived); d_survived = nullptr; survivedCap = 0; }
            if (d_triCount) { cudaFree(d_triCount); d_triCount = nullptr; triCountAlloced = false; }
        }
    } scratch_;
};

} // namespace rsx
