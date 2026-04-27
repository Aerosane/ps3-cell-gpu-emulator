// rsx_raster_bridge.cpp — Implementation.

#include "rsx_raster_bridge.h"
#include "rsx_defs.h"
#include "rsx_vp_shader.h"
#include "rsx_fp_shader.h"
#include "rsx_texture.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────
// DXT (S3TC) block-compression helpers — decode 4×4 blocks to RGBA8
// ─────────────────────────────────────────────────────────────────
namespace {

// Decode a 565 color to 8-bit RGB
inline void decode565(uint16_t c, uint8_t& r, uint8_t& g, uint8_t& b) {
    r = ((c >> 11) & 0x1F) * 255 / 31;
    g = ((c >>  5) & 0x3F) * 255 / 63;
    b = ((c)       & 0x1F) * 255 / 31;
}

void decodeDXT1Block(const uint8_t* block, uint32_t out[16]) {
    uint16_t c0 = block[0] | (block[1] << 8);
    uint16_t c1 = block[2] | (block[3] << 8);
    uint8_t r0, g0, b0, r1, g1, b1;
    decode565(c0, r0, g0, b0);
    decode565(c1, r1, g1, b1);

    uint8_t palette[4][4];
    palette[0][0] = r0; palette[0][1] = g0; palette[0][2] = b0; palette[0][3] = 255;
    palette[1][0] = r1; palette[1][1] = g1; palette[1][2] = b1; palette[1][3] = 255;
    if (c0 > c1) {
        palette[2][0] = (2*r0+r1)/3; palette[2][1] = (2*g0+g1)/3;
        palette[2][2] = (2*b0+b1)/3; palette[2][3] = 255;
        palette[3][0] = (r0+2*r1)/3; palette[3][1] = (g0+2*g1)/3;
        palette[3][2] = (b0+2*b1)/3; palette[3][3] = 255;
    } else {
        palette[2][0] = (r0+r1)/2; palette[2][1] = (g0+g1)/2;
        palette[2][2] = (b0+b1)/2; palette[2][3] = 255;
        palette[3][0] = 0; palette[3][1] = 0;
        palette[3][2] = 0; palette[3][3] = 0;  // transparent black
    }

    uint32_t indices = block[4] | (block[5]<<8) | (block[6]<<16) | (block[7]<<24);
    for (int i = 0; i < 16; ++i) {
        int idx = (indices >> (i * 2)) & 3;
        out[i] = ((uint32_t)palette[idx][3] << 24) |
                 ((uint32_t)palette[idx][0] << 16) |
                 ((uint32_t)palette[idx][1] <<  8) |
                  (uint32_t)palette[idx][2];
    }
}

void decodeDXT3Block(const uint8_t* block, uint32_t out[16]) {
    // First 8 bytes: explicit alpha (4 bits per texel)
    uint64_t alphaData = 0;
    for (int i = 0; i < 8; ++i) alphaData |= (uint64_t)block[i] << (i * 8);

    // Last 8 bytes: DXT1 color block (ignoring DXT1 alpha)
    decodeDXT1Block(block + 8, out);

    // Override alpha with explicit values
    for (int i = 0; i < 16; ++i) {
        uint8_t a4 = (alphaData >> (i * 4)) & 0xF;
        uint8_t a8 = a4 * 17;  // expand 4-bit to 8-bit
        out[i] = (out[i] & 0x00FFFFFF) | ((uint32_t)a8 << 24);
    }
}

void decodeDXT5Block(const uint8_t* block, uint32_t out[16]) {
    // First 8 bytes: interpolated alpha
    uint8_t a0 = block[0], a1 = block[1];
    uint8_t alphaPalette[8];
    alphaPalette[0] = a0;
    alphaPalette[1] = a1;
    if (a0 > a1) {
        for (int i = 2; i < 8; ++i)
            alphaPalette[i] = ((8-i)*a0 + (i-1)*a1) / 7;
    } else {
        for (int i = 2; i < 6; ++i)
            alphaPalette[i] = ((6-i)*a0 + (i-1)*a1) / 5;
        alphaPalette[6] = 0;
        alphaPalette[7] = 255;
    }

    // 6 bytes of 3-bit alpha indices (48 bits for 16 texels)
    uint64_t aBits = 0;
    for (int i = 0; i < 6; ++i) aBits |= (uint64_t)block[2+i] << (i * 8);

    // Last 8 bytes: DXT1 color block
    decodeDXT1Block(block + 8, out);

    for (int i = 0; i < 16; ++i) {
        int aIdx = (aBits >> (i * 3)) & 7;
        out[i] = (out[i] & 0x00FFFFFF) | ((uint32_t)alphaPalette[aIdx] << 24);
    }
}

// Decode full DXT texture to RGBA8 buffer
void decodeDXTTexture(const uint8_t* src, uint32_t W, uint32_t H,
                      int dxtType, std::vector<uint32_t>& rgba8) {
    rgba8.resize(W * H);
    uint32_t bw = (W + 3) / 4;  // blocks wide
    uint32_t bh = (H + 3) / 4;  // blocks high
    uint32_t blockSize = (dxtType == 1) ? 8 : 16;

    for (uint32_t by = 0; by < bh; ++by) {
        for (uint32_t bx = 0; bx < bw; ++bx) {
            uint32_t blockIdx = by * bw + bx;
            const uint8_t* block = src + blockIdx * blockSize;
            uint32_t pixels[16];
            switch (dxtType) {
            case 1: decodeDXT1Block(block, pixels); break;
            case 3: decodeDXT3Block(block, pixels); break;
            case 5: decodeDXT5Block(block, pixels); break;
            }
            for (int py = 0; py < 4; ++py) {
                for (int px = 0; px < 4; ++px) {
                    uint32_t x = bx * 4 + px;
                    uint32_t y = by * 4 + py;
                    if (x < W && y < H)
                        rgba8[y * W + x] = pixels[py * 4 + px];
                }
            }
        }
    }
}

} // anonymous namespace

namespace rsx {

// ─────────────────────────────────────────────────────────────────
// NV40 vertex-array decoder. Pulls per-vertex attributes from VRAM
// using the slot table in RSXState.vertexArrays[]. Convention used:
//   slot 0 → position (x, y, z [,w])
//   slot 3 → diffuse color (r, g, b, a)
//   slot 8 → texcoord 0 (u, v)
// Unused / disabled slots fall back to defaults (color = opaque white,
// uv = 0). Supports VERTEX_F and VERTEX_UB (common D3DCOLOR-style
// 4x u8 for colors). Unknown types are treated as passthrough zeros
// for that attribute.
// ─────────────────────────────────────────────────────────────────
static inline bool decode_attr(const uint8_t* vram, uint32_t vramSize,
                               const RSXState::VertexArray& va,
                               uint32_t index, int maxChannels,
                               float out[4]) {
    for (int i = 0; i < 4; ++i) out[i] = 0.0f;
    if (!va.enabled) return false;

    uint32_t type   = va.format & 0xF;
    uint32_t size   = (va.format >> 4) & 0xF;
    uint32_t stride = (va.format >> 8) & 0xFF;
    if (stride == 0 || size == 0) return false;

    uint32_t off = va.offset + index * stride;
    if ((uint64_t)off + stride > vramSize) return false;
    const uint8_t* p = vram + off;

    int channels = (int)size;
    if (channels > maxChannels) channels = maxChannels;

    switch (type) {
    case VERTEX_F: {
        for (int i = 0; i < channels; ++i) {
            // PS3 stores vertex data big-endian in guest RAM; we byteswap
            // to decode each 32-bit float correctly on the x86 host.
            uint32_t be;
            std::memcpy(&be, p + i * 4, 4);
            uint32_t le = __builtin_bswap32(be);
            float f;
            std::memcpy(&f, &le, 4);
            out[i] = f;
        }
        return true;
    }
    case VERTEX_UB: {
        // D3DCOLOR byte order: stored BGRA in memory, emit as RGBA.
        // When size == 4 and we're reading a color slot, remap.
        if (size == 4 && maxChannels == 4) {
            out[2] = p[0] / 255.0f;  // B
            out[1] = p[1] / 255.0f;  // G
            out[0] = p[2] / 255.0f;  // R
            out[3] = p[3] ? (p[3] / 255.0f) : 1.0f;  // A (default 1.0 if unset)
        } else {
            for (int i = 0; i < channels; ++i) out[i] = p[i] / 255.0f;
        }
        return true;
    }
    case VERTEX_S1: {
        // Signed normalized 16-bit: [-32768..32767] → [-1.0..1.0]
        for (int i = 0; i < channels; ++i) {
            uint16_t be;
            std::memcpy(&be, p + i * 2, 2);
            uint16_t le = __builtin_bswap16(be);
            int16_t s16;
            std::memcpy(&s16, &le, 2);
            out[i] = s16 / 32767.0f;
        }
        return true;
    }
    case VERTEX_SF: {
        // 16-bit half-float
        for (int i = 0; i < channels; ++i) {
            uint16_t be;
            std::memcpy(&be, p + i * 2, 2);
            uint16_t h = __builtin_bswap16(be);
            // IEEE 754 half → float conversion
            uint32_t sign = (uint32_t)(h >> 15) << 31;
            uint32_t exp5 = (h >> 10) & 0x1F;
            uint32_t man  = h & 0x3FF;
            uint32_t f32;
            if (exp5 == 0) {
                if (man == 0) f32 = sign;
                else {
                    // Subnormal: normalize
                    exp5 = 1;
                    while (!(man & 0x400)) { man <<= 1; exp5--; }
                    man &= 0x3FF;
                    f32 = sign | ((exp5 + 127 - 15) << 23) | (man << 13);
                }
            } else if (exp5 == 31) {
                f32 = sign | 0x7F800000 | (man << 13);
            } else {
                f32 = sign | ((exp5 + 127 - 15) << 23) | (man << 13);
            }
            float val;
            std::memcpy(&val, &f32, 4);
            out[i] = val;
        }
        return true;
    }
    case VERTEX_S32K: {
        // Signed 16-bit integer (not normalized)
        for (int i = 0; i < channels; ++i) {
            uint16_t be;
            std::memcpy(&be, p + i * 2, 2);
            uint16_t le = __builtin_bswap16(be);
            int16_t s16;
            std::memcpy(&s16, &le, 2);
            out[i] = (float)s16;
        }
        return true;
    }
    case VERTEX_CMP: {
        // Compressed 11/11/10 packed into 32 bits (X11Y11Z10 signed)
        uint32_t be;
        std::memcpy(&be, p, 4);
        uint32_t w = __builtin_bswap32(be);
        // X: bits [0:10] signed 11-bit, Y: [11:21] signed 11-bit, Z: [22:31] signed 10-bit
        int32_t ix = (int32_t)(w << 21) >> 21;
        int32_t iy = (int32_t)(w << 10) >> 21;
        int32_t iz = (int32_t)(w)       >> 22;
        out[0] = ix / 1023.0f;
        out[1] = iy / 1023.0f;
        out[2] = iz / 511.0f;
        if (channels > 3) out[3] = 1.0f;
        return true;
    }
    case VERTEX_UB256: {
        // Unsigned byte, not normalized (raw 0-255 as float)
        for (int i = 0; i < channels; ++i) out[i] = (float)p[i];
        return true;
    }
    default:
        return false;
    }
}

static void decode_vertex_stream(const RSXState& s,
                                 const uint8_t* vram, uint32_t vramSize,
                                 uint32_t first, uint32_t count,
                                 std::vector<RasterVertex>& out) {
    out.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = first + i;
        float pos[4]   = { 0, 0, 0, 1 };
        float color[4] = { 1, 1, 1, 1 };
        float uv[4]    = { 0, 0, 0, 0 };

        // VA0 is always position
        decode_attr(vram, vramSize, s.vertexArrays[0], idx, 4, pos);

        // Scan remaining VAs for color (type=UB) and UV (type=F, size=2)
        for (int va = 1; va < 16; ++va) {
            if (!s.vertexArrays[va].enabled) continue;
            uint32_t type = s.vertexArrays[va].format & 0xF;
            uint32_t sz   = (s.vertexArrays[va].format >> 4) & 0xF;
            if (type == VERTEX_UB)
                decode_attr(vram, vramSize, s.vertexArrays[va], idx, 4, color);
            else if (type == VERTEX_F && sz == 2)
                decode_attr(vram, vramSize, s.vertexArrays[va], idx, 4, uv);
        }

        RasterVertex& v = out[i];
        memset(&v, 0, sizeof(v));
        v.x = pos[0]; v.y = pos[1]; v.z = pos[2];
        v.w = 1.0f;  // pre-transformed vertices: W=1 (no perspective)
        v.r = color[0]; v.g = color[1]; v.b = color[2]; v.a = color[3];
        v.u = uv[0];    v.v = uv[1];
    }
}

void RasterBridge::onSurfaceSetup(const RSXState& s) {
    if (!rast_) return;
    // (Re)init if dims changed. CudaRasterizer::init tears down the old
    // framebuffer internally.
    if (rast_->width()  != s.surfaceWidth ||
        rast_->height() != s.surfaceHeight) {
        rast_->init(s.surfaceWidth, s.surfaceHeight);
    }
    counters.surfaceSetups++;
}

void RasterBridge::onViewport(const RSXState& s) {
    if (!rast_) return;
    rast_->setViewport((float)s.viewportX, (float)s.viewportY,
                       (float)s.viewportW, (float)s.viewportH);
}

void RasterBridge::onScissor(const RSXState& s) {
    if (!rast_) return;
    rast_->setScissor((int32_t)s.scissorX, (int32_t)s.scissorY,
                      s.scissorW, s.scissorH);
}

void RasterBridge::onClearSurface(const RSXState& s, uint32_t mask) {
    if (!rast_) return;

    // NV4097 CLEAR_SURFACE: bit0=Z, bit1=S, bit4=R, bit5=G, bit6=B, bit7=A
    if (mask & 0xF0) {
        rast_->clear(s.colorClearValue);
        for (uint32_t p = 1; p < rast_->mrtCount(); ++p)
            rast_->clearPlane(p, s.colorClearValue);
    }
    if (mask & 0x01) rast_->clearDepth(1.0f);
    counters.clears++;
}

void RasterBridge::onBeginEnd(const RSXState&, uint32_t) {
    // Rasterizer has no explicit render-pass begin/end — draws map to
    // kernel launches directly. Here only to match the hook signature.
}

static inline DepthFunc nv_to_depthFunc(uint32_t v) {
    // NV4097 depth func values follow GL enum 0x200..0x207:
    //   NEVER, LESS, EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL, ALWAYS.
    if (v >= 0x200 && v <= 0x207) {
        return static_cast<DepthFunc>(v - 0x200);
    }
    return DepthFunc::Less;
}

static inline StencilFunc nv_to_stencilFunc(uint32_t v) {
    if (v >= 0x200 && v <= 0x207) {
        return static_cast<StencilFunc>(v - 0x200);
    }
    return StencilFunc::Always;
}

static inline StencilOp nv_to_stencilOp(uint32_t v) {
    // NV/GL stencil op enums:
    //   KEEP=0x1E00, REPLACE=0x1E01, INCR=0x1E02, DECR=0x1E03,
    //   ZERO=0, INVERT=0x150A, INCR_WRAP=0x8507, DECR_WRAP=0x8508.
    switch (v) {
    case 0x1E00: return StencilOp::Keep;
    case 0x0000: return StencilOp::Zero;
    case 0x1E01: return StencilOp::Replace;
    case 0x1E02: return StencilOp::IncrSat;
    case 0x1E03: return StencilOp::DecrSat;
    case 0x150A: return StencilOp::Invert;
    case 0x8507: return StencilOp::IncrWrap;
    case 0x8508: return StencilOp::DecrWrap;
    default:     return StencilOp::Keep;
    }
}

static inline BlendFactor nv_to_blendFactor(uint32_t v) {
    // NV/GL blend factor enums — mixed ranges.
    switch (v) {
    case 0x0000: return BlendFactor::Zero;
    case 0x0001: return BlendFactor::One;
    case 0x0300: return BlendFactor::SrcColor;
    case 0x0301: return BlendFactor::OneMinusSrcColor;
    case 0x0302: return BlendFactor::SrcAlpha;
    case 0x0303: return BlendFactor::OneMinusSrcAlpha;
    case 0x0304: return BlendFactor::DstAlpha;
    case 0x0305: return BlendFactor::OneMinusDstAlpha;
    case 0x0306: return BlendFactor::DstColor;
    case 0x0307: return BlendFactor::OneMinusDstColor;
    case 0x0308: return BlendFactor::SrcAlphaSaturate;
    case 0x8001: return BlendFactor::ConstColor;
    case 0x8002: return BlendFactor::OneMinusConstColor;
    case 0x8003: return BlendFactor::ConstAlpha;
    case 0x8004: return BlendFactor::OneMinusConstAlpha;
    default:     return BlendFactor::One;
    }
}

static inline BlendEquation nv_to_blendEquation(uint32_t v) {
    switch (v) {
    case 0x8006: return BlendEquation::Add;
    case 0x8007: return BlendEquation::Min;
    case 0x8008: return BlendEquation::Max;
    case 0x800A: return BlendEquation::Subtract;
    case 0x800B: return BlendEquation::RevSubtract;
    default:     return BlendEquation::Add;
    }
}

static inline CullMode nv_to_cullMode(uint32_t v, bool enabled) {
    if (!enabled) return CullMode::None;
    switch (v) {
    case 0x0404: return CullMode::Front;
    case 0x0405: return CullMode::Back;
    case 0x0408: return CullMode::FrontAndBack;
    default:     return CullMode::Back;
    }
}

void RasterBridge::applyPipelineState(const RSXState& s) {
    if (!rast_) return;

    // Depth
    rast_->setDepthTest(s.depthTestEnable);
    rast_->setDepthFunc(nv_to_depthFunc(s.depthFunc));
    rast_->setDepthWrite(s.depthMask);

    // Blend: SFactor/DFactor/Equation split RGB (low16) from alpha (high16).
    rast_->setBlend(s.blendEnable);
    if (s.blendEnable) {
        BlendFactor srcRGB = nv_to_blendFactor(s.blendSFactor & 0xFFFF);
        BlendFactor srcA   = nv_to_blendFactor((s.blendSFactor >> 16) & 0xFFFF);
        BlendFactor dstRGB = nv_to_blendFactor(s.blendDFactor & 0xFFFF);
        BlendFactor dstA   = nv_to_blendFactor((s.blendDFactor >> 16) & 0xFFFF);
        rast_->setBlendFunc(srcRGB, dstRGB, srcA, dstA);
        BlendEquation eqRGB = nv_to_blendEquation(s.blendEquation & 0xFFFF);
        BlendEquation eqA   = nv_to_blendEquation((s.blendEquation >> 16) & 0xFFFF);
        rast_->setBlendEquation(eqRGB, eqA);

        uint32_t c = s.blendColor;
        rast_->setBlendColor(((c >> 16) & 0xFF) / 255.f,
                             ((c >>  8) & 0xFF) / 255.f,
                             ( c        & 0xFF) / 255.f,
                             ((c >> 24) & 0xFF) / 255.f);
    }

    // Stencil
    rast_->setStencilTest(s.stencilTestEnable);
    if (s.stencilTestEnable) {
        rast_->setStencilFunc(nv_to_stencilFunc(s.stencilFunc),
                              (uint8_t)(s.stencilRef & 0xFF),
                              (uint8_t)(s.stencilFuncMask & 0xFF));
        rast_->setStencilOp(nv_to_stencilOp(s.stencilOpFail),
                            nv_to_stencilOp(s.stencilOpZFail),
                            nv_to_stencilOp(s.stencilOpZPass));
        rast_->setStencilWriteMask((uint8_t)(s.stencilWriteMask & 0xFF));
    }

    // Color mask
    rast_->setColorMask(s.colorMask);

    // Alpha function (full 8 GL compare modes)
    rast_->setAlphaFunc(s.alphaFunc);

    // Polygon offset (depth bias)
    rast_->setPolygonOffset(s.polyOffsetFillEnable, s.polyOffsetFactor, s.polyOffsetBias);

    // Flat shading (provoking vertex) vs smooth interpolation
    rast_->setFlatShade(s.shadeMode == 0x1D00);

    // Logic op (bitwise framebuffer ops)
    rast_->setLogicOp(s.logicOpEnable, s.logicOp);

    // Dither
    rast_->setDither(s.ditherEnable);

    // Two-sided stencil
    rast_->setTwoSidedStencil(s.twoSidedStencilEnable);
    if (s.twoSidedStencilEnable) {
        rast_->setBackStencilFunc(nv_to_stencilFunc(s.backStencilFunc),
                                  (uint8_t)(s.backStencilFuncRef & 0xFF),
                                  (uint8_t)(s.backStencilFuncMask & 0xFF));
        rast_->setBackStencilOp(nv_to_stencilOp(s.backStencilOpFail),
                                nv_to_stencilOp(s.backStencilOpZFail),
                                nv_to_stencilOp(s.backStencilOpZPass));
        rast_->setBackStencilWriteMask((uint8_t)(s.backStencilWriteMask & 0xFF));
    }

    // Fog
    rast_->setFogParams(s.fogMode, s.fogParam0, s.fogParam1);

    // Cull
    rast_->setCullMode(nv_to_cullMode(s.cullFace, s.cullFaceEnable));

    // (Texture binding moved to onDrawArrays to use draw-time state.)

    // MRT: decode SURFACE_COLOR_TARGET → 1..4 active color planes.
    // The RSX encoding is sparse (0, 1, 2, 3, 0x13, 0x17, 0x1F) and
    // maps directly to plane count.
    uint32_t mrt = 1;
    switch (s.surfaceColorTarget) {
    case SURFACE_TARGET_NONE: mrt = 0; break;
    case SURFACE_TARGET_A:    mrt = 1; break;
    case SURFACE_TARGET_B:    mrt = 1; break;   // single plane, B bound to slot 0
    case SURFACE_TARGET_AB:
    case SURFACE_TARGET_MRT1: mrt = 2; break;
    case SURFACE_TARGET_MRT2: mrt = 3; break;
    case SURFACE_TARGET_MRT3: mrt = 4; break;
    default: mrt = 1; break;
    }
    if (mrt == 0) mrt = 1;
    if (rast_->mrtCount() != mrt) rast_->setMRTCount(mrt);
}

void RasterBridge::onDrawArrays(const RSXState& s, uint32_t first, uint32_t count) {
    if (!rast_ || count == 0) return;

    // Translate FIFO-side pipeline state into rasterizer setters before
    // every draw. Games frequently toggle depth/blend/cull mid-frame.
    applyPipelineState(s);

    // Choose the vertex source.
    std::vector<RasterVertex> decoded;
    const RasterVertex* base = nullptr;
    if (vram_ && s.vertexArrays[0].enabled) {
        decode_vertex_stream(s, vram_, vramSize_, first, count, decoded);
        base = decoded.data();
    } else if (pool_) {
        if ((uint64_t)first + count > poolCount_) return;
        base = pool_ + first;
    } else {
        return;
    }

    // If a vertex program has been uploaded, run it per-vertex on the
    // decoded/pool stream. VP input registers map 1:1 to vertex
    // attribute indices: VA[k] → input[k]. Output register 0 is HPOS
    // (clip space), o[1] is COL0, o[7] is TEX0, etc.
    std::vector<RasterVertex> transformed;
    if (s.vpValid) {
        if (base == nullptr) return;
        transformed.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            VPFloat4 inputs[16] = {};
            // Populate VP inputs from each enabled VA → input[va_idx]
            for (int va = 0; va < 16; ++va) {
                if (!s.vertexArrays[va].enabled) continue;
                float attr[4] = {0, 0, 0, 1};
                decode_attr(vram_, vramSize_, s.vertexArrays[va],
                            first + i, 4, attr);
                inputs[va].v[0] = attr[0]; inputs[va].v[1] = attr[1];
                inputs[va].v[2] = attr[2]; inputs[va].v[3] = attr[3];
                // Position VA: ensure w=1 for homogeneous coords when
                // the attribute has fewer than 4 components.
                if (va == 0) {
                    uint32_t sz = (s.vertexArrays[0].format >> 4) & 0xF;
                    if (sz < 4) inputs[0].v[3] = 1.0f;
                }
            }
            // Fallback: if no VAs enabled, use pool vertex data
            if (!vram_) {
                inputs[0].v[0] = base[i].x; inputs[0].v[1] = base[i].y;
                inputs[0].v[2] = base[i].z; inputs[0].v[3] = 1.0f;
            }

            const VPFloat4* consts = reinterpret_cast<const VPFloat4*>(s.vpConstants);
            VPFloat4 outputs[16] = {};

            vp_execute(s.vpData, 512u * 4u, s.vpStart,
                       inputs, consts, outputs);

            RasterVertex& v = transformed[i];
            memset(&v, 0, sizeof(v));
            // VP outputs HPOS in clip/NDC space; perform perspective
            // divide (w) and map NDC [-1,1] → pixel space using the
            // current surface dimensions. Y is flipped for screen-down.
            float ox = outputs[0].v[0];
            float oy = outputs[0].v[1];
            float oz = outputs[0].v[2];
            float ow = outputs[0].v[3];
            if (ow == 0.0f) ow = 1.0f;
            float nx = ox / ow;
            float ny = oy / ow;
            float nz = oz / ow;
            // Use viewport offset/scale if explicitly set by game
            if (s.vpOffsetScaleSet) {
                v.x = s.vpScale[0] * nx + s.vpOffset[0];
                v.y = s.vpScale[1] * ny + s.vpOffset[1];
                v.z = s.vpScale[2] * nz + s.vpOffset[2];
            } else {
                float W = (float)(s.surfaceWidth  ? s.surfaceWidth  : 1280);
                float H = (float)(s.surfaceHeight ? s.surfaceHeight : 720);
                v.x = (nx * 0.5f + 0.5f) * W;
                v.y = (1.0f - (ny * 0.5f + 0.5f)) * H;
                v.z = nz * 0.5f + 0.5f;
            }
            v.w = ow;  // preserve clip-space W for perspective-correct interpolation
            // Color from o[1] (COL0), UV from o[7] (TEX0)
            v.r = outputs[1].v[0]; v.g = outputs[1].v[1];
            v.b = outputs[1].v[2]; v.a = outputs[1].v[3];
            v.u = outputs[7].v[0]; v.v = outputs[7].v[1];
            // Extended VP outputs → FP inputs
            v.col1[0] = outputs[2].v[0]; v.col1[1] = outputs[2].v[1];
            v.col1[2] = outputs[2].v[2]; v.col1[3] = outputs[2].v[3];
            v.fog = outputs[5].v[0];
            v.tex1[0] = outputs[8].v[0]; v.tex1[1] = outputs[8].v[1];
            v.tex2[0] = outputs[9].v[0]; v.tex2[1] = outputs[9].v[1];
            v.tex3[0] = outputs[10].v[0]; v.tex3[1] = outputs[10].v[1];
        }
        base = transformed.data();
    }

    // ── Per-pixel texture binding (units 0-3) ─────────────────
    // Upload enabled texture units to the GPU rasterizer at draw time.
    bool texBoundForDraw = false;
    for (int tu = 0; tu < 4; ++tu) {
        if (!vram_ || !s.textures[tu].enabled ||
            s.textures[tu].width == 0 || s.textures[tu].height == 0) continue;
        const auto& t = s.textures[tu];
        // For unit 0, use cache; units 1-3 always upload (simple for now)
        if (tu == 0) {
            bool stale =
                !cachedTexValid_ ||
                t.offset != cachedTexOff_ || t.width != cachedTexW_ ||
                t.height != cachedTexH_   || t.format != cachedTexFmt_;
            if (!stale) { texBoundForDraw = true; continue; }
        }
        uint8_t fmt = ((t.format >> 8) & 0xFF) & 0x9F;
        uint32_t W = t.width, H = t.height;
        uint32_t bpp;
        switch (fmt) {
        case 0x81: bpp = 1; break;  // B8
        case 0x82: bpp = 2; break;  // A1R5G5B5
        case 0x83: bpp = 2; break;  // A4R4G4B4
        case 0x84: bpp = 2; break;  // R5G6B5
        case 0x85: bpp = 4; break;  // A8R8G8B8
        case 0x86: bpp = 0; break;  // DXT1 (block compressed)
        case 0x87: bpp = 0; break;  // DXT23 (block compressed)
        case 0x88: bpp = 0; break;  // DXT45 (block compressed)
        case 0x8B: bpp = 2; break;  // G8B8
        case 0x94: bpp = 2; break;  // X16 (single 16-bit channel)
        case 0x97: bpp = 2; break;  // R5G5B5A1
        case 0x9E: bpp = 4; break;  // D8R8G8B8
        default:   bpp = 4; break;
        }
        uint64_t need;
        if (bpp == 0) {
            // DXT block compressed: 4×4 blocks
            uint32_t bw = (W + 3) / 4, bh = (H + 3) / 4;
            uint32_t blockSize = (fmt == 0x86) ? 8 : 16;
            need = (uint64_t)bw * bh * blockSize;
        } else {
            need = (uint64_t)W * H * bpp;
        }
        if ((uint64_t)t.offset + need > vramSize_) continue;
        const uint8_t* src = vram_ + t.offset;
        std::vector<uint32_t> rgba8(W * H);
        if (fmt == 0x85) {
            // A8R8G8B8 — big-endian
            const uint32_t* s32 = reinterpret_cast<const uint32_t*>(src);
            for (uint32_t i = 0; i < W * H; ++i) {
                uint32_t px = __builtin_bswap32(s32[i]);
                if ((px >> 24) == 0) px |= 0xFF000000u;
                rgba8[i] = px;
            }
        } else if (fmt == 0x9E) {
            // D8R8G8B8 — like A8R8G8B8 but alpha byte is "don't care"
            const uint32_t* s32 = reinterpret_cast<const uint32_t*>(src);
            for (uint32_t i = 0; i < W * H; ++i) {
                uint32_t px = __builtin_bswap32(s32[i]);
                rgba8[i] = px | 0xFF000000u;
            }
        } else if (fmt == 0x81) {
            // B8 — luminance
            for (uint32_t i = 0; i < W * H; ++i) {
                uint8_t v = src[i];
                rgba8[i] = 0xFF000000u |
                           ((uint32_t)v << 16) |
                           ((uint32_t)v <<  8) | (uint32_t)v;
            }
        } else if (fmt == 0x8B) {
            // G8B8 — two-channel, big-endian
            for (uint32_t i = 0; i < W * H; ++i) {
                uint8_t g = src[i * 2];
                uint8_t b = src[i * 2 + 1];
                rgba8[i] = 0xFF000000u | ((uint32_t)g << 8) | (uint32_t)b;
            }
        } else if (fmt == 0x84) {
            // R5G6B5 — 16-bit color, big-endian
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
            for (uint32_t i = 0; i < W * H; ++i) {
                uint16_t px = __builtin_bswap16(s16[i]);
                uint8_t rr = ((px >> 11) & 0x1F) * 255 / 31;
                uint8_t gg = ((px >>  5) & 0x3F) * 255 / 63;
                uint8_t bb = ((px)       & 0x1F) * 255 / 31;
                rgba8[i] = 0xFF000000u | ((uint32_t)rr << 16) |
                            ((uint32_t)gg << 8) | (uint32_t)bb;
            }
        } else if (fmt == 0x82) {
            // A1R5G5B5 — 16-bit, big-endian
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
            for (uint32_t i = 0; i < W * H; ++i) {
                uint16_t px = __builtin_bswap16(s16[i]);
                uint8_t aa = (px >> 15) ? 0xFF : 0x00;
                uint8_t rr = ((px >> 10) & 0x1F) * 255 / 31;
                uint8_t gg = ((px >>  5) & 0x1F) * 255 / 31;
                uint8_t bb = ((px)       & 0x1F) * 255 / 31;
                rgba8[i] = ((uint32_t)aa << 24) | ((uint32_t)rr << 16) |
                            ((uint32_t)gg << 8) | (uint32_t)bb;
            }
        } else if (fmt == 0x83) {
            // A4R4G4B4 — 16-bit, big-endian
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
            for (uint32_t i = 0; i < W * H; ++i) {
                uint16_t px = __builtin_bswap16(s16[i]);
                uint8_t aa = ((px >> 12) & 0xF) * 17;
                uint8_t rr = ((px >>  8) & 0xF) * 17;
                uint8_t gg = ((px >>  4) & 0xF) * 17;
                uint8_t bb = ((px)       & 0xF) * 17;
                rgba8[i] = ((uint32_t)aa << 24) | ((uint32_t)rr << 16) |
                            ((uint32_t)gg << 8) | (uint32_t)bb;
            }
        } else if (fmt == 0x94) {
            // X16 — 16-bit single channel, big-endian
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
            for (uint32_t i = 0; i < W * H; ++i) {
                uint16_t px = __builtin_bswap16(s16[i]);
                uint8_t v = px >> 8;  // top 8 bits
                rgba8[i] = 0xFF000000u | ((uint32_t)v << 16) |
                            ((uint32_t)v << 8) | (uint32_t)v;
            }
        } else if (fmt == 0x97) {
            // R5G5B5A1 — 16-bit, big-endian (alpha in LSB)
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
            for (uint32_t i = 0; i < W * H; ++i) {
                uint16_t px = __builtin_bswap16(s16[i]);
                uint8_t rr = ((px >> 11) & 0x1F) * 255 / 31;
                uint8_t gg = ((px >>  6) & 0x1F) * 255 / 31;
                uint8_t bb = ((px >>  1) & 0x1F) * 255 / 31;
                uint8_t aa = (px & 1) ? 0xFF : 0x00;
                rgba8[i] = ((uint32_t)aa << 24) | ((uint32_t)rr << 16) |
                            ((uint32_t)gg << 8) | (uint32_t)bb;
            }
        } else if (fmt == 0x86) {
            // DXT1 (S3TC) — block compressed
            decodeDXTTexture(src, W, H, 1, rgba8);
        } else if (fmt == 0x87) {
            // DXT23 (S3TC) — block compressed with explicit alpha
            decodeDXTTexture(src, W, H, 3, rgba8);
        } else if (fmt == 0x88) {
            // DXT45 (S3TC) — block compressed with interpolated alpha
            decodeDXTTexture(src, W, H, 5, rgba8);
        } else {
            // Unknown format: magenta debug
            for (uint32_t i = 0; i < W * H; ++i) rgba8[i] = 0xFFFF00FFu;
        }
        rast_->setTexture2D(rgba8.data(), W, H, tu);
        // Extract wrap modes from TEXTURE_ADDRESS register
        uint32_t addr = t.address;
        uint8_t wS = (addr) & 0xF;        // wrapS bits [3:0]
        uint8_t wT = (addr >> 8) & 0xF;   // wrapT bits [11:8]
        if (wS == 0) wS = 1;  // default REPEAT
        if (wT == 0) wT = 1;
        rast_->setTextureWrap(tu, wS, wT);
        // Extract mag filter from TEXTURE_FILTER register: bits [27:24]
        // RSX: 1=NEAREST, 2=LINEAR
        uint8_t mag = (t.filter >> 24) & 0xF;
        rast_->setTextureMagFilter(tu, (mag == 1) ? 0 : 1);  // 0=NEAREST, 1=LINEAR
        if (tu == 0) {
            cachedTexOff_ = t.offset;
            cachedTexW_ = W;
            cachedTexH_ = H;
            cachedTexFmt_ = t.format;
            cachedTexValid_ = true;
        }
        texBoundForDraw = true;
    }

    // ── Per-pixel FP decode + upload ─────────────────────────────
    // Decode the guest FP microcode into packed instructions and upload
    // to the GPU rasterizer for per-pixel execution. Falls back to
    // texture-replace when no FP is available.
    if (vram_ && s.fpOffset != 0) {
        uint32_t fpOff = s.fpOffset & 0x0FFFFFFFu;
        if (fpOff < vramSize_ && (vramSize_ - fpOff) >= 16) {
            const uint32_t* fpData = reinterpret_cast<const uint32_t*>(vram_ + fpOff);
            uint32_t fpMaxWords = (uint32_t)((vramSize_ - fpOff) / 4);
            if (fpMaxWords > 16384) fpMaxWords = 16384;

            // Decode and pack FP instructions.
            // RSX FP embeds constants inline: when any source uses type=CONST,
            // the next 4 words after that instruction are the constant's 4 floats
            // (byte-swapped like all RSX FP data). We extract them and assign
            // sequential constant indices for the GPU-side array.
            std::vector<uint32_t> packed;
            std::vector<float> fpConstants; // 4 floats per constant
            uint32_t pc = 0;
            while (pc * 4 + 3 < fpMaxWords && pc < 256) {
                FPDecodedInsn insn = fp_decode(fpData + pc * 4);

                // Check if any source uses inline constant
                bool hasConst = false;
                for (int si = 0; si < 3; ++si) {
                    if (insn.src[si].regType == 2) { // FP_REG_CONSTANT
                        hasConst = true;
                        break;
                    }
                }

                // Extract inline constant data if present
                uint32_t constIdx = 0;
                if (hasConst) {
                    uint32_t constWordOff = (pc + 1) * 4;
                    if (constWordOff + 3 < fpMaxWords) {
                        constIdx = (uint32_t)(fpConstants.size() / 4);
                        for (int ci = 0; ci < 4; ++ci) {
                            uint32_t raw = fp_swap_word(fpData[constWordOff + ci]);
                            float fval;
                            memcpy(&fval, &raw, sizeof(float));
                            fpConstants.push_back(fval);
                        }
                        for (int si = 0; si < 3; ++si) {
                            if (insn.src[si].regType == 2) {
                                insn.src[si].regIdx = constIdx;
                            }
                        }
                    }
                }

                uint32_t w0 = ((insn.opcode & 0x7F)) |
                              ((insn.maskX ? 1u : 0u) << 7) |
                              ((insn.maskY ? 1u : 0u) << 8) |
                              ((insn.maskZ ? 1u : 0u) << 9) |
                              ((insn.maskW ? 1u : 0u) << 10) |
                              ((insn.texUnit & 0xF) << 11) |
                              ((insn.inputAttr & 0xF) << 15) |
                              ((insn.saturate ? 1u : 0u) << 19) |
                              ((insn.endFlag ? 1u : 0u) << 20) |
                              ((insn.noDest ? 1u : 0u) << 21) |
                              ((insn.setCond ? 1u : 0u) << 22);

                auto packSrc = [](const FPDecodedSrc& s) -> uint32_t {
                    return (s.regType & 3) |
                           ((s.regIdx & 0xFF) << 2) |
                           ((s.swzX & 3) << 10) |
                           ((s.swzY & 3) << 12) |
                           ((s.swzZ & 3) << 14) |
                           ((s.swzW & 3) << 16) |
                           ((s.neg ? 1u : 0u) << 18) |
                           ((s.abs ? 1u : 0u) << 19);
                };

                uint32_t w1 = (insn.dstReg & 0xFF) | (packSrc(insn.src[0]) << 8);
                uint32_t w2 = packSrc(insn.src[1]);
                uint32_t w3 = packSrc(insn.src[2]);

                // w4: condition code fields
                uint32_t w4 = (insn.execLT ? 1u : 0u) |
                              ((insn.execEQ ? 1u : 0u) << 1) |
                              ((insn.execGR ? 1u : 0u) << 2) |
                              ((insn.condSwzX & 3) << 3) |
                              ((insn.condSwzY & 3) << 5) |
                              ((insn.condSwzZ & 3) << 7) |
                              ((insn.condSwzW & 3) << 9) |
                              ((insn.condModRegIdx & 1) << 11) |
                              ((insn.condRegIdx & 1) << 12);

                packed.push_back(w0);
                packed.push_back(w1);
                packed.push_back(w2);
                packed.push_back(w3);
                packed.push_back(w4);
                packed.push_back(0); // reserved

                if (insn.endFlag) break;
                // Skip inline constant data (4 extra words) when present
                pc += hasConst ? 2 : 1;
            }

            uint32_t insnCount = (uint32_t)(packed.size() / 6);
            if (insnCount > 0) {
                uint32_t constCount = (uint32_t)(fpConstants.size() / 4);
                rast_->setFragmentProgram(packed.data(), insnCount,
                                          constCount > 0 ? fpConstants.data() : nullptr,
                                          constCount);
            }
        }
    }

    // ── Fallback: per-vertex texture modulation (no FP active) ──
    // When no fragment program is set and no GPU-side texture is bound,
    // modulate vertex colors by texture unit 0 at each vertex's UV.
    std::vector<RasterVertex> textured;
    if (!texBoundForDraw && vram_ && s.textures[0].enabled && base != nullptr &&
        s.fpOffset == 0) {
        uint8_t texFmt = ((s.textures[0].format >> 8) & 0xFF) & 0x9F;
        bool fmtKnown = (texFmt == 0x85 || texFmt == 0x81 || texFmt == 0x82 ||
                         texFmt == 0x83 || texFmt == 0x84 || texFmt == 0x86 ||
                         texFmt == 0x87 || texFmt == 0x88 || texFmt == 0x8B ||
                         texFmt == 0x94 || texFmt == 0x97 || texFmt == 0x9E);
        if (fmtKnown) {
            ps3rsx::HostTextureSamplerCtx tctx{ vram_, vramSize_, &s };
            textured.resize(count);
            for (uint32_t i = 0; i < count; ++i) {
                RasterVertex v = base[i];
                float uvw[3] = { v.u, v.v, 0.f };
                float rgba[4] = { 1.f, 1.f, 1.f, 1.f };
                ps3rsx::rsx_host_sampler(&tctx, 0, uvw, rgba);
                v.r *= rgba[0]; v.g *= rgba[1];
                v.b *= rgba[2]; v.a *= rgba[3];
                textured[i] = v;
            }
            base = textured.data();
        }
    }

    switch (s.currentPrim) {
    case PRIM_POINTS:
        rast_->drawPoints(base, count);
        break;
    case PRIM_LINES:
        rast_->drawLines(base, count - (count % 2));
        break;
    case PRIM_LINE_LOOP: {
        if (count < 2) break;
        std::vector<RasterVertex> exp;
        exp.reserve(count * 2);
        for (uint32_t i = 0; i + 1 < count; ++i) {
            exp.push_back(base[i]);
            exp.push_back(base[i + 1]);
        }
        exp.push_back(base[count - 1]);
        exp.push_back(base[0]);
        rast_->drawLines(exp.data(), (uint32_t)exp.size());
        break;
    }
    case PRIM_LINE_STRIP: {
        if (count < 2) break;
        // Expand to a line list: (v0,v1),(v1,v2),...
        std::vector<RasterVertex> exp;
        exp.reserve((count - 1) * 2);
        for (uint32_t i = 0; i + 1 < count; ++i) {
            exp.push_back(base[i]);
            exp.push_back(base[i + 1]);
        }
        rast_->drawLines(exp.data(), (uint32_t)exp.size());
        break;
    }
    case PRIM_TRIANGLE_STRIP: {
        if (count < 3) break;
        std::vector<RasterVertex> exp;
        exp.reserve((count - 2) * 3);
        for (uint32_t i = 0; i + 2 < count; ++i) {
            // Flip winding on every odd triangle to preserve facing.
            if (i & 1) {
                exp.push_back(base[i + 1]);
                exp.push_back(base[i]);
                exp.push_back(base[i + 2]);
            } else {
                exp.push_back(base[i]);
                exp.push_back(base[i + 1]);
                exp.push_back(base[i + 2]);
            }
        }
        // DEBUG: check vertex colors before draw
        static int expDbg = 0;
        if (expDbg < 2 && !exp.empty()) {
            printf("[VERTS-IN] count=%zu v0: pos=(%.1f,%.1f) col=(%.3f,%.3f,%.3f,%.3f) uv=(%.3f,%.3f)\n",
                   exp.size(), exp[0].x, exp[0].y,
                   exp[0].r, exp[0].g, exp[0].b, exp[0].a,
                   exp[0].u, exp[0].v);
            expDbg++;
        }
        rast_->drawTriangles(exp.data(), (uint32_t)exp.size());
        break;
    }
    case PRIM_TRIANGLE_FAN: {
        if (count < 3) break;
        std::vector<RasterVertex> exp;
        exp.reserve((count - 2) * 3);
        for (uint32_t i = 1; i + 1 < count; ++i) {
            exp.push_back(base[0]);
            exp.push_back(base[i]);
            exp.push_back(base[i + 1]);
        }
        rast_->drawTriangles(exp.data(), (uint32_t)exp.size());
        break;
    }
    case PRIM_QUADS: {
        if (count < 4) break;
        uint32_t quads = count / 4;
        std::vector<RasterVertex> exp;
        exp.reserve(quads * 6);
        for (uint32_t q = 0; q < quads; ++q) {
            const auto& a = base[q * 4 + 0];
            const auto& b = base[q * 4 + 1];
            const auto& c = base[q * 4 + 2];
            const auto& d = base[q * 4 + 3];
            exp.push_back(a); exp.push_back(b); exp.push_back(c);
            exp.push_back(a); exp.push_back(c); exp.push_back(d);
        }
        rast_->drawTriangles(exp.data(), (uint32_t)exp.size());
        break;
    }
    case PRIM_QUAD_STRIP: {
        if (count < 4) break;
        std::vector<RasterVertex> exp;
        uint32_t nQuads = (count - 2) / 2;
        exp.reserve(nQuads * 6);
        for (uint32_t q = 0; q < nQuads; ++q) {
            // Each quad: (v[2q], v[2q+1], v[2q+3], v[2q+2])
            const auto& a = base[q * 2 + 0];
            const auto& b = base[q * 2 + 1];
            const auto& c = base[q * 2 + 3];
            const auto& d = base[q * 2 + 2];
            exp.push_back(a); exp.push_back(b); exp.push_back(c);
            exp.push_back(a); exp.push_back(c); exp.push_back(d);
        }
        rast_->drawTriangles(exp.data(), (uint32_t)exp.size());
        break;
    }
    case PRIM_POLYGON:
        // Convex polygon → triangle fan from first vertex
        if (count >= 3) {
            std::vector<RasterVertex> exp;
            exp.reserve((count - 2) * 3);
            for (uint32_t i = 1; i + 1 < count; ++i) {
                exp.push_back(base[0]);
                exp.push_back(base[i]);
                exp.push_back(base[i + 1]);
            }
            rast_->drawTriangles(exp.data(), (uint32_t)exp.size());
        }
        break;
    case PRIM_TRIANGLES:
    default:
        rast_->drawTriangles(base, count - (count % 3));
        break;
    }
    counters.draws++;
}

void RasterBridge::onDrawIndexed(const RSXState& s, uint32_t first, uint32_t count,
                                 uint32_t /*indexFormat*/) {
    if (!rast_ || count == 0) return;
    applyPipelineState(s);

    // Path A — host-staged vertex + index pools (legacy test path).
    if (pool_ && idxPool_) {
        if ((uint64_t)first + count > idxCount_) return;
        rast_->drawIndexed(pool_, poolCount_, idxPool_ + first, count, false);
        counters.drawIndexed++;
        return;
    }

    // Path B — VRAM-resident indices + VRAM-decoded vertex stream.
    // Mirrors what real PS3 games drive via cellGcmSetDrawIndexArray:
    // SET_INDEX_ARRAY_ADDRESS captured into s.indexArrayAddress, the
    // DRAW_INDEX_ARRAY method then specifies (first, count). U16 LE
    // is the common format (s.indexArrayFormat low bit = 0).
    if (vram_ && s.vertexArrays[0].enabled) {
        const bool u32 = (s.indexArrayFormat & 1u) != 0;
        const uint32_t stride = u32 ? 4u : 2u;
        const uint64_t need   = (uint64_t)(first + count) * stride;
        if ((uint64_t)s.indexArrayAddress + need > vramSize_) return;

        // Decode both streams into host buffers, then drive the rasterizer.
        std::vector<RasterVertex> verts;
        // Determine highest index we'll touch so decode is bounded.
        const uint8_t* idxSrc = vram_ + s.indexArrayAddress + (uint64_t)first * stride;
        std::vector<uint32_t> idx32(count);
        uint32_t maxIdx = 0;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t v = u32
                ? *reinterpret_cast<const uint32_t*>(idxSrc + i * 4u)
                : (uint32_t)*reinterpret_cast<const uint16_t*>(idxSrc + i * 2u);
            idx32[i] = v;
            if (v > maxIdx) maxIdx = v;
        }
        decode_vertex_stream(s, vram_, vramSize_, 0, maxIdx + 1, verts);
        if (verts.empty()) return;

        // CudaRasterizer::drawIndexed expects uint16_t indices; downcast
        // when safe, otherwise expand triangles ourselves.
        if (maxIdx <= 0xFFFFu) {
            std::vector<uint16_t> idx16(count);
            for (uint32_t i = 0; i < count; ++i) idx16[i] = (uint16_t)idx32[i];
            rast_->drawIndexed(verts.data(), (uint32_t)verts.size(),
                               idx16.data(), count, false);
        } else {
            // Fallback: gather vertices through the index list.
            std::vector<RasterVertex> expanded(count);
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t v = idx32[i];
                expanded[i] = (v < verts.size()) ? verts[v] : RasterVertex{};
            }
            rast_->drawTriangles(expanded.data(), count);
        }
        counters.drawIndexed++;
        return;
    }
}

void RasterBridge::onFlip(const RSXState&, uint32_t) {
    // Flip is a present barrier on real HW. For the headless rasterizer
    // it's a sync point; the caller reads back color after FLIP completes.
    counters.flips++;
}

} // namespace rsx

// ═══════════════════════════════════════════════════════════════
// C ABI shim — same symbol names as rsx_vulkan_emitter_shim.cpp so
// linking one OR the other controls which backend the FIFO drives.
// Only link one of the two shims per binary.
// ═══════════════════════════════════════════════════════════════

#include "rsx_defs.h"

extern "C" {

void rsx_emitter_onSurfaceSetup(void* br, const rsx::RSXState* s) {
    static_cast<rsx::RasterBridge*>(br)->onSurfaceSetup(*s);
}
void rsx_emitter_onViewport(void* br, const rsx::RSXState* s) {
    static_cast<rsx::RasterBridge*>(br)->onViewport(*s);
}
void rsx_emitter_onScissor(void* br, const rsx::RSXState* s) {
    static_cast<rsx::RasterBridge*>(br)->onScissor(*s);
}
void rsx_emitter_onClearSurface(void* br, const rsx::RSXState* s, uint32_t mask) {
    static_cast<rsx::RasterBridge*>(br)->onClearSurface(*s, mask);
}
void rsx_emitter_onBeginEnd(void* br, const rsx::RSXState* s, uint32_t prim) {
    static_cast<rsx::RasterBridge*>(br)->onBeginEnd(*s, prim);
}
void rsx_emitter_onDrawArrays(void* br, const rsx::RSXState* s,
                              uint32_t first, uint32_t count) {
    static_cast<rsx::RasterBridge*>(br)->onDrawArrays(*s, first, count);
}
void rsx_emitter_onDrawIndexed(void* br, const rsx::RSXState* s,
                               uint32_t first, uint32_t count, uint32_t fmt) {
    static_cast<rsx::RasterBridge*>(br)->onDrawIndexed(*s, first, count, fmt);
}
void rsx_emitter_onFlip(void* br, const rsx::RSXState* s, uint32_t surfaceOffset) {
    static_cast<rsx::RasterBridge*>(br)->onFlip(*s, surfaceOffset);
}

} // extern "C"
