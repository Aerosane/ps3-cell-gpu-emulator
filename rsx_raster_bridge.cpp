// rsx_raster_bridge.cpp — Implementation.

#include "rsx_raster_bridge.h"
#include "rsx_defs.h"
#include "rsx_vp_shader.h"
#include "rsx_fp_shader.h"
#include "rsx_texture.h"

#include <cstdio>
#include <cstring>
#include <vector>

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
    default:
        // Unsupported type today — treat attribute as defaulted and
        // let the caller fall back.
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
        v.x = pos[0]; v.y = pos[1]; v.z = pos[2];
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
            float W = (float)(s.surfaceWidth  ? s.surfaceWidth  : 1280);
            float H = (float)(s.surfaceHeight ? s.surfaceHeight : 720);
            v.x = (nx * 0.5f + 0.5f) * W;
            v.y = (1.0f - (ny * 0.5f + 0.5f)) * H;
            v.z = nz * 0.5f + 0.5f;
            // Color from o[1] (COL0), UV from o[7] (TEX0)
            v.r = outputs[1].v[0]; v.g = outputs[1].v[1];
            v.b = outputs[1].v[2]; v.a = outputs[1].v[3];
            v.u = outputs[7].v[0]; v.v = outputs[7].v[1];
        }
        base = transformed.data();
    }

    // ── Per-pixel texture binding ───────────────────────────────
    // Upload texture unit 0 to the GPU rasterizer at draw time so we
    // use the RSXState that's actually current. The rasterizer kernel
    // does per-pixel UV interpolation + texture modulation, which is
    // far superior to per-vertex FP sampling.
    bool texBoundForDraw = false;
    if (vram_ && s.textures[0].enabled &&
        s.textures[0].width > 0 && s.textures[0].height > 0) {
        const auto& t = s.textures[0];
        bool stale =
            !cachedTexValid_ ||
            t.offset != cachedTexOff_ || t.width != cachedTexW_ ||
            t.height != cachedTexH_   || t.format != cachedTexFmt_;
        if (stale) {
            uint8_t fmt = ((t.format >> 8) & 0xFF) & 0x9F;
            uint32_t W = t.width, H = t.height;
            uint64_t need = (uint64_t)W * H * (fmt == 0x81 ? 1u : 4u);
            if ((uint64_t)t.offset + need <= vramSize_) {
                const uint8_t* src = vram_ + t.offset;
                std::vector<uint32_t> rgba8(W * H);
                if (fmt == 0x85) {
                    // A8R8G8B8 — guest stores as big-endian A8R8G8B8;
                    // our rasterizer expects 0xAARRGGBB host-endian.
                    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(src);
                    for (uint32_t i = 0; i < W * H; ++i) {
                        uint32_t px = __builtin_bswap32(s32[i]);
                        // Ensure alpha is opaque if guest left it 0
                        if ((px >> 24) == 0) px |= 0xFF000000u;
                        rgba8[i] = px;
                    }
                } else if (fmt == 0x81) {
                    for (uint32_t i = 0; i < W * H; ++i) {
                        uint8_t v = src[i];
                        rgba8[i] = 0xFF000000u |
                                   ((uint32_t)v << 16) |
                                   ((uint32_t)v <<  8) | (uint32_t)v;
                    }
                } else {
                    for (uint32_t i = 0; i < W * H; ++i) rgba8[i] = 0xFFFF00FFu;
                }
                rast_->setTexture2D(rgba8.data(), W, H);
                cachedTexOff_ = t.offset;
                cachedTexW_ = W;
                cachedTexH_ = H;
                cachedTexFmt_ = t.format;
                cachedTexValid_ = true;
            }
        }
        texBoundForDraw = cachedTexValid_;
    }

    // ── Per-vertex FP shading ─────────────────────────────────
    // Run the FP for each vertex and replace colors with FP output.
    // Skip when the rasterizer already has a per-pixel texture bound
    // (the rasterizer's per-pixel UV interpolation + texture modulation
    // produces far better results than per-vertex FP with its degenerate
    // corner-UV sampling).
    // TODO: True per-pixel FP execution for non-modulate shaders.
    std::vector<RasterVertex> fpShaded;
    if (!texBoundForDraw && vram_ && s.fpOffset != 0 && base != nullptr) {
        const uint32_t* fpData = reinterpret_cast<const uint32_t*>(
            vram_ + (s.fpOffset & 0x0FFFFFFFu));
        uint32_t fpOff = s.fpOffset & 0x0FFFFFFFu;
        if (fpOff < vramSize_ && (vramSize_ - fpOff) >= 16) {
            uint32_t fpMaxWords = (uint32_t)((vramSize_ - fpOff) / 4);
            if (fpMaxWords > 16384) fpMaxWords = 16384;

            ps3rsx::HostTextureSamplerCtx tctx{ vram_, vramSize_, &s };
            fpShaded.resize(count);
            for (uint32_t i = 0; i < count; ++i) {
                const RasterVertex& v = base[i];
                FPFloat4 fpIn[16] = {};
                fpIn[0] = FPFloat4{{ v.x, v.y, v.z, 1.0f }};
                fpIn[1] = FPFloat4{{ v.r, v.g, v.b, v.a }};
                fpIn[2] = FPFloat4{{ 0, 0, 0, 0 }};
                fpIn[3] = FPFloat4{{ 0, 0, 0, 0 }};
                fpIn[4] = FPFloat4{{ v.u, v.v, 0, 1 }};

                FPFloat4 fpOut[4] = {};
                fp_execute(fpData, fpMaxWords, fpIn, fpOut,
                           ps3rsx::rsx_host_sampler, &tctx);

                RasterVertex& o = fpShaded[i];
                o = v;
                o.r = fpOut[0].v[0];
                o.g = fpOut[0].v[1];
                o.b = fpOut[0].v[2];
                o.a = fpOut[0].v[3];
            }
            base = fpShaded.data();
        }
    }
    // ── Fallback: per-vertex texture modulation (no FP active) ──
    // When no fragment program is set and no GPU-side texture is bound,
    // modulate vertex colors by texture unit 0 at each vertex's UV.
    std::vector<RasterVertex> textured;
    if (!texBoundForDraw && fpShaded.empty() && vram_ && s.textures[0].enabled && base != nullptr) {
        uint8_t texFmt = ((s.textures[0].format >> 8) & 0xFF) & 0x9F;
        bool fmtKnown = (texFmt == 0x85 || texFmt == 0x81);
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
