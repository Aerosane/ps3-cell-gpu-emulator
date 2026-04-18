// rsx_raster_bridge.cpp — Implementation.

#include "rsx_raster_bridge.h"
#include "rsx_defs.h"
#include "rsx_vp_shader.h"

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
            float f;
            std::memcpy(&f, p + i * 4, 4);
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
            out[3] = p[3] / 255.0f;  // A
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

        decode_attr(vram, vramSize, s.vertexArrays[0], idx, 4, pos);
        if (s.vertexArrays[3].enabled)
            decode_attr(vram, vramSize, s.vertexArrays[3], idx, 4, color);
        if (s.vertexArrays[8].enabled)
            decode_attr(vram, vramSize, s.vertexArrays[8], idx, 4, uv);

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
    // rsx_command_processor uses logical flags from rsx_defs.h:
    // CLEAR_COLOR = 0x01, CLEAR_DEPTH = 0x02, CLEAR_STENCIL = 0x04.
    if (mask & 0x01) rast_->clear(s.colorClearValue);
    if (mask & 0x02) rast_->clearDepth(1.0f);
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
    // decoded/pool stream. Output register 0 is HPOS (expected in
    // screen-space xyz for this simplified pipeline — a full MVP-based
    // clip-space VP is still TODO), o[1] is diffuse color, o[4] is uv.
    std::vector<RasterVertex> transformed;
    if (s.vpValid) {
        if (base == nullptr) return;
        transformed.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            VPFloat4 inputs[16] = {};
            inputs[0].v[0] = base[i].x; inputs[0].v[1] = base[i].y;
            inputs[0].v[2] = base[i].z; inputs[0].v[3] = 1.0f;
            inputs[3].v[0] = base[i].r; inputs[3].v[1] = base[i].g;
            inputs[3].v[2] = base[i].b; inputs[3].v[3] = base[i].a;
            inputs[8].v[0] = base[i].u; inputs[8].v[1] = base[i].v;

            const VPFloat4* consts = reinterpret_cast<const VPFloat4*>(s.vpConstants);
            VPFloat4 outputs[16] = {};
            // Default pass-through: pos=in0, color=in3, uv=in8
            outputs[0] = inputs[0];
            outputs[1] = inputs[3];
            outputs[4] = inputs[8];

            vp_execute(s.vpData, 512u * 4u, s.vpStart,
                       inputs, consts, outputs);

            RasterVertex& v = transformed[i];
            v.x = outputs[0].v[0]; v.y = outputs[0].v[1]; v.z = outputs[0].v[2];
            v.r = outputs[1].v[0]; v.g = outputs[1].v[1];
            v.b = outputs[1].v[2]; v.a = outputs[1].v[3];
            v.u = outputs[4].v[0]; v.v = outputs[4].v[1];
        }
        base = transformed.data();
    }

    switch (s.currentPrim) {
    case PRIM_POINTS:
        rast_->drawPoints(base, count);
        break;
    case PRIM_LINES:
        rast_->drawLines(base, count - (count % 2));
        break;
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
    case PRIM_TRIANGLES:
    default:
        rast_->drawTriangles(base, count - (count % 3));
        break;
    }
    counters.draws++;
}

void RasterBridge::onDrawIndexed(const RSXState& s, uint32_t first, uint32_t count,
                                 uint32_t /*indexFormat*/) {
    if (!rast_ || !pool_ || !idxPool_ || count == 0) return;
    if ((uint64_t)first + count > idxCount_) return;
    applyPipelineState(s);
    rast_->drawIndexed(pool_, poolCount_, idxPool_ + first, count, false);
    counters.drawIndexed++;
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
