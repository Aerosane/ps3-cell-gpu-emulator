// rsx_raster_bridge.cpp — Implementation.

#include "rsx_raster_bridge.h"
#include "rsx_defs.h"

#include <vector>

namespace rsx {

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

void RasterBridge::onDrawArrays(const RSXState& s, uint32_t first, uint32_t count) {
    if (!rast_ || !pool_ || count == 0) return;
    if ((uint64_t)first + count > poolCount_) return;
    const RasterVertex* base = pool_ + first;
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

void RasterBridge::onDrawIndexed(const RSXState&, uint32_t first, uint32_t count,
                                 uint32_t /*indexFormat*/) {
    if (!rast_ || !pool_ || !idxPool_ || count == 0) return;
    if ((uint64_t)first + count > idxCount_) return;
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
