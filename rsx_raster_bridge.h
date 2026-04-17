#pragma once
// rsx_raster_bridge.h — Drives a CudaRasterizer from RSX FIFO state.
//
// Sits in the same slot as VulkanEmitter but instead of recording a
// command stream, it calls directly into the CUDA rasterizer. Intended
// as the production path; the Vulkan translator remains for portability
// experiments.
//
// Vertex source: since NV40 vertex-array decode isn't implemented yet,
// the bridge accepts an externally-supplied vertex pool. When a draw
// method fires, the bridge pulls verts[first, first+count) into a
// drawTriangles / drawIndexed call. The caller is responsible for
// populating the pool before pushing the draw command.

#include "rsx_raster.h"
#include <cstdint>
#include <vector>

namespace rsx { struct RSXState; }

namespace rsx {

class RasterBridge {
public:
    RasterBridge() = default;

    int  attach(CudaRasterizer* r) { rast_ = r; return r ? 0 : -1; }
    void setVertexPool(const RasterVertex* v, uint32_t count) {
        pool_ = v; poolCount_ = count;
    }
    void setIndexPool(const uint16_t* i, uint32_t count) {
        idxPool_ = i; idxCount_ = count;
    }

    // Hooks called from rsx_command_processor.cu via the weak-symbol shim.
    void onSurfaceSetup(const RSXState& s);
    void onViewport(const RSXState& s);
    void onScissor(const RSXState& s);
    void onClearSurface(const RSXState& s, uint32_t mask);
    void onBeginEnd(const RSXState& s, uint32_t prim);
    void onDrawArrays(const RSXState& s, uint32_t first, uint32_t count);
    void onDrawIndexed(const RSXState& s, uint32_t first, uint32_t count,
                       uint32_t indexFormat);
    void onFlip(const RSXState& s, uint32_t surfaceOffset);

    struct Counters {
        uint32_t surfaceSetups{0};
        uint32_t clears{0};
        uint32_t draws{0};
        uint32_t drawIndexed{0};
        uint32_t flips{0};
    } counters;

private:
    CudaRasterizer* rast_{nullptr};
    const RasterVertex* pool_{nullptr};
    uint32_t poolCount_{0};
    const uint16_t* idxPool_{nullptr};
    uint32_t idxCount_{0};
};

} // namespace rsx
