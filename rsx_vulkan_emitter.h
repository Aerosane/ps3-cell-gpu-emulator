#pragma once
// rsx_vulkan_emitter.h — RSX → Vulkan command translator
//
// Records a structured stream of Vulkan-equivalent operations as the
// RSX FIFO is processed. Decoupled from the live Vulkan driver so the
// translation layer can be unit-tested without a GPU context, and later
// replayed against a real VkCommandBuffer.
//
// Mapping:
//   NV4097_CLEAR_SURFACE       → vkCmdClearAttachments (COLOR/DEPTH bits)
//   NV4097_SET_VIEWPORT_*      → vkCmdSetViewport
//   NV4097_SET_SCISSOR_*       → vkCmdSetScissor
//   NV4097_SET_SURFACE_FORMAT  → VkFramebuffer / begin-render-pass setup
//   NV4097_SET_BEGIN_END (!=0) → vkCmdBeginRenderPass (lazy, on first draw)
//   NV4097_DRAW_ARRAYS         → vkCmdDraw
//   End of frame (FLIP)        → vkCmdEndRenderPass + vkQueuePresent

#include <cstdint>
#include <vector>

namespace rsx {

struct RSXState;

enum class VkOp : uint32_t {
    BeginFrame,
    SetViewport,
    SetScissor,
    BindPipeline,
    BindVertexBuffer,
    ClearAttachment,
    Draw,
    DrawIndexed,
    EndRenderPass,
    Present,
};

struct VkRecord {
    VkOp op;
    // Tagged payload. Only the field that matches `op` is meaningful.
    union {
        struct { uint32_t width, height, format; } beginFrame;
        struct { float x, y, w, h, minDepth, maxDepth; } viewport;
        struct { int32_t x, y; uint32_t w, h; } scissor;
        struct { uint32_t prim; uint32_t pipelineHash; } bindPipeline;
        struct { uint32_t slot; uint64_t offset; uint32_t stride; uint32_t format; } bindVB;
        struct { uint32_t mask; uint32_t color; float depth; uint32_t stencil; } clear;
        struct { uint32_t firstVertex, vertexCount, prim; } draw;
        struct { uint32_t firstIndex, indexCount, prim; uint32_t indexFormat; } drawIndexed;
        struct { uint32_t frameIndex; } endRenderPass;
        struct { uint32_t surfaceOffset; uint32_t frameIndex; } present;
    };
};

// Receives every translated command. The emitter itself is POD-friendly
// so tests can walk the vector directly.
class VulkanEmitter {
public:
    VulkanEmitter() = default;

    void clear() { records_.clear(); frameIndex_ = 0; inRenderPass_ = false; }
    size_t size() const { return records_.size(); }
    const VkRecord& operator[](size_t i) const { return records_[i]; }
    const std::vector<VkRecord>& records() const { return records_; }

    // Emitters — called by the FIFO dispatcher.
    void onSurfaceSetup(const RSXState& s);
    void onViewport(const RSXState& s);
    void onScissor(const RSXState& s);
    void onClearSurface(const RSXState& s, uint32_t mask);
    void onBeginEnd(const RSXState& s, uint32_t prim);
    void onDrawArrays(const RSXState& s, uint32_t first, uint32_t count);
    void onDrawIndexed(const RSXState& s, uint32_t first, uint32_t count, uint32_t indexFormat);
    void onFlip(const RSXState& s, uint32_t surfaceOffset);

    // Debug
    void dump() const;
    const char* opName(VkOp op) const;

    // Counters for quick assertions in tests
    struct Counters {
        uint32_t beginFrame{0}, clears{0}, draws{0}, drawsIndexed{0},
                 viewports{0}, scissors{0}, bindPipelines{0},
                 endRenderPasses{0}, presents{0};
    } counters;

private:
    void maybeBeginFrame(const RSXState& s);
    void maybeBindPipeline(const RSXState& s, uint32_t prim);
    void push(const VkRecord& r) { records_.push_back(r); }

    std::vector<VkRecord> records_;
    uint32_t frameIndex_{0};
    bool     inRenderPass_{false};
    uint32_t lastPipelineHash_{0};
};

} // namespace rsx
