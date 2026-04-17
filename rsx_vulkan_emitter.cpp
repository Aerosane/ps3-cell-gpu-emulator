// rsx_vulkan_emitter.cpp — implementation of RSX → Vulkan translator
//
// Host-side (no CUDA). Compiled with g++.

#include "rsx_vulkan_emitter.h"
#include "rsx_defs.h"

#include <cstdio>

namespace rsx {

const char* VulkanEmitter::opName(VkOp op) const {
    switch (op) {
        case VkOp::BeginFrame:       return "BeginFrame";
        case VkOp::SetViewport:      return "SetViewport";
        case VkOp::SetScissor:       return "SetScissor";
        case VkOp::BindPipeline:     return "BindPipeline";
        case VkOp::BindVertexBuffer: return "BindVertexBuffer";
        case VkOp::ClearAttachment:  return "ClearAttachment";
        case VkOp::Draw:             return "Draw";
        case VkOp::DrawIndexed:      return "DrawIndexed";
        case VkOp::EndRenderPass:    return "EndRenderPass";
        case VkOp::Present:          return "Present";
    }
    return "?";
}

void VulkanEmitter::maybeBeginFrame(const RSXState& s) {
    if (inRenderPass_) return;
    VkRecord r{};
    r.op = VkOp::BeginFrame;
    r.beginFrame.width  = s.surfaceWidth  ? s.surfaceWidth  : 1280;
    r.beginFrame.height = s.surfaceHeight ? s.surfaceHeight : 720;
    r.beginFrame.format = s.surfaceFormat;
    push(r);
    counters.beginFrame++;
    inRenderPass_ = true;
}

void VulkanEmitter::maybeBindPipeline(const RSXState& s, uint32_t prim) {
    // Pipeline hash = prim | (depth<<4) | (blend<<5) | (cull<<6) | (vpStart<<16) | (fpOff>>8)
    uint32_t hash = prim
                  | ((s.depthTestEnable ? 1u : 0u) << 4)
                  | ((s.blendEnable ? 1u : 0u) << 5)
                  | ((s.cullFaceEnable ? 1u : 0u) << 6)
                  | ((uint32_t)s.vpStart << 16)
                  | (s.fpOffset >> 8);
    if (hash == lastPipelineHash_) return;
    lastPipelineHash_ = hash;

    VkRecord r{};
    r.op = VkOp::BindPipeline;
    r.bindPipeline.prim         = prim;
    r.bindPipeline.pipelineHash = hash;
    push(r);
    counters.bindPipelines++;
}

void VulkanEmitter::onSurfaceSetup(const RSXState& /*s*/) {
    // Deferred: the actual BeginFrame happens on the first clear or draw.
    // We just invalidate the active render pass so the next op re-begins it
    // with the new surface dims.
    if (inRenderPass_) {
        VkRecord r{};
        r.op = VkOp::EndRenderPass;
        r.endRenderPass.frameIndex = frameIndex_;
        push(r);
        counters.endRenderPasses++;
        inRenderPass_ = false;
    }
}

void VulkanEmitter::onViewport(const RSXState& s) {
    maybeBeginFrame(s);
    VkRecord r{};
    r.op = VkOp::SetViewport;
    r.viewport.x = (float)s.viewportX;
    r.viewport.y = (float)s.viewportY;
    r.viewport.w = (float)s.viewportW;
    r.viewport.h = (float)s.viewportH;
    r.viewport.minDepth = 0.0f;
    r.viewport.maxDepth = 1.0f;
    push(r);
    counters.viewports++;
}

void VulkanEmitter::onScissor(const RSXState& s) {
    maybeBeginFrame(s);
    VkRecord r{};
    r.op = VkOp::SetScissor;
    r.scissor.x = (int32_t)s.scissorX;
    r.scissor.y = (int32_t)s.scissorY;
    r.scissor.w = s.scissorW;
    r.scissor.h = s.scissorH;
    push(r);
    counters.scissors++;
}

void VulkanEmitter::onClearSurface(const RSXState& s, uint32_t mask) {
    maybeBeginFrame(s);
    VkRecord r{};
    r.op = VkOp::ClearAttachment;
    r.clear.mask    = mask;
    r.clear.color   = s.colorClearValue;
    r.clear.depth   = 1.0f;
    r.clear.stencil = 0;
    push(r);
    counters.clears++;
}

void VulkanEmitter::onBeginEnd(const RSXState& s, uint32_t prim) {
    if (prim == 0) return; // END marker — drawing is emitted on DRAW_ARRAYS
    maybeBeginFrame(s);
    maybeBindPipeline(s, prim);
}

void VulkanEmitter::onDrawArrays(const RSXState& s, uint32_t first, uint32_t count) {
    maybeBeginFrame(s);
    maybeBindPipeline(s, (uint32_t)s.currentPrim);

    VkRecord r{};
    r.op = VkOp::Draw;
    r.draw.firstVertex = first;
    r.draw.vertexCount = count;
    r.draw.prim        = (uint32_t)s.currentPrim;
    push(r);
    counters.draws++;
}

void VulkanEmitter::onDrawIndexed(const RSXState& s, uint32_t first,
                                  uint32_t count, uint32_t indexFormat) {
    maybeBeginFrame(s);
    maybeBindPipeline(s, (uint32_t)s.currentPrim);

    VkRecord r{};
    r.op = VkOp::DrawIndexed;
    r.drawIndexed.firstIndex  = first;
    r.drawIndexed.indexCount  = count;
    r.drawIndexed.prim        = (uint32_t)s.currentPrim;
    r.drawIndexed.indexFormat = indexFormat;
    push(r);
    counters.drawsIndexed++;
}

void VulkanEmitter::onFlip(const RSXState& /*s*/, uint32_t surfaceOffset) {
    if (inRenderPass_) {
        VkRecord e{};
        e.op = VkOp::EndRenderPass;
        e.endRenderPass.frameIndex = frameIndex_;
        push(e);
        counters.endRenderPasses++;
        inRenderPass_ = false;
    }
    VkRecord p{};
    p.op = VkOp::Present;
    p.present.surfaceOffset = surfaceOffset;
    p.present.frameIndex    = frameIndex_;
    push(p);
    counters.presents++;
    frameIndex_++;
    lastPipelineHash_ = 0; // force re-bind next frame
}

void VulkanEmitter::dump() const {
    printf("┌─ Vulkan command stream (%zu records) ─\n", records_.size());
    for (size_t i = 0; i < records_.size(); i++) {
        const VkRecord& r = records_[i];
        switch (r.op) {
        case VkOp::BeginFrame:
            printf("│ [%4zu] BeginFrame %ux%u fmt=%u\n", i,
                   r.beginFrame.width, r.beginFrame.height, r.beginFrame.format);
            break;
        case VkOp::SetViewport:
            printf("│ [%4zu] SetViewport %.0fx%.0f@(%.0f,%.0f)\n", i,
                   r.viewport.w, r.viewport.h, r.viewport.x, r.viewport.y);
            break;
        case VkOp::SetScissor:
            printf("│ [%4zu] SetScissor %ux%u@(%d,%d)\n", i,
                   r.scissor.w, r.scissor.h, r.scissor.x, r.scissor.y);
            break;
        case VkOp::BindPipeline:
            printf("│ [%4zu] BindPipeline prim=%u hash=0x%08x\n", i,
                   r.bindPipeline.prim, r.bindPipeline.pipelineHash);
            break;
        case VkOp::ClearAttachment:
            printf("│ [%4zu] Clear mask=0x%x color=0x%08x\n", i,
                   r.clear.mask, r.clear.color);
            break;
        case VkOp::Draw:
            printf("│ [%4zu] Draw prim=%u first=%u count=%u\n", i,
                   r.draw.prim, r.draw.firstVertex, r.draw.vertexCount);
            break;
        case VkOp::DrawIndexed:
            printf("│ [%4zu] DrawIndexed prim=%u first=%u count=%u\n", i,
                   r.drawIndexed.prim, r.drawIndexed.firstIndex, r.drawIndexed.indexCount);
            break;
        case VkOp::EndRenderPass:
            printf("│ [%4zu] EndRenderPass frame=%u\n", i, r.endRenderPass.frameIndex);
            break;
        case VkOp::Present:
            printf("│ [%4zu] Present frame=%u surface=0x%x\n", i,
                   r.present.frameIndex, r.present.surfaceOffset);
            break;
        default:
            printf("│ [%4zu] %s\n", i, opName(r.op));
            break;
        }
    }
    printf("└ frames=%u beginFrame=%u clears=%u draws=%u pipelines=%u presents=%u\n",
           frameIndex_, counters.beginFrame, counters.clears,
           counters.draws, counters.bindPipelines, counters.presents);
}

} // namespace rsx
