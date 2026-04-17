// rsx_soft_replayer.cpp — software replayer impl
#include "rsx_soft_replayer.h"

#include <algorithm>
#include <cstdio>

namespace rsx {

SoftReplayer::SoftReplayer(uint32_t width, uint32_t height)
    : w_(width), h_(height), fb_(width * height, 0xFF000000u) {
    viewportW_ = width;
    viewportH_ = height;
}

void SoftReplayer::fillRect(int32_t x0, int32_t y0, int32_t x1, int32_t y1, uint32_t color) {
    x0 = std::max(0, std::min((int32_t)w_, x0));
    x1 = std::max(0, std::min((int32_t)w_, x1));
    y0 = std::max(0, std::min((int32_t)h_, y0));
    y1 = std::max(0, std::min((int32_t)h_, y1));
    for (int32_t y = y0; y < y1; y++) {
        uint32_t* row = fb_.data() + (size_t)y * w_;
        for (int32_t x = x0; x < x1; x++) {
            row[x] = color;
            stats_.pixelsDrawn++;
        }
    }
}

void SoftReplayer::applyClear(uint32_t color) {
    clearColor_ = color;
    std::fill(fb_.begin(), fb_.end(), color);
    stats_.pixelsCleared += (uint32_t)fb_.size();
    drawIdx_ = 0;
}

void SoftReplayer::applyDraw(uint32_t /*prim*/, uint32_t vertexCount) {
    // Synthesize a deterministic bounding box per draw. Boxes tile across
    // the viewport so multi-draw frames are visibly distinct.
    uint32_t cols = 4, rows = 4;
    uint32_t cell = drawIdx_++ % (cols * rows);
    uint32_t cx = cell % cols;
    uint32_t cy = cell / cols;

    int32_t cellW = viewportW_ / cols;
    int32_t cellH = viewportH_ / rows;
    int32_t pad = std::max(2, std::min(cellW, cellH) / 6);

    int32_t x0 = viewportX_ + cx * cellW + pad;
    int32_t x1 = viewportX_ + (cx + 1) * cellW - pad;
    int32_t y0 = viewportY_ + cy * cellH + pad;
    int32_t y1 = viewportY_ + (cy + 1) * cellH - pad;

    // Color: clear XOR 0x00FFFFFF, modulated by vertex count so draws differ
    uint32_t color = (clearColor_ ^ 0x00FFFFFFu);
    color = (color & 0xFF000000u) | ((color + vertexCount * 0x112233u) & 0x00FFFFFFu);
    fillRect(x0, y0, x1, y1, color);
}

uint32_t SoftReplayer::replay(const VulkanEmitter& emitter) {
    uint32_t frames = 0;
    for (size_t i = 0; i < emitter.size(); i++) {
        const VkRecord& r = emitter[i];
        switch (r.op) {
        case VkOp::BeginFrame:
            // Framebuffer-sized "attachment" — we don't resize on the fly
            // because the test harness owns the resolution. Just reset draw
            // counter so boxes tile from cell 0 each frame.
            drawIdx_ = 0;
            break;
        case VkOp::SetViewport:
            viewportX_ = (int32_t)r.viewport.x;
            viewportY_ = (int32_t)r.viewport.y;
            viewportW_ = (uint32_t)r.viewport.w;
            viewportH_ = (uint32_t)r.viewport.h;
            break;
        case VkOp::SetScissor:
            // Tracked but not enforced (bounding box already clipped)
            break;
        case VkOp::BindPipeline:
            // No-op for soft replayer
            break;
        case VkOp::ClearAttachment:
            applyClear(r.clear.color);
            break;
        case VkOp::Draw:
            applyDraw(r.draw.prim, r.draw.vertexCount);
            break;
        case VkOp::DrawIndexed:
            applyDraw(r.drawIndexed.prim, r.drawIndexed.indexCount);
            break;
        case VkOp::EndRenderPass:
            break;
        case VkOp::Present:
            frames++;
            stats_.framesRendered++;
            break;
        default:
            break;
        }
    }
    return frames;
}

bool SoftReplayer::savePPM(const std::string& path) const {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fprintf(f, "P6\n%u %u\n255\n", w_, h_);
    // Framebuffer is RGBA little-endian (as uint32_t 0xAARRGGBB). Emit RGB.
    std::vector<uint8_t> row(w_ * 3);
    for (uint32_t y = 0; y < h_; y++) {
        const uint32_t* src = fb_.data() + (size_t)y * w_;
        for (uint32_t x = 0; x < w_; x++) {
            uint32_t c = src[x];
            row[x * 3 + 0] = (uint8_t)((c >> 16) & 0xFF); // R
            row[x * 3 + 1] = (uint8_t)((c >>  8) & 0xFF); // G
            row[x * 3 + 2] = (uint8_t)((c >>  0) & 0xFF); // B
        }
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
    return true;
}

} // namespace rsx
