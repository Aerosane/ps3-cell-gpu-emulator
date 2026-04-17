#pragma once
// rsx_soft_replayer.h — Software replayer for VkRecord streams.
//
// Takes a VulkanEmitter's recorded command stream and renders to an
// in-memory RGBA framebuffer. No Vulkan driver involved — proves the
// command stream carries enough information to produce real pixels
// and provides a deterministic test substrate for the translator.
//
// Not a full rasterizer: triangles fill their axis-aligned bounding
// box with the current clear color XOR 0x00FFFFFF (so draws are
// visibly distinct from clears). Enough to validate ordering and
// regions on a per-frame basis.

#include "rsx_vulkan_emitter.h"
#include <cstdint>
#include <string>
#include <vector>

namespace rsx {

class SoftReplayer {
public:
    struct Stats {
        uint32_t framesRendered{0};
        uint32_t pixelsCleared{0};
        uint32_t pixelsDrawn{0};
    };

    SoftReplayer(uint32_t width, uint32_t height);

    // Replay a recorded stream. Returns number of frames produced
    // (one per Present op). The final frame is retained in framebuffer().
    uint32_t replay(const VulkanEmitter& emitter);

    // Access the current framebuffer (RGBA8, width*height pixels).
    const std::vector<uint32_t>& framebuffer() const { return fb_; }
    uint32_t width()  const { return w_; }
    uint32_t height() const { return h_; }
    const Stats& stats() const { return stats_; }

    // Save the current framebuffer as a PPM image (binary P6).
    bool savePPM(const std::string& path) const;

private:
    void applyClear(uint32_t color);
    void applyDraw(uint32_t prim, uint32_t vertexCount);
    void fillRect(int32_t x0, int32_t y0, int32_t x1, int32_t y1, uint32_t color);

    uint32_t w_, h_;
    std::vector<uint32_t> fb_;
    // Current pipeline context
    uint32_t clearColor_{0xFF000000};
    int32_t  viewportX_{0}, viewportY_{0};
    uint32_t viewportW_{0}, viewportH_{0};
    uint32_t drawIdx_{0}; // number of draws since last clear — spreads boxes
    Stats    stats_;
};

} // namespace rsx
