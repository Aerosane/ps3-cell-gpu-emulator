// test_rsx_replay.cu — End-to-end: FIFO → Emitter → SoftReplayer → PPM
//
// Demonstrates the full RSX pipeline producing a deterministic image
// from a synthetic "game frame" (clear + 8 draws + flip). Saves the
// result to /tmp/rsx_demo.ppm and verifies pixel counts.

#include "rsx_defs.h"
#include "rsx_vulkan_emitter.h"
#include "rsx_soft_replayer.h"

#include <cstdio>
#include <cstdlib>

using namespace rsx;

namespace rsx {
    int  rsx_init(RSXState* state);
    int  rsx_process_fifo(RSXState* state, const uint32_t* fifo, uint32_t fifoSize,
                          uint8_t* vram, uint32_t maxCmds);
    void rsx_shutdown(RSXState* state);
}

static int pass = 0, fail = 0;
#define CHECK(c, m) do { if (c) { printf("  ✓ %s\n", m); pass++; } else { printf("  ✗ %s\n", m); fail++; } } while (0)

static uint32_t fifo_incr(uint32_t method, uint32_t count) {
    return ((count & 0x7FF) << 18) | ((method >> 2) << 2);
}

int main() {
    printf("═══════════════════════════════════════════════════\n");
    printf("  RSX full pipeline demo: FIFO → Emitter → Image\n");
    printf("═══════════════════════════════════════════════════\n\n");

    const uint32_t W = 640, H = 360;
    RSXState st;
    rsx_init(&st);
    VulkanEmitter emit;
    st.vulkanEmitter = &emit;

    // Build a synthetic frame
    uint32_t fifo[256];
    size_t n = 0;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_FORMAT, 1);        fifo[n++] = SURFACE_A8R8G8B8;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1); fifo[n++] = W;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_CLIP_VERTICAL,   1); fifo[n++] = H;
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_HORIZONTAL, 1);   fifo[n++] = (W << 16);
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_VERTICAL,   1);   fifo[n++] = (H << 16);
    fifo[n++] = fifo_incr(NV4097_SET_SCISSOR_HORIZONTAL,  1);   fifo[n++] = (W << 16);
    fifo[n++] = fifo_incr(NV4097_SET_SCISSOR_VERTICAL,    1);   fifo[n++] = (H << 16);
    fifo[n++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1);     fifo[n++] = 0xFF103040u;
    fifo[n++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);             fifo[n++] = CLEAR_COLOR | CLEAR_DEPTH;
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);             fifo[n++] = PRIM_TRIANGLES;
    for (int d = 0; d < 8; d++) {
        uint32_t verts = 3 + d * 3; // variety of vertex counts
        fifo[n++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);
        fifo[n++] = ((verts - 1) << 24) | (uint32_t)(d * verts);
    }
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);             fifo[n++] = 0u;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1); fifo[n++] = 0x00100000u;

    uint8_t* vram = (uint8_t*)calloc(1, 16 * 1024 * 1024);
    rsx_process_fifo(&st, fifo, (uint32_t)n, vram, 2048);

    printf("── Emitter output:\n");
    CHECK(emit.counters.presents == 1, "1 Present");
    CHECK(emit.counters.draws    == 8, "8 Draw records");
    CHECK(emit.counters.clears   == 1, "1 Clear record");

    SoftReplayer rep(W, H);
    uint32_t frames = rep.replay(emit);

    printf("\n── Replayer output:\n");
    CHECK(frames == 1, "1 frame rendered");
    CHECK(rep.stats().pixelsCleared == W * H, "clear touched whole framebuffer");
    CHECK(rep.stats().pixelsDrawn   > 0, "draws produced pixels");

    // Check that center pixel of cell (0,0) is a draw color, not clear
    const auto& fb = rep.framebuffer();
    uint32_t px = fb[(H / 8) * W + (W / 8)];
    CHECK(px != 0xFF103040u, "first draw cell overwrote clear color");

    // Save PPM
    bool saved = rep.savePPM("/tmp/rsx_demo.ppm");
    CHECK(saved, "PPM saved to /tmp/rsx_demo.ppm");

    free(vram);
    rsx_shutdown(&st);

    printf("\n═══════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass, fail);
    printf("  Output:  /tmp/rsx_demo.ppm (%ux%u)\n", W, H);
    printf("═══════════════════════════════════════════════════\n");
    return fail ? 1 : 0;
}
