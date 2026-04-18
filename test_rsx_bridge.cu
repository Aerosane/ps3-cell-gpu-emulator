// test_rsx_bridge.cu — end-to-end: RSX FIFO methods drive the
// CudaRasterizer via the RasterBridge. Same FIFO shape as
// test_rsx_replay.cu but the emitter slot holds a RasterBridge, so the
// pixels come from our CUDA rasterizer instead of the soft replayer.

#include "rsx_defs.h"
#include "rsx_raster.h"
#include "rsx_raster_bridge.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace rsx;

namespace rsx {
    int  rsx_init(RSXState* state);
    int  rsx_process_fifo(RSXState* state, const uint32_t* fifo, uint32_t fifoSize,
                          uint8_t* vram, uint32_t maxCmds);
    void rsx_shutdown(RSXState* state);
}

static int fails = 0;
#define CHECK(c, m) do { if (c) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

static uint32_t fifo_incr(uint32_t method, uint32_t count) {
    return ((count & 0x7FF) << 18) | ((method >> 2) << 2);
}

int main() {
    std::printf("═══════════════════════════════════════════════════\n");
    std::printf("  RSX → RasterBridge → CudaRasterizer pipeline\n");
    std::printf("═══════════════════════════════════════════════════\n\n");

    const uint32_t W = 320, H = 240;

    RSXState st;
    rsx_init(&st);

    CudaRasterizer raster;
    raster.init(W, H);

    RasterBridge bridge;
    bridge.attach(&raster);
    st.vulkanEmitter = &bridge;

    // Three overlapping triangles in the shared vertex pool.
    std::vector<RasterVertex> pool = {
        { 40.f,  40.f, 0, 1,0,0,1 },
        {280.f,  40.f, 0, 1,0,0,1 },
        {160.f, 200.f, 0, 1,0,0,1 },

        { 60.f,  80.f, 0, 0,1,0,1 },
        {260.f,  80.f, 0, 0,1,0,1 },
        {160.f, 180.f, 0, 0,1,0,1 },

        { 80.f, 100.f, 0, 0,0,1,1 },
        {240.f, 100.f, 0, 0,0,1,1 },
        {160.f, 160.f, 0, 0,0,1,1 },
    };
    bridge.setVertexPool(pool.data(), (uint32_t)pool.size());

    // Build a synthetic "game frame" FIFO.
    uint32_t fifo[256];
    size_t n = 0;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1); fifo[n++] = W;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_CLIP_VERTICAL,   1); fifo[n++] = H;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_FORMAT, 1);          fifo[n++] = SURFACE_A8R8G8B8;
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_HORIZONTAL, 1);     fifo[n++] = (W << 16);
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_VERTICAL,   1);     fifo[n++] = (H << 16);
    fifo[n++] = fifo_incr(NV4097_SET_SCISSOR_HORIZONTAL,  1);     fifo[n++] = (W << 16);
    fifo[n++] = fifo_incr(NV4097_SET_SCISSOR_VERTICAL,    1);     fifo[n++] = (H << 16);
    fifo[n++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE,   1);     fifo[n++] = 0xFF202040u;
    fifo[n++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);               fifo[n++] = CLEAR_COLOR | CLEAR_DEPTH;

    // Three draws: red, green, blue — painter's order.
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);  fifo[n++] = PRIM_TRIANGLES;
    fifo[n++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);    fifo[n++] = (2u << 24) | 0;
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);  fifo[n++] = 0;

    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);  fifo[n++] = PRIM_TRIANGLES;
    fifo[n++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);    fifo[n++] = (2u << 24) | 3;
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);  fifo[n++] = 0;

    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);  fifo[n++] = PRIM_TRIANGLES;
    fifo[n++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);    fifo[n++] = (2u << 24) | 6;
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);  fifo[n++] = 0;

    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1); fifo[n++] = 0;

    uint8_t* vram = (uint8_t*)std::calloc(1, 8 * 1024 * 1024);
    int cmds = rsx_process_fifo(&st, fifo, (uint32_t)n, vram, 2048);
    std::printf("  FIFO processed %d commands\n", cmds);

    std::printf("  bridge counters: surf=%u clears=%u draws=%u flips=%u\n",
                bridge.counters.surfaceSetups,
                bridge.counters.clears,
                bridge.counters.draws,
                bridge.counters.flips);
    CHECK(bridge.counters.surfaceSetups >= 1, "Surface setup hook fired");
    CHECK(bridge.counters.clears == 1,        "Clear hook fired once");
    CHECK(bridge.counters.draws == 3,         "Draw hook fired 3 times");
    CHECK(bridge.counters.flips == 1,         "Flip hook fired");

    // Pixel proof: corner should be clear color, center should be blue.
    std::vector<uint32_t> fb(W * H);
    raster.readback(fb.data());
    std::printf("  corner pixel: 0x%08x (want 0xFF202040)\n", fb[5 * W + 5]);
    CHECK(fb[5 * W + 5] == 0xFF202040u, "FIFO clear visible in framebuffer");

    uint32_t c = fb[130 * W + 160];
    std::printf("  center pixel: 0x%08x (want ~0xFF0000FF)\n", c);
    CHECK(((c >> 16) & 0xFF) < 32 && ((c >> 8) & 0xFF) < 32 && (c & 0xFF) > 200,
          "Painter-order: blue on top");

    raster.savePPM("/tmp/rsx_bridge_demo.ppm");
    std::printf("  Saved /tmp/rsx_bridge_demo.ppm\n");

    // ── NV40 vertex-array decode scenario ───────────────────────────
    // Clear state, attach VRAM, detach pool. Pack 3 float3 positions
    // at VRAM offset 0x1000, 3 UB4 colors at 0x2000. Configure slots
    // 0 and 3 via FIFO, draw. The bridge should decode from VRAM.
    std::printf("\n── NV40 vertex-array decode:\n");
    bridge.setVertexPool(nullptr, 0);
    bridge.setVRAM(vram, 8 * 1024 * 1024);

    // Write 3 positions (float3, stride 12) at offset 0x1000.
    float positions[9] = {
         40.f,  40.f, 0.f,
        280.f,  40.f, 0.f,
        160.f, 200.f, 0.f,
    };
    std::memcpy(vram + 0x1000, positions, sizeof(positions));

    // Write 3 colors (UB4, D3DCOLOR = BGRA in memory), stride 4 at 0x2000.
    // Pure yellow for all three: R=255 G=255 B=0 A=255 → memory BGRA = 00FFFFFF.
    uint32_t bgra_yellow = 0xFFFFFF00u; // B=0x00 G=0xFF R=0xFF A=0xFF
    uint32_t cols[3] = { bgra_yellow, bgra_yellow, bgra_yellow };
    std::memcpy(vram + 0x2000, cols, sizeof(cols));

    // Clear → draw with freshly configured streams → flip.
    uint32_t fifo2[64];
    size_t m = 0;
    fifo2[m++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1); fifo2[m++] = 0xFF000000u;
    fifo2[m++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);         fifo2[m++] = CLEAR_COLOR;

    // Slot 0: position float3, stride 12.
    fifo2[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 0*4, 1);
    fifo2[m++] = (12u << 8) | (3u << 4) | VERTEX_F;
    fifo2[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0*4, 1);
    fifo2[m++] = 0x1000u;

    // Slot 3: color UB4, stride 4.
    fifo2[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 3*4, 1);
    fifo2[m++] = (4u << 8) | (4u << 4) | VERTEX_UB;
    fifo2[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 3*4, 1);
    fifo2[m++] = 0x2000u;

    fifo2[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1); fifo2[m++] = PRIM_TRIANGLES;
    fifo2[m++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);   fifo2[m++] = (2u << 24) | 0;
    fifo2[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1); fifo2[m++] = 0;
    fifo2[m++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1); fifo2[m++] = 0;

    uint32_t drawsBefore = bridge.counters.draws;
    rsx_process_fifo(&st, fifo2, (uint32_t)m, vram, 2048);

    CHECK(bridge.counters.draws == drawsBefore + 1,
          "Decode path dispatched a draw");

    raster.readback(fb.data());
    uint32_t yellowPx = fb[130 * W + 160];
    std::printf("  VRAM-decoded tri center: 0x%08x (want yellow)\n", yellowPx);
    CHECK(((yellowPx >> 16) & 0xFF) > 200 &&
          ((yellowPx >>  8) & 0xFF) > 200 &&
          ( yellowPx        & 0xFF) < 32,
          "Decoded VRAM verts + UB4 color produced yellow triangle");
    CHECK(fb[5 * W + 5] == 0xFF000000u, "Clear color outside triangle");

    raster.savePPM("/tmp/rsx_bridge_vram.ppm");

    // ── FIFO pipeline-state translation ─────────────────────────────
    // Push CULL_FACE_ENABLE + CULL_FACE=BACK. With this triangle's winding
    // the rasterizer classifies it as back-facing (screen-space y-down),
    // so BACK culling drops it and the center pixel stays clear.
    std::printf("\n── FIFO pipeline-state translation:\n");
    uint32_t fifo3[32];
    m = 0;
    fifo3[m++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1); fifo3[m++] = 0xFF112233u;
    fifo3[m++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);         fifo3[m++] = CLEAR_COLOR;
    fifo3[m++] = fifo_incr(NV4097_SET_CULL_FACE_ENABLE, 1);  fifo3[m++] = 1;
    fifo3[m++] = fifo_incr(NV4097_SET_CULL_FACE, 1);         fifo3[m++] = 0x0405; // BACK
    fifo3[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo3[m++] = PRIM_TRIANGLES;
    fifo3[m++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);           fifo3[m++] = (2u << 24) | 0;
    fifo3[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo3[m++] = 0;
    fifo3[m++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1); fifo3[m++] = 0;
    rsx_process_fifo(&st, fifo3, (uint32_t)m, vram, 2048);

    raster.readback(fb.data());
    uint32_t culledPx = fb[130 * W + 160];
    std::printf("  cull-back center: 0x%08x (want clear 0xFF112233)\n", culledPx);
    CHECK(culledPx == 0xFF112233u,
          "Triangle culled when FIFO sets CULL_FACE=BACK");

    // Flip to FRONT — triangle reappears (it's back-facing, BACK was the cull).
    m = 0;
    fifo3[m++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1); fifo3[m++] = 0xFF112233u;
    fifo3[m++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);         fifo3[m++] = CLEAR_COLOR;
    fifo3[m++] = fifo_incr(NV4097_SET_CULL_FACE, 1);         fifo3[m++] = 0x0404; // FRONT
    fifo3[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo3[m++] = PRIM_TRIANGLES;
    fifo3[m++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);           fifo3[m++] = (2u << 24) | 0;
    fifo3[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo3[m++] = 0;
    fifo3[m++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1); fifo3[m++] = 0;
    rsx_process_fifo(&st, fifo3, (uint32_t)m, vram, 2048);

    raster.readback(fb.data());
    uint32_t visPx = fb[130 * W + 160];
    std::printf("  cull-front center: 0x%08x (want yellow)\n", visPx);
    CHECK(((visPx >> 16) & 0xFF) > 200 &&
          ((visPx >>  8) & 0xFF) > 200 &&
          ( visPx        & 0xFF) < 32,
          "Triangle visible when FIFO flips CULL_FACE=FRONT");

    raster.shutdown();
    rsx_shutdown(&st);
    std::free(vram);

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
