// test_rsx_bridge.cu — end-to-end: RSX FIFO methods drive the
// CudaRasterizer via the RasterBridge. Same FIFO shape as
// test_rsx_replay.cu but the emitter slot holds a RasterBridge, so the
// pixels come from our CUDA rasterizer instead of the soft replayer.

#include "rsx_defs.h"
#include "rsx_raster.h"
#include "rsx_raster_bridge.h"
#include "rsx_fp_shader.h"

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
    // + 3 UB4 colors into VRAM well past the surface. Configure slots
    // 0 and 3 via FIFO, draw. The bridge should decode from VRAM.
    std::printf("\n── NV40 vertex-array decode:\n");
    bridge.setVertexPool(nullptr, 0);
    bridge.setVRAM(vram, 8 * 1024 * 1024);

    // VRAM layout: surface occupies offsets 0..~0x12B000 (320-wide rows at
    // pitch 5120 × 240 rows). Place vertex buffers past the surface to
    // survive CLEAR_SURFACE.
    static constexpr uint32_t VB_POS1  = 0x200000;
    static constexpr uint32_t VB_COL1  = 0x201000;
    static constexpr uint32_t VB_POS2  = 0x202000;
    static constexpr uint32_t VB_COL2  = 0x203000;

    // Write 3 positions (float3, stride 12) at VB_POS1.
    float positions[9] = {
         40.f,  40.f, 0.f,
        280.f,  40.f, 0.f,
        160.f, 200.f, 0.f,
    };
    std::memcpy(vram + VB_POS1, positions, sizeof(positions));

    // Write 3 colors (UB4, D3DCOLOR = BGRA in memory), stride 4 at VB_COL1.
    // Pure yellow for all three: R=255 G=255 B=0 A=255 → memory BGRA = 00FFFFFF.
    uint32_t bgra_yellow = 0xFFFFFF00u; // B=0x00 G=0xFF R=0xFF A=0xFF
    uint32_t cols[3] = { bgra_yellow, bgra_yellow, bgra_yellow };
    std::memcpy(vram + VB_COL1, cols, sizeof(cols));

    // Clear → draw with freshly configured streams → flip.
    uint32_t fifo2[64];
    size_t m = 0;
    fifo2[m++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1); fifo2[m++] = 0xFF000000u;
    fifo2[m++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);         fifo2[m++] = CLEAR_COLOR;

    // Slot 0: position float3, stride 12.
    fifo2[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 0*4, 1);
    fifo2[m++] = (12u << 8) | (3u << 4) | VERTEX_F;
    fifo2[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0*4, 1);
    fifo2[m++] = VB_POS1;

    // Slot 3: color UB4, stride 4.
    fifo2[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 3*4, 1);
    fifo2[m++] = (4u << 8) | (4u << 4) | VERTEX_UB;
    fifo2[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 3*4, 1);
    fifo2[m++] = VB_COL1;

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

    // ── FIFO blend-state translation ────────────────────────────────
    // Place a second triangle over the first with alpha=128 red. With
    // SRC_ALPHA / ONE_MINUS_SRC_ALPHA the result should be ≈ 50-50
    // mix of red over yellow at the overlap.
    std::printf("\n── FIFO blend-state translation:\n");
    // Overwrite slot-3 colors with half-alpha red (BGRA: B=0 G=0 R=FF A=80)
    uint32_t bgra_red_half = 0x80FF0000u; // B=00 G=00 R=FF A=80
    uint32_t cols2[3] = { bgra_red_half, bgra_red_half, bgra_red_half };
    std::memcpy(vram + VB_COL2, cols2, sizeof(cols2));
    // Second position set, shifted down so overlap center is mixed
    float positions2[9] = {
         40.f, 120.f, 0.f,
        280.f, 120.f, 0.f,
        160.f, 220.f, 0.f,
    };
    std::memcpy(vram + VB_POS2, positions2, sizeof(positions2));

    uint32_t fifo4[64];
    m = 0;
    fifo4[m++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1); fifo4[m++] = 0xFF000000u;
    fifo4[m++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);         fifo4[m++] = CLEAR_COLOR | CLEAR_DEPTH;
    fifo4[m++] = fifo_incr(NV4097_SET_DEPTH_TEST_ENABLE, 1); fifo4[m++] = 0;
    fifo4[m++] = fifo_incr(NV4097_SET_CULL_FACE_ENABLE, 1);  fifo4[m++] = 0;

    // First draw: opaque yellow (slots 0+3 already point at VB_POS1/VB_COL1)
    fifo4[m++] = fifo_incr(NV4097_SET_BLEND_ENABLE, 1);      fifo4[m++] = 0;
    fifo4[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo4[m++] = PRIM_TRIANGLES;
    fifo4[m++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);           fifo4[m++] = (2u << 24) | 0;
    fifo4[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo4[m++] = 0;

    // Retarget vertex arrays to positions2 / cols2, enable blend.
    fifo4[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0*4, 1);
    fifo4[m++] = VB_POS2;
    fifo4[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 3*4, 1);
    fifo4[m++] = VB_COL2;
    fifo4[m++] = fifo_incr(NV4097_SET_BLEND_ENABLE, 1);      fifo4[m++] = 1;
    fifo4[m++] = fifo_incr(NV4097_SET_BLEND_FUNC_SFACTOR, 1);
    fifo4[m++] = 0x03020302; // SRC_ALPHA for RGB + A
    fifo4[m++] = fifo_incr(NV4097_SET_BLEND_FUNC_DFACTOR, 1);
    fifo4[m++] = 0x03030303; // ONE_MINUS_SRC_ALPHA for RGB + A
    fifo4[m++] = fifo_incr(NV4097_SET_BLEND_EQUATION, 1);
    fifo4[m++] = 0x80068006; // ADD,ADD
    fifo4[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo4[m++] = PRIM_TRIANGLES;
    fifo4[m++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);           fifo4[m++] = (2u << 24) | 0;
    fifo4[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo4[m++] = 0;
    fifo4[m++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1); fifo4[m++] = 0;
    rsx_process_fifo(&st, fifo4, (uint32_t)m, vram, 4096);

    raster.readback(fb.data());
    // Sample in the overlap region (y ≈ 150, x ≈ 160)
    uint32_t mixPx = fb[150 * W + 160];
    uint8_t r = (mixPx >> 16) & 0xFF;
    uint8_t g = (mixPx >>  8) & 0xFF;
    uint8_t b =  mixPx        & 0xFF;
    std::printf("  blend-overlap px: 0x%08x  (R=%u G=%u B=%u)\n", mixPx, r, g, b);
    // Expect R=255 (src fills in via SRC_ALPHA*255 + OMSA*255), G ≈ 127
    // (src alpha*0 + OMSA*255 ≈ 128), B = 0.
    CHECK(r > 220 && g > 100 && g < 160 && b < 32,
          "Alpha-blend: red over yellow ≈ orange at overlap");
    raster.savePPM("/tmp/rsx_bridge_blend.ppm");


    // ── FIFO vertex-program interpreter ─────────────────────────────
    // Upload a 2-instruction VP:
    //   MUL o[0], in[0], c[0]   ; scale position
    //   MOV o[1], in[3]         ; pass-through color   (end)
    // Set c[0] = (0.5, 0.5, 1, 1) and re-draw the slot-0/3 triangle from
    // VB_POS1/VB_COL1. The triangle (40,40)-(280,40)-(160,200) should
    // shrink to (20,20)-(140,20)-(80,100).
    std::printf("\n── FIFO vertex-program interpreter:\n");

    // Rebuild original yellow vertex data (cols2 overwrote slot-3 memory
    // for the blend scenario, but slot-0 offset is still VB_POS2 from the
    // blend FIFO — reset both arrays).
    // NOTE: VB_POS1/VB_COL1 already hold the yellow triangle from scenario 2.
    uint32_t fifo5[128];
    m = 0;
    fifo5[m++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1); fifo5[m++] = 0xFF000000u;
    fifo5[m++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);         fifo5[m++] = CLEAR_COLOR | CLEAR_DEPTH;
    fifo5[m++] = fifo_incr(NV4097_SET_DEPTH_TEST_ENABLE, 1); fifo5[m++] = 0;
    fifo5[m++] = fifo_incr(NV4097_SET_BLEND_ENABLE, 1);      fifo5[m++] = 0;
    fifo5[m++] = fifo_incr(NV4097_SET_CULL_FACE_ENABLE, 1);  fifo5[m++] = 0;

    // Point slot 0 back to VB_POS1 (scenario 4 had moved it to VB_POS2).
    fifo5[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0*4, 1);
    fifo5[m++] = VB_POS1;
    fifo5[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 3*4, 1);
    fifo5[m++] = VB_COL1;

    // Upload VP: start at instruction 0.
    fifo5[m++] = fifo_incr(NV4097_SET_TRANSFORM_PROGRAM_START, 1);
    fifo5[m++] = 0;
    // 2 instructions = 8 dwords.
    fifo5[m++] = fifo_incr(NV4097_SET_TRANSFORM_PROGRAM, 8);
    // Instr 1: MUL o[0], in[0].xyzw, c[0].xyzw
    fifo5[m++] = 0x20000000u;
    fifo5[m++] = 0x0080000Du;
    fifo5[m++] = 0x8106C0C0u;
    fifo5[m++] = 0x0001E000u;
    // Instr 2: MOV o[1], in[3].xyzw  (end)
    fifo5[m++] = 0x20000000u;
    fifo5[m++] = 0x0040030Du;
    fifo5[m++] = 0x87000000u;
    fifo5[m++] = 0x0001E005u;

    // Upload c[0] = (0.5, 0.5, 1.0, 1.0).
    // (No explicit LOAD handler in Phase 4a — SET_TRANSFORM_CONSTANT
    // writes relative to its own method offset, so starting at c[0].)
    fifo5[m++] = fifo_incr(NV4097_SET_TRANSFORM_CONSTANT, 4);
    {
        float cv[4] = { 0.5f, 0.5f, 1.0f, 1.0f };
        uint32_t* cu = reinterpret_cast<uint32_t*>(cv);
        fifo5[m++] = cu[0]; fifo5[m++] = cu[1];
        fifo5[m++] = cu[2]; fifo5[m++] = cu[3];
    }

    // Draw.
    fifo5[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo5[m++] = PRIM_TRIANGLES;
    fifo5[m++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);           fifo5[m++] = (2u << 24) | 0;
    fifo5[m++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo5[m++] = 0;
    fifo5[m++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1); fifo5[m++] = 0;
    rsx_process_fifo(&st, fifo5, (uint32_t)m, vram, 4096);

    raster.readback(fb.data());
    // After scaling by (0.5, 0.5), triangle spans roughly x=[20,140] y=[20,100].
    // Sample inside scaled triangle at (80, 40).
    uint32_t vpInsidePx  = fb[40 * W + 80];
    // Sample outside scaled triangle but inside the original (160, 100).
    uint32_t vpOutsidePx = fb[100 * W + 160];
    std::printf("  vp inside  (80,40)  : 0x%08x (want yellow)\n", vpInsidePx);
    std::printf("  vp outside (160,100): 0x%08x (want clear/black)\n", vpOutsidePx);
    CHECK(((vpInsidePx  >> 16) & 0xFF) > 200 &&
          ((vpInsidePx  >>  8) & 0xFF) > 200 &&
          ( vpInsidePx         & 0xFF) < 32,
          "VP-scaled triangle visible at shrunk interior");
    CHECK(((vpOutsidePx >> 16) & 0xFF) < 32 &&
          ((vpOutsidePx >>  8) & 0xFF) < 32,
          "VP-scaled triangle no longer covers original outer region");
    raster.savePPM("/tmp/rsx_bridge_vp.ppm");


    // ── Fragment-program interpreter unit test ──────────────────────
    // Build a minimal FP directly in memory:
    //   MUL r0.xyzw, f[COL0].xyzw, c[0].xyzw ; end
    // With f[COL0] = (1, 1, 0, 1) and c[0] = (0.5, 0.25, 0.125, 1.0) the
    // result in r0 must be (0.5, 0.25, 0, 1).
    std::printf("\n── FP interpreter:\n");
    auto fp_enc = [](uint32_t v) -> uint32_t {
        // fp_swap_word is its own inverse — swap byte pairs so the decoder
        // undoes it when it reads the word back.
        return ((v & 0x00FF00FF) << 8) | ((v & 0xFF00FF00) >> 8);
    };

    // w0 (OPDEST): opcode=FP_MUL(2)<<24 | inputAttr=1<<13 | mask xyzw
    //              | dstReg=0<<1 | endFlag=1
    uint32_t w0 = 1u | (0xFu << 9) | (1u << 13) | (0x02u << 24);
    // w1 (SRC0): regType=INPUT(1), swz xyzw (0,1,2,3)
    uint32_t w1 = 1u | (0u << 2) | (0u << 9) | (1u << 11) | (2u << 13) | (3u << 15);
    // w2 (SRC1): regType=CONSTANT(2), swz xyzw, opcodeHi=0
    uint32_t w2 = 2u | (0u << 2) | (0u << 9) | (1u << 11) | (2u << 13) | (3u << 15);
    // w3 (SRC2): unused (regType=0 => temp 0 — ignored by MUL)
    uint32_t w3 = 0;

    uint32_t fpProg[8];
    fpProg[0] = fp_enc(w0);
    fpProg[1] = fp_enc(w1);
    fpProg[2] = fp_enc(w2);
    fpProg[3] = fp_enc(w3);
    // Inline constant vector for c[0] (follows the instruction because
    // SRC1.regType == CONSTANT). Each component is byte-swapped.
    float cvals[4] = { 0.5f, 0.25f, 0.125f, 1.0f };
    uint32_t* cwords = reinterpret_cast<uint32_t*>(cvals);
    fpProg[4] = fp_enc(cwords[0]);
    fpProg[5] = fp_enc(cwords[1]);
    fpProg[6] = fp_enc(cwords[2]);
    fpProg[7] = fp_enc(cwords[3]);

    FPFloat4 fpInputs[16] = {};
    fpInputs[1] = FPFloat4{{1.0f, 1.0f, 0.0f, 1.0f}};
    FPFloat4 fpOutputs[4] = {};
    fp_execute(fpProg, 8, fpInputs, fpOutputs);
    std::printf("  r0 = (%.3f, %.3f, %.3f, %.3f)\n",
                fpOutputs[0].v[0], fpOutputs[0].v[1],
                fpOutputs[0].v[2], fpOutputs[0].v[3]);
    CHECK(fpOutputs[0].v[0] > 0.49f && fpOutputs[0].v[0] < 0.51f &&
          fpOutputs[0].v[1] > 0.24f && fpOutputs[0].v[1] < 0.26f &&
          fpOutputs[0].v[2] < 0.01f &&
          fpOutputs[0].v[3] > 0.99f && fpOutputs[0].v[3] < 1.01f,
          "FP MUL with inline constant produces (0.5, 0.25, 0, 1)");


    // ── FP texture-sample (TEX) ─────────────────────────────────────
    // Build `TEX r0, f[TEX0]; end` on texture unit 3. Sampler callback
    // returns the concatenated uv and fixed 0.5, 1.0 so we can also
    // verify the tex-unit argument is propagated correctly.
    std::printf("\n── FP TEX sample:\n");
    // w0: opcode=TEX(0x17)<<24 | inputAttr=4(TEX0)<<13 | mask xyzw
    //     | dstReg=0<<1 | texUnit=3<<17 | endFlag=1
    uint32_t tw0 = 1u | (0xFu << 9) | (4u << 13) | (3u << 17) | (0x17u << 24);
    // w1 (SRC0): regType=INPUT(1), swz xyzw
    uint32_t tw1 = 1u | (0u << 2) | (0u << 9) | (1u << 11) | (2u << 13) | (3u << 15);
    uint32_t tw2 = 0;
    uint32_t tw3 = 0;

    uint32_t texProg[4] = { fp_enc(tw0), fp_enc(tw1), fp_enc(tw2), fp_enc(tw3) };

    struct SamplerCtx { uint32_t seenUnit; float seenU, seenV; };
    SamplerCtx ctx{0, 0, 0};
    auto sampler = [](void* ud, uint32_t unit, const float uvw[3], float rgba[4]) {
        SamplerCtx* c = static_cast<SamplerCtx*>(ud);
        c->seenUnit = unit;
        c->seenU = uvw[0];
        c->seenV = uvw[1];
        rgba[0] = uvw[0];  // R = u
        rgba[1] = uvw[1];  // G = v
        rgba[2] = 0.5f;
        rgba[3] = 1.0f;
    };

    FPFloat4 texIn[16] = {};
    texIn[4] = FPFloat4{{0.75f, 0.25f, 0.0f, 1.0f}};  // TEX0 = (.75, .25)
    FPFloat4 texOut[4] = {};
    fp_execute(texProg, 4, texIn, texOut, sampler, &ctx);
    std::printf("  sampler unit=%u uv=(%.3f, %.3f)\n",
                ctx.seenUnit, ctx.seenU, ctx.seenV);
    std::printf("  r0 = (%.3f, %.3f, %.3f, %.3f)\n",
                texOut[0].v[0], texOut[0].v[1],
                texOut[0].v[2], texOut[0].v[3]);
    CHECK(ctx.seenUnit == 3 &&
          ctx.seenU > 0.74f && ctx.seenU < 0.76f &&
          ctx.seenV > 0.24f && ctx.seenV < 0.26f,
          "FP TEX forwarded (unit, u, v) to sampler callback");
    CHECK(texOut[0].v[0] > 0.74f && texOut[0].v[0] < 0.76f &&
          texOut[0].v[1] > 0.24f && texOut[0].v[1] < 0.26f &&
          texOut[0].v[2] > 0.49f && texOut[0].v[2] < 0.51f &&
          texOut[0].v[3] > 0.99f,
          "FP TEX wrote sampled rgba into r0");


    raster.shutdown();
    rsx_shutdown(&st);
    std::free(vram);

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
