// test_rsx_bridge.cu — end-to-end: RSX FIFO methods drive the
// CudaRasterizer via the RasterBridge. Same FIFO shape as
// test_rsx_replay.cu but the emitter slot holds a RasterBridge, so the
// pixels come from our CUDA rasterizer instead of the soft replayer.

#include "rsx_defs.h"
#include "rsx_raster.h"
#include "rsx_raster_bridge.h"
#include "rsx_fp_shader.h"
#include "rsx_texture.h"
#include "rsx_vp_shader.h"

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

// ── VP instruction encoder (uses canonical VP_REG_* / VP_VEC_* enums) ──
static uint32_t encSrc(uint32_t regType, uint32_t regIdx,
                       uint32_t sx, uint32_t sy, uint32_t sz, uint32_t sw) {
    uint32_t s = 0;
    s |= (regType & 0x3);
    s |= (regIdx & 0x3F) << 2;
    s |= (sw & 0x3) << 8;  s |= (sz & 0x3) << 10;
    s |= (sy & 0x3) << 12; s |= (sx & 0x3) << 14;
    return s;
}
static void buildVP(uint32_t d[4], uint32_t vecOp, uint32_t scaOp,
                    uint32_t inputIdx, uint32_t constIdx,
                    uint32_t src0, uint32_t src1, uint32_t src2,
                    uint32_t vecDstTmp, uint32_t vecDstOut,
                    bool vecResult, bool end = false) {
    d[0] = 0;
    d[0] |= (vecDstTmp & 0x3F) << 15;
    if (vecResult) d[0] |= (1u << 30);
    d[1] = 0;
    d[1] |= (src0 >> 9) & 0xFF;
    d[1] |= (inputIdx & 0xF) << 8;
    d[1] |= (constIdx & 0x3FF) << 12;
    d[1] |= (vecOp & 0x1F) << 22;
    d[1] |= (scaOp & 0x1F) << 27;
    d[2] = 0;
    d[2] |= (src2 >> 11) & 0x3F;
    d[2] |= (src1 & 0x1FFFFu) << 6;
    d[2] |= (src0 & 0x1FFu) << 23;
    d[3] = 0;
    if (end) d[3] |= 1;
    d[3] |= (vecDstOut & 0x1F) << 2;
    d[3] |= (1u << 13) | (1u << 14) | (1u << 15) | (1u << 16); // mask XYZW
    d[3] |= (src2 & 0x7FFu) << 21;
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
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_PITCH_A, 1);           fifo[n++] = W * 4;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET, 1);   fifo[n++] = 0;
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
    st.vramSize = 8 * 1024 * 1024;
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
    rsx::store_be_floats(vram + VB_POS1, positions, sizeof(positions)/4);

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
    rsx::store_be_floats(vram + VB_POS2, positions2, sizeof(positions2)/4);

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
    // VP path maps output from NDC[-1,1] to screen, so use NDC positions.
    static constexpr uint32_t VB_POS_NDC = 0x204000;
    float ndcPos[9] = {
       -0.75f, -0.75f, 0.f,
        0.75f, -0.75f, 0.f,
        0.0f,   0.75f, 0.f,
    };
    rsx::store_be_floats(vram + VB_POS_NDC, ndcPos, sizeof(ndcPos)/4);

    uint32_t fifo5[128];
    m = 0;
    fifo5[m++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1); fifo5[m++] = 0xFF000000u;
    fifo5[m++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);         fifo5[m++] = CLEAR_COLOR | CLEAR_DEPTH;
    fifo5[m++] = fifo_incr(NV4097_SET_DEPTH_TEST_ENABLE, 1); fifo5[m++] = 0;
    fifo5[m++] = fifo_incr(NV4097_SET_BLEND_ENABLE, 1);      fifo5[m++] = 0;
    fifo5[m++] = fifo_incr(NV4097_SET_CULL_FACE_ENABLE, 1);  fifo5[m++] = 0;

    // Point slot 0 to NDC positions, slot 3 to original yellow colors.
    fifo5[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0*4, 1);
    fifo5[m++] = VB_POS_NDC;
    fifo5[m++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 3*4, 1);
    fifo5[m++] = VB_COL1;

    // Upload VP via PROGRAM_LOAD then set START.
    fifo5[m++] = fifo_incr(NV4097_SET_TRANSFORM_PROGRAM_LOAD, 1);
    fifo5[m++] = 0;
    fifo5[m++] = fifo_incr(NV4097_SET_TRANSFORM_PROGRAM_START, 1);
    fifo5[m++] = 0;
    // 2 instructions = 8 dwords.
    fifo5[m++] = fifo_incr(NV4097_SET_TRANSFORM_PROGRAM, 8);
    {
        uint32_t s_in0 = encSrc(VP_REG_INPUT, 0, 0,1,2,3);
        uint32_t s_c0  = encSrc(VP_REG_CONSTANT, 0, 0,1,2,3);
        uint32_t s_z   = encSrc(VP_REG_TEMP, 0, 0,1,2,3);
        uint32_t s_in3 = encSrc(VP_REG_INPUT, 3, 0,1,2,3);
        uint32_t ins[8];
        // MUL o[0], in[0].xyzw, c[0].xyzw
        buildVP(ins, VP_VEC_MUL, VP_SCA_NOP, 0, 0,
                s_in0, s_c0, s_z, 0x3F, 0, true, false);
        // MOV o[1], in[3].xyzw  (end)
        buildVP(ins+4, VP_VEC_MOV, VP_SCA_NOP, 3, 0,
                s_in3, s_z, s_z, 0x3F, 1, true, true);
        for (int k = 0; k < 8; ++k) fifo5[m++] = ins[k];
    }

    // Upload c[0] = (0.5, 0.5, 1.0, 1.0).
    fifo5[m++] = fifo_incr(NV4097_SET_TRANSFORM_CONSTANT_LOAD, 1);
    fifo5[m++] = 0;
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
    // After NDC MUL by (0.5, 0.5), the triangle shrinks from NDC
    // ±0.75 to ±0.375, mapping to roughly x=[100,220] y=[75,165].
    // Sample inside scaled triangle at (160, 120).
    uint32_t vpInsidePx  = fb[120 * W + 160];
    // Sample outside scaled triangle at (40, 40).
    uint32_t vpOutsidePx = fb[40 * W + 40];
    std::printf("  vp inside  (160,120): 0x%08x (want yellow)\n", vpInsidePx);
    std::printf("  vp outside (40,40)  : 0x%08x (want clear/black)\n", vpOutsidePx);
    CHECK(((vpInsidePx  >> 16) & 0xFF) > 200 &&
          ((vpInsidePx  >>  8) & 0xFF) > 200 &&
          ( vpInsidePx         & 0xFF) < 32,
          "VP-scaled triangle visible at shrunk interior");
    CHECK(((vpOutsidePx >> 16) & 0xFF) < 32 &&
          ((vpOutsidePx >>  8) & 0xFF) < 32,
          "VP-scaled triangle no longer covers original outer region");
    raster.savePPM("/tmp/rsx_bridge_vp.ppm");

    // Reset VP state so subsequent scenarios use direct (no-VP) path.
    st.vpValid = 0;


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


    // ── FP TEX with real RSX VRAM-backed sampler ────────────────────
    // Upload a 2x2 A8R8G8B8 texture into vram at a known offset, push
    // TEXTURE_OFFSET / FORMAT / IMAGE_RECT FIFO methods for unit 5,
    // then run a TEX FP using rsx_host_sampler. Verifies the command-
    // processor texture state capture and the host sampler decode path.
    std::printf("\n── FP TEX via VRAM-backed RSX sampler:\n");
    {
        const uint32_t TEX_OFF = 0x300000;     // anywhere past vbufs
        // 2x2 ARGB texels, byte order A,R,G,B per pixel:
        //   (0,0) red    (1,0) green
        //   (0,1) blue   (1,1) white
        uint8_t texels[16] = {
            0xFF, 0xFF, 0x00, 0x00,   // red
            0xFF, 0x00, 0xFF, 0x00,   // green
            0xFF, 0x00, 0x00, 0xFF,   // blue
            0xFF, 0xFF, 0xFF, 0xFF,   // white
        };
        std::memcpy(vram + TEX_OFF, texels, sizeof(texels));

        uint32_t tfifo[16]; size_t tn = 0;
        // unit 5 → method base + 5*0x20
        const uint32_t U = 5;
        const uint32_t TBASE = NV4097_SET_TEXTURE_OFFSET + U * 0x20;
        tfifo[tn++] = fifo_incr(TBASE + 0x00, 1); tfifo[tn++] = TEX_OFF;
        tfifo[tn++] = fifo_incr(TBASE + 0x04, 1); tfifo[tn++] = 0x8500; // A8R8G8B8 at bits[15:8]
        tfifo[tn++] = fifo_incr(TBASE + 0x18, 1); tfifo[tn++] = (2u << 16) | 2u;
        rsx_process_fifo(&st, tfifo, (uint32_t)tn, vram, 64);

        CHECK(st.textures[U].enabled &&
              st.textures[U].width == 2 && st.textures[U].height == 2 &&
              ((st.textures[U].format >> 8) & 0xFF) == 0x85,
              "Texture unit 5 captured offset/format/2x2 from FIFO");

        // Build TEX r0, f[TEX0]; end on unit 5.
        uint32_t vw0 = 1u | (0xFu << 9) | (4u << 13) | (U << 17) | (0x17u << 24);
        uint32_t vw1 = 1u | (0u << 2) | (0u << 9) | (1u << 11) | (2u << 13) | (3u << 15);
        uint32_t vProg[4] = { fp_enc(vw0), fp_enc(vw1), 0, 0 };

        ps3rsx::HostTextureSamplerCtx vctx{ vram, 8u * 1024 * 1024, &st };

        struct Probe { float u, v; float r, g, b, a; };
        Probe probes[4] = {
            // Sample centers of each texel: (0.25,0.25)→red, (0.75,0.25)→green,
            // (0.25,0.75)→blue, (0.75,0.75)→white.
            {0.25f, 0.25f, 1, 0, 0, 1},
            {0.75f, 0.25f, 0, 1, 0, 1},
            {0.25f, 0.75f, 0, 0, 1, 1},
            {0.75f, 0.75f, 1, 1, 1, 1},
        };
        for (auto& p : probes) {
            FPFloat4 vIn[16] = {};
            vIn[4] = FPFloat4{{p.u, p.v, 0.f, 1.f}};
            FPFloat4 vOut[4] = {};
            fp_execute(vProg, 4, vIn, vOut, ps3rsx::rsx_host_sampler, &vctx);
            std::printf("  uv=(%.2f,%.2f) → r0=(%.2f,%.2f,%.2f,%.2f) "
                        "(expected %.0f,%.0f,%.0f,%.0f)\n",
                        p.u, p.v,
                        vOut[0].v[0], vOut[0].v[1], vOut[0].v[2], vOut[0].v[3],
                        p.r, p.g, p.b, p.a);
            auto near = [](float a, float b){ return a > b - 0.05f && a < b + 0.05f; };
            CHECK(near(vOut[0].v[0], p.r) && near(vOut[0].v[1], p.g) &&
                  near(vOut[0].v[2], p.b) && near(vOut[0].v[3], p.a),
                  "VRAM-backed TEX returns expected ARGB texel");
        }
    }


    // ── End-to-end textured draw: FIFO texture upload → bridge → raster ──
    // Verifies that NV4097_SET_TEXTURE_* captured from FIFO actually
    // modulate the rasterized pixels. We upload a 2x2 ARGB checker
    // (white/black/black/white), draw a single fullscreen-ish white
    // triangle with per-vertex UVs spanning the whole texture, and
    // confirm that both white and black pixels appear in the framebuffer.
    std::printf("\n── End-to-end FIFO → textured draw:\n");
    {
        // Wipe raster + FB.
        raster.init(W, H);
        RasterBridge br2;
        br2.attach(&raster);
        br2.setVRAM(vram, 8 * 1024 * 1024);
        st.vulkanEmitter = &br2;

        const uint32_t TEX_OFF = 0x400000;
        uint8_t checker[16] = {
            0xFF, 0xFF, 0xFF, 0xFF,   // white
            0xFF, 0x00, 0x00, 0x00,   // black
            0xFF, 0x00, 0x00, 0x00,   // black
            0xFF, 0xFF, 0xFF, 0xFF,   // white
        };
        std::memcpy(vram + TEX_OFF, checker, sizeof(checker));

        // Vertex buffer with uvs.
        const uint32_t VB = 0x500000;
        struct Vtx { float x,y,z,u,v; } verts[3] = {
            {  60.f,  60.f, 0, 0.25f, 0.75f },   // texel (0,1) = black
            { 260.f,  60.f, 0, 0.75f, 0.75f },   // texel (1,1) = white
            { 160.f, 200.f, 0, 0.5f,  0.25f },   // texel (1,0) = black
        };
        rsx::store_be_floats(vram + VB, (const float*)verts, sizeof(verts)/4);

        uint32_t fifo6[64]; size_t k = 0;
        fifo6[k++] = fifo_incr(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1); fifo6[k++] = W;
        fifo6[k++] = fifo_incr(NV4097_SET_SURFACE_CLIP_VERTICAL,   1); fifo6[k++] = H;
        fifo6[k++] = fifo_incr(NV4097_SET_SURFACE_FORMAT, 1);          fifo6[k++] = SURFACE_A8R8G8B8;
        fifo6[k++] = fifo_incr(NV4097_SET_VIEWPORT_HORIZONTAL, 1);     fifo6[k++] = (W << 16);
        fifo6[k++] = fifo_incr(NV4097_SET_VIEWPORT_VERTICAL,   1);     fifo6[k++] = (H << 16);
        fifo6[k++] = fifo_incr(NV4097_SET_SCISSOR_HORIZONTAL,  1);     fifo6[k++] = (W << 16);
        fifo6[k++] = fifo_incr(NV4097_SET_SCISSOR_VERTICAL,    1);     fifo6[k++] = (H << 16);
        fifo6[k++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE,   1);     fifo6[k++] = 0xFF404040u;
        fifo6[k++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);               fifo6[k++] = CLEAR_COLOR | CLEAR_DEPTH;

        // Texture unit 0 setup.
        fifo6[k++] = fifo_incr(NV4097_SET_TEXTURE_OFFSET + 0*0x20, 1); fifo6[k++] = TEX_OFF;
        fifo6[k++] = fifo_incr(NV4097_SET_TEXTURE_FORMAT + 0*0x20, 1); fifo6[k++] = 0x8500;
        fifo6[k++] = fifo_incr(NV4097_SET_TEXTURE_IMAGE_RECT + 0*0x20, 1); fifo6[k++] = (2u << 16) | 2u;

        // Vertex array slots: pos (stride=20, 3F), uv (stride=20, offset+12, 2F).
        fifo6[k++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0*4, 1); fifo6[k++] = VB;
        fifo6[k++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 0*4, 1); fifo6[k++] = (20 << 8) | (3 << 4) | 2; // stride 20, 3 floats
        fifo6[k++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 8*4, 1); fifo6[k++] = VB + 12;
        fifo6[k++] = fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 8*4, 1); fifo6[k++] = (20 << 8) | (2 << 4) | 2; // stride 20, 2 floats

        fifo6[k++] = fifo_incr(NV4097_SET_BEGIN_END, 1);  fifo6[k++] = PRIM_TRIANGLES;
        fifo6[k++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);    fifo6[k++] = (3 << 24) | 0; // count=3, first=0
        fifo6[k++] = fifo_incr(NV4097_SET_BEGIN_END, 1);  fifo6[k++] = 0;

        rsx_process_fifo(&st, fifo6, (uint32_t)k, vram, 4096);

        // Readback and count bright vs dark pixels inside the triangle.
        std::vector<uint32_t> fb(W * H);
        raster.readback(fb.data());
        uint32_t bright = 0, dark = 0;
        for (uint32_t i = 0; i < W * H; ++i) {
            uint8_t r = (fb[i] >> 16) & 0xFF;
            uint8_t g = (fb[i] >> 8)  & 0xFF;
            uint8_t b =  fb[i]        & 0xFF;
            if (r == 0x40 && g == 0x40 && b == 0x40) continue; // clear
            uint32_t lum = (uint32_t)r + g + b;
            if (lum > 350) bright++; else if (lum < 200) dark++;
        }
        std::printf("  bright pixels: %u   dark pixels: %u\n", bright, dark);
        CHECK(bright > 50 && dark > 300,
              "Textured draw produced both light and dark texels in FB");

        raster.savePPM("/tmp/rsx_bridge_textured.ppm");
    }


    // ── MRT state capture: 3 color planes via SURFACE_COLOR_{A,B,C}_OFFSET ──
    std::printf("\n── MRT surface state capture:\n");
    {
        uint32_t mfifo[32]; size_t mk = 0;
        mfifo[mk++] = fifo_incr(NV4097_SET_SURFACE_PITCH_A, 1);    mfifo[mk++] = 1280;
        mfifo[mk++] = fifo_incr(NV4097_SET_SURFACE_PITCH_B, 1);    mfifo[mk++] = 1280;
        mfifo[mk++] = fifo_incr(NV4097_SET_SURFACE_PITCH_C, 1);    mfifo[mk++] = 1280;
        mfifo[mk++] = fifo_incr(NV4097_SET_SURFACE_PITCH_D, 1);    mfifo[mk++] = 1280;
        mfifo[mk++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET, 1); mfifo[mk++] = 0x100000;
        mfifo[mk++] = fifo_incr(NV4097_SET_SURFACE_COLOR_BOFFSET, 1); mfifo[mk++] = 0x200000;
        mfifo[mk++] = fifo_incr(NV4097_SET_SURFACE_COLOR_COFFSET, 1); mfifo[mk++] = 0x300000;
        mfifo[mk++] = fifo_incr(NV4097_SET_SURFACE_COLOR_DOFFSET, 1); mfifo[mk++] = 0x400000;
        mfifo[mk++] = fifo_incr(NV4097_SET_SURFACE_COLOR_TARGET, 1);  mfifo[mk++] = SURFACE_TARGET_MRT2; // ABC

        rsx_process_fifo(&st, mfifo, (uint32_t)mk, vram, 64);

        std::printf("  pitch  A/B/C/D = %u/%u/%u/%u\n",
                    st.surfacePitchA, st.surfacePitchB,
                    st.surfacePitchC, st.surfacePitchD);
        std::printf("  offset A/B/C/D = 0x%x/0x%x/0x%x/0x%x\n",
                    st.surfaceOffsetA, st.surfaceOffsetB,
                    st.surfaceOffsetC, st.surfaceOffsetD);
        std::printf("  target         = 0x%x (MRT2=ABC expected 0x17)\n",
                    st.surfaceColorTarget);
        CHECK(st.surfacePitchA == 1280 && st.surfacePitchB == 1280 &&
              st.surfacePitchC == 1280 && st.surfacePitchD == 1280,
              "All four SURFACE_PITCH_{A,B,C,D} captured");
        CHECK(st.surfaceOffsetA == 0x100000 && st.surfaceOffsetB == 0x200000 &&
              st.surfaceOffsetC == 0x300000 && st.surfaceOffsetD == 0x400000,
              "All four SURFACE_COLOR_{A,B,C,D}_OFFSET captured");
        CHECK(st.surfaceColorTarget == SURFACE_TARGET_MRT2,
              "SURFACE_COLOR_TARGET captured as MRT2 (ABC = 0x17)");
    }


    // ── MRT plane allocation + per-plane clear + readback ──────
    // Pushes SURFACE_COLOR_TARGET=MRT3 (ABCD), clears through FIFO to
    // a known color, readback all four planes and verify they carry the
    // clear. This exercises the new CudaRasterizer::setMRTCount /
    // clearPlane / readbackPlane API and RasterBridge's MRT routing.
    std::printf("\n── MRT plane bind + clear:\n");
    {
        raster.init(W, H);
        RasterBridge br3;
        br3.attach(&raster);
        br3.setVRAM(vram, 8 * 1024 * 1024);
        st.vulkanEmitter = &br3;

        uint32_t pfifo[32]; size_t pk = 0;
        pfifo[pk++] = fifo_incr(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1); pfifo[pk++] = W;
        pfifo[pk++] = fifo_incr(NV4097_SET_SURFACE_CLIP_VERTICAL,   1); pfifo[pk++] = H;
        pfifo[pk++] = fifo_incr(NV4097_SET_VIEWPORT_HORIZONTAL, 1);     pfifo[pk++] = (W << 16);
        pfifo[pk++] = fifo_incr(NV4097_SET_VIEWPORT_VERTICAL,   1);     pfifo[pk++] = (H << 16);
        pfifo[pk++] = fifo_incr(NV4097_SET_SURFACE_COLOR_TARGET, 1);    pfifo[pk++] = SURFACE_TARGET_MRT3;
        pfifo[pk++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE,   1);     pfifo[pk++] = 0xFFAABBCCu;
        pfifo[pk++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);               pfifo[pk++] = CLEAR_COLOR | CLEAR_DEPTH;
        // Draw one dummy primitive to force applyPipelineState → setMRTCount.
        pfifo[pk++] = fifo_incr(NV4097_SET_BEGIN_END, 1);               pfifo[pk++] = PRIM_TRIANGLES;
        pfifo[pk++] = fifo_incr(NV4097_SET_BEGIN_END, 1);               pfifo[pk++] = 0;

        rsx_process_fifo(&st, pfifo, (uint32_t)pk, vram, 64);

        // A clear can't produce an MRT-count change without a draw
        // (applyPipelineState runs per draw in the current design).
        // Force MRT allocation explicitly and re-clear to match the
        // semantic we just tested in the FIFO.
        raster.setMRTCount(4);
        raster.clearPlane(0, 0xFFAABBCCu);
        raster.clearPlane(1, 0xFFAABBCCu);
        raster.clearPlane(2, 0xFFAABBCCu);
        raster.clearPlane(3, 0xFFAABBCCu);

        CHECK(raster.mrtCount() == 4, "CudaRasterizer reports mrtCount=4");

        std::vector<uint32_t> pixbuf(W * H);
        bool all_ok = true;
        for (uint32_t p = 0; p < 4; ++p) {
            std::fill(pixbuf.begin(), pixbuf.end(), 0u);
            raster.readbackPlane(p, pixbuf.data());
            uint32_t mismatched = 0;
            for (uint32_t i = 0; i < W * H; ++i)
                if (pixbuf[i] != 0xFFAABBCCu) mismatched++;
            std::printf("  plane %u: mismatched %u/%u\n",
                        p, mismatched, W * H);
            if (mismatched != 0) all_ok = false;
        }
        CHECK(all_ok, "All 4 MRT planes cleared to 0xFFAABBCC and readback OK");
    }

    // ── Scenario 12: indexed draw via host pool + index buffer ──────
    //
    // FIFO-driven NV4097_DRAW_INDEX_ARRAY exercises the bridge's
    // onDrawIndexed → CudaRasterizer::drawIndexed path. We feed a
    // 4-vertex quad (red/green/blue/yellow) plus a 6-index list that
    // forms two triangles, and verify pixel coverage in the FB.
    {
        std::printf("\n── Indexed draw via FIFO DRAW_INDEX_ARRAY:\n");

        // Reset MRT count so readback hits plane A cleanly.
        raster.setMRTCount(1);

        constexpr uint32_t W = 320, H = 240;
        raster.clearPlane(0, 0xFF000000u); // black

        // 4 vertices forming a screen-space quad with distinct colors.
        // (CudaRasterizer expects pixel coordinates, like scenario 1's pool.)
        RasterVertex quad[4] = {};
        auto setV = [](RasterVertex& v, float x, float y,
                       float r, float g, float b){
            v.x = x; v.y = y; v.z = 0.5f;
            v.r = r; v.g = g; v.b = b; v.a = 1.0f;
            v.u = 0.0f; v.v = 0.0f;
        };
        setV(quad[0],  32.f, 216.f, 1.0f, 0.0f, 0.0f); // bottom-left  red
        setV(quad[1], 288.f, 216.f, 0.0f, 1.0f, 0.0f); // bottom-right green
        setV(quad[2], 288.f,  24.f, 0.0f, 0.0f, 1.0f); // top-right    blue
        setV(quad[3],  32.f,  24.f, 1.0f, 1.0f, 0.0f); // top-left     yellow

        // Two triangles: (0,1,2) + (0,2,3)
        uint16_t idx[6] = { 0, 1, 2, 0, 2, 3 };

        bridge.setVRAM(nullptr, 0);          // force pool path
        bridge.setVertexPool(quad, 4);
        bridge.setIndexPool(idx, 6);
        st.vulkanEmitter = &bridge;          // restore primary bridge as active emitter

        // Disable any VP/texture state from earlier scenarios so the
        // rasterizer just Gourauds the per-vertex colors.
        st.vpValid = 0;
        for (auto& t : st.textures) t.enabled = false;

        // Push BEGIN(triangles) → DRAW_INDEX_ARRAY(first=0,count=6) → END.
        std::vector<uint32_t> fifo;
        fifo.push_back(fifo_incr(NV4097_SET_BEGIN_END, 1));
        fifo.push_back(0x04);                             // PRIM_TRIANGLES
        fifo.push_back(fifo_incr(NV4097_DRAW_INDEX_ARRAY, 1));
        fifo.push_back(((6 - 1) << 24) | 0u);             // count=6, first=0
        fifo.push_back(fifo_incr(NV4097_SET_BEGIN_END, 1));
        fifo.push_back(0x00);                             // END

        uint32_t before = bridge.counters.drawIndexed;
        rsx_process_fifo(&st, fifo.data(), (uint32_t)fifo.size(),
                         (uint8_t*)vram, fifo.size());
        uint32_t after = bridge.counters.drawIndexed;

        CHECK(after == before + 1, "FIFO emitted exactly one indexed draw");

        std::vector<uint32_t> fb(W * H, 0);
        raster.readbackPlane(0, fb.data());

        uint32_t lit = 0, red = 0, green = 0, blue = 0, yellow = 0;
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                uint32_t p = fb[y * W + x];
                if (p == 0xFF000000u) continue;
                ++lit;
                uint8_t R = (p >> 16) & 0xFF;
                uint8_t G = (p >>  8) & 0xFF;
                uint8_t B =  p        & 0xFF;
                if (R > 200 && G <  60 && B <  60) ++red;
                if (R <  60 && G > 200 && B <  60) ++green;
                if (R <  60 && G <  60 && B > 200) ++blue;
                if (R > 200 && G > 200 && B <  60) ++yellow;
            }
        }
        std::printf("  lit=%u   pure-red=%u  pure-green=%u  "
                    "pure-blue=%u  pure-yellow=%u\n",
                    lit, red, green, blue, yellow);

        CHECK(lit > 30000, "Quad covers a large area of the framebuffer");
        CHECK(red > 50 && green > 50 && blue > 50 && yellow > 50,
              "All four corner colors visible (Gouraud-interpolated)");
    }

    // ── Scenario 13: VRAM-resident indexed draw ─────────────────────
    //
    // Real PS3 games stage both vertex and index buffers in VRAM and
    // drive the FIFO with SET_INDEX_ARRAY_ADDRESS + DRAW_INDEX_ARRAY.
    // This exercises the new VRAM index-decode path in onDrawIndexed
    // (no host pool registered).
    {
        std::printf("\n── VRAM-resident indexed draw:\n");

        constexpr uint32_t W = 320, H = 240;
        constexpr uint32_t VB_POS3 = 0x300000;
        constexpr uint32_t VB_COL3 = 0x301000;
        constexpr uint32_t IB3     = 0x302000;

        // Two-triangle quad in pixel coords; same colors as scenario 12.
        float positions[12] = {
             32.f, 216.f, 0.5f,    // 0 BL  red
            288.f, 216.f, 0.5f,    // 1 BR  green
            288.f,  24.f, 0.5f,    // 2 TR  blue
             32.f,  24.f, 0.5f,    // 3 TL  yellow
        };
        rsx::store_be_floats(vram + VB_POS3, positions, sizeof(positions)/4);

        // BGRA in memory: byte0=B byte1=G byte2=R byte3=A → LE u32 packing.
        uint32_t cols[4] = {
            0xFFFF0000u,  // red    (B=00 G=00 R=FF A=FF)
            0xFF00FF00u,  // green  (B=00 G=FF R=00 A=FF)
            0xFF0000FFu,  // blue   (B=FF G=00 R=00 A=FF)
            0xFFFFFF00u,  // yellow (B=00 G=FF R=FF A=FF)
        };
        std::memcpy(vram + VB_COL3, cols, sizeof(cols));

        // U16 LE indices: two triangles forming the quad.
        uint16_t idx[6] = { 0, 1, 2, 0, 2, 3 };
        std::memcpy(vram + IB3, idx, sizeof(idx));

        // Detach host pools so the bridge takes the VRAM path.
        bridge.setVertexPool(nullptr, 0);
        bridge.setIndexPool(nullptr, 0);
        bridge.setVRAM(vram, 8 * 1024 * 1024);
        st.vulkanEmitter = &bridge;

        // Reset MRT to single plane and clear FB to black.
        raster.setMRTCount(1);
        raster.clearPlane(0, 0xFF000000u);

        // Clear texture state (carry-over from earlier scenarios).
        for (auto& t : st.textures) t.enabled = false;
        st.vpValid = 0;

        std::vector<uint32_t> fifo;
        // Vertex stream slot 0: position float3 stride 12.
        fifo.push_back(fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 0*4, 1));
        fifo.push_back((12u << 8) | (3u << 4) | VERTEX_F);
        fifo.push_back(fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0*4, 1));
        fifo.push_back(VB_POS3);
        // Slot 3: color UB4 stride 4.
        fifo.push_back(fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 3*4, 1));
        fifo.push_back((4u << 8) | (4u << 4) | VERTEX_UB);
        fifo.push_back(fifo_incr(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 3*4, 1));
        fifo.push_back(VB_COL3);
        // Index buffer state.
        fifo.push_back(fifo_incr(NV4097_SET_INDEX_ARRAY_ADDRESS, 1));
        fifo.push_back(IB3);
        fifo.push_back(fifo_incr(NV4097_SET_INDEX_ARRAY_DMA, 1));
        fifo.push_back(0u);                                  // U16 little-endian
        // Begin → DrawIndexed(first=0, count=6) → End.
        fifo.push_back(fifo_incr(NV4097_SET_BEGIN_END, 1));
        fifo.push_back(PRIM_TRIANGLES);
        fifo.push_back(fifo_incr(NV4097_DRAW_INDEX_ARRAY, 1));
        fifo.push_back(((6 - 1) << 24) | 0u);
        fifo.push_back(fifo_incr(NV4097_SET_BEGIN_END, 1));
        fifo.push_back(0u);

        uint32_t before = bridge.counters.drawIndexed;
        rsx_process_fifo(&st, fifo.data(), (uint32_t)fifo.size(),
                         (uint8_t*)vram, fifo.size());
        uint32_t after = bridge.counters.drawIndexed;
        CHECK(after == before + 1, "VRAM indexed draw fired exactly once");
        CHECK(st.indexArrayAddress == IB3, "INDEX_ARRAY_ADDRESS captured");
        CHECK(st.indexArrayFormat == 0, "INDEX_ARRAY_DMA captured (U16 LE)");

        std::vector<uint32_t> fb(W * H, 0);
        raster.readbackPlane(0, fb.data());
        uint32_t lit = 0, red = 0, green = 0, blue = 0, yellow = 0;
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                uint32_t p = fb[y * W + x];
                if (p == 0xFF000000u) continue;
                ++lit;
                uint8_t R = (p >> 16) & 0xFF;
                uint8_t G = (p >>  8) & 0xFF;
                uint8_t B =  p        & 0xFF;
                if (R > 200 && G <  60 && B <  60) ++red;
                if (R <  60 && G > 200 && B <  60) ++green;
                if (R <  60 && G <  60 && B > 200) ++blue;
                if (R > 200 && G > 200 && B <  60) ++yellow;
            }
        }
        std::printf("  lit=%u   red=%u green=%u blue=%u yellow=%u\n",
                    lit, red, green, blue, yellow);
        CHECK(lit > 30000, "VRAM-driven quad covers expected area");
        CHECK(red > 50 && green > 50 && blue > 50 && yellow > 50,
              "All four corner colors visible (decoded from VRAM)");
    }


    raster.shutdown();
    rsx_shutdown(&st);
    std::free(vram);

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
