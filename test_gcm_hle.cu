// test_gcm_hle.cu — exercises gcm_hle.h by building a complete FIFO
// using ONLY the cellGcmSet* shim functions (the same shape libgcm
// games would use), then driving the RasterBridge with that buffer.
//
// If this passes we can wire the same builders as libgcm import bodies
// when the ELF loader resolves them.

#include "rsx_defs.h"
#include "rsx_raster.h"
#include "rsx_raster_bridge.h"
#include "gcm_hle.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace rsx;
using namespace gcm;

namespace rsx {
    int  rsx_init(RSXState* state);
    int  rsx_process_fifo(RSXState* state, const uint32_t* fifo, uint32_t fifoSize,
                          uint8_t* vram, uint32_t maxCmds);
    void rsx_shutdown(RSXState* state);
}

static int fails = 0;
#define CHECK(c, m) do { if (c) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  cellGcm HLE shim → FIFO → RasterBridge end-to-end\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint32_t W = 320, H = 240;

    RSXState st;
    rsx_init(&st);

    CudaRasterizer raster;
    raster.init(W, H);

    RasterBridge bridge;
    bridge.attach(&raster);
    st.vulkanEmitter = &bridge;

    // ── Stage geometry + indices in VRAM ──────────────────────
    constexpr uint32_t VRAM_BYTES = 8 * 1024 * 1024;
    uint8_t* vram = (uint8_t*)std::calloc(1, VRAM_BYTES);

    constexpr uint32_t VB_POS = 0x300000;
    constexpr uint32_t VB_COL = 0x301000;
    constexpr uint32_t IB     = 0x302000;
    constexpr uint32_t SURF_A = 0x000000;

    float positions[12] = {
         32.f, 216.f, 0.5f,
        288.f, 216.f, 0.5f,
        288.f,  24.f, 0.5f,
         32.f,  24.f, 0.5f,
    };
    std::memcpy(vram + VB_POS, positions, sizeof(positions));

    // BGRA in memory for VERTEX_UB.
    uint32_t cols[4] = {
        0xFFFF0000u,  // red
        0xFF00FF00u,  // green
        0xFF0000FFu,  // blue
        0xFFFFFF00u,  // yellow
    };
    std::memcpy(vram + VB_COL, cols, sizeof(cols));

    uint16_t idx[6] = { 0, 1, 2, 0, 2, 3 };
    std::memcpy(vram + IB, idx, sizeof(idx));

    bridge.setVRAM(vram, VRAM_BYTES);
    bridge.setVertexPool(nullptr, 0);
    bridge.setIndexPool(nullptr, 0);

    // ── Build the FIFO entirely through the gcm shim ──────────
    std::vector<uint32_t> cmdbuf(256, 0);
    GcmCtx ctx{};
    gcm_init_ctx(&ctx, cmdbuf.data(), (uint32_t)cmdbuf.size());

    cellGcmSetSurface(&ctx, SURFACE_A8R8G8B8, W, H,
                      W * 4, SURF_A,
                      DEPTH_Z24S8, W * 4, 0);
    cellGcmSetViewport(&ctx, 0, 0, W, H);
    cellGcmSetClearColor(&ctx, 0xFF000000u);
    cellGcmSetClearSurface(&ctx, CLEAR_COLOR | CLEAR_DEPTH);

    cellGcmSetVertexDataArray(&ctx, /*slot*/ 0, /*stride*/ 12,
                              /*comps*/ 3, VERTEX_F, VB_POS);
    cellGcmSetVertexDataArray(&ctx, /*slot*/ 3, /*stride*/  4,
                              /*comps*/ 4, VERTEX_UB, VB_COL);

    cellGcmSetDrawIndexArray(&ctx, IB, /*u32*/ false,
                             PRIM_TRIANGLES, /*first*/ 0, /*count*/ 6);

    cellGcmFinish(&ctx, /*ref*/ 0xC0FFEEu);
    cellGcmSetFlip(&ctx, SURF_A);

    const uint32_t fifoWords = gcm_used(&ctx);
    std::printf("\n  GCM emitted %u FIFO dwords\n", fifoWords);
    CHECK(fifoWords > 0 && fifoWords < cmdbuf.size(),
          "GCM builder produced bounded FIFO");

    // ── Drive the bridge ──────────────────────────────────────
    rsx_process_fifo(&st, cmdbuf.data(), fifoWords, vram, fifoWords);

    std::printf("  bridge counters: surf=%u clears=%u draws=%u "
                "drawIndexed=%u flips=%u\n",
                bridge.counters.surfaceSetups, bridge.counters.clears,
                bridge.counters.draws, bridge.counters.drawIndexed,
                bridge.counters.flips);

    CHECK(bridge.counters.clears      >= 1, "FIFO produced a CLEAR");
    CHECK(bridge.counters.drawIndexed == 1, "FIFO produced exactly one indexed draw");
    CHECK(bridge.counters.flips       >= 1, "FIFO produced a FLIP");
    CHECK(st.indexArrayAddress == IB,        "INDEX_ARRAY_ADDRESS captured via GCM");
    CHECK(st.surfaceColorTarget != 0 || st.surfaceFormat == SURFACE_A8R8G8B8,
          "Surface state captured via cellGcmSetSurface");

    // ── Scenario 2: depth/blend/cull/transform-program builders ──
    std::printf("\n  Scenario 2: depth/blend/cull/transform builders\n");
    std::vector<uint32_t> cmd2(512, 0);
    GcmCtx c2{};
    gcm_init_ctx(&c2, cmd2.data(), (uint32_t)cmd2.size());

    cellGcmSetDepthTestEnable(&c2, true);
    cellGcmSetDepthFunc(&c2, 0x0203);          // GL_LEQUAL
    cellGcmSetDepthMask(&c2, true);
    cellGcmSetBlendEnable(&c2, true);
    cellGcmSetBlendFunc(&c2,
                        /*srgb*/0x0302, /*drgb*/0x0303,    // GL_SRC_ALPHA / GL_ONE_MINUS_SRC_ALPHA
                        /*sa*/  0x0001, /*da*/  0x0000);   // GL_ONE / GL_ZERO
    cellGcmSetCullFaceEnable(&c2, true);
    cellGcmSetCullFace(&c2, 0x0405);            // GL_BACK

    // Tiny fake VP: 2 instructions × 4 dwords each
    uint32_t vpInsts[8] = {
        0x401F9C6Cu, 0x0040000Du, 0x8106C083u, 0x6041FFFFu,  // mov o0, v0
        0x401F9C6Cu, 0x0040080Du, 0x8106C083u, 0x6041FFFFu,  // mov o1, v3
    };
    cellGcmSetTransformProgram(&c2, /*loadSlot*/ 0, vpInsts, /*count*/ 2);
    cellGcmSetTransformConstant(&c2, /*slot*/ 0, 1.0f, 2.0f, 3.0f, 4.0f);

    rsx_process_fifo(&st, cmd2.data(), gcm_used(&c2), vram, gcm_used(&c2));

    CHECK(st.depthTestEnable,                    "DEPTH_TEST_ENABLE captured");
    CHECK(st.depthFunc == 0x0203,                "DEPTH_FUNC=GL_LEQUAL captured");
    CHECK(st.depthMask,                          "DEPTH_MASK captured");
    CHECK(st.blendEnable,                        "BLEND_ENABLE captured");
    CHECK((st.blendSFactor & 0xFFFF) == 0x0302,  "BLEND_FUNC sRGB captured");
    CHECK((st.blendDFactor & 0xFFFF) == 0x0303,  "BLEND_FUNC dRGB captured");
    CHECK(st.cullFaceEnable,                     "CULL_FACE_ENABLE captured");
    CHECK(st.cullFace == 0x0405,                 "CULL_FACE=GL_BACK captured");
    CHECK(st.vpValid != 0,                       "Transform program upload valid bit set");
    CHECK(st.vpData[0] == vpInsts[0] &&
          st.vpData[7] == vpInsts[7],            "Transform program first+last word match");

    // Transform constant — VP constants live in vpConstants if present;
    // for now just ensure the FIFO didn't crash and an entry was emitted.
    CHECK(gcm_used(&c2) > 12,                    "Scenario 2 emitted state setters");

    // ── Verify pixels (scenario 1) ────────────────────────────
    raster.setMRTCount(1);
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
    std::printf("  framebuffer: lit=%u  red=%u green=%u blue=%u yellow=%u\n",
                lit, red, green, blue, yellow);
    CHECK(lit > 30000, "Quad covers expected screen area");
    CHECK(red > 50 && green > 50 && blue > 50 && yellow > 50,
          "All four corner colors visible");

    raster.shutdown();
    rsx_shutdown(&st);
    std::free(vram);

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
