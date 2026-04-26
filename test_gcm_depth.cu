// test_gcm_depth.cu — PPC-driven depth test
//
// PPC binary enables depth test + LESS + depth-write, then draws:
//   tri A: near (z=0.2), fully GREEN, covers region
//   tri B: far  (z=0.8), fully RED,   covers the SAME region, issued AFTER A
// With depth test on, tri A (near) must win: region stays green.
// Without depth test (control case later), B would overwrite A.
//
// Also draws a separate isolated red triangle elsewhere to ensure RED
// pixels do appear somewhere — proves the depth test rejecting pixels
// is due to depth, not a generic path bug.

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

#include "ppc_defs.h"
#include "rsx_defs.h"
#include "rsx_raster.h"
#include "rsx_raster_bridge.h"

using namespace ppc;
using namespace rsx;

extern "C" {
    int   megakernel_init();
    int   megakernel_load(uint64_t offset, const void* data, size_t size);
    int   megakernel_set_entry(uint64_t pc, uint64_t sp, uint64_t toc);
    float megakernel_run(uint32_t maxCycles);
    int   megakernel_read_state(PPEState* out);
    int   megakernel_read_mem(uint64_t offset, void* dst, size_t size);
    void  megakernel_shutdown();
}
namespace rsx {
    int  rsx_init(RSXState* state);
    int  rsx_process_fifo(RSXState* state, const uint32_t* fifo, uint32_t fifoSize,
                          uint8_t* vram, uint32_t maxCmds);
    void rsx_shutdown(RSXState* state);
}

static uint32_t ppc_addis(int rD, int rA, int16_t imm) { return (15u<<26) | ((uint32_t)rD<<21) | ((uint32_t)rA<<16) | (uint16_t)imm; }
static uint32_t ppc_ori  (int rA, int rS, uint16_t imm){ return (24u<<26) | ((uint32_t)rS<<21) | ((uint32_t)rA<<16) | imm; }
static uint32_t ppc_sc   () { return (17u<<26) | (1u<<1); }
static void emit_li32(std::vector<uint32_t>& c, int rD, uint32_t val) {
    c.push_back(ppc_addis(rD, 0, (int16_t)(uint16_t)(val >> 16)));
    c.push_back(ppc_ori  (rD, rD, (uint16_t)val));
}
static void emit_method(std::vector<uint32_t>& c, uint32_t method, uint32_t data) {
    emit_li32(c, 3, method);
    emit_li32(c, 4, data);
    emit_li32(c, 11, 0xC710u);
    c.push_back(ppc_sc());
}

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  PPC depth-test scene (near/far occlusion)\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint32_t W = 320, H = 240;
    constexpr uint64_t kLoadAddr = 0x10000;

    constexpr uint32_t VRAM_BYTES = 2u * 1024u * 1024u;
    std::vector<uint8_t> vram(VRAM_BYTES, 0);

    constexpr uint32_t VB_POS = 0x100000;
    constexpr uint32_t VB_COL = 0x101000;

    // 9 verts total: 3 near-green overlap, 3 far-red overlap, 3 isolated red.
    float positions[9 * 3] = {
        // Near green triangle (z=0.2)
          80.f, 180.f, 0.2f,
         220.f, 180.f, 0.2f,
         150.f,  60.f, 0.2f,
        // Far red triangle — same footprint (z=0.8)
          80.f, 180.f, 0.8f,
         220.f, 180.f, 0.8f,
         150.f,  60.f, 0.8f,
        // Isolated red triangle (far right strip), z=0.3 — proves red CAN render
         270.f, 220.f, 0.3f,
         310.f, 220.f, 0.3f,
         290.f, 180.f, 0.3f,
    };
    rsx::store_be_floats(vram.data() + VB_POS, positions, sizeof(positions)/4);

    uint32_t cols[9] = {
        0xFF00FF00u, 0xFF00FF00u, 0xFF00FF00u,    // near green
        0xFFFF0000u, 0xFFFF0000u, 0xFFFF0000u,    // far red (occluded)
        0xFFFF0000u, 0xFFFF0000u, 0xFFFF0000u,    // isolated red
    };
    std::memcpy(vram.data() + VB_COL, cols, sizeof(cols));

    std::vector<uint32_t> code;
    code.reserve(512);

    emit_method(code, NV4097_SET_SURFACE_CLIP_HORIZONTAL, W);
    emit_method(code, NV4097_SET_SURFACE_CLIP_VERTICAL,   H);
    emit_method(code, NV4097_SET_SURFACE_PITCH_A,         W * 4);
    emit_method(code, NV4097_SET_SURFACE_COLOR_AOFFSET,   0);
    emit_method(code, NV4097_SET_SURFACE_FORMAT,          SURFACE_A8R8G8B8);
    emit_method(code, NV4097_SET_VIEWPORT_HORIZONTAL, (W << 16));
    emit_method(code, NV4097_SET_VIEWPORT_VERTICAL,   (H << 16));
    emit_method(code, NV4097_SET_COLOR_CLEAR_VALUE,   0xFF202020u);
    emit_method(code, NV4097_CLEAR_SURFACE,           CLEAR_COLOR | CLEAR_DEPTH);

    // Depth pipeline state
    emit_method(code, NV4097_SET_DEPTH_TEST_ENABLE, 1);
    emit_method(code, NV4097_SET_DEPTH_FUNC,        CMP_LESS);
    emit_method(code, NV4097_SET_DEPTH_MASK,        1);

    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 0 * 4,
                (12u << 8) | (3u << 4) | VERTEX_F);
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0 * 4, VB_POS);
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 3 * 4,
                (4u << 8) | (4u << 4) | VERTEX_UB);
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 3 * 4, VB_COL);

    auto emit_draw = [&](uint32_t prim, uint32_t first, uint32_t count) {
        emit_method(code, NV4097_SET_BEGIN_END, prim);
        emit_method(code, NV4097_DRAW_ARRAYS, ((count - 1) << 24) | first);
        emit_method(code, NV4097_SET_BEGIN_END, 0u);
    };
    // Draw order: near green FIRST, then far red. If depth LESS is working,
    // the far-red triangle's overlapping pixels must be rejected.
    emit_draw(PRIM_TRIANGLES, 0, 3);   // near green
    emit_draw(PRIM_TRIANGLES, 3, 3);   // far red (same footprint)
    emit_draw(PRIM_TRIANGLES, 6, 3);   // isolated red

    emit_method(code, NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 0);

    emit_li32(code, 11, 1);
    code.push_back(ppc_sc());

    std::printf("\n  PPC program: %zu instructions\n", code.size());

    std::vector<uint8_t> codeBE(code.size() * 4);
    for (size_t i = 0; i < code.size(); ++i) {
        uint32_t be = __builtin_bswap32(code[i]);
        std::memcpy(&codeBE[i * 4], &be, 4);
    }

    if (!megakernel_init()) { std::printf("  FAIL: megakernel_init\n"); return 1; }
    megakernel_load(kLoadAddr, codeBE.data(), codeBE.size());
    megakernel_set_entry(kLoadAddr, 0x00F00000ULL, 0);
    float ms = megakernel_run(16384);

    PPEState st{}; megakernel_read_state(&st);
    std::printf("  megakernel: %.3f ms  pc=0x%llx  halted=%u\n",
                ms, (unsigned long long)st.pc, st.halted);
    CHECK(st.halted == 1, "PPC program halted");

    uint32_t cursor = 0;
    megakernel_read_mem(PS3_GCM_FIFO_BASE, &cursor, sizeof(cursor));
    std::vector<uint32_t> fifo(cursor, 0);
    megakernel_read_mem(PS3_GCM_FIFO_BASE + 4, fifo.data(), cursor * 4);
    megakernel_shutdown();

    CudaRasterizer raster; raster.init(W, H);
    RasterBridge bridge; bridge.attach(&raster); bridge.setVRAM(vram.data(), VRAM_BYTES);
    RSXState rs; rsx_init(&rs); rs.vulkanEmitter = &bridge;

    rsx_process_fifo(&rs, fifo.data(), cursor, vram.data(), cursor);

    CHECK(rs.depthTestEnable, "RSX depth test enabled from PPC");
    CHECK(rs.depthFunc == CMP_LESS, "RSX depth func = LESS from PPC");
    CHECK(bridge.counters.draws == 3, "3 draws dispatched");

    std::vector<uint32_t> fb(W * H, 0);
    raster.readbackPlane(0, fb.data());

    // Count green vs red pixels in the shared-footprint region (x in [60..240]).
    uint32_t greenOverlap = 0, redOverlap = 0;
    uint32_t redIsolated = 0;
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t p = fb[y * W + x];
            if ((p & 0x00FFFFFFu) == 0x00202020u) continue;
            uint8_t R = (p >> 16) & 0xFF, G = (p >> 8) & 0xFF, B = p & 0xFF;
            bool inOverlap = (x >= 60 && x <= 240 && y >= 50 && y <= 190);
            bool inRightStrip = (x >= 260);
            if (inOverlap) {
                if (G > 200 && R < 60) ++greenOverlap;
                if (R > 200 && G < 60) ++redOverlap;
            } else if (inRightStrip) {
                if (R > 200 && G < 60) ++redIsolated;
            }
        }
    }
    std::printf("  overlap region: green=%u red=%u    isolated red=%u\n",
                greenOverlap, redOverlap, redIsolated);

    CHECK(greenOverlap > 3000, "Near-green triangle visible in overlap");
    CHECK(redOverlap   == 0,   "Far-red triangle fully occluded by depth test");
    CHECK(redIsolated  > 50,   "Isolated red triangle visible (sanity)");

    raster.shutdown();
    rsx_shutdown(&rs);

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
