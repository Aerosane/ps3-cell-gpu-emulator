// test_gcm_prims.cu — PPC-driven multi-primitive frame
//
// Same pipeline as test_gcm_frame, but the PPC program issues THREE
// draw_arrays calls in one frame with different primitive types:
//   - TRIANGLES: one solid triangle upper-left
//   - TRIANGLE_STRIP: a quad (4 verts → 2 tris) on the right
//   - QUADS: another quad on the bottom
// Then flips. Verifies bridge's primitive expansion (strip, quads).

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

static uint32_t ppc_addi (int rD, int rA, int16_t imm) { return (14u<<26) | ((uint32_t)rD<<21) | ((uint32_t)rA<<16) | (uint16_t)imm; }
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
    std::printf("  PPC multi-primitive frame (TRIANGLES + STRIP + QUADS)\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint32_t W = 320, H = 240;
    constexpr uint64_t kLoadAddr = 0x10000;

    constexpr uint32_t VRAM_BYTES = 2u * 1024u * 1024u;
    std::vector<uint8_t> vram(VRAM_BYTES, 0);

    // VBs staged past surface clear region (H*pitchA=307200).
    constexpr uint32_t VB_POS = 0x100000;
    constexpr uint32_t VB_COL = 0x101000;

    // 3 + 4 + 4 + 4 + 4 = 19 vertices across 5 primitive types.
    float positions[19 * 3] = {
        // triangle — upper-left red
          20.f,  60.f, 0.5f,
         100.f,  60.f, 0.5f,
          60.f,  20.f, 0.5f,
        // triangle strip quad — right side, green
         200.f, 200.f, 0.5f,
         300.f, 200.f, 0.5f,
         200.f, 100.f, 0.5f,
         300.f, 100.f, 0.5f,
        // quad — lower-left blue
          20.f, 220.f, 0.5f,
         120.f, 220.f, 0.5f,
         120.f, 150.f, 0.5f,
          20.f, 150.f, 0.5f,
        // triangle fan — center yellow (4 verts = 2 tris off pivot)
         160.f, 120.f, 0.5f,
         200.f, 100.f, 0.5f,
         220.f, 140.f, 0.5f,
         180.f, 150.f, 0.5f,
        // lines — 2 segments (4 verts), cyan
         150.f,  30.f, 0.5f,
         260.f,  30.f, 0.5f,
         150.f,  50.f, 0.5f,
         260.f,  50.f, 0.5f,
    };
    std::memcpy(vram.data() + VB_POS, positions, sizeof(positions));

    // BGRA stored → little-endian uint32 per vertex.
    uint32_t cols[19] = {
        0xFFFF0000u, 0xFFFF0000u, 0xFFFF0000u,
        0xFF00FF00u, 0xFF00FF00u, 0xFF00FF00u, 0xFF00FF00u,
        0xFF0000FFu, 0xFF0000FFu, 0xFF0000FFu, 0xFF0000FFu,
        0xFFFFFF00u, 0xFFFFFF00u, 0xFFFFFF00u, 0xFFFFFF00u,
        0xFF00FFFFu, 0xFF00FFFFu, 0xFF00FFFFu, 0xFF00FFFFu,
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
    emit_draw(PRIM_TRIANGLES,      0, 3);
    emit_draw(PRIM_TRIANGLE_STRIP, 3, 4);
    emit_draw(PRIM_QUADS,          7, 4);
    emit_draw(PRIM_TRIANGLE_FAN,   11, 4);
    emit_draw(PRIM_LINES,          15, 4);

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
    std::printf("  ring cursor = %u dwords\n", cursor);
    std::vector<uint32_t> fifo(cursor, 0);
    megakernel_read_mem(PS3_GCM_FIFO_BASE + 4, fifo.data(), cursor * 4);
    megakernel_shutdown();

    CudaRasterizer raster;
    raster.init(W, H);
    RasterBridge bridge;
    bridge.attach(&raster);
    bridge.setVRAM(vram.data(), VRAM_BYTES);
    RSXState rs; rsx_init(&rs); rs.vulkanEmitter = &bridge;

    rsx_process_fifo(&rs, fifo.data(), cursor, vram.data(), cursor);

    std::printf("\n  bridge counters: surf=%u clears=%u draws=%u flips=%u\n",
                bridge.counters.surfaceSetups, bridge.counters.clears,
                bridge.counters.draws, bridge.counters.flips);
    CHECK(bridge.counters.draws == 5, "All five draw_arrays dispatched");
    CHECK(bridge.counters.flips == 1, "FLIP dispatched");

    std::vector<uint32_t> fb(W * H, 0);
    raster.readbackPlane(0, fb.data());

    uint32_t red = 0, green = 0, blue = 0, yellow = 0, cyan = 0, clear = 0;
    for (uint32_t i = 0; i < W * H; ++i) {
        uint32_t p = fb[i];
        if ((p & 0x00FFFFFFu) == 0x00202020u) { ++clear; continue; }
        uint8_t R = (p >> 16) & 0xFF, G = (p >> 8) & 0xFF, B = p & 0xFF;
        if (R > 200 && G <  60 && B <  60) ++red;
        if (R <  60 && G > 200 && B <  60) ++green;
        if (R <  60 && G <  60 && B > 200) ++blue;
        if (R > 200 && G > 200 && B <  60) ++yellow;
        if (R <  60 && G > 200 && B > 200) ++cyan;
    }
    std::printf("  pixels: clear=%u red=%u green=%u blue=%u yellow=%u cyan=%u\n",
                clear, red, green, blue, yellow, cyan);
    CHECK(red   > 500,  "TRIANGLES primitive rasterized (red region)");
    CHECK(green > 3000, "TRIANGLE_STRIP quad rasterized (green region)");
    CHECK(blue  > 3000, "QUADS primitive rasterized (blue region)");
    CHECK(yellow > 100, "TRIANGLE_FAN rasterized (yellow region)");
    CHECK(cyan   > 50,  "LINES rasterized (cyan segments)");

    raster.shutdown();
    rsx_shutdown(&rs);

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
