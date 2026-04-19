// test_gcm_indexed.cu — PPC-driven indexed draw (DRAW_INDEX_ARRAY).
// Upload 4 vertices forming a quad + 6 U16 indices forming 2 triangles.
// Exercises the VRAM-resident index + vertex path in RasterBridge::onDrawIndexed.

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
    int   megakernel_load(uint64_t, const void*, size_t);
    int   megakernel_set_entry(uint64_t, uint64_t, uint64_t);
    float megakernel_run(uint32_t);
    int   megakernel_read_state(PPEState*);
    int   megakernel_read_mem(uint64_t, void*, size_t);
    void  megakernel_shutdown();
}
namespace rsx {
    int  rsx_init(RSXState*);
    int  rsx_process_fifo(RSXState*, const uint32_t*, uint32_t, uint8_t*, uint32_t);
    void rsx_shutdown(RSXState*);
}

static uint32_t ppc_addis(int rD, int rA, int16_t imm) { return (15u<<26)|((uint32_t)rD<<21)|((uint32_t)rA<<16)|(uint16_t)imm; }
static uint32_t ppc_ori  (int rA, int rS, uint16_t imm){ return (24u<<26)|((uint32_t)rS<<21)|((uint32_t)rA<<16)|imm; }
static uint32_t ppc_sc   () { return (17u<<26)|(1u<<1); }
static void emit_li32(std::vector<uint32_t>& c, int rD, uint32_t v) {
    c.push_back(ppc_addis(rD, 0, (int16_t)(uint16_t)(v>>16)));
    c.push_back(ppc_ori  (rD, rD, (uint16_t)v));
}
static void emit_method(std::vector<uint32_t>& c, uint32_t m, uint32_t d) {
    emit_li32(c, 3, m); emit_li32(c, 4, d); emit_li32(c, 11, 0xC710u); c.push_back(ppc_sc());
}

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  PPC indexed draw (DRAW_INDEX_ARRAY)\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint32_t W = 320, H = 240;
    constexpr uint64_t kLoadAddr = 0x10000;
    constexpr uint32_t VRAM_BYTES = 2u * 1024u * 1024u;
    std::vector<uint8_t> vram(VRAM_BYTES, 0);

    // Quad: 4 shared vertices, 6 indices = 2 triangles.
    // Corners of a central 160x120 rect → 80..240 x, 60..180 y.
    constexpr uint32_t VB_POS = 0x100000;
    constexpr uint32_t VB_COL = 0x101000;
    constexpr uint32_t IB_OFF = 0x102000;
    float quadPos[12] = {
        80.f,  60.f,  0.5f,   // 0 TL
        240.f, 60.f,  0.5f,   // 1 TR
        80.f,  180.f, 0.5f,   // 2 BL
        240.f, 180.f, 0.5f,   // 3 BR
    };
    uint32_t quadCol[4] = {
        0xFFFF0000u, // red
        0xFF00FF00u, // green
        0xFF0000FFu, // blue
        0xFFFFFF00u, // yellow
    };
    uint16_t idx[6] = { 0, 1, 2,  2, 1, 3 };
    std::memcpy(vram.data() + VB_POS, quadPos, sizeof(quadPos));
    std::memcpy(vram.data() + VB_COL, quadCol, sizeof(quadCol));
    std::memcpy(vram.data() + IB_OFF, idx,     sizeof(idx));

    std::vector<uint32_t> code; code.reserve(512);
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

    // Index array: U16, VRAM-resident.
    emit_method(code, NV4097_SET_INDEX_ARRAY_ADDRESS, IB_OFF);
    emit_method(code, NV4097_SET_INDEX_ARRAY_DMA,     0);  // format U16 (bit0=0)

    emit_method(code, NV4097_SET_BEGIN_END, PRIM_TRIANGLES);
    // DRAW_INDEX_ARRAY payload = ((count-1)<<24) | first; 6 indices from index 0.
    emit_method(code, NV4097_DRAW_INDEX_ARRAY, 0x05000000u);
    emit_method(code, NV4097_SET_BEGIN_END, 0u);

    emit_method(code, NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 0);
    emit_li32(code, 11, 1); code.push_back(ppc_sc());

    std::printf("\n  PPC program: %zu instructions\n", code.size());

    std::vector<uint8_t> codeBE(code.size() * 4);
    for (size_t i = 0; i < code.size(); ++i) {
        uint32_t be = __builtin_bswap32(code[i]);
        std::memcpy(&codeBE[i * 4], &be, 4);
    }
    if (!megakernel_init()) return 1;
    megakernel_load(kLoadAddr, codeBE.data(), codeBE.size());
    megakernel_set_entry(kLoadAddr, 0x00F00000ULL, 0);
    megakernel_run(16384);
    PPEState st{}; megakernel_read_state(&st);
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

    CHECK(rs.indexArrayAddress == IB_OFF, "IndexArrayAddress captured from FIFO");
    CHECK(bridge.counters.drawIndexed >= 1, "Indexed draw path hit");

    std::vector<uint32_t> fb(W * H, 0);
    raster.readbackPlane(0, fb.data());

    uint32_t lit = 0, clear = 0;
    uint32_t red = 0, green = 0, blue = 0, yellow = 0;
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t p = fb[y * W + x];
            uint8_t R = (p>>16)&0xFF, G = (p>>8)&0xFF, B = p&0xFF;
            bool cl = ((p & 0x00FFFFFFu) == 0x00202020u);
            if (cl) { ++clear; continue; }
            ++lit;
            if (R > 180 && G < 80  && B < 80 ) ++red;
            if (R < 80  && G > 180 && B < 80 ) ++green;
            if (R < 80  && G < 80  && B > 180) ++blue;
            if (R > 180 && G > 180 && B < 80 ) ++yellow;
        }
    }
    std::printf("  lit=%u clear=%u  corners R=%u G=%u B=%u Y=%u\n",
                lit, clear, red, green, blue, yellow);

    CHECK(lit > 15000 && lit < 21000, "Indexed quad covers ~160x120 rect (~19200 px)");
    CHECK(red > 50 && green > 50 && blue > 50 && yellow > 50,
          "All 4 vertex colors interpolated across the quad");
    CHECK(clear > (W * H - 20000),
          "Pixels outside quad retain clear color");

    raster.shutdown(); rsx_shutdown(&rs);
    std::printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
