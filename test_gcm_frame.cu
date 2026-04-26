// test_gcm_frame.cu — FULL FRAME rendered by PPC code
//
// Scenario: PPC binary builds a complete frame via sc 0xC710 syscalls
// (SURFACE / VIEWPORT / CLEAR_COLOR / CLEAR_SURFACE / VERTEX_DATA /
// BEGIN / DRAW_ARRAYS / END / FLIP), halts. The host reads the ring
// and pumps it through RasterBridge + CudaRasterizer with VRAM that it
// pre-staged with vertex positions and per-vertex colors. Then we read
// the framebuffer back and verify pixels actually rendered.
//
// This is the first end-to-end proof that a guest PPC program can drive
// the full PS3 graphics pipeline on our emulator.

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

// ── PPC encoding helpers ─────────────────────────────────────────
static uint32_t ppc_addi (int rD, int rA, int16_t imm) {
    return (14u<<26) | ((uint32_t)rD<<21) | ((uint32_t)rA<<16) | (uint16_t)imm;
}
static uint32_t ppc_addis(int rD, int rA, int16_t imm) {
    return (15u<<26) | ((uint32_t)rD<<21) | ((uint32_t)rA<<16) | (uint16_t)imm;
}
static uint32_t ppc_ori  (int rA, int rS, uint16_t imm) {
    return (24u<<26) | ((uint32_t)rS<<21) | ((uint32_t)rA<<16) | imm;
}
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
    std::printf("  PPC syscall bridge renders a FULL frame end-to-end\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint32_t W = 320, H = 240;
    constexpr uint64_t kLoadAddr = 0x10000;

    // ── Stage VRAM ────────────────────────────────────────────
    constexpr uint32_t VRAM_BYTES = 2u * 1024u * 1024u;
    std::vector<uint8_t> vram(VRAM_BYTES, 0);

    // Surface spans 0 .. 320*240*4 = 300 KB. Place vertex buffers well
    // past it so rsx_clear_surface doesn't stomp them.
    constexpr uint32_t VB_POS = 0x100000;
    constexpr uint32_t VB_COL = 0x101000;
    constexpr uint32_t SURF_A = 0x00000;

    float positions[9] = {
         40.f, H - 20.f, 0.5f,
         W - 40.f, H - 20.f, 0.5f,
         W / 2.f, 20.f, 0.5f,
    };
    rsx::store_be_floats(vram.data() + VB_POS, positions, sizeof(positions)/4);

    // VERTEX_UB: bytes stored BGRA, little-endian uint32.
    uint32_t cols[3] = {
        0xFFFF0000u,   // red
        0xFF00FF00u,   // green
        0xFF0000FFu,   // blue
    };
    std::memcpy(vram.data() + VB_COL, cols, sizeof(cols));

    // ── Build PPC program ─────────────────────────────────────
    std::vector<uint32_t> code;
    code.reserve(256);

    emit_method(code, NV4097_SET_SURFACE_CLIP_HORIZONTAL, W);
    emit_method(code, NV4097_SET_SURFACE_CLIP_VERTICAL,   H);
    emit_method(code, NV4097_SET_SURFACE_PITCH_A,         W * 4);
    emit_method(code, NV4097_SET_SURFACE_COLOR_AOFFSET,   SURF_A);
    emit_method(code, NV4097_SET_SURFACE_FORMAT,          SURFACE_A8R8G8B8);

    emit_method(code, NV4097_SET_VIEWPORT_HORIZONTAL, (W << 16));
    emit_method(code, NV4097_SET_VIEWPORT_VERTICAL,   (H << 16));

    emit_method(code, NV4097_SET_COLOR_CLEAR_VALUE,   0xFF202020u);
    emit_method(code, NV4097_CLEAR_SURFACE,           CLEAR_COLOR | CLEAR_DEPTH);

    // Vertex slot 0 — position (stride 12, 3 floats)
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 0 * 4,
                (12u << 8) | (3u << 4) | VERTEX_F);
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0 * 4, VB_POS);

    // Vertex slot 3 — color (stride 4, 4 ubytes)
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 3 * 4,
                (4u << 8) | (4u << 4) | VERTEX_UB);
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 3 * 4, VB_COL);

    emit_method(code, NV4097_SET_BEGIN_END,           PRIM_TRIANGLES);
    // DRAW_ARRAYS payload: (count-1)<<24 | first  => (3-1)<<24 = 0x02000000
    emit_method(code, NV4097_DRAW_ARRAYS,             0x02000000u);
    emit_method(code, NV4097_SET_BEGIN_END,           0u);

    emit_method(code, NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, SURF_A);

    // Halt.
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

    float ms = megakernel_run(/*maxCycles*/ 8192);

    PPEState st{};
    megakernel_read_state(&st);
    std::printf("  megakernel: %.3f ms  pc=0x%llx  halted=%u\n",
                ms, (unsigned long long)st.pc, st.halted);
    CHECK(st.halted == 1, "PPC program halted");

    uint32_t cursor = 0;
    megakernel_read_mem(PS3_GCM_FIFO_BASE, &cursor, sizeof(cursor));
    std::printf("  ring cursor = %u dwords (expect 36: 18 methods × 2)\n", cursor);
    CHECK(cursor >= 30, "Ring captured a full frame's worth of methods");

    std::vector<uint32_t> fifo(cursor, 0);
    megakernel_read_mem(PS3_GCM_FIFO_BASE + 4, fifo.data(), cursor * 4);

    megakernel_shutdown();

    // ── Set up rasterizer + bridge and pump the ring ─────────
    CudaRasterizer raster;
    raster.init(W, H);

    RasterBridge bridge;
    bridge.attach(&raster);
    bridge.setVRAM(vram.data(), VRAM_BYTES);

    RSXState rs;
    rsx_init(&rs);
    rs.vulkanEmitter = &bridge;

    rsx_process_fifo(&rs, fifo.data(), cursor, vram.data(), cursor);

    std::printf("\n  bridge counters: surf=%u clears=%u draws=%u flips=%u\n",
                bridge.counters.surfaceSetups, bridge.counters.clears,
                bridge.counters.draws, bridge.counters.flips);

    CHECK(bridge.counters.surfaceSetups >= 1, "SURFACE via PPC→bridge");
    CHECK(bridge.counters.clears        >= 1, "CLEAR   via PPC→bridge");
    CHECK(bridge.counters.draws         == 1, "DRAW_ARRAYS via PPC→bridge");
    CHECK(bridge.counters.flips         >= 1, "FLIP    via PPC→bridge");

    // ── Readback & pixel analysis ────────────────────────────
    raster.setMRTCount(1);
    std::vector<uint32_t> fb(W * H, 0);
    raster.readbackPlane(0, fb.data());

    uint32_t clearPixels = 0, lit = 0, red = 0, green = 0, blue = 0;
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t p = fb[y * W + x];
            if ((p & 0x00FFFFFFu) == 0x00202020u) { ++clearPixels; continue; }
            uint8_t R = (p >> 16) & 0xFF;
            uint8_t G = (p >>  8) & 0xFF;
            uint8_t B =  p        & 0xFF;
            ++lit;
            if (R > 200 && G <  60 && B <  60) ++red;
            if (R <  60 && G > 200 && B <  60) ++green;
            if (R <  60 && G <  60 && B > 200) ++blue;
        }
    }
    std::printf("  pixels: clear=%u  lit=%u  red=%u green=%u blue=%u\n",
                clearPixels, lit, red, green, blue);
    CHECK(clearPixels > W * H / 4, "Clear color painted most of surface");
    CHECK(lit > 1000,               "Triangle rasterized non-trivially");
    CHECK(red > 30 && green > 30 && blue > 30,
          "All three vertex colors visible (Gouraud interpolation)");

    raster.shutdown();
    rsx_shutdown(&rs);

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
