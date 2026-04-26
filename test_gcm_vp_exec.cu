// test_gcm_vp_exec.cu — PPC uploads a real VP microcode program that scales
// position by c[0]=(0.5, 0.5, 1, 1). The RSX bridge VP interpreter
// runs the program at draw time. We expect the drawn triangle to shrink
// to ~1/4 footprint vs the same geometry without VP.

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

#include "ppc_defs.h"
#include "rsx_defs.h"
#include "rsx_vp_shader.h"
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

// ── NV40 VP encoders (mirrors test_rsx_shaders.cu helpers) ────────
static uint32_t encodeSrc(uint32_t regType, uint32_t regIdx,
                          uint32_t sx, uint32_t sy, uint32_t sz, uint32_t sw,
                          bool neg = false) {
    uint32_t s = 0;
    s |= (regType & 0x3);
    s |= (regIdx & 0x3F) << 2;
    s |= (sw & 0x3) << 8;
    s |= (sz & 0x3) << 10;
    s |= (sy & 0x3) << 12;
    s |= (sx & 0x3) << 14;
    s |= (neg ? 1 : 0) << 16;
    return s;
}
static void buildVPInsn(uint32_t d[4], uint32_t vecOp, uint32_t scaOp,
                        uint32_t inputIdx, uint32_t constIdx,
                        uint32_t src0, uint32_t src1, uint32_t src2,
                        uint32_t vecDstTmp, uint32_t vecDstOut,
                        bool vecResult,
                        bool maskX, bool maskY, bool maskZ, bool maskW,
                        bool end = false) {
    uint32_t d0 = 0;
    d0 |= (vecDstTmp & 0x3F) << 14;
    if (vecResult) d0 |= (1u << 29);
    uint32_t d1 = 0;
    uint32_t src0h = (src0 >> 9) & 0xFF;
    d1 |= src0h;
    d1 |= (inputIdx & 0xF) << 8;
    d1 |= (constIdx & 0x3FF) << 12;
    d1 |= (vecOp & 0x1F) << 22;
    d1 |= (scaOp & 0x1F) << 27;
    uint32_t d2 = 0;
    uint32_t src2h = (src2 >> 11) & 0x3F;
    d2 |= src2h;
    d2 |= (src1 & 0x1FFFFu) << 6;
    uint32_t src0l = src0 & 0x1FFu;
    d2 |= src0l << 23;
    uint32_t d3 = 0;
    if (end) d3 |= 1;
    d3 |= (vecDstOut & 0x1F) << 2;
    if (maskW) d3 |= (1u << 13);
    if (maskZ) d3 |= (1u << 14);
    if (maskY) d3 |= (1u << 15);
    if (maskX) d3 |= (1u << 16);
    uint32_t src2l = src2 & 0x7FFu;
    d3 |= src2l << 21;
    d[0] = d0; d[1] = d1; d[2] = d2; d[3] = d3;
}

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  PPC VP execution: MUL o[0], in[0], c[0]   (half-scale)\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint32_t W = 320, H = 240;
    constexpr uint64_t kLoadAddr = 0x10000;
    constexpr uint32_t VRAM_BYTES = 2u * 1024u * 1024u;
    std::vector<uint8_t> vram(VRAM_BYTES, 0);

    // Triangle spanning most of the upper-left quadrant: roughly 140x140.
    constexpr uint32_t VB_POS = 0x100000;
    constexpr uint32_t VB_COL = 0x101000;
    float pos[9] = {
        20.f,  20.f, 0.5f,
        300.f, 20.f, 0.5f,
        160.f, 200.f, 0.5f,
    };
    uint32_t col[3] = { 0xFF00FF00u, 0xFF00FF00u, 0xFF00FF00u };
    rsx::store_be_floats(vram.data() + VB_POS, pos, sizeof(pos)/4);
    std::memcpy(vram.data() + VB_COL, col, sizeof(col));

    // Build VP microcode: one instruction.
    //   MUL o[0], in[0], c[0]    write vec result, writemask xyzw, end
    uint32_t prog[4];
    uint32_t s_in0 = encodeSrc(VP_REG_INPUT,    0, 0,1,2,3);
    uint32_t s_c0  = encodeSrc(VP_REG_CONSTANT, 0, 0,1,2,3);
    uint32_t s_z   = encodeSrc(VP_REG_TEMP,     0, 0,1,2,3);
    buildVPInsn(prog, VP_VEC_MUL, VP_SCA_NOP,
                /*inputIdx*/0, /*constIdx*/0,
                /*src0*/s_in0, /*src1*/s_c0, /*src2*/s_z,
                /*tmp*/0, /*out*/0,
                /*vecResult*/true,
                /*mask*/true, true, true, true,
                /*end*/true);

    // Program assembly
    std::vector<uint32_t> code; code.reserve(1024);
    emit_method(code, NV4097_SET_SURFACE_CLIP_HORIZONTAL, W);
    emit_method(code, NV4097_SET_SURFACE_CLIP_VERTICAL,   H);
    emit_method(code, NV4097_SET_SURFACE_PITCH_A,         W * 4);
    emit_method(code, NV4097_SET_SURFACE_COLOR_AOFFSET,   0);
    emit_method(code, NV4097_SET_SURFACE_FORMAT,          SURFACE_A8R8G8B8);
    emit_method(code, NV4097_SET_VIEWPORT_HORIZONTAL, (W << 16));
    emit_method(code, NV4097_SET_VIEWPORT_VERTICAL,   (H << 16));
    emit_method(code, NV4097_SET_COLOR_CLEAR_VALUE,   0xFF202020u);
    emit_method(code, NV4097_CLEAR_SURFACE,           CLEAR_COLOR | CLEAR_DEPTH);

    // Upload c[0] = (0.5, 0.5, 1, 1) at slot 0.
    auto emit_const_vec4 = [&](uint32_t slot, float x, float y, float z, float w) {
        uint32_t base = NV4097_SET_TRANSFORM_CONSTANT + slot * 16;
        uint32_t fx, fy, fz, fw;
        std::memcpy(&fx, &x, 4); std::memcpy(&fy, &y, 4);
        std::memcpy(&fz, &z, 4); std::memcpy(&fw, &w, 4);
        emit_method(code, base + 0,  fx);
        emit_method(code, base + 4,  fy);
        emit_method(code, base + 8,  fz);
        emit_method(code, base + 12, fw);
    };
    emit_const_vec4(0, 0.5f, 0.5f, 1.0f, 1.0f);

    // Upload VP microcode at offset 0.
    emit_method(code, NV4097_SET_TRANSFORM_PROGRAM_START, 0);
    for (int i = 0; i < 4; ++i) {
        emit_method(code, NV4097_SET_TRANSFORM_PROGRAM + i * 4, prog[i]);
    }

    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 0 * 4,
                (12u << 8) | (3u << 4) | VERTEX_F);
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 0 * 4, VB_POS);
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 3 * 4,
                (4u << 8) | (4u << 4) | VERTEX_UB);
    emit_method(code, NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 3 * 4, VB_COL);
    emit_method(code, NV4097_SET_BEGIN_END, PRIM_TRIANGLES);
    emit_method(code, NV4097_DRAW_ARRAYS,  0x02000000u);
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
    megakernel_run(65536);
    PPEState st{}; megakernel_read_state(&st);
    CHECK(st.halted == 1, "PPC program halted");

    uint32_t cursor = 0;
    megakernel_read_mem(PS3_GCM_FIFO_BASE, &cursor, sizeof(cursor));
    std::vector<uint32_t> fifo(cursor, 0);
    megakernel_read_mem(PS3_GCM_FIFO_BASE + 4, fifo.data(), cursor * 4);
    megakernel_shutdown();
    std::printf("  FIFO cursor=%u words\n", cursor);

    CudaRasterizer raster; raster.init(W, H);
    RasterBridge bridge; bridge.attach(&raster); bridge.setVRAM(vram.data(), VRAM_BYTES);
    RSXState rs; rsx_init(&rs); rs.vulkanEmitter = &bridge;
    rsx_process_fifo(&rs, fifo.data(), cursor, vram.data(), cursor);

    CHECK(rs.vpValid == 1, "VP uploaded and vpValid set");

    std::vector<uint32_t> fb(W * H, 0);
    raster.readbackPlane(0, fb.data());

    uint32_t green = 0;
    uint32_t greenInUpperLeft = 0; // expect VP-scaled triangle here
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t p = fb[y * W + x];
            uint8_t R=(p>>16)&0xFF, G=(p>>8)&0xFF, B=p&0xFF;
            bool isGreen = (R < 60 && G > 200 && B < 60);
            if (isGreen) {
                ++green;
                if (x < W/2 && y < H/2) ++greenInUpperLeft;
            }
        }
    }
    std::printf("  green=%u  greenInUpperLeft=%u\n", green, greenInUpperLeft);

    // Un-transformed triangle would rasterize ~25000 px. Scaled by 0.5
    // in x and y -> ~6250 px, all contained in upper-left quadrant.
    CHECK(green > 4000 && green < 10000,
          "VP-scaled triangle covers ~6k px (~1/4 of unscaled)");
    CHECK(greenInUpperLeft == green,
          "All lit pixels lie in upper-left quadrant (scale moved geometry)");

    raster.shutdown(); rsx_shutdown(&rs);
    std::printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
