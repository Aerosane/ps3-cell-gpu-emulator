// test_gcm_fp_tex.cu — PPC-driven fragment-program pointer + multi-texture-unit setup.
// NV4097_SET_SHADER_PROGRAM packs FP VRAM offset (bits 31:4) + DMA control bits (3:0).
// NV4097_SET_TEXTURE_* covers 16 units at stride 0x20.
//
// This test uploads unit 0, unit 3, and unit 7 with distinct offsets/formats/rects,
// and sets a fragment program pointer. It then verifies the state lands correctly.

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

static void emit_texture_unit(std::vector<uint32_t>& c, uint32_t unit,
                              uint32_t offset, uint32_t format,
                              uint32_t control0, uint32_t rect) {
    const uint32_t BASE = NV4097_SET_TEXTURE_OFFSET + unit * 0x20;
    emit_method(c, BASE + 0x00, offset);
    emit_method(c, BASE + 0x04, format);
    emit_method(c, BASE + 0x0C, control0);
    emit_method(c, BASE + 0x18, rect);
}

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  PPC fragment-program + multi-texture-unit state\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint64_t kLoadAddr = 0x10000;
    std::vector<uint32_t> code; code.reserve(512);

    // Fragment program: VRAM offset 0x200000, DMA control = 0x1 (VIDEO memory).
    // Packing: (offset & ~0xF) | control[3:0].
    const uint32_t fpOffset  = 0x00200000u;
    const uint32_t fpControl = 0x1u;
    emit_method(code, NV4097_SET_SHADER_PROGRAM, fpOffset | fpControl);

    // Three texture units with distinct parameters.
    struct TexDesc { uint32_t unit, offset, format, ctrl, w, h; };
    const TexDesc td[3] = {
        { 0, 0x00300000u, 0x0085AE02u, 0x00000000u, 256, 256 },
        { 3, 0x00400000u, 0x0085AE02u, 0x80000000u, 512, 128 },
        { 7, 0x00500000u, 0x0081AE02u, 0x40000000u,  64,  64 },
    };
    for (auto& t : td) {
        emit_texture_unit(code, t.unit, t.offset, t.format, t.ctrl,
                          (t.w << 16) | t.h);
    }

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
    megakernel_run(32768);
    PPEState st{}; megakernel_read_state(&st);
    CHECK(st.halted == 1, "PPC program halted");

    uint32_t cursor = 0;
    megakernel_read_mem(PS3_GCM_FIFO_BASE, &cursor, sizeof(cursor));
    std::vector<uint32_t> fifo(cursor, 0);
    megakernel_read_mem(PS3_GCM_FIFO_BASE + 4, fifo.data(), cursor * 4);
    megakernel_shutdown();
    std::printf("  FIFO cursor=%u words\n", cursor);

    RSXState rs; rsx_init(&rs);
    rsx_process_fifo(&rs, fifo.data(), cursor, nullptr, cursor);

    CHECK(rs.fpOffset  == fpOffset,  "Fragment-program offset captured (bits [31:4])");
    CHECK(rs.fpControl == fpControl, "Fragment-program DMA control captured (bits [3:0])");

    for (auto& t : td) {
        const auto& U = rs.textures[t.unit];
        bool ok = U.enabled && U.offset == t.offset && U.format == t.format
               && U.control0 == t.ctrl
               && U.width  == t.w && U.height == t.h;
        char msg[128];
        std::snprintf(msg, sizeof(msg),
            "Texture unit %u: offset=%08x fmt=%08x w=%u h=%u",
            t.unit, U.offset, U.format, U.width, U.height);
        CHECK(ok, msg);
    }

    CHECK(!rs.textures[1].enabled && !rs.textures[2].enabled &&
          !rs.textures[4].enabled && !rs.textures[5].enabled,
          "Untouched texture units remain disabled");

    rsx_shutdown(&rs);
    std::printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
