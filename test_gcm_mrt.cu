// test_gcm_mrt.cu — PPC-driven MRT surface setup (up to 4 color planes).
// RSX supports TARGET_MRT0..MRT3. This test programs all four color
// offsets + TARGET_MRT3 from the PPC side and verifies the state lands.
// Render-backend MRT execution is separate; this is the FIFO plumbing test.

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
    std::printf("  PPC MRT surface setup (4 color planes + TARGET_MRT3)\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint64_t kLoadAddr = 0x10000;
    const uint32_t offA = 0x00100000u;
    const uint32_t offB = 0x00200000u;
    const uint32_t offC = 0x00300000u;
    const uint32_t offD = 0x00400000u;

    std::vector<uint32_t> code; code.reserve(256);
    emit_method(code, NV4097_SET_SURFACE_COLOR_AOFFSET, offA);
    emit_method(code, NV4097_SET_SURFACE_COLOR_BOFFSET, offB);
    emit_method(code, NV4097_SET_SURFACE_COLOR_COFFSET, offC);
    emit_method(code, NV4097_SET_SURFACE_COLOR_DOFFSET, offD);
    emit_method(code, NV4097_SET_SURFACE_COLOR_TARGET,  SURFACE_TARGET_MRT3);
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

    RSXState rs; rsx_init(&rs);
    rsx_process_fifo(&rs, fifo.data(), cursor, nullptr, cursor);

    CHECK(rs.surfaceOffsetA == offA, "surfaceOffsetA = 0x00100000");
    CHECK(rs.surfaceOffsetB == offB, "surfaceOffsetB = 0x00200000");
    CHECK(rs.surfaceOffsetC == offC, "surfaceOffsetC = 0x00300000");
    CHECK(rs.surfaceOffsetD == offD, "surfaceOffsetD = 0x00400000");
    CHECK(rs.surfaceColorTarget == SURFACE_TARGET_MRT3,
          "surfaceColorTarget = MRT3 (A+B+C+D)");

    std::printf("  surfaceOffsets: A=%08x B=%08x C=%08x D=%08x target=%u\n",
                rs.surfaceOffsetA, rs.surfaceOffsetB,
                rs.surfaceOffsetC, rs.surfaceOffsetD,
                rs.surfaceColorTarget);

    rsx_shutdown(&rs);
    std::printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
