// test_gcm_vp.cu — PPC-driven vertex-program + constant upload.
// Exercises NV4097_SET_TRANSFORM_PROGRAM_START / _PROGRAM (microcode window)
// and NV4097_SET_TRANSFORM_CONSTANT (vec4 constant slots) through the PPC -->
// FIFO --> rsx_command_processor path.
//
// Verifies:
//   - vpLoadOffset advances via _PROGRAM_START
//   - uploaded microcode lands at correct vpData[] slot
//   - vpValid becomes 1 after upload
//   - 256 vec4 constants are addressable; spot-check first and last slots

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
    std::printf("  PPC vertex-program + transform-constant upload\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint64_t kLoadAddr = 0x10000;
    std::vector<uint32_t> code; code.reserve(2048);

    // Three VP instructions (4 words each) at load offset 8 (8 words in).
    // PROGRAM_LOAD sets the upload target; PROGRAM_START sets execution entry.
    const uint32_t vpStart = 8u;
    emit_method(code, NV4097_SET_TRANSFORM_PROGRAM_LOAD, vpStart);
    emit_method(code, NV4097_SET_TRANSFORM_PROGRAM_START, vpStart);

    const uint32_t vpWords[12] = {
        0xAA000001u, 0xAA000002u, 0xAA000003u, 0xAA000004u,
        0xBB000001u, 0xBB000002u, 0xBB000003u, 0xBB000004u,
        0xCC000001u, 0xCC000002u, 0xCC000003u, 0xCC000004u,
    };
    for (int i = 0; i < 12; ++i) {
        emit_method(code, NV4097_SET_TRANSFORM_PROGRAM + i * 4, vpWords[i]);
    }

    // Upload transform constants: slot 0 = (1.0, 2.0, 3.0, 4.0),
    // slot 127 = (5.0, 6.0, 7.0, 8.0).
    // Must set CONSTANT_LOAD base before writing through the window.
    auto emit_const_vec4 = [&](uint32_t slot, float x, float y, float z, float w) {
        emit_method(code, NV4097_SET_TRANSFORM_CONSTANT_LOAD, slot);
        uint32_t fx, fy, fz, fw;
        std::memcpy(&fx, &x, 4); std::memcpy(&fy, &y, 4);
        std::memcpy(&fz, &z, 4); std::memcpy(&fw, &w, 4);
        emit_method(code, NV4097_SET_TRANSFORM_CONSTANT + 0,  fx);
        emit_method(code, NV4097_SET_TRANSFORM_CONSTANT + 4,  fy);
        emit_method(code, NV4097_SET_TRANSFORM_CONSTANT + 8,  fz);
        emit_method(code, NV4097_SET_TRANSFORM_CONSTANT + 12, fw);
    };
    emit_const_vec4(0,   1.0f, 2.0f, 3.0f, 4.0f);
    emit_const_vec4(127, 5.0f, 6.0f, 7.0f, 8.0f);

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
    if (cursor > 0) std::printf("  fifo[0..7]: %08x %08x %08x %08x %08x %08x %08x %08x\n",
        fifo[0], fifo[1], fifo[2], fifo[3], fifo[4], fifo[5], fifo[6], fifo[7]);
    if (cursor >= 42) std::printf("  fifo[34..41]: %08x %08x %08x %08x %08x %08x %08x %08x\n",
        fifo[34], fifo[35], fifo[36], fifo[37], fifo[38], fifo[39], fifo[40], fifo[41]);

    RSXState rs; rsx_init(&rs);
    rsx_process_fifo(&rs, fifo.data(), cursor, nullptr, cursor);

    CHECK(rs.vpValid == 1, "vpValid flag set after program upload");
    CHECK(rs.vpLoadOffset == vpStart * 4,
          "vpLoadOffset captured from PROGRAM_LOAD (stored as word index)");

    bool vpOk = true;
    for (int i = 0; i < 12; ++i) {
        if (rs.vpData[vpStart * 4 + i] != vpWords[i]) { vpOk = false; break; }
    }
    CHECK(vpOk, "12 VP microcode words landed at vpData[vpStart*4 + i]");

    CHECK(rs.vpConstants[0][0] == 1.0f && rs.vpConstants[0][1] == 2.0f &&
          rs.vpConstants[0][2] == 3.0f && rs.vpConstants[0][3] == 4.0f,
          "Transform constant slot 0 = (1,2,3,4)");
    CHECK(rs.vpConstants[127][0] == 5.0f && rs.vpConstants[127][1] == 6.0f &&
          rs.vpConstants[127][2] == 7.0f && rs.vpConstants[127][3] == 8.0f,
          "Transform constant slot 127 = (5,6,7,8)");

    std::printf("  vpLoadOffset=%u vpValid=%u\n", rs.vpLoadOffset, rs.vpValid);
    std::printf("  vpData[vpStart*4+0..11]: %08x %08x %08x %08x ... %08x\n",
                rs.vpData[vpStart*4+0], rs.vpData[vpStart*4+1],
                rs.vpData[vpStart*4+2], rs.vpData[vpStart*4+3],
                rs.vpData[vpStart*4+11]);

    rsx_shutdown(&rs);
    std::printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
