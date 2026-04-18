// test_gcm_syscall.cu — PPC code drives RSX through HLE syscall bridge
//
// Closes the loop: PPC binary emits FIFO method headers/payloads via two
// custom HLE syscalls (0xC710 / 0xC711). The HLE handler in ppc_interpreter
// writes them into a guest-visible ring at PS3_GCM_FIFO_BASE. After the
// PPC thread halts via SYS_PROCESS_EXIT, the host copies the ring out and
// feeds it to rsx_process_fifo, then reads RSXState back to confirm the
// methods landed.
//
// Uses the cooperative megakernel (extern "C" megakernel_*) rather than
// the warp JIT, because the warp runner only follows B/BC branches and
// would terminate at the first `sc` without dispatching the rest of the
// program.

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "ppc_defs.h"
#include "rsx_defs.h"

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
static uint32_t ppc_li(int rD, int16_t imm) { return ppc_addi(rD, 0, imm); }

// Load any 32-bit value into rD via lis + ori. Always emits 2 instructions
// so block sizes are predictable.
static void emit_li32(std::vector<uint32_t>& code, int rD, uint32_t val) {
    uint16_t hi = (uint16_t)(val >> 16);
    uint16_t lo = (uint16_t) val;
    code.push_back(ppc_addis(rD, 0, (int16_t)hi));
    code.push_back(ppc_ori  (rD, rD, lo));
}

static void emit_gcm_method(std::vector<uint32_t>& code,
                            uint32_t method, uint32_t data)
{
    emit_li32(code, 3, method);
    emit_li32(code, 4, data);
    emit_li32(code, 11, 0xC710u);
    code.push_back(ppc_sc());
}

static int fails = 0;
#define CHECK(c, m) do { if (c) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  PPC HLE syscall → FIFO ring → RSX dispatcher\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint64_t kLoadAddr = 0x10000;

    std::vector<uint32_t> code;
    code.reserve(128);

    emit_gcm_method(code, NV4097_SET_SURFACE_CLIP_HORIZONTAL, 320);
    emit_gcm_method(code, NV4097_SET_SURFACE_CLIP_VERTICAL,   240);
    emit_gcm_method(code, NV4097_SET_DEPTH_TEST_ENABLE,         1);
    emit_gcm_method(code, NV4097_SET_DEPTH_FUNC,           0x0203);
    emit_gcm_method(code, NV4097_SET_BLEND_ENABLE,              1);
    emit_gcm_method(code, NV4097_SET_CULL_FACE_ENABLE,          1);
    emit_gcm_method(code, NV4097_SET_CULL_FACE,            0x0405);
    emit_gcm_method(code, NV4097_SET_ALPHA_TEST_ENABLE,         1);
    emit_gcm_method(code, NV4097_SET_ALPHA_FUNC,           0x0204);
    emit_gcm_method(code, NV4097_SET_ALPHA_REF,              0x80);
    emit_gcm_method(code, NV4097_SET_SHADE_MODE,           0x1D00);
    emit_gcm_method(code, NV4097_SET_FRONT_FACE,           0x0900);
    emit_gcm_method(code, NV4097_SET_COLOR_CLEAR_VALUE, 0xFF400080u);

    // Halt.
    emit_li32(code, 11, 1);  // SYS_PROCESS_EXIT
    code.push_back(ppc_sc());

    std::printf("\n  PPC program: %zu instructions (%zu bytes)\n",
                code.size(), code.size() * 4);

    // Convert to big-endian for guest memory.
    std::vector<uint8_t> codeBE(code.size() * 4);
    for (size_t i = 0; i < code.size(); ++i) {
        uint32_t be = __builtin_bswap32(code[i]);
        std::memcpy(&codeBE[i * 4], &be, 4);
    }

    if (!megakernel_init()) {
        std::printf("  FAIL: megakernel_init\n");
        return 1;
    }

    megakernel_load(kLoadAddr, codeBE.data(), codeBE.size());
    megakernel_set_entry(kLoadAddr, /*sp*/ 0x00F00000ULL, /*toc*/ 0);

    float ms = megakernel_run(/*maxCycles*/ 4096);

    PPEState st{};
    megakernel_read_state(&st);
    std::printf("  megakernel: %.3f ms  pc=0x%llx  halted=%u\n",
                ms, (unsigned long long)st.pc, st.halted);
    CHECK(st.halted == 1, "PPC program halted via SYS_PROCESS_EXIT");

    // ── Read back FIFO ring ────────────────────────────────
    uint32_t cursor = 0;
    megakernel_read_mem(PS3_GCM_FIFO_BASE, &cursor, sizeof(cursor));
    std::printf("  GCM ring cursor: %u dwords\n", cursor);
    CHECK(cursor > 0,         "Ring received at least one word");
    CHECK((cursor & 1u) == 0, "Cursor is even (header+payload pairs)");
    CHECK(cursor / 2u == 13,  "Ring received exactly 13 method emissions");

    std::vector<uint32_t> fifo(cursor, 0);
    if (cursor > 0) {
        megakernel_read_mem(PS3_GCM_FIFO_BASE + 4,
                            fifo.data(), cursor * sizeof(uint32_t));
    }

    megakernel_shutdown();

    // ── Pump through RSX dispatcher ────────────────────────
    RSXState rs;
    rsx_init(&rs);
    std::vector<uint8_t> dummyVram(64 * 1024, 0);
    rsx_process_fifo(&rs, fifo.data(), cursor, dummyVram.data(), cursor);

    std::printf("\n  RSXState after FIFO pump:\n");
    std::printf("    surface w/h = %u / %u\n", rs.surfaceWidth, rs.surfaceHeight);
    std::printf("    depth %d/0x%x  blend %d  cull %d/0x%x\n",
                rs.depthTestEnable, rs.depthFunc, rs.blendEnable,
                rs.cullFaceEnable, rs.cullFace);
    std::printf("    alpha %d func 0x%x ref 0x%x  shade 0x%x  front 0x%x  clear 0x%08x\n",
                rs.alphaTestEnable, rs.alphaFunc, rs.alphaRef,
                rs.shadeMode, rs.frontFace, rs.colorClearValue);

    CHECK(rs.surfaceWidth  == 320,    "SURFACE_CLIP_HORIZONTAL via PPC syscall");
    CHECK(rs.surfaceHeight == 240,    "SURFACE_CLIP_VERTICAL   via PPC syscall");
    CHECK(rs.depthTestEnable,         "DEPTH_TEST_ENABLE       via PPC syscall");
    CHECK(rs.depthFunc == 0x0203,     "DEPTH_FUNC              via PPC syscall");
    CHECK(rs.blendEnable,             "BLEND_ENABLE            via PPC syscall");
    CHECK(rs.cullFaceEnable,          "CULL_FACE_ENABLE        via PPC syscall");
    CHECK(rs.cullFace      == 0x0405, "CULL_FACE=BACK          via PPC syscall");
    CHECK(rs.alphaTestEnable,         "ALPHA_TEST_ENABLE       via PPC syscall");
    CHECK(rs.alphaFunc     == 0x0204, "ALPHA_FUNC=GREATER      via PPC syscall");
    CHECK(rs.alphaRef      == 0x80,   "ALPHA_REF=128           via PPC syscall");
    CHECK(rs.shadeMode     == 0x1D00, "SHADE_MODE=FLAT         via PPC syscall");
    CHECK(rs.frontFace     == 0x0900, "FRONT_FACE=CW           via PPC syscall");
    CHECK(rs.colorClearValue == 0xFF400080u,
                                      "COLOR_CLEAR_VALUE 32-bit literal via PPC syscall");

    rsx_shutdown(&rs);

    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
