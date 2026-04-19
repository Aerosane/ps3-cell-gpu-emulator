// test_gcm_elf.cu — Load a graphics-driving program via the PS3 ELF
// loader, run it on the cooperative megakernel, and verify rendered
// pixels match. Same gameplay as test_gcm_frame, but the program is
// now wrapped in a real ELF64 big-endian PPC executable — the host
// parses it via ps3_load_elf() first, then dispatches to the
// megakernel using the parsed entry point.
//
// This is the first end-to-end proof that a PPC graphics program
// stored as a real ELF loads correctly and still drives the RSX.

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

#include "elf_loader.h"
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

// ── PPC encode ───────────────────────────────────────────────────
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

// ── Big-endian writers ───────────────────────────────────────────
static void be_u16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back((uint8_t)(v >> 8)); b.push_back((uint8_t)v);
}
static void be_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 3; i >= 0; --i) b.push_back((uint8_t)(v >> (i*8)));
}
static void be_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 7; i >= 0; --i) b.push_back((uint8_t)(v >> (i*8)));
}

static std::vector<uint8_t> build_ppc64_elf(uint64_t loadAddr,
                                            const uint32_t* code,
                                            size_t codeWords) {
    constexpr size_t EHDR = 64;
    constexpr size_t PHDR = 56;
    const size_t codeBytes = codeWords * 4;
    const size_t codeOff   = EHDR + PHDR;
    std::vector<uint8_t> f;
    f.reserve(codeOff + codeBytes);

    f.push_back(0x7F); f.push_back('E'); f.push_back('L'); f.push_back('F');
    f.push_back(2); f.push_back(2); f.push_back(1); f.push_back(0); f.push_back(0);
    while (f.size() < 16) f.push_back(0);

    be_u16(f, 0x0002); be_u16(f, 0x0015); be_u32(f, 1);
    be_u64(f, loadAddr); be_u64(f, EHDR); be_u64(f, 0);
    be_u32(f, 0);
    be_u16(f, EHDR); be_u16(f, PHDR); be_u16(f, 1);
    be_u16(f, 0); be_u16(f, 0); be_u16(f, 0);

    be_u32(f, 0x00000001); be_u32(f, 0x00000005);
    be_u64(f, codeOff); be_u64(f, loadAddr); be_u64(f, loadAddr);
    be_u64(f, codeBytes); be_u64(f, codeBytes); be_u64(f, 0x10000);

    for (size_t i = 0; i < codeWords; ++i) be_u32(f, code[i]);
    return f;
}

static int fails = 0;
#define CHECK(x,m) do { if (x) std::printf("  OK: %s\n", m); \
    else { std::printf("  FAIL: %s\n", m); fails++; } } while(0)

int main() {
    std::printf("══════════════════════════════════════════════════════\n");
    std::printf("  ELF-loaded PPC → RSX end-to-end\n");
    std::printf("══════════════════════════════════════════════════════\n");

    constexpr uint32_t W = 320, H = 240;
    constexpr uint64_t kLoadAddr = 0x10000;
    constexpr uint32_t VRAM_BYTES = 2u * 1024u * 1024u;
    std::vector<uint8_t> vram(VRAM_BYTES, 0);
    constexpr uint32_t VB_POS = 0x100000;
    constexpr uint32_t VB_COL = 0x101000;

    float positions[9] = {
          40.f, H - 20.f, 0.5f,
         W - 40.f, H - 20.f, 0.5f,
         W / 2.f,  20.f,   0.5f,
    };
    std::memcpy(vram.data() + VB_POS, positions, sizeof(positions));
    uint32_t cols[3] = { 0xFFFF0000u, 0xFF00FF00u, 0xFF0000FFu };
    std::memcpy(vram.data() + VB_COL, cols, sizeof(cols));

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
    emit_method(code, NV4097_SET_BEGIN_END, PRIM_TRIANGLES);
    emit_method(code, NV4097_DRAW_ARRAYS,   0x02000000u);
    emit_method(code, NV4097_SET_BEGIN_END, 0u);
    emit_method(code, NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 0);
    emit_li32(code, 11, 1); code.push_back(ppc_sc());

    // ── Build ELF and hand it to the PS3 ELF loader ──────────
    std::vector<uint8_t> elfFile = build_ppc64_elf(kLoadAddr, code.data(), code.size());
    std::printf("  ELF size: %zu bytes  (%zu code words)\n",
                elfFile.size(), code.size());

    constexpr size_t HOST_MEM = 16u * 1024u * 1024u;
    std::vector<uint8_t> hostMem(HOST_MEM, 0);
    PS3ExeInfo info{};
    int rc = ps3_load_elf(elfFile.data(), elfFile.size(),
                          hostMem.data(), hostMem.size(), &info);
    std::printf("  ps3_load_elf rc=%d entry=0x%llx is_prx=%d\n",
                rc, (unsigned long long)info.entry_point, (int)info.is_prx);
    CHECK(rc == ELF_OK, "ps3_load_elf returned OK");
    CHECK(info.entry_point == kLoadAddr, "entry point matches load addr");

    // ── Boot via megakernel from the loaded image ─────────────
    if (!megakernel_init()) { std::printf("FAIL init\n"); return 1; }
    // Load only the span actually touched by the loader (start of the
    // PT_LOAD region up to a little past end of code).
    const size_t loadSpan = (size_t)kLoadAddr + code.size() * 4 + 0x100;
    megakernel_load(0, hostMem.data(), loadSpan);
    megakernel_set_entry(info.entry_point, 0x00F00000ULL, 0);
    float ms = megakernel_run(16384);
    PPEState st{}; megakernel_read_state(&st);
    std::printf("  megakernel: %.3f ms  pc=0x%llx halted=%u\n",
                ms, (unsigned long long)st.pc, st.halted);
    CHECK(st.halted == 1, "ELF-loaded PPC program halted");

    uint32_t cursor = 0;
    megakernel_read_mem(PS3_GCM_FIFO_BASE, &cursor, sizeof(cursor));
    std::vector<uint32_t> fifo(cursor, 0);
    megakernel_read_mem(PS3_GCM_FIFO_BASE + 4, fifo.data(), cursor * 4);
    megakernel_shutdown();

    CudaRasterizer raster; raster.init(W, H);
    RasterBridge bridge; bridge.attach(&raster); bridge.setVRAM(vram.data(), VRAM_BYTES);
    RSXState rs; rsx_init(&rs); rs.vulkanEmitter = &bridge;
    rsx_process_fifo(&rs, fifo.data(), cursor, vram.data(), cursor);

    CHECK(bridge.counters.draws == 1, "Draw dispatched through ELF-loaded path");
    CHECK(bridge.counters.flips == 1, "Flip dispatched");

    std::vector<uint32_t> fb(W * H, 0);
    raster.readbackPlane(0, fb.data());
    uint32_t lit = 0, red = 0, green = 0, blue = 0;
    for (uint32_t i = 0; i < W * H; ++i) {
        uint32_t p = fb[i];
        if ((p & 0x00FFFFFFu) == 0x00202020u) continue;
        uint8_t R = (p >> 16) & 0xFF, G = (p >> 8) & 0xFF, B = p & 0xFF;
        ++lit;
        if (R > 200 && G <  60 && B <  60) ++red;
        if (R <  60 && G > 200 && B <  60) ++green;
        if (R <  60 && G <  60 && B > 200) ++blue;
    }
    std::printf("  lit=%u red=%u green=%u blue=%u\n", lit, red, green, blue);
    CHECK(lit > 1000, "Triangle rasterized");
    CHECK(red > 30 && green > 30 && blue > 30,
          "Gouraud interpolation visible from ELF-loaded draw");

    raster.shutdown(); rsx_shutdown(&rs);
    std::printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
