// test_ppc_vmx.cu — Targeted VMX (Altivec) unit tests on the megakernel.
//
// Covers the subset added in commit db7fdaf:
//   Primary-4 ALU:   vxor, vor, vand, vandc, vnor
//   Primary-31 mem:  lvx, stvx (quadword load/store, 16B aligned)
//
// Each test writes a tiny PPC program into guest memory, seeds registers,
// runs a few cycles, ends with `sc` (r11 = SYS_PROCESS_EXIT) which halts
// the megakernel, then reads back vr[] to assert the expected result.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <initializer_list>

#include "ppc_defs.h"

extern "C" {
    int   megakernel_init();
    int   megakernel_load(uint64_t, const void*, size_t);
    int   megakernel_set_entry(uint64_t, uint64_t, uint64_t);
    float megakernel_run(uint32_t);
    int   megakernel_read_state(ppc::PPEState*);
    int   megakernel_write_state(const ppc::PPEState*);
    int   megakernel_read_mem(uint64_t, void*, size_t);
    int   megakernel_write_mem(uint64_t, const void*, size_t);
    void  megakernel_shutdown();
}

static int fails = 0;
#define CHECK(c, m) do { if (c) std::printf("  OK   %s\n", m); \
    else { std::printf("  FAIL %s\n", m); ++fails; } } while (0)

// Encoders ---------------------------------------------------------------
static uint32_t enc_vx_p4(uint32_t vd, uint32_t va, uint32_t vb, uint32_t xo) {
    return (4u << 26) | ((vd & 31) << 21) | ((va & 31) << 16)
         | ((vb & 31) << 11) | (xo & 0x7FFu);
}
static uint32_t enc_x_p31(uint32_t rt, uint32_t ra, uint32_t rb, uint32_t xo) {
    return (31u << 26) | ((rt & 31) << 21) | ((ra & 31) << 16)
         | ((rb & 31) << 11) | ((xo & 0x3FFu) << 1);
}
static constexpr uint32_t SC_INSN = 0x44000002u;

static void load_prog_be(uint64_t addr, std::initializer_list<uint32_t> ops) {
    std::vector<uint8_t> bytes;
    bytes.reserve(ops.size() * 4);
    for (uint32_t op : ops) {
        bytes.push_back((uint8_t)(op >> 24));
        bytes.push_back((uint8_t)(op >> 16));
        bytes.push_back((uint8_t)(op >> 8));
        bytes.push_back((uint8_t)op);
    }
    megakernel_load(addr, bytes.data(), bytes.size());
}

// Run program at 0x10000 seeded with `seed` applied after set_entry.
// Returns final state.
static ppc::PPEState run_and_read(void (*seed)(ppc::PPEState&)) {
    megakernel_set_entry(0x10000, 0x1000000, 0);
    ppc::PPEState st{};
    megakernel_read_state(&st);
    if (seed) seed(st);
    st.pc = 0x10000;
    st.halted = 0;
    megakernel_write_state(&st);
    megakernel_run(16);
    ppc::PPEState out{};
    megakernel_read_state(&out);
    return out;
}

static void alu_test(const char* name, uint32_t xo,
                     const uint32_t a[4], const uint32_t b[4],
                     const uint32_t expect[4]) {
    // <vx-op vr0, vr1, vr2>; li r11,1; sc
    load_prog_be(0x10000, {
        enc_vx_p4(0, 1, 2, xo),
        0x39600001u,   // li r11, 1  (SYS_PROCESS_EXIT)
        SC_INSN,
    });
    const uint32_t *av = a, *bv = b;
    struct Seed { const uint32_t *a, *b; };
    Seed seed{av, bv};
    static thread_local Seed* g_seed = nullptr;
    g_seed = &seed;
    auto apply = [](ppc::PPEState& s) {
        for (int i = 0; i < 4; ++i) { s.vr[1][i] = g_seed->a[i]; s.vr[2][i] = g_seed->b[i]; }
    };
    ppc::PPEState out = run_and_read(apply);
    bool ok = out.vr[0][0]==expect[0] && out.vr[0][1]==expect[1]
           && out.vr[0][2]==expect[2] && out.vr[0][3]==expect[3]
           && out.halted;
    if (!ok) {
        std::printf("    halted=%d pc=0x%llx\n", (int)out.halted, (unsigned long long)out.pc);
        std::printf("    got   %08x %08x %08x %08x\n",
                    out.vr[0][0], out.vr[0][1], out.vr[0][2], out.vr[0][3]);
        std::printf("    want  %08x %08x %08x %08x\n",
                    expect[0], expect[1], expect[2], expect[3]);
    }
    CHECK(ok, name);
}

int main() {
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  PPC VMX (Altivec) interpreter tests\n");
    std::printf("═══════════════════════════════════════════════════════\n");
    if (!megakernel_init()) {
        std::printf("FATAL: megakernel_init failed\n");
        return 2;
    }

    const uint32_t A[4] = {0xAAAAAAAAu, 0x12345678u, 0xFFFFFFFFu, 0x00000000u};
    const uint32_t B[4] = {0x55555555u, 0x0F0F0F0Fu, 0x0F0F0F0Fu, 0xDEADBEEFu};

    { uint32_t w[4]; for (int i=0;i<4;++i) w[i]=A[i]^B[i];      alu_test("vxor",  1220, A, B, w); }
    { uint32_t w[4]; for (int i=0;i<4;++i) w[i]=A[i]|B[i];      alu_test("vor",   1156, A, B, w); }
    { uint32_t w[4]; for (int i=0;i<4;++i) w[i]=A[i]&B[i];      alu_test("vand",  1028, A, B, w); }
    { uint32_t w[4]; for (int i=0;i<4;++i) w[i]=A[i]&~B[i];     alu_test("vandc", 1092, A, B, w); }
    { uint32_t w[4]; for (int i=0;i<4;++i) w[i]=~(A[i]|B[i]);   alu_test("vnor",  1284, A, B, w); }

    // lvx / stvx round-trip.
    {
        const uint32_t src_ea = 0x20000;
        const uint32_t dst_ea = 0x20010;
        const uint32_t src[4] = {0xDEADBEEFu, 0x01020304u, 0xCAFEBABEu, 0x1F2E3D4Cu};
        uint8_t srcBytes[16];
        for (int i = 0; i < 4; ++i) {
            srcBytes[i*4+0] = (uint8_t)(src[i] >> 24);
            srcBytes[i*4+1] = (uint8_t)(src[i] >> 16);
            srcBytes[i*4+2] = (uint8_t)(src[i] >>  8);
            srcBytes[i*4+3] = (uint8_t)(src[i]      );
        }
        megakernel_write_mem(src_ea, srcBytes, 16);
        uint32_t lvx  = enc_x_p31(3, 0, 4, 103);
        uint32_t stvx = enc_x_p31(3, 0, 5, 231);
        load_prog_be(0x10000, {
            lvx, stvx,
            0x39600001u,   // li r11, 1
            SC_INSN,
        });
        megakernel_set_entry(0x10000, 0x1000000, 0);
        ppc::PPEState st{};
        megakernel_read_state(&st);
        st.gpr[4] = src_ea;
        st.gpr[5] = dst_ea;
        st.pc = 0x10000; st.halted = 0;
        megakernel_write_state(&st);
        megakernel_run(16);
        ppc::PPEState out{};
        megakernel_read_state(&out);

        bool okLoad = out.vr[3][0]==src[0] && out.vr[3][1]==src[1]
                   && out.vr[3][2]==src[2] && out.vr[3][3]==src[3];
        if (!okLoad) {
            std::printf("    lvx got   %08x %08x %08x %08x\n",
                        out.vr[3][0], out.vr[3][1], out.vr[3][2], out.vr[3][3]);
            std::printf("    lvx want  %08x %08x %08x %08x\n",
                        src[0], src[1], src[2], src[3]);
        }
        CHECK(okLoad, "lvx loaded expected quadword");

        uint8_t got[16] = {};
        megakernel_read_mem(dst_ea, got, 16);
        bool okStore = std::memcmp(got, srcBytes, 16) == 0;
        if (!okStore) {
            std::printf("    stvx got   ");
            for (int i = 0; i < 16; ++i) std::printf("%02x ", got[i]);
            std::printf("\n    stvx want  ");
            for (int i = 0; i < 16; ++i) std::printf("%02x ", srcBytes[i]);
            std::printf("\n");
        }
        CHECK(okStore, "stvx wrote same bytes back to memory");
    }

    megakernel_shutdown();
    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
