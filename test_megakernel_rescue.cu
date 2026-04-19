// test_megakernel_rescue.cu — Validate the megakernel's safety nets:
//
//   1. PC += 4 on unimplemented opcode (instead of spinning).
//   2. Zero-fetch rescue: PC = LR when fetched insn is 0 and LR is sane.
//   3. Code-range rescue: PC = LR when fetched PC is outside
//      [d_codeBegin, d_codeEnd).
//
// Each test sets up a scenario and asserts execution either advances
// past the offending PC or transfers to a valid location (terminated by
// sc SYS_PROCESS_EXIT).

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
    int   megakernel_set_code_range(uint64_t, uint64_t);
    float megakernel_run(uint32_t);
    int   megakernel_read_state(ppc::PPEState*);
    int   megakernel_write_state(const ppc::PPEState*);
    int   megakernel_write_mem(uint64_t, const void*, size_t);
    void  megakernel_shutdown();
}

static int fails = 0;
#define CHECK(c, m) do { if (c) std::printf("  OK   %s\n", m); \
    else { std::printf("  FAIL %s\n", m); ++fails; } } while (0)

static void load_be(uint64_t addr, std::initializer_list<uint32_t> ops) {
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

static constexpr uint32_t LI_R11_1 = 0x39600001u;  // li r11, 1 (SYS_PROCESS_EXIT)
static constexpr uint32_t SC       = 0x44000002u;  // sc
// A known "unimplemented" opcode: primary 56 (psq_l on PPC) isn't in our
// dispatch. Use some reserved primary — e.g. primary 5 (not used in
// PowerPC). Encoding: (5 << 26) = 0x14000000.
static constexpr uint32_t UNIMPL_PRIMARY5 = 0x14000000u;

int main() {
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  Megakernel safety-net tests\n");
    std::printf("═══════════════════════════════════════════════════════\n");
    if (!megakernel_init()) { std::printf("FATAL init\n"); return 2; }

    // --- 1. PC += 4 on unimplemented opcode ---------------------------
    // Program: <unimpl>; li r11,1; sc.
    // If PC doesn't advance on the unimpl, we'd spin forever; the run
    // bound (64 cycles) lets us detect that as "halted==0".
    {
        load_be(0x10000, { UNIMPL_PRIMARY5, LI_R11_1, SC });
        megakernel_set_entry(0x10000, 0x1000000, 0);
        megakernel_run(64);
        ppc::PPEState st{};
        megakernel_read_state(&st);
        CHECK(st.halted == 1, "unimplemented op: PC advanced and reached sc");
    }

    // --- 2. Zero-fetch rescue: PC=LR on all-zeros insn ---------------
    // Lay out:
    //   0x10000: bl 0x10100           (branches to a zero hole; LR=0x10004)
    // But the branch target at 0x10100 is ALL ZEROS.  Rescue should set
    // PC = LR = 0x10004 which contains li r11,1 + sc.
    //
    // We skip the bl and directly simulate: set PC=0x10100 (zeros), LR=0x20000.
    // At 0x20000 put li r11,1; sc. Expect rescue to re-enter at LR.
    {
        // clear 0x10100 region (already zero after init cudaMemset). Put
        // tail at 0x20000.
        load_be(0x20000, { LI_R11_1, SC });
        megakernel_set_entry(0x10000, 0x1000000, 0);
        // Override PC to a zero region, LR to our tail.
        ppc::PPEState st{};
        megakernel_read_state(&st);
        st.pc     = 0x10100;
        st.lr     = 0x20000;
        st.halted = 0;
        megakernel_write_state(&st);
        megakernel_run(32);
        ppc::PPEState out{};
        megakernel_read_state(&out);
        CHECK(out.halted == 1, "zero-fetch rescue: PC=LR reached sc");
    }

    // --- 3. Code-range rescue: PC out-of-range → PC=LR ----------------
    //   Set code range [0x10000, 0x30000).  Put tail at 0x20000 (in
    //   range).  Force PC to 0x40000 (out-of-range) and LR=0x20000.
    {
        megakernel_set_code_range(0x10000, 0x30000);
        // Ensure 0x40000 contains non-zero garbage so this isn't just
        // the zero-fetch path.
        uint32_t garbage_be = 0xDE;
        uint8_t gb[4] = { 0x12, 0x34, 0x56, 0x78 };  (void)garbage_be;
        megakernel_write_mem(0x40000, gb, 4);
        load_be(0x20000, { LI_R11_1, SC });
        megakernel_set_entry(0x10000, 0x1000000, 0);
        ppc::PPEState st{};
        megakernel_read_state(&st);
        st.pc     = 0x40000;
        st.lr     = 0x20000;
        st.halted = 0;
        megakernel_write_state(&st);
        megakernel_run(32);
        ppc::PPEState out{};
        megakernel_read_state(&out);
        CHECK(out.halted == 1, "out-of-range rescue: PC=LR reached sc");
        // Turn off the range to avoid affecting later tests.
        megakernel_set_code_range(0, 0);
    }

    megakernel_shutdown();
    std::printf("\n%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    return fails ? 1 : 0;
}
