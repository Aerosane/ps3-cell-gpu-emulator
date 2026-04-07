// test_rpcs3_bridge.cu — RPCS3 SPU CUDA Bridge Test Suite
//
// Simulates what RPCS3 would do: load SPU program → init state → call bridge → verify
//
#include "rpcs3_spu_bridge.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

static int total_pass = 0, total_fail = 0;

// Helper: store big-endian instruction into LS
static void ls_store_insn(uint8_t* ls, uint32_t addr, uint32_t insn) {
    uint32_t be = ((insn >> 24) & 0xFF) | ((insn >> 8) & 0xFF00) |
                  ((insn << 8) & 0xFF0000) | ((insn << 24) & 0xFF000000);
    memcpy(ls + (addr & 0x3FFFF), &be, 4);
}

// SPU instruction encoders (matching spu_defs.h constants)
// RR: op11[31:21] rb[20:14] ra[13:7] rt[6:0]
static uint32_t spu_rr(uint32_t op11, uint32_t rb, uint32_t ra, uint32_t rt) {
    return (op11 << 21) | ((rb & 0x7F) << 14) | ((ra & 0x7F) << 7) | (rt & 0x7F);
}
// RI10: op8[31:24] i10[23:14] ra[13:7] rt[6:0]
static uint32_t spu_ri10(uint32_t op8, int32_t i10, uint32_t ra, uint32_t rt) {
    return (op8 << 24) | (((uint32_t)i10 & 0x3FF) << 14) | ((ra & 0x7F) << 7) | (rt & 0x7F);
}
// RI16: op9[31:23] i16[22:7] rt[6:0]
static uint32_t spu_ri16(uint32_t op9, int32_t i16, uint32_t rt) {
    return (op9 << 23) | (((uint32_t)i16 & 0xFFFF) << 7) | (rt & 0x7F);
}

// il rt, i16: op9 = 0x081
static uint32_t spu_il(uint32_t rt, int32_t i16) {
    return spu_ri16(0x081, i16, rt);
}

// br i16: op9 = 0x064
static uint32_t spu_br(int32_t target_word) {
    return spu_ri16(0x064, target_word, 0);
}

// ai rt, ra, i10: op8 = 0x1C
static uint32_t spu_ai(uint32_t rt, uint32_t ra, int32_t i10) {
    return spu_ri10(0x1C, i10, ra, rt);
}

// a rt, ra, rb: op11 = 0x0C0
static uint32_t spu_a(uint32_t rt, uint32_t ra, uint32_t rb) {
    return spu_rr(0x0C0, rb, ra, rt);
}

// cgti rt, ra, i10: op8 = 0x4C
static uint32_t spu_cgti(uint32_t rt, uint32_t ra, int32_t i10) {
    return spu_ri10(0x4C, i10, ra, rt);
}

// brz rt, i16: op9 = 0x040
static uint32_t spu_brz(uint32_t rt, int32_t target_word) {
    return spu_ri16(0x040, target_word, rt);
}

// brnz rt, i16: op9 = 0x042
static uint32_t spu_brnz(uint32_t rt, int32_t target_word) {
    return spu_ri16(0x042, target_word, rt);
}

// stop: op11 = 0x000
static uint32_t spu_stop() {
    return 0x00000000;
}

// nop: op11 = 0x201
static uint32_t spu_nop() {
    return (0x201 << 21);
}

// ═══════════════════════════════════════════════════════════════
// Test 1: Bridge Init + Basic ALU
// ═══════════════════════════════════════════════════════════════

static void test_bridge_alu() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: Bridge Init + ALU             ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t* ls = (uint8_t*)calloc(1, 256 * 1024);
    SPUBridgeState state = {};

    // Program: il r3, 42; il r4, 58; a r5, r3, r4; stop
    ls_store_insn(ls, 0x0000, spu_il(3, 42));
    ls_store_insn(ls, 0x0004, spu_il(4, 58));
    ls_store_insn(ls, 0x0008, spu_a(5, 3, 4));
    ls_store_insn(ls, 0x000C, spu_stop());

    SPUBridgeStats stats = {};
    int ret = spu_bridge_run(ls, &state, 100, &stats);

    printf("  Executed: %d instructions\n", ret);
    printf("  r3=%u r4=%u r5=%u (expect 42, 58, 100)\n",
           state.gpr[3][0], state.gpr[4][0], state.gpr[5][0]);
    printf("  PC=0x%04X stopped=%u\n", state.pc, state.stopped);
    printf("  Time: %.3f ms total, %.3f ms exec\n", stats.total_ms, stats.exec_ms);

    bool pass = (state.gpr[3][0] == 42 && state.gpr[4][0] == 58 &&
                 state.gpr[5][0] == 100 && state.stopped == 1);
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    free(ls);
}

// ═══════════════════════════════════════════════════════════════
// Test 2: Loop Execution
// ═══════════════════════════════════════════════════════════════

static void test_bridge_loop() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: Loop Execution                ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t* ls = (uint8_t*)calloc(1, 256 * 1024);
    SPUBridgeState state = {};

    // Program: count down from 1000 → 0, accumulate in r4
    // il r3, 1000  (counter)
    ls_store_insn(ls, 0x0000, spu_il(3, 1000));
    // il r4, 0     (accumulator)
    ls_store_insn(ls, 0x0004, spu_il(4, 0));
    // loop (0x0008): ai r4, r4, 1   (acc++)
    ls_store_insn(ls, 0x0008, spu_ai(4, 4, 1));
    // ai r3, r3, -1  (counter--)
    ls_store_insn(ls, 0x000C, spu_ai(3, 3, -1));
    // brnz r3, loop  (if counter != 0, goto word 2 = 0x0008)
    ls_store_insn(ls, 0x0010, spu_brnz(3, 2));
    // stop
    ls_store_insn(ls, 0x0014, spu_stop());

    SPUBridgeStats stats = {};
    int ret = spu_bridge_run(ls, &state, 100000, &stats);

    printf("  r3=%u (expect 0), r4=%u (expect 1000)\n", state.gpr[3][0], state.gpr[4][0]);
    printf("  Instructions: %d, Time: %.3f ms exec\n", ret, stats.exec_ms);
    if (stats.exec_ms > 0)
        printf("  Throughput: %.1f MIPS\n", stats.mips);

    bool pass = (state.gpr[3][0] == 0 && state.gpr[4][0] == 1000 && state.stopped == 1);
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    free(ls);
}

// ═══════════════════════════════════════════════════════════════
// Test 3: Cached Execution (second run should be faster)
// ═══════════════════════════════════════════════════════════════

static void test_bridge_cache() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: Cubin Cache (warm vs cold)    ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t* ls = (uint8_t*)calloc(1, 256 * 1024);
    SPUBridgeState state = {};

    // Simple program: il r3, 777; stop
    ls_store_insn(ls, 0x0000, spu_il(3, 777));
    ls_store_insn(ls, 0x0004, spu_stop());

    // Cold run
    SPUBridgeStats stats1 = {};
    spu_bridge_run(ls, &state, 100, &stats1);
    float cold_ms = stats1.total_ms;

    // Warm run (same LS → cache hit)
    state = {};
    SPUBridgeStats stats2 = {};
    spu_bridge_run(ls, &state, 100, &stats2);
    float warm_ms = stats2.total_ms;

    printf("  Cold: %.3f ms (compile + exec)\n", cold_ms);
    printf("  Warm: %.3f ms (cached exec)\n", warm_ms);
    printf("  Speedup: %.1fx\n", cold_ms / (warm_ms > 0 ? warm_ms : 0.001f));
    printf("  r3=%u (expect 777)\n", state.gpr[3][0]);

    bool pass = (state.gpr[3][0] == 777 && warm_ms < cold_ms);
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    free(ls);
}

// ═══════════════════════════════════════════════════════════════
// Test 4: Memory Operations (Load/Store Quadword)
// ═══════════════════════════════════════════════════════════════

static void test_bridge_memory() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 4: Memory Load/Store             ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t* ls = (uint8_t*)calloc(1, 256 * 1024);
    SPUBridgeState state = {};

    // Store 0xDEADBEEF at LS address 0x1000 (big-endian)
    uint32_t val = 0xDEADBEEF;
    uint32_t be = ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
                  ((val << 8) & 0xFF0000) | ((val << 24) & 0xFF000000);
    memcpy(ls + 0x1000, &be, 4);

    // Program: load quadword from addr in r2, read word[0] to verify
    // il r2, 0x100  (addr = 0x100 << 4 = 0x1000)
    // Actually lqd uses i10 << 4 + ra for address. Let's use lqd r3, 0(r2) with r2 = 0x1000
    // il r2, 0x1000 — but il is 16-bit signed, max 32767. 0x1000 = 4096, fits.
    ls_store_insn(ls, 0x0000, spu_il(2, 0x1000));
    // lqd r3, 0(r2) — op8 = 0x34, i10 = 0
    ls_store_insn(ls, 0x0004, spu_ri10(0x34, 0, 2, 3));
    // stop
    ls_store_insn(ls, 0x0008, spu_stop());

    SPUBridgeStats stats = {};
    spu_bridge_run(ls, &state, 100, &stats);

    // lqd loads 16 bytes (quadword) — word[0] should be 0xDEADBEEF
    printf("  r3[0]=0x%08X (expect 0xDEADBEEF)\n", state.gpr[3][0]);
    printf("  Time: %.3f ms\n", stats.exec_ms);

    bool pass = (state.gpr[3][0] == 0xDEADBEEF);
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    free(ls);
}

// ═══════════════════════════════════════════════════════════════
// Test 5: New Opcodes (CBX/CWX, MPYHHA, FESD/FRDS, DF*, FSCRRD)
// ═══════════════════════════════════════════════════════════════

// RI7: op11[31:21] i7[20:14] ra[13:7] rt[6:0]
static uint32_t spu_ri7(uint32_t op11, int32_t i7, uint32_t ra, uint32_t rt) {
    return (op11 << 21) | (((uint32_t)i7 & 0x7F) << 14) | ((ra & 0x7F) << 7) | (rt & 0x7F);
}

static void test_new_opcodes() {
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 5: New Opcodes (21 ops)          ║\n");
    printf("╚═══════════════════════════════════════╝\n");

    uint8_t* ls = (uint8_t*)calloc(1, 256 * 1024);
    SPUBridgeState state = {};
    int sub_pass = 0, sub_fail = 0;

    // ----- Test CBX (0x1D4): Generate Controls for Byte Insertion -----
    // r1 = 0, r2 = 5 → t = (~(0+5)) & 0xF = ~5 & 0xF = 0xA
    // Result: base pattern 0x10111213 14151617 18191A1B 1C1D1E1F with byte[10] = 0x03
    {
        memset(ls, 0, 256*1024);
        memset(&state, 0, sizeof(state));
        // il r1, 0
        ls_store_insn(ls, 0x0000, spu_il(1, 0));
        // il r2, 5
        ls_store_insn(ls, 0x0004, spu_il(2, 5));
        // cbx r3, r1, r2 — op11=0x1D4
        ls_store_insn(ls, 0x0008, spu_rr(0x1D4, 2, 1, 3));
        ls_store_insn(ls, 0x000C, spu_stop());
        SPUBridgeStats stats = {};
        spu_bridge_run(ls, &state, 100, &stats);
        // Byte index 10 in big-endian: r3[2] bits [23:16] = 0x03
        // word 2 (index 2) = 0x18191A1B, byte 10 = high byte of lower half = bits[23:16]
        // Actually: byte 10 in 16-byte reg: word[10/4]=word[2], within word: byte 10%4=2 → bits[15:8]
        // Big endian mapping: byte 0 = word[0] bits[31:24], byte 1 = bits[23:16], etc.
        // Byte 10: word[2], sub-byte 2 → bits[15:8]
        // So word[2] = 0x1803_1A1B → actually the CBX sets it to 0x03
        // Expected: word[2] should differ from 0x18191A1B
        uint32_t w0 = state.gpr[3][0], w1 = state.gpr[3][1], w2 = state.gpr[3][2], w3 = state.gpr[3][3];
        bool ok = (w0 == 0x10111213 && w1 == 0x14151617 && w3 == 0x1C1D1E1F);
        // Byte 10 is in word 2 (bytes 8-11), sub-byte 2, which is bits[15:8]
        ok = ok && ((w2 & 0xFF00) == 0x0300);  // byte[10] = 0x03, rest should be normal
        ok = ok && ((w2 & 0xFFFF00FF) == 0x18190A1B || (w2 & 0x0000FF00) == 0x0300);
        // Let's just check w2 has 0x03 somewhere where 0x1A was
        printf("  CBX: r3 = %08X %08X %08X %08X\n", w0, w1, w2, w3);
        // t = (~5) & 0xF = 0xFFFFFFFA & 0xF = 0xA = 10
        // byte[10] means word index 10/4=2, sub-byte 10%4=2
        // In our encoding: set_reg_byte(d, 10, 0x03)
        // set_reg_byte: byte 10 → r[10>>2]=r[2], shift = 8*(3-(10&3)) = 8*(3-2) = 8
        // So byte 10 = bits[15:8] of word[2] = 0x03
        // Expected word[2] = 0x1819_03_1B
        bool cbx_ok = (w0 == 0x10111213 && w1 == 0x14151617 && w2 == 0x1819031B && w3 == 0x1C1D1E1F);
        printf("  CBX: %s (expect %08X got %08X)\n", cbx_ok ? "✅" : "❌", 0x1819031B, w2);
        if (cbx_ok) sub_pass++; else sub_fail++;
    }

    // ----- Test CWX (0x1D6): Generate Controls for Word Insertion -----
    // r1 = 0, r2 = 4 → t = (~(0+4)) & 0xC) >> 2 = (0xFFFFFFFB & 0xC) >> 2 = (0x8)>>2 = 2
    {
        memset(ls, 0, 256*1024);
        memset(&state, 0, sizeof(state));
        ls_store_insn(ls, 0x0000, spu_il(1, 0));
        ls_store_insn(ls, 0x0004, spu_il(2, 4));
        ls_store_insn(ls, 0x0008, spu_rr(0x1D6, 2, 1, 3));
        ls_store_insn(ls, 0x000C, spu_stop());
        SPUBridgeStats stats = {};
        spu_bridge_run(ls, &state, 100, &stats);
        // t = (~4 & 0xC) >> 2 = (0xFB & 0xC) >> 2 = 0x8 >> 2 = 2
        // word[2] = 0x00010203, rest = base pattern
        bool cwx_ok = (state.gpr[3][0] == 0x10111213 && state.gpr[3][1] == 0x14151617 &&
                       state.gpr[3][2] == 0x00010203 && state.gpr[3][3] == 0x1C1D1E1F);
        printf("  CWX: r3 = %08X %08X %08X %08X\n",
               state.gpr[3][0], state.gpr[3][1], state.gpr[3][2], state.gpr[3][3]);
        printf("  CWX: %s\n", cwx_ok ? "✅" : "❌");
        if (cwx_ok) sub_pass++; else sub_fail++;
    }

    // ----- Test MPYHHA (0x346): Multiply High signed + Accumulate -----
    // r1[w] = 0x00030000 (high hw = 3), r2[w] = 0x00050000 (high hw = 5)
    // r3[w] = 10 (accumulator)
    // Result: r3[w] = 10 + (3 * 5) = 25
    {
        memset(ls, 0, 256*1024);
        memset(&state, 0, sizeof(state));
        for (int w = 0; w < 4; w++) {
            state.gpr[1][w] = 0x00030000;
            state.gpr[2][w] = 0x00050000;
            state.gpr[3][w] = 10;
        }
        // mpyhha r3, r1, r2 — op11=0x346
        ls_store_insn(ls, 0x0000, spu_rr(0x346, 2, 1, 3));
        ls_store_insn(ls, 0x0004, spu_stop());
        SPUBridgeStats stats = {};
        spu_bridge_run(ls, &state, 100, &stats);
        bool mpyhha_ok = true;
        for (int w = 0; w < 4; w++) {
            if (state.gpr[3][w] != 25) mpyhha_ok = false;
        }
        printf("  MPYHHA: r3 = %u %u %u %u (expect 25)\n",
               state.gpr[3][0], state.gpr[3][1], state.gpr[3][2], state.gpr[3][3]);
        printf("  MPYHHA: %s\n", mpyhha_ok ? "✅" : "❌");
        if (mpyhha_ok) sub_pass++; else sub_fail++;
    }

    // ----- Test FESD (0x3B8): Float Extend Single to Double -----
    // r1[0] = float(3.14), r1[2] = float(2.71) → r3 = {double(3.14), double(2.71)}
    {
        memset(ls, 0, 256*1024);
        memset(&state, 0, sizeof(state));
        float f0 = 3.14f, f1 = 2.71f;
        uint32_t u0, u1;
        memcpy(&u0, &f0, 4); memcpy(&u1, &f1, 4);
        state.gpr[1][0] = u0; state.gpr[1][1] = 0; state.gpr[1][2] = u1; state.gpr[1][3] = 0;
        // fesd r3, r1 — op11=0x3B8, rb=0
        ls_store_insn(ls, 0x0000, spu_rr(0x3B8, 0, 1, 3));
        ls_store_insn(ls, 0x0004, spu_stop());
        SPUBridgeStats stats = {};
        spu_bridge_run(ls, &state, 100, &stats);
        // Reconstruct doubles
        uint64_t d0 = ((uint64_t)state.gpr[3][0] << 32) | state.gpr[3][1];
        uint64_t d1 = ((uint64_t)state.gpr[3][2] << 32) | state.gpr[3][3];
        double rd0, rd1; memcpy(&rd0, &d0, 8); memcpy(&rd1, &d1, 8);
        bool fesd_ok = (rd0 > 3.13 && rd0 < 3.15 && rd1 > 2.70 && rd1 < 2.72);
        printf("  FESD: %.6f %.6f (expect ~3.14, ~2.71)\n", rd0, rd1);
        printf("  FESD: %s\n", fesd_ok ? "✅" : "❌");
        if (fesd_ok) sub_pass++; else sub_fail++;
    }

    // ----- Test FRDS (0x3B9): Float Round Double to Single -----
    // r1 = {double(1.5), double(2.5)} → r3[0] = float(1.5), r3[2] = float(2.5)
    {
        memset(ls, 0, 256*1024);
        memset(&state, 0, sizeof(state));
        double d0 = 1.5, d1 = 2.5;
        uint64_t u0, u1; memcpy(&u0, &d0, 8); memcpy(&u1, &d1, 8);
        state.gpr[1][0] = (uint32_t)(u0 >> 32); state.gpr[1][1] = (uint32_t)u0;
        state.gpr[1][2] = (uint32_t)(u1 >> 32); state.gpr[1][3] = (uint32_t)u1;
        ls_store_insn(ls, 0x0000, spu_rr(0x3B9, 0, 1, 3));
        ls_store_insn(ls, 0x0004, spu_stop());
        SPUBridgeStats stats = {};
        spu_bridge_run(ls, &state, 100, &stats);
        float rf0, rf1;
        uint32_t ru0 = state.gpr[3][0], ru1 = state.gpr[3][2];
        memcpy(&rf0, &ru0, 4); memcpy(&rf1, &ru1, 4);
        bool frds_ok = (rf0 == 1.5f && rf1 == 2.5f && state.gpr[3][1] == 0 && state.gpr[3][3] == 0);
        printf("  FRDS: %.1f %.1f (expect 1.5, 2.5)\n", rf0, rf1);
        printf("  FRDS: %s\n", frds_ok ? "✅" : "❌");
        if (frds_ok) sub_pass++; else sub_fail++;
    }

    // ----- Test DFA (0x2CC) already exists, but test DFMA (0x35C) -----
    // rt = 1.0, ra = 2.0, rb = 3.0 → rt = fma(2.0, 3.0, 1.0) = 7.0
    {
        memset(ls, 0, 256*1024);
        memset(&state, 0, sizeof(state));
        double t = 1.0, a = 2.0, b = 3.0;
        uint64_t ut, ua, ub;
        memcpy(&ut, &t, 8); memcpy(&ua, &a, 8); memcpy(&ub, &b, 8);
        // rt=3 (accumulator), ra=1, rb=2
        state.gpr[1][0] = (uint32_t)(ua >> 32); state.gpr[1][1] = (uint32_t)ua;
        state.gpr[1][2] = (uint32_t)(ua >> 32); state.gpr[1][3] = (uint32_t)ua;  // both doubles = 2.0
        state.gpr[2][0] = (uint32_t)(ub >> 32); state.gpr[2][1] = (uint32_t)ub;
        state.gpr[2][2] = (uint32_t)(ub >> 32); state.gpr[2][3] = (uint32_t)ub;
        state.gpr[3][0] = (uint32_t)(ut >> 32); state.gpr[3][1] = (uint32_t)ut;
        state.gpr[3][2] = (uint32_t)(ut >> 32); state.gpr[3][3] = (uint32_t)ut;
        // dfma r3, r1, r2 — rt += ra*rb
        ls_store_insn(ls, 0x0000, spu_rr(0x35C, 2, 1, 3));
        ls_store_insn(ls, 0x0004, spu_stop());
        SPUBridgeStats stats = {};
        spu_bridge_run(ls, &state, 100, &stats);
        uint64_t res0 = ((uint64_t)state.gpr[3][0] << 32) | state.gpr[3][1];
        double dr0; memcpy(&dr0, &res0, 8);
        bool dfma_ok = (dr0 == 7.0);
        printf("  DFMA: %.1f (expect 7.0, fma(2*3+1))\n", dr0);
        printf("  DFMA: %s\n", dfma_ok ? "✅" : "❌");
        if (dfma_ok) sub_pass++; else sub_fail++;
    }

    // ----- Test DFCEQ (0x3C3): Double FP Compare Equal -----
    {
        memset(ls, 0, 256*1024);
        memset(&state, 0, sizeof(state));
        double v = 42.0;
        uint64_t uv; memcpy(&uv, &v, 8);
        for (int w = 0; w < 4; w++) {
            state.gpr[1][w] = (w % 2 == 0) ? (uint32_t)(uv >> 32) : (uint32_t)uv;
            state.gpr[2][w] = (w % 2 == 0) ? (uint32_t)(uv >> 32) : (uint32_t)uv;
        }
        ls_store_insn(ls, 0x0000, spu_rr(0x3C3, 2, 1, 3));
        ls_store_insn(ls, 0x0004, spu_stop());
        SPUBridgeStats stats = {};
        spu_bridge_run(ls, &state, 100, &stats);
        // Both doubles equal → both doublewords should be all 1s
        bool dfceq_ok = (state.gpr[3][0] == 0xFFFFFFFF && state.gpr[3][1] == 0xFFFFFFFF &&
                         state.gpr[3][2] == 0xFFFFFFFF && state.gpr[3][3] == 0xFFFFFFFF);
        printf("  DFCEQ: %08X %08X %08X %08X (expect all FF)\n",
               state.gpr[3][0], state.gpr[3][1], state.gpr[3][2], state.gpr[3][3]);
        printf("  DFCEQ: %s\n", dfceq_ok ? "✅" : "❌");
        if (dfceq_ok) sub_pass++; else sub_fail++;
    }

    // ----- Test FSCRRD (0x398): Read FP Status (stub returns 0) -----
    {
        memset(ls, 0, 256*1024);
        memset(&state, 0, sizeof(state));
        state.gpr[3][0] = 0xDEAD; // pre-fill to verify it gets cleared
        ls_store_insn(ls, 0x0000, spu_rr(0x398, 0, 0, 3));
        ls_store_insn(ls, 0x0004, spu_stop());
        SPUBridgeStats stats = {};
        spu_bridge_run(ls, &state, 100, &stats);
        bool fscrrd_ok = (state.gpr[3][0] == 0 && state.gpr[3][1] == 0 &&
                          state.gpr[3][2] == 0 && state.gpr[3][3] == 0);
        printf("  FSCRRD: %08X (expect 0)\n", state.gpr[3][0]);
        printf("  FSCRRD: %s\n", fscrrd_ok ? "✅" : "❌");
        if (fscrrd_ok) sub_pass++; else sub_fail++;
    }

    printf("\n  Subtotal: %d/%d new opcode tests passed\n", sub_pass, sub_pass + sub_fail);
    bool pass = (sub_fail == 0);
    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    free(ls);
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  🎮 RPCS3 SPU CUDA Bridge Test Suite     ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Simulates RPCS3 → GPU SPU offloading    ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    if (!spu_bridge_available()) {
        printf("  ❌ No CUDA GPU available\n");
        return 1;
    }

    spu_bridge_init();

    test_bridge_alu();
    test_bridge_loop();
    test_bridge_cache();
    test_bridge_memory();
    test_new_opcodes();

    spu_bridge_print_stats();
    spu_bridge_shutdown();

    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", total_pass, total_fail);
    printf("═══════════════════════════════════════════\n");

    return total_fail > 0 ? 1 : 0;
}
