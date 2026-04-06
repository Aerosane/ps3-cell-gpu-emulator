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

    spu_bridge_print_stats();
    spu_bridge_shutdown();

    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", total_pass, total_fail);
    printf("═══════════════════════════════════════════\n");

    return total_fail > 0 ? 1 : 0;
}
