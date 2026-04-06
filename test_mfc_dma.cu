// test_mfc_dma.cu — MFC DMA Engine Tests
//
// Tests the MFC DMA engine for SPU ↔ main memory transfers.
// Validates PUT, GET, GETLLAR, PUTLLC, list transfers, and tag tracking.

#include "spu_mfc_dma.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  ✅ %s\n", msg); tests_passed++; } \
    else { printf("  ❌ %s\n", msg); tests_failed++; } \
} while(0)

// Helper: byte-swap 32-bit for big-endian LS storage
static uint32_t bswap32(uint32_t v) {
    return ((v>>24)&0xFF)|((v>>8)&0xFF00)|((v<<8)&0xFF0000)|((v<<24)&0xFF000000);
}

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  🚀 MFC DMA Engine Test Suite            ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Tests SPU ↔ Main Memory transfers       ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    // Allocate simulated memory
    uint8_t* ls = (uint8_t*)calloc(1, 256 * 1024);        // 256KB SPU Local Store
    uint8_t* main_mem = (uint8_t*)calloc(1, 16 * 1024 * 1024); // 16MB main memory
    uint32_t main_size = 16 * 1024 * 1024;

    MFCEngine mfc;
    mfc_init(&mfc);

    // ====================================================
    printf("╔═══════════════════════════════════════╗\n");
    printf("║  TEST 1: MFC GET (Main → LS)           ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    {
        // Put test data in main memory
        const char* msg = "Hello from main memory!";
        memcpy(main_mem + 0x1000, msg, strlen(msg) + 1);

        MFCCommand cmd = {
            .cmd = MFC_GET_CMD,
            .tag = 0,
            .size = 32,
            .lsa = 0x100,
            .eal = 0x1000,
            .eah = 0,
        };

        mfc_execute(&mfc, &cmd, ls, main_mem, main_size);

        CHECK(memcmp(ls + 0x100, msg, strlen(msg)) == 0, "GET: data transferred to LS");
        CHECK(mfc.total_gets == 1, "GET count = 1");
        CHECK((mfc.tag_status & 1) != 0, "Tag 0 completed");
    }

    // ====================================================
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 2: MFC PUT (LS → Main)           ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    {
        // Put data in LS
        const char* msg = "SPU says hello!";
        memcpy(ls + 0x200, msg, strlen(msg) + 1);

        MFCCommand cmd = {
            .cmd = MFC_PUT_CMD,
            .tag = 1,
            .size = 32,
            .lsa = 0x200,
            .eal = 0x2000,
            .eah = 0,
        };

        mfc_execute(&mfc, &cmd, ls, main_mem, main_size);

        CHECK(memcmp(main_mem + 0x2000, msg, strlen(msg)) == 0, "PUT: data transferred to main mem");
        CHECK(mfc.total_puts == 1, "PUT count = 1");
        CHECK((mfc.tag_status & 2) != 0, "Tag 1 completed");
    }

    // ====================================================
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 3: GETLLAR + PUTLLC (Atomic)     ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    {
        // Set up a 128-byte cache line in main memory
        for (int i = 0; i < 128; i++) main_mem[0x3000 + i] = (uint8_t)i;

        // GETLLAR: reserve the line
        MFCCommand get_cmd = {
            .cmd = MFC_GETLLAR_CMD,
            .tag = 2,
            .size = 128,
            .lsa = 0x300,
            .eal = 0x3000,
            .eah = 0,
        };
        mfc_execute(&mfc, &get_cmd, ls, main_mem, main_size);

        CHECK(mfc.reservation_valid, "GETLLAR: reservation acquired");
        CHECK(mfc.reservation_addr == 0x3000, "GETLLAR: correct address");
        CHECK(ls[0x300] == 0 && ls[0x301] == 1 && ls[0x37F] == 127,
              "GETLLAR: data loaded to LS");

        // Modify the line in LS
        ls[0x300] = 0xFF;
        ls[0x301] = 0xAB;

        // PUTLLC: conditional write-back (should succeed)
        MFCCommand put_cmd = {
            .cmd = MFC_PUTLLC_CMD,
            .tag = 3,
            .size = 128,
            .lsa = 0x300,
            .eal = 0x3000,
            .eah = 0,
        };
        int rc = mfc_execute(&mfc, &put_cmd, ls, main_mem, main_size);

        CHECK(rc == 0, "PUTLLC: succeeded (reservation held)");
        CHECK(main_mem[0x3000] == 0xFF && main_mem[0x3001] == 0xAB,
              "PUTLLC: data written to main mem");
        CHECK(!mfc.reservation_valid, "PUTLLC: reservation consumed");

        // Try PUTLLC again (should fail — no reservation)
        MFCCommand put_cmd2 = {
            .cmd = MFC_PUTLLC_CMD,
            .tag = 4,
            .size = 128,
            .lsa = 0x300,
            .eal = 0x3000,
            .eah = 0,
        };
        rc = mfc_execute(&mfc, &put_cmd2, ls, main_mem, main_size);
        CHECK(rc == 1, "PUTLLC: failed (no reservation)");
    }

    // ====================================================
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 4: Contested PUTLLC              ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    {
        // Reserve a line
        for (int i = 0; i < 128; i++) main_mem[0x4000 + i] = (uint8_t)(i + 10);

        MFCCommand get_cmd = {
            .cmd = MFC_GETLLAR_CMD, .tag = 5, .size = 128,
            .lsa = 0x400, .eal = 0x4000, .eah = 0,
        };
        mfc_execute(&mfc, &get_cmd, ls, main_mem, main_size);

        // Simulate another SPU modifying the line (contested access)
        main_mem[0x4000] = 0xDE;

        // PUTLLC should fail because main_mem changed
        MFCCommand put_cmd = {
            .cmd = MFC_PUTLLC_CMD, .tag = 6, .size = 128,
            .lsa = 0x400, .eal = 0x4000, .eah = 0,
        };
        int rc = mfc_execute(&mfc, &put_cmd, ls, main_mem, main_size);
        CHECK(rc == 1, "PUTLLC: failed (line contested by other SPU)");
    }

    // ====================================================
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 5: Queue + Tag Tracking          ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    {
        MFCEngine mfc2;
        mfc_init(&mfc2);

        // Queue 3 commands with different tags
        for (int i = 0; i < 3; i++) {
            memset(main_mem + 0x5000 + i * 256, 0xA0 + i, 64);
            MFCCommand cmd = {
                .cmd = MFC_GET_CMD,
                .tag = (uint8_t)(10 + i),
                .size = 64,
                .lsa = (uint32_t)(0x500 + i * 0x100),
                .eal = (uint32_t)(0x5000 + i * 256),
                .eah = 0,
            };
            mfc_enqueue(&mfc2, &cmd);
        }

        CHECK(mfc2.queue_size == 3, "Queue: 3 commands pending");

        // Tags should be incomplete
        mfc2.tag_mask = (1u << 10) | (1u << 11) | (1u << 12);
        CHECK(mfc_read_tag_status(&mfc2) == 0, "Tags 10-12: incomplete before process");

        // Process queue
        int processed = mfc_process_queue(&mfc2, ls, main_mem, main_size);
        CHECK(processed == 3, "Queue: 3 commands processed");
        CHECK(mfc2.queue_size == 0, "Queue: empty after process");

        // Tags should now be complete
        uint32_t status = mfc_read_tag_status(&mfc2);
        CHECK(status == mfc2.tag_mask, "Tags 10-12: all complete");

        // Verify data
        CHECK(ls[0x500] == 0xA0 && ls[0x600] == 0xA1 && ls[0x700] == 0xA2,
              "Queue: all transfers correct");
    }

    // ====================================================
    printf("\n╔═══════════════════════════════════════╗\n");
    printf("║  TEST 6: Bandwidth Benchmark           ║\n");
    printf("╚═══════════════════════════════════════╝\n");
    {
        MFCEngine mfc3;
        mfc_init(&mfc3);

        // Fill main memory with pattern
        memset(main_mem, 0xCC, 16384);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; i++) {
            MFCCommand cmd = {
                .cmd = MFC_GET_CMD, .tag = 0, .size = 16384,
                .lsa = 0, .eal = 0, .eah = 0,
            };
            mfc_execute(&mfc3, &cmd, ls, main_mem, main_size);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbps = (10000.0 * 16384.0) / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        printf("  10K × 16KB GETs: %.1f ms (%.1f GB/s)\n", ms, gbps);
        CHECK(gbps > 1.0, "Bandwidth > 1 GB/s");
        CHECK(ls[0] == 0xCC, "Data verified");
    }

    // ====================================================
    mfc_print_stats(&mfc);

    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════\n");

    free(ls);
    free(main_mem);
    return tests_failed > 0 ? 1 : 0;
}
