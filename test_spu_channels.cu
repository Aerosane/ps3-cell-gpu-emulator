// test_spu_channels.cu — SPU Channel System Tests
//
// Validates mailbox FIFO, MFC command sequencing, signal notification,
// tag tracking, atomic status, and integrated DMA flow.

#include "spu_channels.h"
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

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  📡 SPU Channel System Test Suite        ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Mailbox + MFC + Signals + Events        ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    uint8_t* ls = (uint8_t*)calloc(1, 256 * 1024);
    uint8_t* main_mem = (uint8_t*)calloc(1, 16 * 1024 * 1024);
    uint32_t main_size = 16 * 1024 * 1024;

    // ====================================================
    printf("═══ TEST 1: Mailbox Communication ═══\n");
    {
        SPUChannelState ch;
        spu_channels_init(&ch);

        // PPE writes 3 values to SPU inbound mailbox
        CHECK(spu_ppe_write_inbound(&ch, 0xDEAD), "PPE → inbound[0]");
        CHECK(spu_ppe_write_inbound(&ch, 0xBEEF), "PPE → inbound[1]");
        CHECK(spu_ppe_write_inbound(&ch, 0xCAFE), "PPE → inbound[2]");

        // RCHCNT should show 3 available
        CHECK(spu_rchcnt(&ch, SPU_RdInMbox) == 3, "RCHCNT inbound = 3");

        // SPU reads from inbound
        uint32_t val;
        CHECK(spu_rdch(&ch, SPU_RdInMbox, &val) == 0 && val == 0xDEAD, "SPU RDCH inbound = 0xDEAD");
        CHECK(spu_rdch(&ch, SPU_RdInMbox, &val) == 0 && val == 0xBEEF, "SPU RDCH inbound = 0xBEEF");

        // SPU writes to outbound
        CHECK(spu_wrch(&ch, SPU_WrOutMbox, 0x1234, ls, main_mem, main_size) == 0,
              "SPU WRCH outbound = 0x1234");

        // Outbound full (capacity 1) — should block
        CHECK(spu_wrch(&ch, SPU_WrOutMbox, 0x5678, ls, main_mem, main_size) == 1,
              "SPU WRCH outbound blocks (full)");

        // PPE reads outbound
        CHECK(spu_ppe_read_outbound(&ch, &val) && val == 0x1234, "PPE reads outbound = 0x1234");

        // Now outbound has room
        CHECK(spu_rchcnt(&ch, SPU_WrOutMbox) == 1, "RCHCNT outbound = 1 (available)");

        // Interrupt mailbox
        CHECK(spu_wrch(&ch, SPU_WrOutIntrMbox, 0xFFFF, ls, main_mem, main_size) == 0,
              "SPU WRCH interrupt mbox");
        CHECK(spu_ppe_read_interrupt(&ch, &val) && val == 0xFFFF, "PPE reads interrupt = 0xFFFF");

        // Inbound empty blocks
        uint32_t dummy;
        spu_rdch(&ch, SPU_RdInMbox, &dummy); // read 3rd
        CHECK(spu_rdch(&ch, SPU_RdInMbox, &dummy) == 1, "RDCH inbound blocks (empty)");

        // Fill inbound to capacity (4)
        spu_ppe_write_inbound(&ch, 1);
        spu_ppe_write_inbound(&ch, 2);
        spu_ppe_write_inbound(&ch, 3);
        spu_ppe_write_inbound(&ch, 4);
        CHECK(!spu_ppe_write_inbound(&ch, 5), "Inbound full at 4 (rejects 5th)");
    }

    // ====================================================
    printf("\n═══ TEST 2: MFC Command Sequence ═══\n");
    {
        SPUChannelState ch;
        spu_channels_init(&ch);

        // Fill main memory with pattern
        memset(main_mem + 0x10000, 0xAB, 256);

        // SPU builds MFC GET command via channel writes
        // (This is how real SPU code issues DMA)
        spu_wrch(&ch, MFC_LSA,   0x1000, ls, main_mem, main_size);
        spu_wrch(&ch, MFC_EAH,   0,      ls, main_mem, main_size);
        spu_wrch(&ch, MFC_EAL,   0x10000,ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Size,  256,    ls, main_mem, main_size);
        spu_wrch(&ch, MFC_TagID, 5,      ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Cmd,   MFC_GET_CMD, ls, main_mem, main_size);

        CHECK(ls[0x1000] == 0xAB && ls[0x10FF] == 0xAB, "MFC GET via channels: data in LS");

        // Check tag status
        spu_wrch(&ch, MFC_WrTagMask, (1u << 5), ls, main_mem, main_size);
        uint32_t stat;
        spu_rdch(&ch, MFC_RdTagStat, &stat);
        CHECK(stat == (1u << 5), "Tag 5 completed after GET");

        // Now PUT data back to different location
        memset(ls + 0x1000, 0xCD, 128);
        spu_wrch(&ch, MFC_LSA,   0x1000, ls, main_mem, main_size);
        spu_wrch(&ch, MFC_EAL,   0x20000,ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Size,  128,    ls, main_mem, main_size);
        spu_wrch(&ch, MFC_TagID, 7,      ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Cmd,   MFC_PUT_CMD, ls, main_mem, main_size);

        CHECK(main_mem[0x20000] == 0xCD && main_mem[0x2007F] == 0xCD,
              "MFC PUT via channels: data in main mem");
    }

    // ====================================================
    printf("\n═══ TEST 3: GETLLAR/PUTLLC via Channels ═══\n");
    {
        SPUChannelState ch;
        spu_channels_init(&ch);

        // Set up 128-byte line
        for (int i = 0; i < 128; i++) main_mem[0x30000 + i] = (uint8_t)i;

        // GETLLAR
        spu_wrch(&ch, MFC_LSA,   0x2000, ls, main_mem, main_size);
        spu_wrch(&ch, MFC_EAL,   0x30000,ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Size,  128,    ls, main_mem, main_size);
        spu_wrch(&ch, MFC_TagID, 0,      ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Cmd,   MFC_GETLLAR_CMD, ls, main_mem, main_size);

        uint32_t stat;
        spu_rdch(&ch, MFC_RdAtomicStat, &stat);
        CHECK(stat == 4, "GETLLAR atomic status = 4 (success)");
        CHECK(ls[0x2000] == 0 && ls[0x2001] == 1, "GETLLAR data in LS");

        // Modify and PUTLLC
        ls[0x2000] = 0xFF;
        spu_wrch(&ch, MFC_LSA,   0x2000, ls, main_mem, main_size);
        spu_wrch(&ch, MFC_EAL,   0x30000,ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Size,  128,    ls, main_mem, main_size);
        spu_wrch(&ch, MFC_TagID, 0,      ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Cmd,   MFC_PUTLLC_CMD, ls, main_mem, main_size);

        spu_rdch(&ch, MFC_RdAtomicStat, &stat);
        CHECK(stat == 0, "PUTLLC atomic status = 0 (success)");
        CHECK(main_mem[0x30000] == 0xFF, "PUTLLC wrote to main mem");
    }

    // ====================================================
    printf("\n═══ TEST 4: Signal Notification ═══\n");
    {
        SPUChannelState ch;
        spu_channels_init(&ch);

        // No signal pending — should block
        uint32_t val;
        CHECK(spu_rdch(&ch, SPU_RdSigNotify1, &val) == 1, "Signal1 blocks (not pending)");
        CHECK(spu_rchcnt(&ch, SPU_RdSigNotify1) == 0, "RCHCNT signal1 = 0");

        // PPE sends signal (overwrite mode)
        spu_ppe_signal1(&ch, 0xAAAA, true);
        CHECK(spu_rchcnt(&ch, SPU_RdSigNotify1) == 1, "RCHCNT signal1 = 1 after send");
        CHECK(spu_rdch(&ch, SPU_RdSigNotify1, &val) == 0 && val == 0xAAAA,
              "Signal1 read = 0xAAAA");
        CHECK(spu_rchcnt(&ch, SPU_RdSigNotify1) == 0, "Signal1 consumed");

        // OR mode: multiple sends accumulate
        spu_ppe_signal2(&ch, 0x00F0, false);
        spu_ppe_signal2(&ch, 0x0F00, false);
        CHECK(spu_rdch(&ch, SPU_RdSigNotify2, &val) == 0 && val == 0x0FF0,
              "Signal2 OR mode = 0x0FF0");
    }

    // ====================================================
    printf("\n═══ TEST 5: Event System ═══\n");
    {
        SPUChannelState ch;
        spu_channels_init(&ch);

        // Set event mask for tag group + signal1
        spu_wrch(&ch, SPU_WrEventMask, SPU_EVENT_TG | SPU_EVENT_S1,
                 ls, main_mem, main_size);

        uint32_t val;
        spu_rdch(&ch, SPU_RdEventMask, &val);
        CHECK(val == (SPU_EVENT_TG | SPU_EVENT_S1), "Event mask set correctly");

        // No events yet
        spu_rdch(&ch, SPU_RdEventStat, &val);
        CHECK(val == 0, "No events pending");

        // Trigger signal → sets S1 event
        spu_ppe_signal1(&ch, 0x42, true);
        spu_rdch(&ch, SPU_RdEventStat, &val);
        CHECK(val == SPU_EVENT_S1, "S1 event raised after signal");

        // Acknowledge event
        spu_wrch(&ch, SPU_WrEventAck, SPU_EVENT_S1, ls, main_mem, main_size);
        spu_rdch(&ch, SPU_RdEventStat, &val);
        CHECK(val == 0, "S1 event cleared after ack");

        // Tag update raises TG event
        spu_wrch(&ch, MFC_WrTagUpdate, MFC_TAG_UPDATE_IMMEDIATE,
                 ls, main_mem, main_size);
        spu_rdch(&ch, SPU_RdEventStat, &val);
        CHECK(val == SPU_EVENT_TG, "TG event raised after tag update");
    }

    // ====================================================
    printf("\n═══ TEST 6: Decrementer + SRR0 ═══\n");
    {
        SPUChannelState ch;
        spu_channels_init(&ch);

        spu_wrch(&ch, SPU_WrDec, 1000000, ls, main_mem, main_size);
        uint32_t val;
        spu_rdch(&ch, SPU_RdDec, &val);
        CHECK(val == 1000000, "Decrementer read back");

        spu_wrch(&ch, SPU_WrSRR0, 0x400, ls, main_mem, main_size);
        spu_rdch(&ch, SPU_RdSRR0, &val);
        CHECK(val == 0x400, "SRR0 read back");

        spu_rdch(&ch, SPU_RdMachStat, &val);
        CHECK(val == 1, "Machine status = running");
    }

    // ====================================================
    printf("\n═══ TEST 7: Full DMA Round-Trip ═══\n");
    {
        SPUChannelState ch;
        spu_channels_init(&ch);

        // Simulate a typical SPU workload:
        // 1. GET data from main memory
        // 2. Process it (modify in LS)
        // 3. PUT results back

        const char* input = "GPU-accelerated SPU emulation!";
        uint32_t len = (uint32_t)strlen(input) + 1;
        memcpy(main_mem + 0x50000, input, len);

        // GET
        spu_wrch(&ch, MFC_LSA,   0x3000, ls, main_mem, main_size);
        spu_wrch(&ch, MFC_EAH,   0,      ls, main_mem, main_size);
        spu_wrch(&ch, MFC_EAL,   0x50000,ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Size,  64,     ls, main_mem, main_size);
        spu_wrch(&ch, MFC_TagID, 15,     ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Cmd,   MFC_GET_CMD, ls, main_mem, main_size);

        // Wait for tag 15
        spu_wrch(&ch, MFC_WrTagMask, (1u << 15), ls, main_mem, main_size);
        uint32_t stat;
        spu_rdch(&ch, MFC_RdTagStat, &stat);
        CHECK(stat == (1u << 15), "Round-trip: GET complete");

        // "Process" — uppercase first byte
        ls[0x3000] = 'g'; // lowercase the G

        // PUT result
        spu_wrch(&ch, MFC_LSA,   0x3000, ls, main_mem, main_size);
        spu_wrch(&ch, MFC_EAL,   0x60000,ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Size,  64,     ls, main_mem, main_size);
        spu_wrch(&ch, MFC_TagID, 16,     ls, main_mem, main_size);
        spu_wrch(&ch, MFC_Cmd,   MFC_PUT_CMD, ls, main_mem, main_size);

        // Send result notification via mailbox
        spu_wrch(&ch, SPU_WrOutMbox, 0x60000, ls, main_mem, main_size);

        // PPE side: read mailbox to get result address
        uint32_t result_addr;
        spu_ppe_read_outbound(&ch, &result_addr);
        CHECK(result_addr == 0x60000, "Round-trip: PPE got result addr");
        CHECK(main_mem[0x60000] == 'g', "Round-trip: processed data in main mem");
        CHECK(memcmp(main_mem + 0x60001, input + 1, len - 1) == 0,
              "Round-trip: rest of data intact");
    }

    // ====================================================
    printf("\n═══ TEST 8: Throughput Benchmark ═══\n");
    {
        SPUChannelState ch;
        spu_channels_init(&ch);
        memset(main_mem, 0xDD, 4096);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100000; i++) {
            spu_wrch(&ch, MFC_LSA,   0, ls, main_mem, main_size);
            spu_wrch(&ch, MFC_EAL,   0, ls, main_mem, main_size);
            spu_wrch(&ch, MFC_Size,  4096, ls, main_mem, main_size);
            spu_wrch(&ch, MFC_TagID, 0, ls, main_mem, main_size);
            spu_wrch(&ch, MFC_Cmd,   MFC_GET_CMD, ls, main_mem, main_size);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double cmd_per_sec = 100000.0 / (ms / 1000.0);
        double gbps = (100000.0 * 4096.0) / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        printf("  100K MFC GETs (4KB): %.1f ms — %.0f cmd/s, %.1f GB/s\n",
               ms, cmd_per_sec, gbps);
        CHECK(cmd_per_sec > 100000, "MFC throughput > 100K cmd/s");

        spu_channels_print_stats(&ch);
    }

    // ====================================================
    printf("\n═══════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════\n");

    free(ls);
    free(main_mem);
    return tests_failed > 0 ? 1 : 0;
}
