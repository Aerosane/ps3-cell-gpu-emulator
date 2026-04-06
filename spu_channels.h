// spu_channels.h — SPU Channel System for CUDA Backend
//
// Implements SPU ↔ PPE communication channels:
//   - Mailboxes (inbound/outbound/interrupt)
//   - MFC command parameter registers (LSA, EAH, EAL, Size, TagID, Cmd)
//   - Tag mask/status for DMA completion tracking
//   - Signal notification registers
//   - Event system (decrementer, reservation lost, etc.)
//   - Decrementer timer
//
// Integrates with spu_mfc_dma.h for DMA command dispatch.

#pragma once
#include "spu_mfc_dma.h"
#include <cstdint>
#include <cstring>
#include <cstdio>

// ============================================================
// SPU Channel Numbers (matching RPCS3)
// ============================================================

enum SPUChannel : uint32_t {
    SPU_RdEventStat     = 0,
    SPU_WrEventMask     = 1,
    SPU_WrEventAck      = 2,
    SPU_RdSigNotify1    = 3,
    SPU_RdSigNotify2    = 4,
    SPU_WrDec           = 7,
    SPU_RdDec           = 8,
    SPU_RdEventMask     = 11,
    SPU_RdMachStat      = 13,
    SPU_WrSRR0          = 14,
    SPU_RdSRR0          = 15,
    MFC_LSA             = 16,
    MFC_EAH             = 17,
    MFC_EAL             = 18,
    MFC_Size            = 19,
    MFC_TagID           = 20,
    MFC_Cmd             = 21,
    MFC_WrTagMask       = 22,
    MFC_WrTagUpdate     = 23,
    MFC_RdTagStat       = 24,
    MFC_RdListStallStat = 25,
    MFC_WrListStallAck  = 26,
    MFC_RdAtomicStat    = 27,
    SPU_WrOutMbox       = 28,
    SPU_RdInMbox        = 29,
    SPU_WrOutIntrMbox   = 30,
};

// Tag update modes
enum MFCTagUpdate : uint32_t {
    MFC_TAG_UPDATE_IMMEDIATE = 0,
    MFC_TAG_UPDATE_ANY       = 1,
    MFC_TAG_UPDATE_ALL       = 2,
};

// SPU Events
enum SPUEvent : uint32_t {
    SPU_EVENT_LR = 0x400,   // Lock Line Reservation Lost
    SPU_EVENT_S1 = 0x200,   // Signal Notification 1 available
    SPU_EVENT_S2 = 0x100,   // Signal Notification 2 available
    SPU_EVENT_LE = 0x80,    // Outbound Mailbox available
    SPU_EVENT_ME = 0x40,    // Outbound Interrupt Mailbox available
    SPU_EVENT_TM = 0x20,    // Decrementer negative
    SPU_EVENT_MB = 0x10,    // Inbound Mailbox available
    SPU_EVENT_QV = 0x8,     // MFC Command Queue available
    SPU_EVENT_TG = 0x1,     // MFC Tag Group status update
};

// ============================================================
// Channel FIFO (for mailboxes)
// ============================================================

struct ChannelFIFO {
    uint32_t data[4];   // Up to 4 entries (inbound has 4, outbound has 1)
    uint32_t head;
    uint32_t count;
    uint32_t capacity;
};

static inline void fifo_init(ChannelFIFO* f, uint32_t cap) {
    memset(f, 0, sizeof(*f));
    f->capacity = cap;
}

static inline bool fifo_push(ChannelFIFO* f, uint32_t val) {
    if (f->count >= f->capacity) return false;
    f->data[(f->head + f->count) % f->capacity] = val;
    f->count++;
    return true;
}

static inline bool fifo_pop(ChannelFIFO* f, uint32_t* val) {
    if (f->count == 0) return false;
    *val = f->data[f->head % f->capacity];
    f->head = (f->head + 1) % f->capacity;
    f->count--;
    return true;
}

static inline bool fifo_empty(const ChannelFIFO* f) { return f->count == 0; }
static inline bool fifo_full(const ChannelFIFO* f) { return f->count >= f->capacity; }

// ============================================================
// SPU Channel State
// ============================================================

struct SPUChannelState {
    // Mailboxes
    ChannelFIFO in_mbox;        // PPE → SPU (4 entries)
    ChannelFIFO out_mbox;       // SPU → PPE (1 entry)
    ChannelFIFO out_intr_mbox;  // SPU → PPE interrupt (1 entry)

    // Signal notification registers
    uint32_t signal1;
    uint32_t signal2;
    bool     signal1_pending;
    bool     signal2_pending;

    // MFC command build registers (written sequentially before MFC_Cmd)
    uint32_t mfc_lsa;
    uint32_t mfc_eah;
    uint32_t mfc_eal;
    uint32_t mfc_size;
    uint32_t mfc_tag;

    // Tag tracking
    uint32_t tag_mask;
    uint32_t tag_update_mode;

    // Event system
    uint32_t event_mask;
    uint32_t event_status;

    // Decrementer (80MHz on PS3)
    uint32_t decrementer;
    uint64_t dec_start_tick;

    // SRR0 (save/restore register)
    uint32_t srr0;

    // Atomic status (for GETLLAR/PUTLLC results)
    uint32_t atomic_status;

    // MFC DMA engine
    MFCEngine mfc;

    // Stats
    uint64_t total_rdch;
    uint64_t total_wrch;
};

// ============================================================
// Initialize
// ============================================================

static inline void spu_channels_init(SPUChannelState* ch) {
    memset(ch, 0, sizeof(*ch));
    fifo_init(&ch->in_mbox, 4);
    fifo_init(&ch->out_mbox, 1);
    fifo_init(&ch->out_intr_mbox, 1);
    mfc_init(&ch->mfc);
    ch->mfc.tag_status = 0xFFFFFFFF;
}

// ============================================================
// WRCH — SPU writes to a channel
// ============================================================
// Returns:
//   0  = success
//   1  = channel would block (FIFO full / needs host intervention)
//   -1 = unknown channel

static inline int spu_wrch(SPUChannelState* ch, uint32_t channel, uint32_t value,
                            uint8_t* ls, uint8_t* main_mem, uint32_t main_size)
{
    ch->total_wrch++;

    switch (channel) {
    case SPU_WrEventMask:
        ch->event_mask = value;
        return 0;

    case SPU_WrEventAck:
        ch->event_status &= ~value;
        return 0;

    case SPU_WrDec:
        ch->decrementer = value;
        return 0;

    case SPU_WrSRR0:
        ch->srr0 = value;
        return 0;

    case MFC_LSA:
        ch->mfc_lsa = value & 0x3FFF0; // 16-byte aligned, 256KB
        return 0;

    case MFC_EAH:
        ch->mfc_eah = value;
        return 0;

    case MFC_EAL:
        ch->mfc_eal = value;
        return 0;

    case MFC_Size:
        ch->mfc_size = value & 0x7FFF; // Max 16KB
        return 0;

    case MFC_TagID:
        ch->mfc_tag = value & 0x1F; // 0-31
        return 0;

    case MFC_Cmd: {
        // Build and enqueue the MFC command
        MFCCommand cmd = {
            .cmd  = (uint8_t)(value & 0xFF),
            .tag  = (uint8_t)ch->mfc_tag,
            .size = (uint16_t)ch->mfc_size,
            .lsa  = ch->mfc_lsa,
            .eal  = ch->mfc_eal,
            .eah  = ch->mfc_eah,
        };

        // For atomic commands, process immediately
        if (cmd.cmd == MFC_GETLLAR_CMD || cmd.cmd == MFC_PUTLLC_CMD ||
            cmd.cmd == MFC_PUTLLUC_CMD || cmd.cmd == MFC_PUTQLLUC_CMD) {
            int rc = mfc_execute(&ch->mfc, &cmd, ls, main_mem, main_size);
            if (cmd.cmd == MFC_PUTLLC_CMD) {
                ch->atomic_status = (rc == 0) ? 0 : 1; // SUCCESS or FAILURE
            } else if (cmd.cmd == MFC_GETLLAR_CMD) {
                ch->atomic_status = 4; // MFC_GETLLAR_SUCCESS
            }
            return 0;
        }

        // For normal DMA, enqueue and process immediately (sequential model)
        if (mfc_enqueue(&ch->mfc, &cmd) == 0) {
            mfc_process_queue(&ch->mfc, ls, main_mem, main_size);
            return 0;
        }
        return 1; // Queue full
    }

    case MFC_WrTagMask:
        ch->tag_mask = value;
        ch->mfc.tag_mask = value;
        return 0;

    case MFC_WrTagUpdate:
        ch->tag_update_mode = value;
        // In our sequential model, tags complete immediately
        ch->event_status |= SPU_EVENT_TG;
        return 0;

    case MFC_WrListStallAck:
        // Acknowledge stall-and-notify — no-op in sequential model
        return 0;

    case SPU_WrOutMbox:
        if (fifo_push(&ch->out_mbox, value)) return 0;
        return 1; // Would block

    case SPU_RdInMbox:
        // Write to inbound mailbox is done from PPE side, not SPU
        // This shouldn't be called as WRCH
        return -1;

    case SPU_WrOutIntrMbox:
        if (fifo_push(&ch->out_intr_mbox, value)) return 0;
        return 1; // Would block

    default:
        return -1;
    }
}

// ============================================================
// RDCH — SPU reads from a channel
// ============================================================
// Returns:
//   0  = success, value written to *out
//   1  = channel would block (FIFO empty / not ready)
//   -1 = unknown channel

static inline int spu_rdch(SPUChannelState* ch, uint32_t channel, uint32_t* out) {
    ch->total_rdch++;

    switch (channel) {
    case SPU_RdEventStat:
        *out = ch->event_status & ch->event_mask;
        return 0;

    case SPU_RdEventMask:
        *out = ch->event_mask;
        return 0;

    case SPU_RdSigNotify1:
        if (ch->signal1_pending) {
            *out = ch->signal1;
            ch->signal1_pending = false;
            return 0;
        }
        return 1; // Would block

    case SPU_RdSigNotify2:
        if (ch->signal2_pending) {
            *out = ch->signal2;
            ch->signal2_pending = false;
            return 0;
        }
        return 1; // Would block

    case SPU_RdDec:
        *out = ch->decrementer; // Simplified: no tick-based decrement
        return 0;

    case SPU_RdMachStat:
        *out = 1; // Running
        return 0;

    case SPU_RdSRR0:
        *out = ch->srr0;
        return 0;

    case MFC_RdTagStat:
        *out = mfc_read_tag_status(&ch->mfc);
        return 0;

    case MFC_RdListStallStat:
        *out = 0; // No stalls in sequential model
        return 0;

    case MFC_RdAtomicStat:
        *out = ch->atomic_status;
        return 0;

    case SPU_RdInMbox:
        if (fifo_pop(&ch->in_mbox, out)) return 0;
        return 1; // Would block

    case SPU_WrOutMbox:
        // Read count (RCHCNT) — handled separately
        return -1;

    default:
        return -1;
    }
}

// ============================================================
// RCHCNT — Read channel count (available entries)
// ============================================================

static inline uint32_t spu_rchcnt(const SPUChannelState* ch, uint32_t channel) {
    switch (channel) {
    case SPU_RdEventStat:   return (ch->event_status & ch->event_mask) ? 1 : 0;
    case SPU_RdSigNotify1:  return ch->signal1_pending ? 1 : 0;
    case SPU_RdSigNotify2:  return ch->signal2_pending ? 1 : 0;
    case SPU_RdInMbox:      return ch->in_mbox.count;
    case SPU_WrOutMbox:     return ch->out_mbox.capacity - ch->out_mbox.count;
    case SPU_WrOutIntrMbox: return ch->out_intr_mbox.capacity - ch->out_intr_mbox.count;
    case MFC_RdTagStat:     return 1; // Always available in sequential model
    case MFC_RdAtomicStat:  return 1;
    case MFC_Cmd:           return 16 - ch->mfc.queue_size; // Queue vacancy
    default:                return 0;
    }
}

// ============================================================
// PPE-side operations (host calls these)
// ============================================================

// PPE writes to SPU inbound mailbox
static inline bool spu_ppe_write_inbound(SPUChannelState* ch, uint32_t value) {
    return fifo_push(&ch->in_mbox, value);
}

// PPE reads from SPU outbound mailbox
static inline bool spu_ppe_read_outbound(SPUChannelState* ch, uint32_t* value) {
    return fifo_pop(&ch->out_mbox, value);
}

// PPE reads from SPU interrupt mailbox
static inline bool spu_ppe_read_interrupt(SPUChannelState* ch, uint32_t* value) {
    return fifo_pop(&ch->out_intr_mbox, value);
}

// PPE sends signal notification
static inline void spu_ppe_signal1(SPUChannelState* ch, uint32_t value, bool overwrite) {
    if (overwrite) {
        ch->signal1 = value;
    } else {
        ch->signal1 |= value; // OR mode
    }
    ch->signal1_pending = true;
    ch->event_status |= SPU_EVENT_S1;
}

static inline void spu_ppe_signal2(SPUChannelState* ch, uint32_t value, bool overwrite) {
    if (overwrite) {
        ch->signal2 = value;
    } else {
        ch->signal2 |= value;
    }
    ch->signal2_pending = true;
    ch->event_status |= SPU_EVENT_S2;
}

// ============================================================
// Print channel stats
// ============================================================

static inline void spu_channels_print_stats(const SPUChannelState* ch) {
    printf("[CH] Stats: %lu RDCH, %lu WRCH\n", ch->total_rdch, ch->total_wrch);
    printf("[CH] Mailbox: in=%u/%u out=%u/%u intr=%u/%u\n",
           ch->in_mbox.count, ch->in_mbox.capacity,
           ch->out_mbox.count, ch->out_mbox.capacity,
           ch->out_intr_mbox.count, ch->out_intr_mbox.capacity);
    mfc_print_stats(&ch->mfc);
}
