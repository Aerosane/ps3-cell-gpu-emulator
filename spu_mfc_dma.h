// spu_mfc_dma.h — MFC DMA Engine for CUDA SPU Backend
//
// Implements SPU MFC (Memory Flow Controller) DMA commands.
// Handles data transfer between main memory (host) and SPU Local Store (GPU).
//
// MFC commands are queued by the SPU via WRCH to channel MFC_Cmd (ch21).
// The DMA engine processes them on the host side, performing cudaMemcpy
// between host RAM and GPU device memory.
//
// Supported commands:
//   PUT  (0x20): LS → Main Memory (SPU writes to main mem)
//   GET  (0x40): Main Memory → LS (SPU reads from main mem)
//   PUTL (0x24): List PUT (scatter)
//   GETL (0x44): List GET (gather)
//   GETLLAR (0xD0): Get Lock Line and Reserve (atomic 128B)
//   PUTLLC  (0xB4): Put Lock Line Conditional (atomic 128B)
//   PUTLLUC (0xB0): Put Lock Line Unconditional
//   BARRIER (0xC0): DMA barrier
//   SYNC    (0xCC): DMA sync

#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <atomic>

// ============================================================
// MFC Command Opcodes (matching RPCS3's MFC enum)
// ============================================================

enum MFCCmd : uint8_t {
    MFC_PUT_CMD      = 0x20,
    MFC_PUTB_CMD     = 0x21,
    MFC_PUTF_CMD     = 0x22,
    MFC_PUTS_CMD     = 0x28,
    MFC_GET_CMD      = 0x40,
    MFC_GETB_CMD     = 0x41,
    MFC_GETF_CMD     = 0x42,
    MFC_GETS_CMD     = 0x48,
    MFC_PUTL_CMD     = 0x24,
    MFC_PUTLB_CMD    = 0x25,
    MFC_GETL_CMD     = 0x44,
    MFC_GETLB_CMD    = 0x45,
    MFC_GETLLAR_CMD  = 0xD0,
    MFC_PUTLLC_CMD   = 0xB4,
    MFC_PUTLLUC_CMD  = 0xB0,
    MFC_PUTQLLUC_CMD = 0xB8,
    MFC_SNDSIG_CMD   = 0xA0,
    MFC_BARRIER_CMD  = 0xC0,
    MFC_EIEIO_CMD    = 0xC8,
    MFC_SYNC_CMD     = 0xCC,
};

// MFC masks
static constexpr uint8_t MFC_BARRIER_MASK = 0x01;
static constexpr uint8_t MFC_FENCE_MASK   = 0x02;
static constexpr uint8_t MFC_LIST_MASK    = 0x04;

// ============================================================
// MFC Command Structure
// ============================================================

struct MFCCommand {
    uint8_t  cmd;       // MFC opcode
    uint8_t  tag;       // DMA tag (0-31) for tracking
    uint16_t size;      // Transfer size in bytes (max 16KB for normal, 128B for atomic)
    uint32_t lsa;       // Local Store Address (0-0x3FFFF)
    uint32_t eal;       // Effective Address Low (main memory)
    uint32_t eah;       // Effective Address High (usually 0 for PS3)
};

// ============================================================
// MFC DMA Engine
// ============================================================

struct MFCEngine {
    // Command queue (SPU can queue up to 16 commands)
    MFCCommand queue[16];
    uint32_t   queue_size;

    // Tag completion tracking (32 tags, bit per tag)
    uint32_t   tag_status;      // 1 = completed
    uint32_t   tag_mask;        // Mask for status queries

    // Reservation (for GETLLAR/PUTLLC atomics)
    uint32_t   reservation_addr;    // EA of reserved 128-byte line
    uint8_t    reservation_data[128]; // Cached data
    bool       reservation_valid;
    uint64_t   reservation_time;

    // Stats
    uint64_t   total_puts;
    uint64_t   total_gets;
    uint64_t   total_bytes;
    uint64_t   total_atomics;
};

// Initialize MFC engine
static inline void mfc_init(MFCEngine* mfc) {
    memset(mfc, 0, sizeof(*mfc));
    mfc->tag_status = 0xFFFFFFFF; // All tags initially complete
}

// Check if a command is a PUT variant
static inline bool mfc_is_put(uint8_t cmd) {
    return (cmd & 0x30) == 0x20; // PUT: 0x2X
}

// Check if a command is a GET variant
static inline bool mfc_is_get(uint8_t cmd) {
    return (cmd & 0x30) == 0x40; // GET: 0x4X (actually 0x40-0x4F)
}

// Check if a command is a list transfer
static inline bool mfc_is_list(uint8_t cmd) {
    return (cmd & MFC_LIST_MASK) != 0;
}

// ============================================================
// DMA Execution
// ============================================================

// Execute a single MFC DMA command.
// Transfers data between SPU Local Store and main memory.
//
// Parameters:
//   cmd       — MFC command to execute
//   ls        — pointer to 256KB SPU Local Store (host-side copy)
//   main_mem  — pointer to PS3 main memory (host-side, 256MB)
//   main_size — size of main memory in bytes
//
// Returns: 0 on success, -1 on error
static inline int mfc_execute(MFCEngine* mfc, const MFCCommand* cmd,
                               uint8_t* ls, uint8_t* main_mem, uint32_t main_size)
{
    uint32_t lsa = cmd->lsa & 0x3FFFF;   // Mask to 256KB LS
    uint32_t ea  = cmd->eal;              // Main memory address
    uint32_t size = cmd->size;

    // Validate addresses
    if (lsa + size > 0x40000) {
        fprintf(stderr, "[MFC] LSA overflow: 0x%X + %u > 256KB\n", lsa, size);
        return -1;
    }

    switch (cmd->cmd & ~(MFC_BARRIER_MASK | MFC_FENCE_MASK)) {
    case MFC_PUT_CMD:   // LS → Main Memory
    case MFC_PUTS_CMD:
        if (ea + size <= main_size) {
            memcpy(main_mem + ea, ls + lsa, size);
            mfc->total_puts++;
            mfc->total_bytes += size;
        } else {
            fprintf(stderr, "[MFC] PUT EA overflow: 0x%08X + %u\n", ea, size);
        }
        break;

    case MFC_GET_CMD:   // Main Memory → LS
    case MFC_GETS_CMD:
        if (ea + size <= main_size) {
            memcpy(ls + lsa, main_mem + ea, size);
            mfc->total_gets++;
            mfc->total_bytes += size;
        } else {
            fprintf(stderr, "[MFC] GET EA overflow: 0x%08X + %u\n", ea, size);
        }
        break;

    case MFC_GETLLAR_CMD: // Get Lock Line and Reserve (128 bytes)
        if (ea + 128 <= main_size) {
            memcpy(ls + lsa, main_mem + ea, 128);
            memcpy(mfc->reservation_data, main_mem + ea, 128);
            mfc->reservation_addr = ea;
            mfc->reservation_valid = true;
            mfc->total_atomics++;
        }
        break;

    case MFC_PUTLLC_CMD: // Put Lock Line Conditional (128 bytes)
        if (mfc->reservation_valid && mfc->reservation_addr == ea) {
            // Check if reservation still holds (compare cached vs current)
            if (ea + 128 <= main_size &&
                memcmp(mfc->reservation_data, main_mem + ea, 128) == 0) {
                // Reservation holds — write through
                memcpy(main_mem + ea, ls + lsa, 128);
                mfc->reservation_valid = false;
                mfc->total_atomics++;
                // Success: ch_atomic_stat = 0 (MFC_PUTLLC_SUCCESS)
                return 0;
            } else {
                // Reservation lost
                mfc->reservation_valid = false;
                return 1; // Signal failure (MFC_PUTLLC_FAILURE)
            }
        } else {
            mfc->reservation_valid = false;
            return 1; // No valid reservation
        }
        break;

    case MFC_PUTLLUC_CMD:  // Put Lock Line Unconditional
    case MFC_PUTQLLUC_CMD:
        if (ea + 128 <= main_size) {
            memcpy(main_mem + ea, ls + lsa, 128);
            mfc->reservation_valid = false;
            mfc->total_atomics++;
        }
        break;

    case MFC_BARRIER_CMD:
    case MFC_EIEIO_CMD:
    case MFC_SYNC_CMD:
        // Barriers/sync are no-ops in our sequential model
        break;

    case MFC_SNDSIG_CMD: // Send Signal
        // Signal notification — would need SPU signal channel support
        // For now, treat as no-op
        break;

    default:
        fprintf(stderr, "[MFC] Unknown cmd: 0x%02X\n", cmd->cmd);
        return -1;
    }

    // Mark tag as completed
    mfc->tag_status |= (1u << cmd->tag);
    return 0;
}

// Execute a list DMA command (scatter/gather).
// List entries are stored in LS at the specified LSA.
// Each entry is 8 bytes: { u16 notify, u16 size, u32 eal }
static inline int mfc_execute_list(MFCEngine* mfc, const MFCCommand* cmd,
                                    uint8_t* ls, uint8_t* main_mem, uint32_t main_size)
{
    uint32_t list_addr = cmd->lsa & 0x3FFFF;
    uint32_t list_size = cmd->size; // Number of bytes in list (each entry = 8 bytes)
    uint32_t num_entries = list_size / 8;

    bool is_get = mfc_is_get(cmd->cmd);

    for (uint32_t i = 0; i < num_entries; i++) {
        // Read list entry from LS (big-endian format)
        uint32_t entry_off = list_addr + i * 8;
        if (entry_off + 8 > 0x40000) break;

        // Entry format: [notify:16][size:16][eal:32] (big-endian in LS)
        uint16_t entry_size = (uint16_t)(ls[entry_off + 2] << 8) | ls[entry_off + 3];
        uint32_t entry_ea = (uint32_t)(ls[entry_off + 4] << 24) |
                            (uint32_t)(ls[entry_off + 5] << 16) |
                            (uint32_t)(ls[entry_off + 6] << 8)  |
                            (uint32_t)(ls[entry_off + 7]);

        if (entry_size == 0) continue;

        // The actual LS target for list transfers uses the EAL from the command
        // as the base, but in practice RPCS3 uses lsa from the command header
        // This is simplified — real SPU uses stall-and-notify for complex lists

        MFCCommand sub = {
            .cmd  = is_get ? MFC_GET_CMD : MFC_PUT_CMD,
            .tag  = cmd->tag,
            .size = entry_size,
            .lsa  = cmd->eal + i * entry_size, // Simplified: sequential LS layout
            .eal  = entry_ea,
            .eah  = cmd->eah,
        };
        mfc_execute(mfc, &sub, ls, main_mem, main_size);
    }

    mfc->tag_status |= (1u << cmd->tag);
    return 0;
}

// Enqueue an MFC command. Returns 0 on success, -1 if queue full.
static inline int mfc_enqueue(MFCEngine* mfc, const MFCCommand* cmd) {
    if (mfc->queue_size >= 16) {
        fprintf(stderr, "[MFC] Queue full!\n");
        return -1;
    }
    mfc->queue[mfc->queue_size++] = *cmd;
    // Clear tag completion for this tag
    mfc->tag_status &= ~(1u << cmd->tag);
    return 0;
}

// Process all queued commands. Returns number processed.
static inline int mfc_process_queue(MFCEngine* mfc, uint8_t* ls,
                                     uint8_t* main_mem, uint32_t main_size)
{
    int processed = 0;
    for (uint32_t i = 0; i < mfc->queue_size; i++) {
        MFCCommand* cmd = &mfc->queue[i];
        if (mfc_is_list(cmd->cmd)) {
            mfc_execute_list(mfc, cmd, ls, main_mem, main_size);
        } else {
            mfc_execute(mfc, cmd, ls, main_mem, main_size);
        }
        processed++;
    }
    mfc->queue_size = 0;
    return processed;
}

// Check tag completion status
static inline uint32_t mfc_read_tag_status(MFCEngine* mfc) {
    return mfc->tag_status & mfc->tag_mask;
}

// Print MFC stats
static inline void mfc_print_stats(const MFCEngine* mfc) {
    fprintf(stderr, "[MFC] Stats: %lu PUTs, %lu GETs, %lu atomics, %.1f MB transferred\n",
            mfc->total_puts, mfc->total_gets, mfc->total_atomics,
            mfc->total_bytes / (1024.0 * 1024.0));
}
