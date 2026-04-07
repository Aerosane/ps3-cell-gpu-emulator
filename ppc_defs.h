// ppc_defs.h — PowerPC 64-bit (PPE) register and opcode definitions
// Cell Broadband Engine: PPE is a dual-issue in-order PPC64 core
//
// This file defines the register state, opcode encoding, and memory
// layout for GPU-resident PowerPC interpretation.
//
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define PPE_HD __host__ __device__
#else
#define PPE_HD
#endif

namespace ppc {

// ═══════════════════════════════════════════════════════════════
// PPE Register File — 64-bit PowerPC
// ═══════════════════════════════════════════════════════════════

struct PPEState {
    // General Purpose Registers (64-bit in PPC64, but PS3 games mostly use 32-bit mode)
    uint64_t gpr[32];

    // Floating Point Registers (64-bit double, also used for single via fpscr)
    double   fpr[32];

    // Special Purpose Registers
    uint64_t lr;       // Link Register (return address)
    uint64_t ctr;      // Count Register (loop counter / indirect branch target)
    uint64_t xer;      // Fixed-point exception register (SO, OV, CA, byte count)
    uint32_t cr;       // Condition Register (8 × 4-bit fields, CR0-CR7)
    uint32_t fpscr;    // FP status and control

    // Program Counter
    uint64_t pc;
    uint64_t npc;      // Next PC (for branch delay handling)

    // Machine State Register
    uint64_t msr;

    // SPR table for mtspr/mfspr (only commonly used ones)
    uint64_t sprg[4];  // SPRG0-3 (OS scratch)
    uint64_t srr0;     // Save/Restore Register 0 (exception return addr)
    uint64_t srr1;     // Save/Restore Register 1 (exception MSR)
    uint64_t dec;      // Decrementer (timer)
    uint64_t tbl;      // Time Base Lower
    uint64_t tbu;      // Time Base Upper

    // Execution state
    uint32_t halted;   // 1 = stopped (HLE syscall / breakpoint)
    uint32_t cycles;   // Cycle counter for throttling
};

// ═══════════════════════════════════════════════════════════════
// PPC Opcode Field Extraction
// ═══════════════════════════════════════════════════════════════
//
// PPC instructions are 32 bits, big-endian, with a 6-bit primary opcode:
//
//  31       26 25    21 20    16 15    11 10         1 0
// +----------+--------+--------+--------+------------+-+
// |  OPCD    |   rD   |   rA   |   rB   |   XO       |Rc|
// +----------+--------+--------+--------+------------+-+
//
// I-form (branches): OPCD | LI(24) | AA | LK
// D-form (loads/imm): OPCD | rD | rA | SIMM/UIMM(16)
// X-form (ALU/FP):  OPCD | rD | rA | rB | XO(10) | Rc

PPE_HD inline uint32_t OPCD(uint32_t inst) { return (inst >> 26) & 0x3F; }
PPE_HD inline uint32_t RD(uint32_t inst)   { return (inst >> 21) & 0x1F; }
PPE_HD inline uint32_t RS(uint32_t inst)   { return (inst >> 21) & 0x1F; } // alias
PPE_HD inline uint32_t RA(uint32_t inst)   { return (inst >> 16) & 0x1F; }
PPE_HD inline uint32_t RB(uint32_t inst)   { return (inst >> 11) & 0x1F; }
PPE_HD inline uint32_t RC(uint32_t inst)   { return inst & 1; }
PPE_HD inline uint32_t XO_10(uint32_t inst){ return (inst >> 1) & 0x3FF; } // 10-bit extended
PPE_HD inline uint32_t XO_9(uint32_t inst) { return (inst >> 1) & 0x1FF; } // 9-bit extended

// D-form immediate (sign-extended to 64-bit)
PPE_HD inline int64_t SIMM16(uint32_t inst) {
    int16_t imm = (int16_t)(inst & 0xFFFF);
    return (int64_t)imm;
}
PPE_HD inline uint64_t UIMM16(uint32_t inst) { return inst & 0xFFFF; }

// I-form branch displacement (26-bit, sign-extended, shifted left 2)
PPE_HD inline int64_t LI26(uint32_t inst) {
    int32_t raw = (int32_t)(inst & 0x03FFFFFC);
    if (raw & 0x02000000) raw |= 0xFC000000; // sign-extend
    return (int64_t)raw;
}

// B-form branch displacement (16-bit, sign-extended, shifted left 2)
PPE_HD inline int64_t BD16(uint32_t inst) {
    int16_t raw = (int16_t)(inst & 0xFFFC);
    return (int64_t)raw;
}

// Branch flags
PPE_HD inline uint32_t AA(uint32_t inst) { return (inst >> 1) & 1; }
PPE_HD inline uint32_t LK(uint32_t inst) { return inst & 1; }
PPE_HD inline uint32_t BO(uint32_t inst) { return (inst >> 21) & 0x1F; }
PPE_HD inline uint32_t BI(uint32_t inst) { return (inst >> 16) & 0x1F; }

// SPR encoding (split field: bits 16-20 | bits 11-15, swapped)
PPE_HD inline uint32_t SPR(uint32_t inst) {
    uint32_t lo = (inst >> 16) & 0x1F;
    uint32_t hi = (inst >> 11) & 0x1F;
    return (lo << 5) | hi;
}

// CRM field for mtcrf
PPE_HD inline uint32_t CRM(uint32_t inst) { return (inst >> 12) & 0xFF; }

// MB/ME for rlwinm (rotate and mask)
PPE_HD inline uint32_t SH(uint32_t inst) { return (inst >> 11) & 0x1F; }
PPE_HD inline uint32_t MB(uint32_t inst) { return (inst >> 6) & 0x1F; }
PPE_HD inline uint32_t ME(uint32_t inst) { return (inst >> 1) & 0x1F; }

// ═══════════════════════════════════════════════════════════════
// Primary Opcodes (6-bit)
// ═══════════════════════════════════════════════════════════════

enum PrimaryOp : uint32_t {
    OP_TDI     = 2,   // Trap Doubleword Immediate
    OP_TWI     = 3,   // Trap Word Immediate
    OP_MULLI   = 7,   // Multiply Low Immediate
    OP_SUBFIC  = 8,   // Subtract From Immediate Carrying
    OP_CMPLI   = 10,  // Compare Logical Immediate
    OP_CMPI    = 11,  // Compare Immediate
    OP_ADDIC   = 12,  // Add Immediate Carrying
    OP_ADDIC_D = 13,  // Add Immediate Carrying and Record
    OP_ADDI    = 14,  // Add Immediate (also li)
    OP_ADDIS   = 15,  // Add Immediate Shifted (also lis)
    OP_BC      = 16,  // Branch Conditional
    OP_SC      = 17,  // System Call
    OP_B       = 18,  // Branch
    OP_GRP19   = 19,  // Group 19: bclr, bcctr, crand, etc.
    OP_RLWIMI  = 20,  // Rotate Left Word Imm then Mask Insert
    OP_RLWINM  = 21,  // Rotate Left Word Imm then AND with Mask
    OP_RLWNM   = 23,  // Rotate Left Word then AND with Mask
    OP_ORI     = 24,  // OR Immediate (also nop)
    OP_ORIS    = 25,  // OR Immediate Shifted
    OP_XORI    = 26,  // XOR Immediate
    OP_XORIS   = 27,  // XOR Immediate Shifted
    OP_ANDI    = 28,  // AND Immediate
    OP_ANDIS   = 29,  // AND Immediate Shifted
    OP_GRP30   = 30,  // Group 30: 64-bit rotate/shift
    OP_GRP31   = 31,  // Group 31: ALU, load/store indexed, SPR, etc.
    OP_LWZ     = 32,  // Load Word and Zero
    OP_LWZU    = 33,  // Load Word and Zero with Update
    OP_LBZ     = 34,  // Load Byte and Zero
    OP_LBZU    = 35,  // Load Byte and Zero with Update
    OP_STW     = 36,  // Store Word
    OP_STWU    = 37,  // Store Word with Update
    OP_STB     = 38,  // Store Byte
    OP_STBU    = 39,  // Store Byte with Update
    OP_LHZ     = 40,  // Load Half Word and Zero
    OP_LHZU    = 41,  // Load Half Word and Zero with Update
    OP_LHA     = 42,  // Load Half Word Algebraic
    OP_LHAU    = 43,  // Load Half Word Algebraic with Update
    OP_STH     = 44,  // Store Half Word
    OP_STHU    = 45,  // Store Half Word with Update
    OP_LMW     = 46,  // Load Multiple Word
    OP_STMW    = 47,  // Store Multiple Word
    OP_LFS     = 48,  // Load Floating-Point Single
    OP_LFSU    = 49,  // Load Floating-Point Single with Update
    OP_LFD     = 50,  // Load Floating-Point Double
    OP_LFDU    = 51,  // Load Floating-Point Double with Update
    OP_STFS    = 52,  // Store Floating-Point Single
    OP_STFSU   = 53,  // Store Floating-Point Single with Update
    OP_STFD    = 54,  // Store Floating-Point Double
    OP_STFDU   = 55,  // Store Floating-Point Double with Update
    OP_GRP58   = 58,  // Group 58: LD/LDU/LWA (DS-form)
    OP_GRP59   = 59,  // Group 59: single-precision FP
    OP_GRP62   = 62,  // Group 62: STD/STDU (DS-form)
    OP_GRP63   = 63,  // Group 63: double-precision FP
};

// ═══════════════════════════════════════════════════════════════
// Extended Opcodes (Group 31 — ALU/SPR/Load-Store indexed)
// ═══════════════════════════════════════════════════════════════

enum XO31 : uint32_t {
    XO_CMP     = 0,
    XO_TW      = 4,      // Trap Word
    XO_SUBFC   = 8,
    XO_MULHDU  = 9,      // Multiply High Doubleword Unsigned
    XO_ADDC    = 10,
    XO_MULHWU  = 11,
    XO_MFCR    = 19,     // also MFOCRF
    XO_LWARX   = 20,     // Load Word and Reserve Indexed
    XO_LDX     = 21,     // Load Doubleword Indexed
    XO_LWZX    = 23,
    XO_SLW     = 24,
    XO_CNTLZW  = 26,
    XO_SLD     = 27,     // Shift Left Doubleword
    XO_AND     = 28,
    XO_CMPL    = 32,
    XO_SUBF    = 40,
    XO_LDUX    = 53,     // Load Doubleword with Update Indexed
    XO_DCBST   = 54,     // Data Cache Block Store
    XO_LWZUX   = 55,
    XO_CNTLZD  = 58,     // Count Leading Zeros Doubleword
    XO_ANDC    = 60,
    XO_TD      = 68,     // Trap Doubleword
    XO_MULHD   = 73,     // Multiply High Doubleword
    XO_MULHW   = 75,
    XO_MFMSR   = 83,
    XO_LDARX   = 84,     // Load Doubleword and Reserve Indexed
    XO_DCBF    = 86,     // Data Cache Block Flush
    XO_LBZX    = 87,
    XO_NEG     = 104,
    XO_LBZUX   = 119,
    XO_NOR     = 124,
    XO_SUBFE   = 136,
    XO_ADDE    = 138,
    XO_MTCRF   = 144,    // also MTOCRF
    XO_MTMSR   = 146,
    XO_STDX    = 149,    // Store Doubleword Indexed
    XO_STWCX   = 150,    // Store Word Conditional Indexed
    XO_STWX    = 151,
    XO_STDUX   = 181,    // Store Doubleword with Update Indexed
    XO_STWUX   = 183,
    XO_SUBFZE  = 200,
    XO_ADDZE   = 202,
    XO_STDCX   = 214,    // Store Doubleword Conditional Indexed
    XO_STBX    = 215,
    XO_SUBFME  = 232,
    XO_MULLD   = 233,    // Multiply Low Doubleword
    XO_ADDME   = 234,
    XO_MULLW   = 235,
    XO_DCBTST  = 246,    // Data Cache Block Touch for Store
    XO_STBUX   = 247,
    XO_ADD     = 266,
    XO_DCBT    = 278,    // Data Cache Block Touch
    XO_LHZX    = 279,
    XO_EQV     = 284,    // Equivalent
    XO_LHZUX   = 311,
    XO_XOR     = 316,
    XO_MFSPR   = 339,
    XO_LWAX    = 341,    // Load Word Algebraic Indexed
    XO_LHAX    = 343,
    XO_MFTB    = 371,    // Move From Time Base
    XO_LWAUX   = 373,    // Load Word Algebraic with Update Indexed
    XO_LHAUX   = 375,    // Load Half Word Algebraic with Update Indexed
    XO_STHX    = 407,
    XO_ORC     = 412,
    XO_STHUX   = 439,    // Store Half Word with Update Indexed
    XO_OR      = 444,
    XO_DIVDU   = 457,    // Divide Doubleword Unsigned
    XO_DIVWU   = 459,
    XO_MTSPR   = 467,
    XO_NAND    = 476,
    XO_DIVD    = 489,    // Divide Doubleword
    XO_DIVW    = 491,
    XO_LWBRX   = 534,    // Load Word Byte-Reversed Indexed
    XO_LFSX    = 535,    // Load Floating-Point Single Indexed
    XO_SRW     = 536,
    XO_SRD     = 539,    // Shift Right Doubleword
    XO_LFSUX   = 567,    // Load Floating-Point Single with Update Indexed
    XO_SYNC    = 598,
    XO_LFDX    = 599,    // Load Floating-Point Double Indexed
    XO_LFDUX   = 631,    // Load Floating-Point Double with Update Indexed
    XO_STWBRX  = 662,    // Store Word Byte-Reversed Indexed
    XO_STFSX   = 663,    // Store Floating-Point Single Indexed
    XO_STFSUX  = 695,    // Store Floating-Point Single with Update Indexed
    XO_STFDX   = 727,    // Store Floating-Point Double Indexed
    XO_STFDUX  = 759,    // Store Floating-Point Double with Update Indexed
    XO_LHBRX   = 790,    // Load Half Word Byte-Reversed Indexed
    XO_SRAW    = 792,
    XO_SRAD    = 794,    // Shift Right Algebraic Doubleword
    XO_SRAWI   = 824,
    XO_EIEIO   = 854,
    XO_STHBRX  = 918,    // Store Half Word Byte-Reversed Indexed
    XO_EXTSH   = 922,
    XO_EXTSB   = 954,
    XO_ICBI    = 982,    // Instruction Cache Block Invalidate
    XO_STFIWX  = 983,    // Store Floating-Point as Integer Word Indexed
    XO_EXTSW   = 986,    // Extend Sign Word
    XO_DCBZ    = 1014,   // Data Cache Block Zero
};

// ═══════════════════════════════════════════════════════════════
// Extended Opcodes (Group 19 — Branch/CR operations)
// ═══════════════════════════════════════════════════════════════

enum XO19 : uint32_t {
    XO_MCRF    = 0,
    XO_BCLR    = 16,   // Branch Conditional to Link Register
    XO_CRNOR   = 33,
    XO_CRANDC  = 129,
    XO_ISYNC   = 150,
    XO_CRXOR   = 193,
    XO_CRNAND  = 225,
    XO_CRAND   = 257,
    XO_CREQV   = 289,
    XO_CRORC   = 417,
    XO_CROR    = 449,
    XO_BCCTR   = 528,  // Branch Conditional to Count Register
};

// ═══════════════════════════════════════════════════════════════
// SPR Numbers (for mfspr/mtspr)
// ═══════════════════════════════════════════════════════════════

enum SPRNum : uint32_t {
    SPR_XER  = 1,
    SPR_LR   = 8,
    SPR_CTR  = 9,
    SPR_DSISR= 18,
    SPR_DAR  = 19,
    SPR_DEC  = 22,
    SPR_SRR0 = 26,
    SPR_SRR1 = 27,
    SPR_TBL  = 268,
    SPR_TBU  = 269,
    SPR_SPRG0= 272,
    SPR_SPRG1= 273,
    SPR_SPRG2= 274,
    SPR_SPRG3= 275,
    SPR_PVR  = 287,   // Processor Version Register (read-only, Cell PPE = 0x00700000)
};

// ═══════════════════════════════════════════════════════════════
// PS3 Memory Map Constants
// ═══════════════════════════════════════════════════════════════

static constexpr uint64_t PS3_RAM_SIZE     = 256ULL * 1024 * 1024;  // 256 MB main RAM
static constexpr uint64_t PS3_VRAM_SIZE    = 256ULL * 1024 * 1024;  // 256 MB VRAM (RSX)
static constexpr uint64_t PS3_RAM_BASE     = 0x00000000ULL;
static constexpr uint64_t PS3_VRAM_BASE    = 0x10000000ULL;         // RSX local memory
static constexpr uint64_t PS3_MMIO_BASE    = 0x28000000ULL;         // MMIO region

// Sandbox total: 512MB VRAM on GPU
static constexpr uint64_t PS3_SANDBOX_SIZE = PS3_RAM_SIZE + PS3_VRAM_SIZE;

// ═══════════════════════════════════════════════════════════════
// CellOS HLE Syscall Numbers (most commonly used)
// ═══════════════════════════════════════════════════════════════

enum LV2Syscall : uint32_t {
    SYS_PROCESS_EXIT         = 1,
    SYS_PROCESS_GETPID       = 2,
    SYS_THREAD_CREATE        = 24,
    SYS_THREAD_EXIT          = 25,
    SYS_THREAD_JOIN          = 26,
    SYS_MUTEX_CREATE         = 100,
    SYS_MUTEX_LOCK           = 101,
    SYS_MUTEX_UNLOCK         = 103,
    SYS_COND_CREATE          = 110,
    SYS_COND_WAIT            = 114,
    SYS_COND_SIGNAL          = 115,
    SYS_MEMORY_ALLOCATE      = 348,
    SYS_MEMORY_FREE          = 349,
    SYS_MEMORY_GET_PAGE_SIZE = 352,
    SYS_SPU_THREAD_GROUP_CREATE  = 170,
    SYS_SPU_THREAD_INITIALIZE   = 172,
    SYS_SPU_THREAD_GROUP_START   = 173,
    SYS_SPU_THREAD_GROUP_JOIN    = 178,
    SYS_SPU_THREAD_WRITE_LS     = 181,
    SYS_SPU_THREAD_READ_LS      = 182,
    SYS_FS_OPEN              = 801,
    SYS_FS_CLOSE             = 804,
    SYS_FS_READ              = 802,
    SYS_FS_WRITE             = 803,
    SYS_FS_STAT              = 818,
    SYS_TICKS_GET            = 141,
};

// ═══════════════════════════════════════════════════════════════
// Condition Register Helpers
// ═══════════════════════════════════════════════════════════════

// CR field layout (4 bits per field): LT | GT | EQ | SO
PPE_HD inline void setCRField(uint32_t& cr, int field, int64_t result, uint64_t xer) {
    uint32_t val = 0;
    if (result < 0)       val = 0x8;  // LT
    else if (result > 0)  val = 0x4;  // GT
    else                  val = 0x2;  // EQ
    if (xer & (1ULL << 31)) val |= 0x1;  // SO from XER
    int shift = (7 - field) * 4;
    cr = (cr & ~(0xF << shift)) | (val << shift);
}

PPE_HD inline bool getCRBit(uint32_t cr, int bit) {
    return (cr >> (31 - bit)) & 1;
}

// XER carry bit
PPE_HD inline bool getCA(uint64_t xer) { return (xer >> 29) & 1; }
PPE_HD inline void setCA(uint64_t& xer, bool ca) {
    xer = ca ? (xer | (1ULL << 29)) : (xer & ~(1ULL << 29));
}

} // namespace ppc
