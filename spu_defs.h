// spu_defs.h — Synergistic Processing Unit (SPU) register and opcode definitions
// Cell BE has 6 SPUs (PS3 exposes 6 to games, 1 reserved for OS)
//
// Each SPU:
//   - 128 × 128-bit registers (quad-word GPRs)
//   - 256KB Local Store (private SRAM, not cached)
//   - MFC (Memory Flow Controller) for DMA to/from main memory
//   - Two pipelines: even (FP/byte/int) and odd (LS load/store/branch/channel)
//
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define SPU_HD __host__ __device__
#else
#define SPU_HD
#endif

namespace spu {

// ═══════════════════════════════════════════════════════════════
// 128-bit Quad-Word Register
// Maps directly to CUDA uint4 / float4 for SIMD operations
// ═══════════════════════════════════════════════════════════════

union QWord {
    uint32_t u32[4];   // preferred slot = [0] (big-endian: word 0 is MSB)
    int32_t  s32[4];
    float    f32[4];
    uint16_t u16[8];
    int16_t  s16[8];
    uint8_t  u8[16];
    int8_t   s8[16];
    uint64_t u64[2];
    double   f64[2];
};

// ═══════════════════════════════════════════════════════════════
// SPU State — one per SPU core
// ═══════════════════════════════════════════════════════════════

static constexpr uint32_t SPU_LS_SIZE     = 256 * 1024;  // 256KB Local Store
static constexpr uint32_t SPU_NUM_REGS    = 128;
static constexpr uint32_t SPU_MFC_QUEUE   = 16;          // 16 DMA commands max

struct MFCCommand {
    uint32_t lsa;      // Local Store Address
    uint64_t ea;       // Effective (main memory) Address
    uint32_t size;     // Transfer size (16B aligned, max 16KB)
    uint32_t tag;      // DMA tag (0-31)
    uint32_t cmd;      // MFC command opcode
    uint32_t active;   // 1 = pending
};

// MFC DMA commands
enum MFCCmd : uint32_t {
    MFC_PUT     = 0x20,   // LS → main memory
    MFC_GET     = 0x40,   // main memory → LS
    MFC_PUTL    = 0x24,   // PUT with list
    MFC_GETL    = 0x44,   // GET with list
    MFC_SNDSIG  = 0xA0,   // Send signal
};

// SPU Channel numbers (for rdch/wrch instructions)
enum SPUChannel : uint32_t {
    SPU_RdEventStat     = 0,
    SPU_WrEventMask     = 1,
    SPU_WrEventAck      = 2,
    SPU_RdSigNotify1    = 3,
    SPU_RdSigNotify2    = 4,
    SPU_WrDec           = 7,
    SPU_RdDec           = 8,
    SPU_RdMachStat      = 13,
    SPU_WrSRR0          = 14,
    SPU_RdSRR0          = 15,
    MFC_WrMSSyncReq     = 9,
    MFC_RdTagMask       = 12,
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
    SPU_WrOutMbox       = 28,   // Write to outbound mailbox (→ PPE)
    SPU_RdInMbox        = 29,   // Read from inbound mailbox (← PPE)
    SPU_WrOutIntrMbox   = 30,   // Write outbound interrupt mailbox
};

struct SPUState {
    // 128 × 128-bit registers
    QWord gpr[SPU_NUM_REGS];

    // Program Counter (18-bit, LS address space)
    uint32_t pc;
    uint32_t npc;

    // MFC staging registers (written via channels before MFC_Cmd)
    uint32_t mfc_lsa;
    uint32_t mfc_eah;
    uint32_t mfc_eal;
    uint32_t mfc_size;
    uint32_t mfc_tag;
    uint32_t mfc_tagMask;

    // MFC DMA queue
    MFCCommand mfcQueue[SPU_MFC_QUEUE];
    uint32_t mfcQueueLen;

    // Mailbox (SPU ↔ PPE communication)
    uint32_t outMbox;          // SPU → PPE (1 deep)
    uint32_t outMboxValid;
    uint32_t inMbox;           // PPE → SPU (4 deep ring)
    uint32_t inMboxValid;
    uint32_t outIntrMbox;      // Interrupt mailbox
    uint32_t outIntrMboxValid;

    // Decrementer (timer)
    uint32_t decrementer;

    // Event status
    uint32_t eventMask;
    uint32_t eventStat;

    // SRR0 (interrupt return address)
    uint32_t srr0;

    // Execution state
    uint32_t halted;
    uint32_t cycles;
    uint32_t spuId;            // 0-5, which SPU this is
};

// ═══════════════════════════════════════════════════════════════
// SPU Instruction Format Extraction
// ═══════════════════════════════════════════════════════════════
//
// SPU instructions are 32-bit, big-endian. Multiple formats:
//
// RRR:  op(4)  rT(7) rB(7) rA(7) rC(7)
// RR:   op(11) rT(7) rB(7) rA(7)
// RI7:  op(11) I7(7) rA(7) rT(7)
// RI8:  op(10) I8(8) rA(7) rT(7)
// RI10: op(8)  I10(10) rA(7) rT(7)
// RI16: op(9)  I16(16) rT(7)
// RI18: op(7)  I18(18) rT(7)

SPU_HD inline uint32_t spu_op4(uint32_t inst)   { return (inst >> 28) & 0xF; }
SPU_HD inline uint32_t spu_op7(uint32_t inst)   { return (inst >> 25) & 0x7F; }
SPU_HD inline uint32_t spu_op8(uint32_t inst)   { return (inst >> 24) & 0xFF; }
SPU_HD inline uint32_t spu_op9(uint32_t inst)   { return (inst >> 23) & 0x1FF; }
SPU_HD inline uint32_t spu_op10(uint32_t inst)  { return (inst >> 22) & 0x3FF; }
SPU_HD inline uint32_t spu_op11(uint32_t inst)  { return (inst >> 21) & 0x7FF; }

// RRR format
SPU_HD inline uint32_t spu_rT_rrr(uint32_t inst) { return (inst >> 21) & 0x7F; }
SPU_HD inline uint32_t spu_rB_rrr(uint32_t inst) { return (inst >> 14) & 0x7F; }
SPU_HD inline uint32_t spu_rA_rrr(uint32_t inst) { return (inst >> 7)  & 0x7F; }
SPU_HD inline uint32_t spu_rC_rrr(uint32_t inst) { return inst & 0x7F; }

// RR format (op11)
SPU_HD inline uint32_t spu_rT_rr(uint32_t inst) { return (inst >> 14) & 0x7F; }  // dest is middle
SPU_HD inline uint32_t spu_rB_rr(uint32_t inst) { return (inst >> 7)  & 0x7F; }
SPU_HD inline uint32_t spu_rA_rr(uint32_t inst) { return inst & 0x7F; }

// RI7 format
SPU_HD inline uint32_t spu_rT_ri7(uint32_t inst) { return inst & 0x7F; }
SPU_HD inline uint32_t spu_rA_ri7(uint32_t inst) { return (inst >> 7) & 0x7F; }
SPU_HD inline int32_t  spu_I7(uint32_t inst) {
    int32_t v = (inst >> 14) & 0x7F;
    return (v & 0x40) ? (v | ~0x7F) : v;  // sign-extend 7-bit
}

// RI8 format
SPU_HD inline uint32_t spu_rT_ri8(uint32_t inst) { return inst & 0x7F; }
SPU_HD inline uint32_t spu_rA_ri8(uint32_t inst) { return (inst >> 7) & 0x7F; }
SPU_HD inline int32_t  spu_I8(uint32_t inst) {
    int32_t v = (inst >> 14) & 0xFF;
    return (v & 0x80) ? (v | ~0xFF) : v;
}

// RI10 format
SPU_HD inline uint32_t spu_rT_ri10(uint32_t inst) { return inst & 0x7F; }
SPU_HD inline uint32_t spu_rA_ri10(uint32_t inst) { return (inst >> 7) & 0x7F; }
SPU_HD inline int32_t  spu_I10(uint32_t inst) {
    int32_t v = (inst >> 14) & 0x3FF;
    return (v & 0x200) ? (v | ~0x3FF) : v;  // sign-extend 10-bit
}

// RI16 format
SPU_HD inline uint32_t spu_rT_ri16(uint32_t inst) { return inst & 0x7F; }
SPU_HD inline int32_t  spu_I16(uint32_t inst) {
    int32_t v = (inst >> 7) & 0xFFFF;
    return (v & 0x8000) ? (v | ~0xFFFF) : v;  // sign-extend 16-bit
}
SPU_HD inline uint32_t spu_I16u(uint32_t inst) { return (inst >> 7) & 0xFFFF; }

// RI18 format
SPU_HD inline uint32_t spu_rT_ri18(uint32_t inst) { return inst & 0x7F; }
SPU_HD inline uint32_t spu_I18(uint32_t inst) { return (inst >> 7) & 0x3FFFF; }

// ═══════════════════════════════════════════════════════════════
// SPU Opcode Constants (11-bit, RR format)
// ═══════════════════════════════════════════════════════════════

namespace op11 {
    // Integer arithmetic
    constexpr uint32_t AH       = 0x0C8;  // Add Halfword
    constexpr uint32_t A        = 0x0C0;  // Add Word
    constexpr uint32_t SFH      = 0x048;  // Subtract from Halfword
    constexpr uint32_t SF       = 0x040;  // Subtract from Word
    constexpr uint32_t ADDX     = 0x340;  // Add Extended
    constexpr uint32_t SFX      = 0x341;  // Subtract from Extended
    constexpr uint32_t CG       = 0x0C2;  // Carry Generate
    constexpr uint32_t BG       = 0x042;  // Borrow Generate
    constexpr uint32_t CGX      = 0x342;  // Carry Generate Extended
    constexpr uint32_t BGX      = 0x343;  // Borrow Generate Extended
    constexpr uint32_t MPY      = 0x3C4;  // Multiply (16-bit)
    constexpr uint32_t MPYU     = 0x3CC;  // Multiply Unsigned (16-bit)
    constexpr uint32_t MPYH     = 0x3C5;  // Multiply High
    constexpr uint32_t MPYS     = 0x3C7;  // Multiply and Shift Right
    constexpr uint32_t MPYHH    = 0x3C6;  // Multiply High High
    constexpr uint32_t MPYHHU   = 0x3CE;  // Multiply High High Unsigned
    constexpr uint32_t CLZ      = 0x2A5;  // Count Leading Zeros
    constexpr uint32_t CNTB     = 0x2B4;  // Count Ones in Bytes
    constexpr uint32_t AVGB     = 0x0D3;  // Average Bytes
    constexpr uint32_t ABSDB    = 0x053;  // Absolute Differences of Bytes

    // Logical
    constexpr uint32_t AND      = 0x0C1;  // AND
    constexpr uint32_t ANDC     = 0x2C1;  // AND with Complement
    constexpr uint32_t OR       = 0x041;  // OR
    constexpr uint32_t ORC      = 0x2C9;  // OR with Complement
    constexpr uint32_t XOR      = 0x241;  // XOR
    constexpr uint32_t NAND     = 0x0C9;  // NAND
    constexpr uint32_t NOR      = 0x049;  // NOR
    constexpr uint32_t EQV      = 0x249;  // Equivalent (XNOR)

    // Shift and Rotate
    constexpr uint32_t SHLH     = 0x05F;  // Shift Left Halfword
    constexpr uint32_t SHL      = 0x05B;  // Shift Left Word
    constexpr uint32_t SHLQBI   = 0x1DB;  // Shift Left Quadword by Bits
    constexpr uint32_t SHLQBY   = 0x1DF;  // Shift Left Quadword by Bytes
    constexpr uint32_t SHLQBYBI = 0x1CF;  // Shift Left Quadword by Bytes from Bit Shift Count
    constexpr uint32_t ROTH     = 0x05C;  // Rotate Halfword
    constexpr uint32_t ROT      = 0x058;  // Rotate Word
    constexpr uint32_t ROTQBY   = 0x1DC;  // Rotate Quadword by Bytes
    constexpr uint32_t ROTQBI   = 0x1D8;  // Rotate Quadword by Bits
    constexpr uint32_t ROTQBYBI = 0x1CC;  // Rotate Quadword by Bytes from Bit Shift Count
    constexpr uint32_t ROTHM    = 0x05D;  // Rotate and Mask Halfword
    constexpr uint32_t ROTM     = 0x059;  // Rotate and Mask Word
    constexpr uint32_t ROTQMBY  = 0x1DD;  // Rotate and Mask Quadword by Bytes
    constexpr uint32_t ROTQMBI  = 0x1D9;  // Rotate and Mask Quadword by Bits
    constexpr uint32_t ROTQMBYBI= 0x1CD;  // Rotate and Mask Quadword by Bytes from Bit Shift Count
    constexpr uint32_t ROTMAH   = 0x05E;  // Rotate and Mask Algebraic Halfword
    constexpr uint32_t ROTMA    = 0x05A;  // Rotate and Mask Algebraic Word

    // Compare
    constexpr uint32_t CEQH     = 0x3C8;  // Compare Equal Halfword
    constexpr uint32_t CEQ      = 0x3C0;  // Compare Equal Word
    constexpr uint32_t CGTH     = 0x248;  // Compare Greater Than Halfword
    constexpr uint32_t CGT      = 0x240;  // Compare Greater Than Word
    constexpr uint32_t CLGTH    = 0x2C8;  // Compare Logical Greater Than Halfword
    constexpr uint32_t CLGT     = 0x2C0;  // Compare Logical Greater Than Word
    constexpr uint32_t CEQB     = 0x3D0;  // Compare Equal Byte
    constexpr uint32_t CGTB     = 0x250;  // Compare Greater Than Byte
    constexpr uint32_t CLGTB    = 0x2D0;  // Compare Logical Greater Than Byte

    // Floating point (single-precision, 4-wide)
    constexpr uint32_t FA       = 0x2C4;  // Floating Add
    constexpr uint32_t FS       = 0x2C5;  // Floating Subtract
    constexpr uint32_t FM       = 0x2C6;  // Floating Multiply
    constexpr uint32_t FCEQ     = 0x3C2;  // Floating Compare Equal
    constexpr uint32_t FCGT     = 0x2C2;  // Floating Compare Greater Than
    constexpr uint32_t FCMEQ    = 0x3CA;  // Floating Compare Magnitude Equal
    constexpr uint32_t FCMGT    = 0x2CA;  // Floating Compare Magnitude Greater Than
    constexpr uint32_t FI       = 0x3D4;  // Floating Interpolate
    constexpr uint32_t CSFLT    = 0x1DA;  // Convert Signed Int to Float
    constexpr uint32_t CFLTS    = 0x1D8;  // Convert Float to Signed Int
    constexpr uint32_t CUFLT    = 0x1DC;  // Convert Unsigned Int to Float  
    constexpr uint32_t CFLTU    = 0x1DA;  // Convert Float to Unsigned Int

    // Double precision
    constexpr uint32_t DFA      = 0x2CC;  // Double Floating Add
    constexpr uint32_t DFS      = 0x2CD;  // Double Floating Subtract
    constexpr uint32_t DFM      = 0x2CE;  // Double Floating Multiply

    // Branch (RR form)
    constexpr uint32_t BI       = 0x1A8;  // Branch Indirect
    constexpr uint32_t BISL     = 0x1A9;  // Branch Indirect and Set Link
    constexpr uint32_t BIZ      = 0x128;  // Branch Indirect if Zero
    constexpr uint32_t BINZ     = 0x129;  // Branch Indirect if Not Zero
    constexpr uint32_t BIHZ     = 0x12A;  // Branch Indirect if Halfword Zero
    constexpr uint32_t BIHNZ    = 0x12B;  // Branch Indirect if Halfword Not Zero

    // Channel
    constexpr uint32_t RDCH     = 0x00D;  // Read Channel
    constexpr uint32_t RCHCNT   = 0x00F;  // Read Channel Count
    constexpr uint32_t WRCH     = 0x10D;  // Write Channel

    // Sign extension
    constexpr uint32_t XSBH     = 0x1B6;  // Extend Sign Byte to Halfword
    constexpr uint32_t XSHW     = 0x1AE;  // Extend Sign Halfword to Word
    constexpr uint32_t XSWD     = 0x1A6;  // Extend Sign Word to Doubleword

    // Load/Store indexed (RR form)
    constexpr uint32_t LQX      = 0x1C4;  // Load Quadword (rA+rB index)
    constexpr uint32_t STQX     = 0x144;  // Store Quadword (rA+rB index)

    // Special
    constexpr uint32_t STOPD    = 0x140;  // Stop and Signal (debug)
    constexpr uint32_t NOP      = 0x201;  // No Operation (execute)
    constexpr uint32_t LNOP     = 0x001;  // No Operation (load)
    constexpr uint32_t SYNC     = 0x002;  // Synchronize
    constexpr uint32_t DSYNC    = 0x003;  // Synchronize Data
    constexpr uint32_t STOP     = 0x000;  // Stop and Signal
}

// ═══════════════════════════════════════════════════════════════
// SPU Opcode Constants (8-bit, RI10 format)
// ═══════════════════════════════════════════════════════════════

namespace op8 {
    constexpr uint32_t AHI      = 0x1D;  // Add Halfword Immediate
    constexpr uint32_t AI       = 0x1C;  // Add Word Immediate
    constexpr uint32_t SFI      = 0x0C;  // Subtract from Word Immediate
    constexpr uint32_t SFHI     = 0x0D;  // Subtract from Halfword Immediate
    constexpr uint32_t ANDI     = 0x14;  // AND Word Immediate
    constexpr uint32_t ORI      = 0x04;  // OR Word Immediate
    constexpr uint32_t XORI     = 0x44;  // XOR Word Immediate
    constexpr uint32_t CEQI     = 0x7C;  // Compare Equal Word Immediate
    constexpr uint32_t CEQHI    = 0x7D;  // Compare Equal Halfword Immediate
    constexpr uint32_t CEQBI    = 0x7E;  // Compare Equal Byte Immediate
    constexpr uint32_t CGTI     = 0x4C;  // Compare Greater Than Word Immediate
    constexpr uint32_t CGTHI    = 0x4D;  // Compare Greater Than Halfword Immediate
    constexpr uint32_t CGTBI    = 0x4E;  // Compare Greater Than Byte Immediate
    constexpr uint32_t CLGTI    = 0x5C;  // Compare Logical Greater Than Word Immediate
    constexpr uint32_t CLGTHI   = 0x5D;  // Compare Logical Greater Than Halfword Immediate
    constexpr uint32_t CLGTBI   = 0x5E;  // Compare Logical Greater Than Byte Immediate
    constexpr uint32_t LQD      = 0x34;  // Load Quadword D-form (LS)
    constexpr uint32_t STQD     = 0x24;  // Store Quadword D-form (LS)
    constexpr uint32_t MPYI     = 0x74;  // Multiply Immediate
    constexpr uint32_t MPYUI    = 0x75;  // Multiply Unsigned Immediate
}

// ═══════════════════════════════════════════════════════════════
// SPU Opcode Constants (9-bit, RI16 format)
// ═══════════════════════════════════════════════════════════════

namespace op9 {
    constexpr uint32_t BR       = 0x064;  // Branch Relative
    constexpr uint32_t BRA      = 0x060;  // Branch Absolute
    constexpr uint32_t BRSL     = 0x066;  // Branch Relative and Set Link
    constexpr uint32_t BRASL    = 0x062;  // Branch Absolute and Set Link
    constexpr uint32_t BRZ      = 0x040;  // Branch if Zero
    constexpr uint32_t BRNZ     = 0x042;  // Branch if Not Zero
    constexpr uint32_t BRHZ     = 0x044;  // Branch if Halfword Zero
    constexpr uint32_t BRHNZ    = 0x046;  // Branch if Halfword Not Zero
    constexpr uint32_t ILH      = 0x083;  // Immediate Load Halfword
    constexpr uint32_t IL       = 0x081;  // Immediate Load Word
    constexpr uint32_t ILHU     = 0x082;  // Immediate Load Halfword Upper
    constexpr uint32_t IOHL     = 0x0C1;  // Immediate OR Halfword Lower
    constexpr uint32_t LQA      = 0x061;  // Load Quadword A-form
    constexpr uint32_t STQA     = 0x041;  // Store Quadword A-form
    constexpr uint32_t LQR      = 0x067;  // Load Quadword Instruction Relative
    constexpr uint32_t STQR     = 0x047;  // Store Quadword Instruction Relative
    constexpr uint32_t FSMBI    = 0x065;  // Form Select Mask for Bytes Immediate
}

// ═══════════════════════════════════════════════════════════════
// SPU Opcode Constants (7-bit, RI18 format)
// ═══════════════════════════════════════════════════════════════

namespace op7 {
    constexpr uint32_t ILA      = 0x21;   // Immediate Load Address (18-bit unsigned)
}

// ═══════════════════════════════════════════════════════════════
// SPU Opcode Constants (4-bit, RRR format)
// ═══════════════════════════════════════════════════════════════

namespace op4 {
    constexpr uint32_t SELB     = 0x8;    // Select Bits
    constexpr uint32_t SHUFB    = 0xB;    // Shuffle Bytes
    constexpr uint32_t FMA      = 0xE;    // Floating Multiply-Add
    constexpr uint32_t FMS      = 0xF;    // Floating Multiply-Subtract
    constexpr uint32_t FNMS     = 0xD;    // Floating Negative Multiply-Subtract
    constexpr uint32_t MPYA     = 0xC;    // Multiply and Add (integer)
    constexpr uint32_t DFMA     = 0x6;    // Double Floating Multiply-Add
    constexpr uint32_t DFMS     = 0x7;    // Double Floating Multiply-Subtract
    constexpr uint32_t DFNMS    = 0x5;    // Double Floating Negative Multiply-Subtract
}

} // namespace spu
