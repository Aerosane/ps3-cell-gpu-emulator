// test_rsx.cu — RSX Reality Synthesizer Command Processor Test Suite
//
// Tests NV47 FIFO parsing, NV4097 method dispatch, surface clear,
// shader upload, draw call tracking, and full-frame simulation.
//
// Compile: nvcc -O3 -arch=sm_70 test_rsx.cu rsx_command_processor.cu -o rsx_test
//
// Part of Project Megakernel — CUDA-based PS3 emulation on Tesla V100.

#include "rsx_defs.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>

using namespace rsx;

// ═══════════════════════════════════════════════════════════════════
// Declarations from rsx_command_processor.cu
// ═══════════════════════════════════════════════════════════════════

namespace rsx {
    int  rsx_init(RSXState* state);
    int  rsx_process_fifo(RSXState* state, const uint32_t* fifo, uint32_t fifoSize,
                          uint8_t* vram, uint32_t maxCmds);
    void rsx_clear_surface(RSXState* state, uint8_t* vram, uint32_t clearMask);
    void rsx_print_state(const RSXState* state);
    void rsx_shutdown(RSXState* state);
}

// ═══════════════════════════════════════════════════════════════════
// Globals
// ═══════════════════════════════════════════════════════════════════

static int total_pass = 0, total_fail = 0;

// ═══════════════════════════════════════════════════════════════════
// Helper: build a FIFO method header (NV47 incrementing format)
// ═══════════════════════════════════════════════════════════════════

static uint32_t fifo_method(uint32_t method, uint32_t count, uint32_t subchannel = 0) {
    // type=0 (incrementing), count, subchannel, method >> 2 in bits [12:2]
    return ((count & 0x7FF) << 18) | ((subchannel & 0x1F) << 13) | ((method >> 2) << 2);
}

static uint32_t fifo_method_ni(uint32_t method, uint32_t count, uint32_t subchannel = 0) {
    // type=2 (non-incrementing)
    return (2u << 29) | ((count & 0x7FF) << 18) | ((subchannel & 0x1F) << 13) | ((method >> 2) << 2);
}

// ═══════════════════════════════════════════════════════════════════
// TEST 1: FIFO Header Parsing
// ═══════════════════════════════════════════════════════════════════

static void test_fifo_parsing() {
    printf("\n╔═══════════════════════════════════════════════╗\n");
    printf("║  TEST 1: FIFO Header Parsing                  ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");

    bool pass = true;

    // Test incrementing method: SET_SURFACE_FORMAT, count=1, subchannel=0
    {
        uint32_t hdr = fifo_method(NV4097_SET_SURFACE_FORMAT, 1, 0);
        FIFOCommand cmd = parseFIFOHeader(hdr);
        printf("  Incrementing: method=0x%04X count=%u sub=%u nonIncr=%d\n",
               cmd.method, cmd.count, cmd.subchannel, cmd.isNonIncr);
        if (cmd.method != NV4097_SET_SURFACE_FORMAT || cmd.count != 1 ||
            cmd.subchannel != 0 || cmd.isNonIncr) {
            printf("  ❌ FAIL (incrementing parse)\n");
            pass = false;
        }
    }

    // Test non-incrementing method: SET_TRANSFORM_PROGRAM, count=4
    {
        uint32_t hdr = fifo_method_ni(NV4097_SET_TRANSFORM_PROGRAM, 4, 0);
        FIFOCommand cmd = parseFIFOHeader(hdr);
        printf("  Non-incr:     method=0x%04X count=%u nonIncr=%d\n",
               cmd.method, cmd.count, cmd.isNonIncr);
        if (cmd.method != NV4097_SET_TRANSFORM_PROGRAM || cmd.count != 4 ||
            !cmd.isNonIncr) {
            printf("  ❌ FAIL (non-incrementing parse)\n");
            pass = false;
        }
    }

    // Test old-style jump
    {
        uint32_t hdr = 0x20001000;  // old-style jump to 0x1000
        FIFOCommand cmd = parseFIFOHeader(hdr);
        printf("  Old jump:     target=0x%08X isJump=%d\n", cmd.jumpTarget, cmd.isJump);
        if (!cmd.isJump || cmd.jumpTarget != 0x00001000) {
            printf("  ❌ FAIL (old-style jump)\n");
            pass = false;
        }
    }

    // Test new-style jump
    {
        uint32_t hdr = 0x00002001;  // new-style jump to 0x2000
        FIFOCommand cmd = parseFIFOHeader(hdr);
        printf("  New jump:     target=0x%08X isJump=%d\n", cmd.jumpTarget, cmd.isJump);
        if (!cmd.isJump || cmd.jumpTarget != 0x00002000) {
            printf("  ❌ FAIL (new-style jump)\n");
            pass = false;
        }
    }

    // Test call
    {
        uint32_t hdr = 0x00004002;  // call to 0x4000
        FIFOCommand cmd = parseFIFOHeader(hdr);
        printf("  Call:         target=0x%08X isCall=%d\n", cmd.jumpTarget, cmd.isCall);
        if (!cmd.isCall || cmd.jumpTarget != 0x00004000) {
            printf("  ❌ FAIL (call)\n");
            pass = false;
        }
    }

    // Test return
    {
        uint32_t hdr = 0x00020000;
        FIFOCommand cmd = parseFIFOHeader(hdr);
        printf("  Return:       isReturn=%d\n", cmd.isReturn);
        if (!cmd.isReturn) {
            printf("  ❌ FAIL (return)\n");
            pass = false;
        }
    }

    // Multi-word method: SET_SURFACE_CLIP_HORIZONTAL + VERTICAL (count=2 incrementing)
    {
        uint32_t hdr = fifo_method(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 2, 0);
        FIFOCommand cmd = parseFIFOHeader(hdr);
        printf("  Multi-word:   method=0x%04X count=%u\n", cmd.method, cmd.count);
        if (cmd.method != NV4097_SET_SURFACE_CLIP_HORIZONTAL || cmd.count != 2) {
            printf("  ❌ FAIL (multi-word)\n");
            pass = false;
        }
    }

    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;
}

// ═══════════════════════════════════════════════════════════════════
// TEST 2: Surface Setup
// ═══════════════════════════════════════════════════════════════════

static void test_surface_setup() {
    printf("\n╔═══════════════════════════════════════════════╗\n");
    printf("║  TEST 2: Surface Setup Commands               ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");

    RSXState state;
    rsx_init(&state);

    // Build FIFO: set format, clip, pitch, offset
    uint32_t fifo[32];
    int p = 0;

    // SET_SURFACE_FORMAT = A8R8G8B8 (8)
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_FORMAT, 1);
    fifo[p++] = SURFACE_A8R8G8B8;

    // SET_SURFACE_CLIP_HORIZONTAL = 1920
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1);
    fifo[p++] = 1920;

    // SET_SURFACE_CLIP_VERTICAL = 1080
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_CLIP_VERTICAL, 1);
    fifo[p++] = 1080;

    // SET_SURFACE_PITCH_A = 1920 * 4
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_PITCH_A, 1);
    fifo[p++] = 1920 * 4;

    // SET_SURFACE_COLOR_AOFFSET = 0x00100000
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_COLOR_AOFFSET, 1);
    fifo[p++] = 0x00100000;

    // SET_SURFACE_COLOR_TARGET = 1 (color A)
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_COLOR_TARGET, 1);
    fifo[p++] = 1;

    int processed = rsx_process_fifo(&state, fifo, p, nullptr, 1000);

    printf("  Commands processed: %d\n", processed);
    printf("  Surface: %ux%u fmt=%u pitch=%u offset=0x%08X target=%u\n",
           state.surfaceWidth, state.surfaceHeight,
           state.surfaceFormat, state.surfacePitchA,
           state.surfaceOffsetA, state.surfaceColorTarget);

    bool pass = (state.surfaceWidth == 1920 &&
                 state.surfaceHeight == 1080 &&
                 state.surfaceFormat == SURFACE_A8R8G8B8 &&
                 state.surfacePitchA == 1920 * 4 &&
                 state.surfaceOffsetA == 0x00100000 &&
                 state.surfaceColorTarget == 1 &&
                 processed == 6);

    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    rsx_shutdown(&state);
}

// ═══════════════════════════════════════════════════════════════════
// TEST 3: Clear Surface
// ═══════════════════════════════════════════════════════════════════

static void test_clear_surface() {
    printf("\n╔═══════════════════════════════════════════════╗\n");
    printf("║  TEST 3: Clear Surface (Software Rasterizer)  ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");

    RSXState state;
    rsx_init(&state);

    // Small surface for testing: 64x64 A8R8G8B8
    const uint32_t W = 64, H = 64;
    const uint32_t pitch = W * 4;
    const uint32_t offset = 0x1000;  // offset within VRAM

    // Allocate VRAM (only need enough for the test)
    uint32_t vramNeeded = offset + pitch * H;
    uint8_t* vram = (uint8_t*)calloc(vramNeeded, 1);

    // Build FIFO: configure surface, set clear color to 0xFF0000FF (red in ARGB),
    // then clear
    uint32_t fifo[32];
    int p = 0;

    fifo[p++] = fifo_method(NV4097_SET_SURFACE_FORMAT, 1);
    fifo[p++] = SURFACE_A8R8G8B8;
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1);
    fifo[p++] = W;
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_CLIP_VERTICAL, 1);
    fifo[p++] = H;
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_PITCH_A, 1);
    fifo[p++] = pitch;
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_COLOR_AOFFSET, 1);
    fifo[p++] = offset;

    // Clear color = 0xFF0000FF (ARGB red)
    fifo[p++] = fifo_method(NV4097_SET_COLOR_CLEAR_VALUE, 1);
    fifo[p++] = 0xFF0000FF;

    // CLEAR_SURFACE with color bit set
    fifo[p++] = fifo_method(NV4097_CLEAR_SURFACE, 1);
    fifo[p++] = CLEAR_COLOR;

    rsx_process_fifo(&state, fifo, p, vram, 1000);

    // Verify: every pixel in the 64×64 region should be 0xFF0000FF
    bool pass = true;
    int mismatch = 0;
    for (uint32_t y = 0; y < H && pass; y++) {
        uint32_t* row = (uint32_t*)(vram + offset + y * pitch);
        for (uint32_t x = 0; x < W; x++) {
            if (row[x] != 0xFF0000FF) {
                if (mismatch < 3) {
                    printf("  Mismatch at (%u,%u): 0x%08X != 0xFF0000FF\n", x, y, row[x]);
                }
                mismatch++;
                pass = false;
            }
        }
    }

    printf("  Clear color: 0x%08X\n", state.colorClearValue);
    printf("  Pixels checked: %u (%ux%u)\n", W * H, W, H);
    if (mismatch > 0)
        printf("  Mismatches: %d\n", mismatch);
    else
        printf("  All pixels match ✓\n");

    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    free(vram);
    rsx_shutdown(&state);
}

// ═══════════════════════════════════════════════════════════════════
// TEST 4: Vertex Program Upload
// ═══════════════════════════════════════════════════════════════════

static void test_vertex_program_upload() {
    printf("\n╔═══════════════════════════════════════════════╗\n");
    printf("║  TEST 4: Vertex Program Upload                ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");

    RSXState state;
    rsx_init(&state);

    // Upload 4 VP instructions (4 words each = 16 words total)
    // Using incrementing method writes starting at SET_TRANSFORM_PROGRAM
    uint32_t fifo[64];
    int p = 0;

    // SET_TRANSFORM_PROGRAM_START = 0
    fifo[p++] = fifo_method(NV4097_SET_TRANSFORM_PROGRAM_START, 1);
    fifo[p++] = 0;

    // Upload 16 words (4 instructions × 4 words) to SET_TRANSFORM_PROGRAM
    fifo[p++] = fifo_method(NV4097_SET_TRANSFORM_PROGRAM, 16);
    // Instruction 0
    fifo[p++] = 0xDEAD0000;
    fifo[p++] = 0xDEAD0001;
    fifo[p++] = 0xDEAD0002;
    fifo[p++] = 0xDEAD0003;
    // Instruction 1
    fifo[p++] = 0xBEEF0100;
    fifo[p++] = 0xBEEF0101;
    fifo[p++] = 0xBEEF0102;
    fifo[p++] = 0xBEEF0103;
    // Instruction 2
    fifo[p++] = 0xCAFE0200;
    fifo[p++] = 0xCAFE0201;
    fifo[p++] = 0xCAFE0202;
    fifo[p++] = 0xCAFE0203;
    // Instruction 3
    fifo[p++] = 0xF00D0300;
    fifo[p++] = 0xF00D0301;
    fifo[p++] = 0xF00D0302;
    fifo[p++] = 0xF00D0303;

    rsx_process_fifo(&state, fifo, p, nullptr, 1000);

    // Verify uploaded data
    printf("  VP start: %u\n", state.vpStart);
    printf("  Inst 0: %08X %08X %08X %08X\n",
           state.vpData[0], state.vpData[1], state.vpData[2], state.vpData[3]);
    printf("  Inst 1: %08X %08X %08X %08X\n",
           state.vpData[4], state.vpData[5], state.vpData[6], state.vpData[7]);
    printf("  Inst 2: %08X %08X %08X %08X\n",
           state.vpData[8], state.vpData[9], state.vpData[10], state.vpData[11]);
    printf("  Inst 3: %08X %08X %08X %08X\n",
           state.vpData[12], state.vpData[13], state.vpData[14], state.vpData[15]);

    bool pass = (state.vpStart == 0 &&
                 state.vpData[0]  == 0xDEAD0000 &&
                 state.vpData[3]  == 0xDEAD0003 &&
                 state.vpData[4]  == 0xBEEF0100 &&
                 state.vpData[7]  == 0xBEEF0103 &&
                 state.vpData[8]  == 0xCAFE0200 &&
                 state.vpData[11] == 0xCAFE0203 &&
                 state.vpData[12] == 0xF00D0300 &&
                 state.vpData[15] == 0xF00D0303);

    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    rsx_shutdown(&state);
}

// ═══════════════════════════════════════════════════════════════════
// TEST 5: Draw Call Tracking
// ═══════════════════════════════════════════════════════════════════

static void test_draw_call_tracking() {
    printf("\n╔═══════════════════════════════════════════════╗\n");
    printf("║  TEST 5: Draw Call Tracking                   ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");

    RSXState state;
    rsx_init(&state);

    uint32_t fifo[64];
    int p = 0;

    // Draw call 1: 6 triangles (18 verts as TRIANGLES)
    fifo[p++] = fifo_method(NV4097_SET_BEGIN_END, 1);
    fifo[p++] = PRIM_TRIANGLES;
    // DRAW_ARRAYS: first=0, count-1 in bits [31:24] → count = (val>>24)+1
    // Encode: first=0, count=18 → data = ((18-1) << 24) | 0 = 0x11000000
    fifo[p++] = fifo_method(NV4097_DRAW_ARRAYS, 1);
    fifo[p++] = ((18 - 1) << 24) | 0;
    fifo[p++] = fifo_method(NV4097_SET_BEGIN_END, 1);
    fifo[p++] = 0; // END

    // Draw call 2: triangle strip with 5 verts (3 tris)
    fifo[p++] = fifo_method(NV4097_SET_BEGIN_END, 1);
    fifo[p++] = PRIM_TRIANGLE_STRIP;
    fifo[p++] = fifo_method(NV4097_DRAW_ARRAYS, 1);
    fifo[p++] = ((5 - 1) << 24) | 0;
    fifo[p++] = fifo_method(NV4097_SET_BEGIN_END, 1);
    fifo[p++] = 0; // END

    // Draw call 3: indexed quads, 8 verts (4 tris)
    fifo[p++] = fifo_method(NV4097_SET_BEGIN_END, 1);
    fifo[p++] = PRIM_QUADS;
    fifo[p++] = fifo_method(NV4097_DRAW_INDEX_ARRAY, 1);
    fifo[p++] = ((8 - 1) << 24) | 0;
    fifo[p++] = fifo_method(NV4097_SET_BEGIN_END, 1);
    fifo[p++] = 0; // END

    rsx_process_fifo(&state, fifo, p, nullptr, 1000);

    printf("  Draw calls:  %u (expected 3)\n", state.drawCallCount);
    printf("  Triangles:   %u (expected 13)\n", state.triangleCount);
    printf("  Commands:    %u\n", state.cmdCount);

    // Expected triangles: 6 (18/3) + 3 (5-2) + 4 (8/4*2) = 13
    bool pass = (state.drawCallCount == 3 && state.triangleCount == 13);

    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    rsx_shutdown(&state);
}

// ═══════════════════════════════════════════════════════════════════
// TEST 6: Full Frame Simulation
// ═══════════════════════════════════════════════════════════════════

static void test_full_frame() {
    printf("\n╔═══════════════════════════════════════════════╗\n");
    printf("║  TEST 6: Full Frame Simulation                ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");

    RSXState state;
    rsx_init(&state);

    const uint32_t W = 128, H = 128;
    const uint32_t pitch = W * 4;
    const uint32_t colorOffset = 0x0000;
    uint32_t vramNeeded = colorOffset + pitch * H;
    uint8_t* vram = (uint8_t*)calloc(vramNeeded, 1);

    uint32_t fifo[256];
    int p = 0;

    // ── Step 1: Surface setup ──────────────────────────────────
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_FORMAT, 1);
    fifo[p++] = SURFACE_A8R8G8B8;
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1);
    fifo[p++] = W;
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_CLIP_VERTICAL, 1);
    fifo[p++] = H;
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_PITCH_A, 1);
    fifo[p++] = pitch;
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_COLOR_AOFFSET, 1);
    fifo[p++] = colorOffset;
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_COLOR_TARGET, 1);
    fifo[p++] = 1;

    // ── Step 2: Viewport / Scissor ─────────────────────────────
    fifo[p++] = fifo_method(NV4097_SET_VIEWPORT_HORIZONTAL, 1);
    fifo[p++] = (W << 16) | 0;
    fifo[p++] = fifo_method(NV4097_SET_VIEWPORT_VERTICAL, 1);
    fifo[p++] = (H << 16) | 0;
    fifo[p++] = fifo_method(NV4097_SET_SCISSOR_HORIZONTAL, 1);
    fifo[p++] = (W << 16) | 0;
    fifo[p++] = fifo_method(NV4097_SET_SCISSOR_VERTICAL, 1);
    fifo[p++] = (H << 16) | 0;

    // ── Step 3: Render state ───────────────────────────────────
    fifo[p++] = fifo_method(NV4097_SET_DEPTH_TEST_ENABLE, 1);
    fifo[p++] = 1;
    fifo[p++] = fifo_method(NV4097_SET_DEPTH_FUNC, 1);
    fifo[p++] = CMP_LEQUAL;
    fifo[p++] = fifo_method(NV4097_SET_BLEND_ENABLE, 1);
    fifo[p++] = 0;
    fifo[p++] = fifo_method(NV4097_SET_CULL_FACE_ENABLE, 1);
    fifo[p++] = 1;
    fifo[p++] = fifo_method(NV4097_SET_CULL_FACE, 1);
    fifo[p++] = 0x0405;  // GL_BACK

    // ── Step 4: Clear to cornflower blue (0xFF6495ED) ──────────
    fifo[p++] = fifo_method(NV4097_SET_COLOR_CLEAR_VALUE, 1);
    fifo[p++] = 0xFF6495ED;
    fifo[p++] = fifo_method(NV4097_CLEAR_SURFACE, 1);
    fifo[p++] = CLEAR_COLOR | CLEAR_DEPTH;

    // ── Step 5: Vertex program upload (2 instructions) ─────────
    fifo[p++] = fifo_method(NV4097_SET_TRANSFORM_PROGRAM_START, 1);
    fifo[p++] = 0;
    fifo[p++] = fifo_method(NV4097_SET_TRANSFORM_PROGRAM, 8);
    fifo[p++] = 0x401F9C6C;  // fake VP instruction 0
    fifo[p++] = 0x0040000D;
    fifo[p++] = 0x8106C083;
    fifo[p++] = 0x60401F80;
    fifo[p++] = 0x401F9C6C;  // fake VP instruction 1
    fifo[p++] = 0x0040000D;
    fifo[p++] = 0x8106C083;
    fifo[p++] = 0x60401F81;

    // ── Step 6: Fragment program ───────────────────────────────
    fifo[p++] = fifo_method(NV4097_SET_SHADER_PROGRAM, 1);
    fifo[p++] = 0x00050000;  // offset=0x50000 in VRAM

    // ── Step 7: Vertex array setup (slot 0: float3 position) ──
    fifo[p++] = fifo_method(NV4097_SET_VERTEX_DATA_ARRAY_FORMAT, 1);
    fifo[p++] = (12 << 8) | (3 << 4) | VERTEX_F;  // stride=12, size=3, type=float
    fifo[p++] = fifo_method(NV4097_SET_VERTEX_DATA_ARRAY_OFFSET, 1);
    fifo[p++] = 0x00010000;

    // ── Step 8: Texture unit 0 ─────────────────────────────────
    fifo[p++] = fifo_method(NV4097_SET_TEXTURE_OFFSET, 1);
    fifo[p++] = 0x00080000;  // texture at VRAM offset 0x80000
    fifo[p++] = fifo_method(NV4097_SET_TEXTURE_FORMAT, 1);
    fifo[p++] = 0x00001A08;  // A8R8G8B8 texture format bits
    fifo[p++] = fifo_method(NV4097_SET_TEXTURE_CONTROL0, 1);
    fifo[p++] = 0x80000000;  // enable

    // ── Step 9: Draw triangles ─────────────────────────────────
    fifo[p++] = fifo_method(NV4097_SET_BEGIN_END, 1);
    fifo[p++] = PRIM_TRIANGLES;

    // Two draw calls: 12 tris (36 verts) + 6 tris (18 verts)
    fifo[p++] = fifo_method(NV4097_DRAW_ARRAYS, 1);
    fifo[p++] = ((36 - 1) << 24) | 0;
    fifo[p++] = fifo_method(NV4097_DRAW_ARRAYS, 1);
    fifo[p++] = ((18 - 1) << 24) | 0;

    fifo[p++] = fifo_method(NV4097_SET_BEGIN_END, 1);
    fifo[p++] = 0; // END

    // ── Step 10: Flip ──────────────────────────────────────────
    fifo[p++] = fifo_method(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1);
    fifo[p++] = colorOffset;

    // Process entire frame
    auto t0 = std::chrono::high_resolution_clock::now();
    int processed = rsx_process_fifo(&state, fifo, p, vram, 10000);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("  Frame processed in %.3f ms (%d method writes)\n", ms, processed);

    // Print full state
    rsx_print_state(&state);

    // Verify clear
    uint32_t* pixels = (uint32_t*)(vram + colorOffset);
    bool clearOK = true;
    for (uint32_t i = 0; i < W * H; i++) {
        if (pixels[i] != 0xFF6495ED) { clearOK = false; break; }
    }

    bool pass = true;

    // Check surface
    if (state.surfaceWidth != W || state.surfaceHeight != H) {
        printf("  ❌ Surface dimensions wrong\n"); pass = false;
    }
    // Check state
    if (!state.depthTestEnable || state.depthFunc != CMP_LEQUAL) {
        printf("  ❌ Depth state wrong\n"); pass = false;
    }
    if (!state.cullFaceEnable || state.cullFace != 0x0405) {
        printf("  ❌ Cull state wrong\n"); pass = false;
    }
    // Check clear
    if (!clearOK) {
        printf("  ❌ Clear pixels wrong\n"); pass = false;
    }
    // Check VP upload
    if (state.vpData[0] != 0x401F9C6C || state.vpData[7] != 0x60401F81) {
        printf("  ❌ VP data wrong\n"); pass = false;
    }
    // Check FP
    if (state.fpOffset != 0x00050000) {
        printf("  ❌ FP offset wrong: 0x%08X\n", state.fpOffset); pass = false;
    }
    // Check vertex array
    if (!state.vertexArrays[0].enabled || state.vertexArrays[0].offset != 0x00010000) {
        printf("  ❌ Vertex array wrong\n"); pass = false;
    }
    // Check texture
    if (!state.textures[0].enabled || state.textures[0].offset != 0x00080000) {
        printf("  ❌ Texture wrong\n"); pass = false;
    }
    // Check draw calls: 2 calls, 12 + 6 = 18 triangles
    if (state.drawCallCount != 2 || state.triangleCount != 18) {
        printf("  ❌ Draw calls: %u (exp 2), tris: %u (exp 18)\n",
               state.drawCallCount, state.triangleCount); pass = false;
    }
    // Check frame count
    if (state.frameCount != 1) {
        printf("  ❌ Frame count: %u (expected 1)\n", state.frameCount); pass = false;
    }

    printf("  Result: %s\n", pass ? "✅ PASS" : "❌ FAIL");
    if (pass) total_pass++; else total_fail++;

    free(vram);
    rsx_shutdown(&state);
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  🎮 RSX Reality Synthesizer — Command Processor  ║\n");
    printf("║  NV47 FIFO + NV4097 3D Engine (Phase 4a)         ║\n");
    printf("║  Project Megakernel — PS3 on Tesla V100          ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    test_fifo_parsing();
    test_surface_setup();
    test_clear_surface();
    test_vertex_program_upload();
    test_draw_call_tracking();
    test_full_frame();

    printf("\n═══════════════════════════════════════════════════\n");
    printf("  Results: %d/%d tests passed\n", total_pass, total_pass + total_fail);
    printf("═══════════════════════════════════════════════════\n");

    return (total_fail == 0) ? 0 : 1;
}
