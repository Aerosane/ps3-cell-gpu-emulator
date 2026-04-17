// test_rsx_vulkan.cu — End-to-end RSX → Vulkan translation test
//
// Builds a FIFO that performs:
//   1. surface format/size setup
//   2. viewport + scissor
//   3. clear color + CLEAR_SURFACE
//   4. BEGIN(TRIANGLES) → DRAW_ARRAYS(count=3) → END
//   5. FLIP
// Then runs the dispatcher with a VulkanEmitter attached and verifies
// the emitted Vulkan command stream matches expectations.

#include "rsx_defs.h"
#include "rsx_vulkan_emitter.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace rsx;

// Declarations from rsx_command_processor.cu
namespace rsx {
    int  rsx_init(RSXState* state);
    int  rsx_process_fifo(RSXState* state, const uint32_t* fifo, uint32_t fifoSize,
                          uint8_t* vram, uint32_t maxCmds);
    void rsx_shutdown(RSXState* state);
}

static int pass = 0, fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  ✓ %s\n", msg); pass++; } \
    else      { printf("  ✗ %s\n", msg); fail++; } \
} while (0)

static uint32_t fifo_incr(uint32_t method, uint32_t count) {
    return ((count & 0x7FF) << 18) | ((method >> 2) << 2);
}

static void test_single_frame() {
    printf("── Test: single-frame clear+draw emits correct Vulkan stream\n");

    RSXState st;
    rsx_init(&st);
    VulkanEmitter emit;
    st.vulkanEmitter = &emit;

    // Build FIFO
    uint32_t fifo[128];
    size_t n = 0;
    // Surface setup: format, clip H, clip V, pitch
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_FORMAT, 1);        fifo[n++] = SURFACE_A8R8G8B8;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1); fifo[n++] = 1280u;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_CLIP_VERTICAL,   1); fifo[n++] = 720u;
    // Viewport (H then V triggers emit)
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_HORIZONTAL, 1);   fifo[n++] = (1280u << 16) | 0u;
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_VERTICAL,   1);   fifo[n++] = (720u  << 16) | 0u;
    // Scissor
    fifo[n++] = fifo_incr(NV4097_SET_SCISSOR_HORIZONTAL, 1);    fifo[n++] = (1280u << 16) | 0u;
    fifo[n++] = fifo_incr(NV4097_SET_SCISSOR_VERTICAL,   1);    fifo[n++] = (720u  << 16) | 0u;
    // Clear
    fifo[n++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1);     fifo[n++] = 0xFF204080u;
    fifo[n++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);             fifo[n++] = CLEAR_COLOR | CLEAR_DEPTH;
    // Draw
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);             fifo[n++] = PRIM_TRIANGLES;
    fifo[n++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);               fifo[n++] = (2u << 24) | 0u; // 3 verts
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);             fifo[n++] = 0u;
    // Flip
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1); fifo[n++] = 0x00100000u;

    uint8_t* vram = (uint8_t*)calloc(1, 8 * 1024 * 1024);
    int cmds = rsx_process_fifo(&st, fifo, (uint32_t)n, vram, 1024);

    CHECK(cmds > 0, "dispatcher processed commands");
    CHECK(emit.counters.beginFrame  == 1, "exactly one BeginFrame emitted");
    CHECK(emit.counters.viewports   == 1, "one Viewport emitted");
    CHECK(emit.counters.scissors    == 1, "one Scissor emitted");
    CHECK(emit.counters.clears      == 1, "one ClearAttachment emitted");
    CHECK(emit.counters.bindPipelines == 1, "one BindPipeline on first draw");
    CHECK(emit.counters.draws       == 1, "one Draw emitted");
    CHECK(emit.counters.presents    == 1, "one Present emitted");
    CHECK(emit.counters.endRenderPasses == 1, "one EndRenderPass before Present");

    // Verify ordering: BeginFrame before Draw before Present
    size_t idxBegin = (size_t)-1, idxDraw = (size_t)-1, idxPresent = (size_t)-1;
    for (size_t i = 0; i < emit.size(); i++) {
        if (emit[i].op == VkOp::BeginFrame && idxBegin   == (size_t)-1) idxBegin   = i;
        if (emit[i].op == VkOp::Draw       && idxDraw    == (size_t)-1) idxDraw    = i;
        if (emit[i].op == VkOp::Present    && idxPresent == (size_t)-1) idxPresent = i;
    }
    CHECK(idxBegin < idxDraw && idxDraw < idxPresent, "ordering BeginFrame < Draw < Present");

    // Find the Draw record and verify parameters
    bool drawOk = false;
    for (size_t i = 0; i < emit.size(); i++) {
        if (emit[i].op == VkOp::Draw) {
            drawOk = (emit[i].draw.vertexCount == 3 &&
                      emit[i].draw.prim        == PRIM_TRIANGLES);
            break;
        }
    }
    CHECK(drawOk, "Draw has vertexCount=3 prim=TRIANGLES");

    bool clearOk = false;
    for (size_t i = 0; i < emit.size(); i++) {
        if (emit[i].op == VkOp::ClearAttachment) {
            clearOk = (emit[i].clear.color == 0xFF204080u &&
                       (emit[i].clear.mask & CLEAR_COLOR));
            break;
        }
    }
    CHECK(clearOk, "Clear color=0xFF204080 with COLOR bit set");

    free(vram);
    rsx_shutdown(&st);
}

static void test_multiframe() {
    printf("── Test: multi-frame stream produces one Present per FLIP\n");

    RSXState st;
    rsx_init(&st);
    VulkanEmitter emit;
    st.vulkanEmitter = &emit;

    uint32_t fifo[256];
    size_t n = 0;
    // Setup once
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1); fifo[n++] = 1920u;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_CLIP_VERTICAL,   1); fifo[n++] = 1080u;
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_HORIZONTAL, 1); fifo[n++] = (1920u << 16);
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_VERTICAL,   1); fifo[n++] = (1080u << 16);
    // 3 frames of: clear → draw → flip
    for (int f = 0; f < 3; f++) {
        fifo[n++] = fifo_incr(NV4097_SET_COLOR_CLEAR_VALUE, 1); fifo[n++] = 0xFF000000u | (f * 0x555555u);
        fifo[n++] = fifo_incr(NV4097_CLEAR_SURFACE, 1);         fifo[n++] = CLEAR_COLOR;
        fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo[n++] = PRIM_TRIANGLES;
        fifo[n++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);           fifo[n++] = ((f * 3u + 2u) << 24);
        fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);         fifo[n++] = 0u;
        fifo[n++] = fifo_incr(NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, 1); fifo[n++] = 0x00100000u + f * 0x400000u;
    }

    uint8_t* vram = (uint8_t*)calloc(1, 16 * 1024 * 1024);
    rsx_process_fifo(&st, fifo, (uint32_t)n, vram, 4096);

    CHECK(emit.counters.presents == 3, "3 Present records");
    CHECK(emit.counters.clears   == 3, "3 Clear records");
    CHECK(emit.counters.draws    == 3, "3 Draw records");
    CHECK(emit.counters.beginFrame == 3, "3 BeginFrame (one per frame after Present invalidates)");
    CHECK(emit.counters.endRenderPasses == 3, "3 EndRenderPass");

    free(vram);
    rsx_shutdown(&st);
}

static void test_pipeline_cache() {
    printf("── Test: repeated draws with same state reuse pipeline\n");

    RSXState st;
    rsx_init(&st);
    VulkanEmitter emit;
    st.vulkanEmitter = &emit;

    uint32_t fifo[128];
    size_t n = 0;
    fifo[n++] = fifo_incr(NV4097_SET_SURFACE_CLIP_HORIZONTAL, 1); fifo[n++] = 1280u;
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_HORIZONTAL, 1); fifo[n++] = (1280u << 16);
    fifo[n++] = fifo_incr(NV4097_SET_VIEWPORT_VERTICAL,   1); fifo[n++] = (720u  << 16);
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);           fifo[n++] = PRIM_TRIANGLES;
    for (int i = 0; i < 5; i++) {
        fifo[n++] = fifo_incr(NV4097_DRAW_ARRAYS, 1);         fifo[n++] = (2u << 24);
    }
    fifo[n++] = fifo_incr(NV4097_SET_BEGIN_END, 1);           fifo[n++] = 0u;

    uint8_t* vram = (uint8_t*)calloc(1, 8 * 1024 * 1024);
    rsx_process_fifo(&st, fifo, (uint32_t)n, vram, 1024);

    CHECK(emit.counters.draws == 5, "5 draws emitted");
    CHECK(emit.counters.bindPipelines == 1, "only 1 BindPipeline for 5 identical draws");

    free(vram);
    rsx_shutdown(&st);
}

int main() {
    printf("═══════════════════════════════════════════════════\n");
    printf("  RSX → Vulkan emitter tests\n");
    printf("═══════════════════════════════════════════════════\n\n");

    test_single_frame();
    printf("\n");
    test_multiframe();
    printf("\n");
    test_pipeline_cache();

    printf("\n═══════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass, fail);
    printf("═══════════════════════════════════════════════════\n");
    return fail ? 1 : 0;
}
