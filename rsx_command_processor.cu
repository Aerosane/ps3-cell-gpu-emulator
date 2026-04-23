// rsx_command_processor.cu — RSX Reality Synthesizer Command Processor
//
// Processes NV47 GPU commands from the FIFO command buffer.
// Phase 4a: Parse + state tracking (no actual rendering yet).
// Phase 4b will add Vulkan rasterization backend.
//
// Part of Project Megakernel — CUDA-based PS3 emulation on Tesla V100.

#include "rsx_defs.h"

// ═══════════════════════════════════════════════════════════════════
// Optional Vulkan emitter bridge — weak symbols so both CUDA builds
// (no emitter linked) and host-only builds (emitter present) work.
// ═══════════════════════════════════════════════════════════════════
extern "C" {
void rsx_emitter_onSurfaceSetup(void* emitter, const rsx::RSXState* s)     __attribute__((weak));
void rsx_emitter_onViewport   (void* emitter, const rsx::RSXState* s)      __attribute__((weak));
void rsx_emitter_onScissor    (void* emitter, const rsx::RSXState* s)      __attribute__((weak));
void rsx_emitter_onClearSurface(void* emitter, const rsx::RSXState* s, uint32_t mask) __attribute__((weak));
void rsx_emitter_onBeginEnd   (void* emitter, const rsx::RSXState* s, uint32_t prim)  __attribute__((weak));
void rsx_emitter_onDrawArrays (void* emitter, const rsx::RSXState* s, uint32_t first, uint32_t count) __attribute__((weak));
void rsx_emitter_onDrawIndexed(void* emitter, const rsx::RSXState* s, uint32_t first, uint32_t count, uint32_t fmt) __attribute__((weak));
void rsx_emitter_onFlip       (void* emitter, const rsx::RSXState* s, uint32_t surfaceOffset) __attribute__((weak));
}

#define RSX_EMIT(fn, ...) do { \
    if (state->vulkanEmitter && rsx_emitter_##fn) \
        rsx_emitter_##fn(state->vulkanEmitter, __VA_ARGS__); \
} while (0)

#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace rsx {

// ═══════════════════════════════════════════════════════════════════
// Initialization / Shutdown
// ═══════════════════════════════════════════════════════════════════

int rsx_init(RSXState* state) {
    if (!state) return -1;
    memset(state, 0, sizeof(RSXState));

    // Default surface: 1280×720 A8R8G8B8 targeting a high VRAM region
    // so clears don't stomp low-memory guest structures (vertex/index
    // buffers that games frequently place at small EAs like 0x400).
    state->surfaceFormat  = SURFACE_A8R8G8B8;
    state->surfaceWidth   = 1280;
    state->surfaceHeight  = 720;
    state->surfacePitchA  = 1280 * 4;
    state->surfacePitchB  = 1280 * 4;
    state->surfaceOffsetA = 0x02000000;   // 32 MB into VRAM
    state->surfaceColorTarget = 1;  // color A only

    // Default viewport / scissor
    state->viewportX = 0;  state->viewportY = 0;
    state->viewportW = 1280; state->viewportH = 720;
    state->scissorX  = 0;  state->scissorY  = 0;
    state->scissorW  = 4096; state->scissorH = 4096;

    // Depth defaults
    state->depthTestEnable = false;
    state->depthFunc       = CMP_LESS;
    state->depthMask       = true;   // depth writes on by default (GL)
    state->depthFormat     = DEPTH_Z24S8;

    // Stencil defaults
    state->stencilTestEnable = false;
    state->stencilFunc       = 0x207;   // ALWAYS
    state->stencilRef        = 0;
    state->stencilFuncMask   = 0xFF;
    state->stencilWriteMask  = 0xFF;
    state->stencilOpFail     = 0x1E00;  // KEEP
    state->stencilOpZFail    = 0x1E00;
    state->stencilOpZPass    = 0x1E00;

    // Blend / cull
    state->blendEnable    = false;
    state->blendSFactor   = 0x00010001; // ONE,ONE  (low16 RGB, high16 A)
    state->blendDFactor   = 0x00000000; // ZERO,ZERO
    state->blendEquation  = 0x80068006; // ADD,ADD
    state->blendColor     = 0;
    state->cullFaceEnable = false;
    state->cullFace       = 0x0405; // GL_BACK

    // Alpha test / front face / color mask / shade mode defaults
    state->alphaTestEnable = false;
    state->alphaFunc       = 0x0207;     // ALWAYS
    state->alphaRef        = 0;
    state->frontFace       = 0x0901;     // CCW
    state->colorMask       = 0xFFFFFFFF; // all writes enabled
    state->shadeMode       = 0x1D01;     // SMOOTH

    // Draw state
    state->currentPrim = PRIM_TRIANGLES;
    state->inBeginEnd  = false;

    return 0;
}

void rsx_shutdown(RSXState* state) {
    if (!state) return;
    // No dynamic allocations yet; just zero the state
    memset(state, 0, sizeof(RSXState));
}

// ═══════════════════════════════════════════════════════════════════
// Clear Surface — software implementation
// ═══════════════════════════════════════════════════════════════════

void rsx_clear_surface(RSXState* state, uint8_t* vram, uint32_t clearMask) {
    if (!state || !vram) return;

    if (clearMask & CLEAR_COLOR) {
        uint32_t offset = state->surfaceOffsetA;
        uint32_t pitch  = state->surfacePitchA;
        uint32_t w      = state->surfaceWidth;
        uint32_t h      = state->surfaceHeight;
        uint32_t color  = state->colorClearValue;

        // Bytes per pixel (assume 4 for A8R8G8B8 / X8R8G8B8)
        uint32_t bpp = 4;
        if (state->surfaceFormat == SURFACE_R5G6B5) bpp = 2;

        for (uint32_t y = 0; y < h; y++) {
            uint32_t rowOff = offset + y * pitch;
            if (rowOff + w * bpp > VRAM_SIZE) break;

            if (bpp == 4) {
                uint32_t* row = (uint32_t*)(vram + rowOff);
                for (uint32_t x = 0; x < w; x++) {
                    row[x] = color;
                }
            } else {
                // R5G6B5: truncate clear value
                uint16_t c16 = (uint16_t)(color & 0xFFFF);
                uint16_t* row = (uint16_t*)(vram + rowOff);
                for (uint32_t x = 0; x < w; x++) {
                    row[x] = c16;
                }
            }
        }
    }

    // Depth / stencil clear is a no-op for Phase 4a
}

// ═══════════════════════════════════════════════════════════════════
// Estimate triangle count from draw call parameters
// ═══════════════════════════════════════════════════════════════════

static uint32_t estimateTriangles(PrimitiveType prim, uint32_t vertexCount) {
    switch (prim) {
        case PRIM_TRIANGLES:      return vertexCount / 3;
        case PRIM_TRIANGLE_STRIP: return (vertexCount >= 3) ? vertexCount - 2 : 0;
        case PRIM_TRIANGLE_FAN:   return (vertexCount >= 3) ? vertexCount - 2 : 0;
        case PRIM_QUADS:          return (vertexCount / 4) * 2;
        case PRIM_QUAD_STRIP:     return (vertexCount >= 4) ? ((vertexCount - 2) / 2) * 2 : 0;
        case PRIM_LINES:          return 0;
        case PRIM_LINE_STRIP:     return 0;
        case PRIM_POINTS:         return 0;
        default:                  return 0;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Method Dispatch — handle a single NV4097 method write
// ═══════════════════════════════════════════════════════════════════

static void dispatchMethod(RSXState* state, uint8_t* vram,
                           uint32_t method, uint32_t data) {
    // ── Surface setup ──────────────────────────────────────────
    switch (method) {
    case NV4097_SET_SURFACE_FORMAT:
        if (data) state->surfaceFormat = data & 0x1F;
        RSX_EMIT(onSurfaceSetup, state);
        return;
    case NV4097_SET_SURFACE_CLIP_HORIZONTAL:
        if (data & 0xFFFF) {
            state->surfaceWidth  = data & 0xFFFF;
            state->viewportX     = 0;
            state->viewportW     = state->surfaceWidth;
        }
        return;
    case NV4097_SET_SURFACE_CLIP_VERTICAL:
        if (data & 0xFFFF) {
            state->surfaceHeight = data & 0xFFFF;
            state->viewportY     = 0;
            state->viewportH     = state->surfaceHeight;
        }
        return;
    case NV4097_SET_SURFACE_PITCH_A:
        if (data) state->surfacePitchA = data;
        return;
    case NV4097_SET_SURFACE_PITCH_B:
        state->surfacePitchB = data;
        return;
    case NV4097_SET_SURFACE_PITCH_C:
        state->surfacePitchC = data;
        return;
    case NV4097_SET_SURFACE_PITCH_D:
        state->surfacePitchD = data;
        return;
    case NV4097_SET_SURFACE_COLOR_AOFFSET:
        if (data) state->surfaceOffsetA = data;
        return;
    case NV4097_SET_SURFACE_COLOR_BOFFSET:
        if (data) state->surfaceOffsetB = data;
        return;
    case NV4097_SET_SURFACE_COLOR_COFFSET:
        if (data) state->surfaceOffsetC = data;
        return;
    case NV4097_SET_SURFACE_COLOR_DOFFSET:
        if (data) state->surfaceOffsetD = data;
        return;
    case NV4097_SET_SURFACE_COLOR_TARGET:
        state->surfaceColorTarget = data;
        return;

    // ── Index buffer state (VRAM-resident indexed draws) ───────
    case NV4097_SET_INDEX_ARRAY_ADDRESS:
        state->indexArrayAddress = data;
        return;
    case NV4097_SET_INDEX_ARRAY_DMA:
        state->indexArrayFormat = data;
        return;

    // ── Viewport / Scissor ─────────────────────────────────────
    case NV4097_SET_VIEWPORT_HORIZONTAL:
        // Some RSX programs (e.g. the PSL1GHT basic-triangle demo) write
        // 0 here and rely on VIEWPORT_SCALE/OFFSET for the actual NDC→
        // raster transform; treat 0-width as "keep surface default".
        if (data != 0) {
            state->viewportX = (uint16_t)(data & 0xFFFF);
            state->viewportW = (uint16_t)(data >> 16);
        }
        return;
    case NV4097_SET_VIEWPORT_VERTICAL:
        if (data != 0) {
            state->viewportY = (uint16_t)(data & 0xFFFF);
            state->viewportH = (uint16_t)(data >> 16);
        }
        RSX_EMIT(onViewport, state);
        return;
    case NV4097_SET_SCISSOR_HORIZONTAL:
        state->scissorX = (uint16_t)(data & 0xFFFF);
        state->scissorW = (uint16_t)(data >> 16);
        return;
    case NV4097_SET_SCISSOR_VERTICAL:
        state->scissorY = (uint16_t)(data & 0xFFFF);
        state->scissorH = (uint16_t)(data >> 16);
        RSX_EMIT(onScissor, state);
        return;

    // ── Depth / Stencil ────────────────────────────────────────
    case NV4097_SET_DEPTH_TEST_ENABLE:
        state->depthTestEnable = (data != 0);
        return;
    case NV4097_SET_DEPTH_FUNC:
        state->depthFunc = data;
        return;
    case NV4097_SET_DEPTH_MASK:
        state->depthMask = (data != 0);
        return;
    case NV4097_SET_STENCIL_TEST_ENABLE:
        state->stencilTestEnable = (data != 0);
        return;
    case NV4097_SET_STENCIL_MASK:
        state->stencilWriteMask = data;
        return;
    case NV4097_SET_STENCIL_FUNC:
        state->stencilFunc = data;
        return;
    case NV4097_SET_STENCIL_FUNC_REF:
        state->stencilRef = data;
        return;
    case NV4097_SET_STENCIL_FUNC_MASK:
        state->stencilFuncMask = data;
        return;
    case NV4097_SET_STENCIL_OP_FAIL:
        state->stencilOpFail = data;
        return;
    case NV4097_SET_STENCIL_OP_ZFAIL:
        state->stencilOpZFail = data;
        return;
    case NV4097_SET_STENCIL_OP_ZPASS:
        state->stencilOpZPass = data;
        return;

    // ── Blend ──────────────────────────────────────────────────
    case NV4097_SET_BLEND_ENABLE:
        state->blendEnable = (data != 0);
        return;
    case NV4097_SET_BLEND_FUNC_SFACTOR:
        state->blendSFactor = data;
        return;
    case NV4097_SET_BLEND_FUNC_DFACTOR:
        state->blendDFactor = data;
        return;
    case NV4097_SET_BLEND_EQUATION:
        state->blendEquation = data;
        return;
    case NV4097_SET_BLEND_COLOR:
        state->blendColor = data;
        return;

    // ── Cull face ──────────────────────────────────────────────
    case NV4097_SET_CULL_FACE_ENABLE:
        state->cullFaceEnable = (data != 0);
        return;
    case NV4097_SET_CULL_FACE:
        state->cullFace = data;
        return;
    case NV4097_SET_FRONT_FACE:
        state->frontFace = data;
        return;

    // ── Alpha test / color mask / shade mode ───────────────────
    case NV4097_SET_ALPHA_TEST_ENABLE:
        state->alphaTestEnable = (data != 0);
        return;
    case NV4097_SET_ALPHA_FUNC:
        state->alphaFunc = data;
        return;
    case NV4097_SET_ALPHA_REF:
        state->alphaRef = data;
        return;
    case NV4097_SET_COLOR_MASK:
        state->colorMask = data;
        return;
    case NV4097_SET_SHADE_MODE:
        state->shadeMode = data;
        return;

    // ── Clear ──────────────────────────────────────────────────
    case NV4097_SET_COLOR_CLEAR_VALUE:
        state->colorClearValue = data;
        return;
    case NV4097_CLEAR_SURFACE:
        rsx_clear_surface(state, vram, data);
        RSX_EMIT(onClearSurface, state, data);
        return;

    // ── Shader programs ────────────────────────────────────────
    case NV4097_SET_TRANSFORM_PROGRAM_START:
        state->vpStart = data;
        return;
    case NV4097_SET_TRANSFORM_PROGRAM_LOAD:
        // Sets the word-index into vpData[] where the next 0x0B80 window
        // write lands. Hardware stores programs as 4-word instructions;
        // "data" is the instruction index.
        state->vpLoadOffset = data * 4;
        return;
    case NV4097_SET_TRANSFORM_CONSTANT_LOAD:
        // Base vec4 index for subsequent 0x1F00 window writes.
        state->vpConstantLoad = data;
        return;
    case NV4097_SET_SHADER_PROGRAM:
        state->fpOffset  = data & 0xFFFFFFF0;  // bits [31:4] = offset
        state->fpControl = data & 0x0F;
        return;

    // ── Draw ───────────────────────────────────────────────────
    case NV4097_SET_BEGIN_END:
        if (data != 0) {
            state->currentPrim = (PrimitiveType)data;
            state->inBeginEnd  = true;
        } else {
            // END — finalize draw batch
            state->inBeginEnd = false;
        }
        RSX_EMIT(onBeginEnd, state, data);
        return;
    case NV4097_DRAW_ARRAYS: {
        uint32_t first = data & 0xFFFFFF;
        uint32_t count = ((data >> 24) & 0xFF) + 1;
        state->drawCallCount++;
        state->triangleCount += estimateTriangles(state->currentPrim, count);
        RSX_EMIT(onDrawArrays, state, first, count);
        return;
    }
    case NV4097_DRAW_INDEX_ARRAY: {
        uint32_t first = data & 0xFFFFFF;
        uint32_t count = ((data >> 24) & 0xFF) + 1;
        state->drawCallCount++;
        state->triangleCount += estimateTriangles(state->currentPrim, count);
        RSX_EMIT(onDrawIndexed, state, first, count, 0u);
        return;
    }

    // ── Flip (present) ─────────────────────────────────────────
    case NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP:
        state->surfaceOffsetA = data;
        state->frameCount++;
        RSX_EMIT(onFlip, state, data);
        return;

    // ── Reference register ─────────────────────────────────────
    case NV4097_SET_REFERENCE:
        state->ref = data;
        return;

    // ── Control / NOP ──────────────────────────────────────────
    case NV4097_NO_OPERATION:
        return;
    case NV4097_SET_OBJECT:
        return;
    case NV4097_SET_CONTEXT_DMA_NOTIFIES:
    case NV4097_SET_CONTEXT_DMA_COLOR_A:
        return;

    default:
        break;
    }

    // ── Vertex array format (16 slots at stride 4) ────────────
    if (method >= NV4097_SET_VERTEX_DATA_ARRAY_FORMAT &&
        method <  NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 16 * 4) {
        uint32_t slot = (method - NV4097_SET_VERTEX_DATA_ARRAY_FORMAT) / 4;
        state->vertexArrays[slot].format  = data;
        state->vertexArrays[slot].enabled = (data != 0);
        return;
    }

    // ── Vertex array offset (16 slots at stride 4) ────────────
    if (method >= NV4097_SET_VERTEX_DATA_ARRAY_OFFSET &&
        method <  NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + 16 * 4) {
        uint32_t slot = (method - NV4097_SET_VERTEX_DATA_ARRAY_OFFSET) / 4;
        state->vertexArrays[slot].offset = data;
        return;
    }

    // ── Textures (16 units, stride 0x20 per unit) ──────────────
    // Check textures before transform program — their address ranges overlap
    if (method >= NV4097_SET_TEXTURE_OFFSET &&
        method <  NV4097_SET_TEXTURE_OFFSET + 16 * 0x20) {
        uint32_t unit      = (method - NV4097_SET_TEXTURE_OFFSET) / 0x20;
        uint32_t regOffset = (method - NV4097_SET_TEXTURE_OFFSET) % 0x20;
        if (unit < 16) {
            switch (regOffset) {
            case 0x00: // TEXTURE_OFFSET
                state->textures[unit].offset  = data;
                state->textures[unit].enabled = true;
                break;
            case 0x04: // TEXTURE_FORMAT
                state->textures[unit].format = data;
                break;
            case 0x0C: // TEXTURE_CONTROL0
                state->textures[unit].control0 = data;
                break;
            case 0x18: // TEXTURE_IMAGE_RECT  (width<<16 | height)
                state->textures[unit].width  = (data >> 16) & 0xFFFF;
                state->textures[unit].height =  data        & 0xFFFF;
                break;
            default:
                break;
            }
        }
        return;
    }

    // ── Transform program upload (sequential words) ────────────
    // NV47 uploads VP microcode through a small window at 0x0B80.
    // Real hardware uses non-incrementing writes; limit to 32 words per batch.
    if (method >= NV4097_SET_TRANSFORM_PROGRAM &&
        method <  NV4097_SET_TRANSFORM_PROGRAM + 32 * 4) {
        uint32_t idx = state->vpLoadOffset +
                       (method - NV4097_SET_TRANSFORM_PROGRAM) / 4;
        if (idx < 512 * 4) {
            state->vpData[idx] = data;
            state->vpValid = 1;
        }
        return;
    }

    // ── Transform constants (468 vec4s on NV47) ────────────────
    // Window at 0x1F00 holds up to 32 words (= 8 vec4s) per batch; the
    // real constant target index is vpConstantLoad + byteIndex/16. This
    // matches set_transform_constant::decode_one in rpcs3/NV47/HW/nv4097.
    if (method >= NV4097_SET_TRANSFORM_CONSTANT &&
        method <  NV4097_SET_TRANSFORM_CONSTANT + 32 * 4) {
        uint32_t byteOff  = method - NV4097_SET_TRANSFORM_CONSTANT;
        uint32_t vecOff   = byteOff / 16;          // vec4 offset within batch
        uint32_t comp     = (byteOff / 4) % 4;     // component inside vec4
        uint32_t vec      = state->vpConstantLoad + vecOff;
        if (vec < 512) {
            float fval;
            memcpy(&fval, &data, sizeof(float));
            state->vpConstants[vec][comp] = fval;
        }
        return;
    }

    // Unhandled method — silently ignore in Phase 4a
}

// ═══════════════════════════════════════════════════════════════════
// FIFO Processor — main command loop
// ═══════════════════════════════════════════════════════════════════

int rsx_process_fifo(RSXState* state, const uint32_t* fifo, uint32_t fifoSize,
                     uint8_t* vram, uint32_t maxCmds) {
    if (!state || !fifo) return -1;

    uint32_t pos  = 0;       // current position in fifo (word index)
    uint32_t cmds = 0;       // commands processed

    while (pos < fifoSize && cmds < maxCmds) {
        uint32_t header = fifo[pos++];
        if (header == 0) continue;  // skip padding

        FIFOCommand cmd = parseFIFOHeader(header);

        // Handle FIFO flow control
        if (cmd.isJump || cmd.isCall) {
            // In real hardware, this redirects the FIFO read.
            // For testing with a flat buffer, we just stop.
            break;
        }
        if (cmd.isReturn) {
            break;
        }

        // Process method + data words
        uint32_t method = cmd.method;
        for (uint32_t i = 0; i < cmd.count && pos < fifoSize; i++) {
            uint32_t data = fifo[pos++];
            dispatchMethod(state, vram, method, data);
            state->cmdCount++;
            cmds++;

            if (!cmd.isNonIncr) {
                method += 4;  // incrementing methods advance the offset
            }
        }
    }

    state->get = pos * 4;  // update GET pointer (byte offset)
    return (int)cmds;
}

// ═══════════════════════════════════════════════════════════════════
// Debug / Statistics
// ═══════════════════════════════════════════════════════════════════

void rsx_print_state(const RSXState* state) {
    if (!state) return;

    printf("┌─────────────────────────────────────────────────┐\n");
    printf("│ RSX State Summary                               │\n");
    printf("├─────────────────────────────────────────────────┤\n");
    printf("│ Surface: %ux%u fmt=%u pitch=%u              \n",
           state->surfaceWidth, state->surfaceHeight,
           state->surfaceFormat, state->surfacePitchA);
    printf("│ Color A offset: 0x%08X                      \n", state->surfaceOffsetA);
    printf("│ Color B offset: 0x%08X                      \n", state->surfaceOffsetB);
    printf("│ Viewport: (%u,%u) %ux%u                    \n",
           state->viewportX, state->viewportY,
           state->viewportW, state->viewportH);
    printf("│ Scissor:  (%u,%u) %ux%u                    \n",
           state->scissorX, state->scissorY,
           state->scissorW, state->scissorH);
    printf("│ Depth: test=%s func=0x%04X                 \n",
           state->depthTestEnable ? "ON" : "OFF", state->depthFunc);
    printf("│ Blend: %s  Cull: %s (0x%04X)               \n",
           state->blendEnable ? "ON" : "OFF",
           state->cullFaceEnable ? "ON" : "OFF",
           state->cullFace);
    printf("│ VP start=%u  FP offset=0x%08X              \n",
           state->vpStart, state->fpOffset);
    printf("│ Clear color: 0x%08X                        \n", state->colorClearValue);
    printf("├─────────────────────────────────────────────────┤\n");
    printf("│ Draw calls:  %u                                \n", state->drawCallCount);
    printf("│ Triangles:   %u                                \n", state->triangleCount);
    printf("│ Commands:    %u                                \n", state->cmdCount);
    printf("│ Frames:      %u                                \n", state->frameCount);
    printf("└─────────────────────────────────────────────────┘\n");

    // Vertex arrays
    int activeVA = 0;
    for (int i = 0; i < 16; i++) {
        if (state->vertexArrays[i].enabled) activeVA++;
    }
    printf("  Active vertex arrays: %d/16\n", activeVA);

    // Textures
    int activeTex = 0;
    for (int i = 0; i < 16; i++) {
        if (state->textures[i].enabled) activeTex++;
    }
    printf("  Active texture units: %d/16\n", activeTex);
}

} // namespace rsx
