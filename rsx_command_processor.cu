// rsx_command_processor.cu — RSX Reality Synthesizer Command Processor
//
// Processes NV47 GPU commands from the FIFO command buffer.
// Phase 4a: Parse + state tracking (no actual rendering yet).
// Phase 4b will add Vulkan rasterization backend.
//
// Part of Project Megakernel — CUDA-based PS3 emulation on Tesla V100.

#include "rsx_defs.h"

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

    // Default surface: 1280×720 A8R8G8B8
    state->surfaceFormat  = SURFACE_A8R8G8B8;
    state->surfaceWidth   = 1280;
    state->surfaceHeight  = 720;
    state->surfacePitchA  = 1280 * 4;
    state->surfacePitchB  = 1280 * 4;
    state->surfaceColorTarget = 1;  // color A only

    // Default viewport / scissor
    state->viewportX = 0;  state->viewportY = 0;
    state->viewportW = 1280; state->viewportH = 720;
    state->scissorX  = 0;  state->scissorY  = 0;
    state->scissorW  = 4096; state->scissorH = 4096;

    // Depth defaults
    state->depthTestEnable = false;
    state->depthFunc       = CMP_LESS;
    state->depthFormat     = DEPTH_Z24S8;

    // Blend / cull
    state->blendEnable    = false;
    state->cullFaceEnable = false;
    state->cullFace       = 0x0405; // GL_BACK

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
        state->surfaceFormat = data & 0x1F;
        return;
    case NV4097_SET_SURFACE_CLIP_HORIZONTAL:
        state->surfaceWidth  = data & 0xFFFF;
        state->viewportX     = 0;
        state->viewportW     = state->surfaceWidth;
        return;
    case NV4097_SET_SURFACE_CLIP_VERTICAL:
        state->surfaceHeight = data & 0xFFFF;
        state->viewportY     = 0;
        state->viewportH     = state->surfaceHeight;
        return;
    case NV4097_SET_SURFACE_PITCH_A:
        state->surfacePitchA = data;
        return;
    case NV4097_SET_SURFACE_COLOR_AOFFSET:
        state->surfaceOffsetA = data;
        return;
    case NV4097_SET_SURFACE_COLOR_BOFFSET:
        state->surfaceOffsetB = data;
        return;
    case NV4097_SET_SURFACE_COLOR_TARGET:
        state->surfaceColorTarget = data;
        return;

    // ── Viewport / Scissor ─────────────────────────────────────
    case NV4097_SET_VIEWPORT_HORIZONTAL:
        state->viewportX = (uint16_t)(data & 0xFFFF);
        state->viewportW = (uint16_t)(data >> 16);
        return;
    case NV4097_SET_VIEWPORT_VERTICAL:
        state->viewportY = (uint16_t)(data & 0xFFFF);
        state->viewportH = (uint16_t)(data >> 16);
        return;
    case NV4097_SET_SCISSOR_HORIZONTAL:
        state->scissorX = (uint16_t)(data & 0xFFFF);
        state->scissorW = (uint16_t)(data >> 16);
        return;
    case NV4097_SET_SCISSOR_VERTICAL:
        state->scissorY = (uint16_t)(data & 0xFFFF);
        state->scissorH = (uint16_t)(data >> 16);
        return;

    // ── Depth / Stencil ────────────────────────────────────────
    case NV4097_SET_DEPTH_TEST_ENABLE:
        state->depthTestEnable = (data != 0);
        return;
    case NV4097_SET_DEPTH_FUNC:
        state->depthFunc = data;
        return;

    // ── Blend ──────────────────────────────────────────────────
    case NV4097_SET_BLEND_ENABLE:
        state->blendEnable = (data != 0);
        return;

    // ── Cull face ──────────────────────────────────────────────
    case NV4097_SET_CULL_FACE_ENABLE:
        state->cullFaceEnable = (data != 0);
        return;
    case NV4097_SET_CULL_FACE:
        state->cullFace = data;
        return;

    // ── Clear ──────────────────────────────────────────────────
    case NV4097_SET_COLOR_CLEAR_VALUE:
        state->colorClearValue = data;
        return;
    case NV4097_CLEAR_SURFACE:
        rsx_clear_surface(state, vram, data);
        return;

    // ── Shader programs ────────────────────────────────────────
    case NV4097_SET_TRANSFORM_PROGRAM_START:
        state->vpStart = data;
        state->vpLoadOffset = data * 4;  // each instruction is 4 words
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
        return;
    case NV4097_DRAW_ARRAYS: {
        uint32_t count = ((data >> 24) & 0xFF) + 1;
        state->drawCallCount++;
        state->triangleCount += estimateTriangles(state->currentPrim, count);
        return;
    }
    case NV4097_DRAW_INDEX_ARRAY: {
        uint32_t count = ((data >> 24) & 0xFF) + 1;
        state->drawCallCount++;
        state->triangleCount += estimateTriangles(state->currentPrim, count);
        return;
    }

    // ── Flip (present) ─────────────────────────────────────────
    case NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP:
        state->surfaceOffsetA = data;
        state->frameCount++;
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
        }
        return;
    }

    // ── Transform constants (256 vec4s) ────────────────────────
    if (method >= NV4097_SET_TRANSFORM_CONSTANT &&
        method <  NV4097_SET_TRANSFORM_CONSTANT + 256 * 16) {
        uint32_t byteOff  = method - NV4097_SET_TRANSFORM_CONSTANT;
        uint32_t floatIdx = byteOff / 4;
        if (floatIdx < 256 * 4) {
            uint32_t vec  = floatIdx / 4;
            uint32_t comp = floatIdx % 4;
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
