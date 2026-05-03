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
void rsx_emitter_onDrawInline (void* emitter, const rsx::RSXState* s, const uint32_t* data, uint32_t words) __attribute__((weak));
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
    state->surfaceOffsetA = 0;            // default safe; games set real offset via FIFO
    state->surfaceColorTarget = 1;  // color A only
    state->vramSize = (uint32_t)VRAM_SIZE;  // default 256MB

    // Default viewport / scissor
    state->viewportX = 0;  state->viewportY = 0;
    state->viewportW = 1280; state->viewportH = 720;
    state->scissorX  = 0;  state->scissorY  = 0;
    state->scissorW  = 4096; state->scissorH = 4096;

    // Viewport offset/scale defaults (identity for 1280×720)
    state->vpOffset[0] = 640.0f; state->vpOffset[1] = 360.0f;
    state->vpOffset[2] = 0.5f;   state->vpOffset[3] = 0.0f;
    state->vpScale[0]  = 640.0f; state->vpScale[1]  = -360.0f;
    state->vpScale[2]  = 0.5f;   state->vpScale[3]  = 0.0f;
    state->vpOffsetScaleSet = false;

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

    // Polygon offset defaults
    state->polyOffsetFillEnable = false;
    state->polyOffsetFactor     = 0.0f;
    state->polyOffsetBias       = 0.0f;

    // Logic op defaults
    state->logicOpEnable = false;
    state->logicOp       = 0x1503; // GL_COPY (passthrough)

    // Dither default
    state->ditherEnable  = true;  // RSX default is enabled

    // Polygon mode defaults (FILL)
    state->frontPolygonMode = 0x1B02;  // GL_FILL
    state->backPolygonMode  = 0x1B02;

    // Two-sided stencil defaults
    state->twoSidedStencilEnable = false;
    // Two-sided color defaults
    state->twoSidedColorEnable = false;
    state->backStencilFunc       = 0x0207; // ALWAYS
    state->backStencilFuncRef    = 0;
    state->backStencilFuncMask   = 0xFF;
    state->backStencilOpFail     = 0x1E00; // KEEP
    state->backStencilOpZFail    = 0x1E00;
    state->backStencilOpZPass    = 0x1E00;
    state->backStencilWriteMask  = 0xFF;

    // Fog defaults
    state->fogMode   = 0x2601; // LINEAR
    state->fogParam0 = 0.0f;
    state->fogParam1 = 1.0f;

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

    if (clearMask & CLEAR_RGBA) {
        uint32_t offset = state->surfaceOffsetA;
        uint32_t pitch  = state->surfacePitchA;
        uint32_t w      = state->surfaceWidth;
        uint32_t h      = state->surfaceHeight;
        uint32_t color  = state->colorClearValue;
        uint32_t maxBytes = state->vramSize ? state->vramSize : VRAM_SIZE;

        // Bytes per pixel (assume 4 for A8R8G8B8 / X8R8G8B8)
        uint32_t bpp = 4;
        if (state->surfaceFormat == SURFACE_R5G6B5) bpp = 2;

        for (uint32_t y = 0; y < h; y++) {
            uint32_t rowOff = offset + y * pitch;
            if (rowOff + w * bpp > maxBytes) break;

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

    // Depth / stencil clear
    if ((clearMask & (CLEAR_Z | CLEAR_STENCIL)) && state->zetaOffset) {
        uint32_t w = state->surfaceWidth;
        uint32_t h = state->surfaceHeight;
        uint32_t pitch = state->surfacePitchZ ? state->surfacePitchZ : (w * 4);
        uint32_t maxBytes = state->vramSize ? state->vramSize : VRAM_SIZE;
        uint32_t clearVal = state->zstencilClearValue;

        if (state->depthFormat == DEPTH_Z24S8) {
            // 32-bit: stencil[31:24] | depth[23:0]
            for (uint32_t y = 0; y < h; y++) {
                uint32_t rowOff = state->zetaOffset + y * pitch;
                if (rowOff + w * 4 > maxBytes) break;
                uint32_t* row = (uint32_t*)(vram + rowOff);
                for (uint32_t x = 0; x < w; x++) {
                    uint32_t cur = row[x];
                    if (clearMask & CLEAR_Z) cur = (cur & 0xFF000000) | (clearVal & 0x00FFFFFF);
                    if (clearMask & CLEAR_STENCIL) cur = (cur & 0x00FFFFFF) | (clearVal & 0xFF000000);
                    row[x] = cur;
                }
            }
        } else if (state->depthFormat == DEPTH_Z16) {
            uint32_t zpitch = state->surfacePitchZ ? state->surfacePitchZ : (w * 2);
            uint16_t z16 = (uint16_t)(clearVal & 0xFFFF);
            for (uint32_t y = 0; y < h; y++) {
                uint32_t rowOff = state->zetaOffset + y * zpitch;
                if (rowOff + w * 2 > maxBytes) break;
                uint16_t* row = (uint16_t*)(vram + rowOff);
                if (clearMask & CLEAR_Z) {
                    for (uint32_t x = 0; x < w; x++) row[x] = z16;
                }
            }
        }
    }
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
        case PRIM_POLYGON:        return (vertexCount >= 3) ? vertexCount - 2 : 0;
        case PRIM_LINES:          return 0;
        case PRIM_LINE_LOOP:      return 0;
        case PRIM_LINE_STRIP:     return 0;
        case PRIM_POINTS:         return 0;
        default:                  return 0;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Method Dispatch — handle a single NV4097 method write
// ═══════════════════════════════════════════════════════════════════

static void dispatchMethod(RSXState* state, uint8_t* vram,
                           uint32_t method, uint32_t data,
                           uint32_t subchannel = 0) {

    // ── NV3062/NV3089 2D engine (subchannels 4-6) ────────────────
    // Must be checked BEFORE the NV4097 switch because method addresses
    // 0x300-0x31C overlap with NV4097 dither/alpha/blend registers.
    if (subchannel == 4) {
        // NV3062 context surfaces
        if (method == NV3062_SET_COLOR_FORMAT) {
            state->blit2DSurface.colorFormat = data;
        } else if (method == NV3062_SET_PITCH) {
            state->blit2DSurface.pitch = data;
        } else if (method == NV3062_SET_OFFSET_DESTIN) {
            state->blit2DSurface.dstOffset = data;
        }
        return;
    }
    if (subchannel == 5 || subchannel == 6) {
        // NV3089 scaled image from memory
        if (method == NV3089_SET_CONTEXT_SURFACE) {
            // links to NV3062 — single context, ignore
        } else if (method == NV3089_SET_OPERATION) {
            state->scaledImage.operation = data;
        } else if (method == NV3089_SET_COLOR_FORMAT) {
            state->scaledImage.colorFormat = data;
        } else if (method == NV3089_CLIP_POINT) {
            state->scaledImage.clipX = data & 0xFFFF;
            state->scaledImage.clipY = (data >> 16) & 0xFFFF;
        } else if (method == NV3089_CLIP_SIZE) {
            state->scaledImage.clipW = data & 0xFFFF;
            state->scaledImage.clipH = (data >> 16) & 0xFFFF;
        } else if (method == NV3089_IMAGE_OUT_POINT) {
            state->scaledImage.outX = data & 0xFFFF;
            state->scaledImage.outY = (data >> 16) & 0xFFFF;
        } else if (method == NV3089_IMAGE_OUT_SIZE) {
            state->scaledImage.outW = data & 0xFFFF;
            state->scaledImage.outH = (data >> 16) & 0xFFFF;
        } else if (method == NV3089_DS_DX) {
            state->scaledImage.dsDx = data;
        } else if (method == NV3089_DT_DY) {
            state->scaledImage.dtDy = data;
        } else if (method == NV3089_IMAGE_IN_SIZE) {
            state->scaledImage.inW = (data >> 16) & 0xFFFF;
            state->scaledImage.inH = data & 0xFFFF;
        } else if (method == NV3089_IMAGE_IN_FORMAT) {
            state->scaledImage.inPitch = (data >> 16) & 0xFFFF;
        } else if (method == NV3089_IMAGE_IN_OFFSET) {
            state->scaledImage.inOffset = data;
        } else if (method == NV3089_IMAGE_IN) {
            // Trigger: execute the 2D scaled blit
            auto& si  = state->scaledImage;
            auto& dst = state->blit2DSurface;
            if (vram && si.inW > 0 && si.inH > 0 && si.outW > 0 && si.outH > 0) {
                uint32_t vramSz = state->vramSize;
                uint32_t srcPitch = si.inPitch ? si.inPitch : (si.inW * 4);
                uint32_t dstPitch = dst.pitch ? dst.pitch : (si.outW * 4);
                uint32_t bpp = 4;  // A8R8G8B8

                if (si.inW == si.outW && si.inH == si.outH) {
                    // 1:1 copy — fast path
                    uint32_t copyW = (si.clipW && si.clipW < si.inW) ? si.clipW : si.inW;
                    uint32_t copyH = (si.clipH && si.clipH < si.inH) ? si.clipH : si.inH;
                    uint32_t rowBytes = copyW * bpp;
                    for (uint32_t y = 0; y < copyH; ++y) {
                        uint32_t srcOff = si.inOffset + y * srcPitch;
                        uint32_t dstOff = dst.dstOffset + (si.outY + y) * dstPitch + si.outX * bpp;
                        if (srcOff + rowBytes <= vramSz && dstOff + rowBytes <= vramSz) {
                            std::memmove(vram + dstOff, vram + srcOff, rowBytes);
                        }
                    }
                } else {
                    // Scaled blit — nearest-neighbor via dsDx/dtDy (20.12 fixed-point)
                    uint32_t dsDx = si.dsDx ? si.dsDx : (1 << 20);
                    uint32_t dtDy = si.dtDy ? si.dtDy : (1 << 20);
                    uint32_t outW = si.outW;
                    uint32_t outH = si.outH;
                    if (si.clipW && si.clipW < outW) outW = si.clipW;
                    if (si.clipH && si.clipH < outH) outH = si.clipH;
                    for (uint32_t dy = 0; dy < outH; ++dy) {
                        uint32_t sy = (dy * dtDy) >> 20;
                        if (sy >= si.inH) sy = si.inH - 1;
                        for (uint32_t dx = 0; dx < outW; ++dx) {
                            uint32_t sx = (dx * dsDx) >> 20;
                            if (sx >= si.inW) sx = si.inW - 1;
                            uint32_t srcOff = si.inOffset + sy * srcPitch + sx * bpp;
                            uint32_t dstOff = dst.dstOffset + (si.outY + dy) * dstPitch + (si.outX + dx) * bpp;
                            if (srcOff + bpp <= vramSz && dstOff + bpp <= vramSz) {
                                std::memcpy(vram + dstOff, vram + srcOff, bpp);
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // ── NV406E subchannel semaphore (subchannels 1-2) ───────────
    // Used by cellGcmSetWriteBackEndLabel and semaphore acquire/release.
    // These are the label-based sync that SPUs poll via sys_rsx_context_attribute.
    if (subchannel == 1 || subchannel == 2) {
        if (method == NV406E_SEMAPHORE_OFFSET) {
            state->labelOffset = data;
        } else if (method == NV406E_SEMAPHORE_ACQUIRE) {
            // Wait until VRAM[labelOffset] == data. In our single-threaded
            // emulator the value is already written, so this is a no-op.
            // A real implementation would spin-wait here.
            state->labelValue = data;
        } else if (method == NV406E_SEMAPHORE_RELEASE) {
            // Write data to VRAM at labelOffset (big-endian, 4 bytes).
            if (vram && state->labelOffset + 4 <= state->vramSize) {
                uint32_t off = state->labelOffset;
                vram[off + 0] = (uint8_t)(data >> 24);
                vram[off + 1] = (uint8_t)(data >> 16);
                vram[off + 2] = (uint8_t)(data >> 8);
                vram[off + 3] = (uint8_t)(data);
            }
        } else if (method == NV406E_SET_REFERENCE) {
            state->ref = data;
        }
        return;
    }

    // ── NV4097 3D engine (subchannel 0) + NV0039 DMA (subchannel 3) ──
    // ── Surface setup ──────────────────────────────────────────
    switch (method) {
    case NV4097_SET_SURFACE_FORMAT:
        if (data) {
            state->surfaceFormat = data & 0x1F;
            state->surfaceAntialias = (data >> 12) & 0xF;  // AA mode: 0=none, 4=2x, 12=4x
        }
        RSX_EMIT(onSurfaceSetup, state);
        return;
    case NV4097_SET_SURFACE_CLIP_HORIZONTAL: {
        // Packed: (width << 16) | origin. Unpacked (test compat): plain width.
        uint16_t width  = (data >> 16) ? ((data >> 16) & 0xFFFF) : (data & 0xFFFF);
        uint16_t origin = (data >> 16) ? (data & 0xFFFF) : 0;
        if (width) {
            state->surfaceWidth  = width;
            state->viewportX     = origin;
            state->viewportW     = width;
        }
        return;
    }
    case NV4097_SET_SURFACE_CLIP_VERTICAL: {
        uint16_t height = (data >> 16) ? ((data >> 16) & 0xFFFF) : (data & 0xFFFF);
        uint16_t origin = (data >> 16) ? (data & 0xFFFF) : 0;
        if (height) {
            state->surfaceHeight = height;
            state->viewportY     = origin;
            state->viewportH     = height;
        }
        return;
    }
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
        state->surfaceOffsetA = data;
        return;
    case NV4097_SET_SURFACE_COLOR_BOFFSET:
        state->surfaceOffsetB = data;
        return;
    case NV4097_SET_SURFACE_COLOR_COFFSET:
        state->surfaceOffsetC = data;
        return;
    case NV4097_SET_SURFACE_COLOR_DOFFSET:
        state->surfaceOffsetD = data;
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

    // Viewport offset/scale — 4 float registers each (consecutive methods)
    case NV4097_SET_VIEWPORT_OFFSET:
    case NV4097_SET_VIEWPORT_OFFSET + 4:
    case NV4097_SET_VIEWPORT_OFFSET + 8:
    case NV4097_SET_VIEWPORT_OFFSET + 12: {
        uint32_t idx = (method - NV4097_SET_VIEWPORT_OFFSET) / 4;
        float f; memcpy(&f, &data, 4);
        state->vpOffset[idx] = f;
        state->vpOffsetScaleSet = true;
        return;
    }
    case NV4097_SET_VIEWPORT_SCALE:
    case NV4097_SET_VIEWPORT_SCALE + 4:
    case NV4097_SET_VIEWPORT_SCALE + 8:
    case NV4097_SET_VIEWPORT_SCALE + 12: {
        uint32_t idx = (method - NV4097_SET_VIEWPORT_SCALE) / 4;
        float f; memcpy(&f, &data, 4);
        state->vpScale[idx] = f;
        state->vpOffsetScaleSet = true;
        return;
    }

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

    // ── Polygon offset (depth bias) ──────────────────────────────
    case NV4097_SET_POLY_OFFSET_FILL_ENABLE:
        state->polyOffsetFillEnable = (data != 0);
        return;
    case NV4097_SET_POLYGON_OFFSET_SCALE_FACTOR: {
        // Float encoded as uint32_t (reinterpret bits)
        float f; memcpy(&f, &data, 4);
        state->polyOffsetFactor = f;
        return;
    }
    case NV4097_SET_POLYGON_OFFSET_BIAS: {
        float f; memcpy(&f, &data, 4);
        state->polyOffsetBias = f;
        return;
    }

    // ── Logic op ─────────────────────────────────────────────────
    case NV4097_SET_LOGIC_OP_ENABLE:
        state->logicOpEnable = (data != 0);
        return;
    case NV4097_SET_LOGIC_OP:
        state->logicOp = data;
        return;

    // ── Dither ───────────────────────────────────────────────────
    case NV4097_SET_DITHER_ENABLE:
        state->ditherEnable = (data != 0);
        return;

    // ── Two-sided stencil ────────────────────────────────────────
    case NV4097_SET_TWO_SIDED_STENCIL_TEST_ENABLE:
        state->twoSidedStencilEnable = (data != 0);
        return;
    // ── Two-sided color (back-face lighting) ─────────────────────
    case NV4097_SET_TWO_SIDE_LIGHT_EN:
        state->twoSidedColorEnable = (data != 0);
        return;
    case NV4097_SET_BACK_STENCIL_FUNC:
        state->backStencilFunc = data;
        return;
    case NV4097_SET_BACK_STENCIL_FUNC_REF:
        state->backStencilFuncRef = data;
        return;
    case NV4097_SET_BACK_STENCIL_FUNC_MASK:
        state->backStencilFuncMask = data;
        return;
    case NV4097_SET_BACK_STENCIL_OP_FAIL:
        state->backStencilOpFail = data;
        return;
    case NV4097_SET_BACK_STENCIL_OP_ZFAIL:
        state->backStencilOpZFail = data;
        return;
    case NV4097_SET_BACK_STENCIL_OP_ZPASS:
        state->backStencilOpZPass = data;
        return;
    case NV4097_SET_BACK_STENCIL_MASK:
        state->backStencilWriteMask = data;
        return;

    // ── Fog ──────────────────────────────────────────────────────
    case NV4097_SET_FOG_MODE:
        state->fogMode = data;
        return;
    case NV4097_SET_FOG_PARAMS: {
        float f; memcpy(&f, &data, 4);
        state->fogParam0 = f;
        return;
    }
    case NV4097_SET_FOG_PARAMS + 4: {
        float f; memcpy(&f, &data, 4);
        state->fogParam1 = f;
        return;
    }

    // ── Depth bounds ────────────────────────────────────────────
    case NV4097_SET_DEPTH_BOUNDS_TEST_ENABLE:
        state->depthBoundsTestEnable = (data != 0);
        return;
    case NV4097_SET_DEPTH_BOUNDS_MIN: {
        float f; memcpy(&f, &data, 4);
        state->depthBoundsMin = f;
        return;
    }
    case NV4097_SET_DEPTH_BOUNDS_MAX: {
        float f; memcpy(&f, &data, 4);
        state->depthBoundsMax = f;
        return;
    }

    // ── Primitive restart ───────────────────────────────────────
    case NV4097_SET_RESTART_INDEX_ENABLE:
        state->restartIndexEnable = (data != 0);
        return;
    case NV4097_SET_RESTART_INDEX:
        state->restartIndex = data;
        return;

    // ── User clip planes ────────────────────────────────────────
    case NV4097_SET_USER_CLIP_PLANE_CONTROL:
        state->userClipPlaneControl = data;
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
        state->vpDirty = true;
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
    case NV4097_SET_SHADER_WINDOW:
        state->shaderWindow = data;
        return;

    case NV4097_SET_SHADER_CONTROL:
        state->sRGBWrite = (data & 2) != 0;  // bit 1 = sRGB write enable
        return;

    case NV4097_SET_POINT_SIZE:
        {
            float sz;
            memcpy(&sz, &data, 4);
            state->pointSize = sz;
        }
        return;
    case NV4097_SET_LINE_WIDTH:
        state->lineWidth = data;
        return;
    case NV4097_SET_POINT_SPRITE_CONTROL:
        state->pointSpriteCtrl = data;
        return;
    case NV4097_SET_COLOR_MASK_MRT:
        state->colorMaskMrt = data;
        return;
    case NV4097_SET_VERTEX_ATTRIB_OUTPUT_MASK:
        state->vpAttribOutputMask = data;
        return;
    case NV4097_SET_FREQUENCY_DIVIDER_OPERATION:
        state->freqDividerOp = data;
        return;
    case NV4097_SET_BEGIN_END_INSTANCE_CNT:
        state->instanceCount = data;
        return;

    // ── Draw ───────────────────────────────────────────────────
    case NV4097_SET_BEGIN_END:
        if (data != 0) {
            state->currentPrim = (PrimitiveType)data;
            state->inBeginEnd  = true;
            state->inlineVertexData.clear();
            state->inlineVertexData.reserve(64);
        } else {
            // END — finalize draw batch; flush inline vertex data if any
            state->inBeginEnd = false;
            if (!state->inlineVertexData.empty()) {
                uint32_t words = (uint32_t)state->inlineVertexData.size();
                state->drawCallCount++;
                RSX_EMIT(onDrawInline, state,
                         state->inlineVertexData.data(), words);
                state->inlineVertexData.clear();
            }
        }
        RSX_EMIT(onBeginEnd, state, data);
        return;
    case NV4097_DRAW_ARRAYS: {
        uint32_t first = data & 0xFFFFFF;
        uint32_t count = ((data >> 24) & 0xFF) + 1;
        state->drawCallCount++;
        uint32_t triEst = estimateTriangles(state->currentPrim, count);
        state->triangleCount += triEst;
        if (state->zcullEnable) state->zcullPixelCount += triEst * 100;
        RSX_EMIT(onDrawArrays, state, first, count);
        return;
    }
    case NV4097_DRAW_INDEX_ARRAY: {
        uint32_t first = data & 0xFFFFFF;
        uint32_t count = ((data >> 24) & 0xFF) + 1;
        state->drawCallCount++;
        uint32_t triEst = estimateTriangles(state->currentPrim, count);
        state->triangleCount += triEst;
        if (state->zcullEnable) state->zcullPixelCount += triEst * 100;
        RSX_EMIT(onDrawIndexed, state, first, count, 0u);
        return;
    }
    case NV4097_DRAW_INLINE_ARRAY:
        // Inline vertex data pushed one word at a time. Games pack
        // position/color per-word for small immediate geometry (UI quads).
        // We accumulate into the inline buffer; flushed at END.
        state->inlineVertexData.push_back(data);
        return;

    // ── Flip (present) ─────────────────────────────────────────
    case NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP:
        state->surfaceOffsetA = data;
        state->frameCount++;
        RSX_EMIT(onFlip, state, data);
        return;

    // ── Reference register ─────────────────────────────────────
    // ── Semaphore / label (SPU↔RSX sync) ────────────────────────
    case NV4097_SET_SEMAPHORE_OFFSET:
        state->semaphoreOffset = data;
        return;
    case NV4097_BACK_END_WRITE_SEMAPHORE_RELEASE:
        // Write value to VRAM at semaphoreOffset (big-endian).
        // SPUs poll this location to detect GPU completion.
        if (vram && state->semaphoreOffset + 4 <= state->vramSize) {
            uint32_t off = state->semaphoreOffset;
            // RSX writes the value in GPU byte order (little-endian on real HW,
            // but PS3 convention is big-endian for cross-unit sync).
            vram[off + 0] = (uint8_t)(data >> 24);
            vram[off + 1] = (uint8_t)(data >> 16);
            vram[off + 2] = (uint8_t)(data >> 8);
            vram[off + 3] = (uint8_t)(data);
        }
        return;
    case NV4097_TEXTURE_READ_SEMAPHORE_RELEASE:
        // Same as BACK_END but signals texture pipe completion.
        if (vram && state->semaphoreOffset + 4 <= state->vramSize) {
            uint32_t off = state->semaphoreOffset;
            vram[off + 0] = (uint8_t)(data >> 24);
            vram[off + 1] = (uint8_t)(data >> 16);
            vram[off + 2] = (uint8_t)(data >> 8);
            vram[off + 3] = (uint8_t)(data);
        }
        return;
    case NV4097_SET_NOTIFY:
    case NV4097_NOTIFY:
        // Notify mechanism — acknowledge without action for now.
        return;

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

    // ── Occlusion query / ZCULL ───────────────────────────────
    case NV4097_SET_ZPASS_PIXEL_COUNT_ENABLE:
        state->zcullEnable = (data != 0);
        return;
    case NV4097_CLEAR_REPORT_VALUE:
        state->zcullPixelCount = 0;
        return;
    case NV4097_SET_RENDER_ENABLE: {
        // Bits [23:0] = VRAM offset of report, bits [25:24] = mode
        // mode: 0=unconditional, 1=conditional wait, 2=conditional render
        uint32_t mode = (data >> 24) & 3;
        uint32_t offset = data & 0x00FFFFFF;
        state->conditionalRenderEnable = (mode == 2);
        state->conditionalRenderOffset = offset;
        return;
    }
    case NV4097_GET_REPORT: {
        // Write the report to VRAM at data offset.
        // Report structure: { u64 timestamp; u32 value; u32 padding }
        // We report the accumulated pixel count (always > 0 to avoid
        // games skipping draw calls via conditional render).
        uint32_t rptOff = data;
        if (vram && rptOff + 16 <= state->vramSize) {
            uint32_t count = state->zcullPixelCount;
            if (count == 0) count = 1;  // never report 0 (would skip draws)
            memset(vram + rptOff, 0, 16);
            // Value at offset 8 (big-endian u32)
            vram[rptOff +  8] = (uint8_t)(count >> 24);
            vram[rptOff +  9] = (uint8_t)(count >> 16);
            vram[rptOff + 10] = (uint8_t)(count >> 8);
            vram[rptOff + 11] = (uint8_t)(count);
        }
        return;
    }

    // Texture/vertex cache invalidation (no-ops for us, caches are coherent)
    case NV4097_INVALIDATE_L2:
    case NV4097_INVALIDATE_VERTEX_FILE:
    case NV4097_INVALIDATE_VERTEX_CACHE_FILE:
        return;

    // Surface zeta (depth buffer) offset — store for depth buffer management
    case NV4097_SET_SURFACE_ZETA_OFFSET:
        state->zetaOffset = data;
        return;

    // Anti-aliasing / smoothing control
    case NV4097_SET_ANTI_ALIASING_CONTROL:
        state->antiAliasingControl = data;
        return;
    case NV4097_SET_LINE_SMOOTH_ENABLE:
        state->lineSmoothEnable = (data != 0);
        return;
    case NV4097_SET_POLY_SMOOTH_ENABLE:
        state->polySmoothEnable = (data != 0);
        return;

    // ZCULL stats (performance hint, no-op in our rasterizer)
    case NV4097_SET_ZCULL_STATS_ENABLE:
        state->zcullStatsEnable = data;
        return;
    case NV4097_SET_SCULL_CONTROL:
        state->scullControl = data;
        return;

    // Shader packer / flat shade / vertex attrib input mask
    case NV4097_SET_SHADER_PACKER:
        state->shaderPacker = data;
        return;
    case NV4097_SET_FLAT_SHADE_OP:
        state->flatShadeOp = data;
        return;
    case NV4097_SET_VERTEX_ATTRIB_INPUT_MASK:
        state->vertexAttribInputMask = data;
        return;

    // Misc pipeline control (no-ops — perf hints / stall points)
    case NV4097_SET_REDUCE_DST_COLOR:
        state->reduceDstColor = data;
        return;
    case NV4097_SET_TRANSFORM_TIMEOUT:
    case NV4097_SET_WAIT_FOR_IDLE:
        // Pipeline flush — all pending draws are complete before proceeding.
        // In our single-threaded command processor, this is implicitly
        // satisfied since draws execute synchronously. The async FIFO
        // path handles this via cudaStreamSynchronize in the bridge.
        return;

    // DMA context bindings (select VRAM vs main memory for buffers)
    case NV4097_SET_CONTEXT_DMA_A:
        state->contextDmaA = data;
        return;
    case NV4097_SET_CONTEXT_DMA_B:
        state->contextDmaB = data;
        return;
    case NV4097_SET_CONTEXT_DMA_COLOR_B:
        state->contextDmaColorB = data;
        return;
    case NV4097_SET_CONTEXT_DMA_STATE:
        state->contextDmaState = data;
        return;
    case NV4097_SET_CONTEXT_DMA_ZETA:
        state->contextDmaZeta = data;
        return;

    // Polygon rasterization mode (wireframe support)
    case NV4097_SET_FRONT_POLYGON_MODE:
        state->frontPolygonMode = data;
        return;
    case NV4097_SET_BACK_POLYGON_MODE:
        state->backPolygonMode = data;
        return;

    // Depth buffer pitch
    case NV4097_SET_SURFACE_PITCH_Z:
        state->surfacePitchZ = data;
        return;

    // Depth+stencil clear value
    case NV4097_SET_ZSTENCIL_CLEAR_VALUE:
        state->zstencilClearValue = data;
        return;

    default:
        state->unknownMethodCount++;
        break;
    }

    // ── Vertex array format (16 slots at stride 4) ────────────
    if (method >= NV4097_SET_VERTEX_DATA_ARRAY_FORMAT &&
        method <  NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + 16 * 4) {
        uint32_t slot = (method - NV4097_SET_VERTEX_DATA_ARRAY_FORMAT) / 4;
        state->vertexArrays[slot].format  = data;
        state->vertexArrays[slot].enabled = (data != 0);
        if (data != 0) state->activeVAMask |=  (1u << slot);
        else           state->activeVAMask &= ~(1u << slot);
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
                state->textures[unit].dirty   = true;
                break;
            case 0x04: // TEXTURE_FORMAT
                state->textures[unit].format = data;
                state->textures[unit].dimension = (uint8_t)((data >> 4) & 0xF);
                state->textures[unit].dirty = true;
                break;
            case 0x0C: // TEXTURE_CONTROL0
                state->textures[unit].control0 = data;
                break;
            case 0x08: // TEXTURE_ADDRESS (wrap modes)
                state->textures[unit].address = data;
                break;
            case 0x10: // TEXTURE_BORDER_COLOR
                state->textures[unit].borderColor = data;
                break;
            case 0x14: // TEXTURE_FILTER (min/mag)
                state->textures[unit].filter = data;
                break;
            case 0x18: // TEXTURE_IMAGE_RECT  (width<<16 | height)
                state->textures[unit].width  = (data >> 16) & 0xFFFF;
                state->textures[unit].height =  data        & 0xFFFF;
                state->textures[unit].dirty = true;
                break;
            default:
                break;
            }
        }
        return;
    }

    // ── Texture control3 (depth/pitch per texture unit) ────────
    if (method >= NV4097_SET_TEXTURE_CONTROL3 &&
        method <  NV4097_SET_TEXTURE_CONTROL3 + 16 * 4) {
        uint32_t unit = (method - NV4097_SET_TEXTURE_CONTROL3) / 4;
        if (unit < 16) {
            state->textures[unit].depth = (data >> 20) & 0xFFF;
            if (state->textures[unit].depth == 0)
                state->textures[unit].depth = 1;
            state->textures[unit].dirty = true;
        }
        return;
    }

    // ── Texture control1 (remap/swizzle per unit) ────────────
    if (method >= NV4097_SET_TEXTURE_CONTROL1 &&
        method <  NV4097_SET_TEXTURE_CONTROL1 + 16 * 4) {
        uint32_t unit = (method - NV4097_SET_TEXTURE_CONTROL1) / 4;
        if (unit < 16) {
            state->textures[unit].control1 = data;
        }
        return;
    }

    // ── Texture control2 (aniso level per unit, 16 units) ─────
    if (method >= NV4097_SET_TEXTURE_CONTROL2 &&
        method <  NV4097_SET_TEXTURE_CONTROL2 + 16 * 4) {
        // bits [3:0] = max aniso, bits [12:4] = ISO level
        // Store for future anisotropic filtering; currently no-op
        return;
    }

    // ── Inline vertex data (SET_VERTEX_DATA4F_M) ──────────────
    // 16 attributes × 4 floats each, stride 0x10 per attribute
    if (method >= NV4097_SET_VERTEX_DATA4F_M &&
        method <  NV4097_SET_VERTEX_DATA4F_M + 16 * 0x10) {
        uint32_t byteOff = method - NV4097_SET_VERTEX_DATA4F_M;
        uint32_t attr = byteOff / 0x10;
        uint32_t comp = (byteOff % 0x10) / 4;
        if (attr < 16 && comp < 4) {
            float fval;
            memcpy(&fval, &data, 4);
            state->vertexData4f[attr][comp] = fval;
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
            state->vpDirty = true;
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

    // ── NV0039 DMA buffer copy (subchannel 3 or default) ──────────
    // Games use this for CPU↔VRAM blits (loading screens, XMB, SW render).
    // We accumulate parameters, then execute on BUFFER_NOTIFY.
    if (method == NV0039_OFFSET_IN) {
        state->dmaTransfer.offsetIn = data;
        return;
    }
    if (method == NV0039_OFFSET_OUT) {
        state->dmaTransfer.offsetOut = data;
        return;
    }
    if (method == NV0039_PITCH_IN) {
        state->dmaTransfer.pitchIn = data;
        return;
    }
    if (method == NV0039_PITCH_OUT) {
        state->dmaTransfer.pitchOut = data;
        return;
    }
    if (method == NV0039_LINE_LENGTH_IN) {
        state->dmaTransfer.lineLength = data;
        return;
    }
    if (method == NV0039_LINE_COUNT) {
        state->dmaTransfer.lineCount = data;
        return;
    }
    if (method == NV0039_FORMAT) {
        state->dmaTransfer.format = data;
        return;
    }
    if (method == NV0039_SET_CONTEXT_DMA_BUFFER_IN) {
        state->dmaTransfer.ctxIn = data;
        return;
    }
    if (method == NV0039_SET_CONTEXT_DMA_BUFFER_OUT) {
        state->dmaTransfer.ctxOut = data;
        return;
    }
    if (method == NV0039_BUFFER_NOTIFY) {
        // Execute the DMA transfer: copy lineCount rows of lineLength bytes
        // from offsetIn to offsetOut within VRAM. Supports byte-swap via FORMAT.
        auto& d = state->dmaTransfer;
        if (vram && d.lineLength > 0 && d.lineCount > 0) {
            uint32_t vramSz = state->vramSize;
            uint32_t inFmt  = d.format & 0xFF;         // 1=byte, 2=LE16, 4=LE32
            uint32_t outFmt = (d.format >> 8) & 0xFF;
            bool needSwap = (inFmt != outFmt && inFmt > 1 && outFmt > 1);

            for (uint32_t row = 0; row < d.lineCount; ++row) {
                uint32_t srcOff = d.offsetIn  + row * d.pitchIn;
                uint32_t dstOff = d.offsetOut + row * d.pitchOut;
                if (srcOff + d.lineLength <= vramSz &&
                    dstOff + d.lineLength <= vramSz) {
                    std::memmove(vram + dstOff, vram + srcOff, d.lineLength);
                    // Byte-swap if format conversion required (LE↔BE)
                    if (needSwap && outFmt == 4) {
                        uint32_t* p = reinterpret_cast<uint32_t*>(vram + dstOff);
                        for (uint32_t w = 0; w < d.lineLength / 4; ++w) {
                            uint32_t v = p[w];
                            p[w] = ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
                                   ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000u);
                        }
                    } else if (needSwap && outFmt == 2) {
                        uint16_t* p = reinterpret_cast<uint16_t*>(vram + dstOff);
                        for (uint32_t w = 0; w < d.lineLength / 2; ++w) {
                            uint16_t v = p[w];
                            p[w] = (v >> 8) | (v << 8);
                        }
                    }
                }
            }
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
            dispatchMethod(state, vram, method, data, cmd.subchannel);
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
    if (state->unknownMethodCount)
        printf("│ Unknown methods: %u                            \n", state->unknownMethodCount);
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
