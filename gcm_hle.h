#pragma once
// gcm_hle.h — Host-side cellGcm command-buffer builders
//
// libgcm.sprx is the PS3 user-mode RSX bridge. Real games never write
// FIFO methods directly; they call cellGcmSet* functions which append
// pre-formed method headers + payloads into a command buffer in main
// memory. RSX consumes that buffer asynchronously through the CMD ring.
//
// This header gives us a host-side shim with the same shape as the
// libgcm calls we care about, so PPC code (and unit tests) can build
// FIFO programs the same way real games do. Once the PPC ELF loader
// can resolve libgcm imports, these builders become the bodies of
// the imported NIDs.
//
// All methods written are big-endian neutral here — they're consumed
// by rsx_process_fifo on the host before being interpreted, which
// expects native-endian uint32 method headers + payload words.

#include "rsx_defs.h"
#include <cstdint>
#include <cstring>

namespace gcm {

// ── Command buffer cursor ──────────────────────────────────────────
//
// Mirrors the cellGcmContextData head/begin/end pointers. Games call
// cellGcmSetCurrentBuffer() to point at a ring; we accept any flat
// buffer here and bump head/end on append.
struct GcmCtx {
    uint32_t* begin;
    uint32_t* head;
    uint32_t* end;       // one-past-last
};

inline void gcm_init_ctx(GcmCtx* c, uint32_t* buf, uint32_t words) {
    c->begin = buf;
    c->head  = buf;
    c->end   = buf + words;
}

inline uint32_t gcm_used(const GcmCtx* c) {
    return (uint32_t)(c->head - c->begin);
}

// FIFO method header. `count` is the number of payload dwords that
// follow. Real RSX uses a SLI/IO bit pattern; rsx_command_processor
// only inspects bits [29:18] (count) and [15:2] (method offset).
static inline uint32_t make_method(uint32_t method, uint32_t count) {
    return ((count & 0x7FFu) << 18) | (method & 0xFFFCu);
}

static inline void emit1(GcmCtx* c, uint32_t method, uint32_t a) {
    if (c->head + 2 > c->end) return;
    *c->head++ = make_method(method, 1);
    *c->head++ = a;
}

// ── State setters ─────────────────────────────────────────────────

inline void cellGcmSetSurface(GcmCtx* c,
                              uint32_t format, uint32_t width, uint32_t height,
                              uint32_t pitchA, uint32_t offsetA,
                              uint32_t depthFmt, uint32_t depthPitch, uint32_t depthOffset)
{
    // Order matters: dimensions must be in RSXState BEFORE the format
    // method fires, because SURFACE_FORMAT triggers onSurfaceSetup which
    // re-allocates the framebuffer based on the current width/height.
    emit1(c, rsx::NV4097_SET_SURFACE_CLIP_HORIZONTAL, width);
    emit1(c, rsx::NV4097_SET_SURFACE_CLIP_VERTICAL,   height);
    emit1(c, rsx::NV4097_SET_SURFACE_PITCH_A,         pitchA);
    emit1(c, rsx::NV4097_SET_SURFACE_COLOR_AOFFSET,   offsetA);
    emit1(c, rsx::NV4097_SET_SURFACE_FORMAT,          format);
    (void)depthFmt; (void)depthPitch; (void)depthOffset;
}

inline void cellGcmSetViewport(GcmCtx* c,
                               uint32_t x, uint32_t y,
                               uint32_t w, uint32_t h)
{
    emit1(c, rsx::NV4097_SET_VIEWPORT_HORIZONTAL, (w << 16) | (x & 0xFFFFu));
    emit1(c, rsx::NV4097_SET_VIEWPORT_VERTICAL,   (h << 16) | (y & 0xFFFFu));
    emit1(c, rsx::NV4097_SET_SCISSOR_HORIZONTAL,  (w << 16) | (x & 0xFFFFu));
    emit1(c, rsx::NV4097_SET_SCISSOR_VERTICAL,    (h << 16) | (y & 0xFFFFu));
}

inline void cellGcmSetClearColor(GcmCtx* c, uint32_t argb) {
    emit1(c, rsx::NV4097_SET_COLOR_CLEAR_VALUE, argb);
}

inline void cellGcmSetClearSurface(GcmCtx* c, uint32_t mask) {
    emit1(c, rsx::NV4097_CLEAR_SURFACE, mask);
}

inline void cellGcmSetVertexDataArray(GcmCtx* c,
                                      uint32_t slot,
                                      uint32_t stride,
                                      uint32_t componentCount,
                                      uint32_t format, // VERTEX_F / VERTEX_UB / ...
                                      uint32_t vramOffset)
{
    emit1(c, rsx::NV4097_SET_VERTEX_DATA_ARRAY_FORMAT + slot * 4u,
          (stride << 8) | (componentCount << 4) | (format & 0xFu));
    emit1(c, rsx::NV4097_SET_VERTEX_DATA_ARRAY_OFFSET + slot * 4u,
          vramOffset);
}

inline void cellGcmSetDrawIndexArray(GcmCtx* c,
                                     uint32_t indexAddress,
                                     bool     u32indices,
                                     uint32_t prim,
                                     uint32_t firstIndex,
                                     uint32_t indexCount)
{
    emit1(c, rsx::NV4097_SET_INDEX_ARRAY_ADDRESS, indexAddress);
    emit1(c, rsx::NV4097_SET_INDEX_ARRAY_DMA,     u32indices ? 1u : 0u);
    emit1(c, rsx::NV4097_SET_BEGIN_END,           prim);
    emit1(c, rsx::NV4097_DRAW_INDEX_ARRAY,
          (((indexCount - 1u) & 0xFFu) << 24) | (firstIndex & 0xFFFFFFu));
    emit1(c, rsx::NV4097_SET_BEGIN_END,           0u);
}

inline void cellGcmSetDrawArrays(GcmCtx* c,
                                 uint32_t prim,
                                 uint32_t firstVertex,
                                 uint32_t vertexCount)
{
    emit1(c, rsx::NV4097_SET_BEGIN_END, prim);
    emit1(c, rsx::NV4097_DRAW_ARRAYS,
          (((vertexCount - 1u) & 0xFFu) << 24) | (firstVertex & 0xFFFFFFu));
    emit1(c, rsx::NV4097_SET_BEGIN_END, 0u);
}

inline void cellGcmSetTexture(GcmCtx* c, uint32_t unit,
                              uint32_t vramOffset,
                              uint32_t format,
                              uint32_t width, uint32_t height)
{
    const uint32_t base = rsx::NV4097_SET_TEXTURE_OFFSET + unit * 0x20u;
    emit1(c, base + 0x00,                  vramOffset);
    emit1(c, base + 0x04,                  format);
    emit1(c, rsx::NV4097_SET_TEXTURE_IMAGE_RECT + unit * 0x20u,
          (width << 16) | (height & 0xFFFFu));
    // Enable bit lives in TEXTURE_CONTROL0; keeping it simple for now.
    emit1(c, base + 0x0C, 0x80000000u);
}

inline void cellGcmSetFlip(GcmCtx* c, uint32_t surfaceOffset) {
    emit1(c, rsx::NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP, surfaceOffset);
}

inline void cellGcmFinish(GcmCtx* c, uint32_t ref) {
    emit1(c, rsx::NV4097_SET_REFERENCE, ref);
}

} // namespace gcm
