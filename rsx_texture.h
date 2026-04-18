// rsx_texture.h — host-side RSX texture sampler
// Knows enough of the NV40/RSX texture format to satisfy the FP
// interpreter's TEX/TXP opcodes for bringup and end-to-end tests.
//
// Only nearest-neighbor wrap-mode sampling for the formats the
// pre-rendered cube/triangle scenes actually exercise:
//
//   format byte 0x85 — A8R8G8B8       (4 bytes/texel, big-endian ARGB)
//   format byte 0x81 — B8             (1 byte/texel, single channel)
//
// These are the two formats used by Cell SDK gcm samples and by every
// PS3 home-screen / first-party intro we care about for boot bringup.
//
#pragma once
#include "rsx_defs.h"
#include "rsx_fp_shader.h"
#include <cstdint>
#include <cmath>
#include <cstring>

namespace ps3rsx {

struct HostTextureSamplerCtx {
    const uint8_t* vram;        // host pointer at VRAM_BASE (== sandbox + RAM_SIZE)
    uint64_t       vramSize;    // bytes
    const rsx::RSXState* state;
};

// Decode one texel at (x,y) for the given texture unit.
// Returns RGBA in [0,1]. Out-of-range / disabled units return opaque black.
static inline void rsx_fetch_texel(const HostTextureSamplerCtx* ctx,
                                   uint32_t unit, int32_t x, int32_t y,
                                   float rgba[4])
{
    rgba[0] = rgba[1] = rgba[2] = 0.f; rgba[3] = 1.f;
    if (!ctx || unit >= 16 || !ctx->state) return;
    const auto& t = ctx->state->textures[unit];
    if (!t.enabled || t.width == 0 || t.height == 0) return;

    // Wrap (REPEAT) — the only mode bringup needs.
    int32_t W = (int32_t)t.width, H = (int32_t)t.height;
    int32_t xw = ((x % W) + W) % W;
    int32_t yw = ((y % H) + H) % H;

    // Format byte is the low 8 bits of TEXTURE_FORMAT.
    uint8_t fmt = (uint8_t)(t.format & 0xFF);
    uint64_t off = (uint64_t)t.offset;

    // Bounds check against host VRAM mirror.
    if (off >= ctx->vramSize) return;
    const uint8_t* base = ctx->vram + off;

    switch (fmt) {
    case 0x85: { // A8R8G8B8
        uint64_t pix = (uint64_t)(yw * W + xw) * 4;
        if (off + pix + 4 > ctx->vramSize) return;
        uint8_t a = base[pix + 0];
        uint8_t r = base[pix + 1];
        uint8_t g = base[pix + 2];
        uint8_t b = base[pix + 3];
        rgba[0] = r / 255.f; rgba[1] = g / 255.f;
        rgba[2] = b / 255.f; rgba[3] = a / 255.f;
        break;
    }
    case 0x81: { // B8 (luminance)
        uint64_t pix = (uint64_t)(yw * W + xw);
        if (off + pix + 1 > ctx->vramSize) return;
        float v = base[pix] / 255.f;
        rgba[0] = rgba[1] = rgba[2] = v; rgba[3] = 1.f;
        break;
    }
    default:
        // Unknown format → magenta debug texel.
        rgba[0] = 1.f; rgba[1] = 0.f; rgba[2] = 1.f; rgba[3] = 1.f;
        break;
    }
}

// FPSampler entry point: nearest-neighbor with REPEAT wrap.
// Compatible with the typedef in rsx_fp_shader.h.
static inline void rsx_host_sampler(void* userdata, uint32_t texUnit,
                                    const float uvw[3], float rgba[4])
{
    auto* ctx = (const HostTextureSamplerCtx*)userdata;
    if (!ctx || texUnit >= 16) {
        rgba[0] = rgba[1] = rgba[2] = 0.f; rgba[3] = 1.f; return;
    }
    const auto& t = ctx->state->textures[texUnit];
    if (!t.enabled || t.width == 0 || t.height == 0) {
        rgba[0] = rgba[1] = rgba[2] = 0.f; rgba[3] = 1.f; return;
    }
    int32_t x = (int32_t)floorf(uvw[0] * (float)t.width);
    int32_t y = (int32_t)floorf(uvw[1] * (float)t.height);
    rsx_fetch_texel(ctx, texUnit, x, y, rgba);
}

} // namespace ps3rsx
