#pragma once
// rsx_defs.h — RSX Reality Synthesizer definitions
//
// NV47-class GPU (G70/GeForce 7800 GTX derivative) as used in PlayStation 3.
// Defines command FIFO format, NV4097 3D methods, and GPU state structures.

#include <cstdint>
#include <cstring>
#include <vector>

namespace rsx {

// ── Host→VRAM helpers (byte-swap to PS3 big-endian) ─────────────
// PS3 Cell stores data big-endian; the RSX vertex decoder expects
// this layout. Unit tests that write vertex data from the x86 host
// must use these to match what a real PS3 would produce.
static inline void store_be_floats(uint8_t* dst, const float* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t le;
        std::memcpy(&le, &src[i], 4);
        uint32_t be = __builtin_bswap32(le);
        std::memcpy(dst + i * 4, &be, 4);
    }
}
static inline void store_be_u32s(uint8_t* dst, const uint32_t* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t be = __builtin_bswap32(src[i]);
        std::memcpy(dst + i * 4, &be, 4);
    }
}

// ═══════════════════════════════════════════════════════════════════
// PS3 Memory Map for RSX
// ═══════════════════════════════════════════════════════════════════

static constexpr uint64_t VRAM_BASE       = 0x10000000ULL;  // 256MB VRAM
static constexpr uint64_t VRAM_SIZE       = 256ULL * 1024 * 1024;
static constexpr uint64_t RSX_CTRL_BASE   = 0x40000000ULL;  // Control registers
static constexpr uint32_t FIFO_SIZE       = 1024 * 1024;    // 1MB command buffer

// ═══════════════════════════════════════════════════════════════════
// FIFO Control Registers (offsets from RSX_CTRL_BASE)
// ═══════════════════════════════════════════════════════════════════

static constexpr uint32_t CTRL_PUT = 0x10;  // Write pointer
static constexpr uint32_t CTRL_GET = 0x14;  // Read pointer
static constexpr uint32_t CTRL_REF = 0x18;  // Reference value

// ═══════════════════════════════════════════════════════════════════
// NV4097 3D Engine Methods (subchannel 0)
// All offsets verified against RPCS3 gcm_enums.h
// ═══════════════════════════════════════════════════════════════════

// Object / control
static constexpr uint32_t NV4097_SET_OBJECT              = 0x00000000;
static constexpr uint32_t NV4097_NO_OPERATION             = 0x00000100;
static constexpr uint32_t NV4097_SET_CONTEXT_DMA_NOTIFIES = 0x00000180;
static constexpr uint32_t NV4097_SET_CONTEXT_DMA_A        = 0x00000184;
static constexpr uint32_t NV4097_SET_CONTEXT_DMA_B        = 0x00000188;
static constexpr uint32_t NV4097_SET_CONTEXT_DMA_COLOR_B  = 0x0000018C;
static constexpr uint32_t NV4097_SET_CONTEXT_DMA_STATE    = 0x00000190;
static constexpr uint32_t NV4097_SET_CONTEXT_DMA_COLOR_A  = 0x00000194;
static constexpr uint32_t NV4097_SET_CONTEXT_DMA_ZETA     = 0x00000198;

// Surface (render target) configuration
static constexpr uint32_t NV4097_SET_SURFACE_CLIP_HORIZONTAL = 0x00000200;
static constexpr uint32_t NV4097_SET_SURFACE_CLIP_VERTICAL   = 0x00000204;
static constexpr uint32_t NV4097_SET_SURFACE_FORMAT          = 0x00000208;
static constexpr uint32_t NV4097_SET_SURFACE_PITCH_A         = 0x0000020C;
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_AOFFSET   = 0x00000210;
static constexpr uint32_t NV4097_SET_SURFACE_ZETA_OFFSET     = 0x00000214;
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_BOFFSET   = 0x00000218;
static constexpr uint32_t NV4097_SET_SURFACE_PITCH_B         = 0x0000021C;
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_TARGET    = 0x00000220;
static constexpr uint32_t NV4097_SET_SURFACE_PITCH_Z         = 0x0000022C;
static constexpr uint32_t NV4097_SET_SURFACE_PITCH_C         = 0x00000280;
static constexpr uint32_t NV4097_SET_SURFACE_PITCH_D         = 0x00000284;
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_COFFSET   = 0x00000288;
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_DOFFSET   = 0x0000028C;

// Alpha / blend / stencil state (0x0300 region)
static constexpr uint32_t NV4097_SET_DITHER_ENABLE            = 0x00000300;
static constexpr uint32_t NV4097_SET_ALPHA_TEST_ENABLE        = 0x00000304;
static constexpr uint32_t NV4097_SET_ALPHA_FUNC               = 0x00000308;
static constexpr uint32_t NV4097_SET_ALPHA_REF                = 0x0000030C;
static constexpr uint32_t NV4097_SET_BLEND_ENABLE             = 0x00000310;
static constexpr uint32_t NV4097_SET_BLEND_FUNC_SFACTOR       = 0x00000314;
static constexpr uint32_t NV4097_SET_BLEND_FUNC_DFACTOR       = 0x00000318;
static constexpr uint32_t NV4097_SET_BLEND_COLOR              = 0x0000031C;
static constexpr uint32_t NV4097_SET_BLEND_EQUATION           = 0x00000320;
static constexpr uint32_t NV4097_SET_COLOR_MASK               = 0x00000324;
static constexpr uint32_t NV4097_SET_STENCIL_TEST_ENABLE      = 0x00000328;
static constexpr uint32_t NV4097_SET_STENCIL_MASK             = 0x0000032C;
static constexpr uint32_t NV4097_SET_STENCIL_FUNC             = 0x00000330;
static constexpr uint32_t NV4097_SET_STENCIL_FUNC_REF         = 0x00000334;
static constexpr uint32_t NV4097_SET_STENCIL_FUNC_MASK        = 0x00000338;
static constexpr uint32_t NV4097_SET_STENCIL_OP_FAIL          = 0x0000033C;
static constexpr uint32_t NV4097_SET_STENCIL_OP_ZFAIL         = 0x00000340;
static constexpr uint32_t NV4097_SET_STENCIL_OP_ZPASS         = 0x00000344;
static constexpr uint32_t NV4097_SET_SHADE_MODE               = 0x00000368;

// Logic op
static constexpr uint32_t NV4097_SET_LOGIC_OP_ENABLE          = 0x00000D04;
static constexpr uint32_t NV4097_SET_LOGIC_OP                 = 0x00000D08;

// Two-sided stencil
static constexpr uint32_t NV4097_SET_TWO_SIDED_STENCIL_TEST_ENABLE = 0x00000D54;
// Two-sided color (back-face colors)
static constexpr uint32_t NV4097_SET_TWO_SIDE_LIGHT_EN             = 0x00000294;
static constexpr uint32_t NV4097_SET_BACK_STENCIL_FUNC        = 0x00000D58;
static constexpr uint32_t NV4097_SET_BACK_STENCIL_FUNC_REF    = 0x00000D5C;
static constexpr uint32_t NV4097_SET_BACK_STENCIL_FUNC_MASK   = 0x00000D60;
static constexpr uint32_t NV4097_SET_BACK_STENCIL_OP_FAIL     = 0x00000D64;
static constexpr uint32_t NV4097_SET_BACK_STENCIL_OP_ZFAIL    = 0x00000D68;
static constexpr uint32_t NV4097_SET_BACK_STENCIL_OP_ZPASS    = 0x00000D6C;
static constexpr uint32_t NV4097_SET_BACK_STENCIL_MASK        = 0x00000D70;

// Fog
static constexpr uint32_t NV4097_SET_FOG_MODE                 = 0x000008A4;
static constexpr uint32_t NV4097_SET_FOG_PARAMS               = 0x000008A8; // 2 floats (param0, param1)

// Depth bounds
static constexpr uint32_t NV4097_SET_DEPTH_BOUNDS_TEST_ENABLE = 0x00000380;
static constexpr uint32_t NV4097_SET_DEPTH_BOUNDS_MIN         = 0x00000384;
static constexpr uint32_t NV4097_SET_DEPTH_BOUNDS_MAX         = 0x00000388;

// Primitive restart
static constexpr uint32_t NV4097_SET_RESTART_INDEX_ENABLE     = 0x000006B4;
static constexpr uint32_t NV4097_SET_RESTART_INDEX            = 0x000006B8;

// User clip planes (6 planes, packed 2 bits each in one register)
static constexpr uint32_t NV4097_SET_USER_CLIP_PLANE_CONTROL  = 0x00001478;

// Scissor
static constexpr uint32_t NV4097_SET_SCISSOR_HORIZONTAL       = 0x000008C0;
static constexpr uint32_t NV4097_SET_SCISSOR_VERTICAL         = 0x000008C4;

// Line / Point rendering
static constexpr uint32_t NV4097_SET_LINE_WIDTH               = 0x000003B8;
static constexpr uint32_t NV4097_SET_POINT_SIZE               = 0x00001EE0;
static constexpr uint32_t NV4097_SET_POINT_SPRITE_CONTROL     = 0x00001EE8;

// Per-MRT color write mask (targets 1-3; target 0 uses SET_COLOR_MASK)
static constexpr uint32_t NV4097_SET_COLOR_MASK_MRT           = 0x00000370;

// VP attribute output mask (which VP outputs are active)
static constexpr uint32_t NV4097_SET_VERTEX_ATTRIB_OUTPUT_MASK = 0x00001FF4;

// Vertex attrib instancing dividers (per-attribute frequency)
static constexpr uint32_t NV4097_SET_FREQUENCY_DIVIDER_OPERATION = 0x00001FC0;

// Texture control1 register (remap/swizzle)
static constexpr uint32_t NV4097_SET_TEXTURE_CONTROL1         = 0x00000B40;

// Shader programs
static constexpr uint32_t NV4097_SET_SHADER_PROGRAM          = 0x000008E4;

// Inline vertex data (4 floats per attribute, 16 attributes)
static constexpr uint32_t NV4097_SET_VERTEX_DATA4F_M         = 0x00001C00;

// Viewport
static constexpr uint32_t NV4097_SET_VIEWPORT_HORIZONTAL  = 0x00000A00;
static constexpr uint32_t NV4097_SET_VIEWPORT_VERTICAL    = 0x00000A04;
// 0x0A20..0x0A2F = VIEWPORT_OFFSET (4 floats), 0x0A30..0x0A3F = VIEWPORT_SCALE
static constexpr uint32_t NV4097_SET_VIEWPORT_OFFSET      = 0x00000A20;
static constexpr uint32_t NV4097_SET_VIEWPORT_SCALE       = 0x00000A30;

// Depth state
static constexpr uint32_t NV4097_SET_DEPTH_FUNC               = 0x00000A6C;
static constexpr uint32_t NV4097_SET_DEPTH_MASK               = 0x00000A70;
static constexpr uint32_t NV4097_SET_DEPTH_TEST_ENABLE        = 0x00000A74;

// Polygon offset (depth bias)
static constexpr uint32_t NV4097_SET_POLY_OFFSET_FILL_ENABLE  = 0x00000A68;
static constexpr uint32_t NV4097_SET_POLYGON_OFFSET_SCALE_FACTOR = 0x00000A78;
static constexpr uint32_t NV4097_SET_POLYGON_OFFSET_BIAS      = 0x00000A7C;

// Transform program (vertex shader)
static constexpr uint32_t NV4097_SET_TRANSFORM_PROGRAM        = 0x00000B80;

// Vertex arrays (16 slots, stride 4 per slot)
static constexpr uint32_t NV4097_SET_VERTEX_DATA_ARRAY_OFFSET = 0x00001680;
static constexpr uint32_t NV4097_SET_VERTEX_DATA_ARRAY_FORMAT = 0x00001740;

// Draw commands
static constexpr uint32_t NV4097_SET_BEGIN_END               = 0x00001808;
static constexpr uint32_t NV4097_DRAW_ARRAYS                 = 0x00001814;
static constexpr uint32_t NV4097_SET_INDEX_ARRAY_ADDRESS     = 0x0000181C;
static constexpr uint32_t NV4097_SET_INDEX_ARRAY_DMA         = 0x00001820;
static constexpr uint32_t NV4097_DRAW_INDEX_ARRAY            = 0x00001824;
static constexpr uint32_t NV4097_DRAW_INLINE_ARRAY           = 0x00001818;

// Instanced draw
static constexpr uint32_t NV4097_SET_BEGIN_END_INSTANCE_CNT  = 0x00001844;

// Cull face / polygon mode
static constexpr uint32_t NV4097_SET_FRONT_POLYGON_MODE      = 0x00001828;
static constexpr uint32_t NV4097_SET_BACK_POLYGON_MODE       = 0x0000182C;
static constexpr uint32_t NV4097_SET_CULL_FACE               = 0x00001830;
static constexpr uint32_t NV4097_SET_FRONT_FACE              = 0x00001834;
static constexpr uint32_t NV4097_SET_CULL_FACE_ENABLE        = 0x0000183C;

// Texture (16 units, stride 0x20 per unit)
static constexpr uint32_t NV4097_SET_TEXTURE_OFFSET          = 0x00001A00;
static constexpr uint32_t NV4097_SET_TEXTURE_FORMAT           = 0x00001A04;
static constexpr uint32_t NV4097_SET_TEXTURE_CONTROL0         = 0x00001A0C;
static constexpr uint32_t NV4097_SET_TEXTURE_ADDRESS          = 0x00001A08;
static constexpr uint32_t NV4097_SET_TEXTURE_FILTER           = 0x00001A14;
static constexpr uint32_t NV4097_SET_TEXTURE_IMAGE_RECT       = 0x00001A18;
static constexpr uint32_t NV4097_SET_TEXTURE_BORDER_COLOR     = 0x00001A10;
static constexpr uint32_t NV4097_SET_TEXTURE_CONTROL3         = 0x00000B00; // depth/pitch per unit

// Clear / present
static constexpr uint32_t NV4097_SET_SHADER_WINDOW           = 0x00001D88;
static constexpr uint32_t NV4097_SET_SHADER_CONTROL          = 0x00001D60;  // bit1=sRGB write
static constexpr uint32_t NV4097_SET_ZSTENCIL_CLEAR_VALUE    = 0x00001D8C;
static constexpr uint32_t NV4097_SET_COLOR_CLEAR_VALUE       = 0x00001D90;
static constexpr uint32_t NV4097_CLEAR_SURFACE               = 0x00001D94;

// Transform constants / programs
static constexpr uint32_t NV4097_SET_TRANSFORM_PROGRAM_LOAD   = 0x00001E9C;
static constexpr uint32_t NV4097_SET_TRANSFORM_PROGRAM_START = 0x00001EA0;
static constexpr uint32_t NV4097_SET_TRANSFORM_CONSTANT_LOAD  = 0x00001EFC;
static constexpr uint32_t NV4097_SET_TRANSFORM_CONSTANT       = 0x00001F00;

// NV4097 surface target encodings.
enum SurfaceColorTarget : uint32_t {
    SURFACE_TARGET_NONE = 0,
    SURFACE_TARGET_A    = 1,
    SURFACE_TARGET_B    = 2,
    SURFACE_TARGET_AB   = 3,
    SURFACE_TARGET_MRT1 = 0x13,  // AB
    SURFACE_TARGET_MRT2 = 0x17,  // ABC
    SURFACE_TARGET_MRT3 = 0x1F,  // ABCD
};

// Flip (surface A offset for display flip)
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP = 0x00000E20;

// Anti-aliasing / line & polygon smoothing
static constexpr uint32_t NV4097_SET_ANTI_ALIASING_CONTROL     = 0x00000D7C;
static constexpr uint32_t NV4097_SET_LINE_SMOOTH_ENABLE        = 0x00000DB4;
static constexpr uint32_t NV4097_SET_POLY_SMOOTH_ENABLE        = 0x00000DB8;

// Shader packer / flat shade / attrib input mask
static constexpr uint32_t NV4097_SET_SHADER_PACKER             = 0x00001D48;
static constexpr uint32_t NV4097_SET_FLAT_SHADE_OP             = 0x00001D98;
static constexpr uint32_t NV4097_SET_VERTEX_ATTRIB_INPUT_MASK  = 0x00001FF0;

// ZCULL control
static constexpr uint32_t NV4097_SET_ZCULL_STATS_ENABLE        = 0x00001DC8;
static constexpr uint32_t NV4097_SET_SCULL_CONTROL             = 0x00001D80;

// Misc pipeline control
static constexpr uint32_t NV4097_SET_REDUCE_DST_COLOR          = 0x00000D2C;
static constexpr uint32_t NV4097_SET_TRANSFORM_TIMEOUT         = 0x00001EF8;
static constexpr uint32_t NV4097_SET_WAIT_FOR_IDLE             = 0x00000110;

// Texture aniso control per unit (16 units at stride 4)
static constexpr uint32_t NV4097_SET_TEXTURE_CONTROL2          = 0x00001D34;

// NV406E Reference register (subchannel agnostic)
static constexpr uint32_t NV4097_SET_REFERENCE               = 0x00000050;

// NV406E subchannel semaphore (subchannels 1-2, cellGcm labels)
static constexpr uint32_t NV406E_SET_REFERENCE               = 0x00000050;
static constexpr uint32_t NV406E_SEMAPHORE_OFFSET            = 0x00000010;
static constexpr uint32_t NV406E_SEMAPHORE_ACQUIRE           = 0x00000014;
static constexpr uint32_t NV406E_SEMAPHORE_RELEASE           = 0x00000018;

// Semaphore / label system (SPU↔RSX sync)
static constexpr uint32_t NV4097_SET_SEMAPHORE_OFFSET        = 0x00001D6C;
static constexpr uint32_t NV4097_BACK_END_WRITE_SEMAPHORE_RELEASE = 0x00001D70;
static constexpr uint32_t NV4097_TEXTURE_READ_SEMAPHORE_RELEASE   = 0x00001D74;
static constexpr uint32_t NV4097_SET_NOTIFY                       = 0x00000104;
static constexpr uint32_t NV4097_NOTIFY                           = 0x00000108;

// ── NV3062/NV3089/NV0039 — 2D engine (DMA blit/transfer) ─────────
// These operate on subchannel 3 (2D context) but games often submit
// them inline in the 3D FIFO. Method offsets are relative to the
// object class (NV3089 = scaled image from memory).
static constexpr uint32_t NV3062_SET_OBJECT                    = 0x00000000;
static constexpr uint32_t NV3062_SET_COLOR_FORMAT              = 0x00000300;
static constexpr uint32_t NV3062_SET_PITCH                     = 0x00000304;
static constexpr uint32_t NV3062_SET_OFFSET_DESTIN             = 0x00000308;

static constexpr uint32_t NV3089_SET_CONTEXT_SURFACE           = 0x00000184;
static constexpr uint32_t NV3089_SET_COLOR_FORMAT              = 0x00000300;
static constexpr uint32_t NV3089_SET_OPERATION                 = 0x000002FC;
static constexpr uint32_t NV3089_CLIP_POINT                    = 0x00000308;
static constexpr uint32_t NV3089_CLIP_SIZE                     = 0x0000030C;
static constexpr uint32_t NV3089_IMAGE_OUT_POINT               = 0x00000310;
static constexpr uint32_t NV3089_IMAGE_OUT_SIZE                = 0x00000314;
static constexpr uint32_t NV3089_DS_DX                         = 0x00000318;
static constexpr uint32_t NV3089_DT_DY                         = 0x0000031C;
static constexpr uint32_t NV3089_IMAGE_IN_SIZE                 = 0x00000400;
static constexpr uint32_t NV3089_IMAGE_IN_FORMAT               = 0x00000404;
static constexpr uint32_t NV3089_IMAGE_IN_OFFSET               = 0x00000408;
static constexpr uint32_t NV3089_IMAGE_IN                      = 0x0000040C;

static constexpr uint32_t NV0039_OFFSET_IN                     = 0x0000030C;
static constexpr uint32_t NV0039_OFFSET_OUT                    = 0x00000310;
static constexpr uint32_t NV0039_PITCH_IN                      = 0x00000314;
static constexpr uint32_t NV0039_PITCH_OUT                     = 0x00000318;
static constexpr uint32_t NV0039_LINE_LENGTH_IN                = 0x0000031C;
static constexpr uint32_t NV0039_LINE_COUNT                    = 0x00000320;
static constexpr uint32_t NV0039_FORMAT                        = 0x00000324;
static constexpr uint32_t NV0039_BUFFER_NOTIFY                 = 0x00000104;
static constexpr uint32_t NV0039_SET_CONTEXT_DMA_BUFFER_IN     = 0x00000184;
static constexpr uint32_t NV0039_SET_CONTEXT_DMA_BUFFER_OUT    = 0x00000188;

// ── Occlusion query / ZCULL ──────────────────────────────────────
static constexpr uint32_t NV4097_SET_ZPASS_PIXEL_COUNT_ENABLE  = 0x00001D78;
static constexpr uint32_t NV4097_GET_REPORT                    = 0x00001800;
static constexpr uint32_t NV4097_CLEAR_REPORT_VALUE            = 0x00001D9C;
static constexpr uint32_t NV4097_SET_RENDER_ENABLE             = 0x00001D7C;
static constexpr uint32_t NV4097_INVALIDATE_L2                 = 0x00001FD0;
static constexpr uint32_t NV4097_INVALIDATE_VERTEX_FILE        = 0x00001710;
static constexpr uint32_t NV4097_INVALIDATE_VERTEX_CACHE_FILE  = 0x00001714;

// ═══════════════════════════════════════════════════════════════════
// Enumerations
// ═══════════════════════════════════════════════════════════════════

// Primitive types
enum PrimitiveType : uint32_t {
    PRIM_POINTS         = 1,
    PRIM_LINES          = 2,
    PRIM_LINE_LOOP      = 3,
    PRIM_LINE_STRIP     = 4,
    PRIM_TRIANGLES      = 5,
    PRIM_TRIANGLE_STRIP = 6,
    PRIM_TRIANGLE_FAN   = 7,
    PRIM_QUADS          = 8,
    PRIM_QUAD_STRIP     = 9,
    PRIM_POLYGON        = 10,
};

// Surface color formats
enum SurfaceFormat : uint32_t {
    SURFACE_R5G6B5         = 3,
    SURFACE_X8R8G8B8       = 5,
    SURFACE_A8R8G8B8       = 8,
    SURFACE_F_W16Z16Y16X16 = 11,
    SURFACE_F_W32Z32Y32X32 = 12,
};

// Depth buffer formats
enum DepthFormat : uint32_t {
    DEPTH_Z16   = 1,
    DEPTH_Z24S8 = 2,
};

// Vertex attribute types
enum VertexType : uint32_t {
    VERTEX_S1     = 1,  // signed normalized 16-bit
    VERTEX_F      = 2,  // 32-bit float
    VERTEX_SF     = 3,  // 16-bit float
    VERTEX_UB     = 4,  // unsigned byte
    VERTEX_S32K   = 5,  // signed 16-bit
    VERTEX_CMP    = 6,  // compressed (11/11/10)
    VERTEX_UB256  = 7,  // unsigned byte (unscaled)
};

// Comparison functions (depth / alpha test)
enum CompareFunc : uint32_t {
    CMP_NEVER    = 0x0200,
    CMP_LESS     = 0x0201,
    CMP_EQUAL    = 0x0202,
    CMP_LEQUAL   = 0x0203,
    CMP_GREATER  = 0x0204,
    CMP_NOTEQUAL = 0x0205,
    CMP_GEQUAL   = 0x0206,
    CMP_ALWAYS   = 0x0207,
};

// Clear surface mask bits
// NV4097 CLEAR_SURFACE mask bits (raw hardware)
static constexpr uint32_t CLEAR_Z       = 0x01;  // bit 0 — depth
static constexpr uint32_t CLEAR_STENCIL = 0x02;  // bit 1 — stencil
static constexpr uint32_t CLEAR_R       = 0x10;  // bit 4
static constexpr uint32_t CLEAR_G       = 0x20;  // bit 5
static constexpr uint32_t CLEAR_B       = 0x40;  // bit 6
static constexpr uint32_t CLEAR_A       = 0x80;  // bit 7
static constexpr uint32_t CLEAR_RGBA    = 0xF0;  // bits 4-7
// Convenience aliases
static constexpr uint32_t CLEAR_COLOR  = CLEAR_RGBA;
static constexpr uint32_t CLEAR_DEPTH  = CLEAR_Z;

// ═══════════════════════════════════════════════════════════════════
// RSX GPU State — software shadow of hardware registers
// ═══════════════════════════════════════════════════════════════════

struct RSXState {
    // FIFO control
    uint32_t put;
    uint32_t get;
    uint32_t ref;

    // Surface (render target)
    uint32_t surfaceFormat;
    uint32_t surfaceAntialias;   // AA mode from SURFACE_FORMAT bits 12-15 (0=none, 4=2x, 12=4x)
    uint32_t surfaceWidth;
    uint32_t surfaceHeight;
    uint32_t surfacePitchA;
    uint32_t surfacePitchB;
    uint32_t surfacePitchC;
    uint32_t surfacePitchD;
    uint32_t surfaceOffsetA;   // color buffer A offset in VRAM
    uint32_t surfaceOffsetB;   // color buffer B
    uint32_t surfaceOffsetC;   // color buffer C (MRT)
    uint32_t surfaceOffsetD;   // color buffer D (MRT)
    uint32_t surfaceColorTarget;  // encoded: 0=NONE,1=A,2=B,3=AB,0x13=MRT1,0x17=MRT2,0x1F=MRT3
    uint32_t depthOffset;
    uint32_t depthPitch;
    uint32_t depthFormat;

    // Index buffer state for VRAM-resident indexed draws.
    // SET_INDEX_ARRAY_ADDRESS holds a byte offset into VRAM, DMA holds
    // the format (0 = U16 little-endian, 1 = U32) packed in low bits.
    uint32_t indexArrayAddress;
    uint32_t indexArrayFormat;

    // Viewport / Scissor
    uint16_t viewportX, viewportY, viewportW, viewportH;
    uint16_t scissorX, scissorY, scissorW, scissorH;
    // Viewport offset/scale — NDC→screen transform (4 floats each)
    float    vpOffset[4];   // x, y, z, w offset
    float    vpScale[4];    // x, y, z, w scale
    bool     vpOffsetScaleSet;  // true if game explicitly set offset/scale

    // Clear
    uint32_t colorClearValue;

    // Depth / Stencil
    bool     depthTestEnable;
    uint32_t depthFunc;
    bool     depthMask;         // depth write enable
    bool     stencilTestEnable;
    uint32_t stencilFunc;       // GL compare 0x200..0x207
    uint32_t stencilRef;
    uint32_t stencilFuncMask;
    uint32_t stencilWriteMask;
    uint32_t stencilOpFail;     // GL op: KEEP/ZERO/REPLACE/...
    uint32_t stencilOpZFail;
    uint32_t stencilOpZPass;

    // Blend
    bool     blendEnable;
    uint32_t blendSFactor;      // low 16 = RGB, high 16 = A (GL factor enums)
    uint32_t blendDFactor;
    uint32_t blendEquation;     // low 16 = RGB, high 16 = A (GL equation enums)
    uint32_t blendColor;        // packed ARGB8888

    // Cull
    bool     cullFaceEnable;
    uint32_t cullFace;

    // Alpha test / front face / color mask / shade mode (Phase 4b)
    bool     alphaTestEnable;
    uint32_t alphaFunc;          // GL compare 0x200..0x207
    uint32_t alphaRef;            // reference value (0..255 packed in low byte)
    uint32_t frontFace;           // 0x0900 CW, 0x0901 CCW
    uint32_t colorMask;           // ARGB bits, 0xFFFFFFFF = all writes enabled
    uint32_t shadeMode;           // 0x1D00 FLAT, 0x1D01 SMOOTH

    // Polygon offset (depth bias)
    bool     polyOffsetFillEnable;
    float    polyOffsetFactor;
    float    polyOffsetBias;

    // Logic op
    bool     logicOpEnable;
    uint32_t logicOp;             // GL logic op enum (0x1500..0x150F)

    // Dither
    bool     ditherEnable;

    // Two-sided stencil (back face)
    bool     twoSidedStencilEnable;
    // Two-sided color (back-face BFC0/BFC1)
    bool     twoSidedColorEnable;
    uint32_t backStencilFunc;
    uint32_t backStencilFuncRef;
    uint32_t backStencilFuncMask;
    uint32_t backStencilOpFail;
    uint32_t backStencilOpZFail;
    uint32_t backStencilOpZPass;
    uint32_t backStencilWriteMask;

    // Fog
    uint32_t fogMode;             // 0x2601=LINEAR, 0x0800=EXP, 0x0801=EXP2
    float    fogParam0;           // start (linear), density (exp/exp2)
    float    fogParam1;           // end (linear), unused (exp)

    // Depth bounds test
    bool     depthBoundsTestEnable;
    float    depthBoundsMin;
    float    depthBoundsMax;

    // Primitive restart
    bool     restartIndexEnable;
    uint32_t restartIndex;

    // User clip plane control (6 planes, 2 bits each: 0=disable, 1=LT, 2=GE)
    uint32_t userClipPlaneControl;

    // Vertex arrays (16 slots)
    struct VertexArray {
        uint32_t offset;
        uint32_t format;    // type | (size << 4) | (stride << 8)
        bool     enabled;
    } vertexArrays[16];
    uint16_t activeVAMask;  // bitmask of enabled vertex arrays

    // Inline vertex data (SET_VERTEX_DATA4F_M): per-attribute constant fallback
    float    vertexData4f[16][4];  // 16 attrs × 4 floats

    // Vertex program (transform program)
    uint32_t vpStart;           // start instruction index
    uint32_t vpData[512 * 4];  // up to 512 instructions × 4 words each
    uint32_t vpLoadOffset;      // current program upload offset (word index)
    uint32_t vpValid;           // set when a vertex program has been uploaded

    // Fragment program
    uint32_t fpOffset;   // VRAM offset of fragment program
    uint32_t fpControl;
    uint32_t shaderWindow;  // bits [15:0]=height, [16]=origin(0=top,1=bottom), [17]=pixCenter

    // Line / point rendering state
    float    pointSize;         // default 1.0 (float register)
    uint32_t lineWidth;         // fixed 8.3 format → actual = lineWidth / 8.0
    uint32_t pointSpriteCtrl;   // bit 0 = enable, bits [8:15] = tex coord mask

    // Additional mask/config state
    uint32_t colorMaskMrt;         // per-MRT (1-3) color write masks
    uint32_t vpAttribOutputMask;   // which VP outputs are active
    uint32_t freqDividerOp;        // vertex attrib instancing dividers
    uint32_t instanceCount;        // instanced draw repeat count (0=disabled)
    bool     sRGBWrite;            // gamma encode on FB write

    // Anti-aliasing / smoothing
    uint32_t antiAliasingControl;   // MSAA hint (no-op in software raster)
    bool     lineSmoothEnable;
    bool     polySmoothEnable;

    // ZCULL / S-CULL
    uint32_t zcullStatsEnable;
    uint32_t scullControl;

    // Misc pipeline
    uint32_t shaderPacker;
    uint32_t flatShadeOp;
    uint32_t vertexAttribInputMask;
    uint32_t reduceDstColor;

    // Transform constants (up to 468 × vec4 on NV47; round up to 512)
    float    vpConstants[512][4];
    uint32_t vpConstantLoad;    // current constant upload base index (vec4 index)

    // Textures (16 units)
    struct TextureUnit {
        uint32_t offset;
        uint32_t format;
        uint32_t width, height, depth;
        uint32_t control0;
        uint32_t address;   // wrap modes: wrapS(4)@0 | wrapT(4)@8 | wrapR(4)@16
        uint32_t filter;    // min/mag filter: min(16)@0 | mag(16)@16
        uint32_t borderColor;
        uint32_t control1;  // remap/swizzle
        uint8_t  dimension; // 1=1D, 2=2D, 3=3D, 6=cubemap
        bool     enabled;
        bool     dirty;     // set when offset/format/dimensions change
    } textures[16];

    // Draw state
    PrimitiveType currentPrim;
    bool          inBeginEnd;
    std::vector<uint32_t> inlineVertexData;  // NV4097_DRAW_INLINE_ARRAY accumulator

    // Statistics
    uint32_t drawCallCount;
    uint32_t triangleCount;
    uint32_t cmdCount;
    uint32_t frameCount;
    uint32_t unknownMethodCount;  // unhandled NV4097 methods

    // Optional Vulkan emitter (host-side; null in CUDA contexts).
    // Type-erased as void* so this header stays includable from both
    // .cu and host .cpp. Populated via rsx_set_vulkan_emitter().
    void*    vulkanEmitter;

    // Actual VRAM allocation size (set by rsx_process_fifo).
    uint32_t vramSize;

    // Semaphore / label system (SPU↔RSX synchronization).
    // SPUs poll a VRAM location; RSX writes a value there when done.
    uint32_t semaphoreOffset;   // VRAM byte offset for next semaphore write

    // NV406E subchannel semaphore (used by cellGcmSetWriteBackEndLabel etc.)
    uint32_t labelOffset;       // VRAM byte offset for label write
    uint32_t labelValue;        // value to write/compare

    // NV0039 DMA buffer copy state (accumulated from method writes).
    struct DmaTransfer {
        uint32_t offsetIn;
        uint32_t offsetOut;
        uint32_t pitchIn;
        uint32_t pitchOut;
        uint32_t lineLength;
        uint32_t lineCount;
        uint32_t format;       // bits [7:0]=in, [15:8]=out (1=byte, 2=LE16, 4=LE32)
        uint32_t ctxIn;        // context DMA for source (0xFEED0000=VRAM, 0xFEED0001=main)
        uint32_t ctxOut;       // context DMA for destination
    } dmaTransfer;

    // NV3062 destination surface for 2D blit.
    struct Blit2DSurface {
        uint32_t colorFormat;  // 0x0A = A8R8G8B8
        uint32_t pitch;        // bytes per row (src<<16 | dst)
        uint32_t dstOffset;    // VRAM offset of destination
    } blit2DSurface;

    // NV3089 scaled image from memory state.
    struct ScaledImage {
        uint32_t operation;    // 3=SRCCOPY
        uint32_t colorFormat;  // 0x0A = A8R8G8B8
        uint32_t clipX, clipY, clipW, clipH;
        uint32_t outX, outY, outW, outH;
        uint32_t dsDx, dtDy;  // 20.12 fixed-point scale factors
        uint32_t inW, inH;
        uint32_t inPitch;      // bits [31:16]=pitch, bits [15:0]=origin
        uint32_t inOffset;     // VRAM offset of source
    } scaledImage;

    // Occlusion query (ZCULL) state.
    bool     zcullEnable;       // zpass pixel count enabled
    uint32_t zcullPixelCount;   // accumulated pixel count since last clear

    // Conditional rendering (skip draw if occlusion query reports 0 pixels).
    bool     conditionalRenderEnable;  // NV4097_SET_RENDER_ENABLE mode=2 (conditional)
    uint32_t conditionalRenderOffset;  // VRAM offset of occlusion report to check

    // Depth buffer base offset in VRAM
    uint32_t zetaOffset;

    // DMA context IDs (used to select VRAM region for read/write)
    uint32_t contextDmaA;        // usually local VRAM
    uint32_t contextDmaB;        // usually main memory
    uint32_t contextDmaColorB;   // color buffer B
    uint32_t contextDmaState;    // report/notify region
    uint32_t contextDmaZeta;     // depth buffer region

    // Polygon rasterization mode (0x1B00=POINT, 0x1B01=LINE, 0x1B02=FILL)
    uint32_t frontPolygonMode;   // default FILL
    uint32_t backPolygonMode;    // default FILL

    // Surface depth pitch (bytes per row for Z buffer)
    uint32_t surfacePitchZ;

    // Zstencil clear value (depth[23:0] | stencil[31:24])
    uint32_t zstencilClearValue;
};

// ═══════════════════════════════════════════════════════════════════
// FIFO Command Header Parsing (NV47 format)
// ═══════════════════════════════════════════════════════════════════

struct FIFOCommand {
    uint32_t method;       // register / method offset
    uint32_t subchannel;   // 0-7
    uint32_t count;        // number of data words following
    bool     isJump;
    bool     isCall;
    bool     isReturn;
    bool     isNonIncr;    // non-incrementing method
    uint32_t jumpTarget;   // for jump / call
};

// NV47 header encoding:
//   Bits [31:29] = type (0 = incrementing, 2 = non-incrementing)
//   Bits [28:18] = count
//   Bits [17:13] = subchannel
//   Bits [12:2]  = method offset >> 2
//   Bits [1:0]   = 0 for normal methods
inline FIFOCommand parseFIFOHeader(uint32_t header) {
    FIFOCommand cmd = {};

    // Old-style jump (bit 29 set, bits [1:0] == 0)
    if ((header & 0x20000003) == 0x20000000) {
        cmd.isJump = true;
        cmd.jumpTarget = header & 0x1FFFFFFC;
        return cmd;
    }
    // New-style jump (bits [1:0] == 1)
    if ((header & 0x00000003) == 0x00000001) {
        cmd.isJump = true;
        cmd.jumpTarget = header & 0xFFFFFFFC;
        return cmd;
    }
    // Call (bits [1:0] == 2)
    if ((header & 0x00000003) == 0x00000002) {
        cmd.isCall = true;
        cmd.jumpTarget = header & 0xFFFFFFFC;
        return cmd;
    }
    // Return
    if (header == 0x00020000) {
        cmd.isReturn = true;
        return cmd;
    }

    // Normal method header
    uint32_t type  = (header >> 29) & 0x7;
    cmd.count      = (header >> 18) & 0x7FF;
    cmd.subchannel = (header >> 13) & 0x1F;
    cmd.method     = ((header >> 2) & 0x7FF) << 2;
    cmd.isNonIncr  = (type == 2);

    return cmd;
}

} // namespace rsx
