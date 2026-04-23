#pragma once
// rsx_defs.h — RSX Reality Synthesizer definitions
//
// NV47-class GPU (G70/GeForce 7800 GTX derivative) as used in PlayStation 3.
// Defines command FIFO format, NV4097 3D methods, and GPU state structures.

#include <cstdint>

namespace rsx {

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
// ═══════════════════════════════════════════════════════════════════

// Object / control
static constexpr uint32_t NV4097_SET_OBJECT              = 0x00000000;
static constexpr uint32_t NV4097_NO_OPERATION             = 0x00000100;
static constexpr uint32_t NV4097_SET_CONTEXT_DMA_NOTIFIES = 0x00000180;
static constexpr uint32_t NV4097_SET_CONTEXT_DMA_COLOR_A  = 0x00000184;

// Surface (render target) configuration
static constexpr uint32_t NV4097_SET_SURFACE_FORMAT       = 0x00000200;
static constexpr uint32_t NV4097_SET_SURFACE_CLIP_HORIZONTAL = 0x00000204;
static constexpr uint32_t NV4097_SET_SURFACE_CLIP_VERTICAL   = 0x00000208;
static constexpr uint32_t NV4097_SET_SURFACE_PITCH_A         = 0x0000020C;
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_AOFFSET   = 0x00000210;
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_BOFFSET   = 0x00000214;
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_TARGET    = 0x00000228;

// Viewport / Scissor
static constexpr uint32_t NV4097_SET_VIEWPORT_HORIZONTAL  = 0x00000A00;
static constexpr uint32_t NV4097_SET_VIEWPORT_VERTICAL    = 0x00000A04;
static constexpr uint32_t NV4097_SET_SCISSOR_HORIZONTAL   = 0x00000A08;
static constexpr uint32_t NV4097_SET_SCISSOR_VERTICAL     = 0x00000A0C;

// Depth / stencil state
static constexpr uint32_t NV4097_SET_DEPTH_FUNC               = 0x00000300;
static constexpr uint32_t NV4097_SET_DEPTH_TEST_ENABLE        = 0x00000304;
static constexpr uint32_t NV4097_SET_DEPTH_MASK               = 0x0000030C;
static constexpr uint32_t NV4097_SET_BLEND_ENABLE             = 0x00000310;
static constexpr uint32_t NV4097_SET_BLEND_FUNC_SFACTOR       = 0x00000314;
static constexpr uint32_t NV4097_SET_BLEND_FUNC_DFACTOR       = 0x00000318;
static constexpr uint32_t NV4097_SET_BLEND_COLOR              = 0x0000031C;
static constexpr uint32_t NV4097_SET_BLEND_EQUATION           = 0x00000320;
static constexpr uint32_t NV4097_SET_STENCIL_TEST_ENABLE      = 0x0000032C;
static constexpr uint32_t NV4097_SET_STENCIL_MASK             = 0x00000330;
static constexpr uint32_t NV4097_SET_STENCIL_FUNC             = 0x00000334;
static constexpr uint32_t NV4097_SET_STENCIL_FUNC_REF         = 0x00000338;
static constexpr uint32_t NV4097_SET_STENCIL_FUNC_MASK        = 0x0000033C;
static constexpr uint32_t NV4097_SET_STENCIL_OP_FAIL          = 0x00000340;
static constexpr uint32_t NV4097_SET_STENCIL_OP_ZFAIL         = 0x00000344;
static constexpr uint32_t NV4097_SET_STENCIL_OP_ZPASS         = 0x00000348;

// Cull face
static constexpr uint32_t NV4097_SET_CULL_FACE_ENABLE         = 0x00000B44;
static constexpr uint32_t NV4097_SET_CULL_FACE                = 0x00000B48;
static constexpr uint32_t NV4097_SET_FRONT_FACE               = 0x00000B4C;

// Alpha test / color mask / shade mode (Phase 4b)
static constexpr uint32_t NV4097_SET_ALPHA_TEST_ENABLE        = 0x00000300 + 0x6C; // 0x36C
static constexpr uint32_t NV4097_SET_ALPHA_FUNC               = 0x00000300 + 0x70; // 0x370
static constexpr uint32_t NV4097_SET_ALPHA_REF                = 0x00000300 + 0x74; // 0x374
static constexpr uint32_t NV4097_SET_COLOR_MASK               = 0x00000358;
static constexpr uint32_t NV4097_SET_SHADE_MODE               = 0x00000368;

// Shader programs
static constexpr uint32_t NV4097_SET_SHADER_PROGRAM          = 0x000008E4;

// Transform program (vertex shader)
// Real NV47 layout:
//   0x1E94 TRANSFORM_EXECUTION_MODE
//   0x1E9C TRANSFORM_PROGRAM_LOAD   (write-pointer index for 0x0B80 window)
//   0x1EA0 TRANSFORM_PROGRAM_START  (execution entry point)
//   0x1EFC TRANSFORM_CONSTANT_LOAD  (write-pointer index for 0x1F00 window)
//   0x1F00 TRANSFORM_CONSTANT       (256 vec4 constant window)
static constexpr uint32_t NV4097_SET_TRANSFORM_PROGRAM        = 0x00000B80;
static constexpr uint32_t NV4097_SET_TRANSFORM_PROGRAM_LOAD   = 0x00001E9C;
static constexpr uint32_t NV4097_SET_TRANSFORM_CONSTANT_LOAD  = 0x00001EFC;
static constexpr uint32_t NV4097_SET_TRANSFORM_CONSTANT       = 0x00001F00;

// Vertex arrays (16 slots, stride 4 per slot)
static constexpr uint32_t NV4097_SET_VERTEX_DATA_ARRAY_OFFSET = 0x00001680;
static constexpr uint32_t NV4097_SET_VERTEX_DATA_ARRAY_FORMAT = 0x00001740;

// Surface state — full MRT (up to 4 color planes).
// Layout on real NV47:
//   0x0208 PITCH_A   0x020C CLIP_V        0x0210 COLOR_AOFFSET
//   0x0214 COLOR_BOFFSET                 0x0218 COLOR_BOFFSET? (varies)
//   0x022C COLOR_COFFSET                 0x0230 COLOR_DOFFSET
//   0x0234 PITCH_B   0x0238 PITCH_C      0x023C PITCH_D
// (Approximate — we follow the method indices from NV4097 docs and
// RPCS3's gcm header.)
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_COFFSET   = 0x0000022C;
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_DOFFSET   = 0x00000230;
static constexpr uint32_t NV4097_SET_SURFACE_PITCH_B         = 0x00000234;
static constexpr uint32_t NV4097_SET_SURFACE_PITCH_C         = 0x00000238;
static constexpr uint32_t NV4097_SET_SURFACE_PITCH_D         = 0x0000023C;

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

// Draw commands
static constexpr uint32_t NV4097_SET_BEGIN_END               = 0x00001808;
static constexpr uint32_t NV4097_DRAW_ARRAYS               = 0x00001814;
static constexpr uint32_t NV4097_DRAW_INDEX_ARRAY           = 0x0000181C;

// Index buffer state. (Real RSX places these at 0x1A0/0x1A4 inside the
// texture range; we use a non-conflicting slot since our texture range
// already starts at 0x1A00. Games never see these registers directly —
// HLE wrappers translate cellGcmSetDrawIndexArray into our offsets.)
static constexpr uint32_t NV4097_SET_INDEX_ARRAY_ADDRESS     = 0x00001828;
static constexpr uint32_t NV4097_SET_INDEX_ARRAY_DMA         = 0x0000182C;

// Texture (16 units, stride 0x20 per unit)
static constexpr uint32_t NV4097_SET_TEXTURE_OFFSET          = 0x00001A00;
static constexpr uint32_t NV4097_SET_TEXTURE_FORMAT           = 0x00001A04;
static constexpr uint32_t NV4097_SET_TEXTURE_CONTROL0         = 0x00001A0C;
static constexpr uint32_t NV4097_SET_TEXTURE_IMAGE_RECT       = 0x00001A18;

// Clear / present
static constexpr uint32_t NV4097_SET_COLOR_CLEAR_VALUE       = 0x00001D90;
static constexpr uint32_t NV4097_CLEAR_SURFACE               = 0x00001D94;

// Transform program start address (real addr 0x1EA0)
static constexpr uint32_t NV4097_SET_TRANSFORM_PROGRAM_START = 0x00001EA0;

// Flip (surface A offset for display flip)
static constexpr uint32_t NV4097_SET_SURFACE_COLOR_AOFFSET_FLIP = 0x00000E20;

// Reference register
static constexpr uint32_t NV4097_SET_REFERENCE               = 0x00000050;

// ═══════════════════════════════════════════════════════════════════
// Enumerations
// ═══════════════════════════════════════════════════════════════════

// Primitive types
enum PrimitiveType : uint32_t {
    PRIM_POINTS         = 1,
    PRIM_LINES          = 2,
    PRIM_LINE_STRIP     = 3,
    PRIM_TRIANGLES      = 4,
    PRIM_TRIANGLE_STRIP = 5,
    PRIM_TRIANGLE_FAN   = 6,
    PRIM_QUADS          = 7,
    PRIM_QUAD_STRIP     = 8,
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
static constexpr uint32_t CLEAR_COLOR   = 0x01;
static constexpr uint32_t CLEAR_DEPTH   = 0x02;
static constexpr uint32_t CLEAR_STENCIL = 0x04;

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

    // Vertex arrays (16 slots)
    struct VertexArray {
        uint32_t offset;
        uint32_t format;    // type | (size << 4) | (stride << 8)
        bool     enabled;
    } vertexArrays[16];

    // Vertex program (transform program)
    uint32_t vpStart;           // start instruction index
    uint32_t vpData[512 * 4];  // up to 512 instructions × 4 words each
    uint32_t vpLoadOffset;      // current program upload offset (word index)
    uint32_t vpValid;           // set when a vertex program has been uploaded

    // Fragment program
    uint32_t fpOffset;   // VRAM offset of fragment program
    uint32_t fpControl;

    // Transform constants (up to 468 × vec4 on NV47; round up to 512)
    float    vpConstants[512][4];
    uint32_t vpConstantLoad;    // current constant upload base index (vec4 index)

    // Textures (16 units)
    struct TextureUnit {
        uint32_t offset;
        uint32_t format;
        uint32_t width, height;
        uint32_t control0;
        bool     enabled;
    } textures[16];

    // Draw state
    PrimitiveType currentPrim;
    bool          inBeginEnd;

    // Statistics
    uint32_t drawCallCount;
    uint32_t triangleCount;
    uint32_t cmdCount;
    uint32_t frameCount;

    // Optional Vulkan emitter (host-side; null in CUDA contexts).
    // Type-erased as void* so this header stays includable from both
    // .cu and host .cpp. Populated via rsx_set_vulkan_emitter().
    void*    vulkanEmitter;
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
