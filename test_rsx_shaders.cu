// test_rsx_shaders.cu — Tests for RSX VP→CUDA and FP→GLSL translators
//
// Validates instruction decoding, code generation, and VP kernel execution
// on V100 via NVRTC.

#include "rsx_defs.h"
#include "rsx_vp_shader.h"
#include "rsx_fp_shader.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda.h>
#include <nvrtc.h>
#include <string>
#include <chrono>

// ═══════════════════════════════════════════════════════════════════
// Test framework
// ═══════════════════════════════════════════════════════════════════

static int g_pass = 0, g_fail = 0;

#define TEST(name) \
    printf("\n[TEST] %s\n", name);

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [  OK]  %s\n", msg); } \
    else      { g_fail++; printf("  [FAIL]  %s\n", msg); } \
} while(0)

// ═══════════════════════════════════════════════════════════════════
// Helper: build NV40 VP instruction manually
// ═══════════════════════════════════════════════════════════════════

// Encode a source operand into 17-bit SRC format
static uint32_t encodeSrc(uint32_t regType, uint32_t regIdx,
                           uint32_t swzX, uint32_t swzY, uint32_t swzZ, uint32_t swzW,
                           bool neg = false) {
    uint32_t s = 0;
    s |= (regType & 0x3);
    s |= (regIdx & 0x3F) << 2;
    s |= (swzW & 0x3) << 8;
    s |= (swzZ & 0x3) << 10;
    s |= (swzY & 0x3) << 12;
    s |= (swzX & 0x3) << 14;
    s |= (neg ? 1 : 0) << 16;
    return s;
}

// Build a VP instruction: vec MOV from input to output
static void buildVPInsn(uint32_t d[4],
                         uint32_t vecOp, uint32_t scaOp,
                         uint32_t inputIdx, uint32_t constIdx,
                         uint32_t src0, uint32_t src1, uint32_t src2,
                         uint32_t vecDstTmp, uint32_t vecDstOut,
                         bool vecResult,
                         bool maskX, bool maskY, bool maskZ, bool maskW,
                         bool end = false) {
    // D0
    uint32_t d0 = 0;
    d0 |= (vecDstTmp & 0x3F) << 14;   // dst_tmp
    if (vecResult) d0 |= (1 << 29);    // vec_result

    // D1
    uint32_t d1 = 0;
    uint32_t src0h = (src0 >> 9) & 0xFF;
    d1 |= src0h;                        // src0h
    d1 |= (inputIdx & 0xF) << 8;       // input_src
    d1 |= (constIdx & 0x3FF) << 12;    // const_src
    d1 |= (vecOp & 0x1F) << 22;        // vec_opcode
    d1 |= (scaOp & 0x1F) << 27;        // sca_opcode

    // D2
    uint32_t d2 = 0;
    uint32_t src2h = (src2 >> 11) & 0x3F;
    d2 |= src2h;                         // src2h
    d2 |= (src1 & 0x1FFFF) << 6;       // src1
    uint32_t src0l = src0 & 0x1FF;
    d2 |= src0l << 23;                  // src0l

    // D3
    uint32_t d3 = 0;
    if (end) d3 |= 1;                   // end flag
    d3 |= (vecDstOut & 0x1F) << 2;     // dst output
    // Write masks (1 = enabled)
    if (maskW) d3 |= (1 << 13);
    if (maskZ) d3 |= (1 << 14);
    if (maskY) d3 |= (1 << 15);
    if (maskX) d3 |= (1 << 16);
    uint32_t src2l = src2 & 0x7FF;
    d3 |= src2l << 21;                  // src2l

    d[0] = d0; d[1] = d1; d[2] = d2; d[3] = d3;
}

// ═══════════════════════════════════════════════════════════════════
// Test 1: VP Instruction Decode
// ═══════════════════════════════════════════════════════════════════

void test_vp_decode() {
    TEST("VP Instruction Decode");

    // Build: VEC MOV from input[0] → output[0] (HPOS), all channels
    uint32_t src0 = encodeSrc(rsx::VP_REG_INPUT, 0, 0,1,2,3);
    uint32_t src1 = encodeSrc(rsx::VP_REG_TEMP, 0, 0,1,2,3);
    uint32_t src2 = encodeSrc(rsx::VP_REG_TEMP, 0, 0,1,2,3);

    uint32_t d[4];
    buildVPInsn(d, rsx::VP_VEC_MOV, rsx::VP_SCA_NOP,
                0, 0, src0, src1, src2,
                0, 0, true, true, true, true, true, true);

    rsx::VPDecodedInsn insn = rsx::vp_decode(d);

    CHECK(insn.vecOp == rsx::VP_VEC_MOV, "vec opcode = MOV");
    CHECK(insn.scaOp == rsx::VP_SCA_NOP, "sca opcode = NOP");
    CHECK(insn.vecWriteResult == true, "writes to output");
    CHECK(insn.vecDstOut == 0, "output reg = 0 (HPOS)");
    CHECK(insn.vecMaskX && insn.vecMaskY && insn.vecMaskZ && insn.vecMaskW,
          "write mask = xyzw");
    CHECK(insn.endFlag == true, "end flag set");
    CHECK(insn.src[0].regType == rsx::VP_REG_INPUT, "src0 = input");
    CHECK(insn.inputIdx == 0, "input index = 0");
}

// ═══════════════════════════════════════════════════════════════════
// Test 2: VP Code Generation
// ═══════════════════════════════════════════════════════════════════

void test_vp_codegen() {
    TEST("VP Code Generation");

    // Build a simple 3-instruction VP:
    //   MOV o0, in0          (position passthrough)
    //   MUL t0, in0, c0      (scale by constant)
    //   MAD o1, in1, c1, t0  (color transform)
    uint32_t prog[12];

    // insn 0: MOV output[0] = input[0]
    uint32_t s0_in0 = encodeSrc(rsx::VP_REG_INPUT, 0, 0,1,2,3);
    uint32_t s_zero = encodeSrc(rsx::VP_REG_TEMP, 0, 0,1,2,3);
    buildVPInsn(&prog[0], rsx::VP_VEC_MOV, rsx::VP_SCA_NOP,
                0, 0, s0_in0, s_zero, s_zero,
                0, 0, true, true, true, true, true);

    // insn 1: MUL t0 = input[0] * const[0]
    uint32_t s_c0 = encodeSrc(rsx::VP_REG_CONSTANT, 0, 0,1,2,3);
    buildVPInsn(&prog[4], rsx::VP_VEC_MUL, rsx::VP_SCA_NOP,
                0, 0, s0_in0, s_c0, s_zero,
                0, 0, false, true, true, true, true);

    // insn 2: MAD output[1] = input[1] * const[1] + t0  [END]
    uint32_t s_in1 = encodeSrc(rsx::VP_REG_INPUT, 1, 0,1,2,3);
    uint32_t s_c1 = encodeSrc(rsx::VP_REG_CONSTANT, 1, 0,1,2,3);
    uint32_t s_t0 = encodeSrc(rsx::VP_REG_TEMP, 0, 0,1,2,3);
    buildVPInsn(&prog[8], rsx::VP_VEC_MAD, rsx::VP_SCA_NOP,
                1, 1, s_in1, s_c1, s_t0,
                0, 1, true, true, true, true, true, true);

    // Translate
    std::string cuda_src = rsx::vp_translate_to_cuda(prog, 12);

    CHECK(cuda_src.find("rsx_vp_kernel") != std::string::npos, "kernel name present");
    CHECK(cuda_src.find("f4mul") != std::string::npos, "MUL emitted");
    CHECK(cuda_src.find("f4add(f4mul(") != std::string::npos, "MAD emitted as add(mul())");
    CHECK(cuda_src.find("out_attr[0]") != std::string::npos, "output[0] written");
    CHECK(cuda_src.find("out_attr[1]") != std::string::npos, "output[1] written");
    CHECK(cuda_src.find("in_attr[") != std::string::npos, "input attrs loaded");
    CHECK(cuda_src.find("float4 t0") != std::string::npos, "temp register declared");

    printf("  Generated %zu bytes of CUDA source, %d instructions\n",
           cuda_src.size(), 3);
}

// ═══════════════════════════════════════════════════════════════════
// Test 3: VP NVRTC Compilation
// ═══════════════════════════════════════════════════════════════════

void test_vp_nvrtc_compile() {
    TEST("VP NVRTC Compilation");

    // Simple passthrough VP
    uint32_t prog[4];
    uint32_t s0 = encodeSrc(rsx::VP_REG_INPUT, 0, 0,1,2,3);
    uint32_t sz = encodeSrc(rsx::VP_REG_TEMP, 0, 0,1,2,3);
    buildVPInsn(prog, rsx::VP_VEC_MOV, rsx::VP_SCA_NOP,
                0, 0, s0, sz, sz,
                0, 0, true, true, true, true, true, true);

    std::string src = rsx::vp_translate_to_cuda(prog, 4);

    // Compile with NVRTC
    nvrtcProgram nvprog;
    nvrtcResult res = nvrtcCreateProgram(&nvprog, src.c_str(), "rsx_vp.cu", 0, NULL, NULL);
    CHECK(res == NVRTC_SUCCESS, "NVRTC program created");

    const char* opts[] = {"--gpu-architecture=compute_70", "-default-device"};
    res = nvrtcCompileProgram(nvprog, 2, opts);

    if (res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(nvprog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(nvprog, log);
        printf("  NVRTC compile log:\n%s\n", log);
        delete[] log;
    }
    CHECK(res == NVRTC_SUCCESS, "NVRTC compilation succeeded");

    if (res == NVRTC_SUCCESS) {
        size_t ptxSize;
        nvrtcGetPTXSize(nvprog, &ptxSize);
        CHECK(ptxSize > 100, "PTX generated (size > 100 bytes)");
        printf("  PTX size: %zu bytes\n", ptxSize);
    }

    nvrtcDestroyProgram(&nvprog);
}

// ═══════════════════════════════════════════════════════════════════
// Test 4: VP Kernel Execution on GPU
// ═══════════════════════════════════════════════════════════════════

void test_vp_gpu_execution() {
    TEST("VP GPU Execution");

    // Build VP: MAD output[0] = input[0] * const[0] + const[1]
    uint32_t prog[4];
    uint32_t s_in = encodeSrc(rsx::VP_REG_INPUT, 0, 0,1,2,3);
    uint32_t s_c0 = encodeSrc(rsx::VP_REG_CONSTANT, 0, 0,1,2,3);
    uint32_t s_c1 = encodeSrc(rsx::VP_REG_CONSTANT, 1, 0,1,2,3);
    buildVPInsn(prog, rsx::VP_VEC_MAD, rsx::VP_SCA_NOP,
                0, 0, s_in, s_c0, s_c1,
                0, 0, true, true, true, true, true, true);

    std::string src = rsx::vp_translate_to_cuda(prog, 4);

    // Compile
    nvrtcProgram nvprog;
    nvrtcCreateProgram(&nvprog, src.c_str(), "rsx_vp.cu", 0, NULL, NULL);
    const char* opts[] = {"--gpu-architecture=compute_70", "-default-device"};
    nvrtcResult res = nvrtcCompileProgram(nvprog, 2, opts);

    if (res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(nvprog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(nvprog, log);
        printf("  Compile error:\n%s\n", log);
        delete[] log;
        nvrtcDestroyProgram(&nvprog);
        g_fail++;
        printf("  [FAIL]  compilation failed, skipping execution\n");
        return;
    }

    size_t ptxSize;
    nvrtcGetPTXSize(nvprog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(nvprog, ptx);
    nvrtcDestroyProgram(&nvprog);

    // Load via CUDA driver API
    CUdevice dev;
    CUcontext ctx;
    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    CUresult cr = cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL);
    CHECK(cr == CUDA_SUCCESS, "cuModuleLoad succeeded");
    delete[] ptx;

    CUfunction func;
    cr = cuModuleGetFunction(&func, mod, "rsx_vp_kernel");
    CHECK(cr == CUDA_SUCCESS, "kernel function found");

    // Setup data: 4 vertices, 1 input attrib each
    int numVerts = 4;
    int numAttribs = 1;
    float4 h_vertices[4] = {
        {1.0f, 2.0f, 3.0f, 1.0f},
        {4.0f, 5.0f, 6.0f, 1.0f},
        {-1.0f, 0.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    };

    // Constants: c[0] = {2,2,2,2}
    // MAD with single constIdx=0: input * c[0] + c[0] = {1*2+2, 2*2+2, 3*2+2, 1*2+2} = {4,6,8,4}
    float4 h_constants[256];
    memset(h_constants, 0, sizeof(h_constants));
    h_constants[0] = {2.0f, 2.0f, 2.0f, 2.0f};

    CUdeviceptr d_verts, d_output, d_consts;
    cuMemAlloc(&d_verts, sizeof(float4) * 4);
    cuMemAlloc(&d_output, sizeof(float4) * 4 * 17);
    cuMemAlloc(&d_consts, sizeof(float4) * 256);
    cuMemcpyHtoD(d_verts, h_vertices, sizeof(float4) * 4);
    cuMemcpyHtoD(d_consts, h_constants, sizeof(float4) * 256);

    void* args[] = {&d_verts, &d_output, &d_consts, &numVerts, &numAttribs};
    cr = cuLaunchKernel(func, 1, 1, 1, 4, 1, 1, 0, 0, args, NULL);
    CHECK(cr == CUDA_SUCCESS, "kernel launched");
    cuCtxSynchronize();

    // Read back
    float4 h_output[4 * 17];
    cuMemcpyDtoH(h_output, d_output, sizeof(float4) * 4 * 17);

    // Check vertex 0, output[0] (HPOS): 1*2+2=4, 2*2+2=6, 3*2+2=8, 1*2+2=4
    float4 hpos = h_output[0 * 17 + 0];
    printf("  Vertex 0 HPOS: (%.1f, %.1f, %.1f, %.1f)\n", hpos.x, hpos.y, hpos.z, hpos.w);
    CHECK(fabsf(hpos.x - 4.0f) < 0.01f && fabsf(hpos.y - 6.0f) < 0.01f &&
          fabsf(hpos.z - 8.0f) < 0.01f && fabsf(hpos.w - 4.0f) < 0.01f,
          "MAD result correct: input*2+2");

    // Check vertex 1: 4*2+2=10, 5*2+2=12, 6*2+2=14, 1*2+2=4
    float4 hpos1 = h_output[1 * 17 + 0];
    printf("  Vertex 1 HPOS: (%.1f, %.1f, %.1f, %.1f)\n", hpos1.x, hpos1.y, hpos1.z, hpos1.w);
    CHECK(fabsf(hpos1.x - 10.0f) < 0.01f && fabsf(hpos1.y - 12.0f) < 0.01f,
          "Vertex 1 MAD correct");

    cuMemFree(d_verts);
    cuMemFree(d_output);
    cuMemFree(d_consts);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
}

// ═══════════════════════════════════════════════════════════════════
// Test 5: VP Scalar Operations
// ═══════════════════════════════════════════════════════════════════

void test_vp_scalar_ops() {
    TEST("VP Scalar Operations");

    // Build: SCA RCP → t1.x (reciprocal of input[0].x)
    //        SCA RSQ → t2.x (rsqrt of input[0].x)
    uint32_t prog[8];
    uint32_t s_in = encodeSrc(rsx::VP_REG_INPUT, 0, 0,0,0,0); // .xxxx
    uint32_t sz   = encodeSrc(rsx::VP_REG_TEMP, 0, 0,1,2,3);

    // insn 0: SCA RCP t1
    buildVPInsn(&prog[0], rsx::VP_VEC_NOP, rsx::VP_SCA_RCP,
                0, 0, sz, sz, s_in,  // scalar uses src2
                0, 0, false, false, false, false, false);
    // Set scalar write mask in D3 manually
    prog[3] &= ~(0xF << 17);
    prog[3] |= (1 << 20);  // sca_mask_x = 1
    // Set sca_dst_tmp = 1
    prog[3] &= ~(0x3F << 7);
    prog[3] |= (1 << 7);

    // insn 1: SCA SIN t2 [END]
    buildVPInsn(&prog[4], rsx::VP_VEC_NOP, rsx::VP_SCA_SIN,
                0, 0, sz, sz, s_in,
                0, 0, false, false, false, false, false, true);
    prog[7] &= ~(0xF << 17);
    prog[7] |= (1 << 20);
    prog[7] &= ~(0x3F << 7);
    prog[7] |= (2 << 7);

    std::string src = rsx::vp_translate_to_cuda(prog, 8);

    CHECK(src.find("1.0f/(") != std::string::npos, "RCP emitted");
    CHECK(src.find("sinf(") != std::string::npos, "SIN emitted");
    CHECK(src.find("f4splat") != std::string::npos, "scalar splat used");
}

// ═══════════════════════════════════════════════════════════════════
// Test 6: FP Instruction Decode
// ═══════════════════════════════════════════════════════════════════

void test_fp_decode() {
    TEST("FP Instruction Decode");

    // Build a minimal FP instruction: MOV r0 = input[1] (COL0)
    // OPDEST: end=0, dest_reg=0, fp16=0, set_cond=0, mask=xyzw, src_attr=1, tex=0, prec=0, opcode=MOV(1), no_dest=0, sat=0
    uint32_t w0 = 0;
    w0 |= (0 & 0x3F) << 1;   // dest_reg = 0
    w0 |= (1 << 9);           // mask_x
    w0 |= (1 << 10);          // mask_y
    w0 |= (1 << 11);          // mask_z
    w0 |= (1 << 12);          // mask_w
    w0 |= (1 & 0xF) << 13;   // src_attr_reg = 1 (COL0)
    w0 |= (rsx::FP_MOV & 0x3F) << 24; // opcode

    // Apply FP byte swap (reverse of fp_swap_word)
    auto fp_encode = [](uint32_t val) -> uint32_t {
        return ((val & 0x00FF00FF) << 8) | ((val & 0xFF00FF00) >> 8);
    };

    // SRC0: reg_type=INPUT(1), reg_idx=1, swizzle=xyzw
    uint32_t w1 = 0;
    w1 |= rsx::FP_REG_INPUT;   // reg_type = 1
    w1 |= (1 << 2);            // reg_idx = 1
    w1 |= (0 << 9);            // swz_x = 0
    w1 |= (1 << 11);           // swz_y = 1
    w1 |= (2 << 13);           // swz_z = 2
    w1 |= (3 << 15);           // swz_w = 3

    uint32_t raw[4] = {fp_encode(w0), fp_encode(w1), fp_encode(0), fp_encode(0)};

    rsx::FPDecodedInsn insn = rsx::fp_decode(raw);

    CHECK(insn.opcode == rsx::FP_MOV, "opcode = MOV");
    CHECK(insn.dstReg == 0, "dest = r0");
    CHECK(insn.maskX && insn.maskY && insn.maskZ && insn.maskW, "mask = xyzw");
    CHECK(insn.inputAttr == 1, "input = COL0");
    CHECK(insn.src[0].regType == rsx::FP_REG_INPUT, "src0 = input");
}

// ═══════════════════════════════════════════════════════════════════
// Test 7: FP Code Generation
// ═══════════════════════════════════════════════════════════════════

void test_fp_codegen() {
    TEST("FP GLSL Code Generation");

    // We can't easily build real FP microcode by hand (big-endian + byte swap),
    // so test the emitter directly with a simple program
    rsx::FPEmitter emit;

    // Manually feed decoded instructions
    rsx::FPDecodedInsn mov = {};
    mov.opcode = rsx::FP_MOV;
    mov.dstReg = 0;
    mov.maskX = mov.maskY = mov.maskZ = mov.maskW = true;
    mov.src[0].regType = rsx::FP_REG_INPUT;
    mov.src[0].regIdx = 1;
    mov.src[0].swzX = 0; mov.src[0].swzY = 1;
    mov.src[0].swzZ = 2; mov.src[0].swzW = 3;
    mov.inputAttr = 1;
    emit.emitInsn(mov);

    rsx::FPDecodedInsn mul = {};
    mul.opcode = rsx::FP_MUL;
    mul.dstReg = 1;
    mul.maskX = mul.maskY = mul.maskZ = mul.maskW = true;
    mul.src[0].regType = rsx::FP_REG_TEMP;
    mul.src[0].regIdx = 0;
    mul.src[0].swzX = 0; mul.src[0].swzY = 1;
    mul.src[0].swzZ = 2; mul.src[0].swzW = 3;
    mul.src[1].regType = rsx::FP_REG_CONSTANT;
    mul.src[1].regIdx = 0;
    mul.src[1].swzX = 0; mul.src[1].swzY = 1;
    mul.src[1].swzZ = 2; mul.src[1].swzW = 3;
    mul.inputAttr = 0;
    emit.emitInsn(mul);

    rsx::FPDecodedInsn tex = {};
    tex.opcode = rsx::FP_TEX;
    tex.dstReg = 2;
    tex.maskX = tex.maskY = tex.maskZ = tex.maskW = true;
    tex.texUnit = 3;
    tex.src[0].regType = rsx::FP_REG_TEMP;
    tex.src[0].regIdx = 0;
    tex.src[0].swzX = 0; tex.src[0].swzY = 1;
    tex.src[0].swzZ = 2; tex.src[0].swzW = 3;
    tex.inputAttr = 0;
    emit.emitInsn(tex);

    CHECK(emit.code.find("v_col0") != std::string::npos, "input COL0 referenced");
    CHECK(emit.code.find("*") != std::string::npos, "MUL generated");
    CHECK(emit.code.find("texture(tex3") != std::string::npos, "TEX unit 3 sampled");
    CHECK(emit.texUnitsMask & (1 << 3), "tex unit 3 tracked");
    CHECK(emit.instrCount == 3, "3 instructions emitted");

    printf("  Generated GLSL body: %zu chars\n", emit.code.size());
}

// ═══════════════════════════════════════════════════════════════════
// Test 8: FP Full GLSL Translation
// ═══════════════════════════════════════════════════════════════════

void test_fp_full_glsl() {
    TEST("FP Full GLSL 450 Output");

    // Build minimal FP via encoded words:
    // Just a MOV r0 = input[1] with end flag
    auto fp_encode = [](uint32_t val) -> uint32_t {
        return ((val & 0x00FF00FF) << 8) | ((val & 0xFF00FF00) >> 8);
    };

    uint32_t w0 = 0;
    w0 |= 1;                           // end = 1
    w0 |= (0 & 0x3F) << 1;            // dest_reg = 0
    w0 |= (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12);  // mask xyzw
    w0 |= (1 & 0xF) << 13;            // src_attr = 1
    w0 |= (rsx::FP_MOV & 0x3F) << 24; // opcode

    uint32_t w1 = 0;
    w1 |= rsx::FP_REG_INPUT;
    w1 |= (1 << 2);
    w1 |= (0 << 9) | (1 << 11) | (2 << 13) | (3 << 15);

    uint32_t raw[4] = {fp_encode(w0), fp_encode(w1), fp_encode(0), fp_encode(0)};

    std::string glsl = rsx::fp_translate_to_glsl(raw, 4);

    CHECK(glsl.find("#version 450") != std::string::npos, "GLSL 450 header");
    CHECK(glsl.find("void main()") != std::string::npos, "main function");
    CHECK(glsl.find("fragColor = r0") != std::string::npos, "output = r0");
    CHECK(glsl.find("v_col0") != std::string::npos, "COL0 input declared");
    CHECK(glsl.find("layout(location = 0) out vec4 fragColor") != std::string::npos,
          "fragment output declared");

    printf("  Full GLSL: %zu chars\n", glsl.size());
}

// ═══════════════════════════════════════════════════════════════════
// Test 9: VP Disassembler
// ═══════════════════════════════════════════════════════════════════

void test_vp_disassemble() {
    TEST("VP Disassembler");

    uint32_t prog[8];
    uint32_t s_in = encodeSrc(rsx::VP_REG_INPUT, 0, 0,1,2,3);
    uint32_t s_c  = encodeSrc(rsx::VP_REG_CONSTANT, 0, 0,1,2,3);
    uint32_t sz   = encodeSrc(rsx::VP_REG_TEMP, 0, 0,1,2,3);

    buildVPInsn(&prog[0], rsx::VP_VEC_DP4, rsx::VP_SCA_NOP,
                0, 0, s_in, s_c, sz,
                0, 0, true, true, true, true, true);
    buildVPInsn(&prog[4], rsx::VP_VEC_MOV, rsx::VP_SCA_RCP,
                0, 0, s_in, sz, s_in,
                1, 1, true, true, true, true, true, true);

    printf("  Disassembly:\n  ");
    rsx::vp_disassemble(prog, 8);
    CHECK(true, "disassembly completed without crash");
}

// ═══════════════════════════════════════════════════════════════════
// Test 10: VP Throughput Benchmark
// ═══════════════════════════════════════════════════════════════════

void test_vp_throughput() {
    TEST("VP Throughput Benchmark");

    // Build a 4-instruction VP (typical simple shader)
    uint32_t prog[16];
    uint32_t s_in = encodeSrc(rsx::VP_REG_INPUT, 0, 0,1,2,3);
    uint32_t s_c0 = encodeSrc(rsx::VP_REG_CONSTANT, 0, 0,1,2,3);
    uint32_t sz   = encodeSrc(rsx::VP_REG_TEMP, 0, 0,1,2,3);

    buildVPInsn(&prog[0], rsx::VP_VEC_MUL, rsx::VP_SCA_NOP,
                0, 0, s_in, s_c0, sz, 0, 0, false, 1,1,1,1);
    buildVPInsn(&prog[4], rsx::VP_VEC_ADD, rsx::VP_SCA_NOP,
                0, 0, sz, sz, s_c0, 1, 0, false, 1,1,1,1);
    buildVPInsn(&prog[8], rsx::VP_VEC_MAD, rsx::VP_SCA_NOP,
                0, 0, s_in, s_c0, sz, 0, 0, true, 1,1,1,1);
    buildVPInsn(&prog[12], rsx::VP_VEC_MOV, rsx::VP_SCA_NOP,
                0, 0, sz, sz, sz, 0, 1, true, 1,1,1,1, true);

    // Measure translation speed
    int iters = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        std::string s = rsx::vp_translate_to_cuda(prog, 16);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    double rate = iters / (ms / 1000.0);
    printf("  Translation: %.0f VP/sec (%.2f ms for %d iterations)\n", rate, ms, iters);
    CHECK(rate > 1000, "translation > 1000 VP/sec");
}

// ═══════════════════════════════════════════════════════════════════

int main() {
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║  RSX Shader Translator Tests                     ║\n");
    printf("╠═══════════════════════════════════════════════════╣\n");

    test_vp_decode();
    test_vp_codegen();
    test_vp_nvrtc_compile();
    test_vp_gpu_execution();
    test_vp_scalar_ops();
    test_fp_decode();
    test_fp_codegen();
    test_fp_full_glsl();
    test_vp_disassemble();
    test_vp_throughput();

    printf("\n╠═══════════════════════════════════════════════════╣\n");
    printf("║  Results: %d passed, %d failed                    \n", g_pass, g_fail);
    printf("╚═══════════════════════════════════════════════════╝\n");
    return g_fail > 0 ? 1 : 0;
}
