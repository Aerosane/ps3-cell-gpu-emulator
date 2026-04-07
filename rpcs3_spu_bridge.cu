// rpcs3_spu_bridge.cu — CUDA SPU Bridge Implementation
//
// Bridges any PS3 emulator's SPU execution to GPU via our HyperJIT.
// Manages CUDA context, device memory, JIT compilation cache, and
// bidirectional LS/register transfer.
//
#include "rpcs3_spu_bridge.h"
#include "spu_defs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// SPU Local Store size
static constexpr uint32_t LS_SIZE = 256 * 1024;  // 256KB

// JIT cache entry
struct SPUCacheEntry {
    uint64_t  codeHash;     // FNV-1a hash of LS code region
    CUmodule  cuModule;
    CUfunction cuFunction;
    uint32_t  hitCount;
    bool      valid;
};

static constexpr int MAX_CACHE = 256;

// Bridge global state
struct BridgeState {
    bool          initialized;
    CUcontext     cuCtx;

    // Device memory (persistent across runs to avoid realloc)
    uint8_t*      d_ls;          // 256KB device Local Store
    uint32_t*     d_regs;        // 128 × 4 uint32 = 2KB
    uint32_t*     d_info;        // [pc, status, stopped, cycles_out]

    // JIT cache
    SPUCacheEntry cache[MAX_CACHE];
    uint32_t      numEntries;

    // Cumulative stats
    uint32_t      totalRuns;
    uint32_t      cacheHits;
    uint32_t      compiles;
    double        totalExecMs;
    double        totalCompileMs;
    uint64_t      totalInsns;
};

static BridgeState g_bridge = {};

// FNV-1a hash
static uint64_t hash_code(const uint8_t* data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// ═══════════════════════════════════════════════════════════════
// NVRTC SPU Kernel Generation
// ═══════════════════════════════════════════════════════════════

// Generate a simple SPU interpreter kernel that runs on GPU.
// This is the fallback — the real HyperJIT would emit optimized code per-program.
// For the bridge, we emit a compact interpreter loop that handles core SPU ops.
static const char* spu_kernel_template =
"typedef unsigned int uint32_t;\n"
"typedef int int32_t;\n"
"typedef unsigned short uint16_t;\n"
"typedef short int16_t;\n"
"typedef unsigned char uint8_t;\n"
"typedef char int8_t;\n"
"typedef unsigned long long uint64_t;\n"
"typedef long long int64_t;\n"
"\n"
"__device__ __forceinline__ uint32_t bswap32(uint32_t x) { return __byte_perm(x, 0, 0x0123); }\n"
"\n"
"__device__ __forceinline__ uint32_t ls_fetch(const uint8_t* ls, uint32_t pc) {\n"
"    uint32_t raw; memcpy(&raw, ls + (pc & 0x3FFFF), 4); return bswap32(raw);\n"
"}\n"
"\n"
"__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {\n"
"    n &= 31; return (x << n) | (x >> ((32 - n) & 31));\n"
"}\n"
"__device__ __forceinline__ uint16_t rotl16(uint16_t x, uint16_t n) {\n"
"    n &= 15; return (uint16_t)((x << n) | (x >> ((16 - n) & 15)));\n"
"}\n"
"\n"
"__device__ __forceinline__ float u2f(uint32_t x) { return __int_as_float((int)x); }\n"
"__device__ __forceinline__ uint32_t f2u(float f) { return (uint32_t)__float_as_int(f); }\n"
"\n"
"__device__ __forceinline__ uint8_t reg_byte(const uint32_t* r, int b) {\n"
"    return (uint8_t)((r[b >> 2] >> (8 * (3 - (b & 3)))) & 0xFF);\n"
"}\n"
"__device__ __forceinline__ void set_reg_byte(uint32_t* r, int b, uint8_t v) {\n"
"    int sh = 8 * (3 - (b & 3));\n"
"    r[b >> 2] = (r[b >> 2] & ~((uint32_t)0xFF << sh)) | ((uint32_t)v << sh);\n"
"}\n"
"__device__ __forceinline__ uint16_t reg_hw(const uint32_t* r, int h) {\n"
"    return (uint16_t)((r[h >> 1] >> (16 * (1 - (h & 1)))) & 0xFFFF);\n"
"}\n"
"__device__ __forceinline__ void set_reg_hw(uint32_t* r, int h, uint16_t v) {\n"
"    int sh = 16 * (1 - (h & 1));\n"
"    r[h >> 1] = (r[h >> 1] & ~((uint32_t)0xFFFF << sh)) | ((uint32_t)v << sh);\n"
"}\n"
"\n"
"extern \"C\" __global__ void spu_bridge_kernel(\n"
"    uint8_t*  __restrict__ ls,\n"
"    uint32_t* __restrict__ regs,\n"
"    uint32_t* __restrict__ info)\n"
"{\n"
"    if (threadIdx.x != 0) return;\n"
"    uint32_t pc = info[0];\n"
"    uint32_t max_insns = info[3];\n"
"    uint32_t cycles = 0;\n"
"\n"
"    #define R(i) regs[(i)*4]\n"
"    #define RW(i,w) regs[(i)*4+(w)]\n"
"\n"
"    while (cycles < max_insns) {\n"
"        uint32_t insn = ls_fetch(ls, pc);\n"
"        uint32_t op4  = (insn >> 28) & 0xF;\n"
"        uint32_t op7  = (insn >> 25) & 0x7F;\n"
"        uint32_t op8  = (insn >> 24) & 0xFF;\n"
"        uint32_t op9  = (insn >> 23) & 0x1FF;\n"
"        uint32_t op10 = (insn >> 22) & 0x3FF;\n"
"        uint32_t op11 = (insn >> 21) & 0x7FF;\n"
"\n"
"        uint32_t rt = insn & 0x7F;\n"
"        uint32_t ra = (insn >> 7) & 0x7F;\n"
"        uint32_t rb = (insn >> 14) & 0x7F;\n"
"        int32_t i16 = (int32_t)((int16_t)((insn >> 7) & 0xFFFF));\n"
"        int32_t i10 = (int32_t)(((int32_t)((insn >> 14) & 0x3FF) << 22) >> 22);\n"
"        int32_t i7  = (int32_t)(((int32_t)((insn >> 14) & 0x7F) << 25) >> 25);\n"
"        uint32_t i18 = (insn >> 7) & 0x3FFFF;\n"
"\n"
"        bool handled = true;\n"
"        switch (op11) {\n"
"        case 0x0C0: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)+RW(rb,w); break;\n"
"        case 0x0C8: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,reg_hw(a,h)+reg_hw(b,h)); break; }\n"
"        case 0x040: for(int w=0;w<4;w++) RW(rt,w)=RW(rb,w)-RW(ra,w); break;\n"
"        case 0x048: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,reg_hw(b,h)-reg_hw(a,h)); break; }\n"
"        case 0x340: for(int w=0;w<4;w++){uint32_t c=RW(rt,w)&1;RW(rt,w)=RW(ra,w)+RW(rb,w)+c;} break;\n"
"        case 0x341: for(int w=0;w<4;w++){uint32_t c=RW(rt,w)&1;RW(rt,w)=RW(rb,w)-RW(ra,w)-1+c;} break;\n"
"        case 0x0C2: for(int w=0;w<4;w++) RW(rt,w)=((uint64_t)RW(ra,w)+(uint64_t)RW(rb,w))>0xFFFFFFFFULL?1:0; break;\n"
"        case 0x042: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)>RW(rb,w)?0:1; break;\n"
"        case 0x342: for(int w=0;w<4;w++){uint64_t s=(uint64_t)RW(ra,w)+(uint64_t)RW(rb,w)+(uint64_t)(RW(rt,w)&1);RW(rt,w)=(uint32_t)(s>>32);} break;\n"
"        case 0x343: for(int w=0;w<4;w++){uint64_t s=(uint64_t)RW(rb,w)+(uint64_t)(RW(rt,w)&1)-(uint64_t)RW(ra,w)-1ULL;RW(rt,w)=(s>>63)?0:1;} break;\n"
"        case 0x0C1: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)&RW(rb,w); break;\n"
"        case 0x041: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)|RW(rb,w); break;\n"
"        case 0x241: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)^RW(rb,w); break;\n"
"        case 0x049: for(int w=0;w<4;w++) RW(rt,w)=~(RW(ra,w)|RW(rb,w)); break;\n"
"        case 0x0C9: for(int w=0;w<4;w++) RW(rt,w)=~(RW(ra,w)&RW(rb,w)); break;\n"
"        case 0x2C1: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)&~RW(rb,w); break;\n"
"        case 0x2C9: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)|~RW(rb,w); break;\n"
"        case 0x249: for(int w=0;w<4;w++) RW(rt,w)=~(RW(ra,w)^RW(rb,w)); break;\n"
"        case 0x058: for(int w=0;w<4;w++) RW(rt,w)=rotl32(RW(ra,w),RW(rb,w)); break;\n"
"        case 0x059: for(int w=0;w<4;w++){uint32_t s=(uint32_t)(-(int32_t)RW(rb,w))&0x3F;RW(rt,w)=s<32?RW(ra,w)>>s:0;} break;\n"
"        case 0x05A: for(int w=0;w<4;w++){uint32_t s=(uint32_t)(-(int32_t)RW(rb,w))&0x3F;RW(rt,w)=(uint32_t)((int32_t)RW(ra,w)>>(s<32?s:31));} break;\n"
"        case 0x05B: for(int w=0;w<4;w++){uint32_t s=RW(rb,w)&0x3F;RW(rt,w)=s<32?RW(ra,w)<<s:0;} break;\n"
"        case 0x05C: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,rotl16(reg_hw(a,h),reg_hw(b,h))); break; }\n"
"        case 0x05D: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int h=0;h<8;h++){uint32_t s=(uint32_t)(-(int16_t)reg_hw(b,h))&0x1F;set_reg_hw(d,h,s<16?(uint16_t)(reg_hw(a,h)>>s):0);} break; }\n"
"        case 0x05E: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int h=0;h<8;h++){uint32_t s=(uint32_t)(-(int16_t)reg_hw(b,h))&0x1F;set_reg_hw(d,h,(uint16_t)((int16_t)reg_hw(a,h)>>(s<16?s:15)));} break; }\n"
"        case 0x05F: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int h=0;h<8;h++){uint32_t s=reg_hw(b,h)&0x1F;set_reg_hw(d,h,s<16?(uint16_t)(reg_hw(a,h)<<s):0);} break; }\n"
"        case 0x1DC: { int n=RW(rb,0)&0xF; uint8_t t[16]; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int i=0;i<16;i++) t[i]=reg_byte(a,(i+n)&0xF); for(int i=0;i<16;i++) set_reg_byte(d,i,t[i]); break; }\n"
"        case 0x1DD: { int n=(-(int32_t)RW(rb,0))&0x1F; uint8_t t[16]; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int i=0;i<16;i++){int si=i-n;t[i]=(si>=0)?reg_byte(a,si):0;} for(int i=0;i<16;i++) set_reg_byte(d,i,t[i]); break; }\n"
"        case 0x1DF: { int n=RW(rb,0)&0x1F; uint8_t t[16]; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int i=0;i<16;i++){int si=i+n;t[i]=(si<16)?reg_byte(a,si):0;} for(int i=0;i<16;i++) set_reg_byte(d,i,t[i]); break; }\n"
"        case 0x1CC: { int n=(RW(rb,0)>>3)&0xF; uint8_t t[16]; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int i=0;i<16;i++) t[i]=reg_byte(a,(i+n)&0xF); for(int i=0;i<16;i++) set_reg_byte(d,i,t[i]); break; }\n"
"        case 0x1CD: { int n=(-((int32_t)RW(rb,0)>>3))&0x1F; uint8_t t[16]; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int i=0;i<16;i++){int si=i-n;t[i]=(si>=0)?reg_byte(a,si):0;} for(int i=0;i<16;i++) set_reg_byte(d,i,t[i]); break; }\n"
"        case 0x1CF: { int n=(RW(rb,0)>>3)&0x1F; uint8_t t[16]; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int i=0;i<16;i++){int si=i+n;t[i]=(si<16)?reg_byte(a,si):0;} for(int i=0;i<16;i++) set_reg_byte(d,i,t[i]); break; }\n"
"        case 0x1D8: { int n=RW(rb,0)&0x7; if(n){uint64_t hi=((uint64_t)RW(ra,0)<<32)|RW(ra,1),lo=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t nh=(hi<<n)|(lo>>(64-n)),nl=(lo<<n)|(hi>>(64-n));RW(rt,0)=(uint32_t)(nh>>32);RW(rt,1)=(uint32_t)nh;RW(rt,2)=(uint32_t)(nl>>32);RW(rt,3)=(uint32_t)nl;}else{for(int w=0;w<4;w++)RW(rt,w)=RW(ra,w);} break; }\n"
"        case 0x1D9: { int n=(-(int32_t)RW(rb,0))&0x7; if(n){uint64_t hi=((uint64_t)RW(ra,0)<<32)|RW(ra,1),lo=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t nh=hi>>n,nl=(lo>>n)|(hi<<(64-n));RW(rt,0)=(uint32_t)(nh>>32);RW(rt,1)=(uint32_t)nh;RW(rt,2)=(uint32_t)(nl>>32);RW(rt,3)=(uint32_t)nl;}else{for(int w=0;w<4;w++)RW(rt,w)=RW(ra,w);} break; }\n"
"        case 0x1DB: { int n=RW(rb,0)&0x7; if(n){uint64_t hi=((uint64_t)RW(ra,0)<<32)|RW(ra,1),lo=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t nh=(hi<<n)|(lo>>(64-n)),nl=lo<<n;RW(rt,0)=(uint32_t)(nh>>32);RW(rt,1)=(uint32_t)nh;RW(rt,2)=(uint32_t)(nl>>32);RW(rt,3)=(uint32_t)nl;}else{for(int w=0;w<4;w++)RW(rt,w)=RW(ra,w);} break; }\n"
"        case 0x078: for(int w=0;w<4;w++) RW(rt,w)=rotl32(RW(ra,w),(uint32_t)(i7&0x1F)); break;\n"
"        case 0x079: { uint32_t s=(uint32_t)(-i7)&0x3F; for(int w=0;w<4;w++) RW(rt,w)=s<32?RW(ra,w)>>s:0; break; }\n"
"        case 0x07A: { uint32_t s=(uint32_t)(-i7)&0x3F; for(int w=0;w<4;w++) RW(rt,w)=(uint32_t)((int32_t)RW(ra,w)>>(s<32?s:31)); break; }\n"
"        case 0x07B: { uint32_t s=(uint32_t)i7&0x3F; for(int w=0;w<4;w++) RW(rt,w)=s<32?RW(ra,w)<<s:0; break; }\n"
"        case 0x07C: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,rotl16(reg_hw(a,h),(uint16_t)(i7&0xF))); break; }\n"
"        case 0x07D: { uint32_t s=(uint32_t)(-i7)&0x1F; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,s<16?(uint16_t)(reg_hw(a,h)>>s):0); break; }\n"
"        case 0x07E: { uint32_t s=(uint32_t)(-i7)&0x1F; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,(uint16_t)((int16_t)reg_hw(a,h)>>(s<16?s:15))); break; }\n"
"        case 0x07F: { uint32_t s=(uint32_t)i7&0x1F; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,s<16?(uint16_t)(reg_hw(a,h)<<s):0); break; }\n"
"        case 0x1FC: { int n=i7&0xF; uint8_t t[16]; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int i=0;i<16;i++) t[i]=reg_byte(a,(i+n)&0xF); for(int i=0;i<16;i++) set_reg_byte(d,i,t[i]); break; }\n"
"        case 0x1FD: { int n=(-i7)&0x1F; uint8_t t[16]; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int i=0;i<16;i++){int si=i-n;t[i]=(si>=0)?reg_byte(a,si):0;} for(int i=0;i<16;i++) set_reg_byte(d,i,t[i]); break; }\n"
"        case 0x1FF: { int n=i7&0x1F; uint8_t t[16]; uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int i=0;i<16;i++){int si=i+n;t[i]=(si<16)?reg_byte(a,si):0;} for(int i=0;i<16;i++) set_reg_byte(d,i,t[i]); break; }\n"
"        case 0x1F8: { int n=i7&0x7; if(n){uint64_t hi=((uint64_t)RW(ra,0)<<32)|RW(ra,1),lo=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t nh=(hi<<n)|(lo>>(64-n)),nl=(lo<<n)|(hi>>(64-n));RW(rt,0)=(uint32_t)(nh>>32);RW(rt,1)=(uint32_t)nh;RW(rt,2)=(uint32_t)(nl>>32);RW(rt,3)=(uint32_t)nl;}else{for(int w=0;w<4;w++)RW(rt,w)=RW(ra,w);} break; }\n"
"        case 0x1F9: { int n=(-i7)&0x7; if(n){uint64_t hi=((uint64_t)RW(ra,0)<<32)|RW(ra,1),lo=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t nh=hi>>n,nl=(lo>>n)|(hi<<(64-n));RW(rt,0)=(uint32_t)(nh>>32);RW(rt,1)=(uint32_t)nh;RW(rt,2)=(uint32_t)(nl>>32);RW(rt,3)=(uint32_t)nl;}else{for(int w=0;w<4;w++)RW(rt,w)=RW(ra,w);} break; }\n"
"        case 0x1FB: { int n=i7&0x7; if(n){uint64_t hi=((uint64_t)RW(ra,0)<<32)|RW(ra,1),lo=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t nh=(hi<<n)|(lo>>(64-n)),nl=lo<<n;RW(rt,0)=(uint32_t)(nh>>32);RW(rt,1)=(uint32_t)nh;RW(rt,2)=(uint32_t)(nl>>32);RW(rt,3)=(uint32_t)nl;}else{for(int w=0;w<4;w++)RW(rt,w)=RW(ra,w);} break; }\n"
"        case 0x1F4: { int t=(~(i7+(int32_t)RW(ra,0)))&0xF; uint32_t*d=&regs[rt*4]; d[0]=0x10111213;d[1]=0x14151617;d[2]=0x18191A1B;d[3]=0x1C1D1E1F; set_reg_byte(d,t,0x03); break; }\n"
"        case 0x1F5: { int t=((~(i7+(int32_t)RW(ra,0)))&0xE)>>1; uint32_t*d=&regs[rt*4]; d[0]=0x10111213;d[1]=0x14151617;d[2]=0x18191A1B;d[3]=0x1C1D1E1F; set_reg_hw(d,t,0x0203); break; }\n"
"        case 0x1F6: { int t=((~(i7+(int32_t)RW(ra,0)))&0xC)>>2; uint32_t*d=&regs[rt*4]; d[0]=0x10111213;d[1]=0x14151617;d[2]=0x18191A1B;d[3]=0x1C1D1E1F; d[t]=0x00010203; break; }\n"
"        case 0x1F7: { int t=((~(i7+(int32_t)RW(ra,0)))&0x8)>>3; uint32_t*d=&regs[rt*4]; d[0]=0x10111213;d[1]=0x14151617;d[2]=0x18191A1B;d[3]=0x1C1D1E1F; d[t*2]=0x00010203;d[t*2+1]=0x04050607; break; }\n"
"        case 0x1C4: { uint32_t addr=(RW(ra,0)+RW(rb,0))&0x3FFF0; for(int w=0;w<4;w++){uint32_t v;memcpy(&v,ls+addr+w*4,4);RW(rt,w)=bswap32(v);} break; }\n"
"        case 0x144: { uint32_t addr=(RW(ra,0)+RW(rb,0))&0x3FFF0; for(int w=0;w<4;w++){uint32_t v=bswap32(RW(rt,w));memcpy(ls+addr+w*4,&v,4);} break; }\n"
"        case 0x3C0: for(int w=0;w<4;w++) RW(rt,w)=(RW(ra,w)==RW(rb,w))?0xFFFFFFFF:0; break;\n"
"        case 0x240: for(int w=0;w<4;w++) RW(rt,w)=((int32_t)RW(ra,w)>(int32_t)RW(rb,w))?0xFFFFFFFF:0; break;\n"
"        case 0x2C0: for(int w=0;w<4;w++) RW(rt,w)=(RW(ra,w)>RW(rb,w))?0xFFFFFFFF:0; break;\n"
"        case 0x3C8: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,(reg_hw(a,h)==reg_hw(b,h))?0xFFFF:0); break; }\n"
"        case 0x248: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,((int16_t)reg_hw(a,h)>(int16_t)reg_hw(b,h))?0xFFFF:0); break; }\n"
"        case 0x2C8: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,(reg_hw(a,h)>reg_hw(b,h))?0xFFFF:0); break; }\n"
"        case 0x3D0: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,(reg_byte(a,q)==reg_byte(b,q))?0xFF:0); break; }\n"
"        case 0x250: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,((int8_t)reg_byte(a,q)>(int8_t)reg_byte(b,q))?0xFF:0); break; }\n"
"        case 0x2D0: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,(reg_byte(a,q)>reg_byte(b,q))?0xFF:0); break; }\n"
"        case 0x3C4: for(int w=0;w<4;w++) RW(rt,w)=(uint32_t)((int32_t)(int16_t)(RW(ra,w)&0xFFFF)*(int32_t)(int16_t)(RW(rb,w)&0xFFFF)); break;\n"
"        case 0x3CC: for(int w=0;w<4;w++) RW(rt,w)=(RW(ra,w)&0xFFFF)*(RW(rb,w)&0xFFFF); break;\n"
"        case 0x3C5: for(int w=0;w<4;w++) RW(rt,w)=((RW(ra,w)>>16)*(RW(rb,w)&0xFFFF))<<16; break;\n"
"        case 0x3C7: for(int w=0;w<4;w++) RW(rt,w)=(uint32_t)(((int32_t)(int16_t)(RW(ra,w)&0xFFFF)*(int32_t)(int16_t)(RW(rb,w)&0xFFFF))>>16); break;\n"
"        case 0x3C6: for(int w=0;w<4;w++) RW(rt,w)=(uint32_t)((int32_t)(int16_t)(RW(ra,w)>>16)*(int32_t)(int16_t)(RW(rb,w)>>16)); break;\n"
"        case 0x3CE: for(int w=0;w<4;w++) RW(rt,w)=(RW(ra,w)>>16)*(RW(rb,w)>>16); break;\n"
"        case 0x2A5: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)?__clz(RW(ra,w)):32; break;\n"
"        case 0x2B4: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,(uint8_t)__popc((uint32_t)reg_byte(a,q))); break; }\n"
"        case 0x0D3: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,(uint8_t)(((uint32_t)reg_byte(a,q)+(uint32_t)reg_byte(b,q)+1)>>1)); break; }\n"
"        case 0x053: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int q=0;q<16;q++){int v=(int)reg_byte(a,q)-(int)reg_byte(b,q);set_reg_byte(d,q,(uint8_t)(v<0?-v:v));} break; }\n"
"        case 0x1B0: { uint32_t r=((RW(ra,0)&1)<<3)|((RW(ra,1)&1)<<2)|((RW(ra,2)&1)<<1)|(RW(ra,3)&1); RW(rt,0)=r;RW(rt,1)=0;RW(rt,2)=0;RW(rt,3)=0; break; }\n"
"        case 0x1B1: { uint32_t*a=&regs[ra*4]; uint32_t r=0; for(int h=0;h<8;h++) r|=((uint32_t)(reg_hw(a,h)&1))<<(7-h); RW(rt,0)=r;RW(rt,1)=0;RW(rt,2)=0;RW(rt,3)=0; break; }\n"
"        case 0x1B2: { uint32_t*a=&regs[ra*4]; uint32_t r=0; for(int q=0;q<16;q++) r|=((uint32_t)(reg_byte(a,q)&1))<<(15-q); RW(rt,0)=r;RW(rt,1)=0;RW(rt,2)=0;RW(rt,3)=0; break; }\n"
"        case 0x1B4: { uint32_t m=RW(ra,0)&0xF; for(int w=0;w<4;w++) RW(rt,w)=(m&(8u>>w))?0xFFFFFFFF:0; break; }\n"
"        case 0x1B5: { uint32_t m=RW(ra,0)&0xFF; uint32_t*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,(m&(0x80u>>h))?0xFFFF:0); break; }\n"
"        case 0x1B6: { uint32_t m=RW(ra,0)&0xFFFF; uint32_t*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,(m&(0x8000u>>q))?0xFF:0); break; }\n"
"        case 0x2B6: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,(uint16_t)(int16_t)(int8_t)(reg_hw(a,h)&0xFF)); break; }\n"
"        case 0x2AE: for(int w=0;w<4;w++) RW(rt,w)=(uint32_t)(int32_t)(int16_t)(RW(ra,w)&0xFFFF); break;\n"
"        case 0x2A6: { int64_t d0=(int64_t)(int32_t)RW(ra,1),d1=(int64_t)(int32_t)RW(ra,3); RW(rt,0)=(uint32_t)(d0>>32);RW(rt,1)=(uint32_t)d0;RW(rt,2)=(uint32_t)(d1>>32);RW(rt,3)=(uint32_t)d1; break; }\n"
"        case 0x1F0: RW(rt,0)=RW(ra,0)|RW(ra,1)|RW(ra,2)|RW(ra,3);RW(rt,1)=0;RW(rt,2)=0;RW(rt,3)=0; break;\n"
"        case 0x253: { uint32_t*a=&regs[ra*4],*b=&regs[rb*4],*d=&regs[rt*4]; for(int w=0;w<4;w++){uint32_t sb=(uint32_t)reg_byte(b,w*4)+reg_byte(b,w*4+1)+reg_byte(b,w*4+2)+reg_byte(b,w*4+3);uint32_t sa=(uint32_t)reg_byte(a,w*4)+reg_byte(a,w*4+1)+reg_byte(a,w*4+2)+reg_byte(a,w*4+3);set_reg_hw(d,w*2,(uint16_t)sb);set_reg_hw(d,w*2+1,(uint16_t)sa);} break; }\n"
"        case 0x2C4: for(int w=0;w<4;w++) RW(rt,w)=f2u(u2f(RW(ra,w))+u2f(RW(rb,w))); break;\n"
"        case 0x2C5: for(int w=0;w<4;w++) RW(rt,w)=f2u(u2f(RW(ra,w))-u2f(RW(rb,w))); break;\n"
"        case 0x2C6: for(int w=0;w<4;w++) RW(rt,w)=f2u(u2f(RW(ra,w))*u2f(RW(rb,w))); break;\n"
"        case 0x3C2: for(int w=0;w<4;w++) RW(rt,w)=(u2f(RW(ra,w))==u2f(RW(rb,w)))?0xFFFFFFFF:0; break;\n"
"        case 0x2C2: for(int w=0;w<4;w++) RW(rt,w)=(u2f(RW(ra,w))>u2f(RW(rb,w)))?0xFFFFFFFF:0; break;\n"
"        case 0x3CA: for(int w=0;w<4;w++) RW(rt,w)=(fabsf(u2f(RW(ra,w)))==fabsf(u2f(RW(rb,w))))?0xFFFFFFFF:0; break;\n"
"        case 0x2CA: for(int w=0;w<4;w++) RW(rt,w)=(fabsf(u2f(RW(ra,w)))>fabsf(u2f(RW(rb,w))))?0xFFFFFFFF:0; break;\n"
"        case 0x1B8: for(int w=0;w<4;w++){float v=u2f(RW(ra,w));RW(rt,w)=f2u(v!=0.0f?1.0f/v:0.0f);} break;\n"
"        case 0x1B9: for(int w=0;w<4;w++){float v=fabsf(u2f(RW(ra,w)));RW(rt,w)=f2u(v>0.0f?1.0f/sqrtf(v):0.0f);} break;\n"
"        case 0x3D4: for(int w=0;w<4;w++) RW(rt,w)=RW(rb,w); break;\n"
"        case 0x2CC: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);double da,db,dal,dbl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8);da+=db;dal+=dbl;memcpy(&ah,&da,8);memcpy(&al,&dal,8);RW(rt,0)=(uint32_t)(ah>>32);RW(rt,1)=(uint32_t)ah;RW(rt,2)=(uint32_t)(al>>32);RW(rt,3)=(uint32_t)al; break; }\n"
"        case 0x2CD: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);double da,db,dal,dbl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8);da-=db;dal-=dbl;memcpy(&ah,&da,8);memcpy(&al,&dal,8);RW(rt,0)=(uint32_t)(ah>>32);RW(rt,1)=(uint32_t)ah;RW(rt,2)=(uint32_t)(al>>32);RW(rt,3)=(uint32_t)al; break; }\n"
"        case 0x2CE: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);double da,db,dal,dbl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8);da*=db;dal*=dbl;memcpy(&ah,&da,8);memcpy(&al,&dal,8);RW(rt,0)=(uint32_t)(ah>>32);RW(rt,1)=(uint32_t)ah;RW(rt,2)=(uint32_t)(al>>32);RW(rt,3)=(uint32_t)al; break; }\n"
"        case 0x1A8: pc=RW(ra,0)&0x3FFFC; cycles++; continue;\n"
"        case 0x1A9: { uint32_t tgt=RW(ra,0)&0x3FFFC; RW(rt,0)=pc+4;RW(rt,1)=0;RW(rt,2)=0;RW(rt,3)=0; pc=tgt; cycles++; continue; }\n"
"        case 0x128: if(R(rt)==0){pc=RW(ra,0)&0x3FFFC;cycles++;continue;} break;\n"
"        case 0x129: if(R(rt)!=0){pc=RW(ra,0)&0x3FFFC;cycles++;continue;} break;\n"
"        case 0x12A: if((R(rt)&0xFFFF)==0){pc=RW(ra,0)&0x3FFFC;cycles++;continue;} break;\n"
"        case 0x12B: if((R(rt)&0xFFFF)!=0){pc=RW(ra,0)&0x3FFFC;cycles++;continue;} break;\n"
"        case 0x258: if((int32_t)RW(ra,0)>(int32_t)RW(rb,0)){info[2]=2;goto done;} break;\n"
"        case 0x2D8: if(RW(ra,0)>RW(rb,0)){info[2]=2;goto done;} break;\n"
"        case 0x3D8: if(RW(ra,0)==RW(rb,0)){info[2]=2;goto done;} break;\n"
"        case 0x000: info[2]=1; goto done;\n"
"        case 0x140: info[2]=1; goto done;\n"
"        case 0x001: case 0x201: case 0x002: case 0x003: case 0x1AC: break;\n"
"        case 0x00C: for(int w=0;w<4;w++) RW(rt,w)=0; break;\n"
"        case 0x10C: break;\n"
"        case 0x00D: info[2]=3;info[5]=ra;info[6]=rt;goto done;\n"
"        case 0x10D: info[2]=4;info[5]=ra;info[6]=rt;goto done;\n"
"        case 0x00F: RW(rt,0)=1;RW(rt,1)=0;RW(rt,2)=0;RW(rt,3)=0; break;\n"

// CBX/CHX/CWX/CDX: generate controls for byte/halfword/word/doubleword insertion (register form)
"        case 0x1D4: { int t=(~(RW(ra,0)+RW(rb,0)))&0xF; uint32_t*d=&regs[rt*4]; d[0]=0x10111213;d[1]=0x14151617;d[2]=0x18191A1B;d[3]=0x1C1D1E1F; set_reg_byte(d,t,0x03); break; }\n"
"        case 0x1D5: { int t=((~(RW(ra,0)+RW(rb,0)))&0xE)>>1; uint32_t*d=&regs[rt*4]; d[0]=0x10111213;d[1]=0x14151617;d[2]=0x18191A1B;d[3]=0x1C1D1E1F; set_reg_hw(d,t,0x0203); break; }\n"
"        case 0x1D6: { int t=((~(RW(ra,0)+RW(rb,0)))&0xC)>>2; uint32_t*d=&regs[rt*4]; d[0]=0x10111213;d[1]=0x14151617;d[2]=0x18191A1B;d[3]=0x1C1D1E1F; d[t]=0x00010203; break; }\n"
"        case 0x1D7: { int t=((~(RW(ra,0)+RW(rb,0)))&0x8)>>3; uint32_t*d=&regs[rt*4]; d[0]=0x10111213;d[1]=0x14151617;d[2]=0x18191A1B;d[3]=0x1C1D1E1F; d[t*2]=0x00010203;d[t*2+1]=0x04050607; break; }\n"

// MPYHHA/MPYHHAU: multiply high halfword with accumulate
"        case 0x346: for(int w=0;w<4;w++) RW(rt,w)+=(uint32_t)((int32_t)(int16_t)(RW(ra,w)>>16)*(int32_t)(int16_t)(RW(rb,w)>>16)); break;\n"
"        case 0x34E: for(int w=0;w<4;w++) RW(rt,w)+=(RW(ra,w)>>16)*(RW(rb,w)>>16); break;\n"

// FESD: floating extend single to double (slot 0→dw0, slot 2→dw1)
"        case 0x3B8: { float f0=u2f(RW(ra,0)),f1=u2f(RW(ra,2)); double d0=(double)f0,d1=(double)f1; uint64_t u0,u1; memcpy(&u0,&d0,8);memcpy(&u1,&d1,8); RW(rt,0)=(uint32_t)(u0>>32);RW(rt,1)=(uint32_t)u0;RW(rt,2)=(uint32_t)(u1>>32);RW(rt,3)=(uint32_t)u1; break; }\n"
// FRDS: floating round double to single (dw0→slot 0, dw1→slot 2)
"        case 0x3B9: { uint64_t u0=((uint64_t)RW(ra,0)<<32)|RW(ra,1),u1=((uint64_t)RW(ra,2)<<32)|RW(ra,3); double d0,d1; memcpy(&d0,&u0,8);memcpy(&d1,&u1,8); RW(rt,0)=f2u((float)d0);RW(rt,1)=0;RW(rt,2)=f2u((float)d1);RW(rt,3)=0; break; }\n"

// FSCRRD/FSCRWR: FP status/control register (stub — read zeros / ignore writes)
"        case 0x398: for(int w=0;w<4;w++) RW(rt,w)=0; break;\n"
"        case 0x3BA: break;\n"

// DFTSV: double FP test special value — test class bits in rb[0:6]
"        case 0x3BF: { uint64_t u0=((uint64_t)RW(ra,0)<<32)|RW(ra,1),u1=((uint64_t)RW(ra,2)<<32)|RW(ra,3); double d0,d1; memcpy(&d0,&u0,8);memcpy(&d1,&u1,8); uint32_t mask=i7&0x7F; uint32_t r0=0,r1=0; uint64_t e0=(u0>>52)&0x7FF,m0=u0&0xFFFFFFFFFFFFFULL; uint64_t e1=(u1>>52)&0x7FF,m1=u1&0xFFFFFFFFFFFFFULL; int s0=(u0>>63)&1,s1=(u1>>63)&1; if((mask&0x01)&&(e0==0&&m0==0&&!s0)) r0=0xFFFFFFFF; if((mask&0x02)&&(e0==0&&m0==0&&s0)) r0=0xFFFFFFFF; if((mask&0x04)&&(e0==0&&m0!=0)) r0=0xFFFFFFFF; if((mask&0x08)&&(e0==0x7FF&&m0==0&&!s0)) r0=0xFFFFFFFF; if((mask&0x10)&&(e0==0x7FF&&m0==0&&s0)) r0=0xFFFFFFFF; if((mask&0x20)&&(e0==0x7FF&&m0!=0)) r0=0xFFFFFFFF; if((mask&0x01)&&(e1==0&&m1==0&&!s1)) r1=0xFFFFFFFF; if((mask&0x02)&&(e1==0&&m1==0&&s1)) r1=0xFFFFFFFF; if((mask&0x04)&&(e1==0&&m1!=0)) r1=0xFFFFFFFF; if((mask&0x08)&&(e1==0x7FF&&m1==0&&!s1)) r1=0xFFFFFFFF; if((mask&0x10)&&(e1==0x7FF&&m1==0&&s1)) r1=0xFFFFFFFF; if((mask&0x20)&&(e1==0x7FF&&m1!=0)) r1=0xFFFFFFFF; RW(rt,0)=r0;RW(rt,1)=r0;RW(rt,2)=r1;RW(rt,3)=r1; break; }\n"

// Double FP compare: DFCEQ, DFCGT, DFCMEQ, DFCMGT
"        case 0x3C3: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);double da,db,dal,dbl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8); uint32_t r0=(da==db)?0xFFFFFFFF:0,r1=(dal==dbl)?0xFFFFFFFF:0; RW(rt,0)=r0;RW(rt,1)=r0;RW(rt,2)=r1;RW(rt,3)=r1; break; }\n"
"        case 0x2C3: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);double da,db,dal,dbl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8); uint32_t r0=(da>db)?0xFFFFFFFF:0,r1=(dal>dbl)?0xFFFFFFFF:0; RW(rt,0)=r0;RW(rt,1)=r0;RW(rt,2)=r1;RW(rt,3)=r1; break; }\n"
"        case 0x3CB: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);double da,db,dal,dbl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8); da=da<0?-da:da;db=db<0?-db:db;dal=dal<0?-dal:dal;dbl=dbl<0?-dbl:dbl; uint32_t r0=(da==db)?0xFFFFFFFF:0,r1=(dal==dbl)?0xFFFFFFFF:0; RW(rt,0)=r0;RW(rt,1)=r0;RW(rt,2)=r1;RW(rt,3)=r1; break; }\n"
"        case 0x2CB: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);double da,db,dal,dbl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8); da=da<0?-da:da;db=db<0?-db:db;dal=dal<0?-dal:dal;dbl=dbl<0?-dbl:dbl; uint32_t r0=(da>db)?0xFFFFFFFF:0,r1=(dal>dbl)?0xFFFFFFFF:0; RW(rt,0)=r0;RW(rt,1)=r0;RW(rt,2)=r1;RW(rt,3)=r1; break; }\n"

// DFMA/DFMS/DFNMS/DFNMA: double FP multiply-add/subtract variants
"        case 0x35C: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);uint64_t th=((uint64_t)RW(rt,0)<<32)|RW(rt,1),tl=((uint64_t)RW(rt,2)<<32)|RW(rt,3);double da,db,dt,dal,dbl,dtl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dt,&th,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8);memcpy(&dtl,&tl,8);dt=fma(da,db,dt);dtl=fma(dal,dbl,dtl);memcpy(&th,&dt,8);memcpy(&tl,&dtl,8);RW(rt,0)=(uint32_t)(th>>32);RW(rt,1)=(uint32_t)th;RW(rt,2)=(uint32_t)(tl>>32);RW(rt,3)=(uint32_t)tl; break; }\n"
"        case 0x35D: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);uint64_t th=((uint64_t)RW(rt,0)<<32)|RW(rt,1),tl=((uint64_t)RW(rt,2)<<32)|RW(rt,3);double da,db,dt,dal,dbl,dtl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dt,&th,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8);memcpy(&dtl,&tl,8);dt=fma(da,db,-dt);dtl=fma(dal,dbl,-dtl);memcpy(&th,&dt,8);memcpy(&tl,&dtl,8);RW(rt,0)=(uint32_t)(th>>32);RW(rt,1)=(uint32_t)th;RW(rt,2)=(uint32_t)(tl>>32);RW(rt,3)=(uint32_t)tl; break; }\n"
"        case 0x35E: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);uint64_t th=((uint64_t)RW(rt,0)<<32)|RW(rt,1),tl=((uint64_t)RW(rt,2)<<32)|RW(rt,3);double da,db,dt,dal,dbl,dtl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dt,&th,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8);memcpy(&dtl,&tl,8);dt=fma(-da,db,dt);dtl=fma(-dal,dbl,dtl);memcpy(&th,&dt,8);memcpy(&tl,&dtl,8);RW(rt,0)=(uint32_t)(th>>32);RW(rt,1)=(uint32_t)th;RW(rt,2)=(uint32_t)(tl>>32);RW(rt,3)=(uint32_t)tl; break; }\n"
"        case 0x35F: { uint64_t ah=((uint64_t)RW(ra,0)<<32)|RW(ra,1),al=((uint64_t)RW(ra,2)<<32)|RW(ra,3);uint64_t bh=((uint64_t)RW(rb,0)<<32)|RW(rb,1),bl=((uint64_t)RW(rb,2)<<32)|RW(rb,3);uint64_t th=((uint64_t)RW(rt,0)<<32)|RW(rt,1),tl=((uint64_t)RW(rt,2)<<32)|RW(rt,3);double da,db,dt,dal,dbl,dtl;memcpy(&da,&ah,8);memcpy(&db,&bh,8);memcpy(&dt,&th,8);memcpy(&dal,&al,8);memcpy(&dbl,&bl,8);memcpy(&dtl,&tl,8);dt=-fma(da,db,dt);dtl=-fma(dal,dbl,dtl);memcpy(&th,&dt,8);memcpy(&tl,&dtl,8);RW(rt,0)=(uint32_t)(th>>32);RW(rt,1)=(uint32_t)th;RW(rt,2)=(uint32_t)(tl>>32);RW(rt,3)=(uint32_t)tl; break; }\n"

// IRET: interrupt return (stub — halt, not supported in GPU context)
"        case 0x1AA: info[2]=2; goto done;\n"
// BISLED: branch indirect if external data (stub — treat as unconditional for now)
"        case 0x1AB: { uint32_t tgt=RW(ra,0)&0x3FFFC; RW(rt,0)=pc+4;RW(rt,1)=0;RW(rt,2)=0;RW(rt,3)=0; pc=tgt; cycles++; continue; }\n"

"        default: handled=false;\n"
"        }\n"
"\n"
"        if (!handled) {\n"
"            handled = true;\n"
"            int cvt_scale_exp = 173 - (int)((insn >> 14) & 0xFF);\n"
"            switch (op10) {\n"
"            case 0x1DA: { float sc=ldexpf(1.0f,cvt_scale_exp); for(int w=0;w<4;w++) RW(rt,w)=f2u((float)(int32_t)RW(ra,w)*sc); break; }\n"
"            case 0x1DB: { float sc=ldexpf(1.0f,cvt_scale_exp); for(int w=0;w<4;w++) RW(rt,w)=f2u((float)RW(ra,w)*sc); break; }\n"
"            case 0x1D8: { float sc=ldexpf(1.0f,-cvt_scale_exp); for(int w=0;w<4;w++){float v=u2f(RW(ra,w))*sc;RW(rt,w)=(uint32_t)(v>=2147483648.0f?0x7FFFFFFF:v<=-2147483649.0f?(int32_t)0x80000000:(int32_t)v);} break; }\n"
"            case 0x1D9: { float sc=ldexpf(1.0f,-cvt_scale_exp); for(int w=0;w<4;w++){float v=u2f(RW(ra,w))*sc;RW(rt,w)=v>=4294967296.0f?0xFFFFFFFF:v<=0.0f?0:(uint32_t)v;} break; }\n"
"            default: handled=false;\n"
"            }\n"
"        }\n"
"\n"
"        if (!handled) {\n"
"            handled = true;\n"
"            switch (op9) {\n"
"            case 0x081: for(int w=0;w<4;w++) RW(rt,w)=(uint32_t)i16; break;\n"
"            case 0x083: { uint32_t v=((uint32_t)(uint16_t)i16)|(((uint32_t)(uint16_t)i16)<<16); for(int w=0;w<4;w++) RW(rt,w)=v; break; }\n"
"            case 0x082: for(int w=0;w<4;w++) RW(rt,w)=((uint32_t)(uint16_t)i16)<<16; break;\n"
"            case 0x0C1: for(int w=0;w<4;w++) RW(rt,w)|=(uint32_t)(uint16_t)i16; break;\n"
"            case 0x040: if(R(rt)==0){pc=(uint32_t)(i16<<2)&0x3FFFC;cycles++;continue;} break;\n"
"            case 0x042: if(R(rt)!=0){pc=(uint32_t)(i16<<2)&0x3FFFC;cycles++;continue;} break;\n"
"            case 0x044: if((R(rt)&0xFFFF)==0){pc=(uint32_t)(i16<<2)&0x3FFFC;cycles++;continue;} break;\n"
"            case 0x046: if((R(rt)&0xFFFF)!=0){pc=(uint32_t)(i16<<2)&0x3FFFC;cycles++;continue;} break;\n"
"            case 0x064: pc=(uint32_t)(i16<<2)&0x3FFFC;cycles++;continue;\n"
"            case 0x060: pc=(uint32_t)(i16<<2)&0x3FFFC;cycles++;continue;\n"
"            case 0x066: { uint32_t tgt=(uint32_t)(i16<<2)&0x3FFFC; RW(rt,0)=pc+4;RW(rt,1)=0;RW(rt,2)=0;RW(rt,3)=0; pc=tgt;cycles++;continue; }\n"
"            case 0x062: { uint32_t tgt=(uint32_t)(i16<<2)&0x3FFFC; RW(rt,0)=pc+4;RW(rt,1)=0;RW(rt,2)=0;RW(rt,3)=0; pc=tgt;cycles++;continue; }\n"
"            case 0x065: { uint32_t m=(uint32_t)(uint16_t)i16; uint32_t*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,(m&(0x8000u>>q))?0xFF:0); break; }\n"
"            case 0x061: { uint32_t addr=((uint32_t)(i16<<2))&0x3FFF0; for(int w=0;w<4;w++){uint32_t v;memcpy(&v,ls+addr+w*4,4);RW(rt,w)=bswap32(v);} break; }\n"
"            case 0x067: { uint32_t addr=(pc+(uint32_t)(i16<<2))&0x3FFF0; for(int w=0;w<4;w++){uint32_t v;memcpy(&v,ls+addr+w*4,4);RW(rt,w)=bswap32(v);} break; }\n"
"            case 0x041: { uint32_t addr=((uint32_t)(i16<<2))&0x3FFF0; for(int w=0;w<4;w++){uint32_t v=bswap32(RW(rt,w));memcpy(ls+addr+w*4,&v,4);} break; }\n"
"            case 0x047: { uint32_t addr=(pc+(uint32_t)(i16<<2))&0x3FFF0; for(int w=0;w<4;w++){uint32_t v=bswap32(RW(rt,w));memcpy(ls+addr+w*4,&v,4);} break; }\n"
"            default: handled=false;\n"
"            }\n"
"        }\n"
"\n"
"        if (!handled) {\n"
"            handled = true;\n"
"            switch (op8) {\n"
"            case 0x1C: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)+(uint32_t)i10; break;\n"
"            case 0x1D: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,reg_hw(a,h)+(uint16_t)i10); break; }\n"
"            case 0x0C: for(int w=0;w<4;w++) RW(rt,w)=(uint32_t)i10-RW(ra,w); break;\n"
"            case 0x0D: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,(uint16_t)i10-reg_hw(a,h)); break; }\n"
"            case 0x14: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)&(uint32_t)i10; break;\n"
"            case 0x15: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,reg_hw(a,h)&(uint16_t)i10); break; }\n"
"            case 0x16: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,reg_byte(a,q)&(uint8_t)i10); break; }\n"
"            case 0x04: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)|(uint32_t)i10; break;\n"
"            case 0x05: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,reg_hw(a,h)|(uint16_t)i10); break; }\n"
"            case 0x06: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,reg_byte(a,q)|(uint8_t)i10); break; }\n"
"            case 0x44: for(int w=0;w<4;w++) RW(rt,w)=RW(ra,w)^(uint32_t)i10; break;\n"
"            case 0x45: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,reg_hw(a,h)^(uint16_t)i10); break; }\n"
"            case 0x46: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,reg_byte(a,q)^(uint8_t)i10); break; }\n"
"            case 0x7C: for(int w=0;w<4;w++) RW(rt,w)=(RW(ra,w)==(uint32_t)i10)?0xFFFFFFFF:0; break;\n"
"            case 0x7D: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,(reg_hw(a,h)==(uint16_t)i10)?0xFFFF:0); break; }\n"
"            case 0x7E: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,(reg_byte(a,q)==(uint8_t)i10)?0xFF:0); break; }\n"
"            case 0x4C: for(int w=0;w<4;w++) RW(rt,w)=((int32_t)RW(ra,w)>(int32_t)i10)?0xFFFFFFFF:0; break;\n"
"            case 0x4D: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,((int16_t)reg_hw(a,h)>(int16_t)(uint16_t)i10)?0xFFFF:0); break; }\n"
"            case 0x4E: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,((int8_t)reg_byte(a,q)>(int8_t)(uint8_t)i10)?0xFF:0); break; }\n"
"            case 0x5C: for(int w=0;w<4;w++) RW(rt,w)=(RW(ra,w)>(uint32_t)i10)?0xFFFFFFFF:0; break;\n"
"            case 0x5D: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int h=0;h<8;h++) set_reg_hw(d,h,(reg_hw(a,h)>(uint16_t)i10)?0xFFFF:0); break; }\n"
"            case 0x5E: { uint32_t*a=&regs[ra*4],*d=&regs[rt*4]; for(int q=0;q<16;q++) set_reg_byte(d,q,(reg_byte(a,q)>(uint8_t)i10)?0xFF:0); break; }\n"
"            case 0x34: { uint32_t addr=((uint32_t)((int32_t)RW(ra,0)+(i10<<4)))&0x3FFF0; for(int w=0;w<4;w++){uint32_t v;memcpy(&v,ls+addr+w*4,4);RW(rt,w)=bswap32(v);} break; }\n"
"            case 0x24: { uint32_t addr=((uint32_t)((int32_t)RW(ra,0)+(i10<<4)))&0x3FFF0; for(int w=0;w<4;w++){uint32_t v=bswap32(RW(rt,w));memcpy(ls+addr+w*4,&v,4);} break; }\n"
"            case 0x74: for(int w=0;w<4;w++) RW(rt,w)=(uint32_t)((int32_t)(int16_t)(RW(ra,w)&0xFFFF)*(int32_t)i10); break;\n"
"            case 0x75: for(int w=0;w<4;w++) RW(rt,w)=(RW(ra,w)&0xFFFF)*(uint32_t)(uint16_t)i10; break;\n"
"            case 0x4F: if((int32_t)RW(ra,0)>(int32_t)i10){info[2]=2;goto done;} break;\n"
"            case 0x5F: if(RW(ra,0)>(uint32_t)i10){info[2]=2;goto done;} break;\n"
"            case 0x7F: if(RW(ra,0)==(uint32_t)i10){info[2]=2;goto done;} break;\n"
"            default: handled=false;\n"
"            }\n"
"        }\n"
"\n"
"        if (!handled) {\n"
"            if (op7 == 0x21) { for(int w=0;w<4;w++) RW(rt,w)=i18; }\n"
"            else if (op7 == 0x08 || op7 == 0x09) { /* HBRA/HBRR: NOP */ }\n"
"            else {\n"
"                uint32_t rt4 = (insn >> 21) & 0x7F;\n"
"                uint32_t rc  = insn & 0x7F;\n"
"                switch (op4) {\n"
"                case 0x8: for(int w=0;w<4;w++) RW(rt4,w)=(RW(rc,w)&RW(rb,w))|(~RW(rc,w)&RW(ra,w)); break;\n"
"                case 0xB: {\n"
"                    uint32_t*ca=&regs[ra*4],*cb=&regs[rb*4],*cc=&regs[rc*4],*cd=&regs[rt4*4];\n"
"                    uint8_t concat[32]; for(int i=0;i<16;i++){concat[i]=reg_byte(ca,i);concat[16+i]=reg_byte(cb,i);}\n"
"                    uint8_t res[16];\n"
"                    for(int i=0;i<16;i++){\n"
"                        uint8_t c=reg_byte(cc,i);\n"
"                        if(c>=0xE0) res[i]=0x80;\n"
"                        else if(c>=0xC0) res[i]=0xFF;\n"
"                        else if(c>=0x80) res[i]=0x00;\n"
"                        else res[i]=concat[c&0x1F];\n"
"                    }\n"
"                    for(int i=0;i<16;i++) set_reg_byte(cd,i,res[i]);\n"
"                    break;\n"
"                }\n"
"                case 0xC: for(int w=0;w<4;w++) RW(rt4,w)=(uint32_t)((int32_t)(int16_t)(RW(ra,w)&0xFFFF)*(int32_t)(int16_t)(RW(rb,w)&0xFFFF)+(int32_t)RW(rc,w)); break;\n"
"                case 0xE: for(int w=0;w<4;w++) RW(rt4,w)=f2u(fmaf(u2f(RW(ra,w)),u2f(RW(rb,w)),u2f(RW(rc,w)))); break;\n"
"                case 0xF: for(int w=0;w<4;w++) RW(rt4,w)=f2u(fmaf(u2f(RW(ra,w)),u2f(RW(rb,w)),-u2f(RW(rc,w)))); break;\n"
"                case 0xD: for(int w=0;w<4;w++) RW(rt4,w)=f2u(-fmaf(u2f(RW(ra,w)),u2f(RW(rb,w)),-u2f(RW(rc,w)))); break;\n"
"                default: info[2]=2; goto done;\n"
"                }\n"
"            }\n"
"        }\n"
"\n"
"        #undef R\n"
"        #undef RW\n"
"        pc = (pc + 4) & 0x3FFFC;\n"
"        cycles++;\n"
"    }\n"
"done:\n"
"    info[0] = pc;\n"
"    info[4] = cycles;\n"
"}\n"
;


// ═══════════════════════════════════════════════════════════════
// Bridge API Implementation
// ═══════════════════════════════════════════════════════════════

int spu_bridge_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

int spu_bridge_init(void) {
    if (g_bridge.initialized) return 0;

    if (!spu_bridge_available()) {
        fprintf(stderr, "[SPU-BRIDGE] No CUDA GPU found\n");
        return -1;
    }

    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&g_bridge.cuCtx, 0, dev);

    // Allocate persistent device memory
    cudaMalloc(&g_bridge.d_ls, LS_SIZE);
    cudaMalloc(&g_bridge.d_regs, 128 * 4 * sizeof(uint32_t));
    cudaMalloc(&g_bridge.d_info, 8 * sizeof(uint32_t));

    memset(&g_bridge.cache, 0, sizeof(g_bridge.cache));
    g_bridge.numEntries = 0;
    g_bridge.totalRuns = 0;
    g_bridge.cacheHits = 0;
    g_bridge.compiles = 0;
    g_bridge.totalExecMs = 0;
    g_bridge.totalCompileMs = 0;
    g_bridge.totalInsns = 0;
    g_bridge.initialized = true;

    fprintf(stderr, "[SPU-BRIDGE] Initialized (256KB LS on GPU)\n");
    return 0;
}

static CUfunction bridge_get_kernel(const uint8_t* ls) {
    // Hash the code portion of LS (first 64KB is typical code region)
    uint64_t h = hash_code(ls, 64 * 1024);

    // Check cache
    for (uint32_t i = 0; i < g_bridge.numEntries; i++) {
        if (g_bridge.cache[i].valid && g_bridge.cache[i].codeHash == h) {
            g_bridge.cache[i].hitCount++;
            g_bridge.cacheHits++;
            return g_bridge.cache[i].cuFunction;
        }
    }

    // Compile fresh kernel
    cudaEvent_t cStart, cStop;
    cudaEventCreate(&cStart); cudaEventCreate(&cStop);
    cudaEventRecord(cStart);

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, spu_kernel_template, "spu_bridge.cu", 0, NULL, NULL);
    const char* opts[] = { "--gpu-architecture=sm_70", "-use_fast_math", "-w" };
    nvrtcResult res = nvrtcCompileProgram(prog, 3, opts);

    if (res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = (char*)malloc(logSize + 1);
        nvrtcGetProgramLog(prog, log); log[logSize] = 0;
        fprintf(stderr, "[SPU-BRIDGE] NVRTC compile failed:\n%s\n", log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return nullptr;
    }

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = (char*)malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUmodule cuMod;
    CUfunction cuFunc;
    cuModuleLoadData(&cuMod, ptx);
    cuModuleGetFunction(&cuFunc, cuMod, "spu_bridge_kernel");
    free(ptx);

    cudaEventRecord(cStop);
    cudaEventSynchronize(cStop);
    float compMs = 0;
    cudaEventElapsedTime(&compMs, cStart, cStop);
    cudaEventDestroy(cStart); cudaEventDestroy(cStop);

    g_bridge.totalCompileMs += compMs;
    g_bridge.compiles++;

    // Store in cache
    if (g_bridge.numEntries < MAX_CACHE) {
        SPUCacheEntry& e = g_bridge.cache[g_bridge.numEntries++];
        e.codeHash = h;
        e.cuModule = cuMod;
        e.cuFunction = cuFunc;
        e.hitCount = 0;
        e.valid = true;
    }

    return cuFunc;
}

int spu_bridge_run(uint8_t* ls, SPUBridgeState* state,
                   uint32_t max_insns, SPUBridgeStats* stats) {
    if (!g_bridge.initialized) {
        if (spu_bridge_init() != 0) return -1;
    }

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart); cudaEventCreate(&tStop);
    cudaEventRecord(tStart);

    // Get compiled kernel (cached or fresh)
    CUfunction func = bridge_get_kernel(ls);
    if (!func) return -1;

    bool cached = (g_bridge.cacheHits > 0 &&
                   g_bridge.cache[g_bridge.numEntries-1].hitCount > 0);

    // Upload LS + registers + info to GPU
    cudaMemcpy(g_bridge.d_ls, ls, LS_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(g_bridge.d_regs, state->gpr, 128 * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t info[8] = { state->pc, state->status, state->stopped, max_insns, 0, 0, 0, 0 };
    cudaMemcpy(g_bridge.d_info, info, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    cudaEvent_t execStart, execStop;
    cudaEventCreate(&execStart); cudaEventCreate(&execStop);
    cudaEventRecord(execStart);

    void* args[] = { &g_bridge.d_ls, &g_bridge.d_regs, &g_bridge.d_info };
    CUresult err = cuLaunchKernel(func, 1,1,1, 1,1,1, 0,0, args, NULL);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[SPU-BRIDGE] Launch failed: %d\n", err);
        cudaEventDestroy(execStart); cudaEventDestroy(execStop);
        cudaEventDestroy(tStart); cudaEventDestroy(tStop);
        return -1;
    }

    cudaEventRecord(execStop);
    cudaEventSynchronize(execStop);
    float execMs = 0;
    cudaEventElapsedTime(&execMs, execStart, execStop);
    cudaEventDestroy(execStart); cudaEventDestroy(execStop);

    // Read back results
    cudaMemcpy(ls, g_bridge.d_ls, LS_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(state->gpr, g_bridge.d_regs, 128 * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(info, g_bridge.d_info, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    state->pc = info[0];
    state->status = info[1];
    state->stopped = info[2];

    uint32_t cyclesRun = info[4];

    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);
    float totalMs = 0;
    cudaEventElapsedTime(&totalMs, tStart, tStop);
    cudaEventDestroy(tStart); cudaEventDestroy(tStop);

    // Update cumulative stats
    g_bridge.totalRuns++;
    g_bridge.totalExecMs += execMs;
    g_bridge.totalInsns += cyclesRun;

    // Fill output stats
    if (stats) {
        stats->total_ms = totalMs;
        stats->exec_ms = execMs;
        stats->cycles = cyclesRun;
        stats->cache_hit = cached ? 1 : 0;
        stats->mips = (execMs > 0) ? (double)cyclesRun / (execMs * 1000.0) : 0;
    }

    return (int)cyclesRun;
}

int spu_bridge_run_cached(uint8_t* ls, SPUBridgeState* state,
                          uint32_t max_insns, SPUBridgeStats* stats) {
    // Same as spu_bridge_run — caching is automatic
    return spu_bridge_run(ls, state, max_insns, stats);
}

void spu_bridge_print_stats(void) {
    double avgMips = (g_bridge.totalExecMs > 0) ?
        (double)g_bridge.totalInsns / (g_bridge.totalExecMs * 1000.0) : 0;

    fprintf(stderr,
        "╔═══════════════════════════════════════════╗\n"
        "║  SPU CUDA Bridge Statistics                ║\n"
        "╠═══════════════════════════════════════════╣\n"
        "║  Total runs:    %-8u                   ║\n"
        "║  Cache hits:    %-8u                   ║\n"
        "║  Compiles:      %-8u                   ║\n"
        "║  Total insns:   %-12llu               ║\n"
        "║  Exec time:     %8.2f ms               ║\n"
        "║  Compile time:  %8.2f ms               ║\n"
        "║  Avg throughput: %.1f MIPS              ║\n"
        "╚═══════════════════════════════════════════╝\n",
        g_bridge.totalRuns,
        g_bridge.cacheHits,
        g_bridge.compiles,
        (unsigned long long)g_bridge.totalInsns,
        g_bridge.totalExecMs,
        g_bridge.totalCompileMs,
        avgMips);
}

void spu_bridge_shutdown(void) {
    if (!g_bridge.initialized) return;

    // Free cached modules
    for (uint32_t i = 0; i < g_bridge.numEntries; i++) {
        if (g_bridge.cache[i].valid && g_bridge.cache[i].cuModule)
            cuModuleUnload(g_bridge.cache[i].cuModule);
    }

    cudaFree(g_bridge.d_ls);
    cudaFree(g_bridge.d_regs);
    cudaFree(g_bridge.d_info);

    cuCtxDestroy(g_bridge.cuCtx);
    g_bridge.initialized = false;
    fprintf(stderr, "[SPU-BRIDGE] Shutdown\n");
}
