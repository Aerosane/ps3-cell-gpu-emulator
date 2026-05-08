# Project Megakernel — GPU-Native PS3 Cell BE Emulator

A CUDA-based emulator that runs the PS3's Cell Broadband Engine entirely on an
NVIDIA GPU. The PPE (PowerPC Processing Element) and 6 SPU (Synergistic
Processing Units) cores execute as a **single cooperative CUDA kernel** on V100
HBM2, with 512 MB of PS3 address space living in GPU global memory.

## Architecture

```
┌──────────────────────────────────────────────┐
│           CUDA Cooperative Kernel             │
│                                               │
│  Block 0 (32 threads)    Block 1 (32 threads) │
│  ┌──────────────┐       ┌──────────────────┐  │
│  │  PPE Core     │       │  SPU 0–5         │  │
│  │  PPC64 interp │       │  128-bit SIMD    │  │
│  │  ~40 instrs   │       │  ~80 instrs      │  │
│  │  FP + branch  │       │  MFC DMA engine  │  │
│  └──────┬───────┘       └────────┬─────────┘  │
│         │     grid.sync()        │             │
│         └────────┬───────────────┘             │
│                  ▼                              │
│         CellMailbox (global mem)               │
│         atomic IPC · mailboxes · signals       │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │  512 MB PS3 Sandbox (HBM2)              │  │
│  │  0x00000000  Main RAM  (256 MB)         │  │
│  │  0x10000000  VRAM      (256 MB)         │  │
│  │  + 6 × 256 KB SPU Local Store (1.5 MB)  │  │
│  └─────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

## Components

| File | Description |
|---|---|
| `ppc_defs.h` | PPC64 register state, opcode enums, field extraction macros, PS3 memory map, CellOS syscall numbers |
| `ppc_interpreter.cu` | CUDA PPE interpreter — ~40 PPC instructions, HLE syscalls, `__byte_perm` BE↔LE swap |
| `spu_defs.h` | SPU 128-bit QWord registers, MFC DMA structures, channel enums, instruction format opcodes |
| `spu_interpreter.cu` | CUDA SPU interpreter — ~80 instructions across all formats, MFC DMA, SHUFB, Local Store |
| `cell_megakernel.cu` | Cooperative Cell BE hypervisor — PPE + 6×SPU via `cooperative_groups::grid`, atomic IPC |
| `spu_jit.cu` | Phase 2a per-block NVRTC JIT compiler (superseded by mega) |
| `spu_jit_mega.cu` | Phase 2b persistent megakernel JIT — BFS block discovery → single CUDA kernel |
| `spu_jit_turbo.cu` | **Phase 2c turbo JIT** — register promotion + unrolled SIMD + multi-instance SPURS |
| `test_cell.cu` | 5-test validation suite + PPE throughput benchmark |
| `test_jit.cu` | Per-block JIT tests |
| `test_mega_jit.cu` | Megakernel JIT tests (4/4 pass) |
| `test_turbo_jit.cu` | Turbo JIT tests + SPURS multi-instance scaling (3/3 pass) |
| `build.sh` | Build script targeting `sm_70` (V100) with relocatable device code |

## Building

Requires CUDA 12.x and an sm_70+ GPU (Tesla V100 / RTX 20xx+).

```bash
cd ps3
bash build.sh
./cell_test
```

## Test Results (V100-PCIE-16GB)

```
Phase 1 — Interpreter:
  ✅ Test 1: PPE Integer ALU          — add, mulli, store verified
  ✅ Test 2: PPE Branch + Loop        — sum(1..10) = 55
  ✅ Test 3: SPU 128-bit SIMD Integer — 4-wide add/sub across all lanes
  ✅ Test 4: SPU Float4 SIMD          — fa, fm, fma on {1,2,3,4}×{5,6,7,8}
  ✅ Test 5: Cell Cooperative Kernel   — PPE + SPU0 cross-block with grid.sync()
  Benchmark: 1.3 MIPS (interpreter baseline)

Phase 2b — Megakernel JIT:
  ✅ 4/4 tests, 12.9 MIPS (10.1× over interpreter)

Phase 2c — Turbo JIT (register promotion + SPURS):
  ✅ Test 1: Turbo Correctness — float SIMD fa, fm, fma verified
  ✅ Test 2: Triple Benchmark:
    ┌─────────────────────────────────────────┐
    │ Interpreter:   1.3 MIPS  (baseline)     │
    │ MegaJIT:      12.9 MIPS  (10.1× interp) │
    │ TurboJIT:     52.2 MIPS  (41.0× interp) │
    │ Turbo/Mega:    4.0×                      │
    └─────────────────────────────────────────┘
  ✅ Test 3: SPURS Multi-Instance Scaling:
      1 thread:    42.5 MIPS
     80 threads:  3,364 MIPS   (79× scale)
    640 threads: 26,074 MIPS  (613× scale)
   5120 threads: 204,337 MIPS (4807× scale) ← 200 GIPS from 5120 CUDA cores!
```

## Key Design Decisions

- **Big-endian handling**: `__byte_perm(x, 0, 0x0123)` — single PTX instruction bswap32
- **SPU decode priority**: Widest opcode first (11-bit → 9-bit → 8-bit → 7-bit → 4-bit)
  to avoid format collisions where e.g. `fa` (11-bit 0x2C4) has top 4 bits matching
  RRR opcode DFNMS
- **Cooperative kernel**: `cudaLaunchCooperativeKernel` with 2 blocks × 32 threads,
  `grid.sync()` at time-slice boundaries for PPE↔SPU visibility
- **Memory**: Single `cudaMalloc(512MB)` for PS3 sandbox; SPU LS separately allocated

## Roadmap

| Phase | Target | Approach | Status |
|---|---|---|---|
| Phase 1 | Interpreter (1.3 MIPS) | Switch-dispatch loop, global mem fetch per insn | ✅ Done |
| Phase 2a | Per-block JIT (2.4×) | NVRTC basic-block → kernel per block | ✅ Done (superseded) |
| Phase 2b | Megakernel JIT (12.9 MIPS) | All blocks → single persistent kernel | ✅ Done |
| Phase 2c | Turbo JIT (52 MIPS / 204 GIPS) | Register promotion + SPURS multi-instance | ✅ Done |
| Phase 3 | PPE AOT (~1000 MIPS) | LLVM IR → PTX for hot PPE blocks | 🔜 Next |
| Phase 4 | RSX GPU | Vulkan command translation via VK_RT layer patterns | Planned |

## Lineage

Architectural patterns borrowed from the [VK_RT](https://github.com/Aerosane/vk-rt-cuda-raytracing) Vulkan interception
layer — specifically `rt_ir_exec.cu`'s per-thread slot bank + switch dispatch loop,
and the CUDA↔Vulkan fd-import interop pipeline.
