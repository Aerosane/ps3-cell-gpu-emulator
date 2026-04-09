#!/bin/bash
# Build PS3 Cell BE GPU Emulator
# Target: Tesla V100 (sm_70), CUDA 12.9
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
NVCC=/usr/local/cuda-12.9/bin/nvcc
NVCC_FLAGS="-O3 -arch=sm_70 --compiler-options=-fPIC -Wno-deprecated-gpu-targets"

echo "╔══════════════════════════════════════════╗"
echo "║  🎮 Project Megakernel — PS3 Emulator   ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Target: Tesla V100 (sm_70)             ║"
echo "║  CUDA: 12.9                             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

echo "[1/5] Building PPE interpreter..."
$NVCC $NVCC_FLAGS -c ppc_interpreter.cu -o ppc_interpreter.o

echo "[2/5] Building SPU interpreter..."
$NVCC $NVCC_FLAGS -c spu_interpreter.cu -o spu_interpreter.o

echo "[3/5] Building Cell cooperative megakernel..."
$NVCC $NVCC_FLAGS -rdc=true -c cell_megakernel.cu -o cell_megakernel.o

echo "[4/5] Building test harness..."
$NVCC $NVCC_FLAGS -rdc=true \
  test_cell.cu ppc_interpreter.o spu_interpreter.o cell_megakernel.o \
  -lcudadevrt \
  -o cell_test

echo "[5/5] Building SPU JIT stack (per-block + mega + turbo + hyper)..."
$NVCC $NVCC_FLAGS \
  test_jit.cu spu_jit.cu spu_interpreter.o \
  -lnvrtc -lcuda \
  -o jit_test

$NVCC $NVCC_FLAGS \
  test_mega_jit.cu spu_jit_mega.cu spu_interpreter.o \
  -lnvrtc -lcuda \
  -o mega_jit_test

$NVCC $NVCC_FLAGS \
  test_turbo_jit.cu spu_jit_turbo.cu spu_jit_mega.cu spu_interpreter.o \
  -lnvrtc -lcuda \
  -o turbo_test

$NVCC $NVCC_FLAGS \
  test_hyper_jit.cu spu_jit_hyper.cu spu_jit_turbo.cu spu_jit_mega.cu spu_interpreter.o \
  -lnvrtc -lcuda \
  -o hyper_test

echo "[6/6] Building PPE JIT test..."
$NVCC $NVCC_FLAGS \
  test_ppc_jit.cu ppc_jit.cu ppc_interpreter.o \
  -lnvrtc -lcuda \
  -o ppc_jit_test

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Build complete"
echo "  ./cell_test  ./jit_test  ./mega_jit_test"
echo "  ./turbo_test  ./hyper_test  ./ppc_jit_test"
echo "═══════════════════════════════════════════"
