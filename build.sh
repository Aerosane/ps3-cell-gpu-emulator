#!/bin/bash
# Build PS3 Cell BE GPU Emulator
# Target: Tesla V100 (sm_70), CUDA 12.5
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS="-O3 -arch=sm_70 --compiler-options=-fPIC -Wno-deprecated-gpu-targets"

echo "╔══════════════════════════════════════════╗"
echo "║  🎮 Project Megakernel — PS3 Emulator   ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Target: Tesla V100 (sm_70)             ║"
echo "║  CUDA: 12.5                             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

echo "[1/4] Building PPE interpreter..."
$NVCC $NVCC_FLAGS -c ppc_interpreter.cu -o ppc_interpreter.o

echo "[2/4] Building SPU interpreter..."
$NVCC $NVCC_FLAGS -c spu_interpreter.cu -o spu_interpreter.o

echo "[3/4] Building Cell cooperative megakernel..."
$NVCC $NVCC_FLAGS -rdc=true -c cell_megakernel.cu -o cell_megakernel.o

echo "[4/5] Building test harness..."
$NVCC $NVCC_FLAGS -rdc=true \
  test_cell.cu ppc_interpreter.o spu_interpreter.o cell_megakernel.o \
  -lcudadevrt \
  -o cell_test

echo "[5/6] Building SPU JIT compiler + tests..."
$NVCC $NVCC_FLAGS \
  test_jit.cu spu_jit.cu spu_interpreter.o \
  -lnvrtc -lcuda \
  -o jit_test

echo "[6/6] Building SPU Megakernel JIT..."
$NVCC $NVCC_FLAGS \
  test_mega_jit.cu spu_jit_mega.cu spu_interpreter.o \
  -lnvrtc -lcuda \
  -o mega_jit_test

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Build complete: ./cell_test  ./jit_test  ./mega_jit_test"
echo "═══════════════════════════════════════════"
