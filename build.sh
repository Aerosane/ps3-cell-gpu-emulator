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
  test_cell.cu ppc_interpreter.o spu_interpreter.o cell_megakernel.o ppc_jit.cu \
  -lcudadevrt -lnvrtc -lcuda \
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

echo "[7/7] Building RSX → Vulkan emitter test..."
$NVCC $NVCC_FLAGS \
  test_rsx_vulkan.cu rsx_command_processor.cu rsx_vulkan_emitter.cpp rsx_vulkan_emitter_shim.cpp \
  -o rsx_vulkan_test

echo "[8/8] Building RSX full-pipeline replay demo..."
$NVCC $NVCC_FLAGS \
  test_rsx_replay.cu rsx_command_processor.cu rsx_vulkan_emitter.cpp rsx_vulkan_emitter_shim.cpp rsx_soft_replayer.cpp \
  -o rsx_replay_test

echo "[9/10] Building RSX CUDA rasterizer test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_rsx_raster.cu rsx_raster.cu \
  -o rsx_raster_test

echo "[10/10] Building RSX → bridge → rasterizer end-to-end test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_rsx_bridge.cu rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o rsx_bridge_test

echo "[11/11] Building ELF loader → PPC warp-JIT bringup test..."
$NVCC $NVCC_FLAGS \
  test_elf_boot.cu ppc_jit.cu \
  -lnvrtc -lcuda \
  -o elf_boot_test

echo "[12/12] Building cellGcm HLE shim test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_hle.cu rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_hle_test

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Build complete"
echo "  ./cell_test  ./jit_test  ./mega_jit_test"
echo "  ./turbo_test  ./hyper_test  ./ppc_jit_test"
echo "  ./rsx_vulkan_test  ./rsx_replay_test"
echo "  ./rsx_raster_test  ./rsx_bridge_test"
echo "  ./elf_boot_test  ./gcm_hle_test"
echo "═══════════════════════════════════════════"
