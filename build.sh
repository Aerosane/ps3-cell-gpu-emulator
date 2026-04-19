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

echo "[12/29] Building cellGcm HLE shim test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_hle.cu rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_hle_test

echo "[13/29] Building MFC DMA engine test..."
$NVCC $NVCC_FLAGS \
  test_mfc_dma.cu \
  -o mfc_dma_test

echo "[14/29] Building SPU channels test..."
$NVCC $NVCC_FLAGS \
  test_spu_channels.cu \
  -o spu_channels_test

echo "[15/29] Building ELF loader unit tests..."
$NVCC $NVCC_FLAGS \
  test_elf_loader.cu \
  -o elf_loader_test

echo "[16/29] Building PPC HLE syscall → RSX FIFO bridge test..."
$NVCC $NVCC_FLAGS \
  test_gcm_syscall.cu ppc_interpreter.o rsx_command_processor.cu \
  -o gcm_syscall_test

echo "[17/29] Building PPC-driven FULL FRAME render test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_frame.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_frame_test

echo "[18/29] Building PPC-driven multi-primitive test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_prims.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_prims_test

echo "[19/29] Building PPC-driven depth-test scene..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_depth.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_depth_test

echo "[20/29] Building PPC-driven alpha-blend test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_blend.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_blend_test

echo "[21/29] Building PPC-driven scissor test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_scissor.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_scissor_test

echo "[22/29] Building PPC-driven back-face cull test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_cull.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_cull_test

echo "[23/29] Building PS3 ELF → PPC → RSX end-to-end test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_elf.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_elf_test

echo "[24/29] Building PPC-driven two-pass stencil masking test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_stencil.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_stencil_test

echo "[25/29] Building PPC-driven indexed draw test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_indexed.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_indexed_test

echo "[26/29] Building PPC-driven vertex-program upload test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_vp.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_vp_test

echo "[27/29] Building PPC-driven FP + multi-texture-unit test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_fp_tex.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_fp_tex_test

echo "[28/29] Building PPC-driven MRT surface setup test..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_mrt.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_mrt_test

echo "[29/29] Building PPC-driven VP execution test (real microcode)..."
$NVCC $NVCC_FLAGS --extended-lambda \
  test_gcm_vp_exec.cu ppc_interpreter.o \
  rsx_command_processor.cu rsx_raster.cu rsx_raster_bridge.cpp \
  -o gcm_vp_exec_test

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Build complete"
echo "  ./cell_test  ./jit_test  ./mega_jit_test"
echo "  ./turbo_test  ./hyper_test  ./ppc_jit_test"
echo "  ./rsx_vulkan_test  ./rsx_replay_test"
echo "  ./rsx_raster_test  ./rsx_bridge_test"
echo "  ./elf_boot_test  ./gcm_hle_test"
echo "  ./mfc_dma_test  ./spu_channels_test  ./elf_loader_test"
echo "  ./gcm_syscall_test  ./gcm_frame_test  ./gcm_prims_test  ./gcm_depth_test  ./gcm_blend_test  ./gcm_scissor_test  ./gcm_cull_test  ./gcm_elf_test  ./gcm_stencil_test  ./gcm_indexed_test  ./gcm_vp_test  ./gcm_fp_tex_test  ./gcm_mrt_test  ./gcm_vp_exec_test"
echo "═══════════════════════════════════════════"
