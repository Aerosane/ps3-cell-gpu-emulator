#!/bin/bash
# run_all_tests.sh — Execute every built test_* binary in this directory
# with a timeout, summarize pass/fail, exit non-zero on any failure.
#
# Use-case: single-command regression check after any change to the
# emulator. Does NOT rebuild — run ./build.sh first.

set -u
cd "$(dirname "$0")"

# Known test binaries — keep alphabetized for stable output.
TESTS=(
    backend_test
    bridge_test
    cell_test
    elf_boot_test
    elf_loader_test
    gcm_blend_test
    gcm_cull_test
    gcm_depth_test
    gcm_elf_test
    gcm_fp_tex_test
    gcm_frame_test
    gcm_hle_test
    gcm_indexed_test
    gcm_mrt_test
    gcm_prims_test
    gcm_scissor_test
    gcm_stencil_test
    gcm_syscall_test
    gcm_vp_exec_test
    gcm_vp_test
    hyper_test
    jit_test
    megakernel_rescue_test
    mega_jit_test
    mfc_dma_test
    ppc_jit_test
    ppc_vmx_test
    ppu_hle_resolve_test
    real_self_disasm_test
    real_self_exec_test
    real_self_test
    rsx_bridge_test
    rsx_raster_test
    rsx_replay_test
    rsx_vulkan_test
    self_phdr_scan_test
    spu_channels_test
    test_channels
    test_mfc_dma
    test_rsx_shaders
    triangle_boot_test
    cube_boot_test
    turbo_test
)

TIMEOUT=${TEST_TIMEOUT:-180}
PASS=0
FAIL=0
SKIP=0
FAILED=()

echo "╔══════════════════════════════════════════════════════╗"
echo "║  PS3 Megakernel — Regression Test Suite              ║"
echo "╠══════════════════════════════════════════════════════╣"
printf "║  timeout per test: %3ds                              ║\n" "$TIMEOUT"
echo "╚══════════════════════════════════════════════════════╝"

for t in "${TESTS[@]}"; do
    if [ ! -x "./$t" ]; then
        printf "  SKIP  %-32s  (not built)\n" "$t"
        SKIP=$((SKIP+1))
        continue
    fi
    # Capture rc only; discard stdout/stderr for brevity. On fail, rerun
    # to capture the tail for diagnostic.
    timeout "$TIMEOUT" "./$t" >/dev/null 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        printf "  PASS  %-32s\n" "$t"
        PASS=$((PASS+1))
    else
        printf "  FAIL  %-32s  rc=%d\n" "$t" "$rc"
        FAIL=$((FAIL+1))
        FAILED+=("$t")
    fi
done

echo "─────────────────────────────────────────────────────────"
echo "Summary: PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
if [ "$FAIL" -gt 0 ]; then
    echo "Failing tests:"
    for t in "${FAILED[@]}"; do echo "  - $t"; done
    exit 1
fi
exit 0
