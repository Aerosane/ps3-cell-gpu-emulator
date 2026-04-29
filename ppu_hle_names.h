// ppu_hle_names.h — static FNID → PS3 function name table for the
// subset of imports used by RPCS3's gs_gcm_basic_triangle.elf sample.
// FNIDs are 32-bit truncated SHA-1 hashes (per ppu_generate_id in
// rpcs3/Emu/Cell/PPUModule.cpp). This table is generated offline by
// SHA-1-hashing known PS3 API names against the Sony-specific 16-byte
// suffix.
//
// A future HLE dispatcher keys `{module, fnid} -> handler` off this
// table.

#pragma once
#include <cstdint>
#include <cstring>

struct PpuHleEntry {
    uint32_t    fnid;
    const char* module;
    const char* name;
};

// Hand-generated via offline SHA-1; see
// rpcs3-src/rpcs3/Emu/Cell/PPUModule.cpp:ppu_generate_id.
static const PpuHleEntry PPU_HLE_NAMES[] = {
    // sys_fs
    { 0x718bf5f8, "sys_fs", "cellFsOpen" },
    { 0x4d5ff8e2, "sys_fs", "cellFsRead" },
    { 0xecdcf2ab, "sys_fs", "cellFsWrite" },
    { 0x2cb51f0d, "sys_fs", "cellFsClose" },
    { 0xa397d042, "sys_fs", "cellFsLseek" },
    { 0xef3efa34, "sys_fs", "cellFsFstat" },

    // cellSysutil
    { 0x02ff3c1b, "cellSysutil", "cellSysutilUnregisterCallback" },
    { 0x0bae8772, "cellSysutil", "cellVideoOutConfigure" },
    { 0x189a74da, "cellSysutil", "cellSysutilCheckCallback" },
    { 0x40e895d3, "cellSysutil", "cellSysutilGetSystemParamInt" },
    { 0x887572d5, "cellSysutil", "cellVideoOutGetState" },
    { 0x938013a0, "cellSysutil", "cellVideoOutGetResolutionAvailability" },
    { 0x9d98afa0, "cellSysutil", "cellSysutilRegisterCallback" },
    { 0xe558748d, "cellSysutil", "cellVideoOutGetResolution" },

    // cellGcmSys
    { 0x055bd74d, "cellGcmSys",  "cellGcmGetTiledPitchSize" },
    { 0x0e6b0dae, "cellGcmSys",  "cellGcmGetCurrentField" },
    { 0x15bae46b, "cellGcmSys",  "_cellGcmInitBody" },
    { 0x21397818, "cellGcmSys",  "_cellGcmSetFlipCommand" },
    { 0x21ac3697, "cellGcmSys",  "cellGcmAddressToOffset" },
    { 0x3a33c1fd, "cellGcmSys",  "_cellGcmFunc15" },
    { 0x4524cccd, "cellGcmSys",  "cellGcmBindTile" },
    { 0x4ae8d215, "cellGcmSys",  "cellGcmSetFlipMode" },
    { 0x51c9d62b, "cellGcmSys",  "cellGcmSetDebugOutputLevel" },
    { 0x5a41c10f, "cellGcmSys",  "cellGcmGetTimeStamp" },
    { 0x5e2ee0f0, "cellGcmSys",  "cellGcmGetDefaultCommandWordSize" },
    { 0x723c5c6c, "cellGcmSys",  "cellGcmSetPrepareFlip" },
    { 0x72a577ce, "cellGcmSys",  "cellGcmGetFlipStatus" },
    { 0x8cdf8c70, "cellGcmSys",  "cellGcmGetDefaultSegmentWordSize" },
    { 0x9a4c1b5f, "cellGcmSys",  "cellGcmSetFlipHandler" },
    { 0x9ba451e4, "cellGcmSys",  "cellGcmSetDefaultFifoSize" },
    { 0x9dc04436, "cellGcmSys",  "cellGcmBindZcull" },
    { 0xa114ec67, "cellGcmSys",  "cellGcmMapMainMemory" },
    { 0xa53d12ae, "cellGcmSys",  "cellGcmSetDisplayBuffer" },
    { 0xa547adde, "cellGcmSys",  "cellGcmGetControlRegister" },
    { 0xa75640e8, "cellGcmSys",  "cellGcmUnbindZcull" },
    { 0xacee8542, "cellGcmSys",  "cellGcmSetWaitFlip" },
    { 0xb2e761d4, "cellGcmSys",  "cellGcmResetFlipStatus" },
    { 0xbd100dbc, "cellGcmSys",  "cellGcmSetTileInfo" },
    { 0xd8f88e1a, "cellGcmSys",  "_cellGcmSetFlipCommandWithWaitLabel" },
    { 0xd9b7653e, "cellGcmSys",  "cellGcmUnbindTile" },
    { 0xe315a0b2, "cellGcmSys",  "cellGcmGetConfiguration" },
    { 0xf80196c1, "cellGcmSys",  "cellGcmGetLabelAddress" },

    // sys_io (pad input)
    { 0x1cf98800, "sys_io",      "cellPadInit" },
    { 0x4d9b75d5, "sys_io",      "cellPadEnd" },
    { 0x8b72cda1, "sys_io",      "cellPadGetData" },
    { 0xa703a51d, "sys_io",      "cellPadGetInfo2" },

    // sysPrxForUser
    { 0x1573dc3f, "sysPrxForUser", "sys_lwmutex_lock" },
    { 0x1bc200f4, "sysPrxForUser", "sys_lwmutex_unlock" },
    { 0x2c847572, "sysPrxForUser", "_sys_process_atexitspawn" },
    { 0x2d36462b, "sysPrxForUser", "_sys_strlen" },
    { 0x2f85c0ef, "sysPrxForUser", "sys_lwmutex_create" },
    { 0x350d454e, "sysPrxForUser", "sys_ppu_thread_get_id" },
    { 0x35168520, "sysPrxForUser", "_sys_heap_malloc" },
    { 0x42b23552, "sysPrxForUser", "sys_prx_register_library" },
    { 0x67f9fedb, "sysPrxForUser", "sys_game_process_exitspawn2" },
    { 0x744680a2, "sysPrxForUser", "sys_initialize_tls" },
    { 0x8461e528, "sysPrxForUser", "sys_time_get_system_time" },
    { 0x8a561d92, "sysPrxForUser", "_sys_heap_free" },
    { 0x96328741, "sysPrxForUser", "_sys_process_at_Exitspawn" },
    { 0x9f04f7af, "sysPrxForUser", "_sys_printf" },
    { 0xa2c7ba64, "sysPrxForUser", "sys_prx_exitspawn_with_level" },
    { 0xaede4b03, "sysPrxForUser", "_sys_heap_delete_heap" },
    { 0xb2fcf2c8, "sysPrxForUser", "_sys_heap_create_heap" },
    { 0xc3476d0c, "sysPrxForUser", "sys_lwmutex_destroy" },
    { 0xe6f2c1e7, "sysPrxForUser", "sys_process_exit" },
};

static inline const PpuHleEntry* ppu_hle_lookup(uint32_t fnid) {
    constexpr size_t N = sizeof(PPU_HLE_NAMES) / sizeof(PpuHleEntry);
    for (size_t i = 0; i < N; ++i) {
        if (PPU_HLE_NAMES[i].fnid == fnid) return &PPU_HLE_NAMES[i];
    }
    return nullptr;
}
