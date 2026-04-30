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

    // cellSpurs — SPURS task scheduler
    { 0x70e3d58a, "cellSpurs", "cellSpursInitialize" },
    { 0xb0c0d66a, "cellSpurs", "cellSpursInitializeWithAttribute" },
    { 0xf7234b81, "cellSpurs", "cellSpursInitializeWithAttribute2" },
    { 0xe48bf572, "cellSpurs", "cellSpursFinalize" },
    { 0xbc7001af, "cellSpurs", "cellSpursAttachLv2EventQueue" },
    { 0xe74dc23c, "cellSpurs", "cellSpursDetachLv2EventQueue" },
    { 0x1bf8a000, "cellSpurs", "cellSpursGetNumSpuThread" },
    { 0xe2df437e, "cellSpurs", "cellSpursGetSpuThreadGroupId" },
    { 0x3942f6bf, "cellSpurs", "cellSpursSetMaxContention" },
    { 0x79b36c3e, "cellSpurs", "cellSpursSetPriorities" },
    { 0x6c01b4fb, "cellSpurs", "cellSpursAttributeSetNamePrefix" },
    { 0x3f422300, "cellSpurs", "cellSpursAttributeSetSpuThreadGroupType" },
    { 0xc23c4581, "cellSpurs", "cellSpursAttributeEnableSpuPrintfIfAvailable" },
    { 0xc97cec02, "cellSpurs", "cellSpursCreateTask" },
    { 0x3fd6c11a, "cellSpurs", "cellSpursCreateTaskset" },
    { 0xefa6c145, "cellSpurs", "cellSpursCreateTasksetWithAttribute" },
    { 0xe591f650, "cellSpurs", "cellSpursJoinTaskset" },
    { 0x899934dc, "cellSpurs", "cellSpursShutdownTaskset" },
    { 0xe7fc14af, "cellSpurs", "cellSpursTasksetAttributeSetName" },
    { 0xbc3dab02, "cellSpurs", "cellSpursEventFlagInitialize" },
    { 0x36c4f4a8, "cellSpurs", "cellSpursEventFlagSet" },
    { 0xe82a263a, "cellSpurs", "cellSpursEventFlagWait" },
    { 0x2f0a4998, "cellSpurs", "cellSpursEventFlagClear" },
    { 0x144d8656, "cellSpurs", "cellSpursGetInfo" },
    { 0xc41beb33, "cellSpurs", "cellSpursSetGlobalExceptionEventHandler" },
    { 0x47b25489, "cellSpurs", "cellSpursAttributeInitialize" },

    // cellGame — game boot/content management
    { 0x188beb05, "cellGame", "cellGameBootCheck" },
    { 0xd0536716, "cellGame", "cellGameContentPermit" },
    { 0x4cbca5b6, "cellGame", "cellGamePatchCheck" },
    { 0x405e5cba, "cellGame", "cellGameDataCheck" },
    { 0x0fa06f6d, "cellGame", "cellGameGetParamInt" },
    { 0xde9c0881, "cellGame", "cellGameGetParamString" },
    { 0xd1a90b31, "cellGame", "cellGameGetSizeKB" },
    { 0xc707e826, "cellGame", "cellGameSetParamString" },
    { 0xfadd9ad8, "cellGame", "cellGameCreateGameData" },
    { 0xf2364153, "cellGame", "cellGameDeleteGameData" },

    // cellSaveData — save/load management
    { 0xf0f530b7, "cellSaveData", "cellSaveDataAutoSave2" },
    { 0x590e6d0b, "cellSaveData", "cellSaveDataAutoLoad2" },
    { 0x3604d4f4, "cellSaveData", "cellSaveDataListSave2" },
    { 0xd739cc4b, "cellSaveData", "cellSaveDataListLoad2" },
    { 0x38a0f7d2, "cellSaveData", "cellSaveDataFixedSave2" },
    { 0x7b5e041a, "cellSaveData", "cellSaveDataFixedLoad2" },
    { 0x1c8b05e2, "cellSaveData", "cellSaveDataDelete2" },

    // cellAudio — audio output
    { 0x980f750b, "cellAudio", "cellAudioInit" },
    { 0x81df6d86, "cellAudio", "cellAudioQuit" },
    { 0x708017e5, "cellAudio", "cellAudioPortOpen" },
    { 0xebb73ea7, "cellAudio", "cellAudioPortClose" },
    { 0x8a8c0417, "cellAudio", "cellAudioPortStart" },
    { 0x3ba2ba64, "cellAudio", "cellAudioPortStop" },
    { 0xffd2b376, "cellAudio", "cellAudioGetPortConfig" },
    { 0x69792b3b, "cellAudio", "cellAudioGetPortTimestamp" },
    { 0x20e88c87, "cellAudio", "cellAudioSetNotifyEventQueue" },
    { 0x83afa9b3, "cellAudio", "cellAudioRemoveNotifyEventQueue" },

    // cellFont — font rendering
    { 0x4b734c8c, "cellFont", "cellFontInit" },
    { 0x3e3712ed, "cellFont", "cellFontEnd" },
    { 0x62f4f193, "cellFont", "cellFontOpenFontFile" },
    { 0x21ef94ac, "cellFont", "cellFontOpenFontMemory" },
    { 0x36149928, "cellFont", "cellFontCreateRenderer" },
    { 0x4b02499c, "cellFont", "cellFontRenderCharGlyphImage" },
    { 0xb61f6678, "cellFont", "cellFontGetHorizontalLayout" },
    { 0xa19dbfba, "cellFont", "cellFontSetupRenderScalePixel" },
    { 0x61717f26, "cellFont", "cellFontGetRenderCharGlyphMetrics" },
    { 0x881e825b, "cellFont", "cellFontSetScalePixel" },
    { 0x6dc15d05, "cellFont", "cellFontBindRenderer" },
    { 0x3085a953, "cellFont", "cellFontGetFontIdCode" },

    // sceNpTrophy — trophy system
    { 0xdd74bdae, "sceNpTrophy", "sceNpTrophyInit" },
    { 0xd972691b, "sceNpTrophy", "sceNpTrophyTerm" },
    { 0x7aa2fcff, "sceNpTrophy", "sceNpTrophyCreateContext" },
    { 0xc10b8fce, "sceNpTrophy", "sceNpTrophyCreateHandle" },
    { 0x287f6018, "sceNpTrophy", "sceNpTrophyRegisterContext" },
    { 0x0cde7100, "sceNpTrophy", "sceNpTrophyDestroyContext" },
    { 0x7775c461, "sceNpTrophy", "sceNpTrophyDestroyHandle" },
    { 0x7a3ceb91, "sceNpTrophy", "sceNpTrophyUnlockTrophy" },
    { 0xc868dea5, "sceNpTrophy", "sceNpTrophyGetGameProgress" },
    { 0x3fbae39e, "sceNpTrophy", "sceNpTrophyGetGameInfo" },
};

static inline const PpuHleEntry* ppu_hle_lookup(uint32_t fnid) {
    constexpr size_t N = sizeof(PPU_HLE_NAMES) / sizeof(PpuHleEntry);
    for (size_t i = 0; i < N; ++i) {
        if (PPU_HLE_NAMES[i].fnid == fnid) return &PPU_HLE_NAMES[i];
    }
    return nullptr;
}
