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

    // cellFs extras (beyond sys_fs basics)
    { 0x9eaf0f23, "cellFs", "cellFsOpendir" },
    { 0x49e735fd, "cellFs", "cellFsReaddir" },
    { 0xe1464921, "cellFs", "cellFsClosedir" },
    { 0x0a7faeba, "cellFs", "cellFsStat" },
    { 0x05b82844, "cellFs", "cellFsMkdir" },
    { 0xadff8f10, "cellFs", "cellFsRmdir" },
    { 0x40ba070f, "cellFs", "cellFsUnlink" },
    { 0x3591cf02, "cellFs", "cellFsRename" },
    { 0x60a6c70f, "cellFs", "cellFsTruncate" },
    { 0xe7d6ba00, "cellFs", "cellFsGetFreeSize" },

    // cellSysmodule — PRX module loading
    { 0xb8a0bf48, "cellSysmodule", "cellSysmoduleLoadModule" },
    { 0x2a1321c1, "cellSysmodule", "cellSysmoduleUnloadModule" },
    { 0x452adbc2, "cellSysmodule", "cellSysmoduleIsLoaded" },
    { 0x6e0040b4, "cellSysmodule", "cellSysmoduleInitialize" },
    { 0x8b59a7b1, "cellSysmodule", "cellSysmoduleFinalize" },

    // cellNetCtl — network control
    { 0xf53f04bb, "cellNetCtl", "cellNetCtlInit" },
    { 0x9f18ccad, "cellNetCtl", "cellNetCtlTerm" },
    { 0x5c413ca9, "cellNetCtl", "cellNetCtlGetState" },
    { 0x08a8a347, "cellNetCtl", "cellNetCtlGetInfo" },
    { 0x2e274b74, "cellNetCtl", "cellNetCtlAddHandler" },
    { 0x438cee7b, "cellNetCtl", "cellNetCtlDelHandler" },

    // cellMsgDialog — message/progress dialog
    { 0x503d1913, "cellMsgDialog", "cellMsgDialogOpen" },
    { 0x860634da, "cellMsgDialog", "cellMsgDialogOpen2" },
    { 0x7da3e04f, "cellMsgDialog", "cellMsgDialogClose" },
    { 0xf0d11ff1, "cellMsgDialog", "cellMsgDialogAbort" },
    { 0xbb49b97e, "cellMsgDialog", "cellMsgDialogProgressBarInc" },
    { 0x6df96f85, "cellMsgDialog", "cellMsgDialogProgressBarSetMsg" },

    // cellOskDialog — on-screen keyboard
    { 0x36cff0d8, "cellOskDialog", "cellOskDialogLoadAsync" },
    { 0xf5daa620, "cellOskDialog", "cellOskDialogUnloadAsync" },
    { 0xcae76f24, "cellOskDialog", "cellOskDialogGetInputText" },
    { 0x15cb6934, "cellOskDialog", "cellOskDialogSetInitialInputDevice" },

    // cellResc — resolution scaling
    { 0x3711bd4f, "cellResc", "cellRescInit" },
    { 0xcb4ba814, "cellResc", "cellRescExit" },
    { 0x089a5067, "cellResc", "cellRescSetDisplayMode" },
    { 0xffd96327, "cellResc", "cellRescSetConvertAndFlip" },
    { 0x770f5ac4, "cellResc", "cellRescSetBufferAddress" },
    { 0x35239700, "cellResc", "cellRescGetNumColorBuffers" },
    { 0xe99bbc60, "cellResc", "cellRescGetBufferSize" },
    { 0x147dde14, "cellResc", "cellRescSetSrc" },
    { 0xbe0760f8, "cellResc", "cellRescSetDsts" },

    // cellSysutil extras — BGM
    { 0xfc6fe574, "cellSysutil", "cellSysutilGetBgmPlaybackStatus" },
    { 0x799b5fae, "cellSysutil", "cellSysutilEnableBgmPlayback" },
    { 0xda004916, "cellSysutil", "cellSysutilDisableBgmPlayback" },

    // cellKb — keyboard input
    { 0x85656de9, "cellKb", "cellKbInit" },
    { 0x8fe7f827, "cellKb", "cellKbEnd" },
    { 0xead11ae9, "cellKb", "cellKbGetInfo" },
    { 0xda870fc8, "cellKb", "cellKbRead" },

    // cellMouse
    { 0x890c9fb8, "cellMouse", "cellMouseInit" },
    { 0x67c67673, "cellMouse", "cellMouseEnd" },
    { 0xb4086711, "cellMouse", "cellMouseGetInfo" },

    // cellCamera
    { 0x30eaee99, "cellCamera", "cellCameraInit" },
    { 0xf1e65d8d, "cellCamera", "cellCameraEnd" },

    // cellUserInfo
    { 0xdcf8eb53, "cellUserInfo", "cellUserInfoGetStat" },
    { 0x4a5b749a, "cellUserInfo", "cellUserInfoGetList" },

    // cellSsl / cellHttp — network stubs
    { 0x8c1678ef, "cellSsl", "cellSslInit" },
    { 0x2a574a25, "cellSsl", "cellSslEnd" },
    { 0xf7644439, "cellHttp", "cellHttpInit" },
    { 0x33457271, "cellHttp", "cellHttpEnd" },

    // cellAdec — audio decoder
    { 0x8f027e01, "cellAdec", "cellAdecOpen" },
    { 0xcd96044f, "cellAdec", "cellAdecClose" },
    { 0xcc534abe, "cellAdec", "cellAdecStartSeq" },
    { 0xccfeefc2, "cellAdec", "cellAdecEndSeq" },
    { 0x4d682b55, "cellAdec", "cellAdecDecodeAu" },
    { 0x0cfa3baf, "cellAdec", "cellAdecGetPcm" },
    { 0x382b0cf4, "cellAdec", "cellAdecGetPcmItem" },
    { 0xf07a2ec7, "cellAdec", "cellAdecQueryAttr" },

    // cellDmux — media demuxer
    { 0x2d4ce3df, "cellDmux", "cellDmuxOpen" },
    { 0x438a5a0f, "cellDmux", "cellDmuxClose" },
    { 0xcd076abf, "cellDmux", "cellDmuxSetStream" },
    { 0xabbbb443, "cellDmux", "cellDmuxResetStream" },
    { 0x143a3968, "cellDmux", "cellDmuxQueryAttr" },
    { 0x181d1978, "cellDmux", "cellDmuxEnableEs" },
    { 0x022dda52, "cellDmux", "cellDmuxDisableEs" },
    { 0x7d539c24, "cellDmux", "cellDmuxGetAu" },

    // cellPamf — PAMF container reader
    { 0xb13dac24, "cellPamf", "cellPamfReaderInitialize" },
    { 0x484cfb6f, "cellPamf", "cellPamfReaderGetNumberOfStreams" },
    { 0xbf6c31af, "cellPamf", "cellPamfReaderGetStreamInfo" },
    { 0x79df04e4, "cellPamf", "cellPamfReaderGetStreamTypeCoding" },
    { 0x847d96d4, "cellPamf", "cellPamfReaderGetEsFilterId" },
    { 0xd245f601, "cellPamf", "cellPamfGetHeaderSize" },

    // cellJpgDec / cellPngDec — image decoders
    { 0x8b18c8eb, "cellJpgDec", "cellJpgDecOpen" },
    { 0xbbbfa42c, "cellJpgDec", "cellJpgDecClose" },
    { 0x5920178f, "cellJpgDec", "cellJpgDecReadHeader" },
    { 0xc2a0d674, "cellJpgDec", "cellJpgDecDecodeData" },
    { 0xb4fdda24, "cellJpgDec", "cellJpgDecCreate" },
    { 0xd1f838b9, "cellJpgDec", "cellJpgDecDestroy" },
    { 0xebeab923, "cellPngDec", "cellPngDecOpen" },
    { 0x01c158db, "cellPngDec", "cellPngDecClose" },
    { 0x8cdae535, "cellPngDec", "cellPngDecReadHeader" },
    { 0x994e6188, "cellPngDec", "cellPngDecDecodeData" },
    { 0xf8d2000f, "cellPngDec", "cellPngDecCreate" },
    { 0x7515cc92, "cellPngDec", "cellPngDecDestroy" },

    // ── cellVdec — video decode ─────────────────────────────────
    { 0xA83F253B, "cellVdec", "cellVdecOpen" },
    { 0xB6D5CACD, "cellVdec", "cellVdecClose" },
    { 0x59F481EF, "cellVdec", "cellVdecStartSeq" },
    { 0x9625F90A, "cellVdec", "cellVdecEndSeq" },
    { 0xC1901AB8, "cellVdec", "cellVdecDecodeAu" },
    { 0xAC73ADA8, "cellVdec", "cellVdecGetPicture" },
    { 0x52B9A8B0, "cellVdec", "cellVdecGetPictureExt" },
    { 0x5877B8AB, "cellVdec", "cellVdecSetFrameRate" },
    { 0xA1FFA426, "cellVdec", "cellVdecQueryAttr" },
    { 0x9AFD0B7D, "cellVdec", "cellVdecOpenEx" },

    // cellL10n — localization / string conversion
    { 0x49a66a63, "cellL10n", "UTF8stoUCS2s" },
    { 0xc5925b36, "cellL10n", "UCS2stoUTF8s" },
    { 0x9e989095, "cellL10n", "UTF8toUCS2" },
    { 0x487b4dc1, "cellL10n", "UCS2toUTF8" },
    { 0xa5be179f, "cellL10n", "UTF16stoUTF8s" },
    { 0x13e7cfc1, "cellL10n", "UTF8stoUTF16s" },
    { 0x3b95766d, "cellL10n", "L10nConvertStr" },

    // cellSync — synchronization primitives
    { 0x2733a870, "cellSync", "cellSyncMutexInitialize" },
    { 0x91e6dcb9, "cellSync", "cellSyncMutexLock" },
    { 0xc463ee6c, "cellSync", "cellSyncMutexTryLock" },
    { 0xb9fbcc28, "cellSync", "cellSyncMutexUnlock" },
    { 0x24f80472, "cellSync", "cellSyncBarrierInitialize" },
    { 0xcb40434b, "cellSync", "cellSyncBarrierNotify" },
    { 0xbb54b37f, "cellSync", "cellSyncBarrierTryNotify" },
    { 0x3c81b4da, "cellSync", "cellSyncBarrierTryWait" },
    { 0xf8ef9193, "cellSync", "cellSyncLFQueueInitialize" },
    { 0x8c497d76, "cellSync", "cellSyncLFQueuePush" },
    { 0x0db15740, "cellSync", "cellSyncLFQueueTryPush" },
    { 0xe5c224e2, "cellSync", "cellSyncLFQueuePop" },
    { 0x65f2b02f, "cellSync", "cellSyncLFQueueTryPop" },
    { 0x8558da70, "cellSync", "cellSyncLFQueueGetSize" },

    // cellGifDec — GIF image decoder
    { 0x66315b6d, "cellGifDec", "cellGifDecCreate" },
    { 0xfe153bbf, "cellGifDec", "cellGifDecDestroy" },
    { 0xec45d754, "cellGifDec", "cellGifDecOpen" },
    { 0x11ae5215, "cellGifDec", "cellGifDecClose" },
    { 0xc52c449b, "cellGifDec", "cellGifDecReadHeader" },
    { 0x0c0c12bd, "cellGifDec", "cellGifDecDecodeData" },

    // sceNpTrophy — PlayStation Network trophy system
    { 0xdd74bdae, "sceNp", "sceNpTrophyInit" },
    { 0xd972691b, "sceNp", "sceNpTrophyTerm" },
    { 0xc10b8fce, "sceNp", "sceNpTrophyCreateHandle" },
    { 0x7775c461, "sceNp", "sceNpTrophyDestroyHandle" },
    { 0x7aa2fcff, "sceNp", "sceNpTrophyCreateContext" },
    { 0x0cde7100, "sceNp", "sceNpTrophyDestroyContext" },
    { 0x287f6018, "sceNp", "sceNpTrophyRegisterContext" },
    { 0x99d5fcdb, "sceNp", "sceNpTrophyGetRequiredDiskSpace" },
    { 0x3fbae39e, "sceNp", "sceNpTrophyGetGameInfo" },
    { 0x7a3ceb91, "sceNp", "sceNpTrophyUnlockTrophy" },
    { 0xdd29bb02, "sceNp", "sceNpTrophyGetTrophyInfo" },
    { 0x74b16e87, "sceNp", "sceNpTrophyGetTrophyUnlockState" },
    { 0xc868dea5, "sceNp", "sceNpTrophyGetGameProgress" },

    // sceNp — PlayStation Network base
    { 0xd59da152, "sceNp", "sceNpInit" },
    { 0xec72d838, "sceNp", "sceNpTerm" },
    { 0xd1fcb865, "sceNp", "sceNpGetNpId" },
    { 0x259ab84f, "sceNp", "sceNpGetOnlineId" },
    { 0x2aba00d1, "sceNp", "sceNpGetUserProfile" },

    // sceNpManager — PSN manager stubs
    { 0x9b98354e, "sceNp", "sceNpManagerGetStatus" },
    { 0x3566cd38, "sceNp", "sceNpManagerGetNetworkTime" },
    { 0x54bcafb8, "sceNp", "sceNpManagerGetOnlineId" },
    { 0xebae6c1a, "sceNp", "sceNpManagerGetNpId" },
    { 0x13079e64, "sceNp", "sceNpManagerRegisterCallback" },
    { 0x39853605, "sceNp", "sceNpManagerUnregisterCallback" },

    // sceNpCommerce2 — store stubs
    { 0xaaeae4a9, "sceNp", "sceNpCommerce2Init" },
    { 0x66da2edd, "sceNp", "sceNpCommerce2Term" },

    // cellRtc — real-time clock
    { 0x9148aab9, "cellRtc", "cellRtcGetCurrentTick" },
    { 0x27ffb3e3, "cellRtc", "cellRtcGetCurrentClockLocalTime" },
    { 0xd85c3c42, "cellRtc", "cellRtcGetCurrentClock" },
    { 0x18466b0e, "cellRtc", "cellRtcConvertLocalTimeToUtc" },
    { 0xead86714, "cellRtc", "cellRtcConvertUtcToLocalTime" },
    { 0x9c1aa28d, "cellRtc", "cellRtcGetTick" },
    { 0x7767eeb3, "cellRtc", "cellRtcSetTick" },
    { 0xf8cf8759, "cellRtc", "cellRtcTickAddSeconds" },
    { 0x17947909, "cellRtc", "cellRtcTickAddMinutes" },
    { 0x9ab17608, "cellRtc", "cellRtcFormatRfc2822" },
    { 0x49119def, "cellRtc", "cellRtcIsLeapYear" },
    { 0x2bb6158e, "cellRtc", "cellRtcGetDaysInMonth" },
    { 0xd68d676a, "cellRtc", "cellRtcGetDayOfWeek" },

    // cellScreenshot — screenshot capture stubs
    { 0x3ecc4646, "cellScreenshot", "cellScreenShotEnable" },
    { 0xa58d3f77, "cellScreenshot", "cellScreenShotDisable" },
    { 0xd1bb8d91, "cellScreenshot", "cellScreenShotSetParameter" },
    { 0xdf3c9711, "cellScreenshot", "cellScreenShotSetOverlayImage" },

    // cellMic — microphone stubs
    { 0x1bf34275, "cellMic", "cellMicInit" },
    { 0xb278948c, "cellMic", "cellMicEnd" },
    { 0xe57b4e17, "cellMic", "cellMicOpen" },
    { 0x287bf034, "cellMic", "cellMicClose" },
    { 0x448f9a7c, "cellMic", "cellMicGetDeviceAttr" },

    // cellSysCache — system cache mount/clear
    { 0x83a4afca, "cellSysCache", "cellSysCacheMount" },
    { 0x78f6204d, "cellSysCache", "cellSysCacheClear" },

    // cellUsbd — USB device stubs
    { 0x23658577, "cellUsbd", "cellUsbdInit" },
    { 0x3782d73f, "cellUsbd", "cellUsbdEnd" },

    // cellImeJp — IME Japanese input stubs
    { 0x4cf28923, "cellImeJp", "cellImeJpOpen" },
    { 0x2a9372c3, "cellImeJp", "cellImeJpClose" },

    // cellSysutil extras
    { 0xf89e3dbd, "cellSysutil", "cellSysutilGetSystemParamString" },
    { 0x38ecda87, "cellSysutil", "cellSysutilGetBgmPlaybackStatus2" },
    { 0x6ae48aa9, "cellSysutil", "cellSysutilEnableBgmPlaybackEx" },
    { 0x78410b3f, "cellSysutil", "cellSysutilDisableBgmPlaybackEx" },

    // cellDiscGame — disc info stubs
    { 0x312c9ac4, "cellDiscGame", "cellDiscGameGetBootDiscInfo" },
    { 0xfee36d78, "cellDiscGame", "cellDiscGameRegisterDiscChangeCallback" },
    { 0x04bb3c46, "cellDiscGame", "cellDiscGameUnregisterDiscChangeCallback" },

    // cellStorage — data import/export
    { 0xe462f5e5, "cellStorage", "cellStorageDataImportMove" },
    { 0x1d911b52, "cellStorage", "cellStorageDataExport" },

    // cellSubDisplay — sub-display output
    { 0xb8fea1c2, "cellSubDisplay", "cellSubDisplayInit" },
    { 0x64c14696, "cellSubDisplay", "cellSubDisplayEnd" },
    { 0x278e51bf, "cellSubDisplay", "cellSubDisplayGetRequiredMemory" },

    // cellSearch — media search
    { 0x45e3ec64, "cellSearch", "cellSearchInitialize" },
    { 0xb79cbc6e, "cellSearch", "cellSearchFinalize" },
    { 0xa660aaea, "cellSearch", "cellSearchStartListSearch" },
    { 0x5e331ebe, "cellSearch", "cellSearchGetListItems" },

    // cellMusic — music playback
    { 0x21929812, "cellMusic", "cellMusicInitialize" },
    { 0x74c0bb2f, "cellMusic", "cellMusicFinalize" },
    { 0xa7bf1b92, "cellMusic", "cellMusicGetPlaybackStatus" },

    // cellPhotoExport — photo registration
    { 0xd3b10cbd, "cellPhotoExport", "cellPhotoRegistFromFile" },
    { 0xb29f2aed, "cellPhotoExport", "cellPhotoExportInitialize" },
    { 0x8d422d8b, "cellPhotoExport", "cellPhotoExportFinalize" },

    // cellRemotePlay / cellBgdl / cellGameUpdate
    { 0xed1c0428, "cellRemotePlay", "cellRemotePlayGetStatus" },
    { 0xf09e1d35, "cellBgdl", "cellBgdlGetInfo" },
    { 0xae299ba5, "cellGameUpdate", "cellGameUpdateInit" },
    { 0x00e38e0b, "cellGameUpdate", "cellGameUpdateTerm" },
    { 0x4b88c75a, "cellGameUpdate", "cellGameUpdateCheckStartAsync" },

    // cellVpost — video post-processing
    { 0xce247a9c, "cellVpost", "cellVpostOpen" },
    { 0x292d1115, "cellVpost", "cellVpostClose" },
    { 0xf62c6004, "cellVpost", "cellVpostExec" },
    { 0xd28c4359, "cellVpost", "cellVpostOpenEx" },

    // cellAtrac — ATRAC3+ audio decode
    { 0xeeb4289f, "cellAtrac", "cellAtracCreateDecoder" },
    { 0x34e241d7, "cellAtrac", "cellAtracDeleteDecoder" },
    { 0xb771667c, "cellAtrac", "cellAtracDecode" },
    { 0x1c5004f5, "cellAtrac", "cellAtracGetStreamDataInfo" },
    { 0x05d6b33e, "cellAtrac", "cellAtracAddStreamData" },
    { 0x4343cb43, "cellAtrac", "cellAtracIsSecondBufferNeeded" },

    // cellVoice — voice chat stubs
    { 0x8cf0742d, "cellVoice", "cellVoiceInit" },
    { 0x176d528e, "cellVoice", "cellVoiceEnd" },
    { 0x25e2f250, "cellVoice", "cellVoiceCreatePort" },
    { 0x47f672fb, "cellVoice", "cellVoiceDeletePort" },

    // sceNpMatching2 — multiplayer matchmaking
    { 0x91a66b60, "sceNpMatching2", "sceNpMatching2Init" },
    { 0x8df1c55b, "sceNpMatching2", "sceNpMatching2Term" },
    { 0x7795cbfb, "sceNpMatching2", "sceNpMatching2CreateContext" },
    { 0x43e7ee1a, "sceNpMatching2", "sceNpMatching2DestroyContext" },

    // sceNpScore — leaderboard stubs
    { 0x368bac16, "sceNpScore", "sceNpScoreInit" },
    { 0xe36d39ae, "sceNpScore", "sceNpScoreTerm" },
    { 0x57efaab2, "sceNpScore", "sceNpScoreCreateTitleCtx" },
    { 0xec58df2b, "sceNpScore", "sceNpScoreDeleteTitleCtx" },

    // sceNpTus — title user storage
    { 0x3a24b2ef, "sceNpTus", "sceNpTusInit" },
    { 0x63496d84, "sceNpTus", "sceNpTusTerm" },
    { 0x6a49f21a, "sceNpTus", "sceNpTusCreateTitleCtx" },
    { 0xdb17369f, "sceNpTus", "sceNpTusDeleteTitleCtx" },

    // cellSaveData — additional variants
    { 0x590e6d0b, "cellSaveData", "cellSaveDataAutoLoad2" },
    { 0xf0f530b7, "cellSaveData", "cellSaveDataAutoSave2" },
    { 0x22126da4, "cellSaveData", "cellSaveDataUserListLoad" },
    { 0x06a7a4a8, "cellSaveData", "cellSaveDataUserListSave" },
    { 0xf14197af, "cellSaveData", "cellSaveDataUserFixedLoad" },
    { 0xa4726925, "cellSaveData", "cellSaveDataUserFixedSave" },

    // cellGame — extras
    { 0x8e422adc, "cellGame", "cellGameSetExitParam" },
    { 0xbaed3165, "cellGame", "cellGameGetLocalWebContentPath" },
    { 0x586bdc5c, "cellGame", "cellGameThemeInstall" },
    { 0xc6bee834, "cellGame", "cellGameThemeInstallFromBuffer" },

    // cellWebBrowser
    { 0xf19fd906, "cellWebBrowser", "cellWebBrowserInitialize" },
    { 0x2b5f3544, "cellWebBrowser", "cellWebBrowserShutdown" },

    // cellPad — extras
    { 0x23ef9b61, "cellPad", "cellPadSetActDirect" },
    { 0x316cdf06, "cellPad", "cellPadGetCapabilityInfo" },
    { 0x1be2bbee, "cellPad", "cellPadGetInfo2" },
    { 0x0f2c5daf, "cellPad", "cellPadSetPortSetting" },
    { 0xbd8f1ead, "cellPad", "cellPadSetSensorMode" },
    { 0x0353d3cf, "cellPad", "cellPadLddRegisterController" },
    { 0x9589f71c, "cellPad", "cellPadLddGetPortNo" },

    // ── cellPrint ────────────────────────────────────────────────
    { 0xB2D72BB9, "cellPrint", "cellPrintOpenConfig" },
    { 0xB75EB4F8, "cellPrint", "cellPrintGetStatus" },
    { 0xFC06712D, "cellPrint", "cellPrintStartJob" },
    { 0xD94E89BC, "cellPrint", "cellPrintEndJob" },

    // ── cellMusicDecode ──────────────────────────────────────────
    { 0x98FDAFD5, "cellMusicDecode", "cellMusicDecodeInitialize" },
    { 0x7659B1E7, "cellMusicDecode", "cellMusicDecodeInitialize2" },
    { 0xDDA1BC5E, "cellMusicDecode", "cellMusicDecodeFinalize" },
    { 0x3BE565BF, "cellMusicDecode", "cellMusicDecodeFinalize2" },
    { 0x9609CA68, "cellMusicDecode", "cellMusicDecodeSelectContents" },
    { 0x18CFE79D, "cellMusicDecode", "cellMusicDecodeSetDecodeCommand" },
    { 0xC8E261CE, "cellMusicDecode", "cellMusicDecodeGetDecodeStatus" },
    { 0x22953249, "cellMusicDecode", "cellMusicDecodeRead" },

    // ── sceNpFriends ─────────────────────────────────────────────
    { 0xA72CA286, "sceNpFriends", "sceNpFriendsInit" },
    { 0xBDEF1053, "sceNpFriends", "sceNpFriendsTerm" },
    { 0x912E5879, "sceNpFriends", "sceNpFriendsGetFriendListEntryCount" },
    { 0xFDD24474, "sceNpFriends", "sceNpFriendsGetFriendListEntry" },
    { 0xA98A49F1, "sceNpFriends", "sceNpFriendsGetFriendPresence" },
    { 0x8AE071CF, "sceNpFriends", "sceNpFriendsGetFriendInfo" },

    // ── cellSail (media player) ──────────────────────────────────
    { 0x948DAF24, "cellSail", "cellSailPlayerInitialize" },
    { 0x1C674A46, "cellSail", "cellSailPlayerFinalize" },
    { 0x928AF120, "cellSail", "cellSailPlayerSetParameter" },
    { 0xD43F8537, "cellSail", "cellSailPlayerGetParameter" },
    { 0x9D9DA3F4, "cellSail", "cellSailPlayerBoot" },
    { 0x254936DB, "cellSail", "cellSailPlayerCreateDescriptor" },
    { 0x2B5F20BE, "cellSail", "cellSailPlayerDestroyDescriptor" },
    { 0x3A6E01B1, "cellSail", "cellSailPlayerOpenStream" },

    // ── cellRudp (reliable UDP) ──────────────────────────────────
    { 0xE26B7209, "cellRudp", "cellRudpInit" },
    { 0xB92636EA, "cellRudp", "cellRudpEnd" },
    { 0xCA071AAC, "cellRudp", "cellRudpCreateContext" },
    { 0x500A408E, "cellRudp", "cellRudpBind" },
    { 0x23293E9B, "cellRudp", "cellRudpSend" },
    { 0x61DB1F47, "cellRudp", "cellRudpReceive" },

    // ── cellHttpUtil ─────────────────────────────────────────────
    { 0x1C7430EE, "cellHttpUtil", "cellHttpUtilParseUri" },
    { 0x77A3D37C, "cellHttpUtil", "cellHttpUtilBuildUri" },
    { 0xD95B0C60, "cellHttpUtil", "cellHttpUtilEscapeUri" },
    { 0xA4C9B7DC, "cellHttpUtil", "cellHttpUtilUnescapeUri" },

    // ── cellSsl ──────────────────────────────────────────────────
    { 0x8C1678EF, "cellSsl", "cellSslInit" },
    { 0x2A574A25, "cellSsl", "cellSslEnd" },
    { 0xCA220E5B, "cellSsl", "cellSslCertificateLoader" },
    { 0x10C1BFFC, "cellSsl", "cellSslCertGetSerialNumber" },
    { 0xAB209809, "cellSsl", "cellSslCertGetPublicKey" },

    // ── cellHttp ─────────────────────────────────────────────────
    { 0xF7644439, "cellHttp", "cellHttpInit" },
    { 0x33457271, "cellHttp", "cellHttpEnd" },
    { 0x598DD02E, "cellHttp", "cellHttpCreateClient" },
    { 0x6907A5FB, "cellHttp", "cellHttpDestroyClient" },
    { 0x21149D0A, "cellHttp", "cellHttpCreateTransaction" },
    { 0xC2D6836B, "cellHttp", "cellHttpDestroyTransaction" },
    { 0x6180BA1A, "cellHttp", "cellHttpSendRequest" },
    { 0xFB40BF8E, "cellHttp", "cellHttpRecvResponse" },

    // ── cellNetCtl ───────────────────────────────────────────────
    { 0xF53F04BB, "cellNetCtl", "cellNetCtlInit" },
    { 0x9F18CCAD, "cellNetCtl", "cellNetCtlTerm" },
    { 0x5C413CA9, "cellNetCtl", "cellNetCtlGetState" },
    { 0x08A8A347, "cellNetCtl", "cellNetCtlGetInfo" },
    { 0x2E274B74, "cellNetCtl", "cellNetCtlAddHandler" },
    { 0x438CEE7B, "cellNetCtl", "cellNetCtlDelHandler" },

    // ── cellFont ─────────────────────────────────────────────────
    { 0x4B734C8C, "cellFont", "cellFontInit" },
    { 0x3E3712ED, "cellFont", "cellFontEnd" },
    { 0xC1B20A00, "cellFont", "cellFontNewLibrary" },
    { 0x859A29D7, "cellFont", "cellFontFreeLibrary" },
    { 0x85CAD796, "cellFont", "cellFontOpenFontset" },
    { 0x62F4F193, "cellFont", "cellFontOpenFontFile" },
    { 0xEC2185A8, "cellFont", "cellFontCloseFont" },
    { 0x4B02499C, "cellFont", "cellFontRenderCharGlyphImage" },

    // ── cellFontFT ───────────────────────────────────────────────
    { 0xE67CA1DB, "cellFontFT", "cellFontFTInit" },
    { 0xECD30388, "cellFontFT", "cellFontFTEnd" },
    { 0xA3F2DCEB, "cellFontFT", "cellFontFTGetInitializedRevisionFlags" },
    { 0x6741115E, "cellFontFT", "cellFontFTLoadModule" },

    // ── cellSpurs ────────────────────────────────────────────────
    { 0x70E3D58A, "cellSpurs", "cellSpursInitialize" },
    { 0xE48BF572, "cellSpurs", "cellSpursFinalize" },
    { 0x1BF8A000, "cellSpurs", "cellSpursGetNumSpuThread" },
    { 0xE2DF437E, "cellSpurs", "cellSpursGetSpuThreadGroupId" },
    { 0x3942F6BF, "cellSpurs", "cellSpursSetMaxContention" },
    { 0x79B36C3E, "cellSpurs", "cellSpursSetPriorities" },
    { 0xBC7001AF, "cellSpurs", "cellSpursAttachLv2EventQueue" },
    { 0xC97CEC02, "cellSpurs", "cellSpursCreateTask" },
    { 0x3FD6C11A, "cellSpurs", "cellSpursCreateTaskset" },
    { 0xE591F650, "cellSpurs", "cellSpursJoinTaskset" },
    { 0x899934DC, "cellSpurs", "cellSpursShutdownTaskset" },
    { 0x71DC1454, "cellSpurs", "cellSpursWakeupTaskset" },

    // ── cellSpursJq ──────────────────────────────────────────────
    { 0x94965961, "cellSpursJq", "cellSpursJqInitialize" },
    { 0x5AFC1E7C, "cellSpursJq", "cellSpursJqFinalize" },
    { 0xC8101522, "cellSpursJq", "cellSpursJqAddJob" },
    { 0xAB27B2C4, "cellSpursJq", "cellSpursJqGetJobCount" },

    // ── sys_net (BSD sockets) ────────────────────────────────────
    { 0xAE26EAB1, "sys_net", "sys_net_bnet_socket" },
    { 0x5A048BCB, "sys_net", "sys_net_bnet_close" },
    { 0xFC5F38C2, "sys_net", "sys_net_bnet_bind" },
    { 0xDCA3F6F6, "sys_net", "sys_net_bnet_listen" },
    { 0x3B214BA1, "sys_net", "sys_net_bnet_accept" },
    { 0x4E89A57A, "sys_net", "sys_net_bnet_connect" },
    { 0x8CC694CE, "sys_net", "sys_net_bnet_sendto" },
    { 0x75B3C344, "sys_net", "sys_net_bnet_recvfrom" },
    { 0x9A5C66E3, "sys_net", "sys_net_bnet_setsockopt" },
    { 0xF448BBF6, "sys_net", "sys_net_bnet_getsockopt" },

    // ── cellUserInfo ─────────────────────────────────────────────
    { 0xDCF8EB53, "cellUserInfo", "cellUserInfoGetStat" },
    { 0xADCCC2B9, "cellUserInfo", "cellUserInfoSelectUser_ListType" },
    { 0x389EB4EC, "cellUserInfo", "cellUserInfoEnableOverlay" },
    { 0x4A5B749A, "cellUserInfo", "cellUserInfoGetList" },

    // ── cellAdec (audio decoder) ─────────────────────────────────
    { 0x8F027E01, "cellAdec", "cellAdecOpen" },
    { 0xCD96044F, "cellAdec", "cellAdecClose" },
    { 0xCC534ABE, "cellAdec", "cellAdecStartSeq" },
    { 0xCCFEEFC2, "cellAdec", "cellAdecEndSeq" },
    { 0x4D682B55, "cellAdec", "cellAdecDecodeAu" },
    { 0x0CFA3BAF, "cellAdec", "cellAdecGetPcm" },

    // ── cellDmux (demuxer) ───────────────────────────────────────
    { 0x2D4CE3DF, "cellDmux", "cellDmuxOpen" },
    { 0x438A5A0F, "cellDmux", "cellDmuxClose" },
    { 0xCD076ABF, "cellDmux", "cellDmuxSetStream" },
    { 0xABBBB443, "cellDmux", "cellDmuxResetStream" },
    { 0x181D1978, "cellDmux", "cellDmuxEnableEs" },
    { 0x022DDA52, "cellDmux", "cellDmuxDisableEs" },

    // ── cellAvconfExt — audio/video config extras ────────────────
    { 0x9df98130, "cellAvconfExt", "cellVideoOutGetDeviceInfo" },
    { 0x1e930eef, "cellAvconfExt", "cellVideoOutGetNumberOfDevice" },
    { 0x75bbb672, "cellAvconfExt", "cellVideoOutGetResolutionAvailability" },
    { 0x887572d5, "cellAvconfExt", "cellVideoOutGetState" },
    { 0x0bae8772, "cellAvconfExt", "cellVideoOutConfigure" },
    { 0x15b0b0cd, "cellAvconfExt", "cellAudioOutGetState" },
    { 0x4692ab35, "cellAvconfExt", "cellAudioOutConfigure" },
    { 0xa0e6fdf0, "cellAvconfExt", "cellAudioOutGetSoundAvailability" },
    { 0xa5927fc5, "cellAvconfExt", "cellAudioOutGetDeviceInfo" },
    { 0x7794d7e7, "cellAvconfExt", "cellAudioOutGetNumberOfDevice" },

    // ── cellSync2 — enhanced sync primitives ─────────────────────
    { 0x55836e73, "cellSync2", "cellSync2MutexNew" },
    { 0x5551b4df, "cellSync2", "cellSync2MutexLock" },
    { 0x5551b540, "cellSync2", "cellSync2MutexUnlock" },
    { 0x5551b56c, "cellSync2", "cellSync2MutexTryLock" },
    { 0xa661b35c, "cellSync2", "cellSync2QueueNew" },
    { 0xa661b4d0, "cellSync2", "cellSync2QueuePush" },
    { 0xa661b4e8, "cellSync2", "cellSync2QueueTryPush" },
    { 0xa661b500, "cellSync2", "cellSync2QueuePop" },
    { 0xa661b518, "cellSync2", "cellSync2QueueTryPop" },
    { 0xa661b530, "cellSync2", "cellSync2QueueGetSize" },

    // ── cellVideoExport — video export stubs ─────────────────────
    { 0xe7998490, "cellVideoExport", "cellVideoExportInitialize" },
    { 0x12998e3a, "cellVideoExport", "cellVideoExportFinalize" },
    { 0x3cf0b78e, "cellVideoExport", "cellVideoExportProgress" },

    // ── cellPhotoImport — photo import stubs ─────────────────────
    { 0x0783bce0, "cellPhotoImport", "cellPhotoImportInitialize" },
    { 0x1c231710, "cellPhotoImport", "cellPhotoImportFinalize" },
    { 0x59405c00, "cellPhotoImport", "cellPhotoImportSelectImage" },

    // ── cellNetCtlExt — extended network ─────────────────────────
    { 0xca8cd5b7, "cellNetCtlExt", "cellNetCtlGetNatInfo" },
    { 0x3b23dbd0, "cellNetCtlExt", "cellNetCtlGetIfAddr" },
    { 0x2a72ed91, "cellNetCtlExt", "cellNetCtlNetStartDialogLoadAsync" },
    { 0x7e4a2c6e, "cellNetCtlExt", "cellNetCtlNetStartDialogAbortAsync" },
    { 0x6f000e53, "cellNetCtlExt", "cellNetCtlNetStartDialogUnloadAsync" },

    // ── sys_io — controller/keyboard/mouse I/O ───────────────────
    { 0x3733ea3c, "sys_io", "sys_io_pad_get_capability" },
    { 0x1cf98800, "sys_io", "sys_io_pad_set_press_mode" },
    { 0x578e3c98, "sys_io", "sys_io_pad_clear_buf" },
    { 0xa703a917, "sys_io", "sys_io_pad_get_data" },
    { 0x6bc09c61, "sys_io", "sys_io_pad_get_data_extra" },

    // ── cellGem (PlayStation Move) ───────────────────────────────
    { 0xabb4b268, "cellGem", "cellGemInit" },
    { 0xa8bc1648, "cellGem", "cellGemEnd" },
    { 0x3e24e759, "cellGem", "cellGemGetState" },
    { 0x13ea53e7, "cellGem", "cellGemGetInfo" },
    { 0x6a666297, "cellGem", "cellGemGetTrackerHue" },
    { 0xfb5887f9, "cellGem", "cellGemCalibrate" },
    { 0xe1f85a80, "cellGem", "cellGemEnableCameraPitchAngleCorrection" },
    { 0xc7622586, "cellGem", "cellGemReset" },
    { 0x6d245f02, "cellGem", "cellGemUpdateStart" },
    { 0x3507f03b, "cellGem", "cellGemUpdateFinish" },

    // ── cellMove (PS Move controller) ────────────────────────────
    { 0x9eb07a5b, "cellMove", "cellMoveInit" },
    { 0xadee0a65, "cellMove", "cellMoveEnd" },
    { 0x82cfb3d1, "cellMove", "cellMoveStart" },
    { 0xd37b8e36, "cellMove", "cellMoveStop" },

    // ── cellOvis (visual development) ────────────────────────────
    { 0xcc78cd7b, "cellOvis", "cellOvisInitialize" },
    { 0x2e70a5a1, "cellOvis", "cellOvisFinalize" },
    { 0x71894bfa, "cellOvis", "cellOvisGetStatus" },

    // ── cellCamera ───────────────────────────────────────────────
    { 0x36e1e930, "cellCamera", "cellCameraInit" },
    { 0x20f3f498, "cellCamera", "cellCameraEnd" },
    { 0x5de25cd1, "cellCamera", "cellCameraOpen" },
    { 0x379c5dd6, "cellCamera", "cellCameraClose" },
    { 0x40f6ead6, "cellCamera", "cellCameraStart" },
    { 0xa6b20b8c, "cellCamera", "cellCameraStop" },
    { 0x60237200, "cellCamera", "cellCameraGetDeviceGUID" },
    { 0x10697d02, "cellCamera", "cellCameraGetType" },
    { 0xb0647e5a, "cellCamera", "cellCameraIsAvailable" },

    // ── cellResc (Resolution Scaler) ─────────────────────────────
    { 0x7f3c66b0, "cellResc", "cellRescInit" },
    { 0x25c107e6, "cellResc", "cellRescExit" },
    { 0x10db5b1a, "cellResc", "cellRescSetDisplayMode" },
    { 0x0d3c22ce, "cellResc", "cellRescSetConvertAndFlip" },
    { 0x1d7deee6, "cellResc", "cellRescSetBufferAddress" },
    { 0x01220224, "cellResc", "cellRescSetSrc" },
    { 0x5a338e69, "cellResc", "cellRescSetDsts" },
    { 0x516ee89e, "cellResc", "cellRescGetNumColorBuffers" },
    { 0x2ea3061e, "cellResc", "cellRescGetFlipStatus" },
    { 0xaa8b2baa, "cellResc", "cellRescResetFlipStatus" },

    // ── cellPamf (PS3 media format) ──────────────────────────────
    { 0x90fc9a36, "cellPamf", "cellPamfReaderInitialize" },
    { 0xd1a40ef4, "cellPamf", "cellPamfReaderGetNumberOfStreams" },
    { 0x041a5c89, "cellPamf", "cellPamfReaderGetStreamTypeCoding" },
    { 0x28b4e2c1, "cellPamf", "cellPamfReaderSetStreamWithTypeAndIndex" },
    { 0x45d62a3b, "cellPamf", "cellPamfReaderGetStreamIndex" },
    { 0x461534a4, "cellPamf", "cellPamfReaderGetNumberOfSpecificStreams" },
    { 0xd6a50759, "cellPamf", "cellPamfStreamGetInfo" },

    // ── sceNpUtil — NP utilities ─────────────────────────────────
    { 0x05af7b56, "sceNpUtil", "sceNpUtilBandwidthTestInitStart" },
    { 0xaa9a4c83, "sceNpUtil", "sceNpUtilBandwidthTestGetStatus" },
    { 0xfade2b8d, "sceNpUtil", "sceNpUtilBandwidthTestShutdown" },

    // ── sceNpSignaling — NP signaling ────────────────────────────
    { 0xa3c4ddeb, "sceNpSignaling", "sceNpSignalingCreateCtx" },
    { 0x7db3e905, "sceNpSignaling", "sceNpSignalingDestroyCtx" },
    { 0x55e42a79, "sceNpSignaling", "sceNpSignalingActivateConnection" },
    { 0x7d5a4a87, "sceNpSignaling", "sceNpSignalingDeactivateConnection" },
    { 0xa10fedd3, "sceNpSignaling", "sceNpSignalingGetLocalNetInfo" },

    // ── sceNpClans — NP clans ────────────────────────────────────
    { 0xaa79031d, "sceNpClans", "sceNpClansInit" },
    { 0x1f51ae44, "sceNpClans", "sceNpClansTerm" },
    { 0x6e24f290, "sceNpClans", "sceNpClansCreateRequest" },
    { 0x3c67c847, "sceNpClans", "sceNpClansDestroyRequest" },

    // ── cellSpursTrace — SPURS tracing ───────────────────────────
    { 0x72ec1bf4, "cellSpursTrace", "cellSpursTraceInitialize" },
    { 0x2db41dea, "cellSpursTrace", "cellSpursTraceFinalize" },
    { 0x9cae4fdc, "cellSpursTrace", "cellSpursTraceStart" },
    { 0x04a6bd22, "cellSpursTrace", "cellSpursTraceStop" },

    // ── cellCrossController — cross-controller ───────────────────
    { 0x174ece14, "cellCrossController", "cellCrossControllerInitialize" },
    { 0x7adf3bab, "cellCrossController", "cellCrossControllerFinalize" },

    // ── cellSysconf — system config ──────────────────────────────
    { 0x00753e2a, "cellSysconf", "cellSysconfAbort" },
    { 0x0beecf67, "cellSysconf", "cellSysconfBtGetDeviceList" },
    { 0xac410de9, "cellSysconf", "cellSysconfGetIntegerVariable" },

    // ── cellMusicExport — music export ───────────────────────────
    { 0xe8ad3dd4, "cellMusicExport", "cellMusicExportInitialize" },
    { 0x4ab73a73, "cellMusicExport", "cellMusicExportFinalize" },
    { 0x61ead640, "cellMusicExport", "cellMusicExportProgress" },

    // ── cellPhotoUtility — photo utility ─────────────────────────
    { 0x5caa19e7, "cellPhotoUtility", "cellPhotoInitialize" },
    { 0xd46fa1f7, "cellPhotoUtility", "cellPhotoFinalize" },
    { 0x2ff6e155, "cellPhotoUtility", "cellPhotoRegistFromFile" },
};

static inline const PpuHleEntry* ppu_hle_lookup(uint32_t fnid) {
    constexpr size_t N = sizeof(PPU_HLE_NAMES) / sizeof(PpuHleEntry);
    for (size_t i = 0; i < N; ++i) {
        if (PPU_HLE_NAMES[i].fnid == fnid) return &PPU_HLE_NAMES[i];
    }
    return nullptr;
}
