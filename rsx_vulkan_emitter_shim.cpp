// rsx_vulkan_emitter_shim.cpp — C ABI bridge between rsx_command_processor.cu
// (weak-symbol dispatch via rsx_emitter_*) and the C++ VulkanEmitter class.

#include "rsx_vulkan_emitter.h"
#include "rsx_defs.h"

using rsx::VulkanEmitter;
using rsx::RSXState;

extern "C" {

void rsx_emitter_onSurfaceSetup(void* emitter, const RSXState* s) {
    static_cast<VulkanEmitter*>(emitter)->onSurfaceSetup(*s);
}
void rsx_emitter_onViewport(void* emitter, const RSXState* s) {
    static_cast<VulkanEmitter*>(emitter)->onViewport(*s);
}
void rsx_emitter_onScissor(void* emitter, const RSXState* s) {
    static_cast<VulkanEmitter*>(emitter)->onScissor(*s);
}
void rsx_emitter_onClearSurface(void* emitter, const RSXState* s, uint32_t mask) {
    static_cast<VulkanEmitter*>(emitter)->onClearSurface(*s, mask);
}
void rsx_emitter_onBeginEnd(void* emitter, const RSXState* s, uint32_t prim) {
    static_cast<VulkanEmitter*>(emitter)->onBeginEnd(*s, prim);
}
void rsx_emitter_onDrawArrays(void* emitter, const RSXState* s,
                              uint32_t first, uint32_t count) {
    static_cast<VulkanEmitter*>(emitter)->onDrawArrays(*s, first, count);
}
void rsx_emitter_onDrawIndexed(void* emitter, const RSXState* s,
                               uint32_t first, uint32_t count, uint32_t fmt) {
    static_cast<VulkanEmitter*>(emitter)->onDrawIndexed(*s, first, count, fmt);
}
void rsx_emitter_onFlip(void* emitter, const RSXState* s, uint32_t surfaceOffset) {
    static_cast<VulkanEmitter*>(emitter)->onFlip(*s, surfaceOffset);
}

} // extern "C"
