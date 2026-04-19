// ppc_interpreter.cu — CUDA PowerPC Interpreter Kernel
// Modeled after VK_RT/layer/rt_ir_exec.cu: per-thread state + opcode dispatch loop.
//
// One CUDA thread = one PPE hardware thread.
// The kernel runs as a persistent megakernel — launched once, interprets until halt.
//
#include "ppc_defs.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

using namespace ppc;

// ═══════════════════════════════════════════════════════════════
// Big-Endian Memory Access via __byte_perm
// PS3 is big-endian; GPU is little-endian. Every memory op swaps.
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

__device__ __forceinline__ uint16_t bswap16(uint16_t x) {
    return (uint16_t)__byte_perm((uint32_t)x, 0, 0x0001);
}

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    return ((uint64_t)bswap32(lo) << 32) | (uint64_t)bswap32(hi);
}

// Memory read/write with endian swap
// `mem` is the base of the 512MB PS3 sandbox in GPU VRAM
__device__ __forceinline__ uint32_t mem_read32(const uint8_t* mem, uint64_t addr) {
    uint32_t raw;
    memcpy(&raw, mem + (addr & (PS3_SANDBOX_SIZE - 1)), 4);
    return bswap32(raw);
}

__device__ __forceinline__ uint16_t mem_read16(const uint8_t* mem, uint64_t addr) {
    uint16_t raw;
    memcpy(&raw, mem + (addr & (PS3_SANDBOX_SIZE - 1)), 2);
    return bswap16(raw);
}

__device__ __forceinline__ uint8_t mem_read8(const uint8_t* mem, uint64_t addr) {
    return mem[addr & (PS3_SANDBOX_SIZE - 1)];
}

__device__ __forceinline__ void mem_write32(uint8_t* mem, uint64_t addr, uint32_t val) {
    uint32_t swapped = bswap32(val);
    memcpy(mem + (addr & (PS3_SANDBOX_SIZE - 1)), &swapped, 4);
}

__device__ __forceinline__ void mem_write16(uint8_t* mem, uint64_t addr, uint16_t val) {
    uint16_t swapped = bswap16(val);
    memcpy(mem + (addr & (PS3_SANDBOX_SIZE - 1)), &swapped, 2);
}

__device__ __forceinline__ void mem_write8(uint8_t* mem, uint64_t addr, uint8_t val) {
    mem[addr & (PS3_SANDBOX_SIZE - 1)] = val;
}

// Floating-point memory ops (IEEE 754 is endian-dependent)
__device__ __forceinline__ float mem_readf32(const uint8_t* mem, uint64_t addr) {
    uint32_t bits = mem_read32(mem, addr);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

__device__ __forceinline__ double mem_readf64(const uint8_t* mem, uint64_t addr) {
    uint64_t bits = bswap64(*(const uint64_t*)(mem + (addr & (PS3_SANDBOX_SIZE - 1))));
    double d;
    memcpy(&d, &bits, 8);
    return d;
}

__device__ __forceinline__ void mem_writef32(uint8_t* mem, uint64_t addr, float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    mem_write32(mem, addr, bits);
}

__device__ __forceinline__ void mem_writef64(uint8_t* mem, uint64_t addr, double d) {
    uint64_t bits;
    memcpy(&bits, &d, 8);
    uint64_t swapped = bswap64(bits);
    memcpy(mem + (addr & (PS3_SANDBOX_SIZE - 1)), &swapped, 8);
}

// 64-bit memory read/write (big-endian)
__device__ __forceinline__ uint64_t mem_read64(const uint8_t* mem, uint64_t addr) {
    uint64_t raw;
    memcpy(&raw, mem + (addr & (PS3_SANDBOX_SIZE - 1)), 8);
    return bswap64(raw);
}

__device__ __forceinline__ void mem_write64(uint8_t* mem, uint64_t addr, uint64_t val) {
    uint64_t be = bswap64(val);
    memcpy(mem + (addr & (PS3_SANDBOX_SIZE - 1)), &be, 8);
}

// ═══════════════════════════════════════════════════════════════
// HLE Syscall Handler
// ═══════════════════════════════════════════════════════════════

// HLE object ID counter (simple incrementing IDs for kernel objects)
static __device__ uint32_t g_hle_next_id = 0x100;
static __device__ uint64_t g_heapPtr = 0x10000000ULL; // bump allocator start at 256MB

__device__ static uint32_t hle_alloc_id() {
    return atomicAdd(&g_hle_next_id, 1);
}

__device__ static void handleSyscall(PPEState& s, uint8_t* mem, uint32_t* hle_log,
                                      volatile uint32_t* hle_signal) {
    uint32_t sc_num = (uint32_t)s.gpr[11]; // r11 = syscall number on CellOS

    switch (sc_num) {

    // ─── Process management ──────────────────────────────────────
    case SYS_PROCESS_EXIT:
    case SYS_PROCESS_EXIT2:
    case SYS_PROCESS_EXIT3:
        s.halted = 1;
        break;

    case SYS_PROCESS_GETPID:
        s.gpr[3] = 0x01000500ULL; // fake PID
        break;

    case SYS_PROCESS_GET_SDK_VERSION:
        s.gpr[3] = 0;             // CELL_OK
        s.gpr[4] = 0x00470000ULL; // SDK 4.70
        break;

    case SYS_PROCESS_GET_PARAMSFO:
        s.gpr[3] = 0; // CELL_OK (no SFO data)
        break;

    // ─── PPU Thread management ───────────────────────────────────
    case SYS_PPU_THREAD_CREATE: {
        // r3 = thread_id_ptr, r4 = entry, r5 = arg, r6 = prio, r7 = stacksize, r8 = flags
        uint32_t tid = hle_alloc_id();
        uint64_t tid_ptr = s.gpr[3];
        if (tid_ptr && tid_ptr < PS3_SANDBOX_SIZE - 8)
            mem_write64(mem, tid_ptr, tid);
        s.gpr[3] = 0; // CELL_OK
        if (hle_signal) atomicAdd((uint32_t*)hle_signal, 1);
        break;
    }
    case SYS_PPU_THREAD_START:
    case SYS_PPU_THREAD_DETACH:
    case SYS_PPU_THREAD_RENAME:
    case SYS_PPU_THREAD_SET_PRIORITY:
        s.gpr[3] = 0; // CELL_OK
        break;

    case SYS_PPU_THREAD_EXIT:
        s.halted = 1;
        break;

    case SYS_PPU_THREAD_YIELD:
        s.gpr[3] = 0;
        break;

    case SYS_PPU_THREAD_JOIN:
        s.gpr[3] = 0; // CELL_OK (single-thread: join returns immediately)
        break;

    case SYS_PPU_THREAD_GET_JOIN_STATE:
        s.gpr[3] = 0;
        s.gpr[4] = 0; // not joinable
        break;

    case SYS_PPU_THREAD_GET_PRIORITY:
        s.gpr[3] = 0;
        s.gpr[4] = 1000; // default priority
        break;

    case SYS_PPU_THREAD_GET_STACK_INFO:
        // r3 = sp_info_ptr; write {addr, size}
        if (s.gpr[3] && s.gpr[3] < PS3_SANDBOX_SIZE - 16) {
            mem_write64(mem, s.gpr[3], 0x0D000000ULL);     // stack bottom
            mem_write64(mem, s.gpr[3] + 8, 0x00100000ULL); // 1MB stack
        }
        s.gpr[3] = 0;
        break;

    // ─── Time ────────────────────────────────────────────────────
    case SYS_TICKS_GET:
        s.gpr[3] = s.cycles * 10; // ~79.8 MHz approximation
        break;

    case SYS_TIME_GET_CURRENT_TIME: {
        // r3 = sec_ptr, r4 = nsec_ptr
        uint64_t fake_sec = 1700000000ULL + s.cycles / 79800000ULL;
        if (s.gpr[3] && s.gpr[3] < PS3_SANDBOX_SIZE - 8)
            mem_write64(mem, s.gpr[3], fake_sec);
        if (s.gpr[4] && s.gpr[4] < PS3_SANDBOX_SIZE - 8)
            mem_write64(mem, s.gpr[4], 0);
        s.gpr[3] = 0;
        break;
    }

    case SYS_TIME_GET_TIMEBASE_FREQUENCY:
        s.gpr[3] = 0;
        s.gpr[4] = 79800000ULL; // 79.8 MHz
        break;

    // ─── TTY / debug output ──────────────────────────────────────
    case SYS_TTY_READ:
        // (fd, buf, len, nread_ptr) — signal 0 bytes read
        if (s.gpr[6] && s.gpr[6] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[6], 0);
        s.gpr[3] = 0;
        break;
    case SYS_TTY_WRITE:
        // (fd, buf, len, nwritten_ptr) — swallow output, pretend all written
        if (s.gpr[6] && s.gpr[6] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[6], (uint32_t)s.gpr[5]);
        s.gpr[3] = 0;
        break;

    // ─── Timer ───────────────────────────────────────────────────
    case SYS_TIMER_CREATE:
        s.gpr[3] = 0;
        s.gpr[4] = hle_alloc_id();
        break;
    case SYS_TIMER_DESTROY:
    case SYS_TIMER_GET_INFO:
    case SYS_TIMER_START:
    case SYS_TIMER_STOP:
    case SYS_TIMER_CONNECT_EVENT_QUEUE:
    case SYS_TIMER_DISCONNECT_EVENT_QUEUE:
        s.gpr[3] = 0;
        break;

    // ─── Mutex ───────────────────────────────────────────────────
    case SYS_MUTEX_CREATE: {
        uint32_t id = hle_alloc_id();
        if (s.gpr[3] && s.gpr[3] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[3], id);
        s.gpr[3] = 0;
        break;
    }
    case SYS_MUTEX_DESTROY:
    case SYS_MUTEX_LOCK:
    case SYS_MUTEX_TRYLOCK:
    case SYS_MUTEX_UNLOCK:
        s.gpr[3] = 0;
        break;

    // ─── Lightweight mutex ───────────────────────────────────────
    case SYS_LWMUTEX_CREATE: {
        uint32_t id = hle_alloc_id();
        if (s.gpr[3] && s.gpr[3] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[3], id);
        s.gpr[3] = 0;
        break;
    }
    case SYS_LWMUTEX_DESTROY:
    case SYS_LWMUTEX_LOCK:
    case SYS_LWMUTEX_UNLOCK:
    case SYS_LWMUTEX_TRYLOCK:
        s.gpr[3] = 0;
        break;

    // ─── Condition variable ──────────────────────────────────────
    case SYS_COND_CREATE: {
        uint32_t id = hle_alloc_id();
        if (s.gpr[3] && s.gpr[3] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[3], id);
        s.gpr[3] = 0;
        break;
    }
    case SYS_COND_DESTROY:
    case SYS_COND_WAIT:
    case SYS_COND_SIGNAL:
    case SYS_COND_SIGNAL_ALL:
    case SYS_COND_SIGNAL_TO:
        s.gpr[3] = 0;
        break;

    // ─── Lightweight condition variable ──────────────────────────
    case SYS_LWCOND_CREATE: {
        uint32_t id = hle_alloc_id();
        if (s.gpr[3] && s.gpr[3] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[3], id);
        s.gpr[3] = 0;
        break;
    }
    case SYS_LWCOND_DESTROY:
    case SYS_LWCOND_QUEUE_WAIT:
    case SYS_LWCOND_SIGNAL:
    case SYS_LWCOND_SIGNAL_ALL:
        s.gpr[3] = 0;
        break;

    // ─── Semaphore ───────────────────────────────────────────────
    case SYS_SEMAPHORE_CREATE: {
        uint32_t id = hle_alloc_id();
        if (s.gpr[3] && s.gpr[3] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[3], id);
        s.gpr[3] = 0;
        break;
    }
    case SYS_SEMAPHORE_DESTROY:
    case SYS_SEMAPHORE_WAIT:
    case SYS_SEMAPHORE_TRYWAIT:
    case SYS_SEMAPHORE_POST:
        s.gpr[3] = 0;
        break;
    case SYS_SEMAPHORE_GET_VALUE:
        s.gpr[3] = 0;
        s.gpr[4] = 1; // value = 1
        break;

    // ─── Event flags ─────────────────────────────────────────────
    case SYS_EVENT_FLAG_CREATE: {
        uint32_t id = hle_alloc_id();
        if (s.gpr[3] && s.gpr[3] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[3], id);
        s.gpr[3] = 0;
        break;
    }
    case SYS_EVENT_FLAG_DESTROY:
    case SYS_EVENT_FLAG_WAIT:
    case SYS_EVENT_FLAG_TRYWAIT:
    case SYS_EVENT_FLAG_SET:
    case SYS_EVENT_FLAG_CLEAR:
    case SYS_EVENT_FLAG_CANCEL:
        s.gpr[3] = 0;
        break;

    // ─── Event queue ─────────────────────────────────────────────
    case SYS_EVENT_QUEUE_CREATE: {
        uint32_t id = hle_alloc_id();
        if (s.gpr[3] && s.gpr[3] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[3], id);
        s.gpr[3] = 0;
        break;
    }
    case SYS_EVENT_QUEUE_DESTROY:
    case SYS_EVENT_QUEUE_DRAIN:
        s.gpr[3] = 0;
        break;
    case SYS_EVENT_QUEUE_RECEIVE:
    case SYS_EVENT_QUEUE_TRYRECEIVE:
        s.gpr[3] = 0;
        s.gpr[4] = 0; // no events (0 received)
        break;
    case SYS_EVENT_PORT_CREATE:
        s.gpr[3] = 0;
        s.gpr[4] = hle_alloc_id();
        break;
    case SYS_EVENT_PORT_DESTROY:
    case SYS_EVENT_PORT_CONNECT:
    case SYS_EVENT_PORT_DISCONNECT:
        s.gpr[3] = 0;
        break;

    // ─── Memory management ───────────────────────────────────────
    case SYS_MEMORY_ALLOCATE: {
        uint64_t size = s.gpr[3];
        uint64_t align = s.gpr[4] ? s.gpr[4] : 4096;
        uint64_t ptr = g_heapPtr;
        ptr = (ptr + align - 1) & ~(align - 1);
        s.gpr[3] = 0;       // CELL_OK
        s.gpr[4] = ptr;     // allocated address
        g_heapPtr = ptr + size;
        break;
    }

    case SYS_MEMORY_FREE:
        s.gpr[3] = 0; // CELL_OK (no-op for bump allocator)
        break;

    case SYS_MEMORY_GET_PAGE_ATTRIBUTE:
        // r3 = address; r4 = sys_page_attr_t* (output)
        if (s.gpr[4] && s.gpr[4] + 32 < PS3_SANDBOX_SIZE) {
            // zero the 32-byte page attribute struct, then stamp
            // attribute=read/write, page_size=0x10000 (64 KB).
            for (int i = 0; i < 32; i += 4) mem_write32(mem, s.gpr[4] + i, 0);
            mem_write32(mem, s.gpr[4] + 0,  0x00000000); // attribute hi
            mem_write32(mem, s.gpr[4] + 4,  0x00040000); // page_size 64KB
            mem_write32(mem, s.gpr[4] + 8,  0);
            mem_write32(mem, s.gpr[4] + 12, 0);
        }
        s.gpr[3] = 0;
        break;

    case SYS_MEMORY_GET_USER_MEMORY_SIZE: {
        // r3 = sys_memory_info_t* (2 × u32 BE: total_user_memory, available)
        const uint32_t total = 0x10000000; // 256 MB
        const uint32_t avail = 0x08000000; // 128 MB
        if (s.gpr[3] && s.gpr[3] + 8 < PS3_SANDBOX_SIZE) {
            mem_write32(mem, s.gpr[3] + 0, total);
            mem_write32(mem, s.gpr[3] + 4, avail);
        }
        s.gpr[3] = 0;
        break;
    }

    case SYS_MEMORY_GET_USER_MEMORY_STAT:
        if (s.gpr[3] && s.gpr[3] + 16 < PS3_SANDBOX_SIZE) {
            for (int i = 0; i < 16; i += 4) mem_write32(mem, s.gpr[3] + i, 0);
        }
        s.gpr[3] = 0;
        break;

    case SYS_MEMORY_CONTAINER_CREATE:
    case SYS_MEMORY_CONTAINER_CREATE2: {
        // r3 = out id*, r4 = size
        uint32_t id = hle_alloc_id();
        if (s.gpr[3] && s.gpr[3] + 4 < PS3_SANDBOX_SIZE)
            mem_write32(mem, s.gpr[3], id);
        s.gpr[3] = 0;
        break;
    }
    case SYS_MEMORY_CONTAINER_DESTROY:
    case SYS_MEMORY_CONTAINER_DESTROY2:
        s.gpr[3] = 0;
        break;
    case SYS_MEMORY_CONTAINER_GET_SIZE:
        if (s.gpr[3] && s.gpr[3] + 8 < PS3_SANDBOX_SIZE) {
            mem_write32(mem, s.gpr[3] + 0, 0x04000000);
            mem_write32(mem, s.gpr[3] + 4, 0x02000000);
        }
        s.gpr[3] = 0;
        break;
    case SYS_MEMORY_ALLOCATE_FROM_CONTAINER: {
        uint64_t size = s.gpr[3];
        uint64_t align = 0x10000;
        uint64_t ptr = (g_heapPtr + align - 1) & ~(align - 1);
        if (s.gpr[5] && s.gpr[5] + 4 < PS3_SANDBOX_SIZE)
            mem_write32(mem, s.gpr[5], (uint32_t)ptr);
        g_heapPtr = ptr + size;
        s.gpr[3] = 0;
        break;
    }

    case SYS_MMAPPER_ALLOCATE_ADDRESS: {
        // r3 = size, r4 = flags, r5 = alignment, r6 = alloc_addr_ptr
        uint64_t size = s.gpr[3];
        uint64_t align = s.gpr[5] ? s.gpr[5] : 0x100000;
        uint64_t ptr = g_heapPtr;
        ptr = (ptr + align - 1) & ~(align - 1);
        if (s.gpr[6] && s.gpr[6] < PS3_SANDBOX_SIZE - 8)
            mem_write64(mem, s.gpr[6], ptr);
        g_heapPtr = ptr + size;
        s.gpr[3] = 0;
        break;
    }
    case SYS_MMAPPER_ALLOCATE_SHARED_MEMORY: {
        uint32_t id = hle_alloc_id();
        if (s.gpr[6] && s.gpr[6] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[6], id);
        s.gpr[3] = 0;
        break;
    }
    case SYS_MMAPPER_MAP_SHARED_MEMORY:
    case SYS_MMAPPER_UNMAP_SHARED_MEMORY:
    case SYS_MMAPPER_FREE_SHARED_MEMORY:
        s.gpr[3] = 0;
        break;

    // ─── SPU management ──────────────────────────────────────────
    case SYS_SPU_THREAD_GROUP_CREATE:
    case SYS_SPU_THREAD_GROUP_DESTROY:
    case SYS_SPU_THREAD_INITIALIZE:
    case SYS_SPU_THREAD_GROUP_START:
    case SYS_SPU_THREAD_GROUP_SUSPEND:
    case SYS_SPU_THREAD_GROUP_RESUME:
    case SYS_SPU_THREAD_GROUP_TERMINATE:
    case SYS_SPU_THREAD_GROUP_JOIN:
    case SYS_SPU_THREAD_SET_ARGUMENT:
    case SYS_SPU_THREAD_GET_EXIT_STATUS:
    case SYS_SPU_THREAD_WRITE_LS:
    case SYS_SPU_THREAD_READ_LS:
    case SYS_SPU_THREAD_WRITE_SNR:
    case SYS_SPU_THREAD_CONNECT_EVENT:
    case SYS_SPU_THREAD_DISCONNECT_EVENT:
    case SYS_SPU_THREAD_BIND_QUEUE:
    case SYS_SPU_THREAD_UNBIND_QUEUE:
    case SYS_RAW_SPU_CREATE:
    case SYS_RAW_SPU_DESTROY:
        if (hle_signal) atomicAdd((uint32_t*)hle_signal, 1);
        s.gpr[3] = 0;
        break;

    case SYS_SPU_IMAGE_OPEN:
    case SYS_SPU_IMAGE_IMPORT:
    case SYS_SPU_IMAGE_CLOSE:
        s.gpr[3] = 0;
        break;

    // ─── PRX (shared libraries) ──────────────────────────────────
    case SYS_PRX_LOAD_MODULE:
    case SYS_PRX_START_MODULE:
    case SYS_PRX_STOP_MODULE:
    case SYS_PRX_UNLOAD_MODULE:
    case SYS_PRX_REGISTER_MODULE:
    case SYS_PRX_GET_MODULE_LIST:
    case SYS_PRX_GET_MODULE_INFO:
    case SYS_PRX_GET_MODULE_ID_BY_NAME:
        s.gpr[3] = 0;
        break;

    // ─── Filesystem (stubs) ──────────────────────────────────────
    case SYS_FS_OPEN: {
        uint32_t fd = hle_alloc_id();
        // r5 = fd_ptr
        if (s.gpr[5] && s.gpr[5] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[5], fd);
        s.gpr[3] = 0;
        break;
    }
    case SYS_FS_CLOSE:
        s.gpr[3] = 0;
        break;
    case SYS_FS_READ:
        s.gpr[3] = 0;
        // r5 = nread_ptr → 0 bytes read
        if (s.gpr[5] && s.gpr[5] < PS3_SANDBOX_SIZE - 8)
            mem_write64(mem, s.gpr[5], 0);
        break;
    case SYS_FS_WRITE:
        s.gpr[3] = 0;
        if (s.gpr[5] && s.gpr[5] < PS3_SANDBOX_SIZE - 8)
            mem_write64(mem, s.gpr[5], s.gpr[4]); // pretend all bytes written
        break;
    case SYS_FS_LSEEK:
        s.gpr[3] = 0;
        if (s.gpr[4] && s.gpr[4] < PS3_SANDBOX_SIZE - 8)
            mem_write64(mem, s.gpr[4], 0);
        break;
    case SYS_FS_STAT:
    case SYS_FS_FSTAT:
        // Zero-fill stat buffer (r4 = buf_ptr)
        if (s.gpr[4] && s.gpr[4] < PS3_SANDBOX_SIZE - 128) {
            for (int i = 0; i < 128; i += 8)
                mem_write64(mem, s.gpr[4] + i, 0);
        }
        s.gpr[3] = 0;
        break;
    case SYS_FS_MKDIR:
    case SYS_FS_RENAME:
    case SYS_FS_RMDIR:
    case SYS_FS_UNLINK:
        s.gpr[3] = 0;
        break;
    case SYS_FS_OPENDIR: {
        uint32_t fd = hle_alloc_id();
        if (s.gpr[4] && s.gpr[4] < PS3_SANDBOX_SIZE - 4)
            mem_write32(mem, s.gpr[4], fd);
        s.gpr[3] = 0;
        break;
    }
    case SYS_FS_READDIR:
        s.gpr[3] = -1; // CELL_FS_ENOENT (end of directory)
        break;
    case SYS_FS_CLOSEDIR:
        s.gpr[3] = 0;
        break;

    // ─── RSX (GPU) management ────────────────────────────────────
    case SYS_RSX_DEVICE_MAP:
    case SYS_RSX_DEVICE_UNMAP:
    case SYS_RSX_CONTEXT_ALLOCATE:
    case SYS_RSX_CONTEXT_FREE:
    case SYS_RSX_CONTEXT_IOMAP:
    case SYS_RSX_CONTEXT_ATTRIBUTE:
        if (hle_signal) atomicAdd((uint32_t*)hle_signal, 1);
        s.gpr[3] = 0;
        break;

    // ─── HLE GCM bridge: PPC code drives RSX via custom syscalls ──
    //
    // Real games never go through `sc` for graphics — they call libgcm
    // user-mode functions that write FIFO words directly into a ring.
    // We don't have an SPRX loader yet, so for synthetic test programs
    // we expose two custom syscalls in an unused range (0xC710/0xC711)
    // that emit FIFO method headers + payload into a guest-visible ring
    // at PS3_GCM_FIFO_BASE. The host pumps that ring into rsx_process_fifo
    // after the PPC thread halts.
    //
    // Layout at PS3_GCM_FIFO_BASE:
    //   [+0]      uint32 cursor (in dwords past the cursor word itself)
    //   [+4..]    native-endian FIFO dwords
    case 0xC710: {
        // Emit single-arg method. r3=method, r4=data
        uint32_t method = (uint32_t)s.gpr[3];
        uint32_t data   = (uint32_t)s.gpr[4];
        uint32_t header = ((1u & 0x7FFu) << 18) | (method & 0xFFFCu);
        // atomic bump of cursor (in dwords)
        uint32_t* base = (uint32_t*)(mem + (PS3_GCM_FIFO_BASE & (PS3_SANDBOX_SIZE - 1)));
        uint32_t pos   = atomicAdd(base, 2u);
        // Write native-endian (rsx_process_fifo expects native uint32)
        if (pos + 2 < (PS3_GCM_FIFO_LIMIT_DWORDS - 1)) {
            base[1 + pos]     = header;
            base[1 + pos + 1] = data;
        }
        s.gpr[3] = 0;
        break;
    }
    case 0xC711: {
        // Emit raw FIFO dword (already-formed header or payload).
        // Lets PPC code burst multi-payload methods one word at a time.
        // r3=raw word
        uint32_t word = (uint32_t)s.gpr[3];
        uint32_t* base = (uint32_t*)(mem + (PS3_GCM_FIFO_BASE & (PS3_SANDBOX_SIZE - 1)));
        uint32_t pos   = atomicAdd(base, 1u);
        if (pos + 1 < (PS3_GCM_FIFO_LIMIT_DWORDS - 1)) {
            base[1 + pos] = word;
        }
        s.gpr[3] = 0;
        break;
    }

    default:
        // Log unknown syscall for debugging
        if (hle_log) {
            uint32_t idx = atomicAdd(hle_log, 1);
            if (idx < 255) {
                hle_log[1 + idx] = sc_num;
            }
        }
        s.gpr[3] = 0; // Fake success to keep game running
        break;
    }
}

// ═══════════════════════════════════════════════════════════════
// Branch Evaluation (BO field decoding)
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ bool evalBranch(PPEState& s, uint32_t bo, uint32_t bi) {
    // BO field encoding:
    // bit 0 (4): branch if condition true
    // bit 1 (3): don't test condition
    // bit 2 (2): decrement CTR
    // bit 3 (1): don't decrement CTR
    // bit 4 (0): branch hint (not used for correctness)

    bool ctr_ok = true;
    if (!(bo & 0x04)) {  // decrement CTR
        s.ctr--;
        ctr_ok = (bo & 0x02) ? (s.ctr == 0) : (s.ctr != 0);
    }
    bool cond_ok = true;
    if (!(bo & 0x10)) {  // test condition
        bool bit = getCRBit(s.cr, bi);
        cond_ok = (bo & 0x08) ? bit : !bit;
    }
    return ctr_ok && cond_ok;
}

// ═══════════════════════════════════════════════════════════════
// Rotate Mask Helper (for rlwinm / rlwimi)
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ uint32_t rotateMask32(uint32_t mb, uint32_t me) {
    uint32_t mask = 0;
    if (mb <= me) {
        for (uint32_t i = mb; i <= me; i++) mask |= (1U << (31 - i));
    } else {
        for (uint32_t i = 0; i <= me; i++)   mask |= (1U << (31 - i));
        for (uint32_t i = mb; i <= 31; i++)   mask |= (1U << (31 - i));
    }
    return mask;
}

__device__ __forceinline__ uint32_t rotl32(uint32_t v, uint32_t n) {
    n &= 31;
    return (v << n) | (v >> (32 - n));
}

// ═══════════════════════════════════════════════════════════════
// 64-bit Rotate and Mask Helpers (for PPC64 rotate instructions)
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ uint64_t rotl64(uint64_t v, uint32_t n) {
    n &= 63;
    return (n == 0) ? v : (v << n) | (v >> (64 - n));
}

// PPC64 mask: bit b to bit e (0=MSB=bit63 in uint64_t)
// If b <= e: contiguous mask. If b > e: wrapping mask.
__device__ __forceinline__ uint64_t mask64(uint32_t b, uint32_t e) {
    uint64_t mb = (~0ULL) >> b;
    uint64_t me = (~0ULL) << (63 - e);
    return (b <= e) ? (mb & me) : (mb | me);
}

// Extract 6-bit sh field for GRP30: sh(0:4) = bits 16:20, sh5 = bit 30
__device__ __forceinline__ uint32_t SH64(uint32_t inst) {
    return ((inst >> 11) & 0x1F) | (((inst >> 1) & 1) << 5);
}

// Extract 6-bit mb/me field for GRP30: mb(0:4) = bits 21:25, mb5 = bit 26
__device__ __forceinline__ uint32_t MB64(uint32_t inst) {
    return ((inst >> 6) & 0x1F) | (((inst >> 5) & 1) << 5);
}

// Extract FP register C (frC) field: bits 21:25 in standard notation = bits 6:10 from LSB
__device__ __forceinline__ uint32_t FRC(uint32_t inst) {
    return (inst >> 6) & 0x1F;
}

// CR field update for 64-bit comparison
__device__ __forceinline__ void setCRField64(uint32_t& cr, int field, int64_t result, uint64_t xer) {
    uint32_t val = 0;
    if (result < 0)       val = 0x8;
    else if (result > 0)  val = 0x4;
    else                  val = 0x2;
    if (xer & (1ULL << 31)) val |= 0x1;
    int shift = (7 - field) * 4;
    cr = (cr & ~(0xFU << shift)) | (val << shift);
}

// Set/clear a single CR bit
__device__ __forceinline__ void setCRBit(uint32_t& cr, int bit, bool val) {
    if (val) cr |=  (1U << (31 - bit));
    else     cr &= ~(1U << (31 - bit));
}

// ═══════════════════════════════════════════════════════════════
// Single-Step Execute — decode and run one PPC instruction
// Returns 0 = ok, 1 = halted, 2 = unimplemented
// ═══════════════════════════════════════════════════════════════

__device__ static int execOne(PPEState& s, uint8_t* mem,
                               uint32_t* hle_log, volatile uint32_t* hle_signal) {
    if (s.halted) return 1;

    // Fetch (big-endian 32-bit instruction)
    uint32_t inst = mem_read32(mem, s.pc);
    uint32_t opcd = OPCD(inst);
    s.npc = s.pc + 4;

    switch (opcd) {

    // ─── Immediate ALU ────────────────────────────────────────

    case OP_ADDI: {  // also "li rD, SIMM" when rA=0
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        s.gpr[rd] = (ra == 0) ? (uint64_t)imm : (uint64_t)((int64_t)s.gpr[ra] + imm);
        break;
    }

    case OP_ADDIS: {  // also "lis rD, SIMM" when rA=0
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst) << 16;
        s.gpr[rd] = (ra == 0) ? (uint64_t)imm : (uint64_t)((int64_t)s.gpr[ra] + imm);
        break;
    }

    case OP_ADDIC: {
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        uint64_t a = s.gpr[ra];
        uint64_t result = a + (uint64_t)imm;
        s.gpr[rd] = result;
        setCA(s.xer, (uint32_t)result < (uint32_t)a);
        break;
    }

    case OP_ADDIC_D: {
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        uint64_t a = s.gpr[ra];
        uint64_t result = a + (uint64_t)imm;
        s.gpr[rd] = result;
        setCA(s.xer, (uint32_t)result < (uint32_t)a);
        setCRField(s.cr, 0, (int32_t)(uint32_t)result, s.xer);
        break;
    }

    case OP_SUBFIC: {
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        uint64_t result = (uint64_t)imm - s.gpr[ra];
        s.gpr[rd] = result;
        setCA(s.xer, (uint32_t)s.gpr[ra] <= (uint32_t)(uint64_t)imm);
        break;
    }

    case OP_MULLI: {
        uint32_t rd = RD(inst), ra = RA(inst);
        int64_t imm = SIMM16(inst);
        s.gpr[rd] = (uint64_t)((int64_t)s.gpr[ra] * imm);
        break;
    }

    // ─── Compare ──────────────────────────────────────────────

    case OP_CMPI: {
        uint32_t bf = RD(inst) >> 2;
        uint32_t L = (RD(inst) >> 1) & 1;
        uint32_t ra = RA(inst);
        int64_t imm = SIMM16(inst);
        if (L) {
            int64_t a = (int64_t)s.gpr[ra];
            int64_t diff = (a > imm) ? 1 : (a < imm) ? -1 : 0;
            setCRField(s.cr, bf, diff, s.xer);
        } else {
            int32_t a = (int32_t)(uint32_t)s.gpr[ra];
            setCRField(s.cr, bf, (int64_t)a - imm, s.xer);
        }
        break;
    }

    case OP_CMPLI: {
        uint32_t bf = RD(inst) >> 2;
        uint32_t L = (RD(inst) >> 1) & 1;
        uint32_t ra = RA(inst);
        uint64_t imm = UIMM16(inst);
        if (L) {
            uint64_t a = s.gpr[ra];
            int64_t diff = (a > imm) ? 1 : (a < imm) ? -1 : 0;
            setCRField(s.cr, bf, diff, s.xer);
        } else {
            uint32_t a = (uint32_t)s.gpr[ra];
            int64_t diff = (a > (uint32_t)imm) ? 1 : (a < (uint32_t)imm) ? -1 : 0;
            setCRField(s.cr, bf, diff, s.xer);
        }
        break;
    }

    // ─── Logical Immediate ────────────────────────────────────

    case OP_ORI: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] | UIMM16(inst);
        break;
    }

    case OP_ORIS: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] | (UIMM16(inst) << 16);
        break;
    }

    case OP_XORI: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] ^ UIMM16(inst);
        break;
    }

    case OP_XORIS: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] ^ (UIMM16(inst) << 16);
        break;
    }

    case OP_ANDI: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] & UIMM16(inst);
        setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    case OP_ANDIS: {
        uint32_t rs = RS(inst), ra = RA(inst);
        s.gpr[ra] = s.gpr[rs] & (UIMM16(inst) << 16);
        setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    // ─── Rotate / Mask ───────────────────────────────────────

    case OP_RLWINM: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint32_t sh = SH(inst), mb = MB(inst), me = ME(inst);
        uint32_t rotated = rotl32((uint32_t)s.gpr[rs], sh);
        uint32_t mask = rotateMask32(mb, me);
        s.gpr[ra] = rotated & mask;
        if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    case OP_RLWIMI: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint32_t sh = SH(inst), mb = MB(inst), me = ME(inst);
        uint32_t rotated = rotl32((uint32_t)s.gpr[rs], sh);
        uint32_t mask = rotateMask32(mb, me);
        s.gpr[ra] = (rotated & mask) | ((uint32_t)s.gpr[ra] & ~mask);
        if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    case OP_RLWNM: {
        uint32_t rs = RS(inst), ra = RA(inst), rb = RB(inst);
        uint32_t mb = MB(inst), me = ME(inst);
        uint32_t rotated = rotl32((uint32_t)s.gpr[rs], (uint32_t)s.gpr[rb] & 0x1F);
        uint32_t mask = rotateMask32(mb, me);
        s.gpr[ra] = rotated & mask;
        if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
        break;
    }

    // ─── Branches ─────────────────────────────────────────────

    case OP_B: {
        int64_t disp = LI26(inst);
        s.npc = AA(inst) ? (uint64_t)disp : s.pc + (uint64_t)disp;
        if (LK(inst)) s.lr = s.pc + 4;
        break;
    }

    case OP_BC: {
        uint32_t bo = BO(inst), bi = BI(inst);
        if (evalBranch(s, bo, bi)) {
            int64_t disp = BD16(inst);
            s.npc = AA(inst) ? (uint64_t)disp : s.pc + (uint64_t)disp;
        }
        if (LK(inst)) s.lr = s.pc + 4;
        break;
    }

    case OP_SC: {
        handleSyscall(s, mem, hle_log, hle_signal);
        break;
    }

    // ─── Group 19 (Branch to LR/CTR, CR ops) ─────────────────

    case OP_GRP19: {
        uint32_t xo = XO_10(inst);
        switch (xo) {
        case XO_BCLR: {
            uint32_t bo = BO(inst), bi = BI(inst);
            if (evalBranch(s, bo, bi)) {
                s.npc = s.lr & ~3ULL;
            }
            if (LK(inst)) s.lr = s.pc + 4;
            break;
        }
        case XO_BCCTR: {
            uint32_t bo = BO(inst), bi = BI(inst);
            if (evalBranch(s, bo, bi)) {
                s.npc = s.ctr & ~3ULL;
            }
            if (LK(inst)) s.lr = s.pc + 4;
            break;
        }
        case XO_MCRF: {
            uint32_t bf = RD(inst) >> 2;
            uint32_t bfa = RA(inst) >> 2;
            int src_shift = (7 - bfa) * 4;
            int dst_shift = (7 - bf) * 4;
            uint32_t val = (s.cr >> src_shift) & 0xF;
            s.cr = (s.cr & ~(0xFU << dst_shift)) | (val << dst_shift);
            break;
        }
        case XO_CROR: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            setCRBit(s.cr, bt, getCRBit(s.cr, ba) | getCRBit(s.cr, bb));
            break;
        }
        case XO_CRXOR: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            setCRBit(s.cr, bt, getCRBit(s.cr, ba) ^ getCRBit(s.cr, bb));
            break;
        }
        case XO_CRAND: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            setCRBit(s.cr, bt, getCRBit(s.cr, ba) & getCRBit(s.cr, bb));
            break;
        }
        case XO_CRNOR: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            setCRBit(s.cr, bt, !(getCRBit(s.cr, ba) | getCRBit(s.cr, bb)));
            break;
        }
        case XO_CRANDC: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            setCRBit(s.cr, bt, getCRBit(s.cr, ba) & !getCRBit(s.cr, bb));
            break;
        }
        case XO_CREQV: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            setCRBit(s.cr, bt, getCRBit(s.cr, ba) == getCRBit(s.cr, bb));
            break;
        }
        case XO_CRNAND: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            setCRBit(s.cr, bt, !(getCRBit(s.cr, ba) & getCRBit(s.cr, bb)));
            break;
        }
        case XO_CRORC: {
            uint32_t bt = RD(inst), ba = RA(inst), bb = RB(inst);
            setCRBit(s.cr, bt, getCRBit(s.cr, ba) | !getCRBit(s.cr, bb));
            break;
        }
        case XO_ISYNC:
            __threadfence();
            break;
        default:
            return 2;
        }
        break;
    }

    // ─── Load/Store (D-form) ──────────────────────────────────

    case OP_LWZ: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.gpr[rd] = mem_read32(mem, ea);
        break;
    }

    case OP_LWZU: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        s.gpr[rd] = mem_read32(mem, ea);
        s.gpr[ra] = ea;
        break;
    }

    case OP_LBZ: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.gpr[rd] = mem_read8(mem, ea);
        break;
    }

    case OP_LBZU: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        s.gpr[rd] = mem_read8(mem, ea);
        s.gpr[ra] = ea;
        break;
    }

    case OP_LHZ: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.gpr[rd] = mem_read16(mem, ea);
        break;
    }

    case OP_LHZU: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        s.gpr[rd] = mem_read16(mem, ea);
        s.gpr[ra] = ea;
        break;
    }

    case OP_LHA: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.gpr[rd] = (uint64_t)(int64_t)(int16_t)mem_read16(mem, ea);
        break;
    }

    case OP_STW: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_write32(mem, ea, (uint32_t)s.gpr[rs]);
        break;
    }

    case OP_STWU: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        mem_write32(mem, ea, (uint32_t)s.gpr[rs]);
        s.gpr[ra] = ea;
        break;
    }

    case OP_STB: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_write8(mem, ea, (uint8_t)s.gpr[rs]);
        break;
    }

    case OP_STBU: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        mem_write8(mem, ea, (uint8_t)s.gpr[rs]);
        s.gpr[ra] = ea;
        break;
    }

    case OP_STH: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_write16(mem, ea, (uint16_t)s.gpr[rs]);
        break;
    }

    case OP_STHU: {
        uint32_t rs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        mem_write16(mem, ea, (uint16_t)s.gpr[rs]);
        s.gpr[ra] = ea;
        break;
    }

    case OP_LHAU: {
        uint32_t rd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        s.gpr[rd] = (uint64_t)(int64_t)(int16_t)mem_read16(mem, ea);
        s.gpr[ra] = ea;
        break;
    }

    // ─── FP Load/Store ────────────────────────────────────────

    case OP_LFS: {
        uint32_t frd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.fpr[frd] = (double)mem_readf32(mem, ea);
        break;
    }

    case OP_LFSU: {
        uint32_t frd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        s.fpr[frd] = (double)mem_readf32(mem, ea);
        s.gpr[ra] = ea;
        break;
    }

    case OP_LFD: {
        uint32_t frd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        s.fpr[frd] = mem_readf64(mem, ea);
        break;
    }

    case OP_LFDU: {
        uint32_t frd = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        s.fpr[frd] = mem_readf64(mem, ea);
        s.gpr[ra] = ea;
        break;
    }

    case OP_STFS: {
        uint32_t frs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_writef32(mem, ea, (float)s.fpr[frs]);
        break;
    }

    case OP_STFSU: {
        uint32_t frs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        mem_writef32(mem, ea, (float)s.fpr[frs]);
        s.gpr[ra] = ea;
        break;
    }

    case OP_STFD: {
        uint32_t frs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        mem_writef64(mem, ea, s.fpr[frs]);
        break;
    }

    case OP_STFDU: {
        uint32_t frs = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + s.gpr[ra];
        mem_writef64(mem, ea, s.fpr[frs]);
        s.gpr[ra] = ea;
        break;
    }

    // ─── Group 31 (ALU extended, SPR, indexed load/store) ─────

    case OP_GRP31: {
        uint32_t xo = XO_10(inst);
        uint32_t rd = RD(inst), ra = RA(inst), rb = RB(inst);

        switch (xo) {

        // ── 32-bit Arithmetic ──────────────────────────────────
        case XO_ADD: {
            s.gpr[rd] = (uint32_t)((uint32_t)s.gpr[ra] + (uint32_t)s.gpr[rb]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_SUBF: {
            s.gpr[rd] = (uint32_t)((uint32_t)s.gpr[rb] - (uint32_t)s.gpr[ra]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_SUBFC: {
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            uint64_t result = (uint64_t)b + (uint64_t)(~a) + 1ULL;
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_ADDC: {
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            uint64_t result = (uint64_t)a + (uint64_t)b;
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_ADDE: {
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            uint64_t result = (uint64_t)a + (uint64_t)b + (getCA(s.xer) ? 1ULL : 0ULL);
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_SUBFE: {
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            uint64_t result = (uint64_t)(~a) + (uint64_t)b + (getCA(s.xer) ? 1ULL : 0ULL);
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_ADDZE: {
            uint32_t a = (uint32_t)s.gpr[ra];
            uint64_t result = (uint64_t)a + (getCA(s.xer) ? 1ULL : 0ULL);
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_SUBFZE: {
            uint32_t a = (uint32_t)s.gpr[ra];
            uint64_t result = (uint64_t)(~a) + (getCA(s.xer) ? 1ULL : 0ULL);
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_ADDME: {
            uint32_t a = (uint32_t)s.gpr[ra];
            uint64_t result = (uint64_t)a + (getCA(s.xer) ? 1ULL : 0ULL) + 0xFFFFFFFFULL;
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_SUBFME: {
            uint32_t a = (uint32_t)s.gpr[ra];
            uint64_t result = (uint64_t)(~a) + (getCA(s.xer) ? 1ULL : 0ULL) + 0xFFFFFFFFULL;
            s.gpr[rd] = (uint32_t)result;
            setCA(s.xer, result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_NEG: {
            s.gpr[rd] = (uint32_t)(-(int32_t)(uint32_t)s.gpr[ra]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_MULLW: {
            int32_t a = (int32_t)(uint32_t)s.gpr[ra];
            int32_t b = (int32_t)(uint32_t)s.gpr[rb];
            s.gpr[rd] = (uint64_t)(int64_t)((int64_t)a * (int64_t)b);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_MULHW: {
            int64_t result = (int64_t)(int32_t)(uint32_t)s.gpr[ra] *
                             (int64_t)(int32_t)(uint32_t)s.gpr[rb];
            s.gpr[rd] = (uint32_t)(result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_MULHWU: {
            uint64_t result = (uint64_t)(uint32_t)s.gpr[ra] *
                              (uint64_t)(uint32_t)s.gpr[rb];
            s.gpr[rd] = (uint32_t)(result >> 32);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_DIVW: {
            int32_t a = (int32_t)(uint32_t)s.gpr[ra];
            int32_t b = (int32_t)(uint32_t)s.gpr[rb];
            s.gpr[rd] = (b != 0 && !(a == (int32_t)0x80000000 && b == -1))
                         ? (uint32_t)(a / b) : 0;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_DIVWU: {
            uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
            s.gpr[rd] = (b != 0) ? (a / b) : 0;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[rd], s.xer);
            break;
        }

        // ── 64-bit Arithmetic ──────────────────────────────────
        case XO_MULLD: {
            s.gpr[rd] = (uint64_t)((int64_t)s.gpr[ra] * (int64_t)s.gpr[rb]);
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_MULHD: {
            __int128 result = (__int128)(int64_t)s.gpr[ra] * (__int128)(int64_t)s.gpr[rb];
            s.gpr[rd] = (uint64_t)(result >> 64);
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_MULHDU: {
            unsigned __int128 result = (unsigned __int128)s.gpr[ra] * (unsigned __int128)s.gpr[rb];
            s.gpr[rd] = (uint64_t)(result >> 64);
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_DIVD: {
            int64_t a = (int64_t)s.gpr[ra], b = (int64_t)s.gpr[rb];
            s.gpr[rd] = (b != 0 && !(a == (int64_t)0x8000000000000000LL && b == -1))
                         ? (uint64_t)(a / b) : 0;
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[rd], s.xer);
            break;
        }
        case XO_DIVDU: {
            uint64_t a = s.gpr[ra], b = s.gpr[rb];
            s.gpr[rd] = (b != 0) ? (a / b) : 0;
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[rd], s.xer);
            break;
        }

        // ── Logical ────────────────────────────────────────────
        case XO_AND: {
            s.gpr[ra] = s.gpr[rd] & s.gpr[rb];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_OR: {
            s.gpr[ra] = s.gpr[rd] | s.gpr[rb];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_XOR: {
            s.gpr[ra] = s.gpr[rd] ^ s.gpr[rb];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_ANDC: {
            s.gpr[ra] = s.gpr[rd] & ~s.gpr[rb];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_ORC: {
            s.gpr[ra] = s.gpr[rd] | ~s.gpr[rb];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_NOR: {
            s.gpr[ra] = ~(s.gpr[rd] | s.gpr[rb]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_NAND: {
            s.gpr[ra] = ~(s.gpr[rd] & s.gpr[rb]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_EQV: {
            s.gpr[ra] = ~(s.gpr[rd] ^ s.gpr[rb]);
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_EXTSB: {
            s.gpr[ra] = (uint64_t)(int64_t)(int8_t)(uint8_t)s.gpr[rd];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_EXTSH: {
            s.gpr[ra] = (uint64_t)(int64_t)(int16_t)(uint16_t)s.gpr[rd];
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_EXTSW: {
            s.gpr[ra] = (uint64_t)(int64_t)(int32_t)(uint32_t)s.gpr[rd];
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_CNTLZW: {
            uint32_t val = (uint32_t)s.gpr[rd];
            s.gpr[ra] = val ? __clz(val) : 32;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_CNTLZD: {
            uint64_t val = s.gpr[rd];
            s.gpr[ra] = val ? __clzll(val) : 64;
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[ra], s.xer);
            break;
        }

        // ── 32-bit Shifts ──────────────────────────────────────
        case XO_SLW: {
            uint32_t sh = (uint32_t)s.gpr[rb] & 0x3F;
            s.gpr[ra] = (sh < 32) ? ((uint32_t)s.gpr[rd] << sh) : 0;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_SRW: {
            uint32_t sh = (uint32_t)s.gpr[rb] & 0x3F;
            s.gpr[ra] = (sh < 32) ? ((uint32_t)s.gpr[rd] >> sh) : 0;
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_SRAW: {
            uint32_t sh = (uint32_t)s.gpr[rb] & 0x3F;
            int32_t val = (int32_t)(uint32_t)s.gpr[rd];
            if (sh == 0) {
                s.gpr[ra] = (uint32_t)val;
                setCA(s.xer, false);
            } else if (sh < 32) {
                bool carry = (val < 0) && ((val & ((1 << sh) - 1)) != 0);
                s.gpr[ra] = (uint32_t)(val >> sh);
                setCA(s.xer, carry);
            } else {
                s.gpr[ra] = (val < 0) ? 0xFFFFFFFF : 0;
                setCA(s.xer, val < 0);
            }
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_SRAWI: {
            uint32_t sh = SH(inst);
            int32_t val = (int32_t)(uint32_t)s.gpr[rd];
            if (sh == 0) {
                s.gpr[ra] = (uint32_t)val;
                setCA(s.xer, false);
            } else {
                bool carry = (val < 0) && ((val & ((1 << sh) - 1)) != 0);
                s.gpr[ra] = (uint32_t)(val >> sh);
                setCA(s.xer, carry);
            }
            if (RC(inst)) setCRField(s.cr, 0, (int32_t)(uint32_t)s.gpr[ra], s.xer);
            break;
        }

        // ── 64-bit Shifts ──────────────────────────────────────
        case XO_SLD: {
            uint32_t sh = (uint32_t)s.gpr[rb] & 0x7F;
            s.gpr[ra] = (sh < 64) ? (s.gpr[rd] << sh) : 0;
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_SRD: {
            uint32_t sh = (uint32_t)s.gpr[rb] & 0x7F;
            s.gpr[ra] = (sh < 64) ? (s.gpr[rd] >> sh) : 0;
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[ra], s.xer);
            break;
        }
        case XO_SRAD: {
            uint32_t sh = (uint32_t)s.gpr[rb] & 0x7F;
            int64_t val = (int64_t)s.gpr[rd];
            if (sh == 0) {
                s.gpr[ra] = (uint64_t)val;
                setCA(s.xer, false);
            } else if (sh < 64) {
                bool carry = (val < 0) && ((val & ((1LL << sh) - 1)) != 0);
                s.gpr[ra] = (uint64_t)(val >> sh);
                setCA(s.xer, carry);
            } else {
                s.gpr[ra] = (val < 0) ? ~0ULL : 0;
                setCA(s.xer, val < 0);
            }
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[ra], s.xer);
            break;
        }
        case 826: case 827: { // SRADI: XO_9=413, XO_10 = 826|827 (bit 30 = sh5)
            uint32_t sh = ((inst >> 11) & 0x1F) | (((inst >> 1) & 1) << 5);
            int64_t val = (int64_t)s.gpr[rd];
            if (sh == 0) {
                s.gpr[ra] = (uint64_t)val;
                setCA(s.xer, false);
            } else {
                bool carry = (val < 0) && ((val & ((1ULL << sh) - 1)) != 0);
                s.gpr[ra] = (uint64_t)(val >> sh);
                setCA(s.xer, carry);
            }
            if (RC(inst)) setCRField64(s.cr, 0, (int64_t)s.gpr[ra], s.xer);
            break;
        }

        // ── Compare ────────────────────────────────────────────
        case XO_CMP: {
            uint32_t bf = rd >> 2;
            uint32_t L = (rd >> 1) & 1;
            if (L) {
                int64_t a = (int64_t)s.gpr[ra], b = (int64_t)s.gpr[rb];
                int64_t diff = (a > b) ? 1 : (a < b) ? -1 : 0;
                setCRField(s.cr, bf, diff, s.xer);
            } else {
                int32_t a = (int32_t)(uint32_t)s.gpr[ra];
                int32_t b = (int32_t)(uint32_t)s.gpr[rb];
                setCRField(s.cr, bf, (int64_t)a - (int64_t)b, s.xer);
            }
            break;
        }
        case XO_CMPL: {
            uint32_t bf = rd >> 2;
            uint32_t L = (rd >> 1) & 1;
            if (L) {
                uint64_t a = s.gpr[ra], b = s.gpr[rb];
                int64_t diff = (a > b) ? 1 : (a < b) ? -1 : 0;
                setCRField(s.cr, bf, diff, s.xer);
            } else {
                uint32_t a = (uint32_t)s.gpr[ra], b = (uint32_t)s.gpr[rb];
                int64_t diff = (a > b) ? 1 : (a < b) ? -1 : 0;
                setCRField(s.cr, bf, diff, s.xer);
            }
            break;
        }

        // ── Indexed Load/Store (32-bit) ────────────────────────
        case XO_LWZX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = mem_read32(mem, ea);
            break;
        }
        case XO_LWZUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            s.gpr[rd] = mem_read32(mem, ea);
            s.gpr[ra] = ea;
            break;
        }
        case XO_LBZX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = mem_read8(mem, ea);
            break;
        }
        case XO_LBZUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            s.gpr[rd] = mem_read8(mem, ea);
            s.gpr[ra] = ea;
            break;
        }
        case XO_LHZX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = mem_read16(mem, ea);
            break;
        }
        case XO_LHZUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            s.gpr[rd] = mem_read16(mem, ea);
            s.gpr[ra] = ea;
            break;
        }
        case XO_LHAX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = (uint64_t)(int64_t)(int16_t)mem_read16(mem, ea);
            break;
        }
        case XO_LHAUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            s.gpr[rd] = (uint64_t)(int64_t)(int16_t)mem_read16(mem, ea);
            s.gpr[ra] = ea;
            break;
        }
        case XO_STWX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_write32(mem, ea, (uint32_t)s.gpr[rd]);
            break;
        }
        case XO_STWUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            mem_write32(mem, ea, (uint32_t)s.gpr[rd]);
            s.gpr[ra] = ea;
            break;
        }
        case XO_STBX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_write8(mem, ea, (uint8_t)s.gpr[rd]);
            break;
        }
        case XO_STBUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            mem_write8(mem, ea, (uint8_t)s.gpr[rd]);
            s.gpr[ra] = ea;
            break;
        }
        case XO_STHX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_write16(mem, ea, (uint16_t)s.gpr[rd]);
            break;
        }
        case XO_STHUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            mem_write16(mem, ea, (uint16_t)s.gpr[rd]);
            s.gpr[ra] = ea;
            break;
        }

        // ── Indexed Load/Store (64-bit) ────────────────────────
        case XO_LDX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = mem_read64(mem, ea);
            break;
        }
        case XO_LDUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            s.gpr[rd] = mem_read64(mem, ea);
            s.gpr[ra] = ea;
            break;
        }
        case XO_STDX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_write64(mem, ea, s.gpr[rd]);
            break;
        }
        case XO_STDUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            mem_write64(mem, ea, s.gpr[rd]);
            s.gpr[ra] = ea;
            break;
        }
        case XO_LWAX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = (uint64_t)(int64_t)(int32_t)mem_read32(mem, ea);
            break;
        }
        case XO_LWAUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            s.gpr[rd] = (uint64_t)(int64_t)(int32_t)mem_read32(mem, ea);
            s.gpr[ra] = ea;
            break;
        }

        // ── Indexed FP Load/Store ──────────────────────────────
        case XO_LFSX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.fpr[rd] = (double)mem_readf32(mem, ea);
            break;
        }
        case XO_LFSUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            s.fpr[rd] = (double)mem_readf32(mem, ea);
            s.gpr[ra] = ea;
            break;
        }
        case XO_LFDX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.fpr[rd] = mem_readf64(mem, ea);
            break;
        }
        case XO_LFDUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            s.fpr[rd] = mem_readf64(mem, ea);
            s.gpr[ra] = ea;
            break;
        }
        case XO_STFSX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_writef32(mem, ea, (float)s.fpr[rd]);
            break;
        }
        case XO_STFSUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            mem_writef32(mem, ea, (float)s.fpr[rd]);
            s.gpr[ra] = ea;
            break;
        }
        case XO_STFDX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_writef64(mem, ea, s.fpr[rd]);
            break;
        }
        case XO_STFDUX: {
            uint64_t ea = s.gpr[ra] + s.gpr[rb];
            mem_writef64(mem, ea, s.fpr[rd]);
            s.gpr[ra] = ea;
            break;
        }
        case XO_STFIWX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            uint64_t bits;
            memcpy(&bits, &s.fpr[rd], 8);
            mem_write32(mem, ea, (uint32_t)bits);
            break;
        }

        // ── Byte-Reversed Load/Store ───────────────────────────
        case XO_LWBRX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            uint32_t raw;
            memcpy(&raw, mem + (ea & (PS3_SANDBOX_SIZE - 1)), 4);
            s.gpr[rd] = raw; // no swap — byte-reversed from BE = native LE
            break;
        }
        case XO_STWBRX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            uint32_t val = (uint32_t)s.gpr[rd];
            memcpy(mem + (ea & (PS3_SANDBOX_SIZE - 1)), &val, 4); // store as LE
            break;
        }
        case XO_LHBRX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            uint16_t raw;
            memcpy(&raw, mem + (ea & (PS3_SANDBOX_SIZE - 1)), 2);
            s.gpr[rd] = raw;
            break;
        }
        case XO_STHBRX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            uint16_t val = (uint16_t)s.gpr[rd];
            memcpy(mem + (ea & (PS3_SANDBOX_SIZE - 1)), &val, 2);
            break;
        }

        // ── Atomic Load/Store (single-thread: simplified) ──────
        case XO_LWARX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = mem_read32(mem, ea);
            break;
        }
        case XO_STWCX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_write32(mem, ea, (uint32_t)s.gpr[rd]);
            // Always succeed in single-thread: set CR0 = EQ
            uint32_t cr0 = 0x2; // EQ
            if (s.xer & (1ULL << 31)) cr0 |= 0x1; // SO
            s.cr = (s.cr & 0x0FFFFFFF) | (cr0 << 28);
            break;
        }
        case XO_LDARX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            s.gpr[rd] = mem_read64(mem, ea);
            break;
        }
        case XO_STDCX: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            mem_write64(mem, ea, s.gpr[rd]);
            uint32_t cr0 = 0x2;
            if (s.xer & (1ULL << 31)) cr0 |= 0x1;
            s.cr = (s.cr & 0x0FFFFFFF) | (cr0 << 28);
            break;
        }

        // ── SPR access ─────────────────────────────────────────
        case XO_MFSPR: {
            uint32_t spr = SPR(inst);
            switch (spr) {
            case SPR_LR:   s.gpr[rd] = s.lr; break;
            case SPR_CTR:  s.gpr[rd] = s.ctr; break;
            case SPR_XER:  s.gpr[rd] = s.xer; break;
            case SPR_TBL:  s.gpr[rd] = s.tbl; break;
            case SPR_TBU:  s.gpr[rd] = s.tbu; break;
            case SPR_DEC:  s.gpr[rd] = s.dec; break;
            case SPR_SRR0: s.gpr[rd] = s.srr0; break;
            case SPR_SRR1: s.gpr[rd] = s.srr1; break;
            case SPR_PVR:  s.gpr[rd] = 0x00700000; break;
            default:
                if (spr >= SPR_SPRG0 && spr <= SPR_SPRG3)
                    s.gpr[rd] = s.sprg[spr - SPR_SPRG0];
                else
                    s.gpr[rd] = 0;
                break;
            }
            break;
        }
        case XO_MTSPR: {
            uint32_t spr = SPR(inst);
            switch (spr) {
            case SPR_LR:   s.lr  = s.gpr[rd]; break;
            case SPR_CTR:  s.ctr = s.gpr[rd]; break;
            case SPR_XER:  s.xer = s.gpr[rd]; break;
            case SPR_DEC:  s.dec = s.gpr[rd]; break;
            case SPR_SRR0: s.srr0 = s.gpr[rd]; break;
            case SPR_SRR1: s.srr1 = s.gpr[rd]; break;
            default:
                if (spr >= SPR_SPRG0 && spr <= SPR_SPRG3)
                    s.sprg[spr - SPR_SPRG0] = s.gpr[rd];
                break;
            }
            break;
        }
        case XO_MFTB: {
            uint32_t tbr = SPR(inst);
            switch (tbr) {
            case 268: s.gpr[rd] = s.tbl; break;
            case 269: s.gpr[rd] = s.tbu; break;
            default:  s.gpr[rd] = 0; break;
            }
            break;
        }
        case XO_MFCR:
            s.gpr[rd] = s.cr;
            break;

        case XO_MTCRF: {
            uint32_t crm = CRM(inst);
            uint32_t val = (uint32_t)s.gpr[rd];
            for (int i = 0; i < 8; i++) {
                if (crm & (1 << (7 - i))) {
                    int shift = (7 - i) * 4;
                    s.cr = (s.cr & ~(0xFU << shift)) | (val & (0xFU << shift));
                }
            }
            break;
        }

        case XO_MFMSR:
            s.gpr[rd] = s.msr;
            break;

        case XO_MTMSR:
            s.msr = s.gpr[rd];
            break;

        // ── Trap ───────────────────────────────────────────────
        case XO_TW: {
            int32_t a = (int32_t)(uint32_t)s.gpr[ra];
            int32_t b = (int32_t)(uint32_t)s.gpr[rb];
            uint32_t to = rd;
            bool trap = ((to & 0x10) && a < b) || ((to & 0x08) && a > b) ||
                        ((to & 0x04) && a == b) ||
                        ((to & 0x02) && (uint32_t)a < (uint32_t)b) ||
                        ((to & 0x01) && (uint32_t)a > (uint32_t)b);
            if (trap) s.halted = 1;
            break;
        }
        case XO_TD: {
            int64_t a = (int64_t)s.gpr[ra];
            int64_t b = (int64_t)s.gpr[rb];
            uint32_t to = rd;
            bool trap = ((to & 0x10) && a < b) || ((to & 0x08) && a > b) ||
                        ((to & 0x04) && a == b) ||
                        ((to & 0x02) && (uint64_t)a < (uint64_t)b) ||
                        ((to & 0x01) && (uint64_t)a > (uint64_t)b);
            if (trap) s.halted = 1;
            break;
        }

        // ── Cache hints (NOP on GPU) ───────────────────────────
        case XO_DCBF:
        case XO_DCBST:
        case XO_DCBT:
        case XO_DCBTST:
        case XO_ICBI:
            break;
        case XO_DCBZ: {
            uint64_t ea = ((ra == 0) ? 0 : s.gpr[ra]) + s.gpr[rb];
            ea &= ~31ULL; // align to 32-byte cache line
            for (int i = 0; i < 32; i += 4)
                mem_write32(mem, ea + i, 0);
            break;
        }

        // ── Synchronization ────────────────────────────────────
        case XO_SYNC:
        case XO_EIEIO:
            __threadfence();
            break;

        default:
            return 2;
        }
        break;
    }

    // ─── Load/Store Multiple (used in function prologues) ─────

    case OP_LMW: {
        uint32_t rd_start = RD(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        for (uint32_t r = rd_start; r < 32; r++, ea += 4) {
            s.gpr[r] = mem_read32(mem, ea);
        }
        break;
    }

    case OP_STMW: {
        uint32_t rs_start = RS(inst), ra = RA(inst);
        uint64_t ea = SIMM16(inst) + ((ra == 0) ? 0 : s.gpr[ra]);
        for (uint32_t r = rs_start; r < 32; r++, ea += 4) {
            mem_write32(mem, ea, (uint32_t)s.gpr[r]);
        }
        break;
    }

    // ─── Trap Immediate ───────────────────────────────────────

    case OP_TWI: {
        uint32_t to = RD(inst), ra_field = RA(inst);
        int32_t a = (int32_t)(uint32_t)s.gpr[ra_field];
        int32_t b = (int32_t)(int16_t)(inst & 0xFFFF);
        bool trap = ((to & 0x10) && a < b) || ((to & 0x08) && a > b) ||
                    ((to & 0x04) && a == b) ||
                    ((to & 0x02) && (uint32_t)a < (uint32_t)b) ||
                    ((to & 0x01) && (uint32_t)a > (uint32_t)b);
        if (trap) s.halted = 1;
        break;
    }

    case OP_TDI: {
        uint32_t to = RD(inst), ra_field = RA(inst);
        int64_t a = (int64_t)s.gpr[ra_field];
        int64_t b = (int64_t)(int16_t)(inst & 0xFFFF);
        bool trap = ((to & 0x10) && a < b) || ((to & 0x08) && a > b) ||
                    ((to & 0x04) && a == b) ||
                    ((to & 0x02) && (uint64_t)a < (uint64_t)b) ||
                    ((to & 0x01) && (uint64_t)a > (uint64_t)b);
        if (trap) s.halted = 1;
        break;
    }

    // ─── Group 30 (64-bit Rotate/Shift) ───────────────────────

    case OP_GRP30: {
        uint32_t rs = RS(inst), ra_field = RA(inst);
        uint32_t xo4 = (inst >> 1) & 0xF;
        uint32_t sh = SH64(inst);
        uint32_t mb = MB64(inst);
        bool rc = inst & 1;

        switch (xo4) {
        case 0: case 1: { // RLDICL: rA = ROTL64(rS, sh) & MASK(mb, 63)
            sh = ((inst >> 11) & 0x1F) | ((xo4 & 1) << 5);
            s.gpr[ra_field] = rotl64(s.gpr[rs], sh) & mask64(mb, 63);
            if (rc) setCRField64(s.cr, 0, (int64_t)s.gpr[ra_field], s.xer);
            break;
        }
        case 2: case 3: { // RLDICR: rA = ROTL64(rS, sh) & MASK(0, me)
            sh = ((inst >> 11) & 0x1F) | ((xo4 & 1) << 5);
            uint32_t me = MB64(inst);
            s.gpr[ra_field] = rotl64(s.gpr[rs], sh) & mask64(0, me);
            if (rc) setCRField64(s.cr, 0, (int64_t)s.gpr[ra_field], s.xer);
            break;
        }
        case 4: case 5: { // RLDIC: rA = ROTL64(rS, sh) & MASK(mb, ~sh)
            sh = ((inst >> 11) & 0x1F) | ((xo4 & 1) << 5);
            uint32_t me = 63 - sh;
            s.gpr[ra_field] = rotl64(s.gpr[rs], sh) & mask64(mb, me);
            if (rc) setCRField64(s.cr, 0, (int64_t)s.gpr[ra_field], s.xer);
            break;
        }
        case 6: case 7: { // RLDIMI: rA = (ROTL64(rS, sh) & mask) | (rA & ~mask)
            sh = ((inst >> 11) & 0x1F) | ((xo4 & 1) << 5);
            uint32_t me = 63 - sh;
            uint64_t m = mask64(mb, me);
            s.gpr[ra_field] = (rotl64(s.gpr[rs], sh) & m) | (s.gpr[ra_field] & ~m);
            if (rc) setCRField64(s.cr, 0, (int64_t)s.gpr[ra_field], s.xer);
            break;
        }
        case 8: { // RLDCL: rA = ROTL64(rS, rB&63) & MASK(mb, 63)
            uint32_t rb_field = RB(inst);
            uint32_t n = (uint32_t)s.gpr[rb_field] & 63;
            s.gpr[ra_field] = rotl64(s.gpr[rs], n) & mask64(mb, 63);
            if (rc) setCRField64(s.cr, 0, (int64_t)s.gpr[ra_field], s.xer);
            break;
        }
        case 9: { // RLDCR: rA = ROTL64(rS, rB&63) & MASK(0, me)
            uint32_t rb_field = RB(inst);
            uint32_t n = (uint32_t)s.gpr[rb_field] & 63;
            uint32_t me = MB64(inst);
            s.gpr[ra_field] = rotl64(s.gpr[rs], n) & mask64(0, me);
            if (rc) setCRField64(s.cr, 0, (int64_t)s.gpr[ra_field], s.xer);
            break;
        }
        default:
            return 2;
        }
        break;
    }

    // ─── Group 58 (LD / LDU / LWA) ───────────────────────────

    case OP_GRP58: {
        uint32_t rd_f = RD(inst), ra_f = RA(inst);
        uint32_t ds_xo = inst & 3;
        int64_t ds = (int64_t)(int16_t)(inst & 0xFFFC); // sign-extend bits 16:29, already <<2 by masking
        uint64_t ea = ds + ((ra_f == 0) ? 0 : s.gpr[ra_f]);

        switch (ds_xo) {
        case 0: // LD
            s.gpr[rd_f] = mem_read64(mem, ea);
            break;
        case 1: // LDU
            s.gpr[rd_f] = mem_read64(mem, ea);
            s.gpr[ra_f] = ea;
            break;
        case 2: // LWA (load word algebraic)
            s.gpr[rd_f] = (uint64_t)(int64_t)(int32_t)mem_read32(mem, ea);
            break;
        default:
            return 2;
        }
        break;
    }

    // ─── Group 62 (STD / STDU) ────────────────────────────────

    case OP_GRP62: {
        uint32_t rs_f = RS(inst), ra_f = RA(inst);
        uint32_t ds_xo = inst & 3;
        int64_t ds = (int64_t)(int16_t)(inst & 0xFFFC);
        uint64_t ea = ds + ((ra_f == 0) ? 0 : s.gpr[ra_f]);

        switch (ds_xo) {
        case 0: // STD
            mem_write64(mem, ea, s.gpr[rs_f]);
            break;
        case 1: // STDU
            mem_write64(mem, ea, s.gpr[rs_f]);
            s.gpr[ra_f] = ea;
            break;
        default:
            return 2;
        }
        break;
    }

    // ─── Group 59 (Single-Precision FP) ───────────────────────

    case OP_GRP59: {
        uint32_t frd = RD(inst), fra = RA(inst), frb = RB(inst);
        uint32_t frc = FRC(inst);
        uint32_t xo5 = (inst >> 1) & 0x1F;
        bool rc = inst & 1;

        switch (xo5) {
        case 18: // FDIVS
            s.fpr[frd] = (double)((float)s.fpr[fra] / (float)s.fpr[frb]);
            break;
        case 20: // FSUBS
            s.fpr[frd] = (double)((float)s.fpr[fra] - (float)s.fpr[frb]);
            break;
        case 21: // FADDS
            s.fpr[frd] = (double)((float)s.fpr[fra] + (float)s.fpr[frb]);
            break;
        case 22: // FSQRTS
            s.fpr[frd] = (double)sqrtf((float)s.fpr[fra]);
            break;
        case 24: // FRES (reciprocal estimate, single)
            s.fpr[frd] = (double)(1.0f / (float)s.fpr[frb]);
            break;
        case 25: // FMULS (note: frC not frB)
            s.fpr[frd] = (double)((float)s.fpr[fra] * (float)s.fpr[frc]);
            break;
        case 28: // FMSUBS
            s.fpr[frd] = (double)((float)((float)s.fpr[fra] * (float)s.fpr[frc] - (float)s.fpr[frb]));
            break;
        case 29: // FMADDS
            s.fpr[frd] = (double)((float)((float)s.fpr[fra] * (float)s.fpr[frc] + (float)s.fpr[frb]));
            break;
        case 30: // FNMSUBS
            s.fpr[frd] = (double)(-(float)((float)s.fpr[fra] * (float)s.fpr[frc] - (float)s.fpr[frb]));
            break;
        case 31: // FNMADDS
            s.fpr[frd] = (double)(-(float)((float)s.fpr[fra] * (float)s.fpr[frc] + (float)s.fpr[frb]));
            break;
        default:
            return 2;
        }
        if (rc) setCRField64(s.cr, 1, (s.fpr[frd] < 0) ? -1 : (s.fpr[frd] > 0) ? 1 : 0, s.xer);
        break;
    }

    // ─── Group 63 (Double-Precision FP) ───────────────────────

    case OP_GRP63: {
        uint32_t frd = RD(inst), fra = RA(inst), frb = RB(inst);
        uint32_t frc = FRC(inst);
        uint32_t xo10 = XO_10(inst);
        uint32_t xo5 = (inst >> 1) & 0x1F;
        bool rc = inst & 1;
        bool handled = true;

        // Try X-form (10-bit XO) first
        switch (xo10) {
        case 0: { // FCMPU
            uint32_t bf = frd >> 2;
            double a = s.fpr[fra], b = s.fpr[frb];
            uint32_t val;
            if (isnan(a) || isnan(b))       val = 0x1; // FU (unordered)
            else if (a < b)                 val = 0x8; // FL
            else if (a > b)                 val = 0x4; // FG
            else                            val = 0x2; // FE
            int shift = (7 - bf) * 4;
            s.cr = (s.cr & ~(0xFU << shift)) | (val << shift);
            break;
        }
        case 32: { // FCMPO
            uint32_t bf = frd >> 2;
            double a = s.fpr[fra], b = s.fpr[frb];
            uint32_t val;
            if (isnan(a) || isnan(b))       val = 0x1;
            else if (a < b)                 val = 0x8;
            else if (a > b)                 val = 0x4;
            else                            val = 0x2;
            int shift = (7 - bf) * 4;
            s.cr = (s.cr & ~(0xFU << shift)) | (val << shift);
            break;
        }
        case 12: // FRSP (round to single precision)
            s.fpr[frd] = (double)(float)s.fpr[frb];
            break;
        case 14: { // FCTIW (convert to int word)
            int32_t ival = (int32_t)s.fpr[frb];
            uint64_t bits = (uint64_t)(uint32_t)ival;
            memcpy(&s.fpr[frd], &bits, 8);
            break;
        }
        case 15: { // FCTIWZ (convert to int word with truncation)
            int32_t ival = (int32_t)s.fpr[frb];
            uint64_t bits = (uint64_t)(uint32_t)ival;
            memcpy(&s.fpr[frd], &bits, 8);
            break;
        }
        case 40: // FNEG
            s.fpr[frd] = -s.fpr[frb];
            break;
        case 72: // FMR
            s.fpr[frd] = s.fpr[frb];
            break;
        case 136: // FNABS
            s.fpr[frd] = -fabs(s.fpr[frb]);
            break;
        case 264: // FABS
            s.fpr[frd] = fabs(s.fpr[frb]);
            break;
        case 38: { // MTFSB1 (set FPSCR bit)
            uint32_t bt = frd;
            if (bt < 32) s.fpscr |= (1U << (31 - bt));
            break;
        }
        case 70: { // MTFSB0 (clear FPSCR bit)
            uint32_t bt = frd;
            if (bt < 32) s.fpscr &= ~(1U << (31 - bt));
            break;
        }
        case 583: { // MFFS (move from FPSCR)
            uint64_t bits = (uint64_t)s.fpscr;
            memcpy(&s.fpr[frd], &bits, 8);
            break;
        }
        case 711: { // MTFSF (move to FPSCR fields)
            uint32_t fm = (inst >> 17) & 0xFF;
            uint64_t bits;
            memcpy(&bits, &s.fpr[frb], 8);
            uint32_t val = (uint32_t)bits;
            for (int i = 0; i < 8; i++) {
                if (fm & (1 << (7 - i))) {
                    int shift = (7 - i) * 4;
                    s.fpscr = (s.fpscr & ~(0xFU << shift)) | (val & (0xFU << shift));
                }
            }
            break;
        }
        case 814: { // FCTID (convert to int doubleword)
            int64_t ival = (int64_t)s.fpr[frb];
            uint64_t bits = (uint64_t)ival;
            memcpy(&s.fpr[frd], &bits, 8);
            break;
        }
        case 815: { // FCTIDZ (convert to int doubleword with truncation)
            int64_t ival = (int64_t)s.fpr[frb];
            uint64_t bits = (uint64_t)ival;
            memcpy(&s.fpr[frd], &bits, 8);
            break;
        }
        case 846: { // FCFID (convert from int doubleword)
            uint64_t bits;
            memcpy(&bits, &s.fpr[frb], 8);
            s.fpr[frd] = (double)(int64_t)bits;
            break;
        }
        default:
            // Try A-form (5-bit XO)
            handled = false;
            break;
        }

        if (!handled) {
            handled = true;
            switch (xo5) {
            case 18: // FDIV
                s.fpr[frd] = s.fpr[fra] / s.fpr[frb];
                break;
            case 20: // FSUB
                s.fpr[frd] = s.fpr[fra] - s.fpr[frb];
                break;
            case 21: // FADD
                s.fpr[frd] = s.fpr[fra] + s.fpr[frb];
                break;
            case 22: // FSQRT
                s.fpr[frd] = sqrt(s.fpr[fra]);
                break;
            case 23: // FSEL
                s.fpr[frd] = (s.fpr[fra] >= 0.0) ? s.fpr[frc] : s.fpr[frb];
                break;
            case 24: // FRES (reciprocal estimate)
                s.fpr[frd] = 1.0 / s.fpr[frb];
                break;
            case 25: // FMUL (note: frC not frB)
                s.fpr[frd] = s.fpr[fra] * s.fpr[frc];
                break;
            case 26: // FRSQRTE (reciprocal sqrt estimate)
                s.fpr[frd] = 1.0 / sqrt(s.fpr[frb]);
                break;
            case 28: // FMSUB
                s.fpr[frd] = s.fpr[fra] * s.fpr[frc] - s.fpr[frb];
                break;
            case 29: // FMADD
                s.fpr[frd] = s.fpr[fra] * s.fpr[frc] + s.fpr[frb];
                break;
            case 30: // FNMSUB
                s.fpr[frd] = -(s.fpr[fra] * s.fpr[frc] - s.fpr[frb]);
                break;
            case 31: // FNMADD
                s.fpr[frd] = -(s.fpr[fra] * s.fpr[frc] + s.fpr[frb]);
                break;
            default:
                return 2;
            }
        }
        if (rc && handled) {
            setCRField64(s.cr, 1, (s.fpr[frd] < 0) ? -1 : (s.fpr[frd] > 0) ? 1 : 0, s.xer);
        }
        break;
    }

    default:
        return 2; // unimplemented primary opcode
    }

    s.pc = s.npc;
    s.cycles++;
    s.tbl++;
    return 0;
}

// ═══════════════════════════════════════════════════════════════
// Persistent Megakernel — one thread = one PPE core
// ═══════════════════════════════════════════════════════════════

__global__ void ppeMegakernel(PPEState* states, uint8_t* mem,
                               uint32_t maxCycles,
                               uint32_t* hle_log,
                               volatile uint32_t* hle_signal,
                               volatile uint32_t* status) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1) return;  // Single PPE thread for now

    PPEState& s = states[tid];

    for (uint32_t cycle = 0; cycle < maxCycles && !s.halted; cycle++) {
        int result = execOne(s, mem, hle_log, hle_signal);
        if (result == 1) break; // halted
        if (result == 2) {
            // Unimplemented — log the PC and instruction, then skip
            uint32_t inst = mem_read32(mem, s.pc - 4); // already advanced
            if (hle_log) {
                uint32_t idx = atomicAdd(hle_log, 1);
                if (idx < 255) {
                    hle_log[1 + idx] = 0xDEAD0000 | OPCD(inst);
                }
            }
        }
    }

    // Signal completion
    if (status) atomicExch((uint32_t*)status, s.halted ? 2 : 1);
}

// ═══════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════

extern "C" {

struct MegakernelCtx {
    PPEState*          d_states;
    uint8_t*           d_mem;
    uint32_t*          d_hle_log;
    uint32_t*          d_hle_signal;
    uint32_t*          d_status;
    cudaStream_t       stream;
    bool               ready;
};

static MegakernelCtx g_ctx = {};

int megakernel_init() {
    if (g_ctx.ready) return 1;

    cudaStreamCreate(&g_ctx.stream);

    // Allocate PPE state (1 core for now)
    cudaMalloc(&g_ctx.d_states, sizeof(PPEState));
    cudaMemset(g_ctx.d_states, 0, sizeof(PPEState));

    // Allocate PS3 memory sandbox (512 MB)
    cudaMalloc(&g_ctx.d_mem, PS3_SANDBOX_SIZE);
    cudaMemset(g_ctx.d_mem, 0, PS3_SANDBOX_SIZE);

    // HLE logging buffer (1024 entries)
    cudaMalloc(&g_ctx.d_hle_log, 1024 * sizeof(uint32_t));
    cudaMemset(g_ctx.d_hle_log, 0, 1024 * sizeof(uint32_t));

    // Signal/status
    cudaMalloc(&g_ctx.d_hle_signal, sizeof(uint32_t));
    cudaMemset(g_ctx.d_hle_signal, 0, sizeof(uint32_t));
    cudaMalloc(&g_ctx.d_status, sizeof(uint32_t));
    cudaMemset(g_ctx.d_status, 0, sizeof(uint32_t));

    g_ctx.ready = true;
    fprintf(stderr, "[PPE] Megakernel initialized (512MB sandbox)\n");
    return 1;
}

// Load raw bytes into PS3 memory at given offset
int megakernel_load(uint64_t offset, const void* data, size_t size) {
    if (!g_ctx.ready || offset + size > PS3_SANDBOX_SIZE) return 0;
    cudaMemcpyAsync(g_ctx.d_mem + offset, data, size,
                     cudaMemcpyHostToDevice, g_ctx.stream);
    return 1;
}

// Read back a region of the device-side sandbox (mirror of megakernel_load).
// Useful for inspecting HLE side-effects (e.g. the GCM bridge ring).
int megakernel_read_mem(uint64_t offset, void* dst, size_t size) {
    if (!g_ctx.ready || offset + size > PS3_SANDBOX_SIZE) return 0;
    cudaStreamSynchronize(g_ctx.stream);
    cudaMemcpy(dst, g_ctx.d_mem + offset, size, cudaMemcpyDeviceToHost);
    return 1;
}

// Set initial PPE state (entry point, stack pointer, etc.)
int megakernel_set_entry(uint64_t pc, uint64_t sp, uint64_t toc) {
    if (!g_ctx.ready) return 0;
    PPEState init = {};
    init.pc = pc;
    init.gpr[1] = sp;          // Stack pointer
    init.gpr[2] = toc;         // Table of Contents (PPC64 ABI)
    init.gpr[3] = 0;           // argc
    init.gpr[13] = sp - 0x7000; // Small data area base (r13)
    init.msr = 0x8000000000000000ULL; // 64-bit mode
    cudaMemcpyAsync(g_ctx.d_states, &init, sizeof(PPEState),
                     cudaMemcpyHostToDevice, g_ctx.stream);
    fprintf(stderr, "[PPE] Entry: PC=0x%llx SP=0x%llx TOC=0x%llx\n",
            (unsigned long long)pc, (unsigned long long)sp, (unsigned long long)toc);
    return 1;
}

// Run N cycles of the megakernel
float megakernel_run(uint32_t maxCycles) {
    if (!g_ctx.ready) return -1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Reset status
    cudaMemsetAsync(g_ctx.d_status, 0, sizeof(uint32_t), g_ctx.stream);

    cudaEventRecord(start, g_ctx.stream);

    ppeMegakernel<<<1, 1, 0, g_ctx.stream>>>(
        g_ctx.d_states, g_ctx.d_mem, maxCycles,
        g_ctx.d_hle_log, g_ctx.d_hle_signal, g_ctx.d_status);

    cudaEventRecord(stop, g_ctx.stream);
    cudaStreamSynchronize(g_ctx.stream);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Readback PPE state for debugging
int megakernel_read_state(PPEState* out) {
    if (!g_ctx.ready || !out) return 0;
    cudaMemcpy(out, g_ctx.d_states, sizeof(PPEState), cudaMemcpyDeviceToHost);
    return 1;
}

// Push a PPE state back to the device (used by the host-side HLE
// dispatcher to simulate a `blr` and return value after handling an
// imported function call).
int megakernel_write_state(const PPEState* in) {
    if (!g_ctx.ready || !in) return 0;
    cudaMemcpy(g_ctx.d_states, in, sizeof(PPEState), cudaMemcpyHostToDevice);
    return 1;
}

// Write guest memory (mirror of megakernel_load for smaller regions).
int megakernel_write_mem(uint64_t offset, const void* src, size_t size) {
    if (!g_ctx.ready || offset + size > PS3_SANDBOX_SIZE) return 0;
    cudaMemcpy(g_ctx.d_mem + offset, src, size, cudaMemcpyHostToDevice);
    return 1;
}

// Readback HLE log
int megakernel_read_hle_log(uint32_t* out, int maxEntries) {
    if (!g_ctx.ready || !out) return 0;
    cudaMemcpy(out, g_ctx.d_hle_log, (1 + maxEntries) * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    return out[0]; // count
}

void megakernel_shutdown() {
    if (!g_ctx.ready) return;
    cudaFree(g_ctx.d_states);
    cudaFree(g_ctx.d_mem);
    cudaFree(g_ctx.d_hle_log);
    cudaFree(g_ctx.d_hle_signal);
    cudaFree(g_ctx.d_status);
    cudaStreamDestroy(g_ctx.stream);
    g_ctx.ready = false;
    fprintf(stderr, "[PPE] Megakernel shutdown\n");
}

} // extern "C"
