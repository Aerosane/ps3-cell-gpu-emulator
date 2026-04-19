// elf_loader.h — PS3 ELF64/SELF executable parser and loader
//
// Parses PPC64 big-endian ELF executables (and detects SELF wrappers) for the
// GPU-native Cell BE emulator.  Host-side only — runs on CPU to populate the
// memory buffer before GPU execution begins.
//
// Supports: PT_LOAD, PT_TLS, PT_SCE_PROCPARAM, SELF magic detection.
//
#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>

// ═══════════════════════════════════════════════════════════════
// Error codes
// ═══════════════════════════════════════════════════════════════

enum PS3ElfError : int {
    ELF_OK           =  0,
    ELF_ERR_MAGIC    = -1,
    ELF_ERR_CLASS    = -2,
    ELF_ERR_MACHINE  = -3,
    ELF_ERR_SEGMENT  = -4,
    ELF_ERR_SIZE     = -5,
};

// ═══════════════════════════════════════════════════════════════
// Big-endian byte-swap helpers (host / CPU)
// ═══════════════════════════════════════════════════════════════

static inline uint16_t elf_bswap16(uint16_t v) {
    return (uint16_t)((v >> 8) | (v << 8));
}

static inline uint32_t elf_bswap32(uint32_t v) {
    return ((v >> 24) & 0x000000FFu)
         | ((v >>  8) & 0x0000FF00u)
         | ((v <<  8) & 0x00FF0000u)
         | ((v << 24) & 0xFF000000u);
}

static inline uint64_t elf_bswap64(uint64_t v) {
    return ((uint64_t)elf_bswap32((uint32_t)v) << 32)
         | (uint64_t)elf_bswap32((uint32_t)(v >> 32));
}

// ═══════════════════════════════════════════════════════════════
// ELF64 structures (on-disk, big-endian fields)
// ═══════════════════════════════════════════════════════════════

static constexpr uint32_t ELF_MAGIC = 0x7F454C46u; // "\177ELF"

// ELF identification indices
enum {
    EI_MAG0       = 0,
    EI_MAG1       = 1,
    EI_MAG2       = 2,
    EI_MAG3       = 3,
    EI_CLASS      = 4,
    EI_DATA       = 5,
    EI_VERSION    = 6,
    EI_OSABI      = 7,
    EI_ABIVERSION = 8,
    EI_NIDENT     = 16,
};

struct Elf64_Ehdr {
    uint8_t  e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};

struct Elf64_Phdr {
    uint32_t p_type;
    uint32_t p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
};

struct Elf64_Shdr {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
};

// Program header types
enum : uint32_t {
    PT_NULL            = 0x00000000u,
    PT_LOAD            = 0x00000001u,
    PT_TLS             = 0x00000007u,
    PT_SCE_PROCPARAM   = 0x60000001u,
    PT_SCE_PPURELEXEC  = 0x60000002u,
};

// ELF type values
enum : uint16_t {
    ET_EXEC = 0x0002u,
    ET_PRX  = 0xFFA4u,
};

// ═══════════════════════════════════════════════════════════════
// SELF / SCE header (Sony encrypted wrapper)
// ═══════════════════════════════════════════════════════════════

static constexpr uint32_t SCE_MAGIC = 0x53434500u; // "SCE\0"

struct SceHeader {
    uint32_t magic;
    uint32_t version;
    uint16_t key_revision;
    uint16_t header_type;
    uint32_t metadata_offset;
    uint64_t header_len;
    uint64_t data_len;
};

// SELF v2 extended header (all fields big-endian u64). Starts at file
// byte 0x20, immediately after the SceHeader. Layout matches RPCS3
// `self_hdr`.
struct SelfHeaderV2 {
    uint64_t header_type;          // 3 for SELF
    uint64_t info_offset;          // app_info table offset
    uint64_t elf_offset;           // embedded ELF header offset
    uint64_t phdr_offset;          // embedded program-header-table offset
    uint64_t shdr_offset;          // embedded section-header-table offset
    uint64_t section_info_offset;  // per-PHDR segment info table offset
    uint64_t sce_version_offset;
    uint64_t control_info_offset;
    uint64_t control_info_size;
    uint64_t padding;
};

// One entry per ELF PHDR in the SELF.  Tells us where each segment's
// actual bytes live in the SELF file, and whether they're compressed
// or encrypted.
struct SelfSectionInfo {
    uint64_t offset;       // byte offset in SELF file where segment data lives
    uint64_t size;         // size of segment data in file (possibly compressed)
    uint32_t compressed;   // 1 = raw, 2 = zlib
    uint32_t unknown1;
    uint32_t encrypted;    // 1 = yes, 2 = no
    uint32_t unknown2;
};

// Legacy/unused — kept for source-compat with any external caller.
struct ExtHeader {
    uint64_t authid;
    uint64_t vendor_id;
    uint64_t self_type;
    uint64_t version;
    uint64_t padding;
    uint32_t app_info_offset;
    uint32_t elf_offset;       // offset to embedded ELF within the SELF
    uint32_t phdr_offset;
    uint32_t shdr_offset;
    uint32_t section_info_offset;
    uint32_t sceversion_offset;
    uint32_t controlinfo_offset;
    uint32_t controlinfo_size;
    uint64_t padding2;
};

// ═══════════════════════════════════════════════════════════════
// PS3 process_param_t (found at PT_SCE_PROCPARAM)
// ═══════════════════════════════════════════════════════════════

static constexpr uint32_t PROC_PARAM_MAGIC = 0x13BCC5F6u;

struct PS3ProcessParam {
    uint32_t size;
    uint32_t magic;
    uint32_t version;
    uint32_t sdk_version;
    int32_t  primary_prio;
    uint32_t primary_stacksize;
    uint32_t malloc_pagesize;
    uint32_t ppc_seg;
};

// ═══════════════════════════════════════════════════════════════
// Loader output — information extracted from the ELF
// ═══════════════════════════════════════════════════════════════

static constexpr int PS3_MAX_SEGMENTS = 32;

struct PS3Segment {
    uint64_t addr;
    uint64_t size;
    uint64_t file_offset;
    uint32_t flags;
    uint32_t type;
};

struct PS3TlsInfo {
    uint64_t addr;
    uint64_t filesz;
    uint64_t memsz;
    uint64_t align;
};

struct PS3ExeInfo {
    uint64_t    entry_point;
    PS3Segment  segments[PS3_MAX_SEGMENTS];
    int         num_segments;

    uint32_t    sdk_version;
    uint32_t    primary_stacksize;
    int32_t     primary_prio;

    bool        is_prx;
    bool        has_tls;
    PS3TlsInfo  tls;

    bool        has_proc_param;
    char        module_name[64];
};

// ═══════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════

/// Read a big-endian uint32 from a raw byte pointer.
static inline uint32_t read_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] <<  8) | (uint32_t)p[3];
}

/// Try to locate the embedded ELF inside a SELF container.
/// Returns the byte offset to the ELF header, or 0 on failure.
static inline uint64_t self_find_elf_offset(const uint8_t* data, size_t size) {
    if (size < sizeof(SceHeader) + sizeof(ExtHeader))
        return 0;

    // The SCE header is followed by the extended header which carries the
    // elf_offset field at a known position.
    const SceHeader* sce = reinterpret_cast<const SceHeader*>(data);
    uint64_t hdr_len = elf_bswap64(sce->header_len);

    // The extended header immediately follows the SCE header.
    if (size < sizeof(SceHeader) + sizeof(ExtHeader))
        return 0;

    const ExtHeader* ext = reinterpret_cast<const ExtHeader*>(data + sizeof(SceHeader));
    uint32_t elf_off = elf_bswap32(ext->elf_offset);

    if (elf_off == 0 || elf_off >= size)
        return 0;

    // Quick sanity: check ELF magic at the reported offset
    if (elf_off + 4 <= size && read_be32(data + elf_off) == ELF_MAGIC)
        return elf_off;

    // Fallback: scan for ELF magic within the header region
    uint64_t scan_end = (hdr_len && hdr_len < size) ? hdr_len : (size < 4096 ? size : 4096);
    for (uint64_t off = 4; off + 4 <= scan_end; off += 4) {
        if (read_be32(data + off) == ELF_MAGIC)
            return off;
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════
// SELF segment staging — uses section_info table to resolve real
// file offsets (and decompresses zlib segments when needed).
// Returns ELF_OK on full success. If any segment is encrypted we
// return ELF_OK but mark that segment as zero-filled (caller sees
// empty instruction stream at that vaddr).
// ═══════════════════════════════════════════════════════════════
#include <zlib.h>
static inline int self_stage_segments(const uint8_t* file_data, size_t file_size,
                                       const uint8_t* elf_data, size_t elf_size,
                                       uint8_t* memory, size_t mem_size)
{
    (void)elf_size;
    if (file_size < 0x70) return ELF_ERR_SIZE;

    const SelfHeaderV2* sh = reinterpret_cast<const SelfHeaderV2*>(file_data + 0x20);
    uint64_t secOff = elf_bswap64(sh->section_info_offset);
    uint64_t elfOff = elf_bswap64(sh->elf_offset);
    if (secOff == 0 || secOff >= file_size) return ELF_OK;  // no section table
    if (elfOff == 0 || elfOff >= file_size) return ELF_OK;

    // Reparse PHDRs from inner ELF.
    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(file_data + elfOff);
    uint64_t phoff    = elf_bswap64(ehdr->e_phoff);
    uint16_t phnum    = elf_bswap16(ehdr->e_phnum);
    uint16_t phentsz  = elf_bswap16(ehdr->e_phentsize);
    if (phnum == 0) return ELF_OK;
    if (elfOff + phoff + (uint64_t)phnum * phentsz > file_size) return ELF_ERR_SIZE;
    if (secOff + (uint64_t)phnum * sizeof(SelfSectionInfo) > file_size) return ELF_ERR_SIZE;

    const SelfSectionInfo* sec =
        reinterpret_cast<const SelfSectionInfo*>(file_data + secOff);

    for (uint16_t i = 0; i < phnum; ++i) {
        const Elf64_Phdr* ph = reinterpret_cast<const Elf64_Phdr*>(
            file_data + elfOff + phoff + (uint64_t)i * phentsz);
        uint32_t p_type   = elf_bswap32(ph->p_type);
        if (p_type != PT_LOAD && p_type != PT_SCE_PROCPARAM) continue;
        uint64_t p_vaddr  = elf_bswap64(ph->p_vaddr);
        uint64_t p_memsz  = elf_bswap64(ph->p_memsz);
        uint64_t p_filesz = elf_bswap64(ph->p_filesz);

        uint64_t secFileOff = elf_bswap64(sec[i].offset);
        uint64_t secSize    = elf_bswap64(sec[i].size);
        uint32_t compressed = elf_bswap32(sec[i].compressed);
        uint32_t encrypted  = elf_bswap32(sec[i].encrypted);

        if (p_vaddr + p_memsz > mem_size) return ELF_ERR_SEGMENT;
        if (secFileOff + secSize > file_size) continue;

        // Zero-fill memory region first (covers BSS gap too).
        memset(memory + p_vaddr, 0, (size_t)p_memsz);

        if (encrypted == 1) {
            // Segment is AES-encrypted — we don't have the key path.
            continue;
        }

        if (compressed == 2) {
            // zlib-inflate into memory[p_vaddr..p_vaddr+p_filesz].
            z_stream zs{};
            zs.next_in  = const_cast<Bytef*>(file_data + secFileOff);
            zs.avail_in = (uInt)secSize;
            zs.next_out = memory + p_vaddr;
            zs.avail_out = (uInt)p_memsz;
            if (inflateInit(&zs) != Z_OK) continue;
            int r = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            if (r != Z_STREAM_END && r != Z_OK) {
                // decompression failed — leave zeros.
                memset(memory + p_vaddr, 0, (size_t)p_memsz);
            }
        } else {
            // Raw copy. Use min(secSize, p_filesz) for bounds.
            uint64_t copySize = secSize < p_filesz ? secSize : p_filesz;
            if (copySize > 0)
                memcpy(memory + p_vaddr, file_data + secFileOff, (size_t)copySize);
        }
    }
    return ELF_OK;
}

// ═══════════════════════════════════════════════════════════════
// Public API — ps3_load_elf
// ═══════════════════════════════════════════════════════════════

/// Load a PS3 ELF64 (or SELF-wrapped ELF) into the supplied memory buffer.
///
/// @param file_data  Raw file contents (ELF or SELF).
/// @param file_size  Size of file_data in bytes.
/// @param memory     Target memory buffer (emulated main memory).
/// @param mem_size   Size of memory buffer in bytes.
/// @param info       [out] Parsed executable information.
/// @return           ELF_OK on success, negative error code on failure.
static int ps3_load_elf(const uint8_t* file_data, size_t file_size,
                        uint8_t* memory, size_t mem_size,
                        PS3ExeInfo* info)
{
    if (!file_data || !memory || !info)
        return ELF_ERR_MAGIC;

    memset(info, 0, sizeof(PS3ExeInfo));

    // ── SELF detection ──────────────────────────────────────
    const uint8_t* elf_data = file_data;
    size_t elf_size = file_size;
    bool is_self = false;

    if (file_size >= 4 && read_be32(file_data) == SCE_MAGIC) {
        uint64_t elf_off = self_find_elf_offset(file_data, file_size);
        if (elf_off == 0)
            return ELF_ERR_MAGIC; // couldn't locate embedded ELF
        elf_data = file_data + elf_off;
        elf_size = file_size - elf_off;
        is_self = true;
    }

    // ── ELF header validation ───────────────────────────────
    if (elf_size < sizeof(Elf64_Ehdr))
        return ELF_ERR_SIZE;

    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(elf_data);

    // Magic check
    if (ehdr->e_ident[EI_MAG0] != 0x7F ||
        ehdr->e_ident[EI_MAG1] != 'E'  ||
        ehdr->e_ident[EI_MAG2] != 'L'  ||
        ehdr->e_ident[EI_MAG3] != 'F')
        return ELF_ERR_MAGIC;

    // Must be 64-bit
    if (ehdr->e_ident[EI_CLASS] != 2)
        return ELF_ERR_CLASS;

    // Must be big-endian
    if (ehdr->e_ident[EI_DATA] != 2)
        return ELF_ERR_CLASS;

    // Machine must be PPC64 (0x15 = 21)
    if (elf_bswap16(ehdr->e_machine) != 0x15)
        return ELF_ERR_MACHINE;

    uint16_t e_type = elf_bswap16(ehdr->e_type);
    info->is_prx = (e_type == ET_PRX);
    info->entry_point = elf_bswap64(ehdr->e_entry);

    // ── Program headers ─────────────────────────────────────
    uint64_t phoff    = elf_bswap64(ehdr->e_phoff);
    uint16_t phnum    = elf_bswap16(ehdr->e_phnum);
    uint16_t phentsize = elf_bswap16(ehdr->e_phentsize);

    if (phoff == 0 || phnum == 0)
        return ELF_OK; // valid ELF, just no segments

    if (phoff + (uint64_t)phnum * phentsize > elf_size)
        return ELF_ERR_SIZE;

    for (uint16_t i = 0; i < phnum; i++) {
        const Elf64_Phdr* phdr = reinterpret_cast<const Elf64_Phdr*>(
            elf_data + phoff + (uint64_t)i * phentsize);

        uint32_t p_type  = elf_bswap32(phdr->p_type);
        uint32_t p_flags = elf_bswap32(phdr->p_flags);
        uint64_t p_offset = elf_bswap64(phdr->p_offset);
        uint64_t p_vaddr  = elf_bswap64(phdr->p_vaddr);
        uint64_t p_filesz = elf_bswap64(phdr->p_filesz);
        uint64_t p_memsz  = elf_bswap64(phdr->p_memsz);
        uint64_t p_align  = elf_bswap64(phdr->p_align);

        switch (p_type) {

        case PT_LOAD: {
            // Bounds check against target memory
            if (p_vaddr + p_memsz > mem_size)
                return ELF_ERR_SEGMENT;

            // Bounds check against source file
            if (p_offset + p_filesz > elf_size)
                return ELF_ERR_SIZE;

            // Copy file-backed portion
            if (p_filesz > 0)
                memcpy(memory + p_vaddr, elf_data + p_offset, (size_t)p_filesz);

            // Zero-fill BSS region (memsz > filesz)
            if (p_memsz > p_filesz)
                memset(memory + p_vaddr + p_filesz, 0, (size_t)(p_memsz - p_filesz));

            // Record segment info
            if (info->num_segments < PS3_MAX_SEGMENTS) {
                PS3Segment& seg = info->segments[info->num_segments++];
                seg.addr        = p_vaddr;
                seg.size        = p_memsz;
                seg.file_offset = p_offset;
                seg.flags       = p_flags;
                seg.type        = PT_LOAD;
            }
            break;
        }

        case PT_TLS: {
            info->has_tls    = true;
            info->tls.addr   = p_vaddr;
            info->tls.filesz = p_filesz;
            info->tls.memsz  = p_memsz;
            info->tls.align  = p_align;
            break;
        }

        case PT_SCE_PROCPARAM: {
            if (p_offset + sizeof(PS3ProcessParam) <= elf_size) {
                const PS3ProcessParam* pp = reinterpret_cast<const PS3ProcessParam*>(
                    elf_data + p_offset);

                if (elf_bswap32(pp->magic) == PROC_PARAM_MAGIC) {
                    info->has_proc_param    = true;
                    info->sdk_version       = elf_bswap32(pp->sdk_version);
                    info->primary_stacksize = elf_bswap32(pp->primary_stacksize);
                    info->primary_prio      = (int32_t)elf_bswap32((uint32_t)pp->primary_prio);
                }
            }

            // Also load into memory like PT_LOAD if within bounds
            if (p_vaddr + p_memsz <= mem_size && p_offset + p_filesz <= elf_size) {
                if (p_filesz > 0)
                    memcpy(memory + p_vaddr, elf_data + p_offset, (size_t)p_filesz);
                if (p_memsz > p_filesz)
                    memset(memory + p_vaddr + p_filesz, 0, (size_t)(p_memsz - p_filesz));
            }
            break;
        }

        default:
            break;
        }
    }

    // For SELF files, the ELF-header p_offset values are not authoritative —
    // segment payload lives at positions recorded in the SelfSectionInfo
    // table (and may be zlib-compressed). Overlay correct data now.
    if (is_self) {
        self_stage_segments(file_data, file_size, elf_data, elf_size,
                            memory, mem_size);
    }

    return ELF_OK;
}

/// Convenience: print parsed ELF info to stdout.
static void ps3_print_elf_info(const PS3ExeInfo* info) {
    printf("  Entry point:  0x%08llX\n", (unsigned long long)info->entry_point);
    printf("  Type:         %s\n", info->is_prx ? "PRX (module)" : "Executable");
    printf("  Segments:     %d\n", info->num_segments);
    for (int i = 0; i < info->num_segments; i++) {
        const PS3Segment& s = info->segments[i];
        printf("    [%d] addr=0x%08llX  size=0x%llX  flags=0x%X\n",
               i, (unsigned long long)s.addr, (unsigned long long)s.size, s.flags);
    }
    if (info->has_proc_param) {
        printf("  SDK version:  0x%08X\n", info->sdk_version);
        printf("  Stack size:   0x%X\n", info->primary_stacksize);
        printf("  Priority:     %d\n", info->primary_prio);
    }
    if (info->has_tls) {
        printf("  TLS:          addr=0x%llX  memsz=0x%llX\n",
               (unsigned long long)info->tls.addr,
               (unsigned long long)info->tls.memsz);
    }
}
