// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}

/// A draw that follows a draw of the same pipeline with only light state changes in between takes
/// the delta path. It keeps the previous draw's pipeline, image views, samplers and texture buffer
/// bindings, and resolves and binds every buffer (index, vertex, uniform, storage, texture) again
/// as a full resolve does, so guest memory written without a notification is streamed as in a
/// full resolve.
/// Light changes are draw parameters, index buffer and vertex stream registers, constant buffer
/// binds and inline constant buffer data (collected in g_delta: the rasterizer resolves in full
/// when they reach the pipeline's texture handles at their GPU addresses or through another
/// mapping of their device memory). Anything else that could change what a resolve produces bumps
/// a serial: a 3D register that changes, any 3D method with side effects (uploads, reports,
/// clears, barriers, ...), constant buffer data reaching the texture or sampler headers (the same
/// way), any other engine, deferred calls on the split's back (except the ones marked
/// GpuSplit::FastDrawNeutral), GPU::TickWork requests, texture cache writes, unmaps and
/// deletions, shader removals, fence releases, HLE macros and frames. The rasterizer takes the
/// delta path only while the serial, the GPU memory map, the command buffer, the render pass and
/// the bound pipeline are the ones the previous draw left.
/// EDEN_NX2_FAST_DRAW_VERIFY=1 resolves those draws in full and counts where the pipeline, views,
/// samplers or kept buffer bindings differ.
namespace VideoCommon::Nx2FastDraw {

enum class Reason : u8 {
    Register,
    Method,
    ConstantData,
    InlineData,
    OtherEngine,
    Call,
    TickWork,
    Invalidation,
    Fence,
    IndirectReset,
    Frame,
    TextureHeaders,
    DeltaOverflow,
    Map,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "register",      "method",    "constant buffer data", "inline data",
    "other engine",  "deferred call", "tick work",        "invalidation",
    "fence release", "indirect reset", "frame",           "texture header data",
    "delta overflow", "memory map",
};

inline std::atomic<bool> g_enabled{true};

/// Any thread: whether fast draws run (always, unless a test turned them off). The rasterizer
/// latches it per draw (on the thread that draws) and starts with a full resolve when it turns on.
[[nodiscard]] inline bool Enabled() {
    return g_enabled.load(std::memory_order_relaxed);
}

/// Tests only.
inline void SetEnabledForTest(bool on) {
    g_enabled.store(on, std::memory_order_relaxed);
}

/// EDEN_NX2_FAST_DRAW_VERIFY=1, read once.
[[nodiscard]] inline bool VerifyEnabled() {
    static const bool verify = [] {
        const char* const value = std::getenv("EDEN_NX2_FAST_DRAW_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

inline std::atomic<u64> g_serial{1};
/// The last break's reason and, for Register and Method, its 3D method (diagnostics).
inline std::atomic<u32> g_last_break{};

inline void Break(Reason reason, u32 method = 0) {
    if (Enabled()) {
        g_serial.fetch_add(1, std::memory_order_relaxed);
        g_last_break.store(static_cast<u32>(reason) | (method << 8), std::memory_order_relaxed);
    }
}

inline u64 Serial() {
    return g_serial.load(std::memory_order_relaxed);
}

/// The device memory a GPU range maps to. A write through another GPU address that maps the
/// same device memory changes the range too.
class DeviceSpans {
public:
    /// Translates [address, address + size) again unless that range of memory was translated last
    /// and memory's mappings haven't changed since.
    void Update(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

    /// Whether [begin, end) of device memory overlaps the range's.
    [[nodiscard]] bool Intersects(DAddr begin, DAddr end) const;

private:
    const Tegra::MemoryManager* memory{};
    GPUVAddr address{};
    u64 size{};
    u64 map_serial{};
    std::vector<std::pair<DAddr, DAddr>> spans; ///< Sorted, disjoint [begin, end) pairs
};

/// Light changes since the last draw that can reach texture handles. Written and read on the
/// thread that runs the 3D engine and the rasterizer (the split's back).
struct Delta {
    static constexpr size_t MaxWrites = 16;

    std::array<u32, 5> binds{}; ///< Constant buffer slots bound per stage
    /// Constant buffer data written, by GPU address (unmapped addresses included)...
    std::array<GPUVAddr, MaxWrites> write_begin{};
    std::array<GPUVAddr, MaxWrites> write_end{};
    u32 num_writes{};
    /// ...and by the device address it landed on.
    std::array<DAddr, MaxWrites> device_begin{};
    std::array<DAddr, MaxWrites> device_end{};
    u32 num_device_writes{};
    /// A write was not recorded: the list is full, or a texture/sampler header write returned
    /// before noting the range. Uniform keeps must not trust this delta.
    bool overflowed{};

    [[nodiscard]] bool Empty() const {
        return !overflowed && num_writes == 0 &&
               (binds[0] | binds[1] | binds[2] | binds[3] | binds[4]) == 0;
    }

    /// Whether data written reaches [address, address + size), whose device memory is spans: at
    /// one of its GPU addresses, or through any GPU address that maps its device memory.
    [[nodiscard]] bool Reaches(GPUVAddr address, u64 size, const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            if (write_begin[write] < address + size && address < write_end[write]) {
                return true;
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (spans.Intersects(device_begin[write], device_end[write])) {
                return true;
            }
        }
        return false;
    }
};

inline Delta g_delta;

/// Appends [begin, end) to a delta's range list, or extends its last range when that ends at
/// begin. False when the list is full.
inline bool AppendRange(std::array<u64, Delta::MaxWrites>& begins,
                        std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (count != 0 && ends[count - 1] == begin) {
        ends[count - 1] = end;
        return true;
    }
    if (count == Delta::MaxWrites) {
        return false;
    }
    begins[count] = begin;
    ends[count] = end;
    ++count;
    return true;
}

inline void NoteBind(size_t stage, u32 slot) {
    if (Enabled()) {
        g_delta.binds[stage] |= 1U << slot;
    }
}

/// Inline constant buffer data written to GPU addresses [begin, end).
inline void NoteConstantData(GPUVAddr begin, GPUVAddr end) {
    if (!Enabled()) {
        return;
    }
    Delta& delta = g_delta;
    if (!AppendRange(delta.write_begin, delta.write_end, delta.num_writes, begin, end)) {
        delta.overflowed = true;
        Break(Reason::DeltaOverflow);
    }
}

/// The device addresses [begin, end) that inline constant buffer data landed on.
inline void NoteDeviceData(DAddr begin, DAddr end) {
    if (!Enabled()) {
        return;
    }
    Delta& delta = g_delta;
    if (!AppendRange(delta.device_begin, delta.device_end, delta.num_device_writes, begin, end)) {
        delta.overflowed = true;
        Break(Reason::DeltaOverflow);
    }
}

/// Macro calls that broke a chain, and constant-buffer selector calls that were kept. The GPU
/// back thread writes this; the fast-draw report reads and clears it.
struct MacroNote {
    u32 method{};
    u32 count{};
    const char* kind{};
    bool kept{};
};

inline std::array<MacroNote, 16> g_macro_notes{};
inline u64 g_macro_note_overflow{};
inline u64 g_rt_clean_skips{};
inline u64 g_storage_reused{};
/// Payload fence releases written when the back reaches them, and those still held for the host
/// GPU. The fencing decision is on the GPU thread; the fast-draw report reads these.
inline std::atomic<u64> g_payload_early{};
inline std::atomic<u64> g_payload_held{};
/// FlushCaching calls with an empty accumulator, and calls that translated ranges.
inline std::atomic<u64> g_inval_empty{};
inline std::atomic<u64> g_inval_ran{};
/// RasterizerMarkRegionCached ranges whose protection was already the target, and ranges
/// reprotected.
inline std::atomic<u64> g_protect_skipped{};
inline std::atomic<u64> g_protect_ran{};

inline void NoteMacro(u32 method, const char* kind, bool kept) {
    for (MacroNote& note : g_macro_notes) {
        if (note.count != 0 && note.method == method && note.kind == kind && note.kept == kept) {
            ++note.count;
            return;
        }
        if (note.count == 0) {
            note = MacroNote{method, 1, kind, kept};
            return;
        }
    }
    ++g_macro_note_overflow;
}

} // namespace VideoCommon::Nx2FastDraw
