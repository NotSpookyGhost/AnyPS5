#include "prx/libc/include/general/LogMacros.hpp"
#include "ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/Structurizer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "CacheKey.hpp"
#include "CompiledVariant.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <cstdlib>
#include <cstring>
#include <list>
#include <stdexcept>
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"

namespace AgcDriver::DriverDetail {

std::shared_ptr<const ShaderSnapshot> ReadRawComputeShader(std::uint64_t address) {
    GuestMemory::CheckRange(reinterpret_cast<const void*>(address), sizeof(std::uint32_t), 256);
    static std::mutex cacheMutex;
    static std::list<std::shared_ptr<const ShaderSnapshot>> cache;
    static std::size_t cacheBytes = 0;
    std::shared_ptr<const ShaderSnapshot> cached;
    {
        std::lock_guard lock(cacheMutex);
        const auto found = std::find_if(cache.begin(), cache.end(), [address](const auto& entry) { return entry->codeAddress == address; });
        if (found != cache.end()) cached = *found;
    }
    if (cached) {
        const auto code = std::as_bytes(std::span(cached->code));
        GuestMemory::FlushGpuWrites(address, code.size());
        if (GuestMemory::CompareMapped(address, code) == GuestMemory::Compare::Equal) {
            std::lock_guard lock(cacheMutex);
            const auto found = std::find(cache.begin(), cache.end(), cached);
            if (found != cache.end()) cache.splice(cache.begin(), cache, found);
            return cached;
        }
    }
    constexpr std::size_t limit = 1024 * 1024;
    const auto ranges = GuestMemory::CommittedRanges(address, limit);
    std::uint64_t end = address;
    for (const auto& range : ranges) {
        if (range.first != end) break;
        end = range.second;
    }
    const auto available = static_cast<std::size_t>(end - address) / sizeof(std::uint32_t);
    ShaderSnapshot snapshot{address, 0, 0, {}, {}};
    while (snapshot.code.size() < available) {
        const auto previous = snapshot.code.size();
        snapshot.code.resize(std::min(available, std::max<std::size_t>(64, previous * 2)));
        GuestMemory::Read(address + previous * sizeof(std::uint32_t),
            std::as_writable_bytes(std::span(snapshot.code).subspan(previous)), alignof(std::uint32_t));
        try {
            const auto decoded = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(snapshot.code);
            const auto& last = decoded.instructions.back();
            snapshot.code.resize(last.programCounter / sizeof(std::uint32_t) + last.wordCount);
            auto result = std::make_shared<const ShaderSnapshot>(std::move(snapshot));
            std::lock_guard lock(cacheMutex);
            const auto found = std::find_if(cache.begin(), cache.end(), [address](const auto& entry) { return entry->codeAddress == address; });
            if (found != cache.end()) {
                if ((*found)->code == result->code) {
                    result = *found;
                    cache.splice(cache.begin(), cache, found);
                    return result;
                }
                cacheBytes -= (*found)->code.size() * sizeof(std::uint32_t);
                cache.erase(found);
            }
            const auto bytes = result->code.size() * sizeof(std::uint32_t);
            while (!cache.empty() && (cache.size() >= 64 || cacheBytes + bytes > 8 * 1024 * 1024)) {
                cacheBytes -= cache.back()->code.size() * sizeof(std::uint32_t);
                cache.pop_back();
            }
            cache.push_front(result);
            cacheBytes += bytes;
            return result;
        } catch (const std::out_of_range&) {
            if (snapshot.code.size() == available) break;
        }
    }
    throw std::runtime_error("AGC driver: raw compute program has no reachable end within mapped code or the size limit");
}



std::shared_ptr<const ShaderRecompiler::SourceHandle> SourceHandleFor(const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request) {
    std::lock_guard lock(snapshot.prepared->mutex);
    for (const auto& entry : snapshot.prepared->entries) {
        if (entry.codeOffset == codeOffset && ShaderRecompiler::MatchesPreparedShader(request, *entry.handle)) return entry.handle;
    }
    if (snapshot.header.empty()) {
        if (snapshot.type != 0 || request.shader.stage != ShaderRecompiler::ShaderStage::Compute) throw std::runtime_error("AGC driver: unregistered program is not a compute shader");
        APS5_LOG_ERR("Compute shader 0x%llx was not registered; preparing its artifact at dispatch", static_cast<unsigned long long>(snapshot.codeAddress));
        auto handle = ShaderRecompiler::PrepareShader(request);
        snapshot.prepared->entries.push_back({codeOffset, handle});
        return handle;
    }
    throw std::runtime_error("AGC driver: prepared shader artifact is missing for the requested static ABI");
}

ShaderRecompiler::RectListShaders PreparedRectangle(const ShaderSnapshot& snapshot, std::uint64_t vertexId, std::uint64_t fragmentId) {
    std::lock_guard lock(snapshot.prepared->mutex);
    for (const auto& entry : snapshot.prepared->rectangles) {
        if (entry.vertexId == vertexId && entry.fragmentId == fragmentId) return entry.shaders;
    }
    throw std::runtime_error("AGC driver: prepared rectangle artifacts are missing");
}

ShaderRecompiler::PreparedShaderInvocation InvocationFor(const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request) {
    require(codeOffset < snapshot.code.size(), "prepared shader code offset is outside the snapshot");
    const auto code = std::span(snapshot.code).subspan(codeOffset);
    require(request.shader.code.data() == code.data() && request.shader.code.size() == code.size(), "prepared invocation does not refer to registered code");
    auto invocationRequest = request;
    std::lock_guard lock(snapshot.prepared->mutex);
    for (const auto& entry : snapshot.prepared->entries) {
        if (entry.codeOffset != codeOffset) continue;
        invocationRequest.shader.code = ShaderRecompiler::GetPreparedCode(*entry.handle);
        if (auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(invocationRequest, entry.handle)) return std::move(*invocation);
    }
    throw std::runtime_error("AGC driver: prepared shader artifact is missing for the requested static ABI");
}
namespace {

template<typename TValue>
std::vector<TValue> ReadHeaderArray(const ShaderSnapshot& snapshot, const TValue* pointer, std::size_t count) {
    if (count == 0) return {};
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (address < snapshot.headerAddress || address - snapshot.headerAddress > snapshot.header.size()) throw std::runtime_error("AGC driver: shader metadata is outside the registered header");
    const auto offset = static_cast<std::size_t>(address - snapshot.headerAddress);
    if (count > (snapshot.header.size() - offset) / sizeof(TValue)) throw std::runtime_error("AGC driver: truncated shader metadata");
    std::vector<TValue> result(count);
    std::memcpy(result.data(), snapshot.header.data() + offset, count * sizeof(TValue));
    return result;
}

Shader ReadHeader(const ShaderSnapshot& snapshot) {
    Shader header;
    std::memcpy(&header, snapshot.header.data(), sizeof(header));
    return header;
}

QueueState RegisteredState(const ShaderSnapshot& snapshot) {
    const auto header = ReadHeader(snapshot);
    QueueState state{};
    for (const auto reg : ReadHeaderArray(snapshot, header.sh_registers, header.num_sh_registers)) {
        state.shader.insert_or_assign(reg.offset, reg.value);
    }
    for (const auto reg : ReadHeaderArray(snapshot, header.cx_registers, header.num_cx_registers)) {
        state.context.insert_or_assign(reg.offset, reg.value);
    }
    if (header.specials != nullptr) {
        const auto special = ReadHeaderArray(snapshot, header.specials, 1).front();
        state.context[special.vgt_shader_stages_en.offset] = special.vgt_shader_stages_en.value;
        state.context[special.vgt_gs_out_prim_type.offset] = special.vgt_gs_out_prim_type.value;
        state.userConfig[special.ge_cntl.offset] = special.ge_cntl.value;
        state.userConfig[special.ge_user_vgpr_en.offset] = special.ge_user_vgpr_en.value;
    }
    return state;
}

std::uint32_t RegisterValue(const Registers& registers, std::uint32_t offset) {
    const auto found = registers.find(offset);
    if (found == registers.end()) throw std::runtime_error("AGC driver: missing static shader ABI register " + std::to_string(offset));
    return found->second;
}

std::vector<PreparedShaders::Entry> PrepareRegistered(const ShaderSnapshot& snapshot, const VulkanDevice& device, const QueueState& state, bool registration) {
    using Stage = ShaderRecompiler::ShaderStage;
    const auto header = ReadHeader(snapshot);
    std::uint32_t programRegister;
    std::uint32_t resourceRegister;
    std::uint32_t firstUser = 0;
    Stage stage;
    switch (snapshot.type) {
    case 0: stage = Stage::Compute; programRegister = 0x20c; resourceRegister = 0x213; break;
    case 1: stage = Stage::Fragment; programRegister = 0x008; resourceRegister = 0x00b; break;
    case 2: stage = Stage::Vertex; programRegister = 0x0c8; resourceRegister = 0x08b; firstUser = 8; break;
    case 4: stage = Stage::Mesh; programRegister = 0x0c8; resourceRegister = 0x08b; break;
    case 5: stage = Stage::Local; programRegister = 0x148; resourceRegister = 0x10b; firstUser = 8; break;
    case 6: stage = Stage::Mesh; programRegister = 0x088; resourceRegister = 0x08b; break;
    case 7: stage = Stage::TessellationControl; programRegister = 0x108; resourceRegister = 0x10b; break;
    default: throw std::runtime_error("AGC driver: unsupported registered shader type");
    }
    const auto high = RegisterValue(state.shader, programRegister + 1);
    if ((high & ~0xffu) != 0) throw std::runtime_error("AGC driver: invalid registered program address");
    const auto address = (static_cast<std::uint64_t>(RegisterValue(state.shader, programRegister)) << 8u) | (static_cast<std::uint64_t>(high) << 40u);
    if (address < snapshot.codeAddress || address - snapshot.codeAddress >= snapshot.code.size() * 4u) throw std::runtime_error("AGC driver: registered entry point is outside shader code");
    const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / 4u);
    const auto code = std::span(snapshot.code).subspan(codeOffset);
    const auto decoded = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(code);
    auto graph = ShaderRecompiler::GraphBuilder{}.Build(decoded);
    ShaderRecompiler::Structurizer{}.Structurize(graph);
    std::optional<ShaderRecompiler::ShaderComputeStageInfo> compute;
    std::optional<ShaderRecompiler::ShaderPixelStageInfo> pixel;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex;
    std::optional<ShaderRecompiler::GraphicsCompileContext> graphics;
    std::uint32_t wave;
    const auto resources = RegisterValue(state.shader, resourceRegister);
    auto userCount = (resources >> 1u) & 0x1fu;
    if (stage != Stage::Compute) userCount |= ((resources >> 27u) & 1u) << 5u;
    if (userCount > 32) throw std::runtime_error("AGC driver: static user SGPR count exceeds the register bank");
    if (stage == Stage::Compute) {
        compute = Graphics::DecodeComputeStageInfo(state.shader);
        const auto special = ReadHeaderArray(snapshot, header.specials, 1).front();
        wave = (special.dispatch_modifier & 0x8000u) != 0 ? 32u : 64u;
    } else if (stage == Stage::Fragment) {
        const auto count = RegisterValue(state.context, 0x1b6) & 0x3fu;
        if (registration && count != 0) return {};
        pixel = Graphics::DecodePixelStageInfo(state.context, {});
        wave = pixel->wave32 ? 32u : 64u;
    } else {
        const auto routing = RegisterValue(state.context, 0x2d5);
        wave = (routing & 0x00400000u) != 0 ? 32u : 64u;
        if ((routing & 0x20u) != 0 || stage == Stage::Mesh || (routing & 4u) != 0) {
            if (registration) return {};
            auto stageState = state;
            stageState.context[0x1b6] = 0;
            const auto stages = Graphics::DecodeShaderStages(stageState);
            graphics = ShaderRecompiler::GraphicsCompileContext{0, {}, stages.mesh, stages.tessellation, {}};
            if (stages.mesh) { stage = Stage::Mesh; firstUser = 0; }
            if (stages.tessellation && snapshot.type == 2) stage = Stage::TessellationEvaluation;
        }
        if (firstUser == 0) userCount += 8;
    }
    std::vector<std::uint32_t> userData(userCount);
    if (stage != Stage::Compute && stage != Stage::Fragment && snapshot.type != 6) vertex = Graphics::DecodeVertexStageInfo(snapshot.header, snapshot.headerAddress, userData, nullptr, true);
    const std::array<ShaderRecompiler::MemoryRegion, 2> memory{{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}, {snapshot.headerAddress, snapshot.header}}};
    ShaderRecompiler::RecompileRequest request{{stage, address, code, snapshot.headerAddress, snapshot.header}, {wave, firstUser, userData, compute, pixel, vertex, memory}, stage == Stage::Compute ? device.ComputeTarget(wave) : device.Target(), {0, 0, 0, 128}, graphics};
    if (graphics && graphics->mesh) request.layout.pushConstantSizeBytes = ShaderRecompiler::MeshDrawPushOffsetBytes;
    std::vector<PreparedShaders::Entry> entries;
    const auto append = [&] { entries.push_back({codeOffset, ShaderRecompiler::PrepareShader(request)}); };
    append();
    if (compute) {
        request.context.compute->partialThreads = {1, 1, 1};
        append();
    } else if (pixel) {
        request.layout.pushConstantSizeBytes = ShaderRecompiler::MeshDrawPushOffsetBytes;
        append();
    }
    return entries;
}

}

std::vector<PreparedGraphicsStage> PrepareGraphicsStages(const DrawDecode& decoded, const ShaderRecompiler::SpirvTarget& target) {
    require(decoded.programs.size() == decoded.roles.size(), "graphics ABI program roles are incomplete");
    for (const auto& program : decoded.programs) require(program.snapshot != nullptr, "graphics ABI has no registered shader snapshot");
    std::vector<ShaderRecompiler::ProgramRole> expected;
    if (decoded.state.stages.path == Graphics::ShaderPath::Tessellation) expected = {ShaderRecompiler::ProgramRole::Local, ShaderRecompiler::ProgramRole::Hull, ShaderRecompiler::ProgramRole::Domain};
    else {
        expected.push_back(ShaderRecompiler::ProgramRole::Main);
        if (decoded.state.stages.path == Graphics::ShaderPath::Geometry && !decoded.programs.empty() && decoded.programs.front().snapshot->type == 4) expected.push_back(ShaderRecompiler::ProgramRole::GeometryBack);
    }
    if (!decoded.roles.empty() && decoded.roles.back() == ShaderRecompiler::ProgramRole::Fragment) expected.push_back(ShaderRecompiler::ProgramRole::Fragment);
    require(expected == decoded.roles, "graphics ABI linked programs do not match the stage routing");
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    for (std::size_t index = 0; index < decoded.programs.size(); ++index) {
        const auto& program = decoded.programs[index];
        require(program.snapshot != nullptr && program.codeOffset < program.snapshot->code.size(), "graphics ABI has an invalid registered entry point");
        const auto code = std::span(program.snapshot->code).subspan(program.codeOffset);
        require(program.binary.code.data() == code.data() && program.binary.code.size() == code.size() && program.binary.codeAddress == program.snapshot->codeAddress + program.codeOffset * sizeof(std::uint32_t), "graphics ABI entry point differs from its snapshot");
        linked.push_back({decoded.roles[index], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
        memory.insert(memory.end(), program.memory.begin(), program.memory.end());
    }
    std::vector<PreparedGraphicsStage> prepared;
    std::uint32_t pushOffset = 0;
    const auto capacity = decoded.state.stages.mesh ? ShaderRecompiler::MeshDrawPushOffsetBytes : Graphics::PipelinePushConstantBytes;
    for (std::size_t index = 0; index < decoded.programs.size(); ++index) {
        if (decoded.roles[index] == ShaderRecompiler::ProgramRole::GeometryBack) continue;
        const auto& program = decoded.programs[index];
        const bool fragment = program.binary.stage == ShaderRecompiler::ShaderStage::Fragment;
        const auto wave = fragment ? decoded.state.stages.fragmentWaveSize : decoded.state.stages.vertexWaveSize;
        std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex;
        if (!fragment) vertex = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, nullptr, true);
        ShaderRecompiler::RecompileRequest request{program.binary, {wave, program.firstUserSgpr, program.userData, {}, fragment ? std::optional(decoded.pixel) : std::nullopt, vertex, memory}, target, {0, 0, pushOffset, capacity - pushOffset}, ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, decoded.state.stages.mesh, decoded.state.stages.tessellation, {}}};
        const auto handle = ShaderRecompiler::PrepareShader(request);
        const auto bytes = handle->artifact->bindings.pushConstantSizeBytes;
        require(bytes <= capacity - pushOffset, "prepared graphics stages exceed the push constant block");
        prepared.push_back({program.snapshot, {program.codeOffset, handle}});
        pushOffset += bytes;
    }
    return prepared;
}

void Driver::ResolveGraphicsStagesAbi(std::span<const Shader* const> stages, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    CheckFailure();
    require(!stages.empty(), "graphics ABI has no shader headers");
    QueueState state{};
    std::shared_ptr<const ShaderRegistry> registry;
    {
        std::lock_guard lock(mutex);
        registry = shaders;
        for (const auto* shader : stages) {
            GuestMemory::CheckRange(shader, sizeof(Shader), alignof(Shader));
            const auto address = reinterpret_cast<std::uintptr_t>(const_cast<const void*>(shader->code));
            require(registry != nullptr && registry->contains(address), "graphics ABI refers to an unregistered shader");
            const auto& snapshot = *registry->at(address);
            require(snapshot.headerAddress == reinterpret_cast<std::uintptr_t>(shader), "graphics ABI refers to a replaced shader header");
            const auto registered = RegisteredState(snapshot);
            for (const auto& [offset, value] : registered.shader) state.shader.insert_or_assign(offset, value);
            for (const auto& [offset, value] : registered.context) state.context.insert_or_assign(offset, value);
            for (const auto& [offset, value] : registered.userConfig) state.userConfig.insert_or_assign(offset, value);
        }
    }
    for (const auto reg : context) state.context.insert_or_assign(reg.offset, reg.value);
    for (const auto reg : primitive) state.userConfig.insert_or_assign(reg.offset, reg.value);
    DrawDecode decoded{};
    decoded.state.stages = Graphics::DecodeShaderStages(state);
    DecodeGraphicsPrograms(decoded, state, *registry, true, false);
    const auto localDevice = device.Load();
    require(localDevice != nullptr, "graphics ABI device is missing");
    auto prepared = PrepareGraphicsStages(decoded, localDevice->Target());
    for (auto& stage : prepared) {
        std::lock_guard lock(stage.snapshot->prepared->mutex);
        auto& entries = stage.snapshot->prepared->entries;
        const auto duplicate = std::ranges::any_of(entries, [&](const auto& existing) { return existing.codeOffset == stage.entry.codeOffset && existing.handle->artifact == stage.entry.handle->artifact; });
        if (!duplicate) entries.push_back(std::move(stage.entry));
    }
    const auto primitiveType = state.userConfig.find(0x242u);
    for (const auto& stage : prepared) {
        if (stage.snapshot->type == 1) continue;
        ResolvePreparedGraphics(*stage.snapshot, {}, primitiveType == state.userConfig.end() ? 0u : primitiveType->second, localDevice->Target());
    }
}

void Driver::ResolveShaderAbi(const Shader* shader, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    CheckFailure();
    GuestMemory::CheckRange(shader, sizeof(Shader), alignof(Shader));
    if (shader->type != 0 && shader->type != 1) {
        const std::array<const Shader*, 1> stages{shader};
        ResolveGraphicsStagesAbi(stages, context, primitive);
        return;
    }
    std::shared_ptr<const ShaderSnapshot> snapshot;
    {
        std::lock_guard lock(mutex);
        const auto address = reinterpret_cast<std::uintptr_t>(const_cast<const void*>(shader->code));
        require(shaders != nullptr && shaders->contains(address), "static ABI refers to an unregistered shader");
        snapshot = shaders->at(address);
        require(snapshot->headerAddress == reinterpret_cast<std::uintptr_t>(shader), "static ABI refers to a replaced shader header");
    }
    auto state = RegisteredState(*snapshot);
    for (const auto reg : context) state.context[reg.offset] = reg.value;
    for (const auto reg : primitive) state.userConfig[reg.offset] = reg.value;
    const auto localDevice = device.Load();
    require(localDevice != nullptr, "shader registration device is missing");
    auto entries = PrepareRegistered(*snapshot, *localDevice, state, false);
    std::lock_guard lock(snapshot->prepared->mutex);
    for (auto& entry : entries) {
        const auto duplicate = std::ranges::any_of(snapshot->prepared->entries, [&](const auto& existing) { return existing.handle->artifact == entry.handle->artifact; });
        if (!duplicate) snapshot->prepared->entries.push_back(std::move(entry));
    }
}

alignas(256) static const std::uint32_t NullPixelCode[64] = {0xbf810000u};
static const Shader NullPixelShader = [] {
    Shader shader{};
    shader.file_header = 0x34333231u;
    shader.version = 0x18u;
    shader.code = NullPixelCode;
    shader.header_size = sizeof(Shader);
    shader.shader_size = sizeof(NullPixelCode);
    shader.type = 1;
    return shader;
}();

std::uint64_t NullPixelProgramAddress() {
    return reinterpret_cast<std::uintptr_t>(NullPixelCode);
}

void Driver::ResolveGraphicsAbi(const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType) {
    CheckFailure();
    if (primitiveType != 0 && primitiveType != 7 && primitiveType != 17) return;
    require(vertex != nullptr && pixel != nullptr, "rectangle ABI requires vertex and fragment shaders");
    std::shared_ptr<const ShaderSnapshot> front;
    std::shared_ptr<const ShaderSnapshot> fragment;
    {
        std::lock_guard lock(mutex);
        GuestMemory::CheckRange(vertex, sizeof(Shader), alignof(Shader));
        GuestMemory::CheckRange(pixel, sizeof(Shader), alignof(Shader));
        const auto lookup = [&](const Shader* shader) {
            const auto address = reinterpret_cast<std::uintptr_t>(const_cast<const void*>(shader->code));
            require(shaders != nullptr && shaders->contains(address), "rectangle ABI refers to an unregistered shader");
            const auto snapshot = shaders->at(address);
            require(snapshot->headerAddress == reinterpret_cast<std::uintptr_t>(shader), "rectangle ABI refers to a replaced shader header");
            return snapshot;
        };
        front = lookup(vertex);
        fragment = lookup(pixel);
    }
    require(front != fragment, "rectangle stages refer to the same shader");
    const auto localDevice = device.Load();
    require(localDevice != nullptr, "shader registration device is missing");
    ResolvePreparedGraphics(*front, fragment, primitiveType, localDevice->Target());
}

void ResolvePreparedGraphics(const ShaderSnapshot& front, const std::shared_ptr<const ShaderSnapshot>& fragment, std::uint32_t primitiveType, const ShaderRecompiler::SpirvTarget& target) {
    require(fragment.get() != &front, "rectangle stages refer to the same shader");
    std::vector<std::shared_ptr<const ShaderSnapshot>> fragments;
    {
        std::lock_guard lock(front.prepared->mutex);
        auto& prepared = *front.prepared;
        std::erase_if(prepared.fragments, [](const auto& entry) { return entry.expired(); });
        if (fragment && !std::ranges::any_of(prepared.fragments, [&](const auto& entry) { return entry.lock() == fragment; })) prepared.fragments.push_back(fragment);
        prepared.rectangleRequested |= primitiveType == 7 || primitiveType == 17;
        if (!prepared.rectangleRequested) return;
        for (const auto& entry : prepared.fragments) {
            if (auto snapshot = entry.lock()) fragments.push_back(std::move(snapshot));
        }
    }
    for (const auto& pixel : fragments) {
        std::scoped_lock lock(front.prepared->mutex, pixel->prepared->mutex);
        require(!front.prepared->entries.empty() && !pixel->prepared->entries.empty(), "rectangle ABI has missing stage artifacts");
        std::vector<PreparedShaders::Rectangle> rectangles;
        for (const auto& vertexEntry : front.prepared->entries) {
            for (const auto& fragmentEntry : pixel->prepared->entries) {
                const auto& vertexArtifact = ShaderRecompiler::GetPreparedArtifact(*vertexEntry.handle);
                const auto& fragmentArtifact = ShaderRecompiler::GetPreparedArtifact(*fragmentEntry.handle);
                const auto exists = [&](const auto& entry) { return entry.vertexId == vertexArtifact.variantId && entry.fragmentId == fragmentArtifact.variantId; };
                if (std::ranges::any_of(front.prepared->rectangles, exists) || std::ranges::any_of(rectangles, exists)) continue;
                ShaderRecompiler::RecompileResult vertexResult;
                ShaderRecompiler::RecompileResult fragmentResult;
                static_cast<ShaderRecompiler::CompiledShaderArtifact&>(vertexResult) = vertexArtifact;
                static_cast<ShaderRecompiler::CompiledShaderArtifact&>(fragmentResult) = fragmentArtifact;
                rectangles.push_back({vertexArtifact.variantId, fragmentArtifact.variantId, ShaderRecompiler::BuildRectListShaders(vertexResult, fragmentResult, target)});
            }
        }
        front.prepared->rectangles.insert(front.prepared->rectangles.end(), std::make_move_iterator(rectangles.begin()), std::make_move_iterator(rectangles.end()));
    }
}

void Driver::RegisterShader(const Shader* shader) {
    CheckFailure();
    GuestMemory::CheckRange(shader, sizeof(Shader), 1);
    Shader fields;
    std::memcpy(&fields, static_cast<const void*>(shader), sizeof(Shader));
    require(fields.file_header == 0x34333231u && fields.version == 0x18u, "invalid shader header");
    require(fields.header_size >= sizeof(Shader), "shader header is smaller than its fixed fields");
    require(fields.shader_size != 0 && (fields.shader_size & 3u) == 0, "invalid shader size");
    GuestMemory::CheckRange(shader, fields.header_size, 1);
    const auto* code = const_cast<const void*>(fields.code);
    GuestMemory::CheckRange(code, fields.shader_size, 256);
    ShaderSnapshot snapshot{reinterpret_cast<std::uintptr_t>(code), reinterpret_cast<std::uintptr_t>(shader), fields.type, {}, {}};
    snapshot.code.resize(fields.shader_size / sizeof(std::uint32_t));
    std::memcpy(snapshot.code.data(), code, fields.shader_size);
    snapshot.header.resize(fields.header_size);
    std::memcpy(snapshot.header.data(), static_cast<const void*>(shader), fields.header_size);

    std::shared_ptr<VulkanDevice> localDevice = device.Load();
    if (localDevice == nullptr) {
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    snapshot.prepared->entries = PrepareRegistered(snapshot, *localDevice, RegisteredState(snapshot), true);
    static const char* traceRegs = std::getenv("APS5_TRACE_SHADER_REGS");
    if (traceRegs != nullptr && (std::string(traceRegs) == "all" || std::strtoull(traceRegs, nullptr, 16) == snapshot.codeAddress)) {
        const auto print = [](const ShaderRegister* registers, std::uint32_t count) {
            for (std::uint32_t i = 0; i < count && registers != nullptr; ++i) {
                ShaderRegister value;
                std::memcpy(&value, static_cast<const void*>(registers + i), sizeof(value));
                std::fprintf(stderr, " %x=%08x", value.offset, value.value);
            }
        };
        std::fprintf(stderr, "[shader] 0x%llx type %u cx", static_cast<unsigned long long>(snapshot.codeAddress), fields.type);
        print(fields.cx_registers, fields.num_cx_registers);
        std::fprintf(stderr, " sh");
        print(fields.sh_registers, fields.num_sh_registers);
        std::fprintf(stderr, "\n");
    }
    std::lock_guard lock(mutex);
    rethrowFailure();
    const auto address = snapshot.codeAddress;

    if (shaders == nullptr) shaders = std::make_shared<ShaderRegistry>();
    else if (shaders.use_count() != 1) shaders = std::make_shared<ShaderRegistry>(*shaders);
    shaders->insert_or_assign(address, std::make_shared<const ShaderSnapshot>(std::move(snapshot)));
    if (shaders->find(NullPixelProgramAddress()) == shaders->end()) {
        ShaderSnapshot null{NullPixelProgramAddress(), reinterpret_cast<std::uintptr_t>(&NullPixelShader), NullPixelShader.type, {}, {}};
        null.code.assign(std::begin(NullPixelCode), std::end(NullPixelCode));
        null.header.resize(sizeof(Shader));
        std::memcpy(null.header.data(), &NullPixelShader, sizeof(Shader));
        null.prepared->entries = PrepareRegistered(null, *localDevice, RegisteredState(null), true);
        shaders->insert_or_assign(NullPixelProgramAddress(), std::make_shared<const ShaderSnapshot>(std::move(null)));
    }
}

}
