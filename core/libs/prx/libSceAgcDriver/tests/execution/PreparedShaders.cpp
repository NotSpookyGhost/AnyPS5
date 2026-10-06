#include "VulkanTestDevice.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename TAction>
void ExpectFailure(TAction action, const char* expected) {
    try { action(); }
    catch (const std::exception& error) {
        Require(std::string(error.what()).find(expected) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("expected preparation failure");
}

void Run(AgcDriver::VulkanDevice& device) {
    alignas(256) std::array<std::uint32_t, 1> code{0xbf810000u};
    std::array<std::uint32_t, 4> users{};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1, {}};
    ShaderRecompiler::RecompileRequest request{{ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}}, {32, 0, users, compute, {}, {}, {}}, device.ComputeTarget(32), {0, 0, 0, 128}};
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{request.shader.codeAddress, 0, 0, {code.begin(), code.end()}, {}};
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, device.Serial(), request)); }, "artifact is missing");
    const auto handle = ShaderRecompiler::PrepareShader(request);
    snapshot.prepared->entries.push_back({0, device.Serial(), handle});
    const auto& artifact = ShaderRecompiler::GetPreparedArtifact(*handle);
    for (std::uint32_t value = 0; value < 8; ++value) {
        users.fill(value);
        request.useCache = value % 2 == 0;
        const auto ready = AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, device.Serial(), request);
        ShaderRecompiler::SrtRuntime runtime{};
        runtime.userData = users;
        const auto capture = ShaderRecompiler::CaptureResources(request, runtime, *ready);
        const auto result = ShaderRecompiler::MaterializeShader(request, *capture, *ready);
        Require(result->variantId == artifact.variantId && result->spirv.data() == artifact.spirv.data(), "invocation replaced the prepared artifact");
        device.Dispatch(*result, 1, 1, 1);
    }
    device.WaitIdle();
    request.context.compute->numThreads[0] = 2;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, device.Serial(), request)); }, "artifact is missing");
    request.context.compute->numThreads[0] = 1;
    request.layout.pushConstantSizeBytes = 124;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, device.Serial(), request)); }, "artifact is missing");
    request.layout.pushConstantSizeBytes = 128;
    code[0] = 0xffffffffu;
    ExpectFailure([&] { static_cast<void>(AgcDriver::DriverDetail::SourceHandleFor(snapshot, 0, device.Serial(), request)); }, "artifact is missing");
}

void Registration() {
    alignas(256) std::array<std::uint32_t, 1> code{0xbf810000u};
    struct Header {
        Shader shader{};
        std::array<ShaderRegister, 7> registers{};
        ShaderSpecialRegs specials{};
    } header;
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = sizeof(code);
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)}, {0x207, 1}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, 0}}};
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    code[0] = 0xffffffffu;
    ExpectFailure([&] { AgcDriverRegisterShader_nid_postfix(&header.shader); }, "");
    std::vector<std::uint32_t> commands;
    for (const auto reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, 0x8001});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    sceAgcDriverSubmitAcb(0x20, &packet);
    AgcDriverWaitIdle_nid_postfix();
    AgcDriverShutdown_nid_postfix();
}

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
        device.reset();
        Registration();
        std::cout << "prepared shader and transactional registration tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}