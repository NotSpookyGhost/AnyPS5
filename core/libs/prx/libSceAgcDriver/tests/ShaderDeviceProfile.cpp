#include "prx/libSceAgcDriver/Execution/include/ShaderDeviceProfile.hpp"
#include "BdaAbi.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

using AgcDriver::Graphics::Require;

struct ProfileInput {
    std::vector<std::uint32_t> capabilities{spv::CapabilityShader, spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess};
    std::array<std::string, 2> extensionStrings{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    std::array<std::string_view, 2> extensions{extensionStrings[0], extensionStrings[1]};
    VkPhysicalDeviceDescriptorIndexingFeatures indexing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDevice8BitStorageFeatures bytes{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, &indexing, VK_TRUE};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, &bytes, VK_TRUE};
    VkPhysicalDeviceFeatures core{};
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    VkPhysicalDeviceLimits limits{};

    ProfileInput() {
        core.shaderInt64 = VK_TRUE;
        core.vertexPipelineStoresAndAtomics = VK_TRUE;
        core.fragmentStoresAndAtomics = VK_TRUE;
        device.pEnabledFeatures = &core;
        device.pNext = &bda;
        limits.maxPushConstantsSize = 128u;
        limits.maxBoundDescriptorSets = 1u;
    }

    ShaderRecompiler::SpirvTarget Target() const {
        ShaderRecompiler::SpirvTarget target{};
        target.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
        target.supportedCapabilities = capabilities;
        target.supportedExtensions = extensions;
        return target;
    }
};

template<typename TAction>
void Reject(TAction action, const char* expected) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        Require(std::string(error.what()).find(expected) != std::string::npos, "unexpected profile validation error");
        return;
    }
    throw std::runtime_error("invalid shader device profile was accepted");
}

void CheckProfile() {
    ProfileInput input;
    const AgcDriver::ShaderDeviceProfile profile(input.Target(), input.device, input.limits);
    input.capabilities.clear();
    input.extensionStrings[0] = "changed";
    input.core.shaderInt64 = VK_FALSE;
    input.limits.maxPushConstantsSize = 0u;
    const auto target = profile.Target();
    Require(target.supportedCapabilities.size() == 4u && std::ranges::is_sorted(target.supportedCapabilities), "profile capabilities were not frozen");
    Require(std::ranges::find(target.supportedExtensions, "SPV_KHR_physical_storage_buffer") != target.supportedExtensions.end(), "profile extension storage is borrowed");
    Require(profile.Limits().maxPushConstantsSize == 128u, "profile limits were not frozen");
    static_assert(!std::is_copy_constructible_v<AgcDriver::ShaderDeviceProfile> && !std::is_move_constructible_v<AgcDriver::ShaderDeviceProfile>);
    const auto invalid = [](auto mutate, const char* expected) {
        ProfileInput candidate;
        mutate(candidate);
        Reject([&] { AgcDriver::ShaderDeviceProfile rejected(candidate.Target(), candidate.device, candidate.limits); }, expected);
    };
    invalid([](auto& value) { value.device.pEnabledFeatures = nullptr; }, "enabled core features");
    invalid([](auto& value) { value.bda.bufferDeviceAddress = VK_FALSE; }, "bufferDeviceAddress");
    invalid([](auto& value) { value.device.pNext = nullptr; }, "bufferDeviceAddress");
    invalid([](auto& value) { value.core.shaderInt64 = VK_FALSE; }, "shaderInt64");
    invalid([](auto& value) { value.bytes.storageBuffer8BitAccess = VK_FALSE; }, "storageBuffer8BitAccess");
    invalid([](auto& value) { value.core.fragmentStoresAndAtomics = VK_FALSE; }, "graphics stores and atomics");
    invalid([](auto& value) { value.limits.maxPushConstantsSize = 127u; }, "exceeds device limits");
    invalid([](auto& value) { value.limits.maxBoundDescriptorSets = 0u; }, "exceeds device limits");
    invalid([](auto& value) { value.capabilities.clear(); }, "missing runtime capabilities");
    invalid([](auto& value) { value.extensions[0] = "missing"; }, "missing runtime extensions");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilitySampledImageArrayDynamicIndexing); }, "shaderSampledImageArrayDynamicIndexing");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityStorageImageArrayNonUniformIndexing); }, "shaderStorageImageArrayNonUniformIndexing");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityStorageImageReadWithoutFormat); }, "shaderStorageImageReadWithoutFormat");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityFragmentBarycentricKHR); }, "fragmentShaderBarycentric");
    invalid([](auto& value) { value.capabilities.push_back(spv::CapabilityMeshShadingEXT); }, "meshShader");
    ProfileInput enabled;
    enabled.capabilities.push_back(spv::CapabilitySampledImageArrayDynamicIndexing);
    enabled.capabilities.push_back(spv::CapabilityStorageImageArrayNonUniformIndexing);
    enabled.core.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    enabled.indexing.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    const AgcDriver::ShaderDeviceProfile accepted(enabled.Target(), enabled.device, enabled.limits);
    Require(accepted.Target().supportedCapabilities.size() == 6u, "enabled descriptor indexing was rejected");
}

void CheckAbi() {
    using ShaderRecompiler::RuntimeAbi::Binding;
    using ShaderRecompiler::RuntimeAbi::BindingNumber;
    using ShaderRecompiler::RuntimeAbi::Stage;
    std::set<std::uint32_t> bindings;
    for (std::uint32_t stage = 0u; stage < ShaderRecompiler::RuntimeAbi::StageCount; ++stage) {
        for (std::uint32_t binding = 0u; binding < static_cast<std::uint32_t>(Binding::Count); ++binding) {
            Require(bindings.insert(BindingNumber(static_cast<Stage>(stage), static_cast<Binding>(binding))).second, "runtime ABI bindings overlap");
        }
    }
    Require(BindingNumber(Stage::Main, Binding::ShaderData) == 49u && BindingNumber(Stage::Fragment, Binding::ShaderData) == 99u, "runtime ABI binding numbers changed");
    Reject([] { BindingNumber(static_cast<Stage>(4u), Binding::Buffers); }, "invalid stage or binding");
    Reject([] { BindingNumber(Stage::Main, Binding::Count); }, "invalid stage or binding");
    Reject([] { ShaderRecompiler::RuntimeAbi::RequireVersion(0u); }, "incompatible version");
    ShaderRecompiler::RuntimeAbi::RequireVersion(ShaderRecompiler::RuntimeAbi::Version);
}

}

int main() {
    try {
        CheckAbi();
        CheckProfile();
        std::cout << "shader runtime ABI and device profile tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
