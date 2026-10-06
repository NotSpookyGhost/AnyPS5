#ifndef CORE_LIBS_AGCDRIVER_GRAPHICS_PIPELINESPECIALIZATION_HPP
#define CORE_LIBS_AGCDRIVER_GRAPHICS_PIPELINESPECIALIZATION_HPP

#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <limits>
#include <vector>

namespace AgcDriver::Graphics {

class PipelineSpecialization {
public:
    explicit PipelineSpecialization(const ShaderRecompiler::RecompileResult& shader) {
        entries.reserve(shader.specialization.size());
        for (std::size_t index = 0; index < shader.specialization.size(); ++index) {
            const auto offset = index * sizeof(ShaderRecompiler::PipelineSpecializationConstant) + offsetof(ShaderRecompiler::PipelineSpecializationConstant, value);
            Require(offset <= std::numeric_limits<std::uint32_t>::max(), "pipeline specialization offset exceeds Vulkan limits");
            entries.push_back({shader.specialization[index].id, static_cast<std::uint32_t>(offset), sizeof(std::uint32_t)});
        }
        info.mapEntryCount = static_cast<std::uint32_t>(entries.size());
        info.pMapEntries = entries.data();
        info.dataSize = shader.specialization.size() * sizeof(ShaderRecompiler::PipelineSpecializationConstant);
        info.pData = shader.specialization.data();
    }

    [[nodiscard]] const VkSpecializationInfo* Info() const { return entries.empty() ? nullptr : &info; }

private:
    std::vector<VkSpecializationMapEntry> entries;
    VkSpecializationInfo info{};
};

}

#endif
