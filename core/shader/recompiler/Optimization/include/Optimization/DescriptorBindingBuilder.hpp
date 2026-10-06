#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DESCRIPTORBINDINGBUILDER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DESCRIPTORBINDINGBUILDER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "Recompiler.hpp"
#include <cstdint>

namespace ShaderRecompiler {

std::uint32_t PointFilteredSamplerWord(std::uint32_t word0, std::uint32_t filter);
struct DescriptorBindingPlan {
    std::vector<std::uint32_t> imageModes;
    std::vector<IrDescriptorBinding> selected;
    std::vector<std::vector<std::uint32_t>> originalElements;
    std::vector<PipelineSpecializationConstant> specialization;
};

class DescriptorBindingBuilder {
public:
    DescriptorBindingPlan Prepare(const IrBindingLayout& layout, const ShaderInfo& info, IrShaderStage stage, const ResourceSnapshot& snapshot, std::span<const std::uint8_t> exportMappings = {}) const;
    void Populate(BindingAllocationResult& allocation, const CompiledBindingLayout& compiled, const DescriptorBindingPlan& plan, const ShaderInfo& info, IrShaderStage stage, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads) const;
    void Populate(BindingAllocationResult& allocation, const IrProgram& program, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings = {}) const;
    void Populate(BindingAllocationResult& allocation, const ShaderInfo& info, IrShaderStage stage, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings = {}) const;
};

}

#endif
