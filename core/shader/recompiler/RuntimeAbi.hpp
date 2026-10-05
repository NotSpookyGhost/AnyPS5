#ifndef CORE_SHADER_RECOMPILER_RUNTIMEABI_HPP
#define CORE_SHADER_RECOMPILER_RUNTIMEABI_HPP

#include <cstdint>
#include <stdexcept>

namespace ShaderRecompiler::RuntimeAbi {

inline constexpr std::uint32_t Version = 1u;
inline constexpr std::uint32_t DescriptorSet = 0u;
inline constexpr std::uint32_t StageCount = 4u;
inline constexpr std::uint32_t PushConstantDwords = 32u;
inline constexpr std::uint32_t FirstImageBinding = 1u;
inline constexpr std::uint32_t FirstComparisonImageBinding = 22u;
inline constexpr std::uint32_t FirstStorageImageBinding = 29u;
inline constexpr std::uint32_t ImageBindingCount = 43u;

enum class Binding : std::uint32_t {
    Buffers = 0u,
    Samplers = 44u,
    Gds = 45u,
    BdaPagetable = 46u,
    FaultBuffer = 47u,
    FlattenedSrt = 48u,
    ShaderData = 49u,
    Count = 50u
};

enum class Stage : std::uint32_t { Main, Fragment, TessellationControl, TessellationEvaluation };

inline void RequireVersion(std::uint32_t version) {
    if (version != Version) throw std::runtime_error("Shader runtime ABI: incompatible version");
}

inline std::uint32_t BindingNumber(Stage stage, Binding binding) {
    const auto group = static_cast<std::uint32_t>(stage);
    const auto index = static_cast<std::uint32_t>(binding);
    const auto count = static_cast<std::uint32_t>(Binding::Count);
    if (group >= StageCount || index >= count) throw std::runtime_error("Shader runtime ABI: invalid stage or binding");
    return group * count + index;
}

static_assert(FirstImageBinding + ImageBindingCount == static_cast<std::uint32_t>(Binding::Samplers));

}

#endif
