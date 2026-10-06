#ifndef CORE_SHADER_RECOMPILER_PIPELINESPECIALIZATION_HPP
#define CORE_SHADER_RECOMPILER_PIPELINESPECIALIZATION_HPP

#include <cstdint>

namespace ShaderRecompiler {

struct PipelineSpecializationConstant {
    std::uint32_t id;
    std::uint32_t value;

    bool operator==(const PipelineSpecializationConstant&) const = default;
};

namespace PipelineSpecialization {

inline constexpr std::uint32_t BufferBase = 0u;
inline constexpr std::uint32_t BufferWords = 3u;
inline constexpr std::uint32_t ImageBase = 128u;
inline constexpr std::uint32_t ImageWords = 5u;
inline constexpr std::uint32_t VertexBase = 512u;
inline constexpr std::uint32_t VertexWords = 6u;

}

}

#endif
