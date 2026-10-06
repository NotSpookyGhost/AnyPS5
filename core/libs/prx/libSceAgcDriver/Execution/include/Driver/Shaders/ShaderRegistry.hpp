#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERREGISTRY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERREGISTRY_HPP

#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver::DriverDetail {

struct PreparedShaders {
    struct Entry {
        std::size_t codeOffset;
        std::shared_ptr<const ShaderRecompiler::SourceHandle> handle;
    };
    struct Rectangle {
        std::uint64_t vertexId;
        std::uint64_t fragmentId;
        ShaderRecompiler::RectListShaders shaders;
    };
    std::mutex mutex;
    std::vector<Entry> entries;
    std::vector<Rectangle> rectangles;
};
struct ShaderSnapshot {
    std::uint64_t codeAddress;
    std::uint64_t headerAddress;
    std::uint8_t type;
    std::vector<std::uint32_t> code;
    std::vector<std::byte> header;
    std::unique_ptr<PreparedShaders> prepared = std::make_unique<PreparedShaders>();
};

using ShaderRegistry = std::map<std::uint64_t, std::shared_ptr<const ShaderSnapshot>>;

std::shared_ptr<const ShaderSnapshot> ReadRawComputeShader(std::uint64_t address);

std::uint64_t NullPixelProgramAddress();
ShaderRecompiler::RectListShaders PreparedRectangle(const ShaderSnapshot& snapshot, std::uint64_t vertexId, std::uint64_t fragmentId);

std::shared_ptr<const ShaderRecompiler::SourceHandle> SourceHandleFor(const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request);
ShaderRecompiler::PreparedShaderInvocation InvocationFor(const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request);

}

#endif
