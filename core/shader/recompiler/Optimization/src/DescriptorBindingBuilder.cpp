#include "Optimization/DescriptorBindingBuilder.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ShaderRecompiler {

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

// Debug aid: APS5_TRACE_BUFFER_WRITTEN=1 prints, after every Populate, how many guest buffer
// elements the program may store to and how many it only loads (this call and cumulative), so the
// replay tool shows what a driver gains from DescriptorBinding::bufferWritten.
bool bufferWrittenTraceEnabled() {
    static const bool enabled = std::getenv("APS5_TRACE_BUFFER_WRITTEN") != nullptr;
    return enabled;
}
struct BufferWrittenCounts {
    std::atomic<unsigned long long> written{0};
    std::atomic<unsigned long long> readOnly{0};
};
BufferWrittenCounts bufferWrittenCounts;

DescriptorKind PhysicalKindFor(DescriptorBindingKind kind) {
    if (kind == DescriptorBindingKind::Samplers) {
        return DescriptorKind::Sampler;
    }
    const ImageResourceClass imageClass = ImageBindingResourceClass(kind);
    if (imageClass == ImageResourceClass::Sampled) {
        return DescriptorKind::SampledImage;
    }
    if (imageClass == ImageResourceClass::Storage) {
        return DescriptorKind::StorageImage;
    }
    return DescriptorKind::StorageBuffer;
}

DescriptorRole RoleFor(DescriptorBindingKind kind) {
    if (kind == DescriptorBindingKind::Buffers) {
        return DescriptorRole::GuestBuffers;
    }
    if (kind == DescriptorBindingKind::Samplers) {
        return DescriptorRole::GuestSamplers;
    }
    if (kind == DescriptorBindingKind::Gds) {
        return DescriptorRole::Gds;
    }
    if (kind == DescriptorBindingKind::BdaPagetable) {
        return DescriptorRole::BdaPagetable;
    }
    if (kind == DescriptorBindingKind::FaultBuffer) {
        return DescriptorRole::FaultBuffer;
    }
    if (kind == DescriptorBindingKind::FlattenedSrt) {
        return DescriptorRole::FlattenedSrt;
    }
    if (kind == DescriptorBindingKind::ShaderData) {
        return DescriptorRole::ShaderData;
    }
    if (ImageBindingResourceClass(kind) != ImageResourceClass::None) {
        return DescriptorRole::GuestImages;
    }
    fail("DescriptorBindingBuilder::Populate binding kind has no descriptor role");
}

DescriptorImageShape ImageShapeForResource(const ImageResource& image) {
    const RdnaImageDimensionInfo& info = RdnaImageDimensionInfoFor(image.dimension);
    if (info.multisampled != 0u) {
        fail("DescriptorBindingBuilder::Populate multisampled image resources have no descriptor image shape");
    }
    if (info.spirvDimension == spv::Dim1D) {
        return info.arrayed != 0u ? DescriptorImageShape::Image1DArray : DescriptorImageShape::Image1D;
    }
    if (info.spirvDimension == spv::Dim3D) {
        return DescriptorImageShape::Image3D;
    }
    if (info.spirvDimension == spv::Dim2D) {
        // Cube images are declared and addressed as 2D arrays of faces (the backend converts
        // cube coordinates to face layers), so they bind as 2D arrays.
        if (image.cube && info.arrayed == 0u) {
            fail("DescriptorBindingBuilder::Populate cube image resource is not arrayed");
        }
        return info.arrayed != 0u ? DescriptorImageShape::Image2DArray : DescriptorImageShape::Image2D;
    }
    fail("DescriptorBindingBuilder::Populate image resource dimension has no descriptor image shape");
}

std::vector<std::uint32_t> GuestBuffersDescriptor(const std::vector<std::uint32_t>& resources, const ResourceSnapshot& snapshot) {
    std::vector<std::uint32_t> result;
    result.reserve(resources.size() * 4u);
    for (const std::uint32_t r : resources) {
        if (r >= snapshot.buffers.size()) {
            fail("DescriptorBindingBuilder::Populate guest buffer index is out of range");
        }
        const DescriptorValue& value = snapshot.buffers[r];
        if (value.dwordCount != 4u) {
            fail("DescriptorBindingBuilder::Populate guest buffer descriptor has an invalid width");
        }
        for (std::uint32_t dword = 0; dword < 4u; dword++) {
            result.push_back(value.dwords[dword]);
        }
    }
    return result;
}

std::vector<std::uint32_t> GuestSamplersDescriptor(const std::vector<std::uint32_t>& resources, const ResourceSnapshot& snapshot) {
    std::vector<std::uint32_t> result;
    std::uint32_t dwordCount = 0;
    for (std::size_t i = 0; i < resources.size(); i++) {
        const std::uint32_t r = resources[i];
        if (r >= snapshot.samplers.size()) {
            fail("DescriptorBindingBuilder::Populate guest sampler index is out of range");
        }
        const DescriptorValue& value = snapshot.samplers[r];
        if (value.dwordCount == 0u) {
            fail("DescriptorBindingBuilder::Populate guest sampler descriptor is empty");
        }
        if (i == 0u) {
            dwordCount = value.dwordCount;
        } else if (value.dwordCount != dwordCount) {
            fail("DescriptorBindingBuilder::Populate guest sampler descriptors have inconsistent widths");
        }
        for (std::uint32_t dword = 0; dword < value.dwordCount; dword++) {
            result.push_back(value.dwords[dword]);
        }
    }
    return result;
}

std::vector<std::uint32_t> ShaderDataDwordsFor(const IrBindingLayout& layout, const ShaderInfo& info, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint32_t> imageModes, std::span<const std::uint8_t> exportMappings) {

    RuntimeAbi::ShaderData data{};
    data.version = RuntimeAbi::Version;
    if (!exportMappings.empty() && exportMappings.size() != data.exportMappings.size()) fail("invalid runtime export mapping count");
    std::copy(exportMappings.begin(), exportMappings.end(), data.exportMappings.begin());
    if (snapshot.images.size() > data.images.size() || snapshot.samplers.size() > data.samplers.size()) fail("runtime resource metadata capacity exceeded");
    data.imageCount = static_cast<std::uint32_t>(snapshot.images.size());
    data.samplerCount = static_cast<std::uint32_t>(snapshot.samplers.size());
    for (std::size_t i = 0; i < layout.userDataRegisters.size(); i++) {
        const std::uint32_t reg = layout.userDataRegisters[i];
        if (reg < userDataBase || reg - userDataBase >= snapshot.userData.size()) {
            fail("DescriptorBindingBuilder::Populate user-data register is out of range");
        }
        if (reg >= data.userData.size()) fail("runtime user-data capacity exceeded");
        data.userData[reg] = snapshot.userData[reg - userDataBase];
    }
    if (layout.dispatchThreadLimit) {
        if (partialThreads == std::array<std::uint32_t, 3>{}) {
            fail("DescriptorBindingBuilder::Populate partial-group shader has no dispatch size");
        }
        std::copy(partialThreads.begin(), partialThreads.end(), data.dispatchThreadLimit.begin());
    }
    for (const auto& binding : layout.descriptors) {
        const bool image = ImageBindingResourceClass(binding.kind) != ImageResourceClass::None;
        if (!image && binding.kind != DescriptorBindingKind::Samplers) continue;
        if (binding.resources.size() > RuntimeAbi::HeapCapacity(binding.kind)) fail("runtime typed heap capacity exceeded");
        for (std::size_t element = 0; element < binding.resources.size(); ++element) {
            const auto resource = binding.resources[element];
            auto& metadata = image ? data.images.at(resource) : data.samplers.at(resource);
            const auto& descriptor = image ? snapshot.images.at(resource) : snapshot.samplers.at(resource);
            std::uint32_t modeIndex = 0u;
            if (image) {
                if (resource >= imageModes.size()) fail("runtime image index exceeds the static capacity");
                modeIndex = imageModes[resource];
                const auto& mode = info.runtimeImageModes.at(resource).at(modeIndex);
                if (DescriptorBindingForImage(mode) != binding.kind) continue;
            }
            if (metadata.elementCount == 0u) {
                metadata.binding = static_cast<std::uint32_t>(binding.kind);
                metadata.firstElement = static_cast<std::uint32_t>(element);
                if (descriptor.dwordCount > metadata.descriptor.size()) fail("runtime resource descriptor width exceeded");
                std::copy_n(descriptor.dwords.begin(), descriptor.dwordCount, metadata.descriptor.begin());
                metadata.flags = image && descriptor.dwords[0] == 0u && (descriptor.dwords[1] & 0xffu) == 0u ? 1u : 0u;
                metadata.flags |= modeIndex << 1u;
            }
            ++metadata.elementCount;
        }
    }
    std::vector<std::uint32_t> result(RuntimeAbi::ShaderDataDwords);
    std::memcpy(result.data(), &data, sizeof(data));
    return result;
}

}

std::uint32_t PointFilteredSamplerWord(std::uint32_t word0, std::uint32_t filter) {
    const bool reduced = ((word0 >> 29u) & 3u) != 0u;
    if (reduced && (((filter >> 20u) & 0xfu) != 0u || ((filter >> 26u) & 3u) == 2u)) {
        fail("DescriptorBindingBuilder: a min or max reduction sampler that filters between texels or mip levels samples an image that needs point filtering (sint, converted or depth-bits format), which is not implemented");
    }
    const bool mipmapped = ((filter >> 26u) & 3u) != 0u;
    return (filter & ~(0xffu << 20u)) | (1u << 24u) | (mipmapped ? 1u << 26u : 0u);
}

void DescriptorBindingBuilder::Populate(BindingAllocationResult& allocation, const IrProgram& program, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings) const {
    Populate(allocation, program.Info(), program.Resources().stage, program.Resources().userDataBase, snapshot, partialThreads, exportMappings);
}

void DescriptorBindingBuilder::Populate(BindingAllocationResult& allocation, const ShaderInfo& info, IrShaderStage stage, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings) const {
    if (stage == IrShaderStage::Pixel && exportMappings.size() != 8u) fail("fragment runtime export mappings are missing");
    const IrBindingLayout& layout = allocation.layout;
    if (info.images.size() > ShaderInfo::MaxImages) fail("runtime image count exceeds the static capacity");
    std::array<std::uint32_t, ShaderInfo::MaxImages> imageModes{};
    for (std::size_t index = 0; index < info.images.size(); ++index) imageModes[index] = ResourceMaterializer::RuntimeImageMode(info.images[index], snapshot.images.at(index), info.runtimeImageModes.at(index));
    allocation.specialization.clear();
    for (std::uint32_t index = 0; index < info.buffers.size(); ++index) {
        const auto& descriptor = snapshot.buffers.at(index);
        if (descriptor.dwordCount != 4u) fail("buffer specialization requires four descriptor words");
        const auto& buffer = info.buffers[index];
        const auto format = (descriptor.dwords[3] >> 12u) & 0x7fu;
        const bool present = descriptor.dwords[0] != 0u || (descriptor.dwords[1] & 0xffffu) != 0u;
        if (buffer.descriptorFormatted && (format > static_cast<std::uint32_t>(IrBufferFormat::Format32_32_32_32Float) || (present && format == 0u))) fail("buffer specialization has an unsupported format");
        const auto first = PipelineSpecialization::BufferBase + index * PipelineSpecialization::BufferWords;
        allocation.specialization.push_back({first, descriptor.dwords[1] & 0xffff0000u});
        allocation.specialization.push_back({first + 1u, descriptor.dwords[3]});
        allocation.specialization.push_back({first + 2u, present ? 1u : 0u});
        if (buffer.formattedReadMask != 0u) {
            for (std::uint32_t component = 0; component < 4u; ++component) {
                if ((buffer.formattedReadMask & (1u << component)) == 0u) continue;
                const auto selector = (descriptor.dwords[3] >> (component * 3u)) & 7u;
                if (selector == 2u || selector == 3u) fail("buffer specialization has a reserved component selector");
            }
        }
    }
    for (std::uint32_t index = 0; index < info.images.size(); ++index) {
        if (info.images[index].indirectRoot != ImageResource::NoIndirectImage) continue;
        const auto first = PipelineSpecialization::ImageBase + index * PipelineSpecialization::ImageWords;
        allocation.specialization.push_back({first, imageModes[index]});
        for (std::uint32_t component = 0; component < 4u; ++component) allocation.specialization.push_back({first + 1u + component, (snapshot.images.at(index).dwords[3] >> (component * 3u)) & 7u});
    }
    const std::vector<std::uint32_t> shaderData = ShaderDataDwordsFor(layout, info, userDataBase, snapshot, partialThreads, imageModes, exportMappings);
    const UnnormalizedProof unnormalized = ProveUnnormalized(info, snapshot);
    const std::vector<std::uint32_t> samplerElements = SamplerElements(layout, info);

    std::vector<DescriptorBinding> bindings;
    bindings.reserve(layout.descriptors.size());
    std::size_t writtenHere = 0;
    std::size_t readOnlyHere = 0;
    for (const IrDescriptorBinding& logical : layout.descriptors) {
        DescriptorBinding physical;
        physical.descriptorSet = RuntimeAbi::DescriptorSet;
        physical.binding = NativeBinding(stage, logical.kind);
        physical.count = logical.resources.empty() ? 1u : static_cast<std::uint32_t>(logical.resources.size());
        physical.kind = PhysicalKindFor(logical.kind);
        physical.role = RoleFor(logical.kind);
        physical.readOnly = false;

        switch (physical.role) {
        case DescriptorRole::GuestBuffers:
            physical.guestDescriptor = GuestBuffersDescriptor(logical.resources, snapshot);
            for (const std::uint32_t resource : logical.resources) {
                const auto& buffer = info.buffers.at(resource);
                physical.bufferAtomic.push_back(buffer.atomic);
                // The tracker merges every buffer access of a source into its resource
                // (ResourceTracker::Merge), so an element without a store or atomic is read-only
                // over its whole extent; stores through pointers (BDA) never bind a V#.
                physical.bufferWritten.push_back(buffer.written || buffer.atomic);
                if (buffer.written || buffer.atomic) ++writtenHere;
                else ++readOnlyHere;
            }
            break;
        case DescriptorRole::GuestImages: {
            const auto& modes = info.runtimeImageModes.at(logical.resources.front());
            const auto shape = std::ranges::find_if(modes, [&](const ImageResource& mode) { return DescriptorBindingForImage(mode) == logical.kind; });
            if (shape == modes.end()) fail("runtime image heap has no static image type");
            physical.imageShape = ImageShapeForResource(*shape);
            std::uint32_t previous = UINT32_MAX;
            std::uint32_t mip = 0u;
            for (const std::uint32_t resource : logical.resources) {
                const auto& image = info.images.at(resource);
                const auto& descriptor = snapshot.images.at(resource);
                const auto& mode = info.runtimeImageModes.at(resource).at(imageModes[resource]);
                mip = resource == previous ? mip + 1u : 0u;
                previous = resource;
                const auto firstMip = (descriptor.dwords[3] >> 12u) & 0xfu;
                const auto lastMip = (descriptor.dwords[3] >> 16u) & 0xfu;
                const bool active = DescriptorBindingForImage(mode) == logical.kind && mode.packedFormat != IrBufferFormat::Fmask8_S2_F1 && (image.mipMode != ImageMipMode::DynamicStorage || mip <= lastMip - firstMip);
                for (std::uint32_t dword = 0u; dword < 8u; ++dword) {
                    auto value = descriptor.dwords[dword];
                    if (dword == 3u && (mode.conversionFormat != IrBufferFormat::Invalid || mode.depthBits)) value = (value & ~0xfffu) | ShaderImageIdentitySwizzle;
                    physical.guestDescriptor.push_back(active ? value : 0u);
                }
                physical.imageWritten.push_back(image.written || image.atomic);
                physical.imageDepthCompare.push_back(image.depthCompare);
                physical.imageAtomic.push_back(image.atomic);
                physical.imageAtomic64.push_back(image.atomic64);
                physical.imageUnnormalized.push_back(unnormalized.images.at(resource));
                physical.imageSamplers.push_back(ImageSamplerMask(info, samplerElements, resource));
            }
            break;
        }
        case DescriptorRole::GuestSamplers:
            physical.guestDescriptor = GuestSamplersDescriptor(logical.resources, snapshot);
            for (std::size_t element = 0; element < logical.resources.size(); ++element) {
                const auto& sampler = info.samplers.at(logical.resources[element]);
                physical.samplerDepthCompare.push_back(sampler.depthCompare);
                physical.samplerUnnormalized.push_back(unnormalized.samplers.at(logical.resources[element]));
                if ((element & 1u) != 0u) {
                    auto& filter = physical.guestDescriptor.at(element * 4u + 2u);
                    filter = PointFilteredSamplerWord(physical.guestDescriptor.at(element * 4u), filter);
                }
            }
            break;
        case DescriptorRole::FlattenedSrt:
            if (snapshot.flattenedSrt.empty()) {
                fail("DescriptorBindingBuilder::Populate flattened SRT snapshot is empty");
            }
            physical.guestDescriptor = snapshot.flattenedSrt;
            break;
        case DescriptorRole::ShaderData:
            if (layout.UsesPushData()) {
                fail("DescriptorBindingBuilder::Populate shader-data binding must not exist when push data is used");
            }
            physical.guestDescriptor = shaderData;
            break;
        case DescriptorRole::Gds:
        case DescriptorRole::BdaPagetable:
        case DescriptorRole::FaultBuffer:
            break;
        }

        if (physical.role == DescriptorRole::GuestBuffers || physical.role == DescriptorRole::GuestImages || physical.role == DescriptorRole::GuestSamplers) {
            if (physical.count == 0u || physical.guestDescriptor.size() % physical.count != 0u) {
                fail("DescriptorBindingBuilder::Populate guest descriptor size is not a multiple of the binding count");
            }
        }

        bindings.push_back(std::move(physical));
    }

    if (bufferWrittenTraceEnabled()) {
        // Cumulative counts include every Populate of the process (the replay tool populates
        // each program twice), so the per-call counts are the ones to sum per program.
        const auto written = bufferWrittenCounts.written.fetch_add(writtenHere) + writtenHere;
        const auto readOnly = bufferWrittenCounts.readOnly.fetch_add(readOnlyHere) + readOnlyHere;
        std::fprintf(stderr, "[bindings] guest buffer elements: %zu written, %zu read-only (total so far: %llu / %llu)\n", writtenHere, readOnlyHere, written, readOnly);
    }
    allocation.bindings = std::move(bindings);
    allocation.pushConstants.clear();
    if (layout.UsesPushData()) {
        allocation.pushConstants.resize(static_cast<std::size_t>(shaderData.size()) * sizeof(std::uint32_t));
        std::memcpy(allocation.pushConstants.data(), shaderData.data(), allocation.pushConstants.size());
    }
}

}
