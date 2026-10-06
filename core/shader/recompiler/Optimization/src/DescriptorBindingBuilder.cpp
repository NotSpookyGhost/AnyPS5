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

std::vector<std::uint32_t> ShaderDataDwordsFor(const IrBindingLayout& layout, const ShaderInfo& info, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint32_t> imageModes, const std::vector<IrDescriptorBinding>& selected) {
    std::vector<std::uint32_t> result(layout.ShaderDataDwords(), 0u);
    for (std::size_t i = 0; i < layout.userDataRegisters.size(); ++i) {
        const auto reg = layout.userDataRegisters[i];
        if (reg < userDataBase || reg - userDataBase >= snapshot.userData.size()) fail("shader user-data register is out of range");
        result[i] = snapshot.userData[reg - userDataBase];
    }
    if (layout.dispatchThreadLimit) {
        if (partialThreads == std::array<std::uint32_t, 3>{}) fail("partial-group shader has no dispatch size");
        std::copy(partialThreads.begin(), partialThreads.end(), result.begin() + layout.DispatchThreadLimitDword());
    }
    if (layout.runtimeImageCount == 0u) return result;
    for (const auto& binding : selected) {
        if (ImageBindingResourceClass(binding.kind) == ImageResourceClass::None) continue;
        for (std::size_t element = 0; element < binding.resources.size(); ++element) {
            const auto resource = binding.resources[element];
            if (info.images.at(resource).indirectRoot == ImageResource::NoIndirectImage) continue;
            const auto& mode = info.runtimeImageModes.at(resource).at(imageModes[resource]);
            if (DescriptorBindingForImage(mode) != binding.kind) continue;
            RuntimeAbi::ResourceMetadata metadata{};
            metadata.binding = static_cast<std::uint32_t>(binding.kind);
            metadata.firstElement = static_cast<std::uint32_t>(element);
            metadata.elementCount = 1u;
            const auto& descriptor = snapshot.images.at(resource);
            if (descriptor.dwordCount != metadata.descriptor.size()) fail("invalid runtime image descriptor width");
            std::copy_n(descriptor.dwords.begin(), metadata.descriptor.size(), metadata.descriptor.begin());
            metadata.flags = imageModes[resource] << 1u;
            if (descriptor.dwords[0] == 0u && (descriptor.dwords[1] & 0xffu) == 0u) metadata.flags |= 1u;
            if (resource >= layout.runtimeImageCount) fail("runtime image metadata exceeds compact layout");
            std::memcpy(result.data() + layout.ImageMetadataDword() + resource * (sizeof(metadata) / sizeof(std::uint32_t)), &metadata, sizeof(metadata));
        }
    }
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
    auto plan = Prepare(allocation.layout, info, stage, snapshot, exportMappings);
    Populate(allocation, allocation, plan, info, stage, userDataBase, snapshot, partialThreads);
    allocation.specialization = std::move(plan.specialization);
}

DescriptorBindingPlan DescriptorBindingBuilder::Prepare(const IrBindingLayout& layout, const ShaderInfo& info, IrShaderStage stage, const ResourceSnapshot& snapshot, std::span<const std::uint8_t> exportMappings) const {
    if (stage == IrShaderStage::Pixel && exportMappings.size() != 8u) fail("fragment runtime export mappings are missing");
    if (info.images.size() > ShaderInfo::MaxImages) fail("runtime image count exceeds the static capacity");
    DescriptorBindingPlan plan;
    auto& imageModes = plan.imageModes;
    imageModes.resize(info.images.size());
    for (std::size_t index = 0; index < info.images.size(); ++index) imageModes[index] = ResourceMaterializer::RuntimeImageMode(info.images[index], snapshot.images.at(index), info.runtimeImageModes.at(index));
    for (std::uint32_t index = 0; index < info.buffers.size(); ++index) {
        const auto& descriptor = snapshot.buffers.at(index);
        if (descriptor.dwordCount != 4u) fail("buffer specialization requires four descriptor words");
        const auto& buffer = info.buffers[index];
        const auto format = (descriptor.dwords[3] >> 12u) & 0x7fu;
        const bool present = descriptor.dwords[0] != 0u || (descriptor.dwords[1] & 0xffffu) != 0u;
        if (buffer.descriptorFormatted && (format > static_cast<std::uint32_t>(IrBufferFormat::Format32_32_32_32Float) || (present && format == 0u))) fail("buffer specialization has an unsupported format");
        const auto first = PipelineSpecialization::BufferBase + index * PipelineSpecialization::BufferWords;
        plan.specialization.push_back({first, descriptor.dwords[1] & 0xffff0000u});
        plan.specialization.push_back({first + 1u, descriptor.dwords[3]});
        plan.specialization.push_back({first + 2u, present ? 1u : 0u});
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
        plan.specialization.push_back({first, imageModes[index]});
        for (std::uint32_t component = 0; component < 4u; ++component) plan.specialization.push_back({first + 1u + component, (snapshot.images.at(index).dwords[3] >> (component * 3u)) & 7u});
    }
    for (std::uint32_t target = 0; target < exportMappings.size(); ++target) {
        for (std::uint32_t component = 0; component < 4u; ++component) plan.specialization.push_back({PipelineSpecialization::ExportBase + target * 4u + component, (exportMappings[target] >> (component * 2u)) & 3u});
    }
    std::vector<std::uint32_t> samplerModes(info.samplers.size(), 0u);
    const auto samplerMode = [](const ImageResource& image) { return image.numericClass == IrTextureNumericClass::Sint || image.conversionFormat != IrBufferFormat::Invalid || image.depthBits ? 2u : 1u; };
    for (const auto& pair : info.sampledPairs) {
        const auto& image = info.images.at(pair.image);
        if (image.indirectRoot == ImageResource::NoIndirectImage) samplerModes.at(pair.sampler) |= samplerMode(info.runtimeImageModes.at(pair.image).at(imageModes[pair.image]));
        else {
            const auto& root = info.images.at(image.indirectRoot);
            for (const auto slot : root.indirectResources) samplerModes.at(pair.sampler) |= samplerMode(info.runtimeImageModes.at(slot).at(imageModes[slot]));
        }
    }
    for (std::uint32_t index = 0; index < info.images.size(); ++index) {
        if (info.images[index].mipMode != ImageMipMode::DynamicStorage) continue;
        const auto word = snapshot.images.at(index).dwords[3];
        const auto first = (word >> 12u) & 0xfu;
        const auto last = (word >> 16u) & 0xfu;
        if (last < first || last - first >= RuntimeAbi::StorageHeapCapacity) fail("invalid dynamic storage mip range");
        plan.specialization.push_back({PipelineSpecialization::MipCountBase + index, last - first + 1u});
    }
    for (std::uint32_t resource = 0; resource < info.images.size(); ++resource) {
        const auto& image = info.images[resource];
        if (image.indirectRoot == ImageResource::NoIndirectImage) continue;
        const auto& modes = info.runtimeImageModes.at(resource);
        if (modes.size() > PipelineSpecialization::ImageModeStride) fail("runtime image mode specialization capacity exceeded");
        std::vector<bool> active(modes.size(), false);
        for (const auto slot : info.images.at(image.indirectRoot).indirectResources) {
            const auto mode = ResourceMaterializer::RuntimeImageMode(image, snapshot.images.at(slot), modes);
            if (mode != imageModes[slot]) fail("indirect image mode numbering disagrees with its root");
            active[mode] = true;
        }
        for (std::uint32_t mode = 0; mode < modes.size(); ++mode) plan.specialization.push_back({PipelineSpecialization::ImageModeBase + resource * PipelineSpecialization::ImageModeStride + mode, active[mode] ? 1u : 0u});
    }
    auto& selected = plan.selected;
    auto& originalElements = plan.originalElements;
    selected.reserve(layout.descriptors.size());
    for (const auto& logical : layout.descriptors) {
        const bool imageHeap = ImageBindingResourceClass(logical.kind) != ImageResourceClass::None;
        const bool samplerHeap = logical.kind == DescriptorBindingKind::Samplers;
        IrDescriptorBinding compact{logical.kind, {}};
        std::vector<std::uint32_t> originals;
        std::uint32_t previous = UINT32_MAX;
        std::uint32_t mip = 0;
        for (std::uint32_t element = 0; element < logical.resources.size(); ++element) {
            const auto resource = logical.resources[element];
            bool active = true;
            if (imageHeap) {
                const auto& image = info.images.at(resource);
                const auto& mode = info.runtimeImageModes.at(resource).at(imageModes[resource]);
                mip = previous == resource ? mip + 1u : 0u;
                previous = resource;
                {
                    const auto word = snapshot.images.at(resource).dwords[3];
                    active = DescriptorBindingForImage(mode) == logical.kind && mode.packedFormat != IrBufferFormat::Fmask8_S2_F1 && (image.mipMode != ImageMipMode::DynamicStorage || mip <= ((word >> 16u) & 0xfu) - ((word >> 12u) & 0xfu));
                }
            } else if (samplerHeap) {
                active = (samplerModes.at(resource) & (1u << (element & 1u))) != 0u;
            }
            if (imageHeap || samplerHeap) plan.specialization.push_back({PipelineSpecialization::DescriptorIndex(static_cast<std::uint32_t>(logical.kind), element), active ? static_cast<std::uint32_t>(compact.resources.size()) : 0u});
            if (active) {
                compact.resources.push_back(resource);
                originals.push_back(element);
            }
        }
        if (imageHeap || samplerHeap) plan.specialization.push_back({PipelineSpecialization::HeapCountBase + static_cast<std::uint32_t>(logical.kind), std::max(1u, static_cast<std::uint32_t>(compact.resources.size()))});
        if ((imageHeap || samplerHeap) && compact.resources.empty()) continue;
        selected.push_back(std::move(compact));
        originalElements.push_back(std::move(originals));
    }
    return plan;
}

void DescriptorBindingBuilder::Populate(BindingAllocationResult& allocation, const CompiledBindingLayout& compiled, const DescriptorBindingPlan& plan, const ShaderInfo& info, IrShaderStage stage, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads) const {
    const auto& layout = compiled.layout;
    const auto& imageModes = plan.imageModes;
    const auto& selected = plan.selected;
    const auto& originalElements = plan.originalElements;
    const std::vector<std::uint32_t> shaderData = ShaderDataDwordsFor(layout, info, userDataBase, snapshot, partialThreads, imageModes, selected);
    const UnnormalizedProof unnormalized = ProveUnnormalized(info, snapshot);
    const std::vector<std::uint32_t> samplerElements = SamplerElements(layout, info);

    std::vector<DescriptorBinding> bindings;
    bindings.reserve(layout.descriptors.size());
    std::size_t writtenHere = 0;
    std::size_t readOnlyHere = 0;
    for (std::size_t bindingIndex = 0; bindingIndex < selected.size(); ++bindingIndex) {
        const auto& logical = selected[bindingIndex];
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
                if ((originalElements[bindingIndex][element] & 1u) != 0u) {
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
