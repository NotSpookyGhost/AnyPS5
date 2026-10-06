#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>
#undef SPV_ENABLE_UTILITY_CODE
#include "SpirvBackend/SpirvSpecialization.hpp"
#include <algorithm>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace ShaderRecompiler {
namespace {

using Instruction = std::vector<std::uint32_t>;

spv::Op Opcode(const Instruction& instruction) {
    return instruction.empty() ? spv::OpNop : static_cast<spv::Op>(instruction[0] & 0xffffu);
}

std::uint32_t Result(const Instruction& instruction) {
    if (instruction.empty()) return 0u;
    bool result = false;
    bool type = false;
    spv::HasResultAndType(Opcode(instruction), &result, &type);
    return result ? instruction.at(type ? 2u : 1u) : 0u;
}

Instruction Make(spv::Op op, std::initializer_list<std::uint32_t> operands) {
    Instruction result{(static_cast<std::uint32_t>(operands.size() + 1u) << 16u) | op};
    result.insert(result.end(), operands);
    return result;
}

struct ScalarType {
    std::uint32_t width;
    bool boolean;
};

class Specialization {
public:
    explicit Specialization(std::span<const std::uint32_t> words) {
        if (words.size() < 5u || words[0] != spv::MagicNumber) throw std::runtime_error("invalid prepared SPIR-V header");
        header.assign(words.begin(), words.begin() + 5);
        for (std::size_t cursor = 5; cursor < words.size();) {
            const auto count = words[cursor] >> 16u;
            if (count == 0u || count > words.size() - cursor) throw std::runtime_error("truncated prepared SPIR-V instruction");
            instructions.emplace_back(words.begin() + cursor, words.begin() + cursor + count);
            const auto& instruction = instructions.back();
            const auto op = Opcode(instruction);
            bool hasResult = false;
            bool hasType = false;
            spv::HasResultAndType(op, &hasResult, &hasType);
            if (hasResult && hasType) resultTypes.emplace(instruction.at(2), instruction.at(1));
            if (op == spv::OpTypeBool) types.emplace(instruction.at(1), ScalarType{1u, true});
            if (op == spv::OpTypeInt) types.emplace(instruction.at(1), ScalarType{instruction.at(2), false});
            if (op == spv::OpConstant && types.contains(instruction.at(1)) && types.at(instruction[1]).width <= 32u) values.emplace(instruction.at(2), instruction.at(3));
            if (op == spv::OpConstantTrue || op == spv::OpConstantFalse) values.emplace(instruction.at(2), op == spv::OpConstantTrue ? 1u : 0u);
            cursor += count;
        }
    }

    std::vector<std::uint32_t> Run() {
        bool changed = true;
        while (changed) {
            changed = fold();
            changed |= prune();
        }
        removeUnusedExtracts();
        orderPhis();
        std::vector<std::uint32_t> words = header;
        bool inserted = false;
        for (const auto& instruction : instructions) {
            if (instruction.empty()) continue;
            if (!inserted && Opcode(instruction) == spv::OpFunction) {
                for (const auto& constant : constants) words.insert(words.end(), constant.begin(), constant.end());
                inserted = true;
            }
            if ((Opcode(instruction) == spv::OpName || Opcode(instruction) == spv::OpDecorate) && removed.contains(instruction.at(1))) continue;
            words.insert(words.end(), instruction.begin(), instruction.end());
        }
        return words;
    }

private:
    std::optional<std::uint32_t> value(std::uint32_t id) const {
        const auto found = values.find(id);
        return found != values.end() ? std::optional(found->second) : std::nullopt;
    }

    std::optional<std::uint32_t> evaluate(const Instruction& instruction) const {
        const auto op = Opcode(instruction);
        if (instruction.size() < 4u) return std::nullopt;
        const auto type = types.find(instruction[1]);
        if (type == types.end() || type->second.width > 32u) return std::nullopt;
        const auto left = value(instruction[3]);
        if (!left) return std::nullopt;
        if (op == spv::OpCopyObject) return left;
        if (op == spv::OpLogicalNot) return *left == 0u;
        if (op == spv::OpNot) return ~*left;
        if (instruction.size() < 5u) return std::nullopt;
        const auto right = value(instruction[4]);
        if (!right) return std::nullopt;
        switch (op) {
        case spv::OpIAdd: return *left + *right;
        case spv::OpISub: return *left - *right;
        case spv::OpIMul: return *left * *right;
        case spv::OpUDiv: return *right != 0u ? std::optional(*left / *right) : std::nullopt;
        case spv::OpUMod: return *right != 0u ? std::optional(*left % *right) : std::nullopt;
        case spv::OpBitwiseAnd: return *left & *right;
        case spv::OpBitwiseOr: return *left | *right;
        case spv::OpBitwiseXor: return *left ^ *right;
        case spv::OpShiftRightLogical: return *right < type->second.width ? std::optional(*left >> *right) : std::nullopt;
        case spv::OpShiftLeftLogical: return *right < type->second.width ? std::optional(*left << *right) : std::nullopt;
        case spv::OpIEqual: return *left == *right;
        case spv::OpINotEqual: return *left != *right;
        case spv::OpULessThan: return *left < *right;
        case spv::OpULessThanEqual: return *left <= *right;
        case spv::OpUGreaterThan: return *left > *right;
        case spv::OpUGreaterThanEqual: return *left >= *right;
        case spv::OpLogicalEqual: return (*left != 0u) == (*right != 0u);
        case spv::OpLogicalNotEqual: return (*left != 0u) != (*right != 0u);
        case spv::OpLogicalAnd: return *left != 0u && *right != 0u;
        case spv::OpLogicalOr: return *left != 0u || *right != 0u;
        case spv::OpBitFieldUExtract: {
            if (instruction.size() != 6u) throw std::runtime_error("invalid prepared bitfield instruction");
            const auto count = value(instruction[5]);
            if (!count || *right > 32u || *count > 32u - *right) return std::nullopt;
            if (*count == 0u) return 0u;
            const auto mask = *count == 32u ? UINT32_MAX : (1u << *count) - 1u;
            return (*left >> *right) & mask;
        }
        default: return std::nullopt;
        }
    }

    bool fold() {
        bool changed = false;
        bool function = false;
        std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> extracts;
        for (auto& instruction : instructions) {
            if (instruction.empty()) continue;
            auto op = Opcode(instruction);
            if (op == spv::OpFunction) function = true;
            if (op == spv::OpFunctionEnd) function = false;
            if (!function) continue;
            if (op == spv::OpSelect && instruction.size() == 6u) {
                if (const auto condition = value(instruction[3])) {
                    instruction = Make(spv::OpCopyObject, {instruction[1], instruction[2], instruction[*condition != 0u ? 4u : 5u]});
                    changed = true;
                }
            }
            if (op == spv::OpVectorExtractDynamic && instruction.size() == 5u) {
                if (const auto index = value(instruction[4])) {
                    instruction = Make(spv::OpCompositeExtract, {instruction[1], instruction[2], instruction[3], *index});
                    changed = true;
                }
            }
            op = Opcode(instruction);
            if (op == spv::OpCompositeExtract && instruction.size() == 5u) extracts.emplace(instruction[2], std::pair{instruction[3], instruction[4]});
            if (op == spv::OpCompositeConstruct && instruction.size() == 7u) {
                const auto first = extracts.find(instruction[3]);
                bool shuffle = first != extracts.end();
                for (std::size_t index = 4; shuffle && index < instruction.size(); ++index) {
                    const auto found = extracts.find(instruction[index]);
                    shuffle = found != extracts.end() && found->second.first == first->second.first;
                }
                if (shuffle) {
                    const auto source = first->second.first;
                    bool identity = resultTypes.contains(source) && resultTypes.at(source) == instruction[1];
                    Instruction replacement = Make(spv::OpVectorShuffle, {instruction[1], instruction[2], source, source});
                    for (std::size_t index = 3; index < instruction.size(); ++index) {
                        const auto channel = extracts.at(instruction[index]).second;
                        replacement.push_back(channel);
                        identity &= channel == index - 3u;
                    }
                    replacement[0] = (static_cast<std::uint32_t>(replacement.size()) << 16u) | spv::OpVectorShuffle;
                    instruction = identity ? Make(spv::OpCopyObject, {instruction[1], instruction[2], source}) : std::move(replacement);
                    changed = true;
                }
            }
            const auto result = evaluate(instruction);
            if (!result) continue;
            const auto type = types.at(instruction[1]);
            const auto bits = type.width == 32u ? *result : *result & ((1u << type.width) - 1u);
            values[instruction[2]] = bits;
            constants.push_back(type.boolean ? Make(bits != 0u ? spv::OpConstantTrue : spv::OpConstantFalse, {instruction[1], instruction[2]}) : Make(spv::OpConstant, {instruction[1], instruction[2], bits}));
            instruction.clear();
            changed = true;
        }
        return changed;
    }

    bool prune() {
        std::map<std::uint32_t, std::vector<std::size_t>> blocks;
        std::vector<std::uint32_t> entries;
        std::uint32_t block = 0;
        bool entry = false;
        for (std::size_t index = 0; index < instructions.size(); ++index) {
            const auto& instruction = instructions[index];
            if (instruction.empty()) continue;
            const auto op = Opcode(instruction);
            if (op == spv::OpFunction) entry = true;
            if (op == spv::OpFunctionEnd) block = 0;
            if (op == spv::OpLabel) {
                block = instruction.at(1);
                if (entry) entries.push_back(block);
                entry = false;
            }
            if (block != 0u) blocks[block].push_back(index);
        }
        bool changed = false;
        std::map<std::uint32_t, std::vector<std::uint32_t>> edges;
        for (const auto& [label, indices] : blocks) {
            auto& terminal = instructions[indices.back()];
            auto op = Opcode(terminal);
            std::uint32_t target = 0;
            bool loop = false;
            for (const auto index : indices) loop |= Opcode(instructions[index]) == spv::OpLoopMerge;
            if (!loop && op == spv::OpBranchConditional) {
                if (const auto condition = value(terminal.at(1))) target = terminal.at(*condition != 0u ? 2u : 3u);
            } else if (!loop && op == spv::OpSwitch) {
                if (const auto selector = value(terminal.at(1))) {
                    target = terminal.at(2);
                    if ((terminal.size() - 3u) % 2u != 0u) throw std::runtime_error("invalid prepared 32-bit switch");
                    for (std::size_t index = 3; index < terminal.size(); index += 2u) if (terminal[index] == *selector) target = terminal[index + 1u];
                }
            }
            if (target != 0u) {
                for (const auto index : indices) if (Opcode(instructions[index]) == spv::OpSelectionMerge) instructions[index].clear();
                terminal = Make(spv::OpBranch, {target});
                op = spv::OpBranch;
                changed = true;
            }
            auto& successors = edges[label];
            if (op == spv::OpBranch) successors.push_back(terminal.at(1));
            else if (op == spv::OpBranchConditional) successors = {terminal.at(2), terminal.at(3)};
            else if (op == spv::OpSwitch) {
                successors.push_back(terminal.at(2));
                const auto selectorType = resultTypes.find(terminal.at(1));
                const auto stride = selectorType != resultTypes.end() && types.contains(selectorType->second) && types.at(selectorType->second).width == 64u ? 3u : 2u;
                for (std::size_t index = 3; index + stride <= terminal.size(); index += stride) successors.push_back(terminal[index + stride - 1u]);
            }
        }
        std::set<std::uint32_t> live;
        auto pending = entries;
        while (!pending.empty()) {
            const auto label = pending.back();
            pending.pop_back();
            if (!blocks.contains(label)) throw std::runtime_error("prepared branch references a missing block");
            if (!live.insert(label).second) continue;
            const auto& successors = edges.at(label);
            pending.insert(pending.end(), successors.begin(), successors.end());
        }
        for (const auto& [label, indices] : blocks) {
            if (!live.contains(label)) {
                if (indices.size() == 2u && Opcode(instructions[indices.back()]) == spv::OpUnreachable) continue;
                for (std::size_t index = 1; index < indices.size(); ++index) {
                    auto& instruction = instructions[indices[index]];
                    if (const auto result = Result(instruction)) removed.insert(result);
                    instruction.clear();
                }
                instructions[indices.back()] = Make(spv::OpUnreachable, {});
                changed = true;
                continue;
            }
            for (const auto index : indices) {
                auto& instruction = instructions[index];
                if (instruction.empty() || Opcode(instruction) != spv::OpPhi) continue;
                Instruction phi(instruction.begin(), instruction.begin() + 3);
                for (std::size_t operand = 3; operand + 1u < instruction.size(); operand += 2u) {
                    const auto parent = instruction[operand + 1u];
                    if (live.contains(parent) && std::ranges::find(edges.at(parent), label) != edges.at(parent).end()) phi.insert(phi.end(), {instruction[operand], parent});
                }
                if (phi.size() == 3u) throw std::runtime_error("reachable prepared phi has no predecessor");
                if (phi.size() == instruction.size() && phi.size() != 5u) continue;
                phi[0] = (static_cast<std::uint32_t>(phi.size()) << 16u) | spv::OpPhi;
                instruction = phi.size() == 5u ? Make(spv::OpCopyObject, {phi[1], phi[2], phi[3]}) : std::move(phi);
                changed = true;
            }
        }
        return changed;
    }

    void orderPhis() {
        std::vector<Instruction> ordered;
        std::vector<Instruction> copies;
        bool prefix = false;
        for (auto& instruction : instructions) {
            if (instruction.empty()) continue;
            const auto op = Opcode(instruction);
            if (op == spv::OpLabel) prefix = true;
            else if (prefix && op == spv::OpCopyObject) {
                copies.push_back(std::move(instruction));
                continue;
            } else if (prefix && op != spv::OpPhi && op != spv::OpLine && op != spv::OpNoLine) {
                for (auto& copy : copies) ordered.push_back(std::move(copy));
                copies.clear();
                prefix = false;
            }
            ordered.push_back(std::move(instruction));
        }
        if (!copies.empty()) throw std::runtime_error("unterminated prepared phi block");
        instructions = std::move(ordered);
    }

    void removeUnusedExtracts() {
        std::set<std::uint32_t> referenced;
        for (const auto& instruction : instructions) {
            if (instruction.empty()) continue;
            const auto op = Opcode(instruction);
            if (op == spv::OpName || op == spv::OpDecorate) continue;
            bool hasResult = false;
            bool hasType = false;
            spv::HasResultAndType(op, &hasResult, &hasType);
            const auto resultIndex = hasResult ? (hasType ? 2u : 1u) : 0u;
            for (std::size_t index = 1; index < instruction.size(); ++index) if (index != resultIndex) referenced.insert(instruction[index]);
        }
        for (auto& instruction : instructions) {
            if (Opcode(instruction) != spv::OpCompositeExtract || referenced.contains(Result(instruction))) continue;
            removed.insert(Result(instruction));
            instruction.clear();
        }
    }

    std::vector<std::uint32_t> header;
    std::vector<Instruction> instructions;
    std::vector<Instruction> constants;
    std::map<std::uint32_t, ScalarType> types;
    std::map<std::uint32_t, std::uint32_t> values;
    std::map<std::uint32_t, std::uint32_t> resultTypes;
    std::set<std::uint32_t> removed;
};

}

std::vector<std::uint32_t> SpecializeSpirv(std::span<const std::uint32_t> words) {
    return Specialization(words).Run();
}

}
