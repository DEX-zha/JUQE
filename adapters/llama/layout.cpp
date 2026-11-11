#include "state.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
void validate_matrix(const juqe_part_info &part)
{
    if (part.kind >= GGML_TYPE_COUNT || part.dimensions[0] > INT64_MAX ||
        part.dimensions[1] > INT64_MAX)
    {
        throw std::invalid_argument("JUQE: matrix type or dimensions are unsupported");
    }
    const auto type = static_cast<ggml_type>(part.kind);
    const int64_t block = ggml_blck_size(type);
    const size_t type_bytes = ggml_type_size(type);
    if (block <= 0 || !type_bytes || part.dimensions[0] % block != 0)
    {
        throw std::invalid_argument("JUQE: matrix dimensions do not match quantization blocks");
    }
    const uint64_t blocks = part.dimensions[0] / block;
    if (blocks > part.bytes / type_bytes)
    {
        throw std::invalid_argument("JUQE: matrix row exceeds its payload");
    }
    const uint64_t row = blocks * type_bytes;
    if (!row || part.bytes % row || part.bytes / row != part.dimensions[1])
    {
        throw std::invalid_argument("JUQE: matrix payload does not match its dimensions");
    }
}
bool zero_router(const llama_model &model, uint32_t id)
{
    const auto *router = model.layers[id].ffn_gate_inp;
    if (model.arch != LLM_ARCH_QWEN4EXP || !router || router->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(router))
    {
        return false;
    }
    std::vector<float> values(ggml_nelements(router));
    ggml_backend_tensor_get(router, values.data(), 0, ggml_nbytes(router));
    return std::all_of(values.begin(), values.end(), [](float value) { return value == 0.0f; });
}
} // namespace

namespace juqe
{
void describe_layers(llama_model &model, juqe_offload &state, const juqe_pack *pack, bool meta_only)
{
    juqe_pack_info_t counts{};
    check(juqe_pack_info(pack, &counts));
    if (counts.layers != model.hparams.n_layer() || counts.experts != model.hparams.n_expert)
    {
        throw std::invalid_argument("JUQE: pack layer/expert counts do not match the model");
    }
    state.experts = counts.experts;
    state.layers.resize(counts.layers);
    for (uint32_t id = 0; id < counts.layers; ++id)
    {
        auto &layer = state.layers[id];
        check(juqe_pack_layer(pack, id, &layer.layout));
        if (!layer.layout.stride)
        {
            if (model.layers[id].ffn_gate_inp)
            {
                throw std::invalid_argument("JUQE: dense pack entry for a routed model layer");
            }
            continue;
        }
        const uint32_t selected = model.hparams.n_expert_used(id);
        if (!model.layers[id].ffn_gate_inp || !selected || selected > state.experts ||
            (state.selected && selected != state.selected))
        {
            throw std::invalid_argument("JUQE: unsupported routed-layer selection count");
        }
        state.selected = selected;
        for (const auto &part : layer.layout.parts)
        {
            validate_matrix(part);
        }
        const auto &gate = layer.layout.parts[0];
        const auto &up = layer.layout.parts[1];
        const auto &down = layer.layout.parts[2];
        if (gate.dimensions[0] != model.hparams.n_embd || gate.dimensions[0] != up.dimensions[0] ||
            gate.dimensions[1] != up.dimensions[1] || down.dimensions[0] != gate.dimensions[1] ||
            down.dimensions[1] != gate.dimensions[0])
        {
            throw std::invalid_argument("JUQE: expert matrix shapes do not match the model");
        }
        const auto requested =
            static_cast<uint32_t>(std::lround(state.config.cache_fraction * state.experts));
        layer.pool.slots = std::min(state.experts, std::max(selected + 2, requested));
        if (!meta_only && state.config.compact_fixed && zero_router(model, id))
        {
            layer.fixed_router = true;
            layer.pool.slots = std::min(layer.pool.slots, selected + 2);
        }
    }
    if (!state.selected)
    {
        throw std::invalid_argument("JUQE: model has no routed expert layers");
    }
}
} // namespace juqe
