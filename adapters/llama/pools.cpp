#include "state.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace
{
size_t add(size_t left, size_t right)
{
    if (right > std::numeric_limits<size_t>::max() - left)
    {
        throw std::overflow_error("JUQE: pool size overflow");
    }
    return left + right;
}
size_t align_up(size_t value, size_t alignment)
{
    if (!alignment)
    {
        throw std::invalid_argument("JUQE: invalid pool alignment");
    }
    return add(value, (alignment - value % alignment) % alignment);
}
ggml_backend_buffer_type_t host_type()
{
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index)
    {
        auto *device = ggml_backend_dev_get(index);
        const auto type = ggml_backend_dev_type(device);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU)
        {
            if (auto *host = ggml_backend_dev_host_buffer_type(device))
            {
                return host;
            }
        }
    }
    throw std::runtime_error("JUQE: no GPU host buffer type is available");
}
} // namespace

namespace juqe
{
void allocate_pools(juqe_offload &state, bool meta_only)
{
    auto *type = host_type();
    const size_t alignment = std::max<size_t>(256, ggml_backend_buft_get_alignment(type));
    size_t total = 0;
    for (auto &layer : state.layers)
    {
        if (!layer.layout.stride)
        {
            continue;
        }
        for (size_t part = 0; part < 3; ++part)
        {
            const uint64_t bytes =
                layer.layout.parts[part].bytes * (static_cast<uint64_t>(layer.pool.slots) + 1);
            if (bytes > std::numeric_limits<size_t>::max())
            {
                throw std::overflow_error("JUQE: pool exceeds address space");
            }
            layer.pool.parts[part].bytes = bytes;
            total = add(align_up(total, alignment), bytes);
        }
    }
    state.pool_bytes = total;
    const ggml_init_params parameters = {(3 * state.layers.size() + 8) * ggml_tensor_overhead(),
                                         nullptr, true};
    state.context.reset(ggml_init(parameters));
    if (!state.context)
    {
        throw std::runtime_error("JUQE: tensor metadata allocation failed");
    }
    uint8_t *base = nullptr;
    if (!meta_only)
    {
        state.buffer.reset(ggml_backend_buft_alloc_buffer(type, total));
        if (!state.buffer)
        {
            throw std::runtime_error("JUQE: pinned expert pool allocation failed");
        }
        ggml_backend_buffer_set_usage(state.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(state.buffer.get()));
    }
    size_t offset = 0;
    constexpr const char *names[] = {"gate", "up", "down"};
    for (size_t id = 0; id < state.layers.size(); ++id)
    {
        auto &layer = state.layers[id];
        if (!layer.layout.stride)
        {
            continue;
        }
        for (size_t part = 0; part < 3; ++part)
        {
            offset = align_up(offset, alignment);
            const auto &layout = layer.layout.parts[part];
            auto *tensor = ggml_new_tensor_3d(
                state.context.get(), static_cast<ggml_type>(layout.kind), layout.dimensions[0],
                layout.dimensions[1], layer.pool.slots + 1);
            ggml_format_name(tensor, "blk.%zu.ffn_%s_exps.weight", id, names[part]);
            if (ggml_nbytes(tensor) != layer.pool.parts[part].bytes)
            {
                throw std::runtime_error(
                    "JUQE: tensor size differs from the validated pool layout");
            }
            if (!meta_only)
            {
                auto *destination = base + offset;
                layer.pool.parts[part].data = destination;
                ggml_backend_tensor_alloc(state.buffer.get(), tensor, destination);
                std::memset(destination + layout.bytes * layer.pool.slots, 0, layout.bytes);
            }
            layer.tensors[part] = tensor;
            offset = add(offset, layer.pool.parts[part].bytes);
        }
    }
}

void assign_tensors(llama_model &model, const juqe_offload &state)
{
    for (size_t id = 0; id < state.layers.size(); ++id)
    {
        const auto &layer = state.layers[id];
        if (!layer.layout.stride)
        {
            continue;
        }
        model.layers[id].ffn_gate_exps = layer.tensors[0];
        model.layers[id].ffn_up_exps = layer.tensors[1];
        model.layers[id].ffn_down_exps = layer.tensors[2];
    }
}
} // namespace juqe
