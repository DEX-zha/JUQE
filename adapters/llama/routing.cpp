#include "state.h"

#include <limits>
#include <stdexcept>

namespace juqe
{
void capture_ids(juqe_offload &state, ggml_tensor *tensor, int id)
{
    if (id < 0 || static_cast<size_t>(id) >= state.layers.size() ||
        !state.layers[id].layout.stride || tensor->type != GGML_TYPE_I32 ||
        tensor->ne[0] != state.selected || tensor->nb[0] != sizeof(int32_t) || tensor->ne[1] < 0 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->ne[1] > 1048576 / state.selected)
    {
        throw std::runtime_error(
            "JUQE: invalid top-k tensor layout: layer=" + std::to_string(id) +
            " layers=" + std::to_string(state.layers.size()) +
            " type=" + std::to_string(tensor->type) + " ne0=" + std::to_string(tensor->ne[0]) +
            " ne1=" + std::to_string(tensor->ne[1]) + " nb0=" + std::to_string(tensor->nb[0]) +
            " selected=" + std::to_string(state.selected));
    }
    // The supported scheduler synchronizes the backend before this callback.
    // Previous layer consumers have completed, so their batch can be released.
    check(juqe_runtime_release(state.runtime.get()));
    state.ids_tensor = tensor;
    state.ids_layer = id;
    const size_t rows = tensor->ne[1];
    state.ids.resize(rows * state.selected);
    for (size_t row = 0; row < rows; ++row)
    {
        ggml_backend_tensor_get(tensor, state.ids.data() + row * state.selected,
                                row * tensor->nb[1], state.selected * sizeof(int32_t));
    }
    for (int32_t expert : state.ids)
    {
        if (expert < 0 || static_cast<uint32_t>(expert) >= state.experts ||
            (state.layers[id].fixed_router && static_cast<uint32_t>(expert) >= state.selected))
        {
            throw std::runtime_error("JUQE: routed expert is outside the configured layer");
        }
    }
}

void prepare_weights(juqe_offload &state, ggml_tensor *tensor, int id)
{
    if (state.ids_layer != id || !state.ids_tensor || tensor->type != GGML_TYPE_F32 ||
        tensor->ne[0] != state.selected || tensor->ne[1] != state.ids_tensor->ne[1] ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 || !ggml_is_contiguous(tensor))
    {
        throw std::runtime_error("JUQE: routed weights do not match the preceding top-k tensor");
    }
    // The server can prune every row from the final layer. There is no expert
    // work for that graph view, but the callback sequence still completes.
    if (state.ids.empty())
    {
        state.ids_layer = -1;
        return;
    }
    state.slots.resize(state.ids.size());
    juqe_batch_info info{};
    check(juqe_runtime_prepare(state.runtime.get(), id, state.ids.data(), state.ids.size(),
                               state.slots.data(), &info));
    for (size_t row = 0; row < static_cast<size_t>(tensor->ne[1]); ++row)
    {
        ggml_backend_tensor_set(state.ids_tensor, state.slots.data() + row * state.selected,
                                row * state.ids_tensor->nb[1], state.selected * sizeof(int32_t));
    }
    state.hits += info.cache_hits;
    state.loads += info.loads;
    state.bytes_read += info.bytes_read;
    state.wait_seconds += info.request_to_ready_seconds;
    state.ids_layer = -1;
}
} // namespace juqe
