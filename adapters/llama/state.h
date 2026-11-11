#pragma once

#include "bridge.h"
#include "config.h"
#include "juqe-offload.h"
#include "llama-model.h"
#include <array>
#include <vector>

namespace juqe
{
struct layer
{
    juqe_layer_info layout{};
    juqe_layer_pool pool{};
    std::array<ggml_tensor *, 3> tensors{};
    bool fixed_router = false;
};
struct context_deleter
{
    void operator()(ggml_context *context) const
    {
        ggml_free(context);
    }
};
struct buffer_deleter
{
    void operator()(ggml_backend_buffer *buffer) const
    {
        ggml_backend_buffer_free(buffer);
    }
};
} // namespace juqe

struct juqe_offload
{
    juqe::config config;
    uint32_t experts = 0;
    uint32_t selected = 0;
    std::vector<juqe::layer> layers;
    // Reverse destruction order joins Rust workers before releasing their pools.
    std::unique_ptr<ggml_context, juqe::context_deleter> context;
    std::unique_ptr<ggml_backend_buffer, juqe::buffer_deleter> buffer;
    juqe::runtime_owner runtime;
    ggml_tensor *ids_tensor = nullptr;
    int ids_layer = -1;
    std::vector<int32_t> ids;
    std::vector<int32_t> slots;
    uint64_t hits = 0;
    uint64_t loads = 0;
    uint64_t bytes_read = 0;
    double wait_seconds = 0;
    size_t pool_bytes = 0;
    bool decoding = false;
};

namespace juqe
{
void describe_layers(llama_model &model, juqe_offload &state, const juqe_pack *pack,
                     bool meta_only);
void allocate_pools(juqe_offload &state, bool meta_only);
void assign_tensors(llama_model &model, const juqe_offload &state);
void capture_ids(juqe_offload &state, ggml_tensor *tensor, int layer);
void prepare_weights(juqe_offload &state, ggml_tensor *tensor, int layer);
} // namespace juqe
