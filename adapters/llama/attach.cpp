#include "llama-impl.h"
#include "state.h"

#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace
{
void start_runtime(juqe_offload &state, const juqe_pack *pack)
{
    std::vector<juqe_layer_pool> pools;
    for (const auto &layer : state.layers)
    {
        pools.push_back(layer.pool);
    }
    const juqe_runtime_config config = {state.config.workers, state.config.queue_capacity, 1};
    juqe_runtime *runtime = nullptr;
    juqe::check(juqe_runtime_create(pack, (state.config.directory + "/experts.bin").c_str(),
                                    &config, pools.data(), pools.size(), &runtime));
    state.runtime.reset(runtime);
    if (state.config.preload)
    {
        for (size_t id = 0; id < state.layers.size(); ++id)
        {
            const auto slots = state.layers[id].pool.slots;
            if (!slots)
            {
                continue;
            }
            std::vector<int32_t> experts(slots), mapped(slots);
            std::iota(experts.begin(), experts.end(), 0);
            juqe_batch_info info{};
            juqe::check(
                juqe_runtime_prepare(runtime, id, experts.data(), slots, mapped.data(), &info));
            juqe::check(juqe_runtime_release(runtime));
        }
    }
}
} // namespace

bool juqe_offload_enabled()
{
    return std::getenv("JUQE_EXPERTS") != nullptr;
}

juqe_offload *juqe_offload_attach(llama_model &model, bool meta_only)
{
    if (!juqe_offload_enabled())
    {
        return nullptr;
    }
    auto state = std::make_unique<juqe_offload>();
    state->config = juqe::read_config();
    juqe_pack *raw_pack = nullptr;
    juqe::check(juqe_pack_open((state->config.directory + "/experts.idx").c_str(), &raw_pack));
    juqe::pack_owner pack(raw_pack);
    juqe::describe_layers(model, *state, pack.get(), meta_only);
    juqe::allocate_pools(*state, meta_only);
    if (!meta_only)
    {
        start_runtime(*state, pack.get());
    }
    // Commit model pointers only after every fallible setup step has succeeded.
    juqe::assign_tensors(model, *state);
    LLAMA_LOG_INFO(
        "[juqe] Rust runtime: %u experts, %u workers, %.2f GB pinned pools, exact routing%s\n",
        state->experts, state->config.workers, state->pool_bytes / 1e9,
        meta_only ? " (metadata only)" : "");
    return state.release();
}

juqe_offload *juqe_offload_attach(llama_model &model)
{
    return juqe_offload_attach(model, false);
}

void juqe_offload_free(juqe_offload *state)
{
    if (!state)
    {
        return;
    }
    LLAMA_LOG_INFO("[juqe] hits=%llu loads=%llu read_bytes=%llu preparation_seconds=%.3f\n",
                   static_cast<unsigned long long>(state->hits),
                   static_cast<unsigned long long>(state->loads),
                   static_cast<unsigned long long>(state->bytes_read), state->wait_seconds);
    delete state;
}

void juqe_offload_set_decoding(llama_model *model, bool decoding)
{
    if (model && model->juqe)
    {
        model->juqe->decoding = decoding;
    }
}

void juqe_offload_observe(llama_model *, juqe_cache_observer *observer)
{
    if (observer)
    {
        throw std::runtime_error(
            "JUQE: research observers are not supported by the operational runtime");
    }
}
void juqe_offload_sample(llama_model *, uint64_t)
{
}
bool juqe_offload_snapshot(llama_model *, int, uint8_t *, uint16_t *, int)
{
    return false;
}
