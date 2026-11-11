#include "llama-impl.h"
#include "state.h"

#include <charconv>
#include <cstring>
#include <exception>

namespace
{
int layer_id(const char *name, const char *prefix)
{
    const size_t length = std::strlen(prefix);
    if (std::strncmp(name, prefix, length) != 0 || name[length] != '-')
    {
        return -1;
    }
    const char *first = name + length + 1;
    const char *last = first + std::strlen(first);
    int layer = -1;
    const auto parsed = std::from_chars(first, last, layer);
    return parsed.ec == std::errc() && parsed.ptr == last && layer >= 0 ? layer : -1;
}
} // namespace

bool juqe_offload_eval_cb(ggml_tensor *tensor, bool ask, void *data)
{
    auto &chain = *static_cast<juqe_cb_chain *>(data);
    const int topk = layer_id(tensor->name, "ffn_moe_topk");
    const int weights = layer_id(tensor->name, "ffn_moe_weights_norm");
    const bool managed = topk >= 0 || weights >= 0;
    if (ask)
    {
        const bool user = chain.user_cb && chain.user_cb(tensor, true, chain.user_ud);
        return managed || user;
    }
    try
    {
        if (managed && chain.j)
        {
            if (topk >= 0)
            {
                juqe::capture_ids(*chain.j, tensor, topk);
            }
            else
            {
                juqe::prepare_weights(*chain.j, tensor, weights);
            }
        }
    }
    catch (const std::exception &error)
    {
        // The legacy scheduler does not propagate callback failure as an inference
        // error. Stop explicitly rather than continuing with incomplete weights.
        GGML_ABORT("%s", error.what());
    }
    catch (...)
    {
        GGML_ABORT("JUQE: unexpected adapter failure");
    }
    if (chain.user_cb && chain.user_cb(tensor, true, chain.user_ud))
    {
        return chain.user_cb(tensor, false, chain.user_ud);
    }
    return true;
}
