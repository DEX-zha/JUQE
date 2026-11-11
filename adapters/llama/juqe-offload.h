#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

struct llama_model;
struct juqe_offload;

/// Enable offloading when JUQE_EXPERTS names a pack directory.
bool juqe_offload_enabled();
juqe_offload *juqe_offload_attach(llama_model &model);
juqe_offload *juqe_offload_attach(llama_model &model, bool meta_only);
void juqe_offload_free(juqe_offload *state);
LLAMA_API void juqe_offload_set_decoding(llama_model *model, bool decoding);

/// Compatibility layout used by the existing engine's callback chain.
struct juqe_cb_chain
{
    juqe_offload *j;
    ggml_backend_sched_eval_callback user_cb;
    void *user_ud;
};
bool juqe_offload_eval_cb(ggml_tensor *tensor, bool ask, void *data);

/// Legacy observer ABI; the operational adapter rejects observer installation.
struct juqe_cache_observer
{
    void *data;
    void (*layer)(void *, int, const int32_t *, const float *, const uint8_t *, const uint8_t *,
                  int, double);
    void (*io)(void *, uint64_t, int, int, bool, uint64_t, double, double, double);
};
LLAMA_API void juqe_offload_observe(llama_model *, juqe_cache_observer *);
LLAMA_API void juqe_offload_sample(llama_model *, uint64_t);
LLAMA_API bool juqe_offload_snapshot(llama_model *, int, uint8_t *, uint16_t *, int);
