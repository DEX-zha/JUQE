#ifndef JUQE_H
#define JUQE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct juqe_pack juqe_pack;
    typedef struct juqe_runtime juqe_runtime;

    typedef struct
    {
        uint32_t layers;
        uint32_t experts;
    } juqe_pack_info_t;
    typedef struct
    {
        uint32_t kind;
        uint64_t dimensions[2];
        uint64_t bytes;
    } juqe_part_info;
    typedef struct
    {
        uint64_t stride;
        juqe_part_info parts[3];
    } juqe_layer_info;
    typedef struct
    {
        uint8_t *data;
        uint64_t bytes;
    } juqe_pool_part;
    typedef struct
    {
        uint32_t slots;
        juqe_pool_part parts[3];
    } juqe_layer_pool;
    typedef struct
    {
        uint32_t workers;
        uint32_t queue_capacity;
        uint32_t direct;
    } juqe_runtime_config;
    typedef struct
    {
        uint64_t cache_hits;
        uint64_t loads;
        uint64_t bytes_read;
        double request_to_ready_seconds;
    } juqe_batch_info;

    /// Functions return zero on success, one on an error, or two after a caught panic.
    /// Handles belong to one control thread. Read the error before another API call.
    size_t juqe_last_error(char *buffer, size_t capacity);

    /// Open an immutable index. Output storage must be writable and disjoint from inputs.
    int32_t juqe_pack_open(const char *path, juqe_pack **output);
    int32_t juqe_pack_info(const juqe_pack *pack, juqe_pack_info_t *output);
    int32_t juqe_pack_layer(const juqe_pack *pack, uint32_t layer, juqe_layer_info *output);
    int32_t juqe_pack_close(juqe_pack *pack);

    /// Borrow writable, disjoint engine pools until runtime_close returns.
    /// Descriptors may be discarded after creation; their allocations must remain live.
    /// direct=0 selects buffered I/O; direct=1 selects Linux direct scatter I/O.
    int32_t juqe_runtime_create(const juqe_pack *pack, const char *weights,
                                const juqe_runtime_config *config, const juqe_layer_pool *pools,
                                size_t layers, juqe_runtime **output);

    /// Prepare one complete batch. Output slots preserve expert order and may alias IDs.
    /// The engine must release the previous batch before preparing another one.
    /// Each array contains count entries (1..1048576); info must be disjoint from both.
    int32_t juqe_runtime_prepare(juqe_runtime *runtime, uint32_t layer, const int32_t *experts,
                                 size_t count, int32_t *slots, juqe_batch_info *info);

    /// Synchronize all GPU and CPU consumers before releasing their batch or runtime.
    int32_t juqe_runtime_release(juqe_runtime *runtime);
    int32_t juqe_runtime_close(juqe_runtime *runtime);

#ifdef __cplusplus
}
#endif
#endif
