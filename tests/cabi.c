#include "juqe.h"

// Keep validation active when the caller compiles with release flags.
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int32_t result)
{
    if (result != 0)
    {
        char error[512];
        juqe_last_error(error, sizeof(error));
        fprintf(stderr, "%s\n", error);
        abort();
    }
}

int main(int argc, char **argv)
{
    assert(argc == 3);
    juqe_pack *pack = NULL;
    check(juqe_pack_open(argv[1], &pack));
    juqe_pack_info_t counts;
    check(juqe_pack_info(pack, &counts));
    assert(counts.layers == 1 && counts.experts == 4);
    juqe_layer_info layout;
    check(juqe_pack_layer(pack, 0, &layout));
    assert(layout.stride == 4096 && layout.parts[0].bytes == 512);
    assert(layout.parts[0].dimensions[0] == 128);
    assert(juqe_pack_info(NULL, &counts) != 0);
    assert(juqe_pack_info(pack, NULL) != 0);

    juqe_layer_pool pool = {.slots = 2};
    for (size_t part = 0; part < 3; ++part)
    {
        pool.parts[part].data = malloc(1536);
        assert(pool.parts[part].data);
        pool.parts[part].bytes = 1536;
        memset(pool.parts[part].data, 0xcc, 1536);
    }
    const juqe_runtime_config config = {.workers = 2, .queue_capacity = 1, .direct = 0};
    juqe_runtime *runtime = NULL;
    juqe_layer_pool overlapping = pool;
    overlapping.parts[1] = overlapping.parts[0];
    assert(juqe_runtime_create(pack, argv[2], &config, &overlapping, 1, &runtime) != 0);
    assert(runtime == NULL);
    check(juqe_runtime_create(pack, argv[2], &config, &pool, 1, &runtime));
    check(juqe_pack_close(pack));

    int32_t ids[] = {3, 1, 3};
    juqe_batch_info info;
    check(juqe_runtime_prepare(runtime, 0, ids, 3, ids, &info));
    assert(ids[0] == ids[2] && ids[0] != ids[1]);
    assert(info.loads == 2 && info.cache_hits == 0 && info.bytes_read == 8192);
    assert(info.request_to_ready_seconds >= 0.0);
    const size_t offsets[] = {0, 1024, 2560};
    for (size_t part = 0; part < 3; ++part)
    {
        for (size_t i = 0; i < 512; ++i)
        {
            assert(pool.parts[part].data[ids[0] * 512 + i] == (3 * 4096 + offsets[part] + i) % 251);
            assert(pool.parts[part].data[ids[1] * 512 + i] == (4096 + offsets[part] + i) % 251);
            assert(pool.parts[part].data[1024 + i] == 0xcc);
        }
    }
    assert(juqe_runtime_prepare(runtime, 0, ids, 3, ids, &info) != 0);
    check(juqe_runtime_release(runtime));
    check(juqe_runtime_release(runtime));
    int32_t invalid = -1;
    assert(juqe_runtime_prepare(runtime, 0, &invalid, 1, ids, &info) != 0);
    check(juqe_runtime_close(runtime));
    check(juqe_runtime_close(NULL));
    for (size_t part = 0; part < 3; ++part)
    {
        free(pool.parts[part].data);
    }
    puts("C ABI: layout, exact bytes, leases, errors, and shutdown passed");
    return 0;
}
