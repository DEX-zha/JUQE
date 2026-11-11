#pragma once

#include "juqe.h"
#include <memory>
#include <stdexcept>
#include <vector>

namespace juqe
{
inline void check(int32_t status)
{
    if (status == 0)
    {
        return;
    }
    std::vector<char> error(juqe_last_error(nullptr, 0));
    juqe_last_error(error.data(), error.size());
    throw std::runtime_error(std::string("JUQE: ") + error.data());
}
struct pack_deleter
{
    void operator()(juqe_pack *pack) const
    {
        juqe_pack_close(pack);
    }
};
struct runtime_deleter
{
    void operator()(juqe_runtime *runtime) const
    {
        juqe_runtime_close(runtime);
    }
};
using pack_owner = std::unique_ptr<juqe_pack, pack_deleter>;
using runtime_owner = std::unique_ptr<juqe_runtime, runtime_deleter>;
} // namespace juqe
