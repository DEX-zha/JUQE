#pragma once

#include <cstdint>
#include <string>

namespace juqe
{
struct config
{
    std::string directory;
    double cache_fraction = 0.5;
    uint32_t workers = 4;
    uint32_t queue_capacity = 64;
    bool compact_fixed = true;
    bool preload = false;
};
config read_config();
} // namespace juqe
