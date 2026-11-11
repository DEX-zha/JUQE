#include "config.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace
{
double number(const char *name, double fallback)
{
    const char *value = std::getenv(name);
    if (!value)
    {
        return fallback;
    }
    errno = 0;
    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (errno || end == value || *end || !std::isfinite(parsed))
    {
        throw std::invalid_argument(std::string(name) + " must be a finite number");
    }
    return parsed;
}
uint32_t count(const char *name, uint32_t fallback, uint32_t maximum)
{
    const double value = number(name, fallback);
    if (value < 1 || value > maximum || std::floor(value) != value)
    {
        throw std::invalid_argument(std::string(name) + " is outside its supported integer range");
    }
    return static_cast<uint32_t>(value);
}
bool flag(const char *name, bool fallback)
{
    const double value = number(name, fallback ? 1 : 0);
    if (value != 0 && value != 1)
    {
        throw std::invalid_argument(std::string(name) + " must be 0 or 1");
    }
    return value != 0;
}
void reject_enabled(const char *name)
{
    if (number(name, 0) != 0)
    {
        throw std::invalid_argument(std::string(name) +
                                    " is not supported by the exact Rust runtime");
    }
}
} // namespace

namespace juqe
{
config read_config()
{
    config result;
    const char *directory = std::getenv("JUQE_EXPERTS");
    if (!directory || !*directory)
    {
        throw std::invalid_argument("JUQE_EXPERTS must name a pack directory");
    }
    result.directory = directory;
    result.cache_fraction = number("JUQE_CACHE_FRAC", 0.5);
    if (result.cache_fraction <= 0 || result.cache_fraction > 1)
    {
        throw std::invalid_argument("JUQE_CACHE_FRAC must be greater than zero and at most one");
    }
    result.workers = count("JUQE_IO_THREADS", 4, 128);
    result.queue_capacity = count("JUQE_QUEUE_CAPACITY", 64, 4096);
    result.compact_fixed = flag("JUQE_COMPACT_FIXED", true);
    result.preload = flag("JUQE_PRELOAD", false) || result.cache_fraction == 1.0;
    const char *cpus = std::getenv("JUQE_IO_CPUS");
    if (cpus && *cpus && std::string(cpus) != "off")
    {
        throw std::invalid_argument("JUQE_IO_CPUS binding is not supported by this adapter");
    }
    if (!flag("JUQE_IO_SCATTER", true))
    {
        throw std::invalid_argument("this adapter requires JUQE_IO_SCATTER=1");
    }
    reject_enabled("JUQE_ZERO_FAST");
    reject_enabled("JUQE_BG_POLICY");
    reject_enabled("JUQE_SKIP_W");
    reject_enabled("JUQE_PIPELINE");
    reject_enabled("JUQE_TEST_SKIP_EVERY");
    return result;
}
} // namespace juqe
