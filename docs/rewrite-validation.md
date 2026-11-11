# Operational rewrite validation

This report records the original adapter-overlay validation. The current build
procedure is documented in [Pinned engine build](engine-build.md).

## Scope

The exact-routing JUQE runtime is integrated with the project's existing
JUQE-patched llama.cpp engine on Linux ARM64 with NVIDIA GB10. Rust owns cache
state, load scheduling, and leases. The engine retains its attention, quantization,
and CUDA kernels. The operating procedure is in [Run JUQE](operations.md).

The tested model is the existing Qwen3.8-Flash-Next UD-IQ4_XS deployment with its
matching expert pack, 512 experts per MoE layer, a 0.5 cache fraction, four I/O
workers, an eight-token microbatch, and a 4,096-token context. The adapter reports
28.46 GB of pinned expert pools; this is not total process or device memory.

## Automated checks

| Check | Result |
| --- | --- |
| Rust workspace on macOS | 23 integration tests passed |
| Rust workspace on Linux | 24 integration tests passed, including required direct I/O |
| Rust formatting, Clippy, and API documentation | Passed with warnings denied where applicable |
| Native C ABI caller on macOS and Linux | Layout, exact bytes, leases, error paths, and shutdown passed |
| Launcher tests on macOS and Linux | Inherited library paths and missing-build diagnostics passed |
| C and C++ formatting | clang-format 19.1.7, Microsoft-based configuration |
| Linux adapter build | Passed with compiler warnings treated as errors |
| Existing engine library | Unchanged by the adapter build |
| Real pack byte checks | 18 checks across three layers and three experts, buffered and direct |

The runtime tests cover protected hits, duplicate requests, capacity rejection,
failed-read rollback, shutdown with retained batches, draining concurrent writes
on failure, and recovery after a caught loader panic. Native C tests exercise
in-place ID remapping and overlapping-pool rejection.

## Inference and server checks

A driver ran four prompts with 16 generated tokens each against the existing C++
JUQE loader and the Rust adapter. The 64 generation records matched byte for byte:
token IDs, FNV logit hashes, and recorded maximum logits. Both paths disabled
pruning. This compares the two JUQE implementations, not an independent fully
resident reference or the earlier pruning configuration.

The Rust server also completed two simultaneous OpenAI-compatible chat requests
using two server slots, both without startup preload and with `--preload`. The responses contained the expected answers, Paris and 4.
The server shut down normally, and the previous Qwen service was restored healthy.

Validation found a real engine edge case: a final-layer top-k tensor can contain
zero token rows. The adapter now accepts that empty batch and completes the callback
sequence without submitting an invalid zero-length Rust request. The successful
server check includes this path.

These short checks establish functional compatibility for the tested configuration.
They do not establish broad answer quality, long-running service reliability,
inference acceleration, lower total memory, or a 10 ms worst-case loading bound.
No Dots3 result is inferred from them.

## Review

CodeRabbit reviewed the rewrite against the previous foundation commit
`f05daf3484c95d112244788fcff274a9b79dc25d`. The review completed with four minor
findings and no blocking findings at the major threshold. All four findings were
addressed: current integration status, README coverage, the complete source map,
and preservation of inherited library search paths. Launcher tests cover the
library-path correction.

A second full review completed with one trivial finding and no blocking findings.
The native C test now keeps assertions enabled even with `NDEBUG`; the corrected
test passed on macOS and Linux with `CC="cc -DNDEBUG"`. The macOS launcher fixture
also resolves temporary-directory aliases before comparing paths. These test-only
corrections do not change the GPU-tested runtime.

Rate-limited attempts were not counted as completed reviews. Local review logs
and raw model traces are excluded from the repository. The GitHub CodeRabbit app
is not connected; CLI review does not imply automatic pull-request review.

The [machine-readable record](evidence/rust-runtime-validation.json) identifies
the final adapter sources, linked libraries, and matching generation records.
