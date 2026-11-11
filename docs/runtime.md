# Runtime ownership

`juqe-runtime` prepares exact expert batches through a bounded worker pool. The
inference adapter supplies storage and holds each returned batch until GPU
consumption completes.

## Request flow

1. Validate the layer, expert IDs, and distinct-expert capacity before submitting I/O.
2. Lease all requested resident experts before reserving missing experts.
3. Reserve missing slots and submit one job per distinct absent expert.
4. Read through parallel workers with private, preallocated scratch buffers.
5. Publish each successful load only after its writer returns.
6. Return slots in the original request order, with leases protecting their bytes.

A batch that needs more slots than the layer provides fails with an actionable
error. A batch blocked by another live batch returns a cache-busy error. Neither
case evicts protected slots or silently reduces the requested experts.

## Bounds and failures

Configuration limits workers to 128 and the waiting job queue to 4,096 entries.
Only one `prepare` call can run at a time. A batch contains at most 4,096 distinct
experts, so the completion channel is bounded by outstanding batch work even
though it uses an unbounded channel primitive. Repeated expert IDs share one load.

Every worker allocates scratch before startup completes. Failed startup closes
the queue and joins workers already started. Read failures and caught loader
panics fail the associated ticket. Preparation drains all submitted writes before
returning an error. Successful loads remain cached and can be reused on retry.
A failed worker channel stops the runtime instead of permitting unsafe reuse.

Shutdown closes submissions, drains queued jobs, and joins all workers. Existing
batches keep their storage alive. Their leases must be released only after the
engine's completion fence. Dropping a Rust batch does not synchronize a GPU.

## Source map

| Responsibility | Source |
| --- | --- |
| Public control path | `crates/juqe-runtime/src/runtime.rs` |
| Configuration limits | `crates/juqe-runtime/src/config.rs` |
| Batch reservations and leases | `crates/juqe-runtime/src/batch.rs` |
| Submission and completion draining | `crates/juqe-runtime/src/dispatch.rs` |
| Worker lifecycle and bounded queue | `crates/juqe-runtime/src/pool.rs` |
| Worker execution and panic boundary | `crates/juqe-runtime/src/worker.rs` |
| Storage integration contract | `crates/juqe-runtime/src/storage.rs` |
| Actionable errors | `crates/juqe-runtime/src/error.rs` |

## Integration status

The llama.cpp adapter binds engine-owned CUDA host pools through `juqe-ffi` and
uses this runtime for expert preparation. Qwen generation and concurrent HTTP
requests have completed on GB10. See [rewrite validation](rewrite-validation.md)
for the exact checked scope and [operations](operations.md) for build and startup.
The runtime preserves exact routing. Speculative loading and pruning are outside
this operational control path.
