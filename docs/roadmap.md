# Migration roadmap

## Milestone 1: Verified foundation

Provide the Rust workspace, development conventions, CodeRabbit review, legacy
index import, deterministic cache, exact reads, and verification CLI. Validate
Unix reads and Linux `O_DIRECT` on existing files without stopping the service.
This milestone makes no inference speedup claim.

## Milestone 2: Scheduler and buffer ownership

Add a bounded queue, demand priority, coalesced requests for the same expert,
preallocated scratch memory per worker, and load and wait counters. Do not cancel
an active writer without a completion fence. Test stress, deadline overruns, and
I/O errors. Add a versioned pack manifest and verify weight identity.

## Milestone 3: Engine adapter

Connect Rust to the existing llama.cpp callbacks through a narrow C ABI, with CUDA
pinned pools and leases released after GPU consumption. Keep the C++ engine
available for exact A/B comparisons of tokens and logits. Verify shutdown, panic
handling at the boundary, destruction after I/O, and the absence of reallocation
while kernels are running.

## Milestone 4: Evaluation and optimization

Use the same memory budget, weights, and requests, with alternating repeated runs.
Report reads, request-to-ready latency, percentiles, maxima, overall throughput,
and quality. Test waves of 1, 2, 4, and 8 misses with GPU activity and background
reads. Establish Dots3 results only through measurements on its actual weights.

Pruning and prediction remain separate experiments. A language change or an SSD
microbenchmark does not validate their reactivation.
