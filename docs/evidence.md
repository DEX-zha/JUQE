# Prototype evidence

These results motivate the migration. **They do not measure the Rust implementation.**
The C++ and Python sources, weights, and raw traces remain in the original research
workspace. A structured summary with its source SHA-256 is preserved in
[prototype-loading-summary.json](evidence/prototype-loading-summary.json).

| Observation | Supported conclusion |
| --- | --- |
| Qwen saved approximately 31.06 GB of weight buffers compared with the resident reference | About 48% of those allocations; this is not total service memory |
| An earlier paired comparison measured JUQE at 21.37 tokens/s and the reference at 29.07 tokens/s | Memory savings did not deliver reference throughput |
| The Qwen oracle improved from 20.97 to 24.76 tokens/s excluding preparation, but reached 20.16 tokens/s including preparation | Overlap has potential, but no deployable predictor was validated |
| Pruning at 0.12 versus 0.10 increased median paired throughput by 2.57% on a controlled workload | A limited speed gain, not additional memory savings |
| Both variants passed the same 21 of 32 requests after additional output-format normalization | The corpus is too small to establish general quality equivalence |
| Scatter reads preserved Qwen weights and logits across three pairs | Exactness was verified on this workload; throughput gains varied from -6.77% to +13.11% |

The learned predictor remains suspended. Its cost and coverage do not justify
automatic reactivation during the rewrite.

## Loading target

The target is **at most 10 ms per expert, from request to readiness**.
A microbenchmark of the **C++ prototype in scatter mode only**, using 12.53 MB
blocks, observed maximum latencies of 2.85, 5.66, 9.43, and 17.10 ms for waves of
1, 2, 4, and 8 requests. These maxima combine configurations with two and four
workers; the respective maxima came from 2, 2, 4, and 2 workers.

Scatter met the target for up to four requests in this sample; eight requests
exceeded it. Without scatter, four requests reached 11.21 ms with two workers and
11.34 ms with four workers, with 10 and 6 requests exceeding 10 ms, respectively.

Each configuration ran 20 repetitions. No GPU compute load was imposed, and this
campaign included neither Dots3 inference nor Rust measurements. The large blocks
used existing bytes with a hypothetical IQ4_XS layout, not Dots3 weights.

The SSD delivered approximately 6.7 GB/s with multiple readers. The earlier
3-3.7 GB/s assumption came from a single reader on small blocks. The earlier
estimate of 2-5 tokens/s for Dots3 is therefore not an established result.

Rust evaluations need their own provenance, measurements, and validation on the
actual CUDA paths before any inference speedup can be claimed.
