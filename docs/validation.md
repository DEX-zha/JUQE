# Rust foundation validation

Validation date: November 11, 2026 (Europe/Paris, UTC+01:00).

## Completed checks

| Check | macOS ARM64 | Linux ARM64 / GB10 |
| --- | --- | --- |
| Rust | 1.99.1 | 1.99.1 |
| rustfmt | Passed | Passed |
| Clippy, all targets, warnings denied | Passed | Passed |
| Tests | 15 passed | 16 passed |
| rustdoc, warnings denied | Passed | Passed |
| Required `O_DIRECT` | Unavailable | Passed with `JUQE_REQUIRE_DIRECT=1` |

The additional Linux checks cover scatter reads and aligned staging fallback for
an incompatible matrix offset. Portable tests cover partial writes, adjacent
slots, concurrent reads, leases and tickets, invalid metadata, and CLI errors.

Timestamp and destination-size tests use separate inputs to verify each error
independently.

## Existing Qwen pack verification

The validation ran 18 `juqe verify` commands across layers 0, 2, and 4, representing
the pack's three layouts; experts 0, 256, and 511; and buffered and direct modes.
Each matrix was compared in full with an independent positional read.

All byte comparisons matched. All nine direct requests used `DirectScatter`.
The program did not run inference or stop any service. The recorded individual
load durations are not a representative benchmark or evidence of a speedup over C++.

## CodeRabbit review

The reviews used CodeRabbit CLI 0.9.4, the assertive profile, repository conventions,
and a major-severity gate. The configuration was validated against the official
schema.

- The first foundation review reported two minor findings and no major findings.
  The timestamp test was corrected, and the historical scatter results were clarified.
- The second review covered the corrections and final adjustments. It reported
  zero findings and passed the gate.

Results and tested-source hashes are preserved in
[rust-foundation-validation.json](evidence/rust-foundation-validation.json) and
[rust-pack-verification.json](evidence/rust-pack-verification.json).
These reports were created after the second review. The review results do not
claim to cover the reports' own wording.

The English documentation revision preserves the tested Rust sources. Historical
review identifiers refer to the original snapshots before the branch history was
consolidated. Finding titles are translated summaries; raw review-log hashes
identify the unchanged original logs.

The English documentation review reported one minor date concern and passed the
major-severity gate. The local log timestamps confirm November 11 in Europe/Paris
(November 10 in UTC). The report now specifies the time zone. This clarification
and the review record were added after that review.

GitHub CI is configured. The local results above do not establish that remote CI
ran. At the time of the CLI reviews, the repository was not connected to the
CodeRabbit GitHub app; the reviews used the CLI's free allowance. Automatic pull
request reviews require the app to be installed on this repository.

## Scope

This milestone validates a Rust foundation and reads of actual model weights.
GPU pools, the bounded scheduler, the C ABI, and integrated-engine logit comparisons
remain planned milestones. No Rust inference throughput is reported.
