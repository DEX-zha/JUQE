# Engine integration contract

The C interface is declared in [juqe.h](../include/juqe.h) and implemented in
`juqe-ffi`. All runtime ownership and scheduling remain in safe Rust. Raw pointer
handling is limited to this crate and the allocation/syscall boundary in `juqe-io`.

## Handle lifecycle

1. Open `experts.idx` with `juqe_pack_open` and inspect its validated geometry.
2. Allocate disjoint, stable gate/up/down pools for each routed layer.
3. Create a runtime with the weight path, resource limits, and pool descriptors.
4. Prepare expert IDs and consume the returned slots only after success.
5. Synchronize GPU and CPU consumption, then release the active batch.
6. Close the runtime before releasing the pool allocations.

Metadata handles can be closed after runtime creation. The runtime copies metadata,
not weight bytes. Descriptors can be discarded after creation, but their allocations
must remain live. Dense layers have no pools. The optional final zero slot belongs
to the engine and is excluded from the Rust cache's writable slot count.

## Foreign memory

Creation validates layer counts, slot counts, allocation extents, pointer arithmetic,
and overlap across pools. The caller remains responsible for actual allocation
validity and lifetime, which cannot be proven from a C pointer. Workers create
writable slices only while holding the destination slot's mutex. Runtime leases
exclude GPU consumption of that slot during writes.

Each handle belongs to one control thread. `prepare` rejects a second active batch.
The ID and output-slot arrays may alias because IDs are copied before output writes.
Other inputs and outputs must be disjoint. Outputs may be uninitialized writable
storage; the implementation writes through raw pointers without first borrowing
an uninitialized Rust value.

## Errors

Functions return zero on success, one on an error, and two for a caught panic.
Read `juqe_last_error` on the calling thread before the next API call. Its return
value includes space for the terminating NUL byte. A null error buffer queries the
required capacity.

Null handles are accepted by close functions. Other null inputs, invalid IDs,
invalid pool layouts, and repeated preparation without release are rejected.
Read failures return only after submitted writes have finished. The adapter must
stop inference on failure; incomplete destinations must never reach a kernel.

## Validation

Run `python3 scripts/check-cabi.py` to compile an actual C caller and link it to
the Rust static library. It checks struct layout through field access, exact loaded
bytes, adjacent-slot preservation, in-place ID mapping, active-batch protection,
null pointers and invalid IDs, overlapping-pool rejection, and shutdown order.
