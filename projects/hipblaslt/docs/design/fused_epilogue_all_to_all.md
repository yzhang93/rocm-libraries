# Fused Epilogue Extensions: GEMM + All-to-All

This document specifies the hipBLASLt API extension for a fused GEMM + all-to-all epilogue: a
multi-device epilogue that redistributes part of a GEMM's output across the participating devices
from the GEMM's own store path, instead of running a separate collective kernel after `D` is
complete.

`GEMM + all-to-all` is its **own fused epilogue family**, not a member of the chainable set
defined in [`fused_epilogue_rmsnorm.md`](fused_epilogue_rmsnorm.md). It borrows that document's
*plumbing* — the same `hipblasLtFusedEpilogueDescriptor_t` builder handle, the same
matmul-descriptor attachment point, and the same opaque cross-call descriptor pattern — because a
caller should configure one kind of fusion the same way regardless of family. It does not borrow
the chain-ordering model: the ordered chain was derived from the CODA reparameterization, which has
no notion of a collective, and all-to-all has no meaningful position in that order.

The device-side implementation this design exposes is the TensileLite `FusedGemmA2A` epilogue and
its four-GPU validation client
([ROCm/rocm-libraries#10925](https://github.com/ROCm/rocm-libraries/pull/10925)). That client is
the reference for every requirement, constraint, and error condition below.

## 1. Motivation and scope

A GEMM whose output must be redistributed across devices currently pays for a second kernel: the
GEMM writes `D`, then a collective reads `D` back and sends each rank its slice, usually through a
staging buffer sized for the whole output plane. The collective's transfers start only after the
GEMM's last tile is stored, so they cannot overlap the GEMM's own math.

Fusing the redistribution into the store path removes the extra launch and the staging buffer, and
lets each finished tile band's transfer overlap the remaining tiles' math. The transfers are issued
to a copy engine (SDMA) rather than performed by the compute units, so the cross-device traffic
does not consume CU cycles or registers.

This design defines:

1. All-to-all as a self-contained fused epilogue family on an ordinary `hipblasLtMatmul` call,
   configured through the existing `hipblasLtFusedEpilogueDescriptor_t` builder and standing on its
   own rather than composing with other epilogue stages.
2. A *communicator*, which every rank registers on its own library handle — one call per handle, and
   one communicator across the group, each handle holding a local view of it. Registration fixes that
   rank's index and the world size and allocates the library-owned completion flags. Its only means of
   cross-rank exchange is a caller-supplied allgather callback, so the library needs no rendezvous of
   its own and works unchanged whether the ranks are threads or processes.
3. The ownership split, which puts nearly everything on the caller: `D` stays an ordinary matmul
   output, the redistributed data lands in a receive buffer the caller allocates and sizes, and the
   copy-engine queues are created by the caller and handed over as addresses the library does not
   interpret. All the library owns is the completion flags and the per-launch counters.
4. Completion semantics, so a caller knows when a received shard is safe to read, and the launch
   ordering rule that follows from a collective running inside a kernel.
5. Reserved extension points for the mirrored `all-to-all -> GEMM` direction, for larger world
   sizes, and for a later decision about composing all-to-all with the chainable stages.

Out of scope for the initial release: multi-node participation, batched GEMM, and world sizes above
8. Multi-process is in scope, and is in fact the expected deployment: one process per rank is the
shape a `torchrun` job takes and any PyTorch integration inherits, so section 5.2 lets it share the
single-process entry point rather than needing a second one.

## 2. All-to-all definition

The GEMM is data-parallel over the free-1 dimension of `D` and replicated over free-0. Using the
client's naming, free-0 of `D` is the *feature* dimension (`M`, the A-side free index) and free-1 is
the *token* dimension (`N`, the B-side free index):

```text
rank r computes  D_r[0:M, 0:N] = op(A) * op(B_r)
```

Every rank holds all `M` features for the tokens it owns. The all-to-all converts that ownership
into its transpose: each rank ends up owning one contiguous *feature shard* for the tokens of every
rank.

Let `W` be the world size, `AM` the redistributed feature extent, and `n_shard = AM / W`:

```text
shard(j) = [ j * n_shard, (j+1) * n_shard )      for j in [0, W)
```

For every ordered pair of ranks `(s, d)`, the band `D_s[shard(d), 0:N]` is delivered to rank `d`.
Rank `d`'s receive buffer is therefore indexed by source rank first:

```text
recv_d[s, t, f] = D_s[ d * n_shard + f , t ]     s in [0,W), t in [0,N), f in [0,n_shard)
```

Two consequences are part of the contract:

- **`D` is still a complete matmul output.** The exported bands are not diverted away from `D`; they
  are written to `D` as usual and the copy engine reads them back out of it. Features
  `[AM, M)` are never exported and are the *local tail* that only `D` carries. Because `[0, AM)` is
  the copy source, it must not alias any other tensor, and a caller that wants to reuse it as scratch
  must wait for the whole launch group, not just its own rank (section 4.2).
- **The self pair `(d, d)` is a real transfer.** A rank's own shard is delivered into its own receive
  buffer through a loopback queue, so `recv_d` has the same `W`-slot shape for every `d` and the
  consumer indexes source ranks uniformly.

`AM = M` redistributes the whole output and leaves an empty local tail; `AM < M` keeps a tail in `D`.

## 3. Fused epilogue family

The redistribution needs no arithmetic; it needs the *address* each finished output element should
also appear at. That decision belongs to the store path, which already computes each tile's global
coordinates. A workgroup that has finished a tile in the exported region knows the destination rank
from its free-0 coordinate alone (`dst_rank = coord0 / n_shard`), so the collective's routing is a
by-product of the epilogue's existing addressing. Placing it there is also what buys the overlap: a
band's transfer is in flight while the remaining workgroups are still doing MFMA work, whereas a
post-GEMM collective cannot start until the last tile is stored.

All-to-all is therefore an epilogue in the same sense the other fusions are — extra work performed
on a tile's values inside the kernel that produced them — but its effect is *distributing*: the main
output `D` keeps its shape, and the redistributed data is a side output that lands in another
device's memory. That puts it in a family of its own. The fused epilogue API offers several
families, each with its own composition rule.

| Family | Composition rule | Members |
|--------|------------------|---------|
| Chainable epilogues | Ordered chain, validated as an order-preserving subsequence of `bias -> residual add -> RMSNorm -> AMax -> requant` | residual add, RMSNorm, partial RMSNorm stats, RMSNorm scale-apply, AMax, requant |
| Gated linear units | Single stage; shape-changing (`[M,2N] -> [M,N]`), not chainable initially | SwiGLU (first), GeGLU, ReGLU |
| **Collectives** | **Single stage; distributing, not chainable initially** | **all-to-all** |

The families share the configuration surface — one builder handle, `Add` to select the fusion,
`SetAttribute` for its parameters, one matmul-descriptor attribute to attach it — but not one
ordering rule. All-to-all is a single-stage family, enforced at `Add` time: adding any other stage
to a handle that already contains all-to-all returns `HIPBLAS_STATUS_INVALID_VALUE`, the same
treatment SwiGLU gets. Section 8 records what relaxing that would take.

One `GEMM + all-to-all` operation is `W` matmul calls, one per rank, each on its own device, stream,
and matmul descriptor, each with its own fused-epilogue handle carrying the single all-to-all stage.
Those calls may be `W` threads of one process or `W` separate processes; nothing in the design
distinguishes them. Two things tie the ranks together: the communicator of section 5.2, registered on
each rank's library handle and carrying that rank's index, and the rank-ordered peer
receive-pointer array, which every rank supplies in full. All `W` calls must be in flight at once
(section 4.2).

## 4. Cross-device realization

### 4.1 Kernel organization (single kernel)

The fused path is one kernel launch per rank. Within a launch:

1. **Store.** Each workgroup stores its tile to `D`. Workgroups whose tile lies in `[0, AM)` store
   with the cache policy the copy engine requires, so the band is visible in memory rather than
   resident in a non-coherent cache. Workgroups in the local tail follow the ordinary store path.
2. **Band completion.** Workgroups covering one (destination rank, token tile) band tally their
   completion in a per-launch counter. The last one to arrive owns the band.
3. **Submit.** That owning workgroup places two commands on the queue for its destination rank: a
   rectangular copy for the band, then an atomic incrementing the destination's arrival counter for
   this source rank. Both on one queue, in that order, because the engine consumes a ring in order —
   so observing the increment implies the band landed. Outbound bands are dispatched ahead of the
   launch's local-tail work so the transfers overlap it.
4. **Drain (optional).** One elected workgroup per launch waits until every source rank's arrival
   flag for *this* device shows a full set of bands, so the kernel cannot retire before the data this
   rank is receiving has landed.

None of this is steerable through the API. The counter buffer, flag array, band-ownership election,
and descriptor format are library-internal; only the queues the descriptors are submitted to are
caller-created (section 4.3). Sections 5 and 7 define what the caller supplies.

### 4.2 Completion semantics

There are two different completions to reason about, and conflating them is the main hazard of the
whole feature:

- **Send completion.** This rank's outbound bands have landed in every peer's receive buffer, so
  `D[0:AM)` — the copy source — is free to be overwritten or freed.
- **Receive completion.** Every peer's inbound band for this rank has landed in this rank's receive
  buffer, so `recv` is free to be read.

Only the second is what the completion mode governs. Send completion is established indirectly:
rank `r`'s outbound band to rank `q` has landed by the time `q`'s kernel retires, so a caller that
synchronizes the whole group has both, while a caller that synchronizes only rank `r`'s stream has
receive completion for `r` and no guarantee about `r`'s sends. This is why the reuse rule for
`D[0:AM)` in section 2 is group-wide rather than per-rank.

This release establishes it one way, `IN_KERNEL`: the kernel does not retire until this rank's
receive buffer is fully populated, so ordinary stream semantics then cover the collective and `recv`
is safe to read once the rank's stream is synchronized.

The kernel's wait is a runtime gate rather than a codegen option — the same binary can skip the
barrier — so a *deferred* mode is cheap to add on the device side. It is not exposed, because
retiring early leaves the caller with nothing to wait on: the arrival flags are library-owned and the
copy engine's queues are not HIP streams, so no stream or device synchronization implies inbound data
has landed. Exposing the mode means also exposing a primitive to wait on, which section 8 records as
the follow-up.

**Launch ordering rule.** Because the wait lives inside the kernel, every participating rank's
matmul must be *enqueued* before any of them is synchronized. Enqueueing rank 0, waiting for it, and
only then enqueueing rank 1 deadlocks: rank 0's kernel is waiting for data that rank 1 has not been
asked to produce yet. The API cannot detect this, so it is a documented caller obligation.

**Reuse across launches.** The library re-initializes the flag and counter state per launch, so
sequential launches on one communicator are safe; the `IN_KERNEL` wait is what makes that reset safe,
since no launch can still be arriving when the next one starts. *Concurrent* launches need distinct
channels and disjoint queue sets instead (sections 4.3 and 5.3). The receive buffer is not cleared by
the library; a caller that wants a stale-data check must clear it itself.

### 4.3 Transport backend

The transfers are issued as rectangular sub-window copy commands on user-mode copy-engine (SDMA)
queues, one queue per (source device, destination rank) pair including the loopback pair, each copy
followed by a counter-incrementing atomic. How those commands are built and published — the ring
cursors, the doorbell write, the packet field packing, and the ownership election that keeps exactly
one producer per band tile — is internal and appears nowhere in the API.

The *queues* are the exception — the one place the transport reaches the public surface. Creating a
queue means allocating its ring, creating the queue itself against a topology node and engine, and
tearing both down afterwards, all of which goes through the kernel-mode thunk (hsakmt). Whoever
creates a queue therefore links hsakmt, so leaving that to the caller is what keeps the dependency out
of hipBLASLt's build: the library receives four addresses per peer and hands them to the kernel
without interpreting any of them.

The trade is that a transport on the public surface cannot be replaced silently; section 8 covers
what that does and does not foreclose. One further transport property drives an API rule: a ring
carries at most one launch at a time, which is what forces the queue provisioning of section 5.3.

Two properties of the descriptor format leak into the *constraints* of section 5.7, being
non-negotiable at runtime: its extent and pitch fields are narrow (14-bit extents, 19-bit source
pitch) and its addressing granularity is 16 bytes. Those become the divisibility and magnitude limits
on `AM`, `n_shard`, and `ldd`.

### 4.4 Numerics

The stage performs no arithmetic. `recv_d[s, t, f]` is a bitwise copy of the value `D_s` stores at
the corresponding coordinate, so a fused all-to-all is bit-exact against the same GEMM run without
it followed by a host-side redistribution of `D`. Nothing here needs numerical validation beyond the
ordinary GEMM; what needs validation is placement and completeness — that every `(s, d)` band
arrives, exactly once, in the right slot.

## 5. API surface

### 5.1 Fuseable-epilogue enum

All-to-all appends one value to `hipblasLtFuseableEpilogue_t`, whose existing values run to
`SWIGLU = 6`:

```c
typedef enum {
  /* ... existing 0-6: residual add, RMSNorm, partial RMSNorm stats,
     RMSNorm scale-apply, AMax, requant, SwiGLU ... */

  /* Collective family: single stage, not chainable (section 3). */
  HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX = 7,
} hipblasLtFuseableEpilogue_t;
```

The `PREFIX` suffix names the *dispatch criterion* — which output elements are exported, here a
positional run taken from the front of the free-0 axis (`[0, AM)`, section 2). It belongs in the name
rather than in a parameter because other criteria, such as a trailing run or routing by an index
array, would each need their own parameter group; they take their own stage values and this one is not
reused for them.

The numeric value carries no ordering information, and none is needed: a `GEMM + all-to-all` handle
holds exactly one stage. Family membership is derived from the stage value rather than declared by the
caller — the way the existing chain already separates the full and decomposed RMSNorm paths — so the
builder applies the right rule per family: the single-stage rejection of section 3 here, a subsequence
check for the chainable set.

### 5.2 Entry points and the communicator

The builder is the shared one — `hipblasLtFusedEpilogueCreate` / `...Add` / `...SetAttribute` /
`...Destroy` — attached through the `HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE` matmul-descriptor
attribute. All-to-all needs no builder entry point of its own, whether it lands before or after the
other families.

The stage adds one new entry point, at the **library handle** rather than at the fused-epilogue
handle: a communicator registration. It is what tells a rank which rank it is, how many peers it has,
and — through the caller's own allgather — where those peers' flag regions live.

```c
/* A true allgather: rank r's sendbuf (bytesPerRank bytes) must land at
   recvbuf + r*bytesPerRank on every rank, and recvbuf must be readable on
   return. The payload is opaque; the caller must not interpret or reorder it. */
typedef hipblasStatus_t (*hipblasLtDeviceCommAllgatherFn)(
    void* userData, const void* sendbuf, void* recvbuf, size_t bytesPerRank);

hipblasStatus_t hipblasLtSetDeviceComm(
    hipblasLtHandle_t              handle,
    uint32_t                       rank,       // this rank's index; rank < world
    uint32_t                       world,      // 1 <= world <= 8
    uint32_t                       nChannels,  // concurrent operations; see section 5.3
    hipblasLtDeviceCommAllgatherFn allgather,  // must not be NULL
    void*                          userData);  // passed back verbatim; never dereferenced
```

The callback is the only capability the caller hands over, and the reason one entry point covers both
deployment shapes: the library performs no rendezvous of its own — no environment variables, no
ports, no MPI. In a single process the callback is a `memcpy`; across processes it is `MPI_Allgather`
or `torch.distributed`'s store, and nothing else changes. The alternative, a list of device ordinals,
cannot express the multi-process case at all, since a rank's process typically sees one device and has
no ordinal for its peers. Callback and `userData` are used only for the duration of the call.

On return the library has allocated this device's flag state — `nChannels` independent regions — and
filled in the peer addresses. That state is invisible to the caller, neither passed in nor readable,
and its lifetime follows the handle; a region's internal partitioning is private.

Three properties of this call shape the rest of the API:

- **It is collective, and there is one communicator per group.** Every rank calls it on its own
  handle, so a group of `W` ranks makes `W` calls — but they form one communicator, of which each
  handle holds a local view. The call blocks until all `W` have reached it, the allgather being unable
  to return sooner, and the library may invoke the callback more than once, always in the same order
  on every rank. A rank that never registers leaves the others waiting inside the callback, which
  makes this the earliest point a launch group can hang: before a solution has been chosen, let alone
  a kernel launched.
- **Registration is optional, but not a fallback.** A caller that never requests all-to-all need not
  call it, and plain GEMM behaves exactly as before. But an all-to-all stage on a handle with no
  communicator is an error at `hipblasLtMatmul`, not a silent degradation to an unfused GEMM.
- **Exactly once per handle.** A second call fails whether or not its arguments match, making `world`
  immutable — which is not merely convenient. `world` participates in solution selection
  (section 5.7), so if it could change there would have to be a rule for what becomes of an
  already-obtained `algo`: an invalidation rule straddling the handle and the matmul descriptor, a
  shape nothing else here needs. Changing communicators means destroying the handle.

The receive buffer is caller-owned and caller-sized. It holds exactly `W * N * n_shard` elements of
`D`'s type, laid out `[source, token, feature]` with feature contiguous and the *unpadded* `N` as the
source stride, so `recv_d[s, t, f]` sits at element offset `s*(N*n_shard) + t*n_shard + f`. Every term
is already in the caller's hands — `W` from the communicator, `N` from `Ddesc`, `n_shard = AM / W` —
and the kernel never touches anything outside that range, a tail token tile's copy extent being
clamped rather than rounded up. The size is therefore independent of the selected solution and needs
no library query.

### 5.3 All-to-all attributes

The per-operation parameters are set on the fused-epilogue handle, so they travel with the stage that
consumes them. Rank and world are *not* here — they are properties of the communicator (section 5.2),
which is what keeps the `W` handles from disagreeing about them.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES` | `const hipblasLtSdmaQueue_t*` | Required for a solution using the SDMA transport. `W` entries; entry `j` is this device's queue targeting rank `j`, with `j == rank` the loopback queue. Read only by solutions that use that transport. |
| `HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS` | `void* const*` | Required. `W` entries in rank order; entry `j` is the address, *in this process*, of rank `j`'s receive buffer, sized as in section 5.2. |
| `HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT` | `int64_t` | Required. `AM`, the number of leading free-0 (feature) positions of `D` that are redistributed. `n_shard = AM / W`. Participates in solution selection; see the ordering rule below. |
| `HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_COMPLETION_MODE` | `hipblasLtA2ACompletionMode_t` | Optional. `IN_KERNEL` is the only accepted value in this release, and the default. |
| `HIPBLASLT_FUSED_EPILOGUE_COMM_CHANNEL` | `uint32_t` | Optional, default `0`. Which of the communicator's `nChannels` flag regions this operation uses; range `[0, nChannels)`. |

```c
typedef enum {
  HIPBLASLT_A2A_COMPLETION_IN_KERNEL = 0, // default; kernel waits for inbound data

  /* Value 1 is reserved for a deferred mode, in which the kernel retires before
     inbound data lands. It ships together with the primitive a caller would wait
     on, not before it. */
} hipblasLtA2ACompletionMode_t;

/* Plain aggregate. The caller reads these four values out of the queue it
   created and fills them in; the library stores them and hands them to the
   kernel without interpreting any of them. */
typedef struct {
  void* queueBuf; // copy-engine ring base
  void* rptr;     // hardware read pointer
  void* wptr;     // hardware write pointer
  void* doorbell; // doorbell
} hipblasLtSdmaQueue_t;
```

`RECV_PTRS` and `SDMA_QUEUES` are the two attributes carrying `W` entries rather than a
single value, so each is set by passing the address of a host array together with a size of `W`
elements. The library copies the entries into the handle, and the caller's array need not outlive the
call. What the entries *name* must outlive it: the receive buffers for as long as the launch group
runs, and the queues for at least as long, since destroying a queue while its ring has a launch in
flight is undefined.

`COMM_CHANNEL` is the concurrency knob. It carries no `ALL_TO_ALL` marker on purpose — it selects a
slice of communicator state rather than naming a dispatch criterion, so any other communicating stage
inherits it unchanged. It is needed because a fused-epilogue handle is not bound to a stream: two
matmuls built from one handle on two streams would otherwise share a flag region.

The queue array is an attribute of that same handle, so distinct channels and the disjoint queue sets
of section 4.3 collapse into one requirement: `k` concurrent operations need `k` fused-epilogue
handles, each with its own channel and its own `W` queues, so a device provisions `k * W` queues.
`nChannels` is a bound the caller declares, not a resource the library supplies. Operations already
ordered with respect to each other reuse both freely. All `W` ranks of one operation must pass the
*same* channel — the opposite of rank, which differs by construction across those same calls. A
mismatch puts ranks on different regions, undetectably.

Channels do not cover the per-launch counter block, which is not keyed by channel: it lives in the
synchronizer the library handle owns, shared with the rest of the library. Concurrent operations stay
exposed there even with distinct channels and disjoint queues. The stage inherits this rather than
introducing it — Stream-K has the same hazard — and both are masked by the shapes that get tested: a
single process that device-synchronizes between launches, or one card per process.

**Ordering rule.** `n_shard = AM / W` becomes the copy's destination width and pitch, so a solution's
tile is chosen against `AM` and `world` (section 5.7). Both must be fixed before
`hipblasLtMatmulAlgoGetHeuristic`. `world` is, automatically, the communicator being one-shot; `AM` is
an ordinary attribute a caller can set late, and setting it after the heuristic invalidates the
returned `algo`. `AM` has the same standing as `M`, `N`, and `K` — worth stating because arriving
through an attribute rather than a layout makes it easy to mistake for a launch-time parameter.

Completion flags and per-launch counters are absent from the table by design: library-maintained,
never caller-supplied. The two caller-owned resources are caller-owned for different reasons — the
receive buffer because it is the stage's output, the queues because creating them is thunk work the
library declines to link (section 4.3). Keeping the buffer separate from the flag state is what lets
it become a consumed workspace in the mirrored direction of section 8.

### 5.4 Usage sketch

This is one rank's code, and it is the same whether the other `W-1` ranks are sibling threads or
separate processes — only the allgather callback differs. `myRank` and `W` come from the caller's own
launcher.

```c
// One-time, per handle: register the communicator. In a single process the
// callback is a memcpy; under torchrun it is the launcher's own allgather.
hipblasLtSetDeviceComm(handle, myRank, W, /*nChannels=*/1, myAllgather, myCtx);

// The receive buffer's size is computable: W * N * n_shard elements of D's type.
const int64_t am      = 512;      // redistributed feature extent
const int64_t nShard  = am / W;
void*         myRecv  = NULL;
hipMalloc(&myRecv, (size_t)W * N * nShard * sizeof(hipblasLtHalf /* D's type */));

// Exchange recv addresses and build the W queues targeting each peer. Both are
// the caller's job: peerRecv[j] must be rank j's buffer as addressed from THIS
// process, and queues[j] is this device's copy-engine queue aimed at rank j.
void*                peerRecv[8]; // filled by the caller's own exchange
hipblasLtSdmaQueue_t queues[8];   // filled from the caller's queue creation

hipblasLtFusedEpilogueDescriptor_t fused;
hipblasLtFusedEpilogueCreate(&fused);
hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX);
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT,
                                   &am, sizeof(am));
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS,
                                   peerRecv, W * sizeof(void*));
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                                   queues, W * sizeof(hipblasLtSdmaQueue_t));

hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &fused, sizeof(fused));

// Only now is the heuristic meaningful: it selects against AM and world (5.3).
hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc,
                                pref, 1, &heuristic, &returned);

hipblasLtMatmul(handle, matmulDesc, &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc,
                D, Ddesc, &heuristic.algo, workspace, workspaceSize, stream);

// Every rank must have reached its hipblasLtMatmul before any rank synchronizes
// (section 4.2). With one process per rank that is the launcher's property, not
// this code's; with W threads in one process, enqueue all W first.
hipStreamSynchronize(stream);

// After this rank's stream is synchronized:
//   myRecv[s, t, f] holds D_s[myRank*nShard + f, t]  for every source rank s
//   D holds this rank's own full result; features [am, M) are its local tail
hipblasLtFusedEpilogueDestroy(fused);
```

### 5.5 Datatype requirements

The initial release supports a `D` element type of BF16 (`HIP_R_16BF`) only, enforced as a solution
rejection at generation time rather than a runtime branch. The element size is a constant in the
epilogue's descriptor-field arithmetic, scaling base folds from elements to bytes and the pitch and
extent fields from elements to the descriptor's 16-byte packet elements, so another type is not a
slower path — it is wrong addresses. The transport is type-agnostic, its emitter taking all extents
pre-scaled and never seeing a data type, so widening this parameterizes the epilogue's two shifts
rather than changing the packet format. `A`, `B`, and the compute type follow the ordinary matmul
rules; the stage does not constrain them.

The receive buffer's element type is `D`'s element type. It carries no scale or metadata of its own.

### 5.6 Strided-batched semantics

Batched GEMM is not supported by the initial implementation: a matmul whose descriptor implies
`batch_count > 1` with an all-to-all stage attached is rejected. The receive-buffer layout of
section 2 has no batch axis, and adding one is a layout change rather than a relaxation, so it is
deferred to section 8 rather than reserved as an attribute now.

### 5.7 Shape and layout requirements

These follow from the store path and the copy descriptor's field widths (section 4.3). All of them
are checked before launch; none is silently relaxed. `MT0` and `MT1` are the selected solution's
free-0 and free-1 macro tile extents, and `E` is the number of `D` elements in the descriptor's
16-byte addressing granularity (8 for BF16).

| Requirement | Why |
|-------------|-----|
| `1 <= W <= 8` | The kernel's per-peer state is a fixed-size array; the drain wait's lane mask also bounds it. |
| `AM % W == 0` | `n_shard` must be an integer. |
| `AM <= M`, `M % MT0 == 0` | Bands must tile exactly; a partial band would leave the collective's tally short. |
| `n_shard % MT0 == 0`, equivalently `AM % (W * MT0) == 0` | Each destination rank must own a whole number of free-0 macro tiles, so a workgroup's destination is a workgroup constant. |
| `D` free-0 stride `== 1` | The copy moves `n_shard` contiguous elements per token row; a strided feature axis would ship unrelated data. |
| `n_shard % E == 0`, `ldd % E == 0` | Copy widths and pitches are expressed in 16-byte elements and are not rounded. |
| `n_shard / E < 2^14`, `MT1 < 2^14` | Copy extents are 14-bit fields. Equivalently `AM < 8 * E * 2^14`. |
| `ldd / E < 2^19` | The source pitch is a 19-bit field. |
| `MT0, MT1 in {128, 256}` | The only tile shapes the current epilogue generates. |
| Data-parallel solutions only (no Stream-K) | The collective's band tally assumes a work-group covers one tile of one destination band. |
| No split-K, and the caller cannot override it | The drain owner is elected against `NumWorkGroups0 * NumWorkGroups1`, which split-K inflates. See below. |
| Single-XCD-coherent store policy, pre-GFX12 descriptor layout | The current backend's cache and packet-format assumptions; enforced as an architecture gate, not a runtime branch. |

Two of these are worth a word on where and how they are checked. The world-size bound is validated at
`SetDeviceComm`, before any solution exists, so it is one library-wide constant rather than a
per-solution property. The shard-divisibility rule should be implemented in its multiplicative form,
`AM % (W * MT0) == 0`: it needs no division, so it is well defined without first assuming that `AM`
divides by `W`.

Because `n_shard % MT0 == 0` and `AM % W == 0` together imply `AM >= W * MT0`, a four-rank group with
128-wide tiles cannot redistribute fewer than 512 features. A caller that wants a smaller `AM` must
reduce `W` or select a smaller tile.

The split-K restriction is stricter than it looks. Any split-K multiplies the number of work-groups
that arrive at the election, so the drain owner can be chosen before the last band is submitted, and
a split-K resolved at runtime makes the count unknowable when the kernel is generated. Both are
silent wrong answers rather than hangs, so caller-side split-K override is disabled outright on these
solutions rather than validated per launch.

## 6. Error conditions and return codes

| Condition | Return code | Detected at |
|-----------|-------------|-------------|
| Unrecognized stage passed to `Add` | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time |
| Any other stage added to a handle that already contains `all-to-all`, in either order (single-stage family, section 3) | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time |
| Duplicate `all-to-all` stage | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time |
| `rank >= world`, or `world` outside `[1, 8]`, or `allgather` is `NULL` | `HIPBLAS_STATUS_INVALID_VALUE` | `SetDeviceComm` |
| `SetDeviceComm` called a second time on one handle, matching arguments or not | `HIPBLAS_STATUS_INVALID_VALUE` | `SetDeviceComm` |
| Ranks disagree about `nChannels` (compared across ranks through the allgather) | `HIPBLAS_STATUS_INVALID_VALUE` | `SetDeviceComm` |
| The caller's allgather callback fails | its own status, propagated | `SetDeviceComm` |
| Flag-region allocation failed | `HIPBLAS_STATUS_ALLOC_FAILED` | `SetDeviceComm` |
| Unknown attribute, or required attribute explicitly `NULL` | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time |
| `A2A_PREFIX_EXTENT <= 0`, or `COMM_CHANNEL` outside `[0, nChannels)` | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time |
| `COMPLETION_MODE` set to anything other than `IN_KERNEL` | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time |
| All-to-all stage present but shard extent, peer-recv pointers, or SDMA queues unset | `HIPBLAS_STATUS_INVALID_VALUE` | attach (`SetAttribute` of `FUSED_EPILOGUE`) |
| A peer-recv pointer or queue entry in `[0, W)` is `NULL` | `HIPBLAS_STATUS_INVALID_VALUE` | attach |
| All-to-all stage present but no communicator registered on the handle | `HIPBLAS_STATUS_INVALID_VALUE` | heuristic / `hipblasLtMatmul` |
| `AM % W != 0`, `AM > M`, or `batch_count > 1` | `HIPBLAS_STATUS_INVALID_VALUE` | before the heuristic |
| No solution's tile divides `n_shard` (section 5.7) | `HIPBLAS_STATUS_NOT_SUPPORTED` — reported as no usable algo | heuristic |
| Remaining shape or layout requirements of section 5.7 violated | `HIPBLAS_STATUS_INVALID_VALUE` | heuristic / `hipblasLtMatmul` |
| `D` element type other than BF16 | `HIPBLAS_STATUS_NOT_SUPPORTED` | heuristic / `hipblasLtMatmul` |
| Selected device architecture has no fused all-to-all implementation | `HIPBLAS_STATUS_NOT_SUPPORTED` | heuristic / `hipblasLtMatmul` |

Note the code the combination case gets. All-to-all is a single-stage family, not a chain member with
an unimplemented neighbor, so combining it is `INVALID_VALUE` at `Add` time rather than
`NOT_SUPPORTED` at launch: the caller is asking for something the API does not define, and should
learn that at the call expressing the mistake. Legalizing a combination later turns `INVALID_VALUE`
into success, a compatible direction.

The rest of the table encodes one distinction: an *unusable request* is an error, an *ill-fitting
solution* is a filter. Everything decidable from the descriptors and the communicator alone — family
membership, rank range, attribute completeness, `AM % W` — is rejected before a kernel is selected.
Only a tile that fails to divide `n_shard` may present as "no usable algo", since a different tile
might fit. Missing capability is never expressed by filtering.

Three obligations cannot be checked at all, and each is a hang or a wrong answer rather than a return
code:

- The launch-ordering rule of section 4.2. A caller that synchronizes rank `r` before enqueueing rank
  `r+1` deadlocks.
- Channel agreement. Every rank of one launch group must pass the same `COMM_CHANNEL`; a mismatch has
  the ranks operating on different flag regions, which the library cannot see.
- Queue-set disjointness for overlapping operations (section 4.3). The queues are the caller's and the
  library does not track them.

## 7. Interaction with existing descriptors and preferences

- The stage reuses `hipblasLtFusedEpilogueDescriptor_t` and the
  `HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE` attachment point unchanged, and adds no new
  `HIPBLASLT_MATMUL_DESC_*` attribute. It adds no new opaque handle either: the collective state hangs
  off the existing library handle, established by the one new entry point of section 5.2.
- `hipblasLtMatmulPreference_t` is unchanged. The flag regions live in library-owned memory on the
  library handle, sized at registration from `nChannels`, and the per-launch counter state is
  internal, so the workspace-size query is not affected by the stage. The receive buffer is not
  workspace: it is an output tensor the caller allocates and sizes itself (section 5.2).
- Handle destruction within a communicator has no defined order. The flag state holds peer addresses
  and teardown is not collective, so destroying one rank's handle while peers still run is
  unspecified. `IN_KERNEL` completion is what makes the safe case safe: a rank may tear down its own
  state once its stream is synchronized. A deferred mode would have to revisit this.
- Latency accounting changes. Because the kernel drains before retiring, each rank's reported kernel
  time includes its wait for peers, so a per-rank timing that used to measure GEMM work now measures
  the group's slowest member. Callers benchmarking the fused path should compare it against
  `GEMM + separate collective` end to end, not against a bare GEMM kernel time.
- On the codegen side the path comes from a single TensileLite problem-type capability
  (`FusedGemmA2A`), off by default, with the section 5.7 requirements enforced as solution
  rejections at generation time so an unsupported configuration fails to generate rather than
  mis-executes.
- hipBLASLt's build is unchanged and gains no dependency on the kernel-mode thunk (hsakmt) — the point
  of the caller-owned queues of section 4.3, since the thunk is linked by whoever creates a queue.
  Callers using the SDMA transport take that on instead, and the cost is worth naming: the ROCm 7.x
  `hsakmt` package needs CMake repair to link at all, and the caller must map HSA agent enumeration
  onto HIP device ordinals by PCI address, the two runtimes ordering devices differently.
- The C++ extension API (`hipblaslt_ext::GemmEpilogue` / `GemmInputs`) is outside this document; the
  C handle API is the defined surface.

## 8. Extensibility

- **`all-to-all -> GEMM` (the mirrored direction).** The same communicator, completion model,
  and shard indexing apply when the redistribution feeds a GEMM instead of following one. There the
  redistributed buffer is a consumed workspace rather than an output tensor, and the stage attaches
  to the *input* side of the matmul, which is a new attachment point rather than a new epilogue
  stage. Splitting the flag array from the receive pointer in the current ABI is what keeps that
  direction open: the flags stay library state either way, while the buffer changes owner.
- **Deferred completion.** The kernel can already skip its drain barrier under a runtime gate, so
  the device side of a deferred mode exists. What it needs is the other half: a library primitive
  that waits on receive completion — an entry point that blocks, or one that enqueues a wait on a
  caller's stream — because the arrival flags stay library-owned and the copy engine's queues are
  outside HIP's stream model. Adding the mode means adding that primitive and defining value `1` of
  `hipblasLtA2ACompletionMode_t`; nothing else in this design changes.
- **Other collectives.** Reduce-scatter is the same routing with an accumulate-on-arrival rule, and
  all-gather is the same routing with `n_shard = AM`. Both fit as further members of the collective
  family, reusing the communicator and the same attribute set; neither needs new enum combinations.
  `COMM_CHANNEL` is already named for this — it selects communicator state rather than anything
  all-to-all-specific, so a second communicating stage inherits it unchanged.
- **Replacing the transport.** The queue attribute of section 5.3 puts SDMA on the public surface, the
  price of keeping hsakmt out of the library's build. It bounds a later release without blocking it: a
  solution on some other mechanism — CU-side remote stores, say — never reads the attribute, so the two
  coexist and it goes vestigial rather than obstructive. What is lost is changing transports
  *silently*; retiring SDMA outright would be an API removal.
- **Composing with the chainable family, as a follow-up.** Making `chainable stages -> all-to-all`
  legal is an additive change: the enum value and the attributes already exist, and the builder rule
  of section 3 relaxes from "reject any companion" to "accept a legal chain prefix, all-to-all
  last". It needs a kernel first, because three choices are not derivable from the stage order alone:
  whether the exported band is the pre- or post-epilogue value; whether the chain's cross-tile
  reduction and the collective's band-ownership election share one handshake or nest (which the
  completion mode of section 4.2 would expose); and how a requantized band's scale metadata travels,
  given that the receive layout of section 2 has no room for it. The reverse order (`all-to-all`
  first, then elementwise work on a value that has already left the device) stays permanently
  illegal.
- **Larger world sizes and multi-node.** `W <= 8` is an implementation limit, not an API one — the
  kernel's per-peer state is a fixed-size array and `world` is already a `uint32_t`. Multi-node needs
  nothing new from the API: the communicator exchanges only through the caller's callback, so a
  node-spanning group differs only in what that callback does. What gates it is the transport's ability
  to reach a peer's memory.
- **Batching** needs a batch axis in the receive layout of section 2 and a batch-aware band tally;
  both are additive to this design.

## 9. Mapping from the TensileLite validation client

The client in [PR #10925](https://github.com/ROCm/rocm-libraries/pull/10925)
(`tensilelite/client/src/FusedA2AClient.cpp`) hands the kernel a flat kernarg segment. This design
partitions those fields by owner, which is the substance of the API extension:

| Client / kernarg | API equivalent | Owner |
|------------------|----------------|-------|
| `FusedW` | `world` on the communicator | caller, at `SetDeviceComm` |
| `FusedMyRank` | `rank` on the communicator | caller, at `SetDeviceComm` |
| `FusedAM` | `A2A_PREFIX_EXTENT` | caller, before the heuristic |
| `FusedDrain` | `A2A_PREFIX_COMPLETION_MODE` | library; always `1` in this release, since `IN_KERNEL` is the only accepted mode |
| `peer_j_recvPtr` | `A2A_PREFIX_RECV_PTRS[j]` | caller; size computed as in section 5.2 |
| `peer_j_queueBuf` / `rptr` / `wptr` / `doorbell` | `A2A_PREFIX_SDMA_QUEUES[j]` | caller; four addresses the caller reads from KFD |
| `peer_j_flagPtr` | — | library, on the library handle, one region per channel |
| — | `COMM_CHANNEL` | caller; the client has no equivalent, since it runs one operation at a time |
| `counter_ptr` (counters, cursors, guard tail) | — | library, per launch |
| `n_shard`, `tilesPerRank`, `tokenTiles`, macro-tile derivations | — | library, derived from `AM`, `W`, and the selected solution |
| client-side shape and packet-field guards | section 5.7 requirements, section 6 return codes | library, at attach / heuristic / launch |
| explicit peer-access enable | caller's responsibility, as part of making `peerRecv[j]` addressable | caller |
| per-peer queue creation and teardown | caller, before `SetAttribute` | caller |
| "enqueue all `W` before synchronizing" | section 4.2 launch ordering rule | caller obligation, undetectable |

The client's dual-segment check is also the shape of the API's correctness contract: after a
synchronized launch group, each rank's `recv` must match every source rank's exported bands, and each
rank's `D` must match its own full local result including the untouched local tail `[AM, M)`.
