# Why RstdScale has no UseRstdScaleEqual predicate

## What a predicate would do

In the TensileLite library-selection pipeline, a *problem predicate* gates
dispatch: a solution is eligible only when every predicate it declares matches
the runtime `ContractionProblem`. A hypothetical `UseRstdScaleEqual` predicate
(mirroring the existing `UsePartialRMSEqual` one) would match only when the
caller sets `useRstdScale = true` on the problem, preventing an RstdScale
kernel from being dispatched for an ordinary GEMM call.

## Why it is omitted

RstdScale solutions are placed in dedicated LibraryLogic files (e.g.
`RstdScale_BF16_TN.yaml`) that live in a separate directory tree from regular
GEMM logic files. `TensileCreateLibrary` compiles them into a distinct device
library (`.hsaco`/`.co`/`.dat`), and the hipBLASLt host loads that library only
when the caller explicitly invokes the RstdScale API path (i.e. calls
`setUseRstdScale(true)` and populates `rstdBuf`).

Because no ordinary GEMM library will ever contain an RstdScale solution, the
predicate is redundant — the segregation of logic files provides the same
guarantee with less machinery. This is the same convention used by PartialRMS
(K1): it has no predicate in the current HEAD either.

## Residual risk

If an RstdScale logic file were inadvertently merged into a regular GEMM
library, the kernel could be dispatched without a valid `rstdBuf` pointer.
The epilogue would then load from address zero (or a stale pointer), producing
garbage output or, on address-space-protected hardware, a GPU fault.

**Mitigation**: keep RstdScale logic files in their own directory tree and never
mix them with regular GEMM logic. If that invariant becomes hard to enforce in
a future integration (e.g. a unified multi-epilogue library), adding the
`UseRstdScaleEqual` predicate to `Contractions.py` and the corresponding
`ContractionPredicates.hpp` entry is a straightforward follow-up — the
`useRstdScale` slot is already present in `ProblemType.__slots__` and
`SizeMapping`.
