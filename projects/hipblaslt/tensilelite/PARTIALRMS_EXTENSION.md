# Fused GEMM + Partial RMSNorm Epilogue (K1 Kernel) — Technical Reference

This document describes the full implementation of the fused GEMM + partial
RMSNorm epilogue added in commit `c8693ebad5e346e220c57ec14bf9099c70d28e67`.
It covers the algorithm, all new Python and C++ source additions, the kernel
argument layout, the client-side plumbing, the standalone second-phase kernel,
and the tooling.

---

## 1. What the Feature Does

### 1.1 Algorithm

The feature implements Phase 1 (K1) of a two-kernel RMSNorm pipeline for
large-language-model inference:

**Phase 1 — K1 (fused GEMM + PartialRMS epilogue, this feature):**

```
h1[m, n]             = A^T @ W   (fp32 MFMA accumulator)
partialBuf[m, t]     = Σ_{n in tile t} h1[m, n]²    (raw sum of squares)
D[m, n]              = bf16( alpha * h1[m, n] * gamma[n] + beta * C[m, n] )
```

If `PartialRMSResidualAdd` is also enabled:

```
h1[m, n]             = A^T @ W + R[m, n]    (residual tensor R added before squaring)
partialBuf[m, t]     = Σ_{n in tile t} h1[m, n]²
D[m, n]              = bf16( alpha * h1[m, n] * gamma[n] + beta * C[m, n] )
```

**Phase 2 — K2 (`partial_rms_epilogue`, separate kernel):**

```
rstd[m]     = rsqrt( (Σ_t partialBuf[m, t]) / N_hidden + eps )
D[m, n]    *= rstd[m]   (in-place update of D)
```

### 1.2 Output Buffers

| Buffer | Type | Layout | Filled by |
|---|---|---|---|
| D | bf16 | col-major [M × N_hidden] | K1 store path |
| partialBuf | fp32 | row-major [M_padded × N_tiles_N] | K1 epilogue |

`N_tiles_N = ceil(N_hidden / MT1)` — the number of workgroups along the N axis.
Each workgroup owns one `MT1`-wide N-tile and writes one `fp32` partial sum per
row to column `WorkGroup1` of `partialBuf`.

K1 does **not** divide by `N_hidden`. That normalization happens in K2.

### 1.3 Fitting K1 into the Two-Kernel Pipeline

The kernel is restricted to the Subtile path on gfx950 with bf16 data. The
`StreamKForceDPOnly=1` constraint ensures every workgroup completes a full tile
before entering the epilogue, so the accumulator is final (no K-split partial
results to merge).

The host launches K1 to produce D and partialBuf, then launches K2 (`partial_rms_epilogue`)
to normalize D in-place using partialBuf.

---

## 2. Solution Parameters

### 2.1 New Parameters

Two new boolean solution parameters were introduced:

| Parameter | Default | File | Meaning |
|---|---|---|---|
| `PartialRMS` | `False` | `Tensile/Common/ValidParameters.py` line 450 | Enable the K1 fused epilogue |
| `PartialRMSResidualAdd` | `False` | `Tensile/Common/ValidParameters.py` line 451 | Add a bf16 col-major residual tensor before squaring |

Both appear in `ValidParameters.py` as boolean options:
```python
"PartialRMS": [False, True],
"PartialRMSResidualAdd": [False, True],
```

Both are declared as required parameters (serialized in solution names and YAML)
in `Tensile/Common/RequiredParameters.py` lines 167–168.

Their defaults in `Tensile/Common/GlobalParameters.py` are:
```python
{"PartialRMS": [False]},
{"PartialRMSResidualAdd": [False]},
```

### 2.2 Validation (`_validatePartialRMS` in Solution.py)

The validator at `Tensile/SolutionStructs/Solution.py` line 215 enforces:

- `UseSubtileImpl` must be `True`.
- ISA must be gfx950 `(9, 5, 0)`.
- Data type must be bf16.
- `StreamKForceDPOnly` must be `True` (complete tiles, no K-split fixup).
- `MIArchVgpr` must be `False` (emitter uses AGPR instructions).
- `PartialRMSResidualAdd` requires `PartialRMS`.
- Mutually exclusive with `RstdScale`.
- `MacroTile1 > 0`.
- No `OutputAmaxD` (kernarg layout conflict).
- No `MultipleBufferSingleKernel`/`AdaptiveGemmGSUA`.
- No `GroupedGemm`.
- Each wave must own at least one N-tile: `MacroTile1 // (MatrixInstN * MIWaveGroup[1]) >= 1`.
- When `MIWaveGroup[1] > 1`: `MIWaveGroup[0]` must be a power of two (bitmask
  arithmetic in the cross-wave reduction). LDS requirement
  (`wg[0] * wg[1] * WavefrontSize * num_rows * 4` bytes) must not exceed `MaxLDS`.

LDS accounting (Solution.py line 5294): when `MIWaveGroup[1] > 1`, the LDS
footprint is boosted to `max(main_loop_lds, partialRMS_lds_bytes)` because the
cross-wave scratch region reuses the main-loop LDS freed at the epilogue.

---

## 3. Problem/Contraction-Level Changes

### 3.1 `Tensile/Contractions.py`

`ProblemType.StateKeys` (line 76) now includes `'usePartialRMS'` and
`'partialRMSResidualAdd'`. `FromOriginalState` (lines 249–250) reads:

```python
rv.usePartialRMS = bool(d.get('UsePartialRMS', False))
rv.partialRMSResidualAdd = bool(d.get('PartialRMSResidualAdd', False))
```

`ProblemType.FromOriginalState` also sets them in the full-form constructor
(lines 752–753):
```python
PartialRMS               = bool(d.get('PartialRMS', False)),
PartialRMSResidualAdd    = bool(d.get('PartialRMSResidualAdd', False)),
```

### 3.2 `Tensile/SolutionStructs/Problem.py`

Adds `'PartialRMS'` and `'PartialRMSResidualAdd'` to the problem-type key list
(lines 655–656).

### 3.3 C++ `include/Tensile/ContractionProblem.hpp`

Three new tensor slots were added to `ContractionProblemGemm::TENSOR` (lines 351–353):
```cpp
RMSGAMMA      = 17, // bf16 input: per-column gamma weight (N_hidden elements)
PARTIALBUF    = 18, // f32 output: partial Σx² [M_padded × N_tiles_N] row-major
RESIDUAL      = 19, // bf16 input: col-major residual [M × N_hidden] (optional)
```

Four new problem fields and accessors (lines 764–774):
```cpp
void setUsePartialRMS(bool v);
void setPartialRMSResidualAdd(bool v);
void setPartialRMSMT0(size_t v);
void setPartialRMSMT1(size_t v);

bool   usePartialRMS()         const;
bool   partialRMSResidualAdd() const;
size_t partialRMSMT0()         const;
size_t partialRMSMT1()         const;
```

Private members (lines 1471–1474):
```cpp
bool   m_usePartialRMS           = false;
bool   m_partialRMSResidualAdd   = false;
size_t m_partialRMSMT0           = 0;
size_t m_partialRMSMT1           = 0;
```

Tensor setup helpers (lines 946–967):
- `setRMSGamma(rocisa::DataType type, size_t nHidden)` — sets `RMSGAMMA` tensor.
- `setPartialBuf(size_t mPadded, size_t nTilesN)` — sets `PARTIALBUF` tensor, marks it as output.
- `setResidual(rocisa::DataType type, size_t M, size_t nHidden)` — sets `RESIDUAL` tensor;
  layout is col-major `[M, N_hidden]` with strides `{1, M}`.

`ContractionInputs` (line 1612) gained:
```cpp
void*       partialBuf = nullptr;
void const* rmsGamma   = nullptr;
void const* residual   = nullptr;
```

### 3.4 C++ `include/Tensile/ContractionSolution.hpp`

`SizeMapping` (lines 184–185) gained:
```cpp
bool partialRMS            = false;
bool partialRMSResidualAdd = false;
```

### 3.5 C++ `src/ContractionProblem.cpp`

No significant additions beyond wiring of the new problem fields.

### 3.6 C++ `src/ContractionSolution.cpp`

Kernel argument appending (lines 1115–1121):
```cpp
if(sizeMapping.partialRMS)
{
    args.template append<void const*>("RMSNormGamma", inputs.rmsGamma);
    args.template append<void*>      ("PartialBuf",   inputs.partialBuf);
    if(sizeMapping.partialRMSResidualAdd)
        args.template append<void const*>("ResidualBuf", inputs.residual);
}
```

---

## 4. The Emitter — `SubtilePartialRMSEmit.py`

**File:** `Tensile/Components/Subtile/SubtilePartialRMSEmit.py`

### 4.1 Class Overview

`SubtilePartialRMSEmitter` emits the entire PartialRMS epilogue as a single
rocisa `Module`. It is instantiated from `KernelWriter.py` immediately after
the `dtileInfo` is known, before the main global-write path.

```python
class SubtilePartialRMSEmitter:
    def __init__(self, writer, kernel): ...
    def emit(self, accVgprBase: int) -> Module: ...
```

`accVgprBase` is the AGPR index of the first D-tile accumulator element
(obtained from `dtileInfo.vgprTiles[0].regList.indices[0]`).

### 4.2 Geometry Derivation

All tile geometry is derived from kernel parameters in `__init__`:

```python
self.mfma_m = kernel["MatrixInstM"]        # 16
self.mfma_n = kernel["MatrixInstN"]        # 16
self.waveSize = kernel["WavefrontSize"]    # 64
self.rows_per_lane = (mfma_m * mfma_n) // waveSize   # = 4 for 16x16 MFMA, wave64

wg = kernel["MIWaveGroup"]
self.wg_m = wg[0]  # waves along M
self.wg_n = wg[1]  # waves along N

self.mma_m = (kernel["MacroTile0"] // mfma_m) // wg_m  # MMA repetitions per wave (M)
self.mma_n = (kernel["MacroTile1"] // mfma_n) // wg_n  # MMA repetitions per wave (N)
self.numRows = mma_m * rows_per_lane                    # VGPR rows per wave
```

### 4.3 AGPR Indexing

Acc VGPRs are ordered N-outer, M-inner:
```python
def _acc_idx(self, base, m, n, k):
    return base + (n * self.mma_m + m) * self.rows_per_lane + k
```

For a 16×16 MFMA with wave64: `rows_per_lane = 4`. AGPR index for element
`(m=0, n=1, k=2, base=0)` is `(1*mma_m + 0)*4 + 2`.

### 4.4 Register Allocation

#### VGPRs (allocated from `vgprPool`):

| Name | Size | Purpose |
|---|---|---|
| `partials` | `numRows` | Per-row partial Σx² (one fp32 per row per wave) |
| `accTmp` | 1 | Scratch for reading/writing AGPRs |
| `gammaTmp` | 1 | Gamma value after bf16→fp32 conversion |
| `laneId` | 1 | `Serial & (waveSize - 1)` |
| `colByte` | 1 | Byte offset into gamma buffer for this lane's column |
| `globalAddr` | 1 | Address scratch for partialBuf writes |
| `resTmp` | 1 | Residual loaded value (residual path only) |
| `resAddr` | 1 | Byte address for residual buffer load (residual path only) |
| `resColM` | 1 | Residual column × M (residual path only) |
| `resRowVgpr` | 1 | Global row index (residual path only) |

#### SGPRs (allocated from `sgprPool`, alignment 4):

| Name | Size | Purpose |
|---|---|---|
| `gammaSrd` | 4 | Gamma buffer SRD (s[n:n+3]) |
| `partialSrd` | 4 | partialBuf SRD (s[n:n+3]) |
| `savedExec` | 2 | Saved EXEC mask before narrowing |
| `laneMaskSgpr` | 2 | Writing-lane predicate |

All pool allocations use `preventOverflow=False` because the epilogue is
temporary (live only during the epilogue) and the SGPR budget has been verified
by the validator.

### 4.5 Execution Sequence

`emit()` calls five sub-methods in order:

```
_setup()               — build SRDs, derive laneId, colByte
_addResidual()         — (optional) load bf16 residual, add to AGPRs
_squareAndLaneSum()    — per-row Σx² from AGPRs across N-tiles
_butterflyReduce()     — intra-wave DPP reduction across mfma_n column lanes
_crossWaveReduce()     — (when wg_n > 1) LDS cross-wave reduction
_writePartials()       — predicated 2D write to partialBuf
_applyGammaOnly()      — bf16 gamma load, fp32 multiply, write back to AGPRs
```

### 4.6 `_setup` — SRDs and Column Byte Offset

Builds the gamma SRD from the named SGPR `RMSNormGamma` and the partialBuf SRD
from `PartialBuf`. Both use `BufferOOB` as the limit and `Srd127_96` as the
upper-word flags.

`colByte` encodes the per-lane byte offset into the gamma buffer:
```
colByte = (laneId % mfma_n) * 2
        + waveN_base_bytes          (when wg_n > 1)
        + WorkGroup1 * MT1 * 2
```

When `wg_n > 1`, the wave's N-column base (`waveN = Serial / (waveSize * wg_m)`)
contributes `waveN * mma_n * mfma_n * 2` bytes to shift to the correct N-wave's
gamma columns.

### 4.7 `_addResidual` — Residual Addition

Adds a bf16 col-major residual tensor R to each accumulator element before the
squaring step. Layout: `R[globalRow, globalCol]` at byte offset
`(globalCol * M + globalRow) * 2`.

The method temporarily **reuses the gammaSrd SGPR block** for the residual SRD
(safe because gamma is not needed until `_applyGammaOnly`). The gamma SRD is
rebuilt at the end of `_addResidual` before returning.

The residual SRD's `NumRecords` field is set to `M * N * 2` bytes so that loads
for out-of-bounds rows (OOB for non-tile-aligned M) return zero.

Each accumulator element `(m, n, k)` gets: `acc[m,n,k] += bf16_to_fp32(R[globalRow, globalCol])`.

### 4.8 `_squareAndLaneSum` — Per-Row Partial Σx²

For each row position `(m, k)`, reads AGPR elements across all N-tiles and
accumulates:
```
partials[m*rows_per_lane+k] = acc[m,0,k]² + acc[m,1,k]² + ... + acc[m,mma_n-1,k]²
```

First N-tile uses `VMulF32` (to initialize), subsequent tiles use `VFmaF32`
(`partial += acc²`). All reads use `VAccvgprReadB32`.

### 4.9 `_butterflyReduce` — Intra-Wave DPP Reduction

Reduces `partials` across the `mfma_n = 16` column-sharing lanes within each
16-lane MFMA row group using `VAddF32` with DPP `row_shr` modifiers.

Strides descend as `[8, 4, 2, 1]` (for mfma_n=16). Each stage shifts lane `j`
by `stride` positions so lane `j-stride`'s value becomes visible. `bound_ctrl=0`
zeros out-of-row lanes, preventing cross-group contamination.

After all stages, the last lane in each row group (`laneId % mfma_n == mfma_n-1`)
holds the full sum for that group. These are the writing lanes.

When `numRows == 1`, explicit `SNop(waitState=0)` waits are inserted between
successive DPP rounds on the same register.

### 4.10 `_crossWaveReduce` — Cross-Wave LDS Reduction

Only executed when `wg_n > 1`. After the butterfly, each wave holds the Σx²
for its N-column slice. This step combines `wg_n` sibling waves through LDS.

**Wave-id convention:** `waveId = waveN * wg_m + waveM`.

**LDS layout:** `waveSize * numRows * 4` bytes per wave slot. Each lane's slot
within a wave is at `lane * numRows * 4` bytes past the wave's base.

Procedure:
1. Barrier to ensure prior LDS users are done.
2. Each wave writes its `partials[0..numRows-1]` to `LDS[waveId * strideW + lane * laneSlotBytes + i*4]`.
3. `s_waitcnt dscnt=0` + barrier.
4. Read all `wg_n` wave slots and accumulate into `partials`.
5. Final barrier.

### 4.11 `_writePartials` — Predicated 2D Write to partialBuf

Computes `N_tiles_N = ceil(SizesFree[1] / MT1)` on-device using integer
arithmetic:
```asm
s_add_u32 nTilesS, SizesFree+1, MT1-1
s_lshr_b32 nTilesS, nTilesS, log2(MT1)
v_mov_b32 ntilesVgpr, nTilesS
```

**Writing-lane predicate:** `laneId % mfma_n == mfma_n - 1` (the last lane in
each row group, which holds the DPP-reduced Σx²). Built into `laneMaskSgpr`
via `VCmpEQU32`.

**EXEC narrowing:** `SAndSaveExecB64` saves EXEC and sets it to `laneMaskSgpr`.
Stores are issued under the narrowed EXEC. `SMovB64 EXEC, savedExec` restores it.

**2D address:** for each `(m, k)`:
```
globalRow = WorkGroup0 * MT0 + m * mfma_m + k + rowGroup * rows_per_lane
byteOff   = (globalRow * N_tiles_N + WorkGroup1) * 4
```

`WorkGroup1` is used directly as the tile-column index (named SGPR, always live).

When `wg_m > 1`, `wgRowBase` is augmented by `waveM * mma_m * mfma_m` so each
M-wave addresses its own row slice.

VOP3 `v_add_u32` only supports inline constants (0–64). When `mBase = m*mfma_m+k`
exceeds 64 (large macro-tiles), a `VMovB32` materializes it into a VGPR first.

### 4.12 `_applyGammaOnly` — Gamma Application

For each N-tile `n`, loads `gamma[n*mfma_n + (laneId%mfma_n)]` as bf16 from
the gamma SRD using `BufferLoadD16B16` with `offen=True` and a static
`offset12 = n * mfma_n * 2`. Converts to fp32 with `VCvtBF16toFP32`, then
multiplies all accumulator elements for that N-tile:
```
for (m, k): acc[m, n, k] *= gamma_fp32
```

This modifies the AGPRs in-place. The downstream global-write path then stores
D as bf16 (standard `alpha * acc + beta * C`).

Note: no `rstd` multiply happens here. K2 applies the normalization factor.

---

## 5. KernelWriter Integration

**File:** `Tensile/KernelWriter.py`, line 5143.

The emitter is invoked in the non-LocalSplitU path, just before the main
global-write index computation:

```python
if kernel["PartialRMS"]:
    from .Components.Subtile.SubtilePartialRMSEmit import SubtilePartialRMSEmitter
    module.addComment1("PartialRMS: fused partial sum-of-squares + gamma epilogue")
    pRmsEmitter = SubtilePartialRMSEmitter(self, kernel)
    dtile_agpr_base = dtileInfo.vgprTiles[0].regList.indices[0] if dtileInfo.vgprTiles else 0
    module.add(pRmsEmitter.emit(dtile_agpr_base))
```

The SGPR store-register loading list (line 9789) appends:
- `"RMSNormGamma"` (2 SGPRs, 64-bit ptr).
- `"PartialBuf"` (2 SGPRs, 64-bit ptr).
- Optionally `"ResidualBuf"` (2 SGPRs) when `PartialRMSResidualAdd`.

---

## 6. The Kernel Signature

**File:** `Tensile/Components/Signature.py`, line 316.

### 6.1 `UserArgumentsInfo`

A new field was added to the dataclass:
```python
rmsNormSize: int = 0
```

This tracks the byte size of the PartialRMS-specific kernel arguments appended
after the activation block.

### 6.2 Argument Append Order

When `kernel["PartialRMS"]` is true, `SignatureDefault.__call__` appends these
arguments in order:

| Slot | Name | Type | Size | Condition |
|---|---|---|---|---|
| N+0 | `RMSNormGamma` | bf16 global buffer ptr | 8 B | always |
| N+1 | `PartialBuf` | f32 global buffer ptr | 8 B | always |
| N+2 | `ResidualBuf` | bf16 global buffer ptr | 8 B | `PartialRMSResidualAdd` only |

`rmsNormSize = 16` (or 24 with residual). This size is added to `totalSize`.

`gammaValueType` is always `"bf16"` (the validator enforces `isBFloat16()`).

### 6.3 Total Argument Layout

```
gemmArgumentSize      (A, B, C, D ptrs + strides + sizes + alpha/beta)
+ scaleASize
+ scaleBSize
+ scaleCSize
+ scaleDSize
+ scaleAlphaVecSize
+ biasSize
+ factorDimSize
+ eSize
+ activationSize
+ rmsNormSize         ← new
= totalSize
```

---

## 7. Full Kernel Argument List (End-to-End)

The kernel arguments are appended in `ContractionSolution.cpp` via the
`KernelArguments` helper. The PartialRMS arguments come at the very end of the
user-argument block, after any activation arguments:

```cpp
// ... standard GEMM args (A, B, C, D, strides, sizes, alpha, beta) ...
// ... optional scale/bias/activation args ...

// PartialRMS args (appended only when sizeMapping.partialRMS):
args.append<void const*>("RMSNormGamma", inputs.rmsGamma);    // 8 bytes, bf16 ptr
args.append<void*>      ("PartialBuf",   inputs.partialBuf);  // 8 bytes, f32 ptr
// PartialRMSResidualAdd args:
args.append<void const*>("ResidualBuf",  inputs.residual);    // 8 bytes, bf16 ptr
```

The SGPR names `RMSNormGamma`, `PartialBuf`, and `ResidualBuf` match exactly
between the signature (Python) and the store-SGPR list (Python), and between
those and the `args.append` calls (C++).

---

## 8. Library-Logic Generation

### 8.1 Pre-Built YAML

`tools/gemm_partial_rms_k1.yaml` is a single-solution TensileLite benchmark
config for a 256×256×64 MT bf16 TN GEMM with PartialRMS. Key parameters:

| Parameter | Value |
|---|---|
| `DataType` | `b` (bf16) |
| `DestDataType` | `b` |
| `ComputeDataType` | `s` (fp32) |
| `TransposeA` / `TransposeB` | `True` / `False` (TN) |
| `MatrixInstruction` | `[16, 16, 32, 1, 1, 4, 8, 4, 2]` (MI16×16, wg0=4, wg1=2) |
| `MacroTile0` | 256 |
| `MacroTile1` | 128 |
| `DepthU` | 64 |
| `StreamK` | 3 |
| `StreamKForceDPOnly` | 1 |
| `PartialRMS` | `True` |
| `UseSubtileImpl` | `True` |

### 8.2 `gen_partialrms_logic.py`

Generates a LibraryLogic YAML from the K1 config and writes it under
`<out-dir>/<chip>/Equality/PartialRMS_BF16_TN.yaml`.

Steps:
1. `setup_tensile(chip)` — initializes the rocisa TI singleton.
2. `build_k1_solution(chip, assembler, isaInfoMap)` — builds and validates the
   K1 solution from the YAML.
3. Extracts the expanded solution state via `sol.getKernels()[0]`.
4. `_to_primitive()` — recursively converts Tensile-typed objects to YAML-safe
   Python primitives (handles `DataType`, `ActivationType`, `SemanticVersion`,
   `ISA` namedtuple).
5. Three mandatory type coercions:
   - `BufferStore` → `bool`
   - `GlobalReadPerMfma` → `float`
   - `StaggerUStride` → `int`
6. Writes a 12-element LibraryLogic list and verifies with a round-trip check.

Usage:
```bash
source ~/.tensile/bin/activate
python tools/gen_partialrms_logic.py --chip gfx950 --out-dir /tmp/partialrms_logic
```

### 8.3 Compiling with TensileCreateLibrary

```bash
python -m Tensile.TensileCreateLibrary \
    --architecture=gfx950 \
    --cxx-compiler=$(which amdclang++) \
    --jobs=$(nproc) \
    /tmp/partialrms_logic/ \
    /tmp/partialrms_output/ \
    HIP
```

Output: `Kernels.so-000-gfx950.hsaco`, `TensileLibrary_..._gfx950.co`,
`TensileLibrary_..._gfx950.dat.zlib`, plus lazy-load index files.

---

## 9. Client-Side Changes

### 9.1 New CLI Flags (`client/main.cpp`, lines 407–411)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--use-partial-rms` | `bool` | `false` | Enable PartialRMS epilogue |
| `--partial-rms-residual-add` | `bool` | `false` | Enable residual addition |
| `--partial-rms-mt0` | `size_t` | `0` | MacroTile0 override (0 = use 16) |
| `--partial-rms-mt1` | `size_t` | `0` | MacroTile1 override (0 = use 16) |
| `--init-rmsGamma` | `InitMode` | `Random` | Init mode for gamma buffer |
| `--init-partialBuf` | `InitMode` | `Zero` | Init mode for partialBuf output |

### 9.2 `ClientProblemFactory` (`client/src/ClientProblemFactory.cpp`)

Reads the flags into private fields (lines 190–196):
```cpp
m_usePartialRMS         = args["use-partial-rms"].as<bool>();
m_partialRMSResidualAdd = args["partial-rms-residual-add"].as<bool>();
m_partialRMSMT0Override = args["partial-rms-mt0"].as<size_t>();
m_partialRMSMT1Override = args["partial-rms-mt1"].as<size_t>();
```

Problem construction (lines 392–464) when `m_usePartialRMS`:
1. Calls `rv.back().setUsePartialRMS(true)` and `setPartialRMSResidualAdd(...)`.
2. Computes:
   - `mPadded = ceil(M / mt0) * mt0`
   - `nTilesN = ceil(nHidden / mt1)`
3. Calls `setRMSGamma(bf16Type, nHidden)` and `setPartialBuf(mPadded, nTilesN)`.
4. Optionally calls `setResidual(bf16Type, M, nHidden)`.

`ClientProblemFactory.hpp` declares:
```cpp
bool        m_usePartialRMS         = false;
bool        m_partialRMSResidualAdd = false;
size_t      m_partialRMSMT0Override = 0;
size_t      m_partialRMSMT1Override = 0;
```

### 9.3 `DataInitialization.cpp`

`partialBuf` is initialized to zero (matching `--init-partialBuf Zero`).
`rmsGamma` is initialized with random data (matching `--init-rmsGamma Random`).

### 9.4 Reference Computation (`client/src/Reference.cpp`)

The reference implementation has two parts:

**D output** (line 2037): When `usePartialRMS()`, the reference computes:
1. `hF = float(A*B)` — raw GEMM accumulator value.
2. If `partialRMSResidualAdd`: `hF += float(R[nCoord * M + mCoord])` (col-major R).
3. `gammaVal = float(gamma[nCoord])`.
4. `dVal = alpha * hF * gammaVal + beta * C[cIndex]`.
5. Store `bf16(dVal)` to D.

**partialBuf** (line 2084): A separate pass after the D loop recomputes Σx² for
each validated `(m_row, t_col)` position:
```cpp
tileSum = Σ_{n=t_col*mt1}^{min((t_col+1)*mt1, N)-1} (A*B)[m_row, n]²
```
If `partialRMSResidualAdd`, the residual value is added before squaring.
This is computed correctly for every validated position rather than attempting
to accumulate during the sparse D validation loop.

### 9.5 `ReferenceValidator.cpp`

Validation line 513 checks both `partialBuf` and D:
```cpp
refPtr = reference.partialBuf;
resPtr = result.partialBuf;
```
Both buffers are validated with the same relative/absolute tolerances.

### 9.6 `ClientWriter.py`

`writeClientConfigIni` (line 580) accepts:
```python
partialRMSMT0=0, partialRMSMT1=0, anyPartialRMSResidualAdd=False
```
and writes `use-partial-rms`, `partial-rms-residual-add`, `partial-rms-mt0`,
`partial-rms-mt1` into the client INI.

Additional consistency checks (lines 803–829):
- When any solution has `usePartialRMS`, all solutions must have the same `MT1`
  (so `N_tiles_N` is consistent across the benchmark pass).
- `PartialRMSResidualAdd` must be uniform across all solutions in a benchmark
  pass.

---

## 10. `PartialRmsEpilogueGenerator.py` — The K2 Kernel

**File:** `PartialRmsEpilogueGenerator.py` (root of tensilelite/).

### 10.1 Purpose

Generates the standalone `partial_rms_epilogue` kernel (K2). This kernel reads
the `partialBuf` written by K1 and applies RMSNorm in-place to the bf16
col-major D matrix.

### 10.2 M Limit

```python
_MAX_M = 32767
```

The column-address computation uses `SMulI32` (signed 32-bit multiply):
`col * M * 2`. For M > 32767, this overflows. The generator raises `ValueError`
at construction time.

### 10.3 Kernel Configuration

| Property | Value |
|---|---|
| Kernel name | `partial_rms_epilogue` |
| Workgroup | 256 threads (4 waves × 64 lanes) |
| Wavefront size | 64 (wave64 only) |
| LDS | 1024 bytes (4 waves × 64 iterations × 4 bytes) |
| Grid X | `ceil(M / 256)` |
| Grid Y | `ceil(N / 256)` |

### 10.4 Kernarg Layout (offsets fixed)

| Offset | Size | Name | Kind | Description |
|---|---|---|---|---|
| 0 | 8 B | `ptrC` | `global_buffer` | bf16 col-major C = D [M × N] |
| 8 | 8 B | `ptrD` | `global_buffer` | f32 row-major partialBuf [M × nD] |
| 16 | 4 B | `M` | `by_value` | Row count |
| 20 | 4 B | `N` | `by_value` | Column count |
| 24 | 4 B | `nD` | `by_value` | Number of columns in partialBuf (= N_tiles_N) |
| 28 | 4 B | `invD` | `by_value` | `1 / N_hidden` (f32) |
| 32 | 4 B | `eps` | `by_value` | Epsilon for numerical stability (f32) |

Total: 40 bytes (aligned to 8).

### 10.5 ABI Register Assignments

```
s0:s1   = kernarg segment pointer
s2      = wgIdX (selects 256-row tile)
s3      = wgIdY (selects 256-column tile)
v0      = Serial (thread ID within block)
```

On architectures with `WorkGroupIdFromTTM`, `s2 = ttmp9`.

### 10.6 Algorithm

**Phase 1 — Reduction (outer loop `sIter = 0..63`):**

Each of the 4 waves handles 64 rows within its block. In iteration `sIter`,
the global row for wave `waveId` is:
```
vRowIter = sRowBase + waveId * 64 + sIter
```

All 64 lanes of the wave collectively load one column each from `partialBuf`
(lane `vLane` starts at column `vLane` and strides by 64). EXEC is narrowed
progressively as lane columns advance past `nD`.

After the inner loop, a 6-stage `ds_bpermute` XOR butterfly (strides 1, 2, 4,
8, 16, 32) reduces all 64 partial sums into every lane.

Then:
```
rstd = rsqrt(invD * total_sum + eps)
```

`rstd` is written to `LDS[waveId * 256 + sIter * 4]` for threads where
`row < M`.

**Phase 2 — Scale (all 256 threads):**

After the barrier, each thread reads `rstd = LDS[Serial * 4]` (its fixed row's
rstd). The column loop over `sColStart..sColEnd-1` loads each bf16 from C,
multiplies by `rstd` (bf16 bit-shift trick: shift left 16, multiply as f32,
shift right 16), and stores back.

### 10.7 Public API

```python
from PartialRmsEpilogueGenerator import build_partial_rms_epilogue, KERNEL_NAME

asmStr, kernelName = build_partial_rms_epilogue(chip, M=0, N=0, K=0)
```

Returns `(asmStr, kernelName)` where `asmStr` contains the full `.amdgcn`
assembly including `.rodata` and `.amdgpu_metadata` sections.

---

## 11. Tools and Helpers

### 11.1 `tools/gemm_partialrms_colv2_helpers.py`

Provides utilities for building and executing the K1 + K2 pipeline:

- `setup_tensile(chip)` — initializes rocisa TI singleton and returns
  `(assembler, isaInfoMap, debugConfig)`.
- `build_k1_solution(chip, assembler, isaInfoMap, miOverride=None, residualAdd=False)` —
  builds the K1 solution from `gemm_partial_rms_k1.yaml`. `miOverride` replaces
  the `MatrixInstruction` parameter (used by tests to exercise different tile configs).
- `generate_asm(solution, assembler, debugConfig)` — compiles the solution to
  assembly string and returns `(asmStr, kernelName)`.
- `compute_sk3_dp_args(...)` — computes Stream-K DP (data-parallel only) kernel
  arguments including the magic-number division constants required by the
  StreamK dispatch logic.
- `_magic_number_alg2(d)` — 32-bit unsigned magic number computation (algorithm 2),
  mirrors `ContractionSolution.cpp`.

### 11.2 `tools/gen_partialrms_logic.py`

Generates a LibraryLogic YAML for `TensileCreateLibrary`. See Section 8.2.

### 11.3 `tools/bench_gemm_rms.py`

Full end-to-end benchmark: K1 + K2 pipeline on a real GPU.

```bash
python tools/bench_gemm_rms.py                           # default 4096×4096×4096
python tools/bench_gemm_rms.py --M 8192 --N-hidden 8192 --K 4096
python tools/bench_gemm_rms.py --warmup 5 --iters 20 --no-verify
python tools/bench_gemm_rms.py --wg-n 1                 # MT1=64 (wg_n=1)
python tools/bench_gemm_rms.py --wg-n 2                 # MT1=128 (default)
```

Reports K1 TFLOPS, K2 bandwidth (GB/s), and combined pipeline TFLOPS.
Verifies D against numpy `RMSNorm(A^T @ W * gamma)` with tolerance `2e-2`.

### 11.4 `tools/test_client_partialrms.sh`

Shell integration test:
1. Generates logic YAML via `gen_partialrms_logic.py`.
2. Compiles a device library in YAML format.
3. Runs `tensilelite-client --use-partial-rms` and checks for `PASSED`.
4. Runs with `--partial-rms-residual-add` and checks for `PASSED`.

Usage:
```bash
tools/test_client_partialrms.sh --chip gfx950 --client /path/to/tensilelite-client
```

### 11.5 `tools/gemm_partial_rms_k1.yaml`

Single pinned solution config. See Section 8.1.

### 11.6 `tools/GEMM_PARTIALRMS_LIBRARY.md`

Documents the two-step library build process (generate YAML → compile).

### 11.7 `tools/TUNING_PIPELINE.md`

Documents the full three-phase TensileLite tuning pipeline and provides a
recommended parameter sweep for the PartialRMS K1 kernel (~576 candidates
before filtering, ~300–400 surviving).

---

## 12. Testing

### 12.1 `Tensile/Tests/unit/test_gemm_partial_rms.py`

Exercises the K1 epilogue in isolation (no K2). Generates approximately 490
test cases.

**Fixture parametrization:**

```python
_WG_CONFIGS = [(1,1), (1,2), (2,1), (4,1), (4,2)]
_RESIDUAL_ADD = [False, True]
```

Ten combinations of `(wg0_waves, wg1_waves, residualAdd)`. Each combination
is compiled once per session via the `k1_kernel` session-scoped fixture.

For each tile config, `_m_shapes_for_mt0(mt0)` generates approximately 25
M values covering: tile multiples (1×–32× MT0), boundary fractions (MT0/4,
MT0/2, MT0-1, MT0+1, 2*MT0-1), small primes (1, 3, 7, 31), and large shapes
(4096, 8192).

**N shapes:** `_N_HIDDEN = [64, 128, 192, 256, 320, 4096, 8192]`

**K values:** `_K_VALUES = [1, 32, 64, 96, 128, 256, 512, 1024, 4096, 31, 97, 127, 4093, 8192]`

**Validation:**
- D (bf16): relative tolerance `2e-2` versus `float32(A^T @ W * gamma)`.
- partialBuf (fp32): relative tolerance `1e-4` versus `Σ_{n in tile} float32(A^T @ W)[m,n]²`.

Residual tests skip shapes where total GPU memory exceeds 128 MB (A + W + D +
residual, all bf16).

### 12.2 `Tensile/Tests/unit/test_gemm_partial_rms_epilogue.py`

Tests the full two-kernel pipeline: K1 → K2 (`partial_rms_epilogue`).

**Fixture parametrization:**
```python
_WG_N_VALUES = [1, 2]
_K_VALUES = [64, 256, 4096]
```

(wg_n=4 is excluded due to LDS limits.)

N shapes are generated by `_n_shapes_for_mt1(mt1)`: tile-aligned multiples
(1×, 3×, 7×) plus partial-tile cases (1, mt1//2-1, mt1-1 elements in the last
tile).

M shapes from `_m_shapes_for_mt0(mt0)` plus irregular primes (127, 193, 317,
511, 769, 1009, 1537, 4097).

**Validation:** D (bf16) against `RMSNorm(A^T @ W * gamma)` with eps=1e-5,
tolerance `2e-2`.

### 12.3 Running the Tests

```bash
# Activate the venv first (requires a gfx950 GPU and amdgpu_exec + ml_dtypes).
source ~/.tensile/bin/activate

# K1-only tests (epilogue isolation).
pytest Tensile/Tests/unit/test_gemm_partial_rms.py -v

# Full pipeline tests (K1 + K2).
pytest Tensile/Tests/unit/test_gemm_partial_rms_epilogue.py -v

# Unit tests only (no GPU required — skips if GPU not present).
tox -e unit
```

---

## 13. How to Run Everything

### 13.1 Build the K1 Solution

```bash
source ~/.tensile/bin/activate
cd /path/to/tensilelite

# Generate the LibraryLogic YAML.
python tools/gen_partialrms_logic.py --chip gfx950 --out-dir /tmp/prms_logic

# Compile the device library.
python -m Tensile.TensileCreateLibrary \
    --architecture=gfx950 \
    --cxx-compiler=$(which amdclang++) \
    --jobs=$(nproc) \
    /tmp/prms_logic/ \
    /tmp/prms_output/ \
    HIP
```

### 13.2 Run the Client Integration Test

```bash
# Requires a gfx950 GPU and a compiled tensilelite-client.
tools/test_client_partialrms.sh \
    --chip gfx950 \
    --client /path/to/build_tmp/client/tensilelite-client
```

### 13.3 Run the Benchmark

```bash
# Requires amdgpu_exec and a gfx950 GPU.
python tools/bench_gemm_rms.py --M 4096 --N-hidden 4096 --K 4096
```

### 13.4 Run Unit Tests

```bash
pytest Tensile/Tests/unit/test_gemm_partial_rms.py -v
pytest Tensile/Tests/unit/test_gemm_partial_rms_epilogue.py -v
```

---

## 14. Full Flow Narrative

**Problem statement:** LLM inference needs `D = RMSNorm(A^T @ W * gamma)` efficiently
on gfx950. The bottleneck is computing the per-row mean-square norm, which requires
a full reduction across the N_hidden dimension. A single-kernel approach would
require synchronization across all workgroups, which scales poorly.

**Two-kernel decomposition:**
1. K1 (this feature): Each workgroup computes `A^T @ W` for its M×MT1 tile and
   writes the partial Σx² for those columns to `partialBuf[m, WorkGroup1]` before
   the store path applies gamma. The gamma multiplication produces a scaled D
   (`h1 * gamma`) rather than the final RMS-normalized D.

2. K2 (`partial_rms_epilogue`): Reads all `N_tiles_N` partial sums per row, sums
   them, divides by N_hidden, adds eps, takes rsqrt, and multiplies D in-place.
   Each K2 workgroup handles a 256×256 tile of D.

**Execution path through the codebase:**

1. Host calls `hipblasLtMatmul` → `tensile_host.cpp` → C++ runtime selects a
   solution with `sizeMapping.partialRMS = true`.

2. `ContractionSolution.cpp:appendKernelArgs` appends `RMSNormGamma`,
   `PartialBuf`, and optionally `ResidualBuf` to the HIP kernel arguments.

3. The kernel (`Cijk_Alik_Bljk_BBS_BH_UserArgs_PartialRMS` or similar) runs the
   Subtile bf16 GEMM on gfx950. After MFMA completion, `SubtilePartialRMSEmitter`
   inserts code that:
   - Optionally adds the residual tensor to AGPRs.
   - Computes Σx² across all N-tiles owned by this wave (squaring AGPRs).
   - Reduces within each wave via DPP row_shr.
   - Optionally reduces across sibling waves via LDS.
   - Writes one fp32 per row to `partialBuf[m, WorkGroup1]`.
   - Multiplies each AGPR by the corresponding gamma (bf16 → fp32).

4. The normal store path executes: `D = bf16(alpha * acc + beta * C)` where
   `acc` now contains `h1 * gamma`.

5. Host launches K2 (`partial_rms_epilogue`) with a 2D grid:
   - Grid X: `ceil(M / 256)` (row tiles).
   - Grid Y: `ceil(N_hidden / 256)` (column tiles).
   - Each block reads the `N_tiles_N` partial sums for its 256 rows, reduces them,
     computes `rstd`, and scales its 256-column slice of D in-place.

**Output:** D contains `RMSNorm(h1) * gamma = RMSNorm(A^T @ W) * gamma`, ready
for the next layer.

---

## 15. Key File Locations

| File | Role |
|---|---|
| `/home/fmoracor/tensilelite/Tensile/Components/Subtile/SubtilePartialRMSEmit.py` | K1 epilogue emitter |
| `/home/fmoracor/tensilelite/Tensile/KernelWriter.py` (line 5143) | Emitter invocation hook |
| `/home/fmoracor/tensilelite/Tensile/Components/Signature.py` (line 316) | Kernel signature extension |
| `/home/fmoracor/tensilelite/Tensile/SolutionStructs/Solution.py` (line 215) | Parameter validation |
| `/home/fmoracor/tensilelite/Tensile/Common/ValidParameters.py` (line 450) | Parameter registry |
| `/home/fmoracor/tensilelite/Tensile/Common/RequiredParameters.py` (line 167) | Required parameter list |
| `/home/fmoracor/tensilelite/Tensile/Contractions.py` (line 76) | ProblemType keys |
| `/home/fmoracor/tensilelite/include/Tensile/ContractionProblem.hpp` (line 351) | Tensor enum + setters |
| `/home/fmoracor/tensilelite/include/Tensile/ContractionSolution.hpp` (line 184) | SizeMapping flags |
| `/home/fmoracor/tensilelite/src/ContractionSolution.cpp` (line 1115) | Argument appending |
| `/home/fmoracor/tensilelite/PartialRmsEpilogueGenerator.py` | K2 kernel generator |
| `/home/fmoracor/tensilelite/client/main.cpp` (line 407) | CLI flags |
| `/home/fmoracor/tensilelite/client/src/ClientProblemFactory.cpp` (line 190) | Problem setup |
| `/home/fmoracor/tensilelite/client/src/Reference.cpp` (line 2037) | Reference computation |
| `/home/fmoracor/tensilelite/client/src/ReferenceValidator.cpp` (line 513) | partialBuf validation |
| `/home/fmoracor/tensilelite/Tensile/ClientWriter.py` (line 580) | Client INI generation |
| `/home/fmoracor/tensilelite/tools/gemm_partialrms_colv2_helpers.py` | Pipeline helpers |
| `/home/fmoracor/tensilelite/tools/gen_partialrms_logic.py` | LibraryLogic YAML generator |
| `/home/fmoracor/tensilelite/tools/bench_gemm_rms.py` | End-to-end benchmark |
| `/home/fmoracor/tensilelite/tools/test_client_partialrms.sh` | Integration test script |
| `/home/fmoracor/tensilelite/tools/gemm_partial_rms_k1.yaml` | Single pinned K1 config |
| `/home/fmoracor/tensilelite/Tensile/Tests/unit/test_gemm_partial_rms.py` | K1 unit tests |
| `/home/fmoracor/tensilelite/Tensile/Tests/unit/test_gemm_partial_rms_epilogue.py` | Pipeline unit tests |
