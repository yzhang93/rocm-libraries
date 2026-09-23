# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Tuple

from geko.config_generator.mi_designer import MFMA, MIDesign
from geko.config_generator.fork_params.optimization_param import BaseParamBuilder
from geko.config_generator.shared_utils import (
    ForkParameter,
    GroupDimension,
    SizeContext,
)


def mark_post_process(fn):
    """Mark a method as a post-processing step."""
    fn._is_post_process = True
    return fn


class BasePostProcessor(BaseParamBuilder):
    """Base class for post-processing MI groups and fork params.

    Runs after MIDesigner + OptimizationParams have produced their
    outputs.  Heuristic-only (generic params don't depend on MI properties).

    Decorate methods with @mark_post_process.  Each receives
    (fork_params, mi_groups) and returns the modified pair.
    Discovery uses vars(type(self)) — same pattern as OptimizationParams.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._post_process_methods: List[str] = []
        for name, method in vars(type(self)).items():
            if getattr(method, "_is_post_process", False):
                self._post_process_methods.append(name)

    def apply(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        size: Tuple[int, int, int, int],
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Run MT_DU adjustments (if active), then all @mark_post_process methods.

        *size* is ``(M, N, B, K)``.
        """
        M, N, B, K = size
        ctx = SizeContext(M=M, N=N, B=B, K=K)
        mt_du = self.config.get("MT_DU")
        if mt_du is not None:
            fork_params, mi_groups = self._apply_mt_du(fork_params, mi_groups, mt_du)
        for method_name in self._post_process_methods:
            fork_params, mi_groups = getattr(self, method_name)(fork_params, mi_groups, ctx)
        # Runs last so it overrides whatever the HW post-processors decided for
        # the parameters the fused epilogue pins.
        if self.config.get("RMS_EPILOGUE", False):
            fork_params, mi_groups = self._apply_rms_epilogue(fork_params, mi_groups)
        elif self.config.get("SUBTILE", False):
            fork_params, mi_groups = self._apply_subtile(fork_params, mi_groups)
        elif self.config.get("MX", False):
            fork_params, mi_groups = self._apply_mx_gemm(fork_params, mi_groups)
        return fork_params, mi_groups

    # -----------------------------------------------------------------
    # Macrotile / Origami tuning (MT_DU) — HW-agnostic
    # -----------------------------------------------------------------

    def _apply_mt_du(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        mt_du: List,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Macrotile / Origami tuning overrides. HW-agnostic.

        Overrides select params with fixed values and filters MI groups
        to only keep entries matching the specified macro tile (MT0, MT1).
        """
        fixed_MT0, fixed_MT1, fixed_DU = mt_du[0], mt_du[1], mt_du[2]

        overrides = {
            "DepthU": [fixed_DU],
            "WorkGroupMapping": [0],
            "WorkGroupMappingXCC": [-1],
            "NonTemporalA": [0],
            "NonTemporalB": [0],
            "NonTemporalC": [0],
            "NonTemporalD": [0],
            "StreamKXCCMapping": [0],
        }
        for name, values in overrides.items():
            if name in fork_params:
                fork_params[name].values = values
            else:
                fork_params[name] = self._make_param(name, values)

        mi_groups = [
            entry for entry in mi_groups
            if _mi_matches_mt(entry, fixed_MT0, fixed_MT1)
        ]

        return fork_params, mi_groups

    # -----------------------------------------------------------------
    # Fused RMSNorm epilogue (MegaFusedEmit) — HW-agnostic
    # -----------------------------------------------------------------

    def _apply_rms_epilogue(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Restrict the search space to RMSEpilogue-legal kernels.

        The fused RMSNorm epilogue rides on the Subtile code path, so this is
        the Subtile search space plus the epilogue's own parameters.
        """
        return self._apply_subtile(
            fork_params,
            mi_groups,
            {
                "RMSEpilogue": [True],
                "RMSEpilogueGammaType": [self.config.get("RMS_EPILOGUE_GAMMA_TYPE", "b")],
                "RMSEpilogueResidualType": [
                    self.config.get("RMS_EPILOGUE_RESIDUAL_TYPE", "b")
                ],
            },
        )

    def _apply_subtile(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        extra_overrides: Dict[str, List] | None = None,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Restrict the search space to Subtile-legal kernels.

        The Subtile emitters read and write AGPRs directly (hence
        MIArchVgpr=False) and Tensile rejects them together with
        PrefetchAcrossPersistent and the custom main-loop schedule. Kernels
        violating any of these are rejected during codegen, so pinning them
        here keeps the search space useful instead of mostly-invalid.

        StreamK is pinned to data-parallel-only because anything else makes
        Tensile emit a second, GSU-reducing store pass alongside the main one.
        That pass re-reads the accumulators the Subtile epilogue already
        consumed, and codegen dies on the exhausted list rather than rejecting
        the kernel.

        The store width is left for Tensile to derive for the same reason: a
        forced width makes the edge store batch more elements than the tile has
        accumulators, which again exhausts that list mid-kernel.
        """
        overrides: Dict[str, List] = {
            "UseSubtileImpl": [True],
            "MIArchVgpr": [False],
            "PrefetchAcrossPersistent": [0],
            "UseCustomMainLoopSchedule": [0],
            "StreamK": [3],
            "StreamKForceDPOnly": [1],
            "StreamKAtomic": [0],
            "StoreVectorWidth": [-1],
            "NumElementsPerBatchStore": [0],
        }
        overrides.update(extra_overrides or {})

        # Subtile supports PrefetchGlobalRead 0/1/2 only.
        pgr = fork_params.get("PrefetchGlobalRead")
        if pgr is not None:
            overrides["PrefetchGlobalRead"] = [v for v in pgr.values if v in (0, 1, 2)] or [2]

        # Subtile needs DepthU to be a multiple of 2 * MatrixInstK * LocalSplitU.
        # That is 64 for 16x16x32 bf16/f16, and 256 for 16x16x128 MXFP8. The
        # actual K is taken from the surviving MI groups below.
        for name, values in overrides.items():
            if name in fork_params:
                fork_params[name].values = values
            else:
                fork_params[name] = self._make_param(name, values)

        # Drop the CMS tiles merged in by the HW post-processor: their
        # handwritten schedules are incompatible with the Subtile path.
        mi_groups = [entry for entry in mi_groups if not _mi_is_cms(entry)]

        # MI entries carry per-tile copies of some of the pinned parameters
        # (e.g. MIArchVgpr); remove them so the forced values win.
        for entry in mi_groups:
            for name in overrides:
                entry.pop(name, None)

        mi_groups = [entry for entry in mi_groups if _mi_supports_subtile(entry)]

        inst_ks = [
            MFMA.from_list(entry["MatrixInstruction"].values).K
            for entry in mi_groups
        ]
        if inst_ks:
            du_mult = 2 * max(inst_ks)
            du = fork_params.get("DepthU")
            if du is not None:
                du.values = [v for v in du.values if v % du_mult == 0] or [du_mult]

        return fork_params, mi_groups

    # -----------------------------------------------------------------
    # MX GEMMs without the fused RMS epilogue (e.g. ScaleAlphaVec consumer)
    # -----------------------------------------------------------------

    def _apply_mx_gemm(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Keep MXFP8 samples in the legal DepthU / PGR region.

        Random GA sampling otherwise spends its iteration budget on DepthU
        values smaller than 2*MatrixInstK (32/64/128 for 16x16x128) and on
        CMS tiles whose handwritten schedules do not cover MX block scaling,
        which yields a 0-sized initial population.
        """
        overrides: Dict[str, List] = {
            "UseSubtileImpl": [True],
            "MIArchVgpr": [False],
            "PrefetchAcrossPersistent": [0],
            "UseCustomMainLoopSchedule": [0],
            "StreamK": [0],
            "StoreVectorWidth": [-1],
            "NumElementsPerBatchStore": [0],
        }
        pgr = fork_params.get("PrefetchGlobalRead")
        if pgr is not None:
            overrides["PrefetchGlobalRead"] = [v for v in pgr.values if v in (0, 1, 2)] or [2]
        for name, values in overrides.items():
            if name in fork_params:
                fork_params[name].values = values
            else:
                fork_params[name] = self._make_param(name, values)

        mi_groups = [entry for entry in mi_groups if not _mi_is_cms(entry)]
        for entry in mi_groups:
            for name in overrides:
                entry.pop(name, None)
        mi_groups = [entry for entry in mi_groups if _mi_supports_subtile(entry)]

        inst_ks = [
            MFMA.from_list(entry["MatrixInstruction"].values).K
            for entry in mi_groups
        ]
        if inst_ks:
            du_mult = 2 * max(inst_ks)
            du = fork_params.get("DepthU")
            if du is not None:
                du.values = [v for v in du.values if v % du_mult == 0] or [du_mult]
        return fork_params, mi_groups


def _mi_matches_mt(entry: Dict[str, ForkParameter], fixed_MT0: int, fixed_MT1: int) -> bool:
    """Check if an MI group entry's macro tile matches the fixed MT."""
    mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(entry["MatrixInstruction"].values))
    return mfma_params.MT0 == fixed_MT0 and mfma_params.MT1 == fixed_MT1


def _mi_is_cms(entry: Dict[str, ForkParameter]) -> bool:
    """Whether an MI group entry came from the CMS kernel registry."""
    cms = entry.get("UseCustomMainLoopSchedule")
    return cms is not None and 1 in cms.values


# gfx950 splits one 512-entry register file into 256 arch VGPRs and 256 AGPRs,
# so a D tile needing more than 256 accumulators per thread spills into arch
# VGPRs. Tensile's Subtile store path mis-maps those spilled registers and dies
# during codegen (mapAcctoArchRegs leaves holes, which surface as
# "replaceHolder(): NoneType"), taking the whole tuning run with it. Keeping the
# D tile inside the AGPR file avoids the spill path entirely.
_MAX_ACCVGPRS_PER_THREAD = 256
_WAVEFRONT_SIZE = 64


def _mi_supports_subtile(entry: Dict[str, ForkParameter]) -> bool:
    """Whether an MI group entry can run on the Subtile code path.

    Subtile needs a 16x16 matrix instruction, both macro-tile dimensions
    64-aligned, and a D tile that fits in the AGPR file.
    """
    gsu = entry.get("GlobalSplitU")
    if gsu is not None and gsu.values != [1]:
        return False

    mi = MFMA.from_list(entry["MatrixInstruction"].values)
    if mi.M * mi.MIBlockM != 16:
        return False
    if mi.N / mi.MIBlockM * mi.B != 16:
        return False

    mfma_params = MIDesign.calculate_mfma_parameters(mi)
    if mfma_params.MT0 % 64 != 0 or mfma_params.MT1 % 64 != 0:
        return False

    threads = mi.waveM * mi.waveN * _WAVEFRONT_SIZE
    accvgprs = (mfma_params.MT0 * mfma_params.MT1) // threads
    return accvgprs <= _MAX_ACCVGPRS_PER_THREAD
