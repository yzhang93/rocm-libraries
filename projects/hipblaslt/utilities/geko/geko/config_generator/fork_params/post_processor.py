# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import logging
from typing import Any, Dict, List, Tuple

from geko.config_generator.mi_designer import MFMA, MIDesign
from geko.config_generator.fork_params.optimization_param import BaseParamBuilder
from geko.config_generator.shared_utils import (
    ForkParameter,
    GroupDimension,
    SizeContext,
)


logger = logging.getLogger("GEKO")


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
        if self.config.get("PARTIAL_RMS", False):
            fork_params, mi_groups = self._apply_partial_rms(fork_params, mi_groups)
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
    # Fused RMSNorm epilogue (PartialRMS) — HW-agnostic
    # -----------------------------------------------------------------

    def _apply_partial_rms(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Restrict the search space to PartialRMS-legal kernels.

        The fused RMSNorm epilogue only exists on the Subtile code path, whose
        emitters read and write AGPRs directly (hence MIArchVgpr=False) and
        which Tensile rejects together with PrefetchAcrossPersistent and the
        custom main-loop schedule. Kernels violating any of these are rejected
        during codegen, so pinning them here keeps the search space useful
        instead of mostly-invalid.

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
            "PartialRMS": [True],
            "UseSubtileImpl": [True],
            "MIArchVgpr": [False],
            "PrefetchAcrossPersistent": [0],
            "UseCustomMainLoopSchedule": [0],
            "StreamK": [3],
            "StreamKForceDPOnly": [1],
            "StreamKAtomic": [0],
            "StoreVectorWidth": [-1],
            "NumElementsPerBatchStore": [0],
            "PartialRMSResidualAdd": [bool(self.config.get("PARTIAL_RMS_RESIDUAL_ADD", False))],
            "PartialRMSQuant": [bool(self.config.get("PARTIAL_RMS_QUANT", False))],
        }

        # Subtile supports PrefetchGlobalRead 0/1/2 only.
        pgr = fork_params.get("PrefetchGlobalRead")
        if pgr is not None:
            overrides["PrefetchGlobalRead"] = [v for v in pgr.values if v in (0, 1, 2)] or [2]

        # Subtile needs DepthU to be a multiple of 2 * MatrixInstK * LocalSplitU,
        # which is 64 for the 16x16x32 bf16/f16 instruction required below.
        du = fork_params.get("DepthU")
        if du is not None:
            overrides["DepthU"] = [v for v in du.values if v % 64 == 0] or [64]

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

        mi_groups = [entry for entry in mi_groups if _mi_supports_partial_rms(entry)]
        mi_groups = self._select_partial_rms_mt1(mi_groups)

        return fork_params, mi_groups

    def _select_partial_rms_mt1(self, mi_groups: GroupDimension) -> GroupDimension:
        """Keep a single MacroTile1 among the PartialRMS candidate tiles.

        The client sizes the RMS partial buffer from the tile shape, so
        ClientWriter refuses a benchmark pass whose solutions disagree on MT1.
        Tune one MT1 per run; PARTIAL_RMS_MT1 picks which, otherwise the MT1
        with the most candidates wins.
        """
        by_mt1: Dict[int, GroupDimension] = {}
        for entry in mi_groups:
            mfma_params = MIDesign.calculate_mfma_parameters(
                MFMA.from_list(entry["MatrixInstruction"].values))
            by_mt1.setdefault(mfma_params.MT1, []).append(entry)

        if not by_mt1:
            return mi_groups

        requested = self.config.get("PARTIAL_RMS_MT1")
        if requested is not None:
            if int(requested) not in by_mt1:
                raise ValueError(
                    f"PARTIAL_RMS_MT1={requested} has no PartialRMS-legal tiles for this "
                    f"problem; available MacroTile1 values: {sorted(by_mt1)}"
                )
            chosen = int(requested)
        else:
            chosen = max(by_mt1, key=lambda mt1: (len(by_mt1[mt1]), mt1))
            logger.warning(
                "PartialRMS: benchmarking one MacroTile1 per run; selected MT1=%d "
                "(%d tiles) out of %s. Set PARTIAL_RMS_MT1 to tune a different one.",
                chosen, len(by_mt1[chosen]), sorted(by_mt1),
            )

        return by_mt1[chosen]


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


def _mi_supports_partial_rms(entry: Dict[str, ForkParameter]) -> bool:
    """Whether an MI group entry can host the PartialRMS epilogue.

    PartialRMS needs a 16x16 matrix instruction (a Subtile requirement), both
    macro-tile dimensions 64-aligned (each workgroup reduces its own
    MacroTile0-wide slice of the reduced axis), and a D tile that fits in the
    AGPR file.
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
