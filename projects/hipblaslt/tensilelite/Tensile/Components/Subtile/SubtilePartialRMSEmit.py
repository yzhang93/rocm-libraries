# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PartialRMS fused epilogue emitter for the Subtile kernel (gfx950, bf16).

Reduces the fp32 GEMM accumulator (AGPRs) over free0 (the N_hidden dimension),
producing one fp32 Σx² per output token (free1 row) per free0 tile (WorkGroup0).
Also applies the gamma weight (bf16) to the accumulator in-place.

This is Phase 1 (K1) of a two-kernel RMSNorm pipeline operating on row-major output:
  - K1 (this kernel): free0=N_hidden, free1=M_tokens. Each WG writes
    partialBuf[token, WorkGroup0] = Σ_{i in tile} h1[token, i]².
  - K2 (row_div): reads all partialBuf tiles per token, reduces, computes
    rstd = rsqrt(Σx²/N_hidden + eps), and divides D in-place.

partialBuf layout contract (2D, row-major):
  - Logical shape [M_padded, n_d], n_d = ceil(SizesFree0 / MT0).
  - partialBuf[token, t] = Σ_{i in WG t's free0 columns} h1[token, i]².
  - Byte offset for (token, t) = (token * n_d + t) * 4.
  - Token index = WorkGroup1*MT1 + intra-tile token offset.
  - Tile column t = WorkGroup0 (the WG's index along free0).
  - n_d is computed on device from SizesFree0; it is not a kernarg.

Reduction stages (free0 axis):
  1. Sum acc²  over all free0 MMA-tiles (mma_m) and k-offsets (rows_per_lane)
     within each wave, yielding one partial[n] per free1 lane-column (mfma_n lanes).
  2. XOR butterfly via ds_bpermute over waveSize/mfma_n row groups so every
     lane holds the full free0 sum for its n column.
  3. (When wg_m > 1) LDS cross-wave reduction over wg_m sibling waves.
  4. Lanes with rowGroup==0 and waveM==0 write partialBuf[token, WG0].

Gamma application:
  Loads gamma (bf16) for each free0 position via BufferLoadD16B16, converts to
  fp32, and multiplies each accumulator element in-place before the store path
  writes D as bf16.

MFMA layout (gfx950, waveSize=64, 16x16 MFMA):
  - lane % mfma_n = free1 column within MMA tile (token lane)
  - rows_per_lane = (mfma_m * mfma_n) // waveSize

Acc VGPR ordering (N-outer, M-inner):
  acc_idx(base, m, n, k) = base + (n*mma_m + m)*rows_per_lane + k

Alpha=1, beta=0 must be passed by the host.
"""

import math

from rocisa.code import Module
from rocisa.container import (
    ContinuousRegister,
    DPPModifiers,
    DSModifiers,
    EXEC,
    MUBUFModifiers,
    accvgpr,
    sgpr,
    vgpr,
)
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    BufferLoadD16B16,
    BufferStoreB32,
    DSBPermuteB32,
    DSLoadB32,
    DSStoreB32,
    SAndSaveExecB64,
    SMovB32,
    SMovB64,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddF32,
    VAddU32,
    VAndB32,
    VCmpEQU32,
    VCmpLtU32,
    VCndMaskB32,
    VCvtBF16toFP32,
    VFmaF32,
    VLShiftLeftB32,
    SAddU32,
    SLShiftLeftB32,
    SLShiftRightB32,
    VMulF32,
    VMulLOU32,
    VMovB32,
    VLShiftRightB32,
    VOrB32,
    VXorB32,
    SMulI32,
)


class SubtilePartialRMSEmitter:
    """Emit the PartialRMS epilogue for the Subtile gfx950 bf16 kernel.

    Computes per-row Σx² from fp32 AGPRs, writes to partialBuf (fp32),
    and applies gamma (bf16) in-place to the accumulator.
    """

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        self.archCaps = writer.states.archCaps

        # Derive all geometry from kernel params; no module-level constants.
        self.mfma_m = kernel["MatrixInstM"]
        self.mfma_n = kernel["MatrixInstN"]
        self.waveSize = kernel["WavefrontSize"]
        self.rows_per_lane = (self.mfma_m * self.mfma_n) // self.waveSize

        wg = kernel["MIWaveGroup"]
        self.wg_m = wg[0]
        self.wg_n = wg[1]

        self.mma_m = (kernel["MacroTile0"] // self.mfma_m) // self.wg_m
        self.mma_n = (kernel["MacroTile1"] // self.mfma_n) // self.wg_n
        self.macro_tile0 = kernel["MacroTile0"]
        self.macro_tile1 = kernel["MacroTile1"]
        self.numRows = self.mma_m * self.rows_per_lane
        self.numPartials = self.mma_n

        # laneSGPRCount: 1 for wave32, 2 for wave64.
        self.lane_sgpr_count = writer.states.laneSGPRCount
        self.residualAdd = bool(kernel.get("PartialRMSResidualAdd", False))

    def _acc_idx(self, base: int, m: int, n: int, k: int) -> int:
        """AGPR index for accumulator element at M-tile m, N-tile n, row-offset k."""
        return base + (n * self.mma_m + m) * self.rows_per_lane + k

    def emit(self, accVgprBase: int) -> Module:
        """Return the full PartialRMS epilogue module.

        accVgprBase: AGPR index of the first D-tile accumulator.
        """
        numAccVgpr = self.mma_m * self.mma_n * self.rows_per_lane
        module = Module("PartialRMS epilogue")
        module.addComment1("PartialRMS: fused partial sum-of-squares + gamma epilogue")
        module.addComment0(
            f"  Acc AGPRs [{accVgprBase}, {accVgprBase + numAccVgpr}), "
            f"mma_m={self.mma_m}, mma_n={self.mma_n}, "
            f"MT0={self.macro_tile0}, MT1={self.macro_tile1}"
        )
        module.addComment0(
            "  partialBuf: raw Σx² per row (K2 divides by N_hidden, not this kernel)"
        )

        # Allocate all VGPRs for temporaries.
        partials = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_partials")
        accTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_accTmp")
        gammaTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_gammaTmp")
        laneId = self.writer.vgprPool.checkOut(1, tag="pRMS_laneId")
        # colByte is computed and consumed outside the EXEC-narrowed window in _writePartials.
        colByte = self.writer.vgprPool.checkOut(1, tag="pRMS_colByte")
        globalAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_globalAddr")
        if self.residualAdd:
            resTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_resTmp")
            resAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_resAddr")

        # Allocate SGPRs: gamma SRD, partialBuf SRD, saved exec.
        # savedExec and laneMaskSgpr must be 2-aligned for 64-bit EXEC operations.
        # rowBase = WorkGroup0 * MT0 is computed into globalAddr on demand (no SGPR).
        # tileCol = sgpr("WorkGroup1"), live named SGPR, no allocation needed.
        # preventOverflow=False: the epilogue SGPRs are temporary (live only during
        # the epilogue) and hardware SGPR budget has been verified to accommodate them.
        gammaSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_gammaSrd",
                                                        preventOverflow=False)
        partialSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_partialSrd",
                                                          preventOverflow=False)
        savedExec = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_savedExec",
            preventOverflow=False,
        )
        laneMaskSgpr = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_laneMask",
            preventOverflow=False,
        )
        if self.residualAdd:
            resSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_resSrd",
                                                          preventOverflow=False)

        # Flush MFMA pipeline before reading AGPRs.
        module.add(
            SWaitCnt(waitAll=True, comment="flush MFMA pipeline before PartialRMS")
        )

        module.add(self._setup(gammaSrd, partialSrd, laneId, colByte))
        if self.residualAdd:
            module.add(
                self._addResidualFree0(accVgprBase, accTmp, resTmp, resAddr, laneId, colByte, resSrd)
            )
        module.add(self._squareAndLaneSumFree0(accVgprBase, partials, accTmp))
        module.add(self._rowGroupReduceFree0(partials))
        if self.wg_m > 1:
            module.add(self._crossWaveReduceFree0(partials))
        module.add(
            self._writePartialsFree0(
                partials, partialSrd, laneId, savedExec, laneMaskSgpr,
                globalAddr, colByte
            )
        )
        module.add(
            self._applyGammaFree0(accVgprBase, gammaSrd, gammaTmp, accTmp, globalAddr)
        )

        if self.residualAdd:
            self.writer.sgprPool.checkIn(resSrd)
        self.writer.sgprPool.checkIn(laneMaskSgpr)
        self.writer.sgprPool.checkIn(savedExec)
        self.writer.sgprPool.checkIn(partialSrd)
        self.writer.sgprPool.checkIn(gammaSrd)
        self.writer.vgprPool.checkIn(globalAddr)
        self.writer.vgprPool.checkIn(colByte)
        self.writer.vgprPool.checkIn(laneId)
        self.writer.vgprPool.checkIn(gammaTmp)
        self.writer.vgprPool.checkIn(accTmp)
        self.writer.vgprPool.checkIn(partials)
        if self.residualAdd:
            self.writer.vgprPool.checkIn(resAddr)
            self.writer.vgprPool.checkIn(resTmp)

        return module

    def _setup(
        self,
        gammaSrd: int,
        partialSrd: int,
        laneId: int,
        colByte: int,
    ) -> Module:
        """Build gamma SRD, partialBuf SRD; derive laneId and colByte.

        Signature append order (matches Signature.py additions):
          slot N+0: RMSNormGamma  (bf16 global buffer pointer, 8 bytes)
          slot N+1: PartialBuf    (fp32 global buffer pointer, 8 bytes) [InOutArray]
          slot N+2: ResidualBuf   (bf16 global buffer pointer, 8 bytes) — only when PartialRMSResidualAdd

        rowBase = WorkGroup0 * MT0 is computed on demand in _writePartials (no SGPR).
        tileCol = sgpr("WorkGroup1"), live named SGPR, no extra allocation needed.
        """
        module = Module("PartialRMS setup")
        module.add(SWaitCnt(kmcnt=0, comment="wait for PartialRMS kernarg s_load"))

        # Gamma SRD (bf16 global buffer).
        module.add(
            SMovB64(
                dst=sgpr(gammaSrd, 2),
                src=sgpr("RMSNormGamma", 2),
                comment="gamma SRD base",
            )
        )
        module.add(
            SMovB32(dst=sgpr(gammaSrd + 2), src="BufferOOB", comment="gamma SRD limit")
        )
        module.add(
            SMovB32(dst=sgpr(gammaSrd + 3), src="Srd127_96", comment="gamma SRD flags")
        )

        # PartialBuf SRD (fp32 global buffer).
        module.add(
            SMovB64(
                dst=sgpr(partialSrd, 2),
                src=sgpr("PartialBuf", 2),
                comment="partialBuf SRD base",
            )
        )
        module.add(
            SMovB32(
                dst=sgpr(partialSrd + 2),
                src="BufferOOB",
                comment="partialBuf SRD limit",
            )
        )
        module.add(
            SMovB32(
                dst=sgpr(partialSrd + 3),
                src="Srd127_96",
                comment="partialBuf SRD flags",
            )
        )

        # laneId = Serial & (waveSize - 1).
        module.add(
            VAndB32(
                dst=vgpr(laneId),
                src0=vgpr("Serial"),
                src1=self.waveSize - 1,
                comment="laneId = Serial & (waveSize-1)",
            )
        )

        # colByte = (laneId % mfma_n) * 2  (byte offset into bf16 gamma per-lane).
        module.add(
            VAndB32(
                dst=vgpr(colByte),
                src0=vgpr(laneId),
                src1=self.mfma_n - 1,
                comment=f"colInMma = laneId % {self.mfma_n}",
            )
        )
        module.add(
            VLShiftLeftB32(
                dst=vgpr(colByte),
                shiftHex=hex(1),
                src=vgpr(colByte),
                comment="colByte = colInMma * 2 (bf16 size)",
            )
        )

        # When wg_n > 1, shift colByte by the wave's column base.
        if self.wg_n > 1:
            waveN = self.writer.vgprPool.checkOut(1, tag="pRMS_setupWaveN")
            tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_setupTmp")
            tmpRes = ContinuousRegister(tmpVgpr, 2)
            module.add(
                vectorStaticDivide(
                    waveN,
                    "Serial",
                    self.waveSize * self.wg_m,
                    tmpRes,
                    comment=f"waveN = Serial / {self.waveSize * self.wg_m}",
                )
            )
            colBaseBytes = self.mma_n * self.mfma_n * 2
            with self.writer.allocTmpSgpr(1, tag="pRMS_setupColBase") as tmpSgprInfo:
                module.add(
                    SMovB32(
                        dst=sgpr(tmpSgprInfo.idx),
                        src=hex(colBaseBytes),
                        comment=f"col base bytes per wave ({colBaseBytes})",
                    )
                )
                module.add(
                    VMulLOU32(
                        dst=vgpr(waveN),
                        src0=sgpr(tmpSgprInfo.idx),
                        src1=vgpr(waveN),
                        comment="waveN * mma_n * mfma_n * 2",
                    )
                )
            module.add(
                VAddU32(
                    vgpr(colByte),
                    vgpr(colByte),
                    vgpr(waveN),
                    comment="colByte += wave column base",
                )
            )
            self.writer.vgprPool.checkIn(tmpVgpr)
            self.writer.vgprPool.checkIn(waveN)

        # Add WorkGroup1 * MT1 * 2 to colByte so each WG addresses its own gamma tile.
        # MT1 * 2 is a power-of-2 because MT1 is a power-of-2 and bf16 is 2 bytes.
        wg1Shift = int(math.log2(self.macro_tile1 * 2))
        with self.writer.allocTmpSgpr(1, tag="pRMS_setupWG1") as wg1S:
            module.add(
                SLShiftLeftB32(
                    dst=sgpr(wg1S.idx),
                    src=sgpr("WorkGroup1"),
                    shiftHex=hex(wg1Shift),
                    comment=f"wg1ColByte = WorkGroup1 * MT1*2 (MT1={self.macro_tile1})",
                )
            )
            module.add(
                VAddU32(
                    vgpr(colByte),
                    vgpr(colByte),
                    sgpr(wg1S.idx),
                    comment="colByte += WorkGroup1 * MT1 * 2",
                )
            )

        return module

    def _addResidualFree0(
        self,
        accVgprBase: int,
        accTmp: int,
        resTmp: int,
        resAddr: int,
        laneId: int,
        colByte: int,
        resSrd: int,
    ) -> Module:
        """Load bf16 row-major residual R and add to each AGPR element.

        R is row-major [M_tokens, N_hidden]. For acc[m, n, k]:
          nhidden_pos = WG0*MT0 + waveMOff + m*mfma_m + rowGroupOff + k
          token       = (colByte >> 1) + n*mfma_n
          byteOff     = (token * SizesFree0 + nhidden_pos) * 2

        rowGroupOff = (laneId >> log2(mfma_n)) * rows_per_lane encodes the
        per-lane free0 row-group offset within each MMA tile.
        colByte already encodes laneId%mfma_n + waveN offset + WG1*MT1, so
        colByte >> 1 is the token index for n=0.
        """
        module = Module("PartialRMS addResidualFree0")
        module.addComment1("PartialRMS residual add (free0): acc[m,n,k] += R[token, nhidden_pos]")

        log2MfmaN = int(math.log2(self.mfma_n))

        # Residual SRD: bounds = M_tokens * N_hidden * 2 bytes so tail-WG lanes
        # that address token >= M_tokens get a safe OOB (returns 0, no fault).
        with self.writer.allocTmpSgpr(1, tag="pRMS_rF0SrdNumRec") as tmpSgpr:
            module.add(SMovB64(dst=sgpr(resSrd, 2), src=sgpr("ResidualBuf", 2),
                               comment="residual SRD base"))
            module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("SizesFree+0"),
                               src1=sgpr("SizesFree+1"),
                               comment="numRecords = N_hidden * M_tokens"))
            module.add(SLShiftLeftB32(dst=sgpr(resSrd + 2), src=sgpr(tmpSgpr.idx),
                                      shiftHex=hex(1),
                                      comment="numRecords *= 2 (bf16 element size)"))
        module.add(SMovB32(dst=sgpr(resSrd + 3), src="Srd127_96", comment="residual SRD flags"))

        # rowGroupOff = (laneId >> log2(mfma_n)) * rows_per_lane.
        rowGroup = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0RowGroup")
        rowGroupOff = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0RGOff")
        module.add(VLShiftRightB32(dst=vgpr(rowGroup), shiftHex=hex(log2MfmaN), src=vgpr(laneId),
                                   comment=f"rowGroup = laneId >> {log2MfmaN}"))
        module.add(VMulLOU32(dst=vgpr(rowGroupOff), src0=self.rows_per_lane, src1=vgpr(rowGroup),
                             comment=f"rowGroupOff = rowGroup * {self.rows_per_lane}"))
        self.writer.vgprPool.checkIn(rowGroup)

        # nhiddenBase = WorkGroup0 * MT0 (+ waveM * mma_m * mfma_m when wg_m > 1).
        mt0Vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0MT0")
        nhiddenBase = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0NHiddenBase")
        module.add(VMovB32(dst=vgpr(mt0Vgpr), src=self.macro_tile0, comment=f"MT0={self.macro_tile0}"))
        module.add(VMulLOU32(dst=vgpr(nhiddenBase), src0=vgpr(mt0Vgpr), src1=sgpr("WorkGroup0"),
                             comment="nhiddenBase = WorkGroup0 * MT0"))
        self.writer.vgprPool.checkIn(mt0Vgpr)

        if self.wg_m > 1:
            waveIdTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0WaveId")
            waveMTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0WaveM")
            tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_rF0TmpDiv")
            tmpRes = ContinuousRegister(tmpVgpr, 2)
            module.add(
                vectorStaticDivide(waveIdTmp, "Serial", self.waveSize, tmpRes,
                                   comment="waveId = Serial / WavefrontSize")
            )
            module.add(VAndB32(dst=vgpr(waveMTmp), src0=vgpr(waveIdTmp), src1=self.wg_m - 1,
                                comment=f"waveM = waveId %% {self.wg_m}"))
            waveStride = self.mma_m * self.mfma_m
            waveStrideV = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0WaveStride")
            module.add(VMovB32(dst=vgpr(waveStrideV), src=waveStride,
                               comment=f"waveStride = mma_m * mfma_m = {waveStride}"))
            module.add(VMulLOU32(dst=vgpr(waveMTmp), src0=vgpr(waveStrideV), src1=vgpr(waveMTmp),
                                 comment="waveMOff = waveM * waveStride"))
            module.add(VAddU32(vgpr(nhiddenBase), vgpr(nhiddenBase), vgpr(waveMTmp),
                               comment="nhiddenBase += waveMOff"))
            self.writer.vgprPool.checkIn(waveStrideV)
            self.writer.vgprPool.checkIn(tmpVgpr)
            self.writer.vgprPool.checkIn(waveMTmp)
            self.writer.vgprPool.checkIn(waveIdTmp)

        # tokenBase = colByte >> 1 (token index for n=0, already encodes waveN + WG1*MT1).
        tokenBase = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0TokenBase")
        module.add(VLShiftRightB32(dst=vgpr(tokenBase), shiftHex=hex(1), src=vgpr(colByte),
                                   comment="tokenBase = colByte >> 1"))

        module.add(
            self._loopResidualFree0(accVgprBase, accTmp, resTmp, resAddr,
                                    nhiddenBase, tokenBase, rowGroupOff, resSrd)
        )

        self.writer.vgprPool.checkIn(tokenBase)
        self.writer.vgprPool.checkIn(nhiddenBase)
        self.writer.vgprPool.checkIn(rowGroupOff)
        return module

    def _loopResidualFree0(
        self,
        accVgprBase: int,
        accTmp: int,
        resTmp: int,
        resAddr: int,
        nhiddenBase: int,
        tokenBase: int,
        rowGroupOff: int,
        resSrd: int,
    ) -> Module:
        """Emit the n/m/k loop that loads R and adds to each accumulator.

        For each (n, m, k): loads R[token_n, nhidden_pos] as bf16, converts to
        fp32, and adds to acc[m, n, k] in place. Byte offset into R:
          (token_n * SizesFree0 + nhidden_pos) * 2
        where token_n = tokenBase + n*mfma_n and
              nhidden_pos = nhiddenBase + rowGroupOff + m*mfma_m + k.
        """
        module = Module("PartialRMS loopResidualFree0")

        # nOffVgpr doubles as scratch for both n-tile offsets and mBase materialization.
        nOffVgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0NOff")
        rowByteBase = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0RowByteBase")
        # OOB clamp target: an out-of-range byte offset makes the residual load
        # return 0, so padding nhidden positions (>= N_hidden) add nothing.
        oobV = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0Oob")
        module.add(VMovB32(dst=vgpr(oobV), src="BufferOOB",
                           comment="OOB byte offset -> residual load returns 0"))
        oobMask = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_rF0OobMask",
            preventOverflow=False)

        for n in range(self.mma_n):
            nOff = n * self.mfma_n
            if nOff > 64:
                module.add(VMovB32(dst=vgpr(nOffVgpr), src=nOff, comment=f"nOff={nOff} (n={n})"))
                module.add(VAddU32(vgpr(rowByteBase), vgpr(tokenBase), vgpr(nOffVgpr),
                                   comment=f"token_n = tokenBase + {nOff} (n={n})"))
            else:
                module.add(VAddU32(vgpr(rowByteBase), vgpr(tokenBase), nOff,
                                   comment=f"token_n = tokenBase + {nOff} (n={n})"))
            # rowByteBase = token_n * SizesFree0 * 2 (byte offset of the token row in R).
            module.add(VMulLOU32(dst=vgpr(rowByteBase), src0=sgpr("SizesFree+0"),
                                 src1=vgpr(rowByteBase), comment="token_n * SizesFree0"))
            module.add(VLShiftLeftB32(dst=vgpr(rowByteBase), shiftHex=hex(1),
                                      src=vgpr(rowByteBase),
                                      comment="rowByteBase = token_n * SizesFree0 * 2"))

            for m in range(self.mma_m):
                for k in range(self.rows_per_lane):
                    mBase = m * self.mfma_m + k
                    # nhidden_pos = nhiddenBase + rowGroupOff + mBase.
                    if mBase > 64:
                        module.add(VMovB32(dst=vgpr(nOffVgpr), src=mBase,
                                           comment=f"mBase={mBase} (m={m},k={k})"))
                        module.add(VAddU32(vgpr(resAddr), vgpr(nhiddenBase), vgpr(nOffVgpr),
                                           comment=f"nhidden_pos = nhiddenBase + {mBase}"))
                    else:
                        module.add(VAddU32(vgpr(resAddr), vgpr(nhiddenBase), mBase,
                                           comment=f"nhidden_pos = nhiddenBase + {mBase}"))
                    module.add(VAddU32(vgpr(resAddr), vgpr(resAddr), vgpr(rowGroupOff),
                                       comment="nhidden_pos += rowGroupOff"))
                    module.add(VCmpLtU32(dst=sgpr(oobMask, self.lane_sgpr_count),
                                         src0=vgpr(resAddr), src1=sgpr("SizesFree+0"),
                                         comment="inRange = nhidden_pos < N_hidden"))
                    # byteAddr = rowByteBase + nhidden_pos * 2.
                    module.add(VLShiftLeftB32(dst=vgpr(resAddr), shiftHex=hex(1),
                                              src=vgpr(resAddr),
                                              comment="nhiddenByte = nhidden_pos * 2"))
                    module.add(VAddU32(vgpr(resAddr), vgpr(resAddr), vgpr(rowByteBase),
                                       comment="byteAddr = rowByteBase + nhiddenByte"))
                    module.add(VCndMaskB32(dst=vgpr(resAddr), src0=vgpr(oobV),
                                           src1=vgpr(resAddr),
                                           src2=sgpr(oobMask, self.lane_sgpr_count),
                                           comment="clamp OOB when nhidden_pos >= N_hidden"))

                    a = self._acc_idx(accVgprBase, m, n, k)
                    module.add(
                        BufferLoadD16B16(
                            vgpr(resTmp), vgpr(resAddr), sgpr(resSrd, 4), 0,
                            MUBUFModifiers(offen=True),
                            comment=f"R[token_n, nhidden_pos] bf16 (m={m},n={n},k={k})",
                        )
                    )
                    module.add(SWaitCnt(vlcnt=0, comment="wait residual load"))
                    module.add(VCvtBF16toFP32(vgpr(resTmp), vgpr(resTmp), None, 0,
                                              comment="residual bf16 -> fp32"))
                    module.add(VAccvgprReadB32(vgpr(accTmp), accvgpr(a),
                                               comment=f"read acc[m={m},n={n},k={k}]"))
                    module.add(VAddF32(dst=vgpr(accTmp), src0=vgpr(accTmp), src1=vgpr(resTmp),
                                       comment="H = GEMM + residual"))
                    module.add(VAccvgprWriteB32(accvgpr(a), vgpr(accTmp),
                                                comment=f"write acc[m={m},n={n},k={k}] += residual"))

        self.writer.sgprPool.checkIn(oobMask)
        self.writer.vgprPool.checkIn(oobV)
        self.writer.vgprPool.checkIn(rowByteBase)
        self.writer.vgprPool.checkIn(nOffVgpr)
        return module

    def _squareAndLaneSumFree0(
        self, accVgprBase: int, partials: int, accTmp: int
    ) -> Module:
        """Step 1 (free0): per-column Σx² over M-rows from fp32 AGPRs.

        For each n: partial[n] = Σ_{m,k} acc[m,n,k]².
        """
        module = Module("PartialRMS squareAndLaneSumFree0")
        module.addComment1("PartialRMS step 1 (free0): per-column Σx² over M-rows")
        for n in range(self.mma_n):
            pidx = partials + n
            first = self._acc_idx(accVgprBase, 0, n, 0)
            module.add(
                VAccvgprReadB32(vgpr(accTmp), accvgpr(first), comment=f"read acc[m=0,n={n},k=0]")
            )
            module.add(
                VMulF32(dst=vgpr(pidx), src0=vgpr(accTmp), src1=vgpr(accTmp),
                        comment=f"partial[n={n}] = acc^2")
            )
            for m in range(self.mma_m):
                for k in range(self.rows_per_lane):
                    if m == 0 and k == 0:
                        continue
                    a = self._acc_idx(accVgprBase, m, n, k)
                    module.add(
                        VAccvgprReadB32(vgpr(accTmp), accvgpr(a),
                                        comment=f"read acc[m={m},n={n},k={k}]")
                    )
                    module.add(
                        VFmaF32(dst=vgpr(pidx), src0=vgpr(accTmp), src1=vgpr(accTmp),
                                src2=vgpr(pidx), comment=f"partial[n={n}] += acc^2")
                    )
        return module

    def _rowGroupReduceFree0(self, partials: int) -> Module:
        """Step 2 (free0): all-reduce each partial[n] across row groups via ds_bpermute XOR butterfly.

        Combines the waveSize//mfma_n row groups so every lane holds the full
        column sum. Uses ds_bpermute_b32 instead of DPP because the reduction
        partners differ by mfma_n (not 1), crossing DPP row boundaries.
        """
        numRounds = int(math.log2(self.waveSize // self.mfma_n))
        module = Module("PartialRMS rowGroupReduceFree0")
        module.addComment1(
            f"PartialRMS step 2 (free0): XOR butterfly over {self.waveSize // self.mfma_n} row groups"
        )
        if numRounds == 0:
            return module

        laneIdV = self.writer.vgprPool.checkOut(1, tag="pRMS_rgrLaneId")
        addrV = self.writer.vgprPool.checkOut(1, tag="pRMS_rgrAddr")
        tmpV = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_rgrTmp")

        module.add(
            VAndB32(dst=vgpr(laneIdV), src0=vgpr("Serial"), src1=self.waveSize - 1,
                    comment="laneId = Serial & (waveSize-1)")
        )
        for i in range(numRounds):
            xorVal = self.mfma_n << i
            module.add(
                VXorB32(dst=vgpr(addrV), src0=vgpr(laneIdV), src1=xorVal,
                        comment=f"partnerLane = laneId ^ {xorVal}")
            )
            module.add(
                VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(2), src=vgpr(addrV),
                               comment="byteAddr = partnerLane * 4")
            )
            for n in range(self.numPartials):
                module.add(
                    DSBPermuteB32(vgpr(tmpV + n), vgpr(addrV), vgpr(partials + n),
                                  comment=f"fetch partner partial[{n}]")
                )
            module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
            for n in range(self.numPartials):
                module.add(
                    VAddF32(dst=vgpr(partials + n), src0=vgpr(partials + n),
                            src1=vgpr(tmpV + n), comment=f"partial[{n}] += partner")
                )

        self.writer.vgprPool.checkIn(tmpV)
        self.writer.vgprPool.checkIn(addrV)
        self.writer.vgprPool.checkIn(laneIdV)
        return module

    def _crossWaveReduceFree0(self, partials: int) -> Module:
        """Step 3 (free0): LDS reduction across wg_m sibling waves sharing waveN.

        Wave-id convention: waveId = waveN * wg_m + waveM.
        Siblings sharing waveN occupy waveIds [waveN*wg_m, waveN*wg_m + wg_m).
        Slot stride: waveSize * numPartials * 4 bytes per wave slot.
        """
        strideW = self.waveSize * self.numPartials * 4
        laneSlotBytes = self.numPartials * 4

        module = Module("PartialRMS crossWaveReduceFree0")
        module.addComment1(
            f"PartialRMS step 3 (free0): cross-wave LDS reduction over wg_m={self.wg_m}"
        )
        module.add(
            self.writer._syncThreads(
                self.kernel,
                "partialRMS free0 cross-wave: ensure siblings done reading LDS before scratch write",
            )
        )

        waveId = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WaveId")
        waveM = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WaveM")
        laneLoc = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0Lane")
        writeAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WriteAddr")
        readAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0ReadAddr")
        readBaseWave = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0ReadBase")
        readTmp = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_xwF0ReadTmp")
        tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_xwF0Tmp")
        tmpRes = ContinuousRegister(tmpVgpr, 2)

        module.add(
            VAndB32(dst=vgpr(laneLoc), src0=vgpr("Serial"), src1=self.waveSize - 1,
                    comment="laneId for LDS addressing")
        )
        module.add(
            vectorStaticDivide(waveId, "Serial", self.waveSize, tmpRes,
                               comment="waveId = Serial / WavefrontSize")
        )
        module.add(
            VAndB32(dst=vgpr(waveM), src0=vgpr(waveId), src1=self.wg_m - 1,
                    comment=f"waveM = waveId %% {self.wg_m}")
        )
        # readBaseWave = waveId XOR waveM = waveN * wg_m (low bits cancel because waveM = waveId & mask).
        module.add(
            VXorB32(dst=vgpr(readBaseWave), src0=vgpr(waveId), src1=vgpr(waveM),
                    comment="readBaseWave = waveN * wg_m")
        )

        with self.writer.allocTmpSgpr(1, tag="pRMS_xwF0AddrSetup") as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(strideW), comment=f"strideW={strideW}"))
            module.add(
                VMulLOU32(dst=vgpr(writeAddr), src0=sgpr(tmpSgpr), src1=vgpr(waveId),
                          comment="writeAddr = waveId * strideW")
            )
            module.add(
                VMulLOU32(dst=vgpr(readAddr), src0=sgpr(tmpSgpr), src1=vgpr(readBaseWave),
                          comment="readAddr = readBaseWave * strideW")
            )
            module.add(
                SMovB32(dst=sgpr(tmpSgpr), src=hex(laneSlotBytes),
                        comment=f"laneSlotBytes={laneSlotBytes}")
            )
            module.add(
                VMulLOU32(dst=vgpr(laneLoc), src0=sgpr(tmpSgpr), src1=vgpr(laneLoc),
                          comment="lane * laneSlotBytes")
            )
            module.add(
                VAddU32(vgpr(writeAddr), vgpr(writeAddr), vgpr(laneLoc),
                        comment="writeAddr += lane*laneSlotBytes")
            )
            module.add(
                VAddU32(vgpr(readAddr), vgpr(readAddr), vgpr(laneLoc),
                        comment="readAddr += lane*laneSlotBytes")
            )

        for i in range(self.numPartials):
            module.add(
                DSStoreB32(dstAddr=vgpr(writeAddr), src=vgpr(partials + i),
                           ds=DSModifiers(offset=i * 4), comment=f"LDS store partial[{i}]")
            )
        module.add(SWaitCnt(dscnt=0, comment="wait LDS writes"))
        module.add(self.writer._syncThreads(self.kernel, "partialRMS free0 cross-wave write"))

        for j in range(self.wg_m):
            for i in range(self.numPartials):
                module.add(
                    DSLoadB32(dst=vgpr(readTmp + i), src=vgpr(readAddr),
                              ds=DSModifiers(offset=i * 4),
                              comment=f"LDS load wave[{j}] partial[{i}]")
                )
            module.add(SWaitCnt(dscnt=0, comment="wait LDS reads"))
            for i in range(self.numPartials):
                if j == 0:
                    module.add(
                        VMovB32(dst=vgpr(partials + i), src=vgpr(readTmp + i),
                                comment=f"partial[{i}] = wave[0]")
                    )
                else:
                    module.add(
                        VAddF32(dst=vgpr(partials + i), src0=vgpr(partials + i),
                                src1=vgpr(readTmp + i), comment=f"partial[{i}] += wave[{j}]")
                    )
            if j < self.wg_m - 1:
                with self.writer.allocTmpSgpr(1, tag="pRMS_xwF0Advance") as tmpSgprInfo:
                    module.add(
                        SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(strideW),
                                comment=f"strideW={strideW}")
                    )
                    module.add(
                        VAddU32(vgpr(readAddr), vgpr(readAddr), sgpr(tmpSgprInfo.idx),
                                comment="advance readAddr to next sibling wave")
                    )

        module.add(self.writer._syncThreads(self.kernel, "partialRMS free0 cross-wave done"))

        self.writer.vgprPool.checkIn(tmpVgpr)
        self.writer.vgprPool.checkIn(readTmp)
        self.writer.vgprPool.checkIn(readAddr)
        self.writer.vgprPool.checkIn(readBaseWave)
        self.writer.vgprPool.checkIn(writeAddr)
        self.writer.vgprPool.checkIn(laneLoc)
        self.writer.vgprPool.checkIn(waveM)
        self.writer.vgprPool.checkIn(waveId)
        return module

    def _writePartialsFree0(
        self,
        partials: int,
        partialSrd: int,
        laneId: int,
        savedExec: int,
        laneMaskSgpr: int,
        globalAddr: int,
        colByte: int,
    ) -> Module:
        """Step 4 (free0): write per-column (per-token) Σx² to partialBuf[token, WorkGroup0].

        Only lanes with rowGroup==0 in waves with waveM==0 write, since every
        lane already holds the all-reduced value after _rowGroupReduceFree0 and
        _crossWaveReduceFree0. Write address: (token * n_d + WorkGroup0) * 4.
        """
        module = Module("PartialRMS writePartialsFree0")
        module.addComment1(
            "PartialRMS step 4 (free0): predicated write of Σx² to partialBuf[token, WG0]"
        )

        lsc = self.lane_sgpr_count
        log2MfmaN = int(math.log2(self.mfma_n))

        # Build write predicate: active iff rowGroup==0 AND waveM==0.
        rgV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0RowGroup")
        module.add(
            VLShiftRightB32(dst=vgpr(rgV), shiftHex=hex(log2MfmaN), src=vgpr(laneId),
                            comment=f"rowGroup = laneId >> {log2MfmaN}")
        )
        if self.wg_m > 1:
            waveIdTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0WaveId")
            waveMv = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0WaveM")
            tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_wF0TmpDiv")
            tmpRes = ContinuousRegister(tmpVgpr, 2)
            module.add(
                vectorStaticDivide(waveIdTmp, "Serial", self.waveSize, tmpRes,
                                   comment="waveId = Serial / WavefrontSize")
            )
            module.add(
                VAndB32(dst=vgpr(waveMv), src0=vgpr(waveIdTmp), src1=self.wg_m - 1,
                        comment=f"waveM = waveId %% {self.wg_m}")
            )
            module.add(
                VOrB32(dst=vgpr(rgV), src0=vgpr(rgV), src1=vgpr(waveMv),
                       comment="selV = rowGroup | waveM (zero iff both zero)")
            )
            self.writer.vgprPool.checkIn(tmpVgpr)
            self.writer.vgprPool.checkIn(waveIdTmp)
            self.writer.vgprPool.checkIn(waveMv)
        module.add(
            VCmpEQU32(dst=sgpr(laneMaskSgpr, lsc), src0=0, src1=vgpr(rgV),
                      comment="laneMask: rowGroup==0 && waveM==0")
        )
        self.writer.vgprPool.checkIn(rgV)

        # Compute n_d = ceil(SizesFree0 / MT0) into ntilesV.
        log2Mt0 = int(math.log2(self.macro_tile0))
        ntilesV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0NTiles")
        with self.writer.allocTmpSgpr(1, tag="pRMS_wF0NTilesS") as ntilesS:
            module.add(
                SAddU32(dst=sgpr(ntilesS.idx), src0=sgpr("SizesFree+0"),
                        src1=self.macro_tile0 - 1,
                        comment=f"N_hidden + MT0-1 (MT0={self.macro_tile0})")
            )
            module.add(
                SLShiftRightB32(dst=sgpr(ntilesS.idx), shiftHex=hex(log2Mt0),
                                src=sgpr(ntilesS.idx),
                                comment=f"n_d = ceil(SizesFree0 / MT0={self.macro_tile0})")
            )
            module.add(VMovB32(dst=vgpr(ntilesV), src=sgpr(ntilesS.idx), comment="ntilesV = n_d"))

        # tokenBase = colByte >> 1 = colInMma + waveN*(mma_n*mfma_n) + WG1*MT1.
        tokenBase = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0TokenBase")
        module.add(
            VLShiftRightB32(dst=vgpr(tokenBase), shiftHex=hex(1), src=vgpr(colByte),
                            comment="tokenBase = colByte >> 1 (token index for n=0)")
        )

        module.add(
            SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(laneMaskSgpr, lsc),
                            comment="save exec; set exec = writing-lane mask")
        )

        # Write partial[n] to partialBuf[token, WorkGroup0].
        nOffVgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0NOff")
        for n in range(self.mma_n):
            nOff = n * self.mfma_n
            if nOff > 64:
                module.add(VMovB32(dst=vgpr(nOffVgpr), src=nOff, comment=f"nOff={nOff} (n={n})"))
                module.add(
                    VAddU32(vgpr(globalAddr), vgpr(tokenBase), vgpr(nOffVgpr),
                            comment=f"token = tokenBase + {nOff} (n={n})")
                )
            else:
                module.add(
                    VAddU32(vgpr(globalAddr), vgpr(tokenBase), nOff,
                            comment=f"token = tokenBase + {nOff} (n={n})")
                )
            module.add(
                VMulLOU32(dst=vgpr(globalAddr), src0=vgpr(ntilesV), src1=vgpr(globalAddr),
                          comment="token * n_d")
            )
            module.add(
                VAddU32(vgpr(globalAddr), vgpr(globalAddr), sgpr("WorkGroup0"),
                        comment="+ WorkGroup0 (free0 tile index)")
            )
            module.add(
                VLShiftLeftB32(dst=vgpr(globalAddr), shiftHex=hex(2), src=vgpr(globalAddr),
                               comment="byteOff = (token*n_d + WG0) * 4")
            )
            module.add(
                BufferStoreB32(
                    src=vgpr(partials + n), vaddr=vgpr(globalAddr),
                    saddr=sgpr(partialSrd, 4), soffset=0, mubuf=MUBUFModifiers(offen=True),
                    comment=f"partialBuf[token, WG0] = Σx² (n={n})",
                )
            )
        module.add(SWaitCnt(vlcnt=0, comment="wait partialBuf stores"))
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc), comment="restore exec mask"))

        self.writer.vgprPool.checkIn(nOffVgpr)
        self.writer.vgprPool.checkIn(tokenBase)
        self.writer.vgprPool.checkIn(ntilesV)
        return module

    def _applyGammaFree0(
        self,
        accVgprBase: int,
        gammaSrd: int,
        gammaTmp: int,
        accTmp: int,
        scratchV: int,
    ) -> Module:
        """Step 5 (free0): load gamma (bf16) by free0 row and multiply each accumulator in-place.

        Gamma is indexed by globalRow (the free0 hidden dimension). One gamma load
        per (m, k) pair; all n-tiles share that gamma value.
        """
        module = Module("PartialRMS applyGammaFree0")
        module.addComment1("PartialRMS step 5 (free0): apply gamma[free0 row] in-place")

        log2MfmaN = int(math.log2(self.mfma_n))

        # rowGroupOff = rowGroup * rows_per_lane, where rowGroup = laneId >> log2(mfma_n).
        laneIdV = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0LaneId")
        rowGroupOff = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0RGOff")
        module.add(
            VAndB32(dst=vgpr(laneIdV), src0=vgpr("Serial"), src1=self.waveSize - 1,
                    comment="laneId = Serial & (waveSize-1)")
        )
        module.add(
            VLShiftRightB32(dst=vgpr(rowGroupOff), shiftHex=hex(log2MfmaN), src=vgpr(laneIdV),
                            comment=f"rowGroup = laneId >> {log2MfmaN}")
        )
        module.add(
            VMulLOU32(dst=vgpr(rowGroupOff), src0=self.rows_per_lane, src1=vgpr(rowGroupOff),
                      comment=f"rowGroupOff = rowGroup * {self.rows_per_lane}")
        )
        self.writer.vgprPool.checkIn(laneIdV)

        # wgRowBase = WorkGroup0 * MT0 (+ waveM * mma_m * mfma_m when wg_m > 1).
        mt0Vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0MT0")
        wgRowBase = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0WgRowBase")
        module.add(VMovB32(dst=vgpr(mt0Vgpr), src=self.macro_tile0, comment=f"MT0={self.macro_tile0}"))
        module.add(
            VMulLOU32(dst=vgpr(wgRowBase), src0=vgpr(mt0Vgpr), src1=sgpr("WorkGroup0"),
                      comment="wgRowBase = WorkGroup0 * MT0")
        )
        self.writer.vgprPool.checkIn(mt0Vgpr)

        if self.wg_m > 1:
            waveIdTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0WaveId")
            waveMTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0WaveM")
            tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_agF0TmpDiv")
            tmpRes = ContinuousRegister(tmpVgpr, 2)
            module.add(
                vectorStaticDivide(waveIdTmp, "Serial", self.waveSize, tmpRes,
                                   comment="waveId = Serial / WavefrontSize")
            )
            module.add(
                VAndB32(dst=vgpr(waveMTmp), src0=vgpr(waveIdTmp), src1=self.wg_m - 1,
                        comment=f"waveM = waveId %% {self.wg_m}")
            )
            waveStride = self.mma_m * self.mfma_m
            waveStrideV = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0WaveStride")
            module.add(
                VMovB32(dst=vgpr(waveStrideV), src=waveStride,
                        comment=f"waveStride = mma_m * mfma_m = {waveStride}")
            )
            module.add(
                VMulLOU32(dst=vgpr(waveMTmp), src0=vgpr(waveStrideV), src1=vgpr(waveMTmp),
                          comment="waveMOff = waveM * waveStride")
            )
            module.add(
                VAddU32(vgpr(wgRowBase), vgpr(wgRowBase), vgpr(waveMTmp),
                        comment="wgRowBase += waveMOff")
            )
            self.writer.vgprPool.checkIn(waveStrideV)
            self.writer.vgprPool.checkIn(tmpVgpr)
            self.writer.vgprPool.checkIn(waveMTmp)
            self.writer.vgprPool.checkIn(waveIdTmp)

        # Reuse scratchV (globalAddr from emit) as gammaByteVgpr.
        gammaByteVgpr = scratchV
        mBaseVgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0MBase")

        for m in range(self.mma_m):
            for k in range(self.rows_per_lane):
                mBase = m * self.mfma_m + k
                # globalRow = wgRowBase + rowGroupOff + mBase.
                if mBase > 64:
                    module.add(
                        VMovB32(dst=vgpr(mBaseVgpr), src=mBase, comment=f"mBase={mBase}")
                    )
                    module.add(
                        VAddU32(vgpr(gammaByteVgpr), vgpr(wgRowBase), vgpr(mBaseVgpr),
                                comment=f"globalRow = wgRowBase + {mBase}")
                    )
                else:
                    module.add(
                        VAddU32(vgpr(gammaByteVgpr), vgpr(wgRowBase), mBase,
                                comment=f"globalRow = wgRowBase + {mBase}")
                    )
                module.add(
                    VAddU32(vgpr(gammaByteVgpr), vgpr(gammaByteVgpr), vgpr(rowGroupOff),
                            comment="globalRow += rowGroupOff")
                )
                # gammaByte = globalRow * 2 (bf16 element size = 2 bytes).
                module.add(
                    VLShiftLeftB32(dst=vgpr(gammaByteVgpr), shiftHex=hex(1),
                                   src=vgpr(gammaByteVgpr), comment="gammaByte = globalRow * 2")
                )
                module.add(
                    BufferLoadD16B16(
                        vgpr(gammaTmp), vgpr(gammaByteVgpr), sgpr(gammaSrd, 4), 0,
                        MUBUFModifiers(offen=True),
                        comment=f"gamma bf16[globalRow] (m={m},k={k})",
                    )
                )
                module.add(SWaitCnt(vlcnt=0, comment="wait gamma load"))
                module.add(
                    VCvtBF16toFP32(vgpr(gammaTmp), vgpr(gammaTmp), None, 0,
                                   comment="gamma bf16 -> fp32")
                )
                for n in range(self.mma_n):
                    a = self._acc_idx(accVgprBase, m, n, k)
                    module.add(
                        VAccvgprReadB32(vgpr(accTmp), accvgpr(a),
                                        comment=f"read acc[m={m},n={n},k={k}]")
                    )
                    module.add(
                        VMulF32(dst=vgpr(accTmp), src0=vgpr(accTmp), src1=vgpr(gammaTmp),
                                comment="acc *= gamma")
                    )
                    module.add(
                        VAccvgprWriteB32(accvgpr(a), vgpr(accTmp),
                                         comment=f"write acc[m={m},n={n},k={k}]")
                    )

        self.writer.vgprPool.checkIn(mBaseVgpr)
        self.writer.vgprPool.checkIn(wgRowBase)
        self.writer.vgprPool.checkIn(rowGroupOff)
        return module
