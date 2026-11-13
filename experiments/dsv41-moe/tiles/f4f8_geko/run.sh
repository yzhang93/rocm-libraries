#!/bin/bash
# run.sh GPU GEMM ORDER: ORDER "mixed_first" runs the mixed sweep with checks, then the MXFP8 control;
# "mxfp8_first" runs the MXFP8 control, then the mixed sweep without checks (replicate).
WS=/mnt/nvme1/vivizhan/dsv41_inferencex
INST=$WS/hipblaslt-gg-f4f8-geko/install
cd $WS/experiments/moe/tiles
GPU=$1; GEMM=$2; ORDER=$3; g=${GEMM/gemm/g}; D=f4f8_geko
run() {  # ab label tiles extra
  LD_LIBRARY_PATH=$INST/lib:$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=$GPU GG_SHIM=libgg_shim_f4f8_geko.so \
    python3 tile_sweep.py --ab $1 --shapes $GEMM --tiles "$(cat $D/$3)" $4 --out $D/$2_${GEMM}_gpu$GPU.json \
    > $D/$2_${GEMM}_gpu$GPU.log 2>&1
  echo "EXIT=$?" >> $D/$2_${GEMM}_gpu$GPU.log
}
if [ "$ORDER" = mixed_first ]; then
  run 3 mixed mixed_$g.arg ""
  run 0 mxfp8 mxfp8_$g.arg --no-check
else
  run 0 mxfp8 mxfp8_$g.arg --no-check
  run 3 mixed mixed_$g.arg --no-check
fi
