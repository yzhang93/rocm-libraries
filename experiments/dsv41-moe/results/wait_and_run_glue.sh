#!/bin/bash
# Wait until one of GPUs 0-3 is idle (no other process on it, under 5% VRAM and 0% use for
# three checks in a row), then start the dsv41 container and run moe_glue_bench.py on that GPU.
WS=/mnt/nvme1/vivizhan/dsv41_inferencex
LOG=$WS/experiments/moe/results/wait_and_run_glue.log
idle_gpu() {
  local use vram
  for g in 0 1 2 3; do
    use=$(rocm-smi -d $g --showuse | grep -oP 'GPU use \(%\): \K[0-9]+')
    vram=$(rocm-smi -d $g --showmemuse | grep -oP 'VRAM%\): \K[0-9]+')
    if [ "${use:-100}" -eq 0 ] && [ "${vram:-100}" -lt 5 ]; then echo $g; return; fi
  done
}
streak=0; last=""
while :; do
  g=$(idle_gpu)
  if [ -n "$g" ] && [ "$g" = "$last" ]; then streak=$((streak + 1)); else streak=0; fi
  last=$g
  echo "$(date -u +%FT%TZ) idle=${g:-none} streak=$streak" >> $LOG
  if [ "$streak" -ge 2 ]; then
    docker start vivizhan-dsv41 >/dev/null
    sleep 30
    if [ "$(docker inspect -f '{{.State.Status}}' vivizhan-dsv41)" != running ]; then
      echo "$(date -u +%FT%TZ) container stopped again; waiting" >> $LOG
      streak=0; sleep 600; continue
    fi
    echo "$(date -u +%FT%TZ) GPU_READY gpu=$g, running" >> $LOG
    docker exec vivizhan-dsv41 bash -c "cd $WS/experiments/moe && \
      LD_LIBRARY_PATH=$WS/hipblaslt-gg-f4f8-geko/install/lib:\$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=$g \
      GG_SHIM=libgg_shim_f4f8_geko.so python3 moe_glue_bench.py" > $WS/experiments/moe/results/moe_glue_bench.log 2>&1
    echo "$(date -u +%FT%TZ) RUN_DONE exit=$?" >> $LOG
    exit 0
  fi
  sleep 300
done
