#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:-phase1to3_2gpu_$(date +%Y%m%d-%H%M%S)}"
CUDA_DEVICES="${2:-0,2}"
CKPT_ROOT="${3:-checkpoints/${RUN_TAG}}"
OUT_ROOT="${4:-outputs/${RUN_TAG}}"

mkdir -p "${OUT_ROOT}" "${CKPT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export NPROC_PER_NODE=2
export MASTER_PORT="${MASTER_PORT:-12347}"

COMMON_ARGS=(
  --checkpoint_dir "${CKPT_ROOT}"
  --output_dir "${OUT_ROOT}"
  --auto_adjust_batch_size false
  --per_device_batch_size 16
  --gradient_accumulation_steps 4
  --target_global_batch_size 128
)

echo "[$(date '+%F %T')] Starting Phase 1 -> Phase 3 two-GPU rerun"
echo "RUN_TAG=${RUN_TAG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "checkpoint_root=${CKPT_ROOT}"
echo "output_root=${OUT_ROOT}"

echo "[$(date '+%F %T')] Phase 1 start"
bash scripts/run_phase1_id.sh "${COMMON_ARGS[@]}" --label_screen "phase1-${RUN_TAG}"
echo "[$(date '+%F %T')] Phase 1 done"

echo "[$(date '+%F %T')] Phase 2 start"
bash scripts/run_phase2_align.sh "${COMMON_ARGS[@]}" --label_screen "phase2-${RUN_TAG}"
echo "[$(date '+%F %T')] Phase 2 done"

echo "[$(date '+%F %T')] Phase 3 start"
bash scripts/run_phase3_3tower.sh "${COMMON_ARGS[@]}" --label_screen "phase3-${RUN_TAG}"
echo "[$(date '+%F %T')] Phase 3 done"
