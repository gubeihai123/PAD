#!/usr/bin/env bash
# Train Phase 2 Prime Pantry alignment tower from Phase 1 best.pt. Set CUDA_VISIBLE_DEVICES outside this script.
set -euo pipefail

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  NPROC="${NPROC_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NPROC="$(python -c "import os; print(len([x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]))")"
else
  NPROC="$(python -c "import torch; print(torch.cuda.device_count())")"
fi
MASTER_PORT="${MASTER_PORT:-12345}"

torchrun --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
  run_amazon_Prime_Pantry_transfer.py \
  --data_dir ./dataset \
  --model_dir ./models \
  --output_dir ./outputs \
  --checkpoint_dir ./checkpoints \
  --llm2vec_output ./dataset/Amazon_Prime_Pantry_llm2vec.pt \
  --mode train \
  --item_tower modal_cat \
  --bert_model_load llm \
  --news_attributes title \
  --embedding_dim 128 \
  --transformer_block 5 \
  --mo_dnn_layers 4 \
  --dnn_layers 0 \
  --drop_rate 0.1 \
  --l2_weight 0.1 \
  --lr 1e-5 \
  --gamma 0.2 \
  --gamma2 -4 \
  --load_ckpt_name best.pt \
  --per_device_batch_size 16 \
  --target_global_batch_size 128 \
  --eval_batch_size 512 \
  --patience 20 \
  --epoch 400 \
  --num_workers 8 \
  --logging_num 4 \
  --testing_num 1 \
  --label_screen phase2 \
  "$@"
