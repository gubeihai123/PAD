#!/usr/bin/env bash
# Generate Prime Pantry LLM2Vec item embeddings. Set CUDA_VISIBLE_DEVICES outside this script.
set -euo pipefail

python create_llm2vec_amazon.py \
  --data_dir ./dataset \
  --model_dir ./models \
  --llama_path ./models/Meta-Llama-3-8B-Instruct \
  --llm2vec_path ./models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
  --llm2vec_output ./dataset/Amazon_Prime_Pantry_llm2vec.pt \
  --batch_size 128 \
  "$@"
