#!/usr/bin/env bash
set -euo pipefail


PROJECT_ROOT="${PROJECT_ROOT:-/mnt/d/Abiyelunwen/new_fire_vlm}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/qwen3vl/train_qwen3vl.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/qwen3vl/val_qwen3vl.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen3vl_2b_lora}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

mkdir -p "${OUTPUT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-512}"

swift sft \
  --model "${MODEL_NAME}" \
  --use_hf true \
  --dataset "${TRAIN_JSONL}" \
  --val_dataset "${VAL_JSONL}" \
  --tuner_type lora \
  --quant_bits 4 \
  --torch_dtype float16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --max_length 1024 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --warmup_ratio 0.03 \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"
