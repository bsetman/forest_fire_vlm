#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/d/Abiyelunwen/new_fire_vlm"
CONFIG="${PROJECT_ROOT}/scripts/train/internvl35_2b_hf_qlora_axolotl.yml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/internvl35_2b_hf_qlora"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
LOG_FILE="${LOG_DIR}/internvl35_2b_hf_qlora_train.log"
FINAL_ADAPTER_DIR="${PROJECT_ROOT}/outputs/adapters/internvl35_2b_hf_qlora_adapter"
TRAIN_JSONL="${PROJECT_ROOT}/data/internvl35/train_internvl35_axolotl.jsonl"
VAL_JSONL="${PROJECT_ROOT}/data/internvl35/val_internvl35_axolotl.jsonl"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${FINAL_ADAPTER_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}"
  exit 1
fi

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[ERROR] Train JSONL not found: ${TRAIN_JSONL}"
  exit 1
fi

if [[ ! -f "${VAL_JSONL}" ]]; then
  echo "[ERROR] Val JSONL not found: ${VAL_JSONL}"
  exit 1
fi

# Disable Axolotl telemetry and run single-GPU QLoRA.
AXOLOTL_DO_NOT_TRACK=1 \
DO_NOT_TRACK=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
axolotl train "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"

# Copy the latest adapter/checkpoint to a fixed path for later inference/evaluation.
LATEST_ADAPTER=""
if [[ -f "${OUTPUT_DIR}/adapter_model.safetensors" || -f "${OUTPUT_DIR}/adapter_model.bin" ]]; then
  LATEST_ADAPTER="${OUTPUT_DIR}"
else
  LATEST_ADAPTER=$(find "${OUTPUT_DIR}" -type f \( -name 'adapter_model.safetensors' -o -name 'adapter_model.bin' \) -printf '%h\n' | sort | tail -n 1 || true)
fi

if [[ -z "${LATEST_ADAPTER}" ]]; then
  echo "[WARN] No adapter_model.safetensors/bin found under ${OUTPUT_DIR}."
  echo "[WARN] Check the Axolotl output directory manually."
  exit 0
fi

rm -rf "${FINAL_ADAPTER_DIR}"
mkdir -p "${FINAL_ADAPTER_DIR}"
cp -a "${LATEST_ADAPTER}/." "${FINAL_ADAPTER_DIR}/"
cp -a "${CONFIG}" "${FINAL_ADAPTER_DIR}/training_config_used.yml"
cp -a "${LOG_FILE}" "${FINAL_ADAPTER_DIR}/train.log"

cat > "${FINAL_ADAPTER_DIR}/README.txt" <<EOF
InternVL3.5-2B-HF QLoRA adapter

Base model:
  OpenGVLab/InternVL3_5-2B-HF

Adapter path:
  ${FINAL_ADAPTER_DIR}

Training data:
  ${TRAIN_JSONL}
  ${VAL_JSONL}

Do not load this adapter on OpenGVLab/InternVL3_5-2B-Instruct.
Use it with OpenGVLab/InternVL3_5-2B-HF only.
EOF

echo "[DONE] Training finished."
echo "[DONE] Final adapter copied to: ${FINAL_ADAPTER_DIR}"
