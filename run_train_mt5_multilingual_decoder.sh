#!/bin/bash
#SBATCH --gres=gpu:a40
#SBATCH --job-name=mt5_multi_decoder
#SBATCH --output=log/mt5_multi_decoder_%j.out
#SBATCH --error=log/mt5_multi_decoder_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-../python3.sif}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

MODEL_NAME="${MODEL_NAME:-google/mt5-small}"
TRAIN_DATASETS="${TRAIN_DATASETS:-yywwrr/mmarco_english,yywwrr/mmarco_french,yywwrr/mmarco_german,yywwrr/mmarco_italian,yywwrr/mmarco_portuguese,yywwrr/mmarco_spanish,yywwrr/mmarco_dutch}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/decoders}"
MAX_LENGTH="${MAX_LENGTH:-32}"
PROMPT_LENGTH="${PROMPT_LENGTH:-32}"

# Total training budget across all languages. src/train_decoder.py splits this
# approximately evenly over TRAIN_DATASETS.
TRAIN_SAMPLES="${TRAIN_SAMPLES:-450000}"
VAL_SAMPLES="${VAL_SAMPLES:-700}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"

echo "Training multilingual mT5-based attack decoder"
echo "model=${MODEL_NAME}"
echo "train_datasets=${TRAIN_DATASETS}"
echo "total_train_samples=${TRAIN_SAMPLES}, val_samples=${VAL_SAMPLES}"
echo "batch_size=${BATCH_SIZE}, epochs=${NUM_EPOCHS}"

singularity exec --nv "${SINGULARITY_IMAGE}" "${PYTHON_BIN}" src/train_decoder.py \
  --model_name "${MODEL_NAME}" \
  --train_datasets "${TRAIN_DATASETS}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --prompt_length "${PROMPT_LENGTH}" \
  --train_samples "${TRAIN_SAMPLES}" \
  --val_samples "${VAL_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_epochs "${NUM_EPOCHS}"
