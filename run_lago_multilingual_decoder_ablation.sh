#!/bin/bash
#SBATCH --gres=gpu:a40
#SBATCH --job-name=lago_mt5_multi
#SBATCH --output=log/lago_mt5_multi_%j.out
#SBATCH --error=log/lago_mt5_multi_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-../python3.sif}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

MULTI_DECODER_DIR="${MULTI_DECODER_DIR:-outputs/decoders/google_mt5-small/yywwrr_mmarco_english_yywwrr_mmarco_french_yywwrr_mmarco_german_yywwrr_mmarco_italian_yywwrr_mmarco_portuguese_yywwrr_mmarco_spanish_yywwrr_mmarco_dutch_maxlength32_train450000_batch128_lr0.0001_wd0.0001_epochs100}"
SOURCE_MODEL_NAME="${SOURCE_MODEL_NAME:-google/mt5-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lago_ablation_multilingual_decoder}"
DATASETS="${DATASETS:-yywwrr/mmarco_english,yywwrr/mmarco_french,yywwrr/mmarco_german,yywwrr/mmarco_italian,yywwrr/mmarco_portuguese,yywwrr/mmarco_spanish,yywwrr/mmarco_dutch}"
LANG_KEYS="${LANG_KEYS:-en,fr,de,it,pt,es,nl}"

GRAPH_TYPES="${GRAPH_TYPES:-none lang2vec random_lang2vec ajsp random_ajsp}"
CONSTRAINT_MODE="${CONSTRAINT_MODE:-totalvariation}"
ALIGN_TRAIN_SAMPLES="${ALIGN_TRAIN_SAMPLES:-10 30 100 300 500 1000}"
VAL_SAMPLES="${VAL_SAMPLES:-200}"
TEST_SAMPLES="${TEST_SAMPLES:-200}"
REG_LAMBDA="${REG_LAMBDA:-0.01}"
EPSILON="${EPSILON:-0.01}"
NUM_ITER="${NUM_ITER:-500}"

echo "Running LAGO ablation with multilingual fine-tuned mT5 decoder"
echo "decoder_checkpoint=${MULTI_DECODER_DIR}"
echo "source_model=${SOURCE_MODEL_NAME}"
echo "graph_types=${GRAPH_TYPES}"
echo "align_train_samples=${ALIGN_TRAIN_SAMPLES}"

for TRAIN_N in ${ALIGN_TRAIN_SAMPLES}; do
  for GRAPH_TYPE in ${GRAPH_TYPES}; do
    MODE="${CONSTRAINT_MODE}"
    if [ "${GRAPH_TYPE}" = "none" ]; then
      MODE="none"
    fi

    echo "Starting graph_type=${GRAPH_TYPE}, constraint_mode=${MODE}, train_samples=${TRAIN_N}"
    singularity exec --nv "${SINGULARITY_IMAGE}" "${PYTHON_BIN}" src/run_alignment_ablation.py \
      --decoder_checkpoint_path "${MULTI_DECODER_DIR}" \
      --source_model_name "${SOURCE_MODEL_NAME}" \
      --datasets "${DATASETS}" \
      --lang_keys "${LANG_KEYS}" \
      --output_dir "${OUTPUT_DIR}" \
      --graph_type "${GRAPH_TYPE}" \
      --constraint_mode "${MODE}" \
      --align_train_samples "${TRAIN_N}" \
      --val_samples "${VAL_SAMPLES}" \
      --test_samples "${TEST_SAMPLES}" \
      --reg_lambda "${REG_LAMBDA}" \
      --epsilon "${EPSILON}" \
      --num_iter "${NUM_ITER}"
    echo "Finished graph_type=${GRAPH_TYPE}, train_samples=${TRAIN_N}"
  done
done
