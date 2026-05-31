#!/bin/bash
#SBATCH --gres=gpu:a40
#SBATCH --job-name=lago_conn
#SBATCH --output=log/lago_conn_%j.out
#SBATCH --error=log/lago_conn_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-../python3.sif}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

DECODER_CHECKPOINT_PATH="${DECODER_CHECKPOINT_PATH:-outputs/decoders/google_mt5-small/yywwrr_mmarco_english_500k_maxlength32_train450000_batch128_lr0.0001_wd0.0001_epochs100}"
DECODER_MODEL_SLUG="${DECODER_MODEL_SLUG:-google_mt5-small}"
SOURCE_MODEL_NAME="${SOURCE_MODEL_NAME:-google/mt5-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lago_ablation_connectivity}"
DATASETS="${DATASETS:-yywwrr/mmarco_english,yywwrr/mmarco_french,yywwrr/mmarco_german,yywwrr/mmarco_italian,yywwrr/mmarco_portuguese,yywwrr/mmarco_spanish,yywwrr/mmarco_dutch}"
LANG_KEYS="${LANG_KEYS:-en,fr,de,it,pt,es,nl}"
DATA_FOLDER="${DATA_FOLDER:-datasets/finetuning_decoder}"

GRAPH_DIR="${GRAPH_DIR:-graphs/mmarco_7}"
GRAPH_FILES="${GRAPH_FILES:-}"
INCLUDE_NONE="${INCLUDE_NONE:-1}"
INCLUDE_REAL="${INCLUDE_REAL:-1}"
INCLUDE_RANDOM="${INCLUDE_RANDOM:-1}"

CONSTRAINT_MODE="${CONSTRAINT_MODE:-totalvariation}"
ALIGN_TRAIN_SAMPLES="${ALIGN_TRAIN_SAMPLES:-10 30 100 300 500 1000}"
VAL_SAMPLES="${VAL_SAMPLES:-200}"
TEST_SAMPLES="${TEST_SAMPLES:-200}"
REG_LAMBDA="${REG_LAMBDA:-0.01}"
EPSILON="${EPSILON:-0.01}"
NUM_ITER="${NUM_ITER:-500}"
SEED="${SEED:-42}"
SKIP_EXISTING_RESULTS="${SKIP_EXISTING_RESULTS:-1}"
RUN_RETRIES="${RUN_RETRIES:-3}"
RUN_RETRY_SLEEP="${RUN_RETRY_SLEEP:-60}"
SOURCE_MODEL_SLUG="${SOURCE_MODEL_NAME//\//_}"

if [ -z "${GRAPH_FILES}" ]; then
  DEFAULT_GRAPH_FILES="
${GRAPH_DIR}/graph_lang2vec_0.45_signed.json
${GRAPH_DIR}/graph_lang2vec_topk3_signed.json
${GRAPH_DIR}/graph_lang2vec_topk6_signed.json
${GRAPH_DIR}/graph_lang2vec_topk9_signed.json
${GRAPH_DIR}/graph_lang2vec_topk12_signed.json
${GRAPH_DIR}/graph_lang2vec_topk15_signed.json
${GRAPH_DIR}/graph_lang2vec_topk18_signed.json
${GRAPH_DIR}/graph_ajsp_90_signed.json
${GRAPH_DIR}/graph_ajsp_topk3_signed.json
${GRAPH_DIR}/graph_ajsp_topk6_signed.json
${GRAPH_DIR}/graph_ajsp_topk9_signed.json
${GRAPH_DIR}/graph_ajsp_topk12_signed.json
${GRAPH_DIR}/graph_ajsp_topk15_signed.json
${GRAPH_DIR}/graph_ajsp_topk18_signed.json
"
  GRAPH_FILES=""
  for CANDIDATE_GRAPH_FILE in ${DEFAULT_GRAPH_FILES}; do
    if [ -f "${CANDIDATE_GRAPH_FILE}" ]; then
      GRAPH_FILES="${GRAPH_FILES} ${CANDIDATE_GRAPH_FILE}"
    fi
  done
fi

if [ -z "${GRAPH_FILES}" ]; then
  echo "No graph files found. Run sbatch run_generate_language_graphs.sh first."
  exit 1
fi

echo "Running LAGO connectivity ablation"
echo "hostname=$(hostname)"
echo "uid=$(id -u), gid=$(id -g)"
getent passwd "$(id -u)" || true
echo "decoder_checkpoint=${DECODER_CHECKPOINT_PATH}"
echo "decoder_model_slug=${DECODER_MODEL_SLUG}"
echo "source_model=${SOURCE_MODEL_NAME}"
echo "output_dir=${OUTPUT_DIR}"
echo "data_folder=${DATA_FOLDER}"
echo "graph_files=${GRAPH_FILES}"
echo "include_none=${INCLUDE_NONE}"
echo "include_real=${INCLUDE_REAL}"
echo "include_random=${INCLUDE_RANDOM}"
echo "align_train_samples=${ALIGN_TRAIN_SAMPLES}"
echo "seed=${SEED}"
echo "skip_existing_results=${SKIP_EXISTING_RESULTS}"
echo "run_retries=${RUN_RETRIES}"

infer_graph_type() {
  local label="$1"
  if [[ "${label}" == lang2vec_* ]]; then
    echo "lang2vec"
  elif [[ "${label}" == ajsp_* ]]; then
    echo "ajsp"
  else
    echo "lang2vec"
  fi
}

infer_random_graph_type() {
  local label="$1"
  if [[ "${label}" == ajsp_* ]]; then
    echo "random_ajsp"
  else
    echo "random_lang2vec"
  fi
}

result_json_path() {
  local graph_label="$1"
  local mode="$2"
  local train_n="$3"
  echo "${OUTPUT_DIR}/${DECODER_MODEL_SLUG}/${SOURCE_MODEL_SLUG}/${graph_label}_${mode}_train${train_n}_ridge${REG_LAMBDA}_eps${EPSILON}_seed${SEED}/results.json"
}

run_with_retry() {
  local attempt=1
  local status=0
  while true; do
    "$@" && return 0
    status=$?
    if [ "${attempt}" -ge "${RUN_RETRIES}" ]; then
      return "${status}"
    fi
    local delay=$((RUN_RETRY_SLEEP * attempt))
    echo "Command failed with status ${status}; retrying in ${delay}s (${attempt}/${RUN_RETRIES})"
    sleep "${delay}"
    attempt=$((attempt + 1))
  done
}

run_condition() {
  local train_n="$1"
  local graph_type="$2"
  local graph_label="$3"
  local mode="$4"
  local graph_file="${5:-}"
  local results_json
  results_json="$(result_json_path "${graph_label}" "${mode}" "${train_n}")"

  if [ "${SKIP_EXISTING_RESULTS}" = "1" ] && [ -s "${results_json}" ]; then
    echo "Skipping existing result: ${results_json}"
    return 0
  fi

  echo "Starting graph_label=${graph_label}, graph_type=${graph_type}, constraint_mode=${mode}, train_samples=${train_n}"
  if [ -n "${graph_file}" ]; then
    run_with_retry singularity exec --nv "${SINGULARITY_IMAGE}" "${PYTHON_BIN}" src/run_alignment_ablation.py \
      --decoder_checkpoint_path "${DECODER_CHECKPOINT_PATH}" \
      --source_model_name "${SOURCE_MODEL_NAME}" \
      --datasets "${DATASETS}" \
      --lang_keys "${LANG_KEYS}" \
      --data_folder "${DATA_FOLDER}" \
      --output_dir "${OUTPUT_DIR}" \
      --graph_type "${graph_type}" \
      --graph_file "${graph_file}" \
      --graph_label "${graph_label}" \
      --constraint_mode "${mode}" \
      --align_train_samples "${train_n}" \
      --val_samples "${VAL_SAMPLES}" \
      --test_samples "${TEST_SAMPLES}" \
      --reg_lambda "${REG_LAMBDA}" \
      --epsilon "${EPSILON}" \
      --num_iter "${NUM_ITER}" \
      --seed "${SEED}"
  else
    run_with_retry singularity exec --nv "${SINGULARITY_IMAGE}" "${PYTHON_BIN}" src/run_alignment_ablation.py \
      --decoder_checkpoint_path "${DECODER_CHECKPOINT_PATH}" \
      --source_model_name "${SOURCE_MODEL_NAME}" \
      --datasets "${DATASETS}" \
      --lang_keys "${LANG_KEYS}" \
      --data_folder "${DATA_FOLDER}" \
      --output_dir "${OUTPUT_DIR}" \
      --graph_type "${graph_type}" \
      --graph_label "${graph_label}" \
      --constraint_mode "${mode}" \
      --align_train_samples "${train_n}" \
      --val_samples "${VAL_SAMPLES}" \
      --test_samples "${TEST_SAMPLES}" \
      --reg_lambda "${REG_LAMBDA}" \
      --epsilon "${EPSILON}" \
      --num_iter "${NUM_ITER}" \
      --seed "${SEED}"
  fi
  echo "Finished graph_label=${graph_label}, train_samples=${train_n}"
}

for TRAIN_N in ${ALIGN_TRAIN_SAMPLES}; do
  if [ "${INCLUDE_NONE}" = "1" ]; then
    run_condition "${TRAIN_N}" "none" "none" "none"
  fi

  for GRAPH_FILE in ${GRAPH_FILES}; do
    BASE_NAME="$(basename "${GRAPH_FILE}")"
    LABEL="${BASE_NAME#graph_}"
    LABEL="${LABEL%_signed.json}"
    GRAPH_TYPE="$(infer_graph_type "${LABEL}")"
    RANDOM_GRAPH_TYPE="$(infer_random_graph_type "${LABEL}")"
    RANDOM_LABEL="random_${LABEL}"

    if [ "${INCLUDE_REAL}" = "1" ]; then
      run_condition "${TRAIN_N}" "${GRAPH_TYPE}" "${LABEL}" "${CONSTRAINT_MODE}" "${GRAPH_FILE}"
    fi
    if [ "${INCLUDE_RANDOM}" = "1" ]; then
      run_condition "${TRAIN_N}" "${RANDOM_GRAPH_TYPE}" "${RANDOM_LABEL}" "${CONSTRAINT_MODE}" "${GRAPH_FILE}"
    fi
  done
done
