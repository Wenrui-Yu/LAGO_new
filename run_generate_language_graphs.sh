#!/bin/bash
#SBATCH --job-name=lago_graphs
#SBATCH --output=log/lago_graphs_%j.out
#SBATCH --error=log/lago_graphs_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-../python3.sif}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
LOCAL_PYTHON_BIN="${LOCAL_PYTHON_BIN:-python3}"

OUTPUT_DIR="${OUTPUT_DIR:-graphs/mmarco_7}"
LANGUAGES="${LANGUAGES:-en,fr,de,it,pt,es,nl}"
SYNTACTIC_THRESHOLDS="${SYNTACTIC_THRESHOLDS:-0.45}"
AJSP_THRESHOLDS="${AJSP_THRESHOLDS:-90}"
SYNTACTIC_TOPK="${SYNTACTIC_TOPK:-3,6,9,12,15,18}"
AJSP_TOPK="${AJSP_TOPK:-3,6,9,12,15,18}"
SIGN_MODE="${SIGN_MODE:-legacy}"
ASJP_OUTPUT_PATH="${ASJP_OUTPUT_PATH:-resources/asjp/output.txt}"
LANG2VEC_PACKAGE_DIR="${LANG2VEC_PACKAGE_DIR:-../lang2vec-master}"
LANG2VEC_DISTANCE_PATH="${LANG2VEC_DISTANCE_PATH:-resources/lang2vec/syntactic_distances.csv}"

echo "Generating LAGO language graphs"
echo "output_dir=${OUTPUT_DIR}"
echo "languages=${LANGUAGES}"
echo "syntactic_thresholds=${SYNTACTIC_THRESHOLDS}"
echo "ajsp_thresholds=${AJSP_THRESHOLDS}"
echo "syntactic_topk=${SYNTACTIC_TOPK}"
echo "ajsp_topk=${AJSP_TOPK}"
echo "sign_mode=${SIGN_MODE}"
echo "asjp_output_path=${ASJP_OUTPUT_PATH}"
echo "lang2vec_package_dir=${LANG2VEC_PACKAGE_DIR}"
echo "lang2vec_distance_path=${LANG2VEC_DISTANCE_PATH}"

if command -v singularity >/dev/null 2>&1 && [ -f "${SINGULARITY_IMAGE}" ]; then
  singularity exec "${SINGULARITY_IMAGE}" "${PYTHON_BIN}" src/generate_language_graphs.py \
    --output_dir "${OUTPUT_DIR}" \
    --languages "${LANGUAGES}" \
    --syntactic_thresholds "${SYNTACTIC_THRESHOLDS}" \
    --ajsp_thresholds "${AJSP_THRESHOLDS}" \
    --syntactic_topk "${SYNTACTIC_TOPK}" \
    --ajsp_topk "${AJSP_TOPK}" \
    --lang2vec_package_dir "${LANG2VEC_PACKAGE_DIR}" \
    --lang2vec_distance_path "${LANG2VEC_DISTANCE_PATH}" \
    --asjp_output_path "${ASJP_OUTPUT_PATH}" \
    --sign_mode "${SIGN_MODE}"
else
  "${LOCAL_PYTHON_BIN}" src/generate_language_graphs.py \
    --output_dir "${OUTPUT_DIR}" \
    --languages "${LANGUAGES}" \
    --syntactic_thresholds "${SYNTACTIC_THRESHOLDS}" \
    --ajsp_thresholds "${AJSP_THRESHOLDS}" \
    --syntactic_topk "${SYNTACTIC_TOPK}" \
    --ajsp_topk "${AJSP_TOPK}" \
    --lang2vec_package_dir "${LANG2VEC_PACKAGE_DIR}" \
    --lang2vec_distance_path "${LANG2VEC_DISTANCE_PATH}" \
    --asjp_output_path "${ASJP_OUTPUT_PATH}" \
    --sign_mode "${SIGN_MODE}"
fi
