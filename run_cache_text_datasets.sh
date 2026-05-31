#!/bin/bash
#SBATCH --job-name=lago_cache_data
#SBATCH --output=log/lago_cache_data_%j.out
#SBATCH --error=log/lago_cache_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-../python3.sif}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
LOCAL_PYTHON_BIN="${LOCAL_PYTHON_BIN:-python3}"

DATASETS="${DATASETS:-yywwrr/mmarco_english,yywwrr/mmarco_french,yywwrr/mmarco_german,yywwrr/mmarco_italian,yywwrr/mmarco_portuguese,yywwrr/mmarco_spanish,yywwrr/mmarco_dutch}"
DATA_FOLDER="${DATA_FOLDER:-datasets/finetuning_decoder}"

echo "Caching text datasets"
echo "datasets=${DATASETS}"
echo "data_folder=${DATA_FOLDER}"

if command -v singularity >/dev/null 2>&1 && [ -f "${SINGULARITY_IMAGE}" ]; then
  singularity exec "${SINGULARITY_IMAGE}" "${PYTHON_BIN}" src/cache_text_datasets.py \
    --datasets "${DATASETS}" \
    --data_folder "${DATA_FOLDER}"
else
  "${LOCAL_PYTHON_BIN}" src/cache_text_datasets.py \
    --datasets "${DATASETS}" \
    --data_folder "${DATA_FOLDER}"
fi
