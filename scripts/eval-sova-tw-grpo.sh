#!/usr/bin/env bash
set -euo pipefail

# Runtime and paths
PROJECT_ROOT="/path/to/SOVA-TW-GRPO"
PROJECT_ENV="/path/to/conda/envs/sova-tw-grpo"
CUDA_TOOLKIT="/path/to/cuda-12.4"
VAL_DATA_JSON="/path/to/datasets/CLEVRER/clevrer_counterfactual_val.json"

# Checkpoint and outputs
MODEL_RUN="Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_sovatwgrpo_a1p70_l0p125_vr2p0"
CHECKPOINT_STEP=500
CHECKPOINT="${PROJECT_ROOT}/outputs/${MODEL_RUN}/checkpoint-${CHECKPOINT_STEP}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${MODEL_RUN}/evaluation"
OUTPUT_JSON="${OUTPUT_DIR}/checkpoint-${CHECKPOINT_STEP}_eval_clevrer.json"

# CLEVRER evaluation parameters
EVAL_BATCH_SIZE=1
VIDEO_MAX_PIXELS=$((256 * 28 * 28))
FPS_MAX_FRAMES=16
SAMPLE_MODE=true

: "${CUDA_VISIBLE_DEVICES:?Run with CUDA_VISIBLE_DEVICES=GPU_ID bash scripts/eval-sova-tw-grpo.sh}"

for required_path in \
    "${PROJECT_ENV}/bin/python" \
    "${CUDA_TOOLKIT}/bin/nvcc" \
    "${CHECKPOINT}/config.json" \
    "${CHECKPOINT}/model.safetensors.index.json" \
    "${CHECKPOINT}/preprocessor_config.json" \
    "${VAL_DATA_JSON}" \
    "${PROJECT_ROOT}/src/eval/eval_clevrer.py"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path is missing: ${required_path}" >&2
        exit 3
    fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

export CUDA_HOME="${CUDA_TOOLKIT}"
export PATH="${CUDA_HOME}/bin:${PROJECT_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VIDEO_MAX_PIXELS
export FPS_MAX_FRAMES
export SAMPLE_MODE

exec "${PROJECT_ENV}/bin/python" src/eval/eval_clevrer.py \
    --model_name "${CHECKPOINT}" \
    --dataset_path "${VAL_DATA_JSON}" \
    --output_path "${OUTPUT_JSON}" \
    --batch_size "${EVAL_BATCH_SIZE}"
