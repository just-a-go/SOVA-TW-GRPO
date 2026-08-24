#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
    echo "This launcher takes no positional arguments; edit the selector block instead." >&2
    exit 2
fi

# Runtime and paths
PROJECT_ROOT="/data0/codefile/wangweiqi/SOVA-TW-GRPO"
PROJECT_ENV="/data0/codefile/wangweiqi/SOVA-TW-GRPO/.conda/envs/sova-tw-grpo"
CUDA_TOOLKIT="/data0/codefile/wangweiqi/SOVA-TW-GRPO/.cuda-12.4"
MODEL_NAME_OR_PATH="/data0/pretrained/wangweiqi/Qwen2.5-VL-7B-Instruct"
TRAIN_DATA_JSON="/data0/data/CD-TW-GRPO/CLEVRER/clevrer_counterfactual_train_cd_tw.json"

# ===== Method selector =====
# LOSS_TYPE: tw_grpo | grpo | ngrpo | avspo
LOSS_TYPE="tw_grpo"

# SOVA settings apply only to tw_grpo.
# SOVA_MODE: positive | negative | bidirectional
#   positive      = all-correct positive branch only
#   negative      = current all-wrong negative branch only
#   bidirectional = positive + negative
# Presets (choose lambdas deliberately; these examples are not accuracy claims):
#   positive:      SOVA_MODE="positive";      lambda+ > 0; lambda- = 0
#   negative:      SOVA_MODE="negative";      lambda+ = 0; lambda- > 0
#   bidirectional: SOVA_MODE="bidirectional"; lambda+ > 0; lambda- > 0
SOVA_MODE="positive"
SOVA_LAMBDA_POSITIVE=0.0625
SOVA_LAMBDA_NEGATIVE=0.0

# Semantic anchors, not strength knobs. Keep fixed for the first comparison.
SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD=1.5
SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD=2.0
# ===== End selector =====

case "${LOSS_TYPE}" in
    tw_grpo)
        USE_SOVA=true
        REWARD_FUNCS=(accuracy format)
        ;;
    grpo|ngrpo|avspo)
        USE_SOVA=false
        REWARD_FUNCS=(origin_accuracy)
        ;;
    *)
        echo "LOSS_TYPE must be one of: tw_grpo, grpo, ngrpo, avspo" >&2
        exit 2
        ;;
esac

# GRPO and sampling
TW_ALPHA=1.70
KL_BETA=0.00
NUM_GENERATIONS=8
NGRPO_NUM_ITERATIONS=2
GENERATE_TEMPERATURE=1.0
MAX_PROMPT_LENGTH=4096
MAX_COMPLETION_LENGTH=4096

# Optimization and data
LEARNING_RATE=1e-6
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=1
TRAIN_SAMPLES=2000
DATA_SEED=42
MAX_GRAD_NORM=20
QUESTION_TYPE="mixed"

# Logging and checkpoints
LOGGING_STEPS=1
SAVE_STEPS=100
REPORT_TO="wandb"
SAVE_ONLY_MODEL=true

# Model execution
USE_BF16=true
TORCH_DTYPE="bfloat16"
GRADIENT_CHECKPOINTING=true
ATTN_IMPLEMENTATION="flash_attention_2"
FREEZE_VISION_MODULES=true

# Distributed training
NUM_PROCESSES=2
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12541}"

: "${CUDA_VISIBLE_DEVICES:?Run with CUDA_VISIBLE_DEVICES=GPU_ID,GPU_ID bash scripts/sova-tw-grpo.sh}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_IDS[@]} != NUM_PROCESSES )); then
    echo "CUDA_VISIBLE_DEVICES must contain exactly ${NUM_PROCESSES} GPU IDs." >&2
    exit 2
fi
if [[ ! "${GPU_IDS[0]}" =~ ^[0-9]+$ || ! "${GPU_IDS[1]}" =~ ^[0-9]+$ || "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain two distinct integer GPU IDs." >&2
    exit 2
fi
if [[ "${USE_SOVA}" == "true" ]] && (( NUM_GENERATIONS != 8 )); then
    echo "SOVA requires the frozen TW-GRPO NUM_GENERATIONS=8." >&2
    exit 2
fi
if [[ "${LOSS_TYPE}" == "ngrpo" ]] && ((
    NGRPO_NUM_ITERATIONS < 2 || GRADIENT_ACCUMULATION_STEPS != 1
)); then
    echo "NGRPO requires reuse >= 2 and gradient accumulation = 1." >&2
    exit 2
fi

for required_path in \
    "${PROJECT_ENV}/bin/python" \
    "${PROJECT_ENV}/bin/torchrun" \
    "${CUDA_TOOLKIT}/bin/nvcc" \
    "${MODEL_NAME_OR_PATH}/config.json" \
    "${TRAIN_DATA_JSON}" \
    "${PROJECT_ROOT}/scripts/zero3_offload.json" \
    "${PROJECT_ROOT}/src/open_r1/grpo.py" \
    "${PROJECT_ROOT}/src/open_r1/grpo_variants.py" \
    "${PROJECT_ROOT}/src/open_r1/sova.py" \
    "${PROJECT_ROOT}/src/open_r1/trainer/grpo_trainer.py"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path is missing: ${required_path}" >&2
        exit 3
    fi
done

if [[ "${USE_SOVA}" == "true" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src" "${PROJECT_ENV}/bin/python" - \
        "${SOVA_MODE}" \
        "${SOVA_LAMBDA_POSITIVE}" \
        "${SOVA_LAMBDA_NEGATIVE}" \
        "${SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD}" \
        "${SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD}" <<'PY'
import sys

from open_r1.sova import validate_sova_configuration

mode = sys.argv[1]
lambda_positive, lambda_negative, virtual_positive, virtual_negative = map(
    float, sys.argv[2:]
)
validate_sova_configuration(
    mode,
    lambda_positive,
    lambda_negative,
    virtual_positive,
    virtual_negative,
)
PY
fi

# Run identity and outputs
ALPHA_TAG="${TW_ALPHA//./p}"
LAMBDA_POSITIVE_TAG="${SOVA_LAMBDA_POSITIVE//./p}"
LAMBDA_NEGATIVE_TAG="${SOVA_LAMBDA_NEGATIVE//./p}"
VIRTUAL_POSITIVE_TAG="${SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD//./p}"
VIRTUAL_NEGATIVE_TAG="${SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD//./p}"
case "${LOSS_TYPE}" in
    tw_grpo)
        if [[ "${USE_SOVA}" == "true" ]]; then
            MODEL_RUN="Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_sovatwgrpo_${SOVA_MODE}_a${ALPHA_TAG}_lp${LAMBDA_POSITIVE_TAG}_ln${LAMBDA_NEGATIVE_TAG}_vp${VIRTUAL_POSITIVE_TAG}_vn${VIRTUAL_NEGATIVE_TAG}"
        else
            MODEL_RUN="Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_twgrpo_a1p70"
        fi
        ;;
    ngrpo)
        MODEL_RUN="Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_ngrpo_core_reuse${NGRPO_NUM_ITERATIONS}_drop_norefill_rmax1p0"
        ;;
    avspo)
        MODEL_RUN="Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_avspo_binary"
        ;;
    grpo)
        MODEL_RUN="Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_grpo_binary"
        ;;
esac
OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_RUN}"

mkdir -p "${OUTPUT_ROOT}"
if ! mkdir "${OUTPUT_DIR}"; then
    echo "Output directory already exists; choose a new configuration or archive it: ${OUTPUT_DIR}" >&2
    exit 4
fi

cd "${PROJECT_ROOT}"
export CUDA_HOME="${CUDA_TOOLKIT}"
export PATH="${PROJECT_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MAX_JOBS="${MAX_JOBS:-1}"
export PRIVATE_DATA_ROOT="${OUTPUT_ROOT}"
export WANDB_PROJECT="Qwen2.5-VL-7B-Video-GRPO"
export WANDB_NAME="${MODEL_RUN}"
export WANDB_MODE="offline"
export DEBUG_MODE="true"
export SAMPLE_MODE="true"
export LOG_PATH="${OUTPUT_DIR}/debug.log"

printf '%s\n' \
    "Run: ${MODEL_RUN}" \
    "Policy: loss_type=${LOSS_TYPE}, beta=${KL_BETA}, generations=${NUM_GENERATIONS}" \
    "Rewards: ${REWARD_FUNCS[*]}" \
    "Optimization: lr=${LEARNING_RATE}, batch=${PER_DEVICE_TRAIN_BATCH_SIZE}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}, epochs=${NUM_TRAIN_EPOCHS}"
if [[ "${USE_SOVA}" == "true" ]]; then
    printf '%s\n' \
        "SOVA: mode=${SOVA_MODE}, lambda_positive=${SOVA_LAMBDA_POSITIVE}, lambda_negative=${SOVA_LAMBDA_NEGATIVE}" \
        "SOVA anchors: positive=${SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD}, negative=${SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD}" \
        "TW-GRPO: alpha=${TW_ALPHA}"
elif [[ "${LOSS_TYPE}" == "ngrpo" ]]; then
    echo "NGRPO core: r_max=1.0, reuse=${NGRPO_NUM_ITERATIONS}, clip=[0.84,1.24], transactional all-correct drop, no refill"
elif [[ "${LOSS_TYPE}" == "avspo" ]]; then
    echo "AVSPO: tau0=0.5, alpha=0.5, eta=0.01, anchor=0.1"
fi

METHOD_ARGS=(--loss_type "${LOSS_TYPE}" --use_sova "${USE_SOVA}")
if [[ "${USE_SOVA}" == "true" ]]; then
    METHOD_ARGS+=(
        --sova_mode "${SOVA_MODE}"
        --sova_lambda_positive "${SOVA_LAMBDA_POSITIVE}"
        --sova_lambda_negative "${SOVA_LAMBDA_NEGATIVE}"
        --sova_positive_virtual_total_reward "${SOVA_POSITIVE_VIRTUAL_TOTAL_REWARD}"
        --sova_negative_virtual_total_reward "${SOVA_NEGATIVE_VIRTUAL_TOTAL_REWARD}"
    )
elif [[ "${LOSS_TYPE}" == "ngrpo" ]]; then
    METHOD_ARGS+=(--ngrpo_num_iterations "${NGRPO_NUM_ITERATIONS}")
fi

exec "${PROJECT_ENV}/bin/torchrun" \
    --nproc_per_node="${NUM_PROCESSES}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset_name xxx \
    --jsonl_path "${TRAIN_DATA_JSON}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --reward_funcs "${REWARD_FUNCS[@]}" \
    --learning_rate "${LEARNING_RATE}" \
    --beta "${KL_BETA}" \
    --alpha "${TW_ALPHA}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --question_type "${QUESTION_TYPE}" \
    --bf16 "${USE_BF16}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --data_seed "${DATA_SEED}" \
    --train_samples "${TRAIN_SAMPLES}" \
    --report_to "${REPORT_TO}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --freeze_vision_modules "${FREEZE_VISION_MODULES}" \
    "${METHOD_ARGS[@]}" \
    --generate_temperature "${GENERATE_TEMPERATURE}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --run_name "${MODEL_RUN}" \
    --save_steps "${SAVE_STEPS}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --save_only_model "${SAVE_ONLY_MODEL}" \
    --num_generations "${NUM_GENERATIONS}"
