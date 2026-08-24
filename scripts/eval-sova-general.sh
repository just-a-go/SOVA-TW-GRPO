#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data0/codefile/wangweiqi/SOVA-TW-GRPO"
PROJECT_ENV="/data0/codefile/wangweiqi/EF-TW-GRPO/.conda/envs/ef-tw-grpo"
CUDA_TOOLKIT="/data0/codefile/wangweiqi/EF-TW-GRPO/.cuda-12.4"
SOURCE_EVAL_ROOT="/data0/codefile/wangweiqi/SCRA-TW-GRPO/data/evaluation"
TARGET_EVAL_ROOT="${PROJECT_ROOT}/data/evaluation"
VIDEO_ROOT="/data0/data/CD-TW-GRPO"
MVBench_FALLBACK_ROOT="/data0/data/VideoMind-data/mvbench"
MVBENCH_TVQA_FRAME_ROOT="${MVBench_FALLBACK_ROOT}/video/tvqa/frames_fps3_hq"
MVBENCH_TVQA_VIDEO_ROOT="${VIDEO_ROOT}/MVBench/tvqa/frames_fps3"

MODEL_RUN="Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_sovatwgrpo_bidirectional_a1p70_lp0p0625_ln0p03125_vp1p5_vn2p0"
CHECKPOINT_STEP=1000
CHECKPOINT="${PROJECT_ROOT}/outputs/${MODEL_RUN}/checkpoint-${CHECKPOINT_STEP}"
RUN_ROOT="${PROJECT_ROOT}/outputs/${MODEL_RUN}"
LOG_ROOT="${RUN_ROOT}/evaluation/general"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

usage() {
    cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=GPU_ID bash scripts/eval-sova-general.sh [selection]

Selections:
  ready        Run NExTQA, MMVU, TempCompass, and VideoMME (default)
  all          Run the five baseline datasets, including MVBench
  nextgqa      Run only NExTQA
  mmvu         Run only MMVU
  mvbench      Run only MVBench
  tempcompass  Run only TempCompass
  videomme     Run only VideoMME

Completed datasets are verified and skipped. Incomplete output files are never
silently resumed because the current evaluator's temporary file is JSONL.
EVAL_BATCH_SIZE defaults to 1 and may be set explicitly for a matched protocol.
EOF
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi

selection="${1:-ready}"
case "${selection}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    ready)
        datasets=(eval_nextgqa eval_mmvu eval_tempcompass eval_videomme)
        ;;
    all)
        datasets=(eval_nextgqa eval_mmvu eval_mvbench eval_tempcompass eval_videomme)
        ;;
    nextgqa|eval_nextgqa)
        datasets=(eval_nextgqa)
        ;;
    mmvu|eval_mmvu)
        datasets=(eval_mmvu)
        ;;
    mvbench|eval_mvbench)
        datasets=(eval_mvbench)
        ;;
    tempcompass|eval_tempcompass)
        datasets=(eval_tempcompass)
        ;;
    videomme|eval_videomme)
        datasets=(eval_videomme)
        ;;
    *)
        echo "Unknown selection: ${selection}" >&2
        usage >&2
        exit 2
        ;;
esac

: "${CUDA_VISIBLE_DEVICES:?Specify one GPU, for example: CUDA_VISIBLE_DEVICES=6 bash scripts/eval-sova-general.sh}"
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly one non-negative GPU ID." >&2
    exit 2
fi
if [[ ! "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_BATCH_SIZE must be a positive integer." >&2
    exit 2
fi

PYTHON="${PROJECT_ENV}/bin/python"
for required_path in \
    "${PYTHON}" \
    "${CUDA_TOOLKIT}/bin/nvcc" \
    "${CHECKPOINT}/config.json" \
    "${CHECKPOINT}/model.safetensors.index.json" \
    "${CHECKPOINT}/preprocessor_config.json" \
    "${PROJECT_ROOT}/scripts/prepare_general_eval_data.py" \
    "${PROJECT_ROOT}/scripts/prepare_mvbench_tvqa_videos.py" \
    "${PROJECT_ROOT}/src/eval/eval_general_videor1.py"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path is missing: ${required_path}" >&2
        exit 3
    fi
done

mkdir -p "${TARGET_EVAL_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

if ! command -v flock >/dev/null 2>&1; then
    echo "Required command is missing: flock" >&2
    exit 3
fi
exec 9>"${RUN_ROOT}/.eval-sova-general.lock"
if ! flock -n 9; then
    echo "Another general evaluation wrapper is already running for this model." >&2
    exit 7
fi

export CUDA_HOME="${CUDA_TOOLKIT}"
export PATH="${CUDA_HOME}/bin:${PROJECT_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VIDEO_MAX_PIXELS=$((256 * 28 * 28))
export FPS_MAX_FRAMES=16
export SAMPLE_MODE=true

prepare_mvbench=false
for dataset in "${datasets[@]}"; do
    if [[ "${dataset}" == eval_mvbench ]]; then
        prepare_mvbench=true
        break
    fi
done
if [[ "${prepare_mvbench}" == true ]]; then
    "${PYTHON}" scripts/prepare_mvbench_tvqa_videos.py \
        --annotations "${SOURCE_EVAL_ROOT}/eval_mvbench.json" \
        --frame-root "${MVBENCH_TVQA_FRAME_ROOT}" \
        --video-root "${MVBENCH_TVQA_VIDEO_ROOT}" \
        --fps 3 \
        --preset ultrafast
fi

prepare_args=(
    --source-root "${SOURCE_EVAL_ROOT}"
    --target-root "${TARGET_EVAL_ROOT}"
    --video-root "${VIDEO_ROOT}"
)
if [[ -d "${MVBench_FALLBACK_ROOT}" ]]; then
    prepare_args+=(--fallback-root "${MVBench_FALLBACK_ROOT}")
fi
"${PYTHON}" scripts/prepare_general_eval_data.py "${prepare_args[@]}" "${datasets[@]}"

expected_samples() {
    case "$1" in
        eval_nextgqa) echo 5553 ;;
        eval_mmvu) echo 625 ;;
        eval_mvbench) echo 4000 ;;
        eval_tempcompass) echo 7540 ;;
        eval_videomme) echo 2700 ;;
        *) return 2 ;;
    esac
}

set_output_paths() {
    local dataset="$1"
    output_file="${RUN_ROOT}/checkpoint-${CHECKPOINT_STEP}_${dataset}_eval04251.json"
    temp_file="${RUN_ROOT}/checkpoint-${CHECKPOINT_STEP}_${dataset}_eval04251_temp.json"
    metrics_file="${RUN_ROOT}/checkpoint-${CHECKPOINT_STEP}_${dataset}_eval04251_temp_metrics.json"
}

output_status() {
    local expected="$1"
    if [[ ! -e "${output_file}" && ! -e "${temp_file}" && ! -e "${metrics_file}" ]]; then
        return 1
    fi
    if [[ ! -f "${output_file}" || -e "${temp_file}" || ! -f "${metrics_file}" ]]; then
        return 2
    fi

    if "${PYTHON}" - "${output_file}" "${metrics_file}" "${expected}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
expected = int(sys.argv[3])

try:
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    result_count = 0
    results_are_objects = True
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            result_count += 1
            results_are_objects &= isinstance(json.loads(line), dict)
except (OSError, json.JSONDecodeError, TypeError):
    raise SystemExit(2)

if (
    isinstance(metrics, dict)
    and metrics.get("completed_samples") == expected
    and metrics.get("total_samples") == expected
    and result_count == expected
    and results_are_objects
):
    raise SystemExit(0)
raise SystemExit(2)
PY
    then
        return 0
    fi
    return 2
}

refuse_incomplete_output() {
    local dataset="$1"
    echo "Refusing to run: incomplete or inconsistent output exists for ${dataset}." >&2
    echo "Inspect these files before moving them aside and rerunning:" >&2
    echo "  ${output_file}" >&2
    echo "  ${temp_file}" >&2
    echo "  ${metrics_file}" >&2
    exit 5
}

for dataset in "${datasets[@]}"; do
    expected="$(expected_samples "${dataset}")"
    set_output_paths "${dataset}"
    if output_status "${expected}"; then
        continue
    else
        status=$?
    fi
    if (( status == 2 )); then
        refuse_incomplete_output "${dataset}"
    fi
done

ran_any=false
for dataset in "${datasets[@]}"; do
    expected="$(expected_samples "${dataset}")"
    set_output_paths "${dataset}"
    if output_status "${expected}"; then
        echo "Skipping completed dataset: ${dataset} (${expected}/${expected})"
        continue
    else
        status=$?
    fi
    if (( status == 2 )); then
        refuse_incomplete_output "${dataset}"
    fi

    ran_any=true
    log_file="${LOG_ROOT}/${dataset}.log"
    echo "Starting ${dataset} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with batch_size=${EVAL_BATCH_SIZE}"
    "${PYTHON}" src/eval/eval_general_videor1.py \
        --model_name "${CHECKPOINT}" \
        --dataset_name "${dataset}" \
        --batch_size "${EVAL_BATCH_SIZE}" 2>&1 | tee "${log_file}"

    if ! output_status "${expected}"; then
        echo "Evaluation ended without a complete ${expected}-sample result: ${dataset}" >&2
        exit 6
    fi
    echo "Completed ${dataset}: ${metrics_file}"
done

if [[ "${ran_any}" == false ]]; then
    echo "All selected datasets were already complete."
else
    echo "All selected evaluations completed successfully."
fi
