#!/usr/bin/env bash
# Remove resume-related files from a training output directory, keeping every
# checkpoint weight and evaluation result.
#
# Usage:
#   bash clean_resume_artifacts.sh <output-dir>            # dry run, prints only
#   DRYRUN=0 bash clean_resume_artifacts.sh <output-dir>   # actually delete

set -uo pipefail
shopt -s nullglob

ROOT="${1:-}"
DRYRUN="${DRYRUN:-1}"

if [ -z "$ROOT" ]; then
    echo "Usage: bash $(basename "$0") <output-dir>" >&2
    echo "  e.g. bash $(basename "$0") outputs/Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_sovatwgrpo" >&2
    exit 2
fi

[ -d "$ROOT" ] || { echo "No such directory: $ROOT" >&2; exit 1; }
cd "$ROOT" || exit 1

echo "Directory : $(pwd)"
echo "Current   : $(du -sh . 2>/dev/null | cut -f1)"
echo "Mode      : $([ "$DRYRUN" = "0" ] && echo 'delete' || echo 'dry run (DRYRUN=1)')"
echo

# ---------------------------------------------------------------- 1. safety check
# Without stage3_gather_16bit_weights_on_model_save, ZeRO-3 may keep the weights
# only inside global_step*/ shards. Deleting global_step* from such a checkpoint
# would delete the model itself.
echo "=== Checking which checkpoints already hold consolidated weights ==="
SAFE_DIRS=()
UNSAFE=0
for d in checkpoint-*/; do
    n=$(ls "$d"model-*.safetensors "$d"pytorch_model*.bin 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "  [skip]  $d  no consolidated weights, keeping its global_step*"
        UNSAFE=1
    else
        echo "  [clean] $d  $n consolidated shard(s)"
        SAFE_DIRS+=("$d")
    fi
done
[ ${#SAFE_DIRS[@]} -eq 0 ] && { echo; echo "Nothing to clean, exiting."; exit 0; }
echo

# ---------------------------------------------------------------- 2. collect targets
TARGETS=()
collect() {
    local d="$1"
    while IFS= read -r -d '' x; do TARGETS+=("$x"); done < <(
        find "$d" -maxdepth 1 -type d -name 'global_step*' -print0 2>/dev/null)
    while IFS= read -r -d '' x; do TARGETS+=("$x"); done < <(
        find "$d" -maxdepth 1 -type f \( \
            -name 'optimizer.pt'  -o -name 'optimizer.bin' -o \
            -name 'scheduler.pt'  -o -name 'rng_state*.pth' -o \
            -name 'latest'        -o -name 'zero_pp_rank_*optim_states.pt' -o \
            -name 'mp_rank_*model_states.pt' \) -print0 2>/dev/null)
}
for d in "${SAFE_DIRS[@]}"; do collect "$d"; done
collect "."   # the output root itself may hold some too

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "No resume-related files found; this directory may already be clean."
    exit 0
fi

# ---------------------------------------------------------------- 3. list
echo "=== Would be deleted ==="
for t in "${TARGETS[@]}"; do
    printf '  %-10s %s\n' "$(du -sh "$t" 2>/dev/null | cut -f1)" "$t"
done
echo
echo "Total reclaimable: $(du -shc "${TARGETS[@]}" 2>/dev/null | tail -1 | cut -f1)"
echo

# ---------------------------------------------------------------- 4. execute
if [ "$DRYRUN" != "0" ]; then
    echo "Dry run finished, nothing deleted. To delete for real:"
    echo "    DRYRUN=0 bash $(basename "$0") $ROOT"
    exit 0
fi

read -r -p "Delete everything listed above? Type yes to continue: " ans
[ "$ans" = "yes" ] || { echo "Cancelled."; exit 1; }

for t in "${TARGETS[@]}"; do rm -rf -- "$t"; done

echo
echo "=== Done ==="
echo "Size after cleanup: $(du -sh . 2>/dev/null | cut -f1)"
[ "$UNSAFE" = "1" ] && echo "Note: some checkpoints were skipped for lacking consolidated weights, see the [skip] lines above."
echo
for d in checkpoint-*/; do echo "--- $d"; ls "$d"; done
