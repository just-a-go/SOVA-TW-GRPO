#!/usr/bin/env bash
# 清理 SOVA-TW-GRPO 输出目录中与「续训」相关的文件，保留全部 checkpoint 权重与评测结果。
#
# 用法:
#   bash clean_resume_artifacts.sh          # 预演，只打印不删除
#   DRYRUN=0 bash clean_resume_artifacts.sh # 真正删除

set -uo pipefail
shopt -s nullglob

ROOT="/data0/codefile/wangweiqi/SOVA-TW-GRPO/outputs/Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_sovatwgrpo_bidirectional_a1p70_lp0p125_ln0p03125_vp1p5_vn2p0"
DRYRUN="${DRYRUN:-1}"

[ -d "$ROOT" ] || { echo "目录不存在: $ROOT"; exit 1; }
cd "$ROOT" || exit 1

echo "工作目录 : $(pwd)"
echo "当前占用 : $(du -sh . 2>/dev/null | cut -f1)"
echo "模式     : $([ "$DRYRUN" = "0" ] && echo '实际删除' || echo '预演 (DRYRUN=1)')"
echo

# ---------------------------------------------------------------- 1. 安全检查
# ZeRO-3 若未开启 stage3_gather_16bit_weights_on_model_save，权重可能只存在于
# global_step*/ 分片中。那种 checkpoint 删掉 global_step* 等于删掉模型。
echo "=== 检查各 checkpoint 是否已有合并权重 ==="
SAFE_DIRS=()
UNSAFE=0
for d in checkpoint-*/; do
    n=$(ls "$d"model-*.safetensors "$d"pytorch_model*.bin 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "  [跳过] $d  没有合并权重，保留其 global_step*"
        UNSAFE=1
    else
        echo "  [可清] $d  合并权重 $n 片"
        SAFE_DIRS+=("$d")
    fi
done
[ ${#SAFE_DIRS[@]} -eq 0 ] && { echo; echo "没有可清理的 checkpoint，退出。"; exit 0; }
echo

# ---------------------------------------------------------------- 2. 收集目标
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
collect "."   # 输出根目录本身也可能有

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "没有找到续训相关文件，可能已经清理过了。"
    exit 0
fi

# ---------------------------------------------------------------- 3. 列出清单
echo "=== 将被删除的内容 ==="
for t in "${TARGETS[@]}"; do
    printf '  %-10s %s\n' "$(du -sh "$t" 2>/dev/null | cut -f1)" "$t"
done
echo
echo "合计可释放: $(du -shc "${TARGETS[@]}" 2>/dev/null | tail -1 | cut -f1)"
echo

# ---------------------------------------------------------------- 4. 执行
if [ "$DRYRUN" != "0" ]; then
    echo "预演结束，未删除任何文件。确认无误后执行:"
    echo "    DRYRUN=0 bash $(basename "$0")"
    exit 0
fi

read -r -p "确认删除以上内容？输入 yes 继续: " ans
[ "$ans" = "yes" ] || { echo "已取消。"; exit 1; }

for t in "${TARGETS[@]}"; do rm -rf -- "$t"; done

echo
echo "=== 完成 ==="
echo "清理后占用: $(du -sh . 2>/dev/null | cut -f1)"
[ "$UNSAFE" = "1" ] && echo "注意: 有 checkpoint 因缺少合并权重被跳过，见上方 [跳过] 行。"
echo
for d in checkpoint-*/; do echo "--- $d"; ls "$d"; done
