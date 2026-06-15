#!/usr/bin/env bash
# show_metrics.sh —— 不联网查看训练 reward
# ============================================================================
# 两个来源:
#   1) debug/metrics.jsonl —— 新代码(train_sd3.py)每 epoch 落的纯文本曲线
#      (只对「加了该改动后启动的 run」有;老 run 没有)
#   2) wandb 离线快照 wandb-summary.json —— 任何 run 都有,但只含【最新】一组值
#
# 用法: bash debug/show_metrics.sh
# ============================================================================
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || true

echo "==================== 1) metrics.jsonl(完整曲线)===================="
if [[ -f debug/metrics.jsonl ]]; then
    echo "共 $(wc -l < debug/metrics.jsonl) 行,最近 15 条:"
    tail -n 15 debug/metrics.jsonl
else
    echo "(无 debug/metrics.jsonl —— 当前 run 是加该日志前启动的,看下面 wandb 快照)"
fi

echo ""
echo "==================== 2) wandb 离线最新快照 ===================="
# 找最新的 offline-run 目录(latest-run 软链可能不存在)
SUMMARY=$(ls -t wandb/*/files/wandb-summary.json 2>/dev/null | head -1)
if [[ -n "${SUMMARY:-}" && -f "$SUMMARY" ]]; then
    echo "来自: $SUMMARY"
    # 只挑 reward / kl 相关键打印(没有 jq 就原样 cat)
    if command -v jq >/dev/null 2>&1; then
        jq 'to_entries | map(select(.key|test("reward|kl|epoch";"i"))) | from_entries' "$SUMMARY"
    else
        cat "$SUMMARY"
    fi
else
    echo "(没找到 wandb 离线快照,确认训练在用 wandb、且没改 WANDB_DIR)"
fi
