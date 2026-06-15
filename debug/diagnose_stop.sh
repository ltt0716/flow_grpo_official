#!/usr/bin/env bash
# diagnose_stop.sh —— 训练异常停止后的"事后验尸",打包推到 git 供分析
# ============================================================================
# 训练莫名停了 -> 在训练机跑本脚本 -> 它收集"为什么停 + 死前学得怎样" -> 推到 git
# -> 通知对端 git pull 读 debug/reports/stopdiag_<时间戳>/
#
# 用法:  bash debug/diagnose_stop.sh
# ============================================================================
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || { echo "✗ 不在 git 仓库"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
OUT="debug/reports/stopdiag_$TS"
mkdir -p "$OUT"

echo "==> [1/6] 训练日志结尾(断在哪)"
if [[ -f debug/train.log ]]; then
    tail -n 200 debug/train.log > "$OUT/train_tail.txt"
else
    echo "(无 debug/train.log)" > "$OUT/train_tail.txt"
fi

echo "==> [2/6] 是否被 OOM-kill(主机内存)"
{ dmesg -T 2>/dev/null | grep -iE "killed process|out of memory|oom-kill" | tail -30; } > "$OUT/dmesg_oom.txt" 2>&1 || true
[[ -s "$OUT/dmesg_oom.txt" ]] || echo "(dmesg 无 OOM 记录,或无权限读 dmesg)" > "$OUT/dmesg_oom.txt"

echo "==> [3/6] wandb 最新快照(死前 reward 涨没涨)"
SUMMARY=$(ls -t wandb/*/files/wandb-summary.json 2>/dev/null | head -1)
if [[ -n "${SUMMARY:-}" && -f "$SUMMARY" ]]; then
    cp "$SUMMARY" "$OUT/wandb_summary.json"
else
    echo "(没找到 wandb summary)" > "$OUT/wandb_summary.json"
fi

echo "==> [4/6] 完整 reward 曲线(若有 metrics.jsonl)"
[[ -f debug/metrics.jsonl ]] && cp debug/metrics.jsonl "$OUT/metrics.jsonl" || echo "(无 metrics.jsonl,旧 run 没这日志)" > "$OUT/metrics.jsonl"

echo "==> [5/6] 存了哪些 checkpoint"
ls -laR logs 2>/dev/null | head -80 > "$OUT/checkpoints.txt" 2>&1 || echo "(无 logs/ 目录)" > "$OUT/checkpoints.txt"

echo "==> [6/6] 主机 / 资源快照"
{
    echo "### hostname"; hostname
    echo; echo "### uptime(看是否刚重启过=pod 被回收)"; uptime
    echo; echo "### 内存"; free -h
    echo; echo "### 磁盘"; df -h
    echo; echo "### GPU"; nvidia-smi 2>&1
} > "$OUT/host.txt" 2>&1

echo "==> 提交并推送"
git add -f "$OUT"
if git commit -q -m "stop diagnosis $TS"; then
    if git push origin HEAD; then
        echo "✅ 已推送,通知对端 git pull 看: $OUT/"
    else
        echo "✗ push 失败(没配 token?),已 commit 在本地,补 push: git push origin HEAD"
    fi
else
    echo "(没有内容可提交)"
fi
