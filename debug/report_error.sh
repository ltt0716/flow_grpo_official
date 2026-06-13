#!/usr/bin/env bash
# report_error.sh —— 把训练报错 + 环境信息打包推到 git,供远程分析
# ============================================================================
# 工作流(在训练机上):
#
#   1) 跑训练时把全部输出存成日志(关键:用 tee 同时上屏 + 落盘):
#        mkdir -p debug
#        accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
#          --num_processes=4 --main_process_port 29501 \
#          scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu \
#          2>&1 | tee debug/train.log
#
#   2) 训练挂了 -> 运行本脚本,它会收集日志尾部 + 环境 + git 信息并 push:
#        bash debug/report_error.sh
#      可选指定别的日志文件:
#        bash debug/report_error.sh path/to/other.log
#
#   3) 在本地 `git pull`,看 debug/reports/<时间戳>/ 里的内容分析原因。
# ============================================================================
set -uo pipefail

# 切到仓库根,保证相对路径稳定
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || { echo "✗ 不在 git 仓库里"; exit 1; }

LOG="${1:-debug/train.log}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="debug/reports/$TS"
mkdir -p "$OUT"

echo "==> [1/5] 收集训练日志: $LOG"
if [[ -f "$LOG" ]]; then
    # 只取尾部 1500 行,避免把几百 MB 的日志塞进仓库
    tail -n 1500 "$LOG" > "$OUT/train_tail.txt"
    echo "    取了尾部 1500 行 -> $OUT/train_tail.txt"
else
    echo "⚠️  找不到日志 '$LOG'。你跑训练时加了 '2>&1 | tee debug/train.log' 吗?" \
        | tee "$OUT/train_tail.txt"
fi

echo "==> [2/5] 收集 git 状态"
{
    echo "### branch & commit"
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    echo; echo "### working tree 改动(你本地改了哪些文件)"
    git status -s
    echo; echo "### 相对上游/远程的差异 commit"
    git log --oneline -10
} > "$OUT/git_state.txt" 2>&1

echo "==> [3/5] 收集 Python / torch / CUDA 环境"
{
    echo "### date";        date
    echo; echo "### conda env"; echo "${CONDA_DEFAULT_ENV:-未知}"
    echo; echo "### python";  python -V 2>&1
    echo; echo "### torch / cuda"
    python - <<'PY' 2>&1
try:
    import torch
    print("torch       :", torch.__version__)
    print("cuda(build) :", torch.version.cuda)
    print("cuda avail  :", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
except Exception as e:
    print("torch import 失败:", repr(e))
PY
} > "$OUT/env.txt" 2>&1

echo "==> [4/5] nvidia-smi + 依赖清单"
nvidia-smi > "$OUT/nvidia_smi.txt" 2>&1 || echo "没有 nvidia-smi" > "$OUT/nvidia_smi.txt"
pip freeze   > "$OUT/pip_freeze.txt" 2>&1 || echo "pip freeze 失败" > "$OUT/pip_freeze.txt"

echo "==> [5/5] 提交并推送"
# -f 强制加:仓库 .gitignore 会拦 *.log,这里日志已存成 .txt,再加 -f 双保险
git add -f "$OUT"
if git commit -q -m "debug report $TS"; then
    if git push origin HEAD; then
        echo "✅ 已推送。在本地执行 'git pull' 后查看: $OUT/"
    else
        echo "✗ push 失败(多半是没配 git 凭据/token)。报告已 commit 在本地,"
        echo "  配好凭据后手动 'git push origin HEAD' 即可。"
    fi
else
    echo "（没有新内容可提交)"
fi
