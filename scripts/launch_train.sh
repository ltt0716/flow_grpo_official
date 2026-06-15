#!/usr/bin/env bash
# launch_train.sh —— 启动 4 卡 PickScore 训练(nohup 后台 + 离线环境)
# ============================================================================
# 用法(先 pull 再跑):
#   git pull && bash scripts/launch_train.sh
#
# 特点:
#   - nohup 后台:终端/SSH 断开也不会杀进程(上次"突然停止"多半是前台被 SIGHUP)
#   - 全离线环境变量:模型已在本地,禁止任何联网,避免 HF 超时
#   - 看日志:   tail -f debug/train.log
#   - 看 reward: tail -f debug/train.log | grep metrics
#   - 停止训练: kill $(cat debug/train.pid)
# 注意:本脚本不在内部 git pull(避免运行中自改),请在外面先 pull。
# ============================================================================
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

# 配置名作为第 1 个参数,默认全新训练;续训传 pickscore_sd3_4gpu_resume
CONFIG_NAME="${1:-pickscore_sd3_4gpu}"
echo "==> 使用配置: $CONFIG_NAME"

export WANDB_MODE=offline            # 离线记录(无外网)
export TOKENIZERS_PARALLELISM=false  # 消掉 tokenizer fork 警告刷屏
export HF_HUB_OFFLINE=1              # 模型全在本地,禁止联网取元数据
export TRANSFORMERS_OFFLINE=1        # 同上,避免再撞 HF 超时
mkdir -p debug

nohup accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=4 --main_process_port 29501 \
  scripts/train_sd3.py --config "config/grpo.py:$CONFIG_NAME" \
  > debug/train.log 2>&1 &

PID=$!
echo "$PID" > debug/train.pid
echo "✅ 训练已后台启动,PID=$PID (存于 debug/train.pid)"
echo "   实时日志: tail -f debug/train.log"
echo "   看 reward: tail -f debug/train.log | grep metrics"
echo "   停止训练: kill $PID    # 或 kill \$(cat debug/train.pid)"
