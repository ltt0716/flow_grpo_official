#!/usr/bin/env bash
# preflight.sh —— 换机器后、启动训练前的检查。全绿再跑,避免排队半天才失败。
# 用法: bash debug/preflight.sh
echo "==================== 起飞前检查 ===================="

echo "## 1) 代码目录"
cd "$(git rev-parse --show-toplevel)" 2>/dev/null && echo "✅ $(pwd)" || { echo "❌ 不在 git 仓库(代码目录没了?需重新 clone)"; exit 1; }

echo ""
echo "## 2) 拉最新代码"
git pull --ff-only 2>&1 | tail -1
echo "当前提交: $(git log --oneline -1)"

echo ""
MODELS_DIR="${FLOW_GRPO_MODELS_DIR:-/opt/nas/p/longtao/models}"
echo "## 3) 模型路径(MODELS_DIR=$MODELS_DIR;可用 FLOW_GRPO_MODELS_DIR 覆盖)"
ALLOK=1
for d in stable-diffusion-3.5-medium PickScore_v1 CLIP-ViT-H-14-laion2B-s32B-b79K; do
    p="$MODELS_DIR/$d"
    if [ -d "$p" ] && [ -n "$(ls -A "$p" 2>/dev/null)" ]; then
        echo "✅ $p"
    else
        echo "❌ 缺失或为空: $p"; ALLOK=0
    fi
done

echo ""
echo "## 4) GPU / torch"
python -c "import torch; print('  CUDA可用:', torch.cuda.is_available(), ' 卡数:', torch.cuda.device_count())" 2>&1

echo ""
echo "===================================================="
if [ "$ALLOK" = "1" ]; then
    echo "✅ 模型齐全。若上面卡数=8,即可启动:"
    echo "   accelerate launch ... --num_processes=8 ... :pickscore_sd3_8gpu"
else
    echo "❌ 模型路径有问题 —— 新机器的 /opt/nas 挂载可能不同。"
    echo "   要么确认正确的模型路径告诉我改 config,要么重新下载模型。"
fi
