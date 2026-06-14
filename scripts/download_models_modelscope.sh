#!/usr/bin/env bash
# download_models_modelscope.sh —— 从魔搭 ModelScope 下载 3 个模型到持久盘
# ============================================================================
# 适用:训练机连不上 HuggingFace/hf-mirror,但 modelscope.cn 通(国内集群常见)。
#
# ⚠️ 模型文件下到 /opt/nas(持久共享盘),不是 pod 本地、更不是 git。
#    pod 重启本地盘会清空,NAS 不会。
#
# 用法:
#   pip install modelscope
#   bash download_models_modelscope.sh
# ============================================================================
set -uo pipefail

# === 已经修改为你集群的专属持久盘路径 ===
MODELS_DIR="${MODELS_DIR:-/opt/nas/p/longtao/models}"
mkdir -p "$MODELS_DIR"
echo "==> 模型将下到: $MODELS_DIR"

dl () {  # dl <modelscope_id> <目标子目录>
    local mid="$1" sub="$2"
    echo ""
    echo "==> 下载 $mid  ->  $MODELS_DIR/$sub"
    modelscope download --model "$mid" --local_dir "$MODELS_DIR/$sub" \
        || echo "✗ $mid 下载失败(可能该 ID 在 modelscope 不存在,见脚本末尾兜底)"
}

# 1) SD3.5-medium —— 已确认在 modelscope(主模型,~20G)
dl "AI-ModelScope/stable-diffusion-3.5-medium"            "stable-diffusion-3.5-medium"
miasnmian
dl "AI-ModelScope/PickScore_v1"                            "PickScore_v1"

# 3) CLIP-ViT-H —— PickScore 的处理器。候选 ID,若 404 见兜底。
dl "AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K"        "CLIP-ViT-H-14-laion2B-s32B-b79K"

echo ""
echo "============================================================"
echo "下完后,把这些路径填进代码(见 RUNBOOK「指向本地模型」一节):"
echo "  SD3.5     : $MODELS_DIR/stable-diffusion-3.5-medium"
echo "  PickScore : $MODELS_DIR/PickScore_v1"
echo "  CLIP-H    : $MODELS_DIR/CLIP-ViT-H-14-laion2B-s32B-b79K"
echo ""
echo "若 PickScore / CLIP-H 在 modelscope 报 404(ID 不存在):"
echo "  - 去 https://modelscope.cn 搜 'PickScore' / 'CLIP-ViT-H-14' 看真实仓库名,改上面的 ID;"
echo "  - 或在能上 HF 的机器下好这两个(较小)再传到 $MODELS_DIR(方案 B)。"
echo "============================================================"