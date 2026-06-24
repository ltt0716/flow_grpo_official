"""
paths.py —— 集中管理模型/输出路径,支持环境变量覆盖,避免写死在代码里。

优先级:专属环境变量 > FLOW_GRPO_MODELS_DIR 派生 > 默认值。
换机器只需在启动前 `export FLOW_GRPO_MODELS_DIR=/这台机器的模型目录`,无需改代码;
不设任何环境变量则用默认值(向后兼容,原来怎么跑还怎么跑)。

常用:
    export FLOW_GRPO_MODELS_DIR=/data/models      # 三个模型都在这个目录下
    export FLOW_GRPO_OUTPUT_DIR=/opt/nas/runs      # checkpoint/日志存持久盘
模型分散时可单独指定:
    export FLOW_GRPO_SD35_PATH=/a/sd35
    export FLOW_GRPO_PICKSCORE_PATH=/b/pickscore
    export FLOW_GRPO_CLIPH_PATH=/c/cliph
"""
import os

# 模型根目录(一个变量管住三个模型的常见情况)。默认 = 当前在用路径。
MODELS_DIR = os.environ.get("FLOW_GRPO_MODELS_DIR", "/opt/nas/p/longtao/models")

# 各模型路径:默认从 MODELS_DIR 派生;模型不在同一目录时用专属环境变量覆盖。
SD35_PATH = os.environ.get(
    "FLOW_GRPO_SD35_PATH", os.path.join(MODELS_DIR, "stable-diffusion-3.5-medium")
)
PICKSCORE_PATH = os.environ.get(
    "FLOW_GRPO_PICKSCORE_PATH", os.path.join(MODELS_DIR, "PickScore_v1")
)
CLIPH_PATH = os.environ.get(
    "FLOW_GRPO_CLIPH_PATH", os.path.join(MODELS_DIR, "CLIP-ViT-H-14-laion2B-s32B-b79K")
)

# 输出/checkpoint 根目录(指向持久盘可防 pod 迁移丢失)。默认沿用相对路径 "logs"。
OUTPUT_DIR = os.environ.get("FLOW_GRPO_OUTPUT_DIR", "logs")
