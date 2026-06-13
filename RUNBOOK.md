# 训练机部署 Runbook —— Flow-GRPO (SD3.5 + PickScore, 4×80G)

> 目标:在训练机上跑通 `pickscore_sd3_4gpu`,看到 reward 随 step 上升。
> 每步做完都有 **✅ 检查点**,过了再走下一步——卡住时能立刻定位。

环境基线:Ubuntu 20.04 · Python 3.12 · CUDA 12.1 · conda 环境 `lt-env` · 4×80G GPU

---

## 第 1 步:clone 你的 fork

```bash
cd ~                      # 或你想放代码的目录
git clone https://github.com/ltt0716/flow_grpo_official.git
cd flow_grpo_official
```

**✅ 检查点**
```bash
ls debug/report_error.sh && git log --oneline -2
```
能看到 `report_error.sh` 和提交 `953c97b` = clone 成功。

> ⚠️ clone 很慢/卡住 = 训练机连 GitHub 不畅,改用 gitee 镜像或打包传。

---

## 第 2 步:激活环境 + 装依赖

```bash
conda activate lt-env

# torch + xformers 从 cu121 源一起装(关键:防 xformers 拉新版顶掉 cu121 的 torch)
pip install torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post3 \
  --index-url https://download.pytorch.org/whl/cu121

# 装仓库本体
pip install -e .
```

**✅ 检查点**
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```
期望输出类似 `2.6.0 12.1 True 4`。`True`(CUDA 可用)和 `4`(看到 4 张卡)是关键。

> ⚠️ 某个包装不上(常见 `deepspeed` / `xformers`)→ 贴报错,别硬刚。
> `deepspeed` 在 4 卡 DDP(multi_gpu.yaml)其实用不到,装不上可先跳过。

---

## 第 3 步:登录 HuggingFace(SD3.5 是 gated 模型)

**先用浏览器**打开并点 "Agree" 同意协议:
`https://huggingface.co/stabilityai/stable-diffusion-3.5-medium`

再在训练机登录(token 从 https://hf.co/settings/tokens 生成):
```bash
huggingface-cli login
```

**✅ 检查点**
```bash
huggingface-cli whoami    # 显示你的用户名 = 登录成功
```

---

## 第 4 步:起飞前检查

```bash
nvidia-smi                # 确认 4 张卡空闲、各 80G
df -h ~                   # 确认家目录有 30–40G 下模型
export WANDB_MODE=offline # 第一次离线,免得卡在 wandb 登录
mkdir -p debug
```

> 磁盘不够 → `export HF_HOME=/数据盘/hf_cache` 把模型缓存挪到大盘。

---

## 第 5 步:启动训练(4 卡 PickScore)

```bash
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=4 --main_process_port 29501 \
  scripts/train_sd3.py --config config/grpo.py:pickscore_sd3_4gpu \
  2>&1 | tee debug/train.log
```

- **第一次会先下十几分钟模型**(SD3.5 + PickScore + CLIP-H),刷下载进度是正常的。
- **✅ 成功的样子**:下载完后屏幕开始刷训练日志,出现 `reward` / `kl` / `loss` 数字,
  且 `reward` 随 step 缓慢上升 = 整条链路通了。

---

## 出错了怎么办

一条命令把报错 + 环境打包推到 fork:
```bash
bash debug/report_error.sh
```
它收集日志尾部 + git 状态 + python/torch/cuda + nvidia-smi + pip freeze 到
`debug/reports/<时间戳>/`,自动 commit & push。然后通知对端 `git pull` 分析。

> push 失败(没配 token)→ 报告已 commit 在本地,配好凭据后 `git push origin HEAD` 即可。

---

## 关键决策备忘(为什么这么配)

- **用 `pickscore_sd3_4gpu` 不用 OCR**:OCR reward 要 `paddleocr`+`paddlepaddle`+`Levenshtein`
  (都不在 setup.py),坑多;PickScore 零额外依赖,只下 HF 模型。OCR 是官方默认示例,非必需。
- **`gpu_number` 必须 = `--num_processes`**:config 里的 batch 数学由 `gpu_number` 推导,
  改卡数就要换对应的 `_1gpu`/`_4gpu` config 或改 `gpu_number`。
- **80G 显存**:默认 batch=8@512 绰绰有余,**不用**开 activation_checkpointing、不用降 batch。
- **先离线 wandb**:第一次只为验证链路;确认能跑后再 `wandb login` 上线看曲线。

---

## 跑通之后(下一步)

- 切真实任务/上 wandb:去掉 `WANDB_MODE=offline`,`wandb login`。
- checkpoint 在 `logs/pickscore/sd3.5-M/`(已被 .gitignore,不会误推回仓库)。
- 想做"文字渲染"任务再装 paddleocr 用 OCR config;想移植手写版的 credit-assignment
  探索(early/late 逐步权重)到官方 `stat_tracking.py`,届时再说。
