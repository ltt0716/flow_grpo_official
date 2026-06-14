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

## 第 3.5 步:解决「连不上 HuggingFace」(国内集群必看)

国内训练机/k8s pod 常常出网受限,直连 huggingface.co 会报
`OSError: [Errno 101] Network is unreachable`。**走 HF 国内镜像(方案 A)。**

**先测连通性**,决定能不能走镜像:
```bash
curl -sI --connect-timeout 5 https://hf-mirror.com | head -1   # 国内镜像
curl -sI --connect-timeout 5 https://huggingface.co | head -1  # 官方
```
- 镜像返回 `HTTP/.. 200/301` → 走下面方案 A
- **两条都超时**(连镜像也不通)→ 先按下面「网络受限排查」找模型来源,别盲跑

### 网络受限排查(连 hf-mirror 都超时时,按序找到一个能用的就停)

> 典型情况:pod 能连 GitHub(clone/push 正常),但 HF / 镜像都被墙。
> 说明是选择性放行,模型多半要走集群自己的渠道,而不是从 pod 直接下。

```bash
# ① 模型是否已预置在共享存储/NAS(最省事,集群常预置热门模型)
ls /opt/nas 2>/dev/null
find /opt/nas /opt/project -iname "*stable-diffusion-3*" 2>/dev/null | head
find / -iname "*PickScore*" 2>/dev/null | head

# ② 是否有现成 http 代理(企业集群常配代理走外网)
env | grep -i proxy
cat /etc/environment 2>/dev/null | grep -i proxy

# ③ 到底能连到哪些外网
curl -sI --connect-timeout 5 https://www.modelscope.cn | head -1  # 阿里魔搭,国内集群常通
curl -sI --connect-timeout 5 https://github.com        | head -1  # 已知通
```

按结果选路:
- **①找到模型** → 改 `config/grpo.py` 里 `config.pretrained.model` 指向该路径,直接跑。
- **②有代理** → `export http_proxy=... https_proxy=...`,回方案 A。
- **③只有 modelscope 通** → 从魔搭下 SD3.5(国内无需翻墙,见方案 C)。
- **全不通、只有 GitHub** → 方案 B:本地下好传到 `/opt/nas`(持久存储)。

### 方案 A:走 HF 镜像 + 预下载模型
```bash
export HF_ENDPOINT=https://hf-mirror.com   # 关键:指向国内镜像
export HF_TOKEN=<你的HF token>             # SD3.5 是 gated,要带 token

# 单进程先把 3 个模型下全(别让 4 个 rank 同时抢下载)
huggingface-cli download stabilityai/stable-diffusion-3.5-medium
huggingface-cli download yuvalkirstain/PickScore_v1
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

> ⚠️ `HF_ENDPOINT` 必须在**启动训练的同一个 shell** 里 export,否则训练进程仍去连官方。
> 建议把这两行 export 也写进 `~/.bashrc`,一劳永逸。

---

## 第 4 步:起飞前检查

```bash
nvidia-smi                # 确认 4 张卡空闲、各 80G
df -h ~                   # 确认家目录有 30–40G 下模型
export HF_ENDPOINT=https://hf-mirror.com   # 同上,确保训练进程也走镜像
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

---

## 附:方案 B —— 机器完全不出网(连镜像也不通)

在**有网的机器**(本地电脑 / 能上 HF 的跳板)下好模型,再传到训练机:

```bash
# ① 有网机器上下载(用镜像加速)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir sd35
huggingface-cli download yuvalkirstain/PickScore_v1 --local-dir pickscore
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir cliph

# ② 打包传到训练机(SD3.5 约 20G,耐心等)
scp -r sd35 pickscore cliph 用户@训练机:/数据盘/models/
```

训练机上有两种用法:
- **填进 HF 缓存**:把文件放到 `$HF_HOME/hub/` 对应目录,代码按原仓库名就能从缓存读;
- **改成本地路径**(更简单):在 `config/grpo.py` 的 `pickscore_sd3_4gpu` 里把
  `config.pretrained.model` 改成 `/数据盘/models/sd35`,reward 同理改 scorer 里的路径。

> 方案 B 较繁琐,仅在「连 hf-mirror.com 都不通」时才用。优先试方案 A。

---

## 附:方案 C —— 从魔搭 ModelScope 下载(国内集群,modelscope 通时)

国内 Tencent/阿里集群常常 HF 全墙、但 `modelscope.cn` 通。SD3.5 在魔搭有镜像。

```bash
pip install modelscope
# 在训练机上直接下到本地目录(无需翻墙、无需 HF token)
python -c "from modelscope import snapshot_download; \
  snapshot_download('AI-ModelScope/stable-diffusion-3.5-medium', local_dir='/opt/nas/models/sd35')"
```

然后把 `config/grpo.py` 里 `config.pretrained.model` 改成 `/opt/nas/models/sd35` 即可。

> PickScore / CLIP-H 魔搭也多有镜像(搜对应名),或走方案 B 单独传这两个小模型。
> 存到 `/opt/nas`(持久盘)而非 pod 本地,pod 重启不会丢。

## 模型路径
/opt/nas/p/longtao/models/stable-diffusion-3.5-medium
/opt/nas/p/longtao/models/PickScore_v1
/opt/nas/p/longtao/models/CLIP-ViT-H-14-laion2B-s32B-b79K