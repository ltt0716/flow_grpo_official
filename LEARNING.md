# Flow-GRPO 学习计划 + 官方代码导读地图

> 目标:吃透"用 RL(GRPO)后训练 Flow Matching 文生图(SD3.5)"。
> 学习主线:**用干净的手写版理解算法 → 在官方版看工程现实 → 论文补理论**。
> 两个仓库:`flow_grpo_official`(在跑的官方版)/ `flow-grpo`(你自己手写的 600 行简化版,学算法用)。

---

## 一句话心智图

Flow Matching 采样是确定性 ODE,没随机性做不了 RL → 把每步 ODE 转移**随机化成高斯转移**(ODE→SDE)
→ 每步成为可算 log-prob 的动作 → 同一 prompt 采 K 条轨迹,**组内相对优势**替代 critic(GRPO)
→ clipped surrogate + KL 更新 LoRA。

---

## Part A:六阶段学习计划

### 阶段 0:全局心智图(半天)
- 读:手写版 `flow-grpo/README.md`
- 目标:能用上面那一段话讲清"为什么这么做"

### 阶段 1:Flow Matching 与 ODE→SDE 核心 ⭐最重要(1–2 天)
- 读:手写 `flow-grpo/flow_grpo/log_prob.py`、`sampling.py`
- 跑:`pytest tests/test_log_prob.py`,改数值看 log-prob 变化
- 对照:官方 `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`(推导更严格)
- **要能回答**:
  - 为什么确定性 ODE 不能做 RL?随机化加在哪一步?
  - `std = eta·√|dt|` 的 eta 是什么旋钮?太大/太小各会怎样?
  - 为什么 log-prob 要对所有像素维度求和?(float32 那个坑就在这)

### 阶段 2:GRPO 算法本身(1 天)
- 读:手写 `advantage.py`、`loss.py`;对照官方 `stat_tracking.py` + `train_sd3.py` loss 部分
- **要能回答**:
  - GRPO 为什么不需要 value network?组内标准化怎么替代 baseline?
  - clipped surrogate 防什么?k3 KL 约束什么?
  - `beta`(KL 系数)调大调小的影响?

### 阶段 3:完整链路与数据流(1 天)
- 读:官方 `train_sd3.py` 主循环 + `DistributedKRepeatSampler`
- 看:正在跑的训练 `tail -f debug/train.log`,把日志阶段和代码对上
- **要能回答**:
  - 一个 step 从 prompt 到 LoRA 更新经过哪些阶段?
  - "卡数×batch ÷ group = prompt 数"为什么要整除?
  - 为什么 reward 要跨卡 gather?

### 阶段 4:Reward 与"过拟合陷阱"(半天)
- 读:手写版 README「指标可信度」+ `rewards.py`
- 核心洞见:**训练 reward 必涨不算成果,OOD/eval reward 涨才算**(本次 eval 0.835→0.865 即活例)
- **要能回答**:为什么不能只看训练 reward?over-optimization 怎么检验?

### 阶段 5:研究贡献 —— credit assignment(持续,你的卖点)
- 读:手写 `advantage.py` 的 `expand_advantage_over_steps`(trajectory/late/early)
- 跑通基线后,移植进官方 `stat_tracking.py` 做对比实验
- **要能回答**:为什么 credit assignment 在多步生成里是真问题?early/late 你预期谁更好、为什么?

### 阶段 6:论文与脉络(1–2 天)
- 读:[Flow-GRPO 论文](https://arxiv.org/abs/2505.05470)、DDPO、DeepSeekMath/GRPO
- **要能回答**:Flow-GRPO 和 DDPO 的关系?哪些是模态无关、可从 LLM 迁移的?

---

## Part B:官方 `scripts/train_sd3.py` 导读地图(975 行,算法约 110 行)

### 🟢 核心算法(只读这 ~110 行,5 块)

| 块 | 行号 | 干什么 | 手写版对照 |
|---|---|---|---|
| ① log-prob 核心 ⭐⭐⭐ | **181–215** `compute_log_prob` | transformer 前向→SDE 一步 mean/std→算 log-prob(**OOM 就在这**) | `log_prob.py` + `sampling.py` 前向 |
| ② 采样 rollout ⭐⭐ | **643** `pipeline_with_logprob`、**668** `reward_fn` | 生成图、逐步记 log_prob、打分 | `sampling.py` `sample_trajectory` |
| ③ 组内优势 ⭐⭐ | **774** `stat_tracker.update`、**745** adv 沿时间维 repeat | reward→组内标准化→展开到每步 | `advantage.py` 两个函数 |
| ④ PPO 更新 ⭐⭐⭐ | **899** ratio、**900–906** clipped surrogate、**908–910** kl+loss | `ratio=exp(logp_new−logp_old)`,min(裁剪),加 KL | `loss.py` `grpo_loss` |
| ⑤ 反传 ⭐ | **946–952** | backward+clip_grad+step | `trainer.py` |

**精读顺序**:① → ④ → ②/③(① 和 ④ 是灵魂)。

### ⚪ 工程外壳(先跳过)
- 42–77 数据集类 / 124–131 文本编码 / 133–167 zero_std 监控指标
- 217–315 `eval()` / 318–332 `save_ckpt` / 334–600 `main()` setup(模型/LoRA/优化器/dataloader/EMA)
- 851–884 dict↔list、tqdm、分批

### 🟡 选读(算法相关但偏工程)
- 79–122 `DistributedKRepeatSampler`("4 个 prompt 怎么来的")
- 428–445 LoRA 注入 + `lora_path` 续训 + reference 模型

### 主循环结构(行号锚点)
```
for epoch:
  602–606  eval + save_ckpt(每60轮)        ⚪
  609–668  采样:生成图+记log_prob+打分      🟢② ⭐
  744–810  reward→组内优势                  🟢③ ⭐
  840  for inner_epoch:                     # PPO 内循环
    859  for i, sample:                      # 分 minibatch
      878  for j in 时间步:                  # 逐去噪步
        885  with accumulate:
          887  compute_log_prob              🟢① ⭐⭐⭐
          899–910  ratio/clip/kl/loss        🟢④ ⭐⭐⭐
          946–952  backward+step             🟢⑤
```

---

## 怎么用好"训练在跑"这段时间
- 边看日志边对代码:每出现一个 `[metrics]`,问"这一步代码做了什么"。
- 从手写版入手理解算法,再去官方看工程化(FSDP/EMA/分布式是"工程"不是"算法",别先陷进去)。
- 看到 887/899 这些行=算法;看到 124/318=外壳可跳过。

## 算法核对照表(官方 ↔ 手写)
| 概念 | 官方 train_sd3.py | 手写 flow-grpo |
|---|---|---|
| 高斯 log-prob + SDE 步 | `compute_log_prob` 181–215 | `log_prob.py` |
| 轨迹采样 | `pipeline_with_logprob` (diffusers_patch) | `sampling.py` |
| 组内相对优势 | `stat_tracking.py` `PerPromptStatTracker` | `advantage.py` |
| credit assignment | (官方无,= 你的增量) | `expand_advantage_over_steps` |
| clipped surrogate + KL | 899–910 | `loss.py` `grpo_loss` |
