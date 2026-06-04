# sany_PID

`sany_PID` 用于研究轻量级强化学习策略对传统 PID 参数整定的辅助作用。项目以水泥浆密度控制过程为对象，包含两类运行情景：预混情景和生产情景。代码围绕“仿真建模—基准 PID 参数计算—PPO-bandit 策略训练—成对评估—补充实验”组织。

本文中的策略网络不是在每个控制周期反复输出控制量，而是在每个运行情景开始前，根据当前工况特征推理一次，输出 PID 参数缩放因子。随后该组 PID 参数在整个情景内保持不变。控制周期内仍由 PID 控制器执行闭环控制。

## 1. 项目结构

```text
sany_PID/
├── scripts/
│   ├── core/
│   │   ├── sim_config.py
│   │   ├── sim_model.py
│   │   └── sim_env.py
│   ├── PID_control/
│   │   ├── PIDcontroller.py
│   │   └── baseline.py
│   ├── rl/
│   │   ├── PPO_bandit.py
│   │   ├── modes.py
│   │   ├── rollout.py
│   │   ├── metrics.py
│   │   ├── plotting.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── reviewer_lightweight_experiments.py
│   ├── examples/
│   │   ├── sim_examples.py
│   │   └── archive_examples.py
│   └── legacy/
│       ├── IMC_PID_tuning.py
│       ├── evaluate_compare.py
│       ├── seed_sweep.py
│       └── train_modes.py
├── README.md
└── requirement.txt
```

其中推荐使用的主流程代码都位于 `scripts/rl/`、`scripts/core/` 和 `scripts/PID_control/` 中。`scripts/examples/` 用于简单仿真示例。`scripts/legacy/` 是历史版本迁移参考，不作为当前论文实验的推荐运行入口。

## 2. 主要模块说明

### 2.1 仿真核心模块

`core/sim_config.py` 定义仿真相关配置和参数结构。

`core/sim_model.py` 实现水泥浆罐体、阀门、混合过程、密度响应、液位响应和观测延迟等过程模型。

`core/sim_env.py` 封装闭环仿真环境，用于在给定 PID 参数和工况扰动下运行一个完整控制情景。

### 2.2 PID 与基准参数模块

`PID_control/PIDcontroller.py` 实现 PID 控制器逻辑，包括比例、积分、微分项计算和控制量约束。

`PID_control/baseline.py` 计算传统 PID 参数和前馈参数，作为 RL 策略缩放的基础参数来源。

### 2.3 强化学习模块

`rl/PPO_bandit.py` 实现 PPO-bandit 策略网络。该策略将当前情景特征映射为 PID 参数缩放因子，而不是直接输出每个控制周期的阀门动作。

`rl/modes.py` 定义不同运行情景的任务规格，包括工况采样、特征构造、动作缩放项、基准参数计算和奖励定义。

`rl/rollout.py` 负责单情景仿真、共享不确定性采样、PID 缩放因子应用、奖励计算和轨迹保存。

`rl/metrics.py` 负责控制性能指标计算、成对比较统计、bootstrap 置信区间、CSV/JSON 汇总和多 seed 结果汇总。

`rl/plotting.py` 负责训练曲线、评估分布图、缩放因子图和成对改进图的绘制。

`rl/train.py` 是当前推荐训练入口，支持预混、生产或两种情景同时训练，也支持多 seed 训练和结果汇总。

`rl/evaluate.py` 是当前推荐评估入口，用于加载训练好的 checkpoint，并在相同随机工况下比较 RL 缩放策略和基准 PID 控制器。

`rl/reviewer_lightweight_experiments.py` 用于补充实验，包括单次策略推理耗时测试和训练区间外极端工况泛化测试。

## 3. 两类运行情景

### 3.1 预混情景 `premix`

预混情景主要关注密度跟踪。策略网络根据当前工况特征输出水泥通道 PID 参数缩放因子：

```text
s_c_p, s_c_i, s_c_d
```

其中 `c` 表示 cement。评估重点包括密度误差、密度 IAE、控制奖励和相对于基准 PID 的改进。

### 3.2 生产情景 `production`

生产情景同时关注液位控制和密度控制。策略网络输出水通道和水泥通道的 PID 参数缩放因子：

```text
s_w_p, s_w_i, s_w_d, s_c_p, s_c_i, s_c_d
```

其中 `w` 表示 water，`c` 表示 cement。评估重点包括液位 IAE、密度 IAE、综合控制奖励和相对于基准 PID 的改进。

## 4. 推荐运行顺序

建议从项目根目录运行所有命令。

推荐顺序如下：

```text
1. 训练预混情景策略
2. 训练生产情景策略
3. 汇总多 seed 训练结果
4. 对最佳 checkpoint 进行成对评估
5. 运行补充实验
```

## 5. 策略训练

### 5.1 训练预混情景

```bash
python -m scripts.rl.train \
  --mode premix \
  --seeds 1 2 3 4 5 \
  --device cuda \
  --updates 1000 \
  --batch_episodes 64 \
  --eval_every 20 \
  --eval_episodes 100 \
  --save_every 50 \
  --num_workers 4 \
  --dt 0.5 \
  --premix_duration 120 \
  --premix_hold 10 \
  --ckpt_dir outputs/premix_train
```

### 5.2 训练生产情景

```bash
python -m scripts.rl.train \
  --mode production \
  --seeds 1 2 3 4 5 \
  --device cuda \
  --updates 1500 \
  --batch_episodes 64 \
  --eval_every 20 \
  --eval_episodes 100 \
  --save_every 50 \
  --num_workers 4 \
  --dt 0.5 \
  --production_duration 240 \
  --ckpt_dir outputs/production_train
```

### 5.3 同时训练两类情景

如果希望在同一次命令中训练预混和生产情景，可以使用：

```bash
python -m scripts.rl.train \
  --mode both \
  --seeds 1 2 3 4 5 \
  --device cuda \
  --updates 1000 \
  --batch_episodes 64 \
  --eval_every 20 \
  --eval_episodes 100 \
  --save_every 50 \
  --num_workers 4 \
  --dt 0.5 \
  --premix_duration 120 \
  --premix_hold 10 \
  --production_duration 240 \
  --ckpt_dir outputs/both_train
```

单独训练两类情景时，输出目录更清晰；同时训练适合快速批量实验。

## 6. 训练输出

多 seed 训练后，典型输出结构如下：

```text
outputs/premix_train/
├── seed1/
│   ├── premix_best.pt
│   ├── premix_ppo_bandit_00050.pt
│   ├── premix_ppo_bandit_00100.pt
│   └── plots/
│       ├── premix_seed1_summary.json
│       ├── premix_seed1_metrics_series.npz
│       └── *.png
├── seed2/
├── seed3/
├── seed4/
├── seed5/
├── train_summary_all.json
└── seed_sweep_summary/
    ├── seed_sweep_table.csv
    ├── seed_sweep_summary.json
    └── seed_sweep_boxplot.png
```

生产情景输出结构类似，文件名前缀由 `premix` 替换为 `production`。

主要文件含义如下：

| 文件 | 含义 |
|---|---|
| `*_best.pt` | 训练过程中按照指定指标保存的最佳策略 checkpoint |
| `*_ppo_bandit_*.pt` | 周期性保存的策略 checkpoint |
| `*_seed*_summary.json` | 单个 seed 的训练与最终评估摘要 |
| `*_seed*_metrics_series.npz` | 训练过程中的奖励、评估指标和损失序列 |
| `train_summary_all.json` | 当前训练命令下所有 seed 和 mode 的总汇总 |
| `seed_sweep_table.csv` | 多 seed 结果表 |
| `seed_sweep_summary.json` | 多 seed 均值、标准差、最优 seed 等摘要 |
| `seed_sweep_boxplot.png` | 多 seed 指标分布图 |

## 7. 多 seed 结果汇总

训练脚本在多 seed 运行结束后会自动生成汇总结果。如果需要单独重新汇总，可以使用 `--summarize_only`。

### 7.1 汇总预混训练结果

```bash
python -m scripts.rl.train \
  --summarize_only \
  --summary_root outputs/premix_train \
  --summary_metric eval_metrics.R_mean \
  --summary_out_dir outputs/premix_train/seed_sweep_summary_manual
```

### 7.2 汇总生产训练结果

```bash
python -m scripts.rl.train \
  --summarize_only \
  --summary_root outputs/production_train \
  --summary_metric eval_metrics.R_mean \
  --summary_out_dir outputs/production_train/seed_sweep_summary_manual
```

`--summary_metric` 可以根据论文关注重点调整。常用指标包括：

```text
eval_metrics.R_mean
eval_metrics.density_iae_mean
eval_metrics.level_iae_mean
```

## 8. 成对评估

评估脚本会在相同随机工况下分别运行 RL 缩放策略和基准 PID 控制器，从而进行公平的成对比较。推荐对训练得到的最佳 checkpoint 单独评估。

### 8.1 评估预混策略

```bash
python -m scripts.rl.evaluate \
  --mode premix \
  --ckpt outputs/premix_train/seed1/premix_best.pt \
  --out_dir outputs/premix_eval/seed1 \
  --N 500 \
  --seed 2026 \
  --device cuda \
  --dt 0.5 \
  --premix_duration 120 \
  --premix_hold 10 \
  --bootstrap_B 2000 \
  --save_full_if_n_le 20
```

### 8.2 评估生产策略

```bash
python -m scripts.rl.evaluate \
  --mode production \
  --ckpt outputs/production_train/seed1/production_best.pt \
  --out_dir outputs/production_eval/seed1 \
  --N 500 \
  --seed 2026 \
  --device cuda \
  --dt 0.5 \
  --production_duration 240 \
  --bootstrap_B 2000 \
  --save_full_if_n_le 20
```

### 8.3 评估输出

典型评估输出包括：

```text
outputs/premix_eval/seed1/premix/
├── premix_paired_compare_summary.json
├── premix_paired_compare_report.txt
├── premix_paired_cases.csv
├── premix_rl_scales.csv
├── premix_paired_compare.npz
└── plots/
    ├── premix_baseline_reward_distribution.png
    ├── premix_rl_reward_distribution.png
    ├── premix_delta_reward_hist.png
    ├── premix_delta_density_iae_hist.png
    ├── premix_delta_reward_scatter.png
    ├── premix_delta_density_iae_scatter.png
    ├── premix_scale_boxplot.png
    └── premix_scale_scatter.png
```

生产情景还会额外生成与外输排量相关的分析图，例如：

```text
production_delta_vs_qs.png
production_delta_qs_bucket.png
```

主要输出文件含义如下：

| 文件 | 含义 |
|---|---|
| `*_paired_compare_summary.json` | 成对评估核心统计结果，包括均值、胜率、改进量和 bootstrap 置信区间 |
| `*_paired_compare_report.txt` | 便于直接阅读的文本版评估报告 |
| `*_paired_cases.csv` | 每个测试情景的基准 PID 与 RL 策略表现 |
| `*_rl_scales.csv` | 每个测试情景中 RL 策略输出的 PID 缩放因子 |
| `*_paired_compare.npz` | 评估数组数据，便于后续重新作图或统计 |
| `plots/` | 评估分布图、成对改进图和缩放因子图 |

## 9. 补充实验

`reviewer_lightweight_experiments.py` 用于生成补充实验结果，主要包括：

```text
1. 单次策略推理耗时测试
2. 训练区间外极端工况泛化测试
```

该脚本只加载已经训练好的 checkpoint，不重新训练策略。

### 9.1 同时运行预混和生产补充实验

```bash
python -m scripts.rl.reviewer_lightweight_experiments \
  --mode both \
  --premix_ckpt outputs/premix_train/seed1/premix_best.pt \
  --production_ckpt outputs/production_train/seed1/production_best.pt \
  --out_dir outputs/reviewer_lightweight \
  --N 1000 \
  --seed 2026 \
  --device cuda \
  --dt 0.5 \
  --premix_duration 120 \
  --premix_hold 10 \
  --production_duration 240 \
  --timing_repeats 10000
```

### 9.2 只运行推理耗时测试

```bash
python -m scripts.rl.reviewer_lightweight_experiments \
  --mode both \
  --premix_ckpt outputs/premix_train/seed1/premix_best.pt \
  --production_ckpt outputs/production_train/seed1/production_best.pt \
  --out_dir outputs/reviewer_lightweight_timing_only \
  --device cuda \
  --no_ood
```

### 9.3 只运行极端工况泛化测试

```bash
python -m scripts.rl.reviewer_lightweight_experiments \
  --mode both \
  --premix_ckpt outputs/premix_train/seed1/premix_best.pt \
  --production_ckpt outputs/production_train/seed1/production_best.pt \
  --out_dir outputs/reviewer_lightweight_ood_only \
  --N 1000 \
  --seed 2026 \
  --device cuda \
  --dt 0.5 \
  --premix_duration 120 \
  --premix_hold 10 \
  --production_duration 240 \
  --no_timing
```

### 9.4 补充实验输出

典型输出包括：

```text
outputs/reviewer_lightweight/
├── premix_single_inference_timing.csv
├── premix_single_inference_timing.json
├── premix_ood_cases.csv
├── premix_ood_summary.csv
├── premix_ood_summary.json
├── premix_reviewer_lightweight_result.json
├── production_single_inference_timing.csv
├── production_single_inference_timing.json
├── production_ood_cases.csv
├── production_ood_summary.csv
├── production_ood_summary.json
├── production_reviewer_lightweight_result.json
└── reviewer_lightweight_all_results.json
```

推理耗时测试记录策略网络单次前向推理时间，不包含闭环仿真时间。

极端工况泛化测试默认包含：

| 情景 | 极端工况 |
|---|---|
| 预混、生产 | 混合时间常数 `100`--`150 s` |
| 预混、生产 | 观测时滞 `20`--`30 s` |
| 生产 | 外输排量 `1.5`--`2.0 m^3/min` |

如需额外测试组合极端工况，可加入：

```bash
--include_combined_extreme
```

## 10. 常用参数说明

### 10.1 训练参数

| 参数 | 说明 |
|---|---|
| `--mode` | 运行情景，可选 `premix`、`production`、`both` |
| `--seed` | 单个随机种子 |
| `--seeds` | 多个随机种子，优先级高于 `--seed` |
| `--device` | 计算设备，例如 `cpu` 或 `cuda` |
| `--updates` | PPO-bandit 更新轮数 |
| `--batch_episodes` | 每轮更新采样的情景数量 |
| `--eval_every` | 每隔多少轮进行一次评估 |
| `--eval_episodes` | 每次训练中评估使用的情景数量 |
| `--save_every` | checkpoint 保存间隔 |
| `--best_metric` | 选择最佳 checkpoint 的指标，默认使用 `R_mean` |
| `--num_workers` | 并行采样进程数 |
| `--ckpt_dir` | checkpoint 和训练输出目录 |
| `--dt` | 控制周期，单位为秒 |
| `--premix_duration` | 预混情景仿真时长 |
| `--premix_hold` | 预混情景前期保持时间 |
| `--production_duration` | 生产情景仿真时长 |

### 10.2 评估参数

| 参数 | 说明 |
|---|---|
| `--mode` | 评估情景，可选 `premix`、`production`、`both` |
| `--ckpt` | 单个情景对应的 checkpoint 路径 |
| `--ckpt_dir` | 同时评估两个情景时的 checkpoint 目录 |
| `--out_dir` | 评估输出目录 |
| `--N` | 成对评估情景数量 |
| `--episodes` | `--N` 的别名 |
| `--seed` | 评估随机种子 |
| `--device` | 计算设备 |
| `--bootstrap_B` | bootstrap 重采样次数 |
| `--save_full_if_n_le` | 当评估情景数不超过该值时保存完整轨迹 |
| `--no_plots` | 只保存数值结果，不生成图 |

### 10.3 补充实验参数

| 参数 | 说明 |
|---|---|
| `--premix_ckpt` | 预混策略 checkpoint 路径 |
| `--production_ckpt` | 生产策略 checkpoint 路径 |
| `--timing_repeats` | 推理耗时测试重复次数 |
| `--timing_warmup` | 推理耗时测试预热次数 |
| `--include_combined_extreme` | 是否加入组合极端工况测试 |
| `--no_timing` | 不运行推理耗时测试 |
| `--no_ood` | 不运行极端工况泛化测试 |

## 11. 实验建议流程

建议保持以下原则：

1. 训练和评估使用相同的 `dt`、`premix_duration`、`premix_hold` 和 `production_duration`。
2. 主结果优先报告独立评估脚本得到的成对比较结果，而不是训练过程中的在线评估结果。
3. 多 seed 训练用于观察策略稳定性；论文主表可选择最佳 seed 或多 seed 汇总，但需要在正文中说明选择规则。
4. 成对评估应固定 `--seed` 和 `--N`，便于复现实验结果。
5. 如果需要展示完整轨迹图，可以适当减小 `--N` 或设置 `--save_full_if_n_le`，避免保存过多轨迹文件。
6. 补充实验中的推理耗时只反映策略参数整定环节开销，不等同于完整闭环仿真耗时。

## 12. 最小完整命令序列

下面给出一组从训练到评估再到补充实验的最小完整命令。路径可根据实际 seed 和输出目录调整。

```bash
python -m scripts.rl.train --mode premix --seeds 1 2 3 4 5 --device cuda --updates 1000 --batch_episodes 64 --eval_every 20 --eval_episodes 100 --save_every 50 --num_workers 4 --dt 0.5 --premix_duration 120 --premix_hold 10 --ckpt_dir outputs/premix_train
```

```bash
python -m scripts.rl.train --mode production --seeds 1 2 3 4 5 --device cuda --updates 1500 --batch_episodes 64 --eval_every 20 --eval_episodes 100 --save_every 50 --num_workers 4 --dt 0.5 --production_duration 240 --ckpt_dir outputs/production_train
```

```bash
python -m scripts.rl.evaluate --mode premix --ckpt outputs/premix_train/seed1/premix_best.pt --out_dir outputs/premix_eval/seed1 --N 500 --seed 2026 --device cuda --dt 0.5 --premix_duration 120 --premix_hold 10 --bootstrap_B 2000 --save_full_if_n_le 20
```

```bash
python -m scripts.rl.evaluate --mode production --ckpt outputs/production_train/seed1/production_best.pt --out_dir outputs/production_eval/seed1 --N 500 --seed 2026 --device cuda --dt 0.5 --production_duration 240 --bootstrap_B 2000 --save_full_if_n_le 20
```

```bash
python -m scripts.rl.reviewer_lightweight_experiments --mode both --premix_ckpt outputs/premix_train/seed1/premix_best.pt --production_ckpt outputs/production_train/seed1/production_best.pt --out_dir outputs/reviewer_lightweight --N 1000 --seed 2026 --device cuda --dt 0.5 --premix_duration 120 --premix_hold 10 --production_duration 240 --timing_repeats 10000
```
