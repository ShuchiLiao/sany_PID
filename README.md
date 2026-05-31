# sany_PID

`sany_PID` 是一个面向固井连续混浆过程的仿真与控制实验项目。项目主要包含三部分：第一，建立水泥浆连续混合过程的仿真模型；第二，构建 PID / FF 基准控制方法；第三，使用 PPO-bandit 对基准控制参数进行缩放整定，并通过配对仿真评估与基准方法的差异。

当前代码支持两类主要工况：

* `premix`：预混阶段，重点关注水泥浆密度控制；
* `production`：生产阶段，重点关注水泥浆密度与液位协同控制。

项目的主要运行入口为：

```text
python -m scripts.rl.train
python -m scripts.rl.evaluate
```

所有命令建议在项目根目录下执行。

---

## 1. 代码结构

当前核心代码位于 `scripts/` 目录下，按功能分为三层。

```text
scripts/
├── core/
│   ├── sim_config.py
│   ├── sim_model.py
│   └── sim_env.py
├── PID_control/
│   ├── PIDcontroller.py
│   └── baseline.py
└── rl/
    ├── PPO_bandit.py
    ├── modes.py
    ├── rollout.py
    ├── metrics.py
    ├── plotting.py
    ├── train.py
    └── evaluate.py
```

各部分作用如下。

| 目录 / 文件                                | 作用                                          |
| -------------------------------------- | ------------------------------------------- |
| `scripts/core/sim_config.py`           | 定义仿真参数、阀门参数、控制参数等配置                         |
| `scripts/core/sim_model.py`            | 定义罐体、阀门、混合过程等物理模型                           |
| `scripts/core/sim_env.py`              | 封装闭环仿真环境                                    |
| `scripts/PID_control/PIDcontroller.py` | PID 控制器实现                                   |
| `scripts/PID_control/baseline.py`      | 基准 PID / FF 参数整定                            |
| `scripts/rl/PPO_bandit.py`             | PPO-bandit 算法主体                             |
| `scripts/rl/modes.py`                  | 定义 `premix` 与 `production` 两类任务模式           |
| `scripts/rl/rollout.py`                | episode 仿真、共享扰动、轨迹生成                        |
| `scripts/rl/metrics.py`                | 控制指标、配对统计、bootstrap、多 seed 汇总               |
| `scripts/rl/plotting.py`               | 训练曲线、评估图、配对对比图                              |
| `scripts/rl/train.py`                  | 训练入口，支持单 seed 和多 seed                       |
| `scripts/rl/evaluate.py`               | 评估入口，支持 baseline-only 和 baseline vs RL 配对比较 |

---

## 2. 环境准备

建议使用 Python 3.10 或更高版本。

如果使用 conda，可先创建并激活环境：

```bash
conda create -n sany python=3.10 -y
```

```bash
conda activate sany
```

安装依赖：

```bash
python -m pip install -U pip
```

```bash
python -m pip install numpy scipy matplotlib torch
```

如果使用 GPU 训练，请根据本机 CUDA 版本安装对应的 PyTorch 版本。CPU 训练可以直接使用上述安装方式。

---

## 3. 基本检查

首次运行前，建议先检查代码是否可以正常编译：

```bash
python -m compileall scripts/core scripts/PID_control scripts/rl
```

如果该命令没有报错，说明 Python 语法和基础导入链路基本正常。

---

## 4. 训练入口

训练统一使用：

```text
python -m scripts.rl.train
```

训练入口支持单 seed 训练，也支持多 seed 批量训练。

---

## 5. premix 正式训练示例

`premix` 工况主要用于预混阶段密度控制。正式实验建议使用多 seed 训练，以减少单一随机种子的偶然性。

推荐命令如下：

```bash
python -m scripts.rl.train --mode premix --seeds 1 2 3 4 5 --device cuda --updates 1000 --batch_episodes 64 --eval_every 20 --eval_episodes 100 --save_every 50 --num_workers 4 --dt 0.5 --premix_duration 120 --premix_hold 10 --ckpt_dir outputs/premix_train
```

参数说明如下。

| 参数                                | 含义                         | 推荐设置                             |
| --------------------------------- | -------------------------- | -------------------------------- |
| `--mode premix`                   | 指定训练 premix 工况             | premix 实验固定使用 `premix`           |
| `--seeds 1 2 3 4 5`               | 多随机种子训练                    | 正式实验建议至少 5 个 seed                |
| `--device cuda`                    | 训练设备                       | 无 GPU 时用 `cpu`；有 CUDA 时可用 `cuda` |
| `--updates 1000`                  | PPO 参数更新次数                 | 正式实验建议 1000–3000 起步              |
| `--batch_episodes 64`             | 每次 update 采样的 episode 数    | 推荐 64；计算资源充足可设 128               |
| `--eval_every 20`                 | 每隔多少次 update 做一次评估         | 推荐 20；训练较长时可设 50                 |
| `--eval_episodes 100`             | 每次评估使用的 episode 数          | 推荐 100；正式结果可设 200                |
| `--save_every 50`                 | 每隔多少次 update 保存 checkpoint | 推荐 50 或 100                      |
| `--num_workers 4`                 | 并行 rollout worker 数量       | CPU 核数较少可设 0 或 2；多核机器可设 4–8      |
| `--dt 0.5`                        | 仿真步长                       | 推荐与论文/实验设定保持一致，通常可用 0.5          |
| `--premix_duration 120`           | premix 单次仿真时长              | 推荐 120 s；如需覆盖更长动态可设 180 s        |
| `--premix_hold 10`                | premix 初始保持时长              | 推荐 5–10 s                        |
| `--ckpt_dir outputs/premix_train` | checkpoint 和训练结果输出目录       | 推荐放在 `outputs/` 下                |

运行后，典型输出包括：

```text
outputs/premix_train/
├── seed1/
├── seed2/
├── seed3/
├── seed4/
├── seed5/
├── train_summary_all.json
└── seed_sweep_summary/
```

每个 seed 子目录中通常包括：

```text
premix_best.pt
premix_ppo_bandit_*.pt
plots/premix_seed*_summary.json
plots/premix_seed*_metrics_series.npz
plots/*.png
```

---

## 6. production 正式训练示例

`production` 工况用于生产阶段密度与液位协同控制。由于 production 状态维度和控制目标更多，训练通常比 premix 更复杂。

推荐命令如下：

```bash
python -m scripts.rl.train --mode production --seeds 1 2 3 4 5 --device cuda --updates 1500 --batch_episodes 64 --eval_every 20 --eval_episodes 100 --save_every 50 --num_workers 4 --dt 0.5 --production_duration 240 --ckpt_dir outputs/production_train
```

参数说明如下。

| 参数                                    | 含义                      | 推荐设置                             |
| ------------------------------------- | ----------------------- | -------------------------------- |
| `--mode production`                   | 指定训练 production 工况      | production 实验固定使用 `production`   |
| `--seeds 1 2 3 4 5`                   | 多随机种子训练                 | 正式实验建议至少 5 个 seed                |
| `--device cuda`                        | 训练设备                    | 无 GPU 时用 `cpu`；有 CUDA 时可用 `cuda` |
| `--updates 1500`                      | PPO 参数更新次数              | production 推荐 1500–3000          |
| `--batch_episodes 64`                 | 每次 update 采样的 episode 数 | 推荐 64；计算资源充足可设 128               |
| `--eval_every 20`                     | 每隔多少次 update 做一次评估      | 推荐 20 或 50                       |
| `--eval_episodes 100`                 | 每次评估使用的 episode 数       | 推荐 100；正式结果可设 200                |
| `--save_every 50`                     | checkpoint 保存间隔         | 推荐 50 或 100                      |
| `--num_workers 4`                     | 并行 rollout worker 数量    | 推荐 4；单机资源有限可设 0 或 2              |
| `--dt 0.5`                            | 仿真步长                    | 推荐与论文/实验设定保持一致，通常可用 0.5          |
| `--production_duration 240`           | production 单次仿真时长       | 推荐 240 s；如需覆盖更长工况可设 300 s        |
| `--ckpt_dir outputs/production_train` | 输出目录                    | 推荐放在 `outputs/` 下                |

运行后，典型输出包括：

```text
outputs/production_train/
├── seed1/
├── seed2/
├── seed3/
├── seed4/
├── seed5/
├── train_summary_all.json
└── seed_sweep_summary/
```

---

## 7. 多 seed 汇总

多 seed 训练完成后，训练脚本会自动生成汇总结果。也可以单独对已有训练结果重新汇总。

premix 汇总示例：

```bash
python -m scripts.rl.train --summarize_only --summary_root outputs/premix_train --summary_metric eval_metrics.R_mean --summary_out_dir outputs/premix_train/seed_sweep_summary_manual
```

production 汇总示例：

```bash
python -m scripts.rl.train --summarize_only --summary_root outputs/production_train --summary_metric eval_metrics.R_mean --summary_out_dir outputs/production_train/seed_sweep_summary_manual
```

参数说明如下。

| 参数                  | 含义             | 推荐设置                       |
| ------------------- | -------------- | -------------------------- |
| `--summarize_only`  | 只做已有结果汇总，不重新训练 | 需要单独汇总时使用                  |
| `--summary_root`    | 需要扫描的训练结果根目录   | 指向包含 `seed1/seed2/...` 的目录 |
| `--summary_metric`  | 用于汇总的指标字段      | 推荐 `eval_metrics.R_mean`   |
| `--summary_out_dir` | 汇总结果输出目录       | 建议放在训练目录下                  |

汇总输出通常包括：

```text
seed_sweep_table.csv
seed_sweep_summary.json
seed_sweep_boxplot.png
```

---

## 8. 评估入口

评估统一使用：

```text
python -m scripts.rl.evaluate
```

评估脚本支持三种方式：

1. `baseline vs RL` 配对评估；
2. `baseline-only` 评估；
3. 同时评估 `premix` 和 `production`。

配对评估时，baseline 和 RL 使用相同的工况采样和相同的随机扰动，因此可以比较两者在同一批 scenario 下的差异。

---

## 9. premix 正式评估示例

如果使用多 seed 训练，建议分别评估每个 seed 的 `premix_best.pt`，再对评估结果做汇总分析。下面以 `seed1` 为例。

```bash
python -m scripts.rl.evaluate --mode premix --ckpt outputs/premix_train/seed1/premix_best.pt --N 500 --seed 2026 --device cuda --dt 0.5 --premix_duration 120 --premix_hold 10 --bootstrap_B 2000 --save_full_if_n_le 20 --out_dir outputs/premix_eval/seed1
```

参数说明如下。

| 参数                       | 含义                       | 推荐设置                           |
| ------------------------ | ------------------------ | ------------------------------ |
| `--mode premix`          | 指定评估 premix 工况           | premix 评估固定用 `premix`          |
| `--ckpt`                 | 待评估的 RL checkpoint       | 通常使用对应 seed 的 `premix_best.pt` |
| `--N 500`                | Monte Carlo 评估 episode 数 | 正式评估推荐 500；资源充足可设 1000         |
| `--seed 2026`            | 评估随机种子                   | 推荐固定一个与训练 seed 不同的评估 seed      |
| `--device cuda`           | 模型推理设备                   | 无 GPU 用 `cpu`；有 CUDA 可用 `cuda` |
| `--dt 0.5`               | 仿真步长                     | 应与训练和论文设定一致                    |
| `--premix_duration 120`  | premix 评估仿真时长            | 应与训练设定一致                       |
| `--premix_hold 10`       | premix 初始保持时长            | 应与训练设定一致                       |
| `--bootstrap_B 2000`     | bootstrap 重采样次数          | 正式结果推荐 2000；更稳健可设 5000         |
| `--save_full_if_n_le 20` | 当 `N` 不超过该值时保存完整轨迹       | 正式大规模评估建议 20，避免保存过多轨迹          |
| `--out_dir`              | 评估结果输出目录                 | 推荐按 mode 和 seed 单独建目录          |

典型输出包括：

```text
outputs/premix_eval/seed1/premix/
├── premix_paired_compare_summary.json
├── premix_paired_compare_report.txt
├── premix_paired_cases.csv
├── premix_rl_scales.csv
├── premix_paired_compare.npz
├── premix_delta_return_hist.png
├── premix_delta_vs_tau_mix.png
├── premix_delta_vs_delay.png
├── premix_delta_tau_bucket.png
├── premix_scale_boxplot.png
└── premix_scales_vs_tau_mix.png
```

---

## 10. production 正式评估示例

下面以 `seed1` 的 production checkpoint 为例。

```bash
python -m scripts.rl.evaluate --mode production --ckpt outputs/production_train/seed1/production_best.pt --N 500 --seed 2026 --device cuda --dt 0.5 --production_duration 240 --bootstrap_B 2000 --save_full_if_n_le 20 --out_dir outputs/production_eval/seed1
```

参数说明如下。

| 参数                          | 含义                       | 推荐设置                               |
| --------------------------- | ------------------------ | ---------------------------------- |
| `--mode production`         | 指定评估 production 工况       | production 评估固定用 `production`      |
| `--ckpt`                    | 待评估的 RL checkpoint       | 通常使用对应 seed 的 `production_best.pt` |
| `--N 500`                   | Monte Carlo 评估 episode 数 | 正式评估推荐 500；资源充足可设 1000             |
| `--seed 2026`               | 评估随机种子                   | 推荐固定一个与训练 seed 不同的评估 seed          |
| `--device cuda`              | 模型推理设备                   | 无 GPU 用 `cpu`；有 CUDA 可用 `cuda`     |
| `--dt 0.5`                  | 仿真步长                     | 应与训练和论文设定一致                        |
| `--production_duration 240` | production 评估仿真时长        | 应与训练设定一致                           |
| `--bootstrap_B 2000`        | bootstrap 重采样次数          | 正式结果推荐 2000；更稳健可设 5000             |
| `--save_full_if_n_le 20`    | 当 `N` 不超过该值时保存完整轨迹       | 正式大规模评估建议 20                       |
| `--out_dir`                 | 评估结果输出目录                 | 推荐按 mode 和 seed 单独建目录              |

典型输出包括：

```text
outputs/production_eval/seed1/production/
├── production_paired_compare_summary.json
├── production_paired_compare_report.txt
├── production_paired_cases.csv
├── production_rl_scales.csv
├── production_paired_compare.npz
├── production_delta_return_hist.png
├── production_delta_vs_tau_mix.png
├── production_delta_vs_delay.png
├── production_delta_vs_qs.png
├── production_delta_qs_bucket.png
├── production_scale_boxplot.png
└── production_scales_vs_tau_mix.png
```

---

## 11. baseline-only 正式评估示例

如果只需要评估基准控制器，不加载 RL checkpoint，可以使用 `--baseline_only`。

同时评估 premix 与 production：

```bash
python -m scripts.rl.evaluate --mode both --baseline_only --N 500 --seed 2026 --device cpu --dt 0.5 --premix_duration 120 --premix_hold 10 --production_duration 240 --bootstrap_B 2000 --save_full_if_n_le 20 --out_dir outputs/baseline_eval
```

参数说明如下。

| 参数                                | 含义                             | 推荐设置                        |
| --------------------------------- | ------------------------------ | --------------------------- |
| `--mode both`                     | 同时评估 premix 和 production       | baseline-only 总体验证可用 `both` |
| `--baseline_only`                 | 不加载 RL checkpoint，只评估 baseline | 基准方法评估时使用                   |
| `--N 500`                         | Monte Carlo episode 数          | 正式评估推荐 500；资源充足可设 1000      |
| `--seed 2026`                     | 评估随机种子                         | 推荐固定                        |
| `--device cpu`                    | 推理设备                           | baseline-only 通常用 `cpu` 即可  |
| `--dt 0.5`                        | 仿真步长                           | 与训练和论文设定一致                  |
| `--premix_duration 120`           | premix 仿真时长                    | 与 premix 实验设定一致             |
| `--premix_hold 10`                | premix 初始保持时长                  | 与 premix 实验设定一致             |
| `--production_duration 240`       | production 仿真时长                | 与 production 实验设定一致         |
| `--bootstrap_B 2000`              | bootstrap 次数                   | 推荐 2000                     |
| `--save_full_if_n_le 20`          | 小样本时保存完整轨迹                     | 正式大样本评估通常不会触发               |
| `--out_dir outputs/baseline_eval` | 输出目录                           | 推荐单独保存 baseline 结果          |

baseline-only 模式下，输出仍然包含 paired summary、report、CSV、NPZ 和图表。由于不加载 RL 模型，RL 分支会使用 baseline 参数，因此两者的 paired delta 通常为 0。

---

## 12. 推荐实验流程

建议完整实验按以下顺序执行。

第一步，训练 premix：

```bash
python -m scripts.rl.train --mode premix --seeds 1 2 3 4 5 --device cuda --updates 1000 --batch_episodes 64 --eval_every 20 --eval_episodes 100 --save_every 50 --num_workers 4 --dt 0.5 --premix_duration 120 --premix_hold 10 --ckpt_dir outputs/premix_train
```

第二步，训练 production：

```bash
python -m scripts.rl.train --mode production --seeds 1 2 3 4 5 --device cuda --updates 1500 --batch_episodes 64 --eval_every 20 --eval_episodes 100 --save_every 50 --num_workers 4 --dt 0.5 --production_duration 240 --ckpt_dir outputs/production_train
```

第三步，评估 premix 每个 seed 的 best checkpoint。以 seed1 为例：

```bash
python -m scripts.rl.evaluate --mode premix --ckpt outputs/premix_train/seed1/premix_best.pt --N 500 --seed 2026 --device cuda --dt 0.5 --premix_duration 120 --premix_hold 10 --bootstrap_B 2000 --save_full_if_n_le 20 --out_dir outputs/premix_eval/seed1
```

第四步，评估 production 每个 seed 的 best checkpoint。以 seed1 为例：

```bash
python -m scripts.rl.evaluate --mode production --ckpt outputs/production_train/seed1/production_best.pt --N 500 --seed 2026 --device cuda --dt 0.5 --production_duration 240 --bootstrap_B 2000 --save_full_if_n_le 20 --out_dir outputs/production_eval/seed1
```

第五步，评估 baseline：

```bash
python -m scripts.rl.evaluate --mode both --baseline_only --N 500 --seed 2026 --device cpu --dt 0.5 --premix_duration 120 --premix_hold 10 --production_duration 240 --bootstrap_B 2000 --save_full_if_n_le 20 --out_dir outputs/baseline_eval
```

---

## 13. 结果文件说明

### 13.1 训练结果

训练结果通常包含：

| 文件                     | 含义                          |
| ---------------------- | --------------------------- |
| `*_best.pt`            | 当前 seed 下评估表现最好的 checkpoint |
| `*_ppo_bandit_*.pt`    | 按保存间隔保存的 checkpoint         |
| `*_summary.json`       | 单个 seed 的训练与评估摘要            |
| `*_metrics_series.npz` | 训练过程中的 reward、loss、评估指标序列   |
| `*.png`                | 训练曲线和评估指标曲线                 |

### 13.2 多 seed 汇总结果

多 seed 汇总通常包含：

| 文件                        | 含义                    |
| ------------------------- | --------------------- |
| `seed_sweep_table.csv`    | 每个 seed 的关键指标表        |
| `seed_sweep_summary.json` | 多 seed 均值、标准差、最小值、最大值 |
| `seed_sweep_boxplot.png`  | 多 seed 指标箱线图          |

### 13.3 评估结果

评估结果通常包含：

| 文件                              | 含义                                |
| ------------------------------- | --------------------------------- |
| `*_paired_compare_summary.json` | 配对评估摘要                            |
| `*_paired_compare_report.txt`   | 可读文本报告                            |
| `*_paired_cases.csv`            | 每个 scenario 的 baseline/RL 指标和扰动参数 |
| `*_rl_scales.csv`               | RL 输出的 PID/FF 缩放因子                |
| `*_paired_compare.npz`          | 数组形式保存的评估结果                       |
| `*_delta_return_hist.png`       | RL 相对 baseline 的 return 差异分布      |
| `*_delta_vs_tau_mix.png`        | return 差异与混合时间常数关系                |
| `*_delta_vs_delay.png`          | return 差异与观测延迟关系                  |
| `*_scale_boxplot.png`           | RL 输出缩放因子的分布                      |
| `*_scales_vs_tau_mix.png`       | 缩放因子与混合时间常数关系                     |

production 模式还会额外输出：

| 文件                               | 含义                 |
| -------------------------------- | ------------------ |
| `production_delta_vs_qs.png`     | return 差异与目标排量关系   |
| `production_delta_qs_bucket.png` | 不同排量区间下的 return 差异 |

---

## 14. 常用参数建议

### 14.1 训练参数

| 参数                 |    小规模调试 |      正式实验推荐 |
| ------------------ | -------: | ----------: |
| `--updates`        |     2–20 |   1000–3000 |
| `--batch_episodes` |      2–8 |      64–128 |
| `--eval_every`     |      1–5 |       20–50 |
| `--eval_episodes`  |     2–10 |     100–200 |
| `--save_every`     |      1–5 |      50–100 |
| `--num_workers`    |        0 |         4–8 |
| `--seeds`          | 1 个 seed | 至少 5 个 seed |

### 14.2 评估参数

| 参数                    | 小规模调试 |               正式实验推荐 |
| --------------------- | ----: | -------------------: |
| `--N`                 |  2–20 |             500–1000 |
| `--bootstrap_B`       |   100 |            2000–5000 |
| `--save_full_if_n_le` |  2–20 |                   20 |
| `--seed`              | 任意固定值 | 固定一个独立评估 seed，如 2026 |

### 14.3 仿真参数

| 参数                      | 含义                | 推荐                |
| ----------------------- | ----------------- | ----------------- |
| `--dt`                  | 仿真步长              | 推荐 0.5，并与论文实验保持一致 |
| `--premix_duration`     | premix 单次仿真时长     | 推荐 120 s          |
| `--premix_hold`         | premix 初始保持时长     | 推荐 5–10 s         |
| `--production_duration` | production 单次仿真时长 | 推荐 240 s          |

---



## 15. 注意事项

1. 所有训练和评估命令建议在项目根目录执行。
2. 推荐使用 `python -m scripts.rl.train` 和 `python -m scripts.rl.evaluate`，避免直接运行脚本文件。
3. 正式训练时建议使用多 seed，并保存每个 seed 的 best checkpoint。
4. 正式评估时建议使用与训练不同的固定评估 seed。
5. `premix` 与 `production` 的 `dt`、duration 等仿真参数应在训练和评估中保持一致。
6. 如果出现 `RuntimeWarning: Mean of empty slice`，通常表示某些指标在当前仿真设置下全为 NaN，例如 settling time 未达到阈值；该 warning 不一定表示程序失败。
7. 结果文件建议统一保存在 `outputs/` 下，避免污染源码目录。
