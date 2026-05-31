# train_modes.py
"""
train_modes.py

训练入口：分别训练两个 mode 的 agent（premix / production）。
- 复用 modes.py 的 ModeSpec（采样、上下文、reward、动作映射）
- 复用 PPO_bandit.py 的 PPOBanditAgent（一次决策 PPO）

本文件的关键点：
1) simulate_episode(): 对接你的 sim_config.py / sim_env.py / sim_model.py，跑一个完整 episode 并返回 Trajectory；
2) 训练时记录 loss/return 曲线并保存最优模型；
3) 对最优模型做 evaluation 并作图。

说明：
- 这里用“上下文老虎机(one-shot)”方式：每个 episode 固定一次性采样 PID/前馈缩放，然后整个 episode 内不再变化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import math
import json
import traceback

import argparse
import numpy as np
import torch
import multiprocessing as mp
from typing import Union
import random

from modes import (
    EpisodeParams,
    Trajectory,
    ModeSpec,
    make_premix_mode,
    make_production_mode,
    action_to_scales,
    premix_reward,
    production_reward
)

from PPO_bandit import PPOBanditAgent, PPOConfig, RolloutBuffer
from scripts.core.sim_config import PlantParams, ValveParams, SimulationConfig, PhysicalConstraintError
from scripts.core.sim_env import CementingSimEnv
from scripts.core.sim_model import SlurryState

# matplotlib 仅用于作图；若你希望无图形依赖，可删掉并自行实现保存逻辑
import matplotlib.pyplot as plt
from copy import deepcopy

# -----------------------------
# Simulator bridge
# -----------------------------
def _rollout_reward_worker(args) -> float:
    p, scales, base_params, override_dt, seed = args
    rrng = np.random.default_rng(int(seed))
    p = deepcopy(p)
    p.dt = float(override_dt)
    return float(simulate_episode(p, scales, rrng, base_params=base_params, return_traj=False))


def simulate_episode(
    p: EpisodeParams,
    scales: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
    return_traj: bool = False,
    *,
    base_params: Dict[str, float],
) -> Union[Trajectory, float]:
    """
    - return_traj=True : 返回 Trajectory（用于画图/对比/调试）
    - return_traj=False: 训练用，直接返回 reward（float），避免构造 Trajectory/多余字段
    """

    # -----------------------------
    # RNG
    # -----------------------------
    if rng is None:
        rng = np.random.default_rng()

    # -----------------------------
    # 1) apply scales to baseline params (episode-fixed)
    # -----------------------------
    Kp_w = base_params.get("Kp_w", 0.0) * scales.get("s_w_p", 1.0)
    Ki_w = base_params.get("Ki_w", 0.0) * scales.get("s_w_i", 1.0)
    Kd_w = base_params.get("Kd_w", 0.0) * scales.get("s_w_d", 1.0)

    Kp_c = base_params.get("Kp_c", 0.0) * scales.get("s_c_p", 1.0)
    Ki_c = base_params.get("Ki_c", 0.0) * scales.get("s_c_i", 1.0)
    Kd_c = base_params.get("Kd_c", 0.0) * scales.get("s_c_d", 1.0)

    ff_w_base = base_params.get("ff_w", 0.0)
    ff_c_base = base_params.get("ff_c", 0.0)
    kff_base = base_params.get("kff", 0.0)

    ff_w = ff_w_base
    ff_c = ff_c_base
    kff = kff_base

    # -----------------------------
    # 2) Build sim config/env
    # -----------------------------
    plant = PlantParams()

    # 根据p.tau_mix随机采样 plant.tau_mix_hat
    if hasattr(plant, "tau_mix_hat"):
        tau_mix = float(np.clip(rng.normal(float(p.tau_mix), float(p.tau_mix) * 0.25), 5.0, 100.0))
        plant.tau_mix_hat = tau_mix

    # 根据p.max_flow 随机采样
    wv_max_flow = float(np.clip(rng.normal(float(p.water_valve_max_flow), float(p.water_valve_max_flow) * 0.1), 1.5/60, 3.0/60))
    cv_max_flow = float(np.clip(rng.normal(float(p.cement_valve_max_flow), float(p.cement_valve_max_flow) * 0.1), 0.5/60, 2.0/60))
    # --- flow noise seeds: different per episode (derived from episode rng) ---
    flow_seed_w = int(rng.integers(0, 2 ** 31 - 1))
    flow_seed_c = int(rng.integers(0, 2 ** 31 - 1))

    water_valve_params = ValveParams(
        max_flow=wv_max_flow,
        flow_noise_enable=True,
        flow_noise_mode="mul",
        flow_noise_tau=1,
        flow_noise_std=0.01,
        flow_noise_seed=flow_seed_w,
    )
    cement_valve_params = ValveParams(
        max_flow=cv_max_flow,
        flow_noise_enable=True,
        flow_noise_mode="mul",
        flow_noise_tau=1,
        flow_noise_std=0.1,
        flow_noise_seed=flow_seed_c,
    )

    tau_delay = float(np.clip(rng.normal(float(p.tau_delay), float(p.tau_delay) * 0.25), 0.0, 20.0))

    qs = float(p.qs) if p.mode == "production" else 0.0

    cfg = SimulationConfig(
        dt=float(p.dt),
        t_end=float(p.duration_s),
        h_sp=float(p.h_sp),
        rho_sp=float(p.rho_sp),
        Qs_nominal=float(qs),
        h_obs_delay=0.0,
        rho_obs_delay=tau_delay,
        enable_logger=False,
        log_to_csv=False,
        control_mode=("siso-density" if p.mode == "premix" else "mimo"),
        h_pid_kp=float(Kp_w),
        h_pid_ki=float(Ki_w),
        h_pid_kd=float(Kd_w),
        rho_pid_kp=float(Kp_c),
        rho_pid_ki=float(Ki_c),
        rho_pid_kd=float(Kd_c),
        use_h_feedforward=True,
        use_density_feedforward=True,
        use_kff_decoupler=False,
        kff=float(kff),
        use_smith_decoupler=False,
    )

    if ff_w_base > 0.0:
        cfg.water_opening_ff = float(ff_w)
    if ff_c_base > 0.0:
        cfg.cement_opening_ff = float(ff_c)

    init_state = SlurryState()
    if hasattr(init_state, "h"):
        init_state.h = float(p.h0)
    if hasattr(init_state, "rho_out"):
        init_state.rho_out = float(p.rho0)

    env = CementingSimEnv(
        plant_params=plant,
        water_valve_params=water_valve_params,
        cement_valve_params=cement_valve_params,
        config=cfg,
        initial_slurry_state=init_state,
    )

    def _dump_physical_crash(tag: str, payload: dict) -> None:
        os.makedirs("crash_logs", exist_ok=True)
        path = os.path.join("crash_logs", f"{tag}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # 这一条 episode 的“定位用元信息”（尽量全）
    episode_meta = {
        "mode": p.mode,
        "dt": float(p.dt),
        "T": float(p.duration_s),
        "h0": float(p.h0),
        "h_sp": float(p.h_sp),
        "rho0": float(p.rho0),
        "rho_sp": float(p.rho_sp),
        "qs": float(p.qs),
        "tau_mix": float(p.tau_mix),
        "tau_delay": float(p.tau_delay),
        "water_valve_max_flow": float(p.water_valve_max_flow),
        "cement_valve_max_flow": float(p.cement_valve_max_flow),
        "base_params": dict(base_params),
        "scales": dict(scales),
        # cfg 里关键字段也记一下（避免 dataclass 里有对象无法 json）
        "cfg": {
            "Qs_nominal": float(cfg.Qs_nominal),
            "control_mode": str(cfg.control_mode),
            "h_sp": float(cfg.h_sp),
            "rho_sp": float(cfg.rho_sp),
            "use_density_feedforward": bool(cfg.use_density_feedforward),
            "use_kff_decoupler": bool(cfg.use_kff_decoupler),
            "kff": float(cfg.kff),
            "water_opening_ff_cfg": float(getattr(cfg, "water_opening_ff", 0.0)),
            "cement_opening_ff_cfg": float(getattr(cfg, "cement_opening_ff", 0.0)),
        },
        "plant": {
            "A": float(plant.tank_cross_section_area),
            "h_min": float(getattr(plant, "h_min", 0.0)),
        },
        "valve_params": {
            "wv_max_flow": float(water_valve_params.max_flow),
            "cv_max_flow": float(cement_valve_params.max_flow),
            "dead_zone": float(getattr(water_valve_params, "dead_zone_opening", 5.0)),
        },
    }


    # -----------------------------
    # 3) Rollout
    # -----------------------------
    if p.mode == "production":
        env.Q_s = float(p.qs)  # 从 t=0 起固定工作排量

    dt = float(p.dt)
    T = float(p.duration_s)
    n_steps = int(math.floor(T / dt)) + 1

    # ---- training fast path: return reward only ----
    if not return_traj:

        # 只存 reward 需要的序列
        rho_hist = np.empty(n_steps, dtype=np.float32)
        rho_hist[0] = float(getattr(env.state, "rho_out", p.rho0))

        h_hist = None
        if p.mode == "production":
            h_hist = np.empty(n_steps, dtype=np.float32)
            h_hist[0] = float(getattr(env.state, "h", p.h0))

        for k in range(1, n_steps):
            t = k * dt
            # env.step(None, t)
            try:
                env.step(None, t)
            except PhysicalConstraintError as e:
                # 记录“出错那一刻”的状态/阀门/流量/ff
                crash = dict(episode_meta)
                crash.update({
                    "k": int(k),
                    "t": float(t),
                    "err": str(e),
                    "traceback": traceback.format_exc(),
                    "state_before_step": {
                        "h": float(getattr(env.state, "h", float("nan"))),
                        "rho_out": float(getattr(env.state, "rho_out", float("nan"))),
                        "x": float(getattr(env.state, "x", float("nan"))),
                        "M": float(getattr(env.state, "M", float("nan"))),
                    },
                    "Q": {
                        "Q_s": float(getattr(env, "Q_s", float("nan"))),
                        "Q_w": float(getattr(env.water_valve, "current_flow", float("nan"))),
                        "Q_c": float(getattr(env.cement_valve, "current_flow", float("nan"))),
                    },
                    "valves": {
                        "water_target_opening": float(getattr(env.water_valve, "target_opening", float("nan"))),
                        "water_current_opening": float(getattr(env.water_valve, "current_opening", float("nan"))),
                        "cement_target_opening": float(getattr(env.cement_valve, "target_opening", float("nan"))),
                        "cement_current_opening": float(getattr(env.cement_valve, "current_opening", float("nan"))),
                    },
                    "ff_in_env": {
                        "water_opening_ff": float(getattr(env, "water_opening_ff", float("nan"))),
                        "cement_opening_ff": float(getattr(env, "cement_opening_ff", float("nan"))),
                    },
                })
                _dump_physical_crash("physical_constraint", crash)

                # 训练不中断：给一个很差的 reward（或者也可以 return -1e9）
                return -10


            rho_hist[k] = float(getattr(env.state, "rho_out", rho_hist[k - 1]))
            if h_hist is not None:
                h_hist[k] = float(getattr(env.state, "h", h_hist[k - 1]))

        # 关键：直接调用 modes.py 里的 reward（你已经替换成 series 版）
        # 下面两行按你的 modes.py 新签名选择其一：
        if p.mode == "premix":
            return float(premix_reward(rho_hist, p))
        else:
            return float(production_reward(rho_hist, h_hist, p))

    # ---- return_traj path: keep full trajectory for plots/debug ----
    t_hist = np.zeros(n_steps, dtype=np.float32)
    h_hist = np.zeros(n_steps, dtype=np.float32)
    rho_hist = np.zeros(n_steps, dtype=np.float32)
    uw_hist = np.zeros(n_steps, dtype=np.float32)
    uc_hist = np.zeros(n_steps, dtype=np.float32)

    t_hist[0] = 0.0
    h_hist[0] = float(getattr(env.state, "h", p.h0))
    rho_hist[0] = float(getattr(env.state, "rho_out", p.rho0))
    uw_hist[0] = float(getattr(env.water_valve, "current_opening", 0.0))
    uc_hist[0] = float(getattr(env.cement_valve, "current_opening", 0.0))

    for k in range(1, n_steps):
        t = k * dt
        env.step(None, t)

        t_hist[k] = float(t)
        h_hist[k] = float(getattr(env.state, "h", h_hist[k - 1]))
        rho_hist[k] = float(getattr(env.state, "rho_out", rho_hist[k - 1]))
        uw_hist[k] = float(getattr(env.water_valve, "current_opening", uw_hist[k - 1]))
        uc_hist[k] = float(getattr(env.cement_valve, "current_opening", uc_hist[k - 1]))

    return Trajectory(t=t_hist, rho=rho_hist, h=h_hist, u_w=uw_hist, u_c=uc_hist)



# -----------------------------
# Training loop helpers
# -----------------------------

@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 0
    num_workers: int = 12
    mp_start: str = "spawn"
    mp_chunksize: int = 8

    # 每轮收集多少个 episode（一次 PPO update）
    batch_episodes: int = 256
    # 训练多少轮 update
    updates: int = 2000

    # checkpoint
    ckpt_dir: str = "./checkpoints"
    save_every: int = 50  # 每多少次 update 保存一次

    # evaluation
    eval_episodes: int = 2000
    eval_every: int = 200  # 每多少次 update 评估一次（用于挑 best）；<=0 则只在末尾评估
    best_metric: str = "ema_return"  # "ema_return" or "mean_return"

    # plotting
    plots_dirname: str = "plots"



def collect_batch(
    mode: ModeSpec,
    agent: PPOBanditAgent,
    rng: np.random.Generator,
    cfg: TrainConfig,
    *,
    override_dt: float = 0.5,
    pool: Optional[mp.pool.Pool] = None,
) -> RolloutBuffer:
    """收集一批 episodes（并行 rollout），填充 RolloutBuffer（1-step bandit PPO）。"""

    # 维度初始化
    obs_dim = mode.build_context(mode.sample_episode(rng)).shape[0]
    act_dim = len(mode.action_specs)
    buf = RolloutBuffer(obs_dim=obs_dim, act_dim=act_dim)

    # 1) 一次性采样 batch 的 EpisodeParams + obs
    ps: List[EpisodeParams] = []
    obs_list: List[np.ndarray] = []
    for _ in range(int(cfg.batch_episodes)):
        p = mode.sample_episode(rng)
        p.dt = float(override_dt)
        ps.append(p)
        obs_list.append(mode.build_context(p))

    obs_batch = np.stack(obs_list, axis=0).astype(np.float32, copy=False)

    # 2) 一次性前向（GPU 友好）
    if hasattr(agent, "act_batch"):
        a_batch, logp_batch, v_batch = agent.act_batch(obs_batch)
    else:
        # fallback（不推荐）
        aL, lpL, vL = [], [], []
        for i in range(obs_batch.shape[0]):
            a, lp, v = agent.act(obs_batch[i])
            aL.append(a); lpL.append(lp); vL.append(v)
        a_batch = np.stack(aL, axis=0)
        logp_batch = np.asarray(lpL, dtype=np.float32)
        v_batch = np.asarray(vL, dtype=np.float32)

    # 3) action -> scales（每个 episode 一个 dict）
    scales_list: List[Dict[str, float]] = []
    base_params_list: List[Dict[str, float]] = []
    seeds: np.ndarray = rng.integers(0, 2**32 - 1, size=obs_batch.shape[0], dtype=np.uint32)

    for i in range(obs_batch.shape[0]):
        scales = action_to_scales(a_batch[i], mode.action_specs)
        scales_list.append(scales)
        base_params_list.append(mode.compute_base_params(ps[i]))

    # 4) 并行 rollout：simulate_episode(..., return_traj=False) -> reward(float)
    #    你需要把 _rollout_reward_worker 改成接收 (p, scales, base_params, override_dt, seed)
    #    并在 worker 里做 rng = default_rng(seed) 后调用 simulate_episode(...)
    worker_args = [
        (ps[i], scales_list[i], base_params_list[i], float(override_dt), int(seeds[i]))
        for i in range(obs_batch.shape[0])
    ]

    if pool is None:
        # 不并行：用于调试
        rews = []
        for (p, scales, base_params, odt, seed) in worker_args:
            rrng = np.random.default_rng(seed)
            p.dt = float(odt)
            rews.append(float(simulate_episode(p, scales, rrng, return_traj=False, base_params=base_params)))
    else:
        rews = pool.map(_rollout_reward_worker, worker_args, chunksize=int(cfg.mp_chunksize))

    rews_np = np.asarray(rews, dtype=np.float32)

    # 5) 填 buffer（1-step）
    for i in range(obs_batch.shape[0]):
        buf.add(
            obs_batch[i],
            a_batch[i],
            float(logp_batch[i]),
            float(v_batch[i]),
            float(rews_np[i]),
        )

    return buf



@torch.no_grad()
def evaluate_agent(
    mode: ModeSpec,
    agent: PPOBanditAgent,
    rng: np.random.Generator,
    cfg: TrainConfig,
    *,
    override_dt: float = 0.5,
    episodes: int = 200,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Deterministic action evaluation, but environment parameters/noise are stochastic (seeded by rng)."""
    rews: List[float] = []
    for _ in range(int(episodes)):
        p = mode.sample_episode(rng)
        p.dt = override_dt
        obs = mode.build_context(p)

        # deterministic action: use mean if supported
        if hasattr(agent, "act_deterministic"):
            a, _, _ = agent.act_deterministic(obs)
        else:
            a, _, _ = agent.act(obs)

        scales = action_to_scales(a, mode.action_specs)
        base_params = mode.compute_base_params(p)
        R = float(simulate_episode(p, scales, rng, return_traj=False, base_params=base_params))
        rews.append(R)

    rews_np = np.array(rews, dtype=np.float32)
    metrics = {
        "R_mean": float(np.mean(rews_np)),
        "R_std": float(np.std(rews_np)),
        "R_p10": float(np.percentile(rews_np, 10)),
        "R_p50": float(np.percentile(rews_np, 50)),
        "R_p90": float(np.percentile(rews_np, 90)),
    }
    return metrics, rews_np

def _compute_ctrl_metrics_from_traj(traj: Trajectory, p: EpisodeParams) -> Dict[str, float]:
    # 与 evaluate_compare.compute_basic_metrics 对齐（这里不引入它，避免循环依赖）
    t = traj.t.astype(np.float64)
    dt = float(p.dt)
    rho = traj.rho.astype(np.float64)
    h = traj.h.astype(np.float64) if p.mode == "production" else None

    iae_rho = float(np.sum(np.abs(rho - float(p.rho_sp))) * dt)
    rho_os = float(np.max(rho) - float(p.rho_sp))
    tv_uc = float(np.sum(np.abs(np.diff(traj.u_c.astype(np.float64)))))

    out = {
        "IAE_rho": iae_rho,
        "overshoot_rho": rho_os,
        "TV_uc": tv_uc,
    }
    if p.mode == "production" and h is not None:
        iae_h = float(np.sum(np.abs(h - float(p.h_sp))) * dt)
        h_os = float(np.max(h) - float(p.h_sp))
        tv_uw = float(np.sum(np.abs(np.diff(traj.u_w.astype(np.float64)))))
        out.update({"IAE_h": iae_h, "overshoot_h": h_os, "TV_uw": tv_uw})
    return out


@torch.no_grad()
def evaluate_agent_with_metrics(
    mode: ModeSpec,
    agent: PPOBanditAgent,
    rng: np.random.Generator,
    cfg: TrainConfig,
    *,
    override_dt: float = 0.5,
    episodes: int = 200,
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    """
    返回：
      - metrics_return: return 的统计
      - metrics_ctrl  : 控制工程指标（均值）
      - rews_np       : 每条episode的return数组（用于直方图/boxplot）
    """
    rews: List[float] = []
    ctrl_acc: Dict[str, List[float]] = {}

    for _ in range(int(episodes)):
        p = mode.sample_episode(rng)
        p.dt = float(override_dt)
        obs = mode.build_context(p)

        if hasattr(agent, "act_deterministic"):
            a, _, _ = agent.act_deterministic(obs)
        else:
            a, _, _ = agent.act(obs)

        scales = action_to_scales(a, mode.action_specs)
        base_params = mode.compute_base_params(p)

        traj = simulate_episode(p, scales, rng, return_traj=True, base_params=base_params)
        # 用现有 reward 函数计算（与你训练一致）
        if p.mode == "premix":
            R = float(premix_reward(traj.rho, p))
        else:
            R = float(production_reward(traj.rho, traj.h, p))
        rews.append(R)

        m = _compute_ctrl_metrics_from_traj(traj, p)
        for k, v in m.items():
            ctrl_acc.setdefault(k, []).append(float(v))

    rews_np = np.asarray(rews, dtype=np.float32)
    metrics_return = {
        "R_mean": float(np.mean(rews_np)),
        "R_std": float(np.std(rews_np)),
        "R_p10": float(np.percentile(rews_np, 10)),
        "R_p50": float(np.percentile(rews_np, 50)),
        "R_p90": float(np.percentile(rews_np, 90)),
    }
    metrics_ctrl = {k: float(np.mean(np.asarray(v, dtype=np.float64))) for k, v in ctrl_acc.items()}
    return metrics_return, metrics_ctrl, rews_np

def _save_checkpoint(
    path: str,
    mode: ModeSpec,
    agent: PPOBanditAgent,
    update: int,
) -> None:
    torch.save(
        {
            "mode": mode.name,
            "update": update,
            "state_dict": agent.state_dict(),
            "obs_dim": agent.cfg.obs_dim,
            "act_dim": agent.cfg.act_dim,
            "config": agent.cfg.__dict__,
        },
        path,
    )


def _ensure_plots_dir(ckpt_dir: str, plots_dirname: str) -> str:
    p = os.path.join(ckpt_dir, plots_dirname)
    os.makedirs(p, exist_ok=True)
    return p

def _save_metrics_npz(path: str, **arrays) -> str:
    """
    保存“指标序列”（小数据），不保存仿真轨迹。
    用 npz 方便后处理/复现画图，无需重跑训练。
    """
    pack = {}
    for k, v in arrays.items():
        if v is None:
            continue
        pack[k] = np.asarray(v)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **pack)
    print(f"Saved metrics npz: {path}")
    return path


def _plot_one_series(
    out_png: str,
    x: List[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> str:
    fig = plt.figure(figsize=(7, 4))
    plt.plot(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _plot_training_curves(
    mode_name: str,
    plots_dir: str,
    returns_mean: List[float],
    returns_ema: List[float],
    loss_pi: List[float],
    loss_v: List[float],
    ent: List[float],
) -> List[str]:
    outs: List[str] = []

    outs.append(
        _plot_one_series(
            os.path.join(plots_dir, f"{mode_name}_train_R_mean.png"),
            returns_mean,
            xlabel="update",
            ylabel="R_mean",
            title=f"{mode_name}: train R_mean",
        )
    )
    outs.append(
        _plot_one_series(
            os.path.join(plots_dir, f"{mode_name}_train_R_ema.png"),
            returns_ema,
            xlabel="update",
            ylabel="R_ema",
            title=f"{mode_name}: train R_ema",
        )
    )
    outs.append(
        _plot_one_series(
            os.path.join(plots_dir, f"{mode_name}_train_loss_pi.png"),
            loss_pi,
            xlabel="update",
            ylabel="loss_pi",
            title=f"{mode_name}: train loss_pi",
        )
    )
    outs.append(
        _plot_one_series(
            os.path.join(plots_dir, f"{mode_name}_train_loss_v.png"),
            loss_v,
            xlabel="update",
            ylabel="loss_v",
            title=f"{mode_name}: train loss_v",
        )
    )
    outs.append(
        _plot_one_series(
            os.path.join(plots_dir, f"{mode_name}_train_entropy.png"),
            ent,
            xlabel="update",
            ylabel="entropy",
            title=f"{mode_name}: train entropy",
        )
    )

    return outs


def _plot_ctrl_metrics_curves(
    mode_name: str,
    plots_dir: str,
    eval_updates: List[int],
    series: Dict[str, List[float]],
) -> Dict[str, str]:
    """
    每个 metric 单独输出一张图：<mode>_eval_<metric>.png
    """
    outs: Dict[str, str] = {}
    if len(eval_updates) == 0:
        return outs

    x = np.asarray(eval_updates, dtype=np.int64)
    for k, v in series.items():
        if v is None or len(v) == 0:
            continue

        y = np.asarray(v, dtype=np.float64)
        out = os.path.join(plots_dir, f"{mode_name}_eval_{k}.png")

        fig = plt.figure(figsize=(7, 4))
        plt.plot(x, y)
        plt.xlabel("update")
        plt.ylabel(k)
        plt.title(f"{mode_name}: eval {k}")
        plt.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)

        outs[k] = out

    return outs



def _plot_eval_results(
    mode_name: str,
    plots_dir: str,
    rewards: np.ndarray,
) -> str:
    fig = plt.figure()
    plt.hist(rewards, bins=40)
    plt.xlabel("episode return")
    plt.ylabel("count")
    plt.title(f"{mode_name}: evaluation return histogram")
    out = os.path.join(plots_dir, f"{mode_name}_eval_hist.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def train_one_mode(
    mode: ModeSpec,
    agent: PPOBanditAgent,
    cfg: TrainConfig,
    *,
    override_dt: float = 0.5,
) -> None:
    """训练一个 mode 对应的 agent，并保存最优模型 + 作图。"""
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    plots_dir = _ensure_plots_dir(cfg.ckpt_dir, cfg.plots_dirname)

    # -----------------------------
    # Global seeding (for reproducibility)
    # -----------------------------
    seed = int(cfg.seed)

    # Python / NumPy
    random.seed(seed)
    np.random.seed(seed)  # legacy global RNG (some libs may use it)
    rng_train = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 1234)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional: stricter determinism (may reduce speed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception:
    #     pass


    ema_return: Optional[float] = None
    beta = 0.98

    returns_mean: List[float] = []
    returns_ema: List[float] = []
    loss_pi: List[float] = []
    loss_v: List[float] = []
    ent: List[float] = []

    eval_updates: List[int] = []
    eval_R_mean: List[float] = []
    eval_IAE_rho: List[float] = []
    eval_OS_rho: List[float] = []
    eval_TV_uc: List[float] = []
    eval_IAE_h: List[float] = []
    eval_OS_h: List[float] = []
    eval_TV_uw: List[float] = []

    best_score = -1e18
    best_path = os.path.join(cfg.ckpt_dir, f"{mode.name}_best.pt")

    # ---- create multiprocessing pool once (spawn) ----
    pool: Optional[mp.pool.Pool] = None
    if int(cfg.num_workers) > 1:
        ctx = mp.get_context(cfg.mp_start)
        pool = ctx.Pool(processes=int(cfg.num_workers), maxtasksperchild=200)

    try:
        for u in range(1, cfg.updates + 1):
            buf = collect_batch(
                mode,
                agent,
                rng_train,
                cfg,
                override_dt=override_dt,
                pool=pool,
            )

            rews = np.array(buf._rew, dtype=np.float32)
            avg_R = float(np.mean(rews))

            if ema_return is None:
                ema_return = avg_R
            else:
                ema_return = beta * ema_return + (1.0 - beta) * avg_R

            losses = agent.update(buf)

            returns_mean.append(avg_R)
            returns_ema.append(float(ema_return))
            loss_pi.append(float(losses.get("policy", np.nan)))
            loss_v.append(float(losses.get("value", np.nan)))
            ent.append(float(losses.get("entropy", np.nan)))

            if u % 10 == 0:
                print(
                    f"[{mode.name}] update={u:05d}  "
                    f"R_mean={avg_R:+.4f}  R_ema={ema_return:+.4f}  "
                    f"loss_pi={loss_pi[-1]:.4f}  loss_v={loss_v[-1]:.4f}"
                )

            if u % cfg.save_every == 0:
                path = os.path.join(cfg.ckpt_dir, f"{mode.name}_ppo_bandit_{u:05d}.pt")
                _save_checkpoint(path, mode, agent, u)
                print(f"Saved checkpoint: {path}")

            do_eval = (cfg.eval_every > 0) and (u % cfg.eval_every == 0)
            if do_eval:
                m_ret, m_ctrl, _ = evaluate_agent_with_metrics(
                    mode, agent, rng_eval, cfg, override_dt=override_dt, episodes=min(300, cfg.eval_episodes)
                )

                eval_updates.append(int(u))
                eval_R_mean.append(float(m_ret["R_mean"]))
                eval_IAE_rho.append(float(m_ctrl.get("IAE_rho", np.nan)))
                eval_OS_rho.append(float(m_ctrl.get("overshoot_rho", np.nan)))
                eval_TV_uc.append(float(m_ctrl.get("TV_uc", np.nan)))

                if mode.name == "production":
                    eval_IAE_h.append(float(m_ctrl.get("IAE_h", np.nan)))
                    eval_OS_h.append(float(m_ctrl.get("overshoot_h", np.nan)))
                    eval_TV_uw.append(float(m_ctrl.get("TV_uw", np.nan)))


                score = m_ret["R_mean"]
                if cfg.best_metric == "ema_return":
                    score = float(ema_return)

                if score > best_score:
                    best_score = score
                    _save_checkpoint(best_path, mode, agent, u)
                    print(f"[BEST] update={u:05d}  score={best_score:+.4f}  -> {best_path}")

        if cfg.eval_every <= 0:
            best_score = returns_ema[-1]
            _save_checkpoint(best_path, mode, agent, cfg.updates)
            print(f"[BEST] (by last ema) score={best_score:+.4f} -> {best_path}")

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    outs_train = _plot_training_curves(mode.name, plots_dir, returns_mean, returns_ema, loss_pi, loss_v, ent)
    print(f"Saved training plots ({len(outs_train)}):")
    for p in outs_train:
        print("  ", p)

    ckpt = torch.load(best_path, map_location="cpu")
    agent.load_state_dict(ckpt["state_dict"])
    metrics, rews = evaluate_agent(
        mode, agent, rng_eval, cfg, override_dt=override_dt, episodes=cfg.eval_episodes
    )
    out_eval = _plot_eval_results(mode.name, plots_dir, rews)
    print(f"[{mode.name}] BEST evaluation metrics: {metrics}")
    print(f"Saved evaluation plot: {out_eval}")

    # -----------------------------
    # Save run summary (AFTER ckpt+metrics exist)
    # -----------------------------
    summary = {
        "mode": mode.name,
        "seed": int(cfg.seed),
        "best_path": best_path,
        "best_update": int(ckpt.get("update", -1)),
        "eval_metrics": metrics,
    }
    summary_path = os.path.join(
        cfg.ckpt_dir,
        cfg.plots_dirname,
        f"{mode.name}_seed{cfg.seed}_summary.json",
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")

    outs_ctrl = _plot_ctrl_metrics_curves(
        mode.name, plots_dir, eval_updates,
        series={
            "R_mean": eval_R_mean,
            "IAE_rho": eval_IAE_rho,
            "OS_rho": eval_OS_rho,
            "TV_uc": eval_TV_uc,
            **({"IAE_h": eval_IAE_h, "OS_h": eval_OS_h, "TV_uw": eval_TV_uw} if mode.name == "production" else {}),
        }
    )
    print(f"Saved eval ctrl-metrics plots ({len(outs_ctrl)}):")
    for k, p in outs_ctrl.items():
        print(f"  {k}: {p}")

    # -----------------------------
    # Save metric time-series (NO trajectories)
    # -----------------------------
    metrics_npz_path = os.path.join(
        cfg.ckpt_dir,
        cfg.plots_dirname,
        f"{mode.name}_seed{cfg.seed}_metrics_series.npz",
    )

    _save_metrics_npz(
        metrics_npz_path,
        # training per-update series
        train_returns_mean=returns_mean,
        train_returns_ema=returns_ema,
        train_loss_pi=loss_pi,
        train_loss_v=loss_v,
        train_entropy=ent,

        # eval-over-training series
        eval_updates=eval_updates,
        eval_R_mean=eval_R_mean,
        eval_IAE_rho=eval_IAE_rho,
        eval_OS_rho=eval_OS_rho,
        eval_TV_uc=eval_TV_uc,
        eval_IAE_h=(eval_IAE_h if mode.name == "production" else []),
        eval_OS_h=(eval_OS_h if mode.name == "production" else []),
        eval_TV_uw=(eval_TV_uw if mode.name == "production" else []),

        # best final evaluation reward distribution
        best_eval_rewards=rews,
    )


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1, help="Random seed for training/eval")
    ap.add_argument("--mode", type=str, default="both", choices=["premix", "production", "both"],
                    help="Which mode to train")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--batch_episodes", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_episodes", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    # # (optional, stricter determinism; may reduce speed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # 1) create modes
    premix = make_premix_mode(duration_default=400.0, hold_default=10.0)
    prod = make_production_mode(duration_default=600.0)

    # 2) build agents (each mode one agent) - obs_dim depends on build_context
    rng = np.random.default_rng(args.seed)  # <-- use seed here
    obs_pre = premix.build_context(premix.sample_episode(rng))
    obs_prod = prod.build_context(prod.sample_episode(rng))

    cfg_pre = PPOConfig(obs_dim=int(obs_pre.shape[0]), act_dim=len(premix.action_specs))
    cfg_prod = PPOConfig(obs_dim=int(obs_prod.shape[0]), act_dim=len(prod.action_specs))

    device = args.device
    agent_pre = PPOBanditAgent(cfg_pre, device=device)
    agent_prod = PPOBanditAgent(cfg_prod, device=device)

    train_cfg = TrainConfig(
        device=device,
        seed=int(args.seed),                 # <-- use seed here
        batch_episodes=int(args.batch_episodes),
        updates=int(args.updates),
        ckpt_dir=str(args.ckpt_dir),
        save_every=int(args.save_every),
        eval_episodes=int(args.eval_episodes),
        eval_every=int(args.eval_every),
        best_metric="ema_return",
    )

    # 3) train
    # if args.mode in ("premix", "both"):
    #     print(f"=== Train premix agent (seed={args.seed}) ===")
    #     train_one_mode(
    #         premix,
    #         agent_pre,
    #         train_cfg,
    #         override_dt=0.5,
    #     )

    if args.mode in ("production", "both"):
        print(f"=== Train production agent (seed={args.seed}) ===")
        train_one_mode(
            prod,
            agent_prod,
            train_cfg,
            override_dt=0.5,
        )


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()


#python train_modes.py --mode production --seed 1 --updates 500 --batch_episodes 128 --eval_every 100 --eval_episodes 100 --ckpt_dir ./checkpoints --device cuda

#./checkpoints/plots/ 下保存训练曲线、评估直方图、summary JSON、以及 metrics_series.npz