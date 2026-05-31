
# evaluate_compare.py
"""
evaluate_compare.py

Compare baseline (coarse-tuned PID/FF) vs PPO-bandit scaled controller using a saved *_best.pt.

Features
--------
1) Load a checkpoint saved by train_modes._save_checkpoint (expects keys:
   {"mode","update","state_dict","obs_dim","act_dim","config"}).

2) Fixed-seed Monte-Carlo evaluation over N sampled scenarios (EpisodeParams):
   - For each scenario p_i, evaluate:
       * Base: scales = 1 for all action dims
       * RL  : scales = action_to_scales( actor_mean(context) )
     Both runs share the SAME sampled (tau_mix_hat, tau_delay) perturbation to ensure a fair paired comparison.

3) If N <= 4: save per-scenario trajectory comparisons (h and rho subplots).
   Else: run statistical analysis and save evidence plots:
      - paired deltas (RL - Base) summary: mean/median/std, bootstrap CI
      - paired significance: Wilcoxon signed-rank (fallback to sign test if SciPy missing)
      - histogram of deltas, scatter vs tau_mix and |ΔQs|
      - bucket plots (quantile bins) for tau_mix and |ΔQs|

Usage
-----
Minimal (edit the constants below if desired):
    python evaluate_compare.py

Optional CLI args:
    python evaluate_compare.py --ckpt ./checkpoints/production_best.pt --N 1000 --seed 0

Notes
-----
- Deterministic policy action uses actor mean (no sampling noise), then clamps to [-1,1].
- Simulation noise (uncertainty) is controlled: tau_mix_hat and tau_delay are sampled ONCE per scenario
  using eval_rng and applied to BOTH Base and RL runs.

"""

from __future__ import annotations

# from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
import os
import math
import argparse

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
import re

from scripts.rl.modes import (
    EpisodeParams,
    Trajectory,
    ModeSpec,
    make_premix_mode,
    make_production_mode,
    action_to_scales,
    premix_reward,
    production_reward
)

from scripts.rl.PPO_bandit import PPOBanditAgent, PPOConfig

from scripts.core.sim_config import PlantParams, ValveParams, SimulationConfig, PhysicalConstraintError
from scripts.core.sim_env import CementingSimEnv
from scripts.core.sim_model import SlurryState
from scripts.PID_control.tune_baseline import opening_to_flow
from copy import deepcopy

# -----------------------------
# Helpers: deterministic action
# -----------------------------

@torch.no_grad()
def act_deterministic(agent: PPOBanditAgent, obs: np.ndarray) -> np.ndarray:
    """
    Deterministic action for evaluation:
    - Prefer the agent's own deterministic path (matches train_modes.evaluate_agent)
    - Fall back to agent.act() if deterministic API not present
    """
    if hasattr(agent, "act_deterministic"):
        a, _, _ = agent.act_deterministic(obs)
        return np.asarray(a, dtype=np.float32)
    else:
        a, _, _ = agent.act(obs)
        return np.asarray(a, dtype=np.float32)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _bootstrap_ci(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    stat: str = "mean",
    B: int = 2000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Basic bootstrap CI for a 1D array.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n <= 1:
        v = float(x[0]) if n == 1 else float("nan")
        return v, v

    stats = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        if stat == "mean":
            stats[b] = float(np.mean(xb))
        elif stat == "median":
            stats[b] = float(np.median(xb))
        else:
            raise ValueError(f"Unknown stat: {stat}")

    lo = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _paired_pvalue_wilcoxon(d: np.ndarray) -> Tuple[str, float]:
    """
    Paired significance test for delta (RL - Base).
    Prefer Wilcoxon signed-rank (SciPy). Fallback to sign test if SciPy not available.
    Returns (method_name, p_value).
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return "wilcoxon", float("nan")

    try:
        from scipy.stats import wilcoxon  # type: ignore
        # zero_method='wilcox' drops zeros; alternative is 'pratt'
        res = wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="auto")
        return "wilcoxon", float(res.pvalue)
    except Exception:
        # Sign test: p = 2*min(P(X<=k), P(X>=k)) for X~Bin(n,0.5)
        n = int(d.size)
        k_pos = int(np.sum(d > 0))
        # compute exact binomial tail (small n ok)
        from math import comb
        probs = np.array([comb(n, k) for k in range(n + 1)], dtype=np.float64) / (2.0 ** n)
        cdf = float(np.sum(probs[: k_pos + 1]))
        sf = float(np.sum(probs[k_pos:]))
        p = 2.0 * min(cdf, sf)
        p = float(min(1.0, max(0.0, p)))
        return "sign_test", p




def _make_sim_config_filtered(**kwargs):
    """
    Create SimulationConfig while being tolerant to config field name differences
    across versions (filters kwargs by dataclass fields if possible).
    """
    try:
        fields = {f.name for f in getattr(SimulationConfig, "__dataclass_fields__", {}).values()}
        if fields:
            kwargs = {k: v for k, v in kwargs.items() if k in fields}
    except Exception:
        pass
    return SimulationConfig(**kwargs)


# -----------------------------
# Simulator bridge (fair paired comparison)
# -----------------------------

def simulate_episode_with_uncertainty(
    p: EpisodeParams,
    scales: Dict[str, float],
    *,
    base_params: Dict[str, float],
    tau_mix_hat: float,
    tau_delay: float,
    wv_max_flow: float,
    cv_max_flow: float,
    wv_noise_seed: int,
    cv_noise_seed: int,
) -> Trajectory:

    """
    Same logic as train_modes.simulate_episode, but:
    - (shared) tau_mix_hat and tau_delay are provided (sampled once per scenario),
      so Base and RL are compared under identical plant/measurement uncertainty.
    """
    # 1) apply scales to baseline params (episode-fixed)
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

    # 2) Build sim config/env
    plant = PlantParams()
    if hasattr(plant, "tau_mix_hat"):
        plant.tau_mix_hat = float(tau_mix_hat)

    water_valve = ValveParams(
        max_flow=float(wv_max_flow),
        flow_noise_enable=True,
        flow_noise_mode="mul",
        flow_noise_tau=1,
        flow_noise_std=0.01,
        flow_noise_seed=int(wv_noise_seed),
    )
    cement_valve = ValveParams(
        max_flow=float(cv_max_flow),
        flow_noise_enable=True,
        flow_noise_mode="mul",
        flow_noise_tau=1,
        flow_noise_std=0.1,
        flow_noise_seed=int(cv_noise_seed),
    )

    qs = float(p.qs) if p.mode == "production" else 0.0

    cfg = _make_sim_config_filtered(
        dt=float(p.dt),
        t_end=float(p.duration_s),
        h_sp=float(p.h_sp),
        rho_sp=float(p.rho_sp),
        Qs_nominal=float(qs),
        h_obs_delay=0.0,
        rho_obs_delay=float(tau_delay),
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
        water_valve_params=water_valve,
        cement_valve_params=cement_valve,
        config=cfg,
        initial_slurry_state=init_state,
    )

    # 3) Rollout
    n_steps = int(math.floor(p.duration_s / p.dt)) + 1
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
        t = k * p.dt
        if p.mode == "production":
            env.Q_s = float(p.qs)
        try:
            env.step(None, t)
        except PhysicalConstraintError:
            # 直接用最后状态填满剩余序列并 break（或 raise 给上层打 -inf）
            t_hist[k:] = t
            h_hist[k:] = h_hist[k - 1]
            rho_hist[k:] = rho_hist[k - 1]
            uw_hist[k:] = uw_hist[k - 1]
            uc_hist[k:] = uc_hist[k - 1]
            break

        t_hist[k] = float(t)
        h_hist[k] = float(getattr(env.state, "h", h_hist[k - 1]))
        rho_hist[k] = float(getattr(env.state, "rho_out", rho_hist[k - 1]))
        uw_hist[k] = float(getattr(env.water_valve, "current_opening", uw_hist[k - 1]))
        uc_hist[k] = float(getattr(env.cement_valve, "current_opening", uc_hist[k - 1]))

    return Trajectory(t=t_hist, rho=rho_hist, h=h_hist, u_w=uw_hist, u_c=uc_hist)


def simulate_episode_with_uncertainty_full(
    p: EpisodeParams,
    scales: Dict[str, float],
    *,
    base_params: Dict[str, float],
    tau_mix_hat: float,
    tau_delay: float,
    wv_max_flow: float,
    cv_max_flow: float,
    wv_noise_seed: int,
    cv_noise_seed: int,
) -> Tuple[Trajectory, Dict[str, np.ndarray]]:
    """
    Same as simulate_episode_with_uncertainty(), but also records full trajectories of flows.

    "波动前"(nominal): 根据阀门开度用 opening_to_flow() 计算的近似流量（不含噪声）。
    "波动后"(actual) : 仿真器中阀门的 current_flow（包含你注入的随机波动）。

    Returns
    -------
    traj : Trajectory
    extra: dict of np.ndarray with keys:
        Q_s, Q_w_nom, Q_w, Q_c_nom, Q_c
    """
    # 1) apply scales to baseline params (episode-fixed)
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

    # 2) Build sim config/env (match simulate_episode_with_uncertainty)
    plant = PlantParams()
    if hasattr(plant, "tau_mix_hat"):
        plant.tau_mix_hat = float(tau_mix_hat)

    water_valve = ValveParams(
        max_flow=float(wv_max_flow),
        flow_noise_enable=True,
        flow_noise_mode="mul",
        flow_noise_tau=1,
        flow_noise_std=0.01,
        flow_noise_seed=int(wv_noise_seed),
    )
    cement_valve = ValveParams(
        max_flow=float(cv_max_flow),
        flow_noise_enable=True,
        flow_noise_mode="mul",
        flow_noise_tau=1,
        flow_noise_std=0.1,
        flow_noise_seed=int(cv_noise_seed),
    )

    qs = float(p.qs) if p.mode == "production" else 0.0

    cfg = _make_sim_config_filtered(
        dt=float(p.dt),
        t_end=float(p.duration_s),
        h_sp=float(p.h_sp),
        rho_sp=float(p.rho_sp),
        Qs_nominal=float(qs),
        h_obs_delay=0.0,
        rho_obs_delay=float(tau_delay),
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
        water_valve_params=water_valve,
        cement_valve_params=cement_valve,
        config=cfg,
        initial_slurry_state=init_state,
    )

    # 3) Rollout + record
    n_steps = int(math.floor(p.duration_s / p.dt)) + 1
    t_hist = np.zeros(n_steps, dtype=np.float32)
    h_hist = np.zeros(n_steps, dtype=np.float32)
    rho_hist = np.zeros(n_steps, dtype=np.float32)
    uw_hist = np.zeros(n_steps, dtype=np.float32)
    uc_hist = np.zeros(n_steps, dtype=np.float32)

    qs_hist = np.zeros(n_steps, dtype=np.float32)
    qw_nom_hist = np.zeros(n_steps, dtype=np.float32)
    qc_nom_hist = np.zeros(n_steps, dtype=np.float32)
    qw_hist = np.zeros(n_steps, dtype=np.float32)
    qc_hist = np.zeros(n_steps, dtype=np.float32)

    def _read_flow(obj, default: float = 0.0) -> float:
        if hasattr(obj, "current_flow"):
            return float(getattr(obj, "current_flow"))
        if hasattr(obj, "flow"):
            return float(getattr(obj, "flow"))
        return float(default)

    t_hist[0] = 0.0
    h_hist[0] = float(getattr(env.state, "h", p.h0))
    rho_hist[0] = float(getattr(env.state, "rho_out", p.rho0))
    uw_hist[0] = float(getattr(env.water_valve, "current_opening", 0.0))
    uc_hist[0] = float(getattr(env.cement_valve, "current_opening", 0.0))

    qs_hist[0] = float(getattr(env, "Q_s", qs))
    qw_hist[0] = _read_flow(env.water_valve, 0.0)
    qc_hist[0] = _read_flow(env.cement_valve, 0.0)
    qw_nom_hist[0] = float(opening_to_flow(uw_hist[0], water_valve))
    qc_nom_hist[0] = float(opening_to_flow(uc_hist[0], cement_valve))

    for k in range(1, n_steps):
        t = k * p.dt
        if p.mode == "production":
            env.Q_s = float(p.qs)
        try:
            env.step(None, t)
        except PhysicalConstraintError:
            t_hist[k:] = t
            h_hist[k:] = h_hist[k - 1]
            rho_hist[k:] = rho_hist[k - 1]
            uw_hist[k:] = uw_hist[k - 1]
            uc_hist[k:] = uc_hist[k - 1]
            qs_hist[k:] = qs_hist[k - 1]
            qw_nom_hist[k:] = qw_nom_hist[k - 1]
            qc_nom_hist[k:] = qc_nom_hist[k - 1]
            qw_hist[k:] = qw_hist[k - 1]
            qc_hist[k:] = qc_hist[k - 1]
            break

        t_hist[k] = float(t)
        h_hist[k] = float(getattr(env.state, "h", h_hist[k - 1]))
        rho_hist[k] = float(getattr(env.state, "rho_out", rho_hist[k - 1]))
        uw_hist[k] = float(getattr(env.water_valve, "current_opening", uw_hist[k - 1]))
        uc_hist[k] = float(getattr(env.cement_valve, "current_opening", uc_hist[k - 1]))

        qs_hist[k] = float(getattr(env, "Q_s", qs_hist[k - 1]))
        qw_hist[k] = _read_flow(env.water_valve, qw_hist[k - 1])
        qc_hist[k] = _read_flow(env.cement_valve, qc_hist[k - 1])
        qw_nom_hist[k] = float(opening_to_flow(uw_hist[k], water_valve))
        qc_nom_hist[k] = float(opening_to_flow(uc_hist[k], cement_valve))

    traj = Trajectory(t=t_hist, rho=rho_hist, h=h_hist, u_w=uw_hist, u_c=uc_hist)
    extra = {"Q_s": qs_hist, "Q_w_nom": qw_nom_hist, "Q_w": qw_hist, "Q_c_nom": qc_nom_hist, "Q_c": qc_hist}
    return traj, extra


def _save_full_traj_csv(
    out_csv: str,
    traj: Trajectory,
    extra: Dict[str, np.ndarray],
    *,
    p: EpisodeParams,
    tag: str,
) -> None:
    """Save full time-series (including flows) to CSV for quick inspection."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    t = np.asarray(traj.t, dtype=np.float64)
    h = np.asarray(getattr(traj, "h", np.full_like(t, np.nan)), dtype=np.float64)
    rho = np.asarray(traj.rho, dtype=np.float64)
    u_w = np.asarray(getattr(traj, "u_w", np.full_like(t, np.nan)), dtype=np.float64)
    u_c = np.asarray(getattr(traj, "u_c", np.full_like(t, np.nan)), dtype=np.float64)

    Q_s = np.asarray(extra.get("Q_s", np.zeros_like(t)), dtype=np.float64)
    Q_w_nom = np.asarray(extra.get("Q_w_nom", np.zeros_like(t)), dtype=np.float64)
    Q_w = np.asarray(extra.get("Q_w", np.zeros_like(t)), dtype=np.float64)
    Q_c_nom = np.asarray(extra.get("Q_c_nom", np.zeros_like(t)), dtype=np.float64)
    Q_c = np.asarray(extra.get("Q_c", np.zeros_like(t)), dtype=np.float64)

    m3min = 60.0
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "tag", "mode", "dt", "duration_s",
            "t_s", "h", "rho",
            "u_w", "u_c",
            "Q_s_m3s", "Q_s_m3min",
            "Q_w_nom_m3s", "Q_w_nom_m3min",
            "Q_w_m3s", "Q_w_m3min",
            "Q_c_nom_m3s", "Q_c_nom_m3min",
            "Q_c_m3s", "Q_c_m3min",
        ])
        for i in range(t.size):
            w.writerow([
                tag, p.mode, float(p.dt), float(p.duration_s),
                float(t[i]), float(h[i]), float(rho[i]),
                float(u_w[i]), float(u_c[i]),
                float(Q_s[i]), float(Q_s[i] * m3min),
                float(Q_w_nom[i]), float(Q_w_nom[i] * m3min),
                float(Q_w[i]), float(Q_w[i] * m3min),
                float(Q_c_nom[i]), float(Q_c_nom[i] * m3min),
                float(Q_c[i]), float(Q_c[i] * m3min),
            ])


def apply_scales_to_params(
    base_params: Dict[str, float],
    scales: Dict[str, float],
) -> Dict[str, float]:
    """Return the *actual* params used after applying scales (for annotation)."""
    out: Dict[str, float] = {}

    out["Kp_w"] = base_params.get("Kp_w", 0.0) * scales.get("s_w_p", 1.0)
    out["Ki_w"] = base_params.get("Ki_w", 0.0) * scales.get("s_w_i", 1.0)
    out["Kd_w"] = base_params.get("Kd_w", 0.0) * scales.get("s_w_d", 1.0)

    out["Kp_c"] = base_params.get("Kp_c", 0.0) * scales.get("s_c_p", 1.0)
    out["Ki_c"] = base_params.get("Ki_c", 0.0) * scales.get("s_c_i", 1.0)
    out["Kd_c"] = base_params.get("Kd_c", 0.0) * scales.get("s_c_d", 1.0)

    out["ff_w"] = base_params.get("ff_w", 0.0)
    out["ff_c"] = base_params.get("ff_c", 0.0)
    out["kff"] = base_params.get("kff", 0.0)

    return out


# -----------------------------
# Metrics
# -----------------------------

def compute_basic_metrics(traj: Trajectory, p: EpisodeParams) -> Dict[str, float]:
    """
    Reward-agnostic metrics for interpretability.

    Notes:
      - Works for both premix and production.
      - If some trajectory fields are absent (e.g., h/u_w in premix), returns NaN for those metrics.
    """
    dt = float(p.dt)

    rho = np.asarray(traj.rho, dtype=np.float64)

    # optional fields (may not exist in premix trajectories depending on your implementation)
    h = getattr(traj, "h", None)
    u_w = getattr(traj, "u_w", None)
    u_c = getattr(traj, "u_c", None)

    h = None if h is None else np.asarray(h, dtype=np.float64)
    u_w = None if u_w is None else np.asarray(u_w, dtype=np.float64)
    u_c = None if u_c is None else np.asarray(u_c, dtype=np.float64)

    # ---- helpers ----
    def _settling_time(y: np.ndarray, sp: float, dt: float, band: float = 0.02, hold_seconds: float = 10.0) -> float:
        """
        First time the signal enters and stays within ±band*|sp| for hold_seconds.
        """
        eps = band * max(1e-6, abs(float(sp)))
        inside = np.abs(y - float(sp)) <= eps
        n = int(inside.size)
        hold = max(1, int(round(hold_seconds / dt)))
        for k in range(n):
            if k + hold <= n and np.all(inside[k:k + hold]):
                return float(k * dt)
        return float((n - 1) * dt)

    def _sat_ratio(u: np.ndarray, umin: float = 0.0, umax: float = 100.0, tol: float = 1e-6) -> float:
        u = np.asarray(u, dtype=np.float64)
        sat = (u <= (umin + tol)) | (u >= (umax - tol))
        return float(np.mean(sat))

    # ---- core metrics ----
    iae_rho = float(np.sum(np.abs(rho - float(p.rho_sp))) * dt)
    rho_os = float(np.max(rho) - float(p.rho_sp))
    Ts_rho = _settling_time(rho, p.rho_sp, dt, band=0.02, hold_seconds=10.0)

    # defaults for fields not applicable / not available
    iae_h = float("nan")
    h_os = float("nan")
    TV_uw = float("nan")
    sat_uw = float("nan")

    TV_uc = float("nan")
    sat_uc = float("nan")

    # cement opening metrics (usually exists in both modes; but guard anyway)
    if u_c is not None and u_c.size >= 2:
        TV_uc = float(np.sum(np.abs(np.diff(u_c))))
        sat_uc = _sat_ratio(u_c)

    # production-only metrics (or premix if you still record them)
    if h is not None:
        iae_h = float(np.sum(np.abs(h - float(p.h_sp))) * dt)
        h_os = float(np.max(h) - float(p.h_sp))

    if u_w is not None and u_w.size >= 2:
        TV_uw = float(np.sum(np.abs(np.diff(u_w))))
        sat_uw = _sat_ratio(u_w)

    return {
        "IAE_h": iae_h,
        "IAE_rho": iae_rho,
        "overshoot_h": h_os,
        "overshoot_rho": rho_os,
        "TV_uw": TV_uw,
        "TV_uc": TV_uc,
        "Ts_rho": Ts_rho,
        "sat_uw": sat_uw,
        "sat_uc": sat_uc,
    }



# -----------------------------
# Plotting
# -----------------------------

def plot_trajectory_compare(
    out_path: str,
    traj_base: Trajectory,
    traj_rl: Trajectory,
    p: EpisodeParams,
    *,
    title_prefix: str = "",
    params_base: Optional[Dict[str, float]] = None,
    params_rl: Optional[Dict[str, float]] = None,
) -> None:
    # ---- helper: pretty format ----
    def _fmt_params(d: Optional[Dict[str, float]], keys: List[str]) -> str:
        if not d:
            return "(none)"
        parts = []
        for k in keys:
            if k in d:
                parts.append(f"{k}={d[k]:.4g}")
        return ", ".join(parts) if parts else "(none)"

    # 你想展示的参数项（按你当前 simulate_episode_with_uncertainty 里会用到的）
    keys_prod = ["Kp_w","Ki_w","Kd_w","Kp_c","Ki_c","Kd_c","ff_w","ff_c","kff"]
    keys_premix = ["Kp_c","Ki_c","Kd_c","ff_c","kff"]  # premix 主要看密度回路即可（更清爽）

    # ---- plotting ----
    if p.mode == "premix":
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)

        # rho only
        ax.plot(traj_base.t, traj_base.rho, label="base")
        ax.plot(traj_rl.t, traj_rl.rho, label="RL")
        ax.axhline(float(p.rho_sp), linestyle="--", linewidth=1, label="rho_sp")
        ax.set_ylabel("rho (kg/m^3)")
        ax.set_xlabel("t (s)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)

        # parameter annotation
        txt = (
            "Base: " + _fmt_params(params_base, keys_premix) + "\n"
            "RL  : " + _fmt_params(params_rl, keys_premix)
        )
        ax.text(
            0.01, 0.99, txt,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.15),
        )

    else:
        # production: h + rho
        fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
        # h
        axes[0].plot(traj_base.t, traj_base.h, label="base")
        axes[0].plot(traj_rl.t, traj_rl.h, label="RL")
        axes[0].axhline(float(p.h_sp), linestyle="--", linewidth=1, label="h_sp")
        axes[0].set_ylabel("h (m)")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.2)

        # rho
        axes[1].plot(traj_base.t, traj_base.rho, label="base")
        axes[1].plot(traj_rl.t, traj_rl.rho, label="RL")
        axes[1].axhline(float(p.rho_sp), linestyle="--", linewidth=1, label="rho_sp")
        axes[1].set_ylabel("rho (kg/m^3)")
        axes[1].set_xlabel("t (s)")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.2)

        # NEW (2) u_w
        axes[2].plot(traj_base.t, traj_base.u_w, label="base")
        axes[2].plot(traj_rl.t, traj_rl.u_w, label="RL")
        axes[2].set_ylabel("u_w (%)")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.2)

        # NEW (3) u_c
        axes[3].plot(traj_base.t, traj_base.u_c, label="base")
        axes[3].plot(traj_rl.t, traj_rl.u_c, label="RL")
        axes[3].set_ylabel("u_c (%)")
        axes[3].set_xlabel("t (s)")
        axes[3].legend(loc="best")
        axes[3].grid(True, alpha=0.2)

        # parameter annotation (放在上图角落，避免挡住 rho)
        txt = (
            "Base: " + _fmt_params(params_base, keys_prod) + "\n"
            "RL  : " + _fmt_params(params_rl, keys_prod)
        )
        axes[0].text(
            0.01, 0.99, txt,
            transform=axes[0].transAxes, va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.15),
        )

    fig.suptitle(
        f"{title_prefix} scenario: mode={p.mode}, tau_mix={p.tau_mix:.1f}, tau_delay={p.tau_delay:.1f}, |Qs|={abs(p.qs):.4f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)



def plot_hist(out_path: str, x: np.ndarray, title: str, xlabel: str) -> None:
    fig = plt.figure()
    plt.hist(x, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.2)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(out_path: str, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    fig = plt.figure()
    plt.scatter(x, y, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bucket_bars(
    out_path: str,
    x: np.ndarray,
    d: np.ndarray,
    *,
    title: str,
    xlabel: str,
    nbins: int = 5,
    seed: int = 0,
) -> None:
    """
    Quantile-bin bucket plot of mean delta with bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    # compute quantile edges (ensure unique)
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(x, qs)
    # make edges strictly increasing by small eps if needed
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    means = []
    lo = []
    hi = []
    labels = []
    for b in range(nbins):
        mask = (x >= edges[b]) & (x <= edges[b + 1] if b == nbins - 1 else x < edges[b + 1])
        db = d[mask]
        if db.size == 0:
            means.append(0.0); lo.append(0.0); hi.append(0.0)
        else:
            means.append(float(np.mean(db)))
            ci = _bootstrap_ci(db, rng, stat="mean", B=1000, alpha=0.05)
            lo.append(ci[0]); hi.append(ci[1])
        labels.append(f"[{edges[b]:.2g},{edges[b+1]:.2g}]")

    x_pos = np.arange(nbins)
    fig = plt.figure(figsize=(10, 4))
    plt.bar(x_pos, means)
    # error bars
    yerr = np.vstack([np.array(means) - np.array(lo), np.array(hi) - np.array(means)])
    plt.errorbar(x_pos, means, yerr=yerr, fmt="none", capsize=3)
    plt.xticks(x_pos, labels, rotation=30, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("mean Δ return (RL - Base)")
    plt.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_scales_boxplot(out_path: str, scales_mat: np.ndarray, names: List[str], title: str) -> None:
    fig = plt.figure(figsize=(10, 4))
    plt.boxplot([scales_mat[:, i] for i in range(scales_mat.shape[1])], showfliers=False)
    plt.xticks(np.arange(1, len(names) + 1), names, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel("scale value")
    plt.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Main evaluation
# -----------------------------

def load_agent_from_ckpt(ckpt_path: str, device: str = "cuda") -> Tuple[str, PPOBanditAgent, Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mode_name = str(ckpt.get("mode", ""))
    cfg_dict = dict(ckpt.get("config", {}))
    # ensure obs_dim/act_dim present
    if "obs_dim" not in cfg_dict:
        cfg_dict["obs_dim"] = int(ckpt.get("obs_dim", 0))
    if "act_dim" not in cfg_dict:
        cfg_dict["act_dim"] = int(ckpt.get("act_dim", 0))

    cfg = PPOConfig(**cfg_dict)
    agent = PPOBanditAgent(cfg, device=device)
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return mode_name, agent, ckpt


def build_mode_from_name(mode_name: str) -> ModeSpec:
    name = mode_name.lower()
    if "premix" in name:
        return make_premix_mode(duration_default=400.0, hold_default=10.0)
    if "production" in name or "prod" in name:
        return make_production_mode(duration_default=600.0)
    # fallback: try both
    return make_production_mode(duration_default=600.0)

def _derive_outdir_from_ckpt(ckpt: str, compare_dirname: str = "compare") -> str:
    ckpt_path = Path(ckpt)

    # 在 ckpt 的各级父目录里寻找形如 seed123 的目录名
    seed_dir = None
    for p in [ckpt_path.parent, *ckpt_path.parents]:
        if re.fullmatch(r"seed\d+", p.name):
            seed_dir = p
            break

    if seed_dir is None:
        # 没找到 seedX：就退化为 ckpt 同级目录下的 compare
        return str(ckpt_path.parent / compare_dirname)

    # 找到 seedX：在该 seed 目录下放 compare
    return str(seed_dir / compare_dirname)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="./checkpoints/production_best.pt", help="path to *_best.pt")
    ap.add_argument("--N", type=int, default=1000, help="number of scenarios")
    ap.add_argument("--seed", type=int, default=0, help="random seed for scenario generation")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--dt", type=float, default=0.5, help="override dt for simulation")
    ap.add_argument("--outdir", type=str, default="compare", help="output directory for plots/results")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    outdir_name = Path(args.outdir).name  # e.g. "compare" or "paper_figs"
    args.outdir = _derive_outdir_from_ckpt(args.ckpt, compare_dirname=outdir_name)
    outdir = _ensure_dir(args.outdir)
    mode_name, agent, ckpt = load_agent_from_ckpt(args.ckpt, device=args.device)
    mode = build_mode_from_name(mode_name)

    print(f"Loaded ckpt: {args.ckpt}")
    print(f"  mode={mode.name}  update={ckpt.get('update')}")
    print(f"  obs_dim={agent.cfg.obs_dim} act_dim={agent.cfg.act_dim} device={args.device}")

    # Scenario generation RNG
    scen_rng = np.random.default_rng(args.seed)
    # Evaluation RNG for uncertainty sampling (separate stream for reproducibility)
    eval_rng = np.random.default_rng(args.seed + 12345)

    N = int(args.N)
    scenarios: List[EpisodeParams] = []
    tau_mix_hat_list: List[float] = []
    tau_delay_list: List[float] = []
    dQs_list: List[float] = []
    wv_max_flow_list: List[float] = []
    cv_max_flow_list: List[float] = []
    wv_noise_seed_list: List[int] = []
    cv_noise_seed_list: List[int] = []

    scale_names = [spec.name for spec in mode.action_specs]
    scales_mat = np.zeros((N, len(scale_names)), dtype=np.float64)

    for i in range(N):
        p = mode.sample_episode(scen_rng)
        p.dt = float(args.dt)
        # Ensure mode is consistent (some samplers embed it)
        p.mode = mode.name
        scenarios.append(p)

        # Sample uncertainty once per scenario (shared between base and RL)
        tm = float(p.tau_mix)
        td = float(p.tau_delay)
        tm_hat = float(np.clip(eval_rng.normal(tm, max(1e-6, 0.25 * tm)), 5.0, 100.0))
        td_hat = float(np.clip(eval_rng.normal(td, max(1e-6, 0.25 * td)), 0.0, 20.0))

        tau_mix_hat_list.append(tm_hat)
        tau_delay_list.append(td_hat)
        dQs_list.append(float(abs(float(p.qs))))

        # 在 for i in range(N): 内，采样 tau 后，新增 max_flow 采样
        wv0 = float(p.water_valve_max_flow)
        cv0 = float(p.cement_valve_max_flow)

        wv_hat = float(np.clip(eval_rng.normal(wv0, max(1e-9, 0.25 * wv0)), 1.5 / 60, 3.0 / 60))
        cv_hat = float(np.clip(eval_rng.normal(cv0, max(1e-9, 0.25 * cv0)), 0.5 / 60, 2.0 / 60))

        wv_max_flow_list.append(wv_hat)
        cv_max_flow_list.append(cv_hat)

        wv_seed = int(eval_rng.integers(0, 2 ** 31 - 1))
        cv_seed = int(eval_rng.integers(0, 2 ** 31 - 1))
        wv_noise_seed_list.append(wv_seed)
        cv_noise_seed_list.append(cv_seed)

    # Compare
    R_base = np.zeros(N, dtype=np.float64)
    R_rl = np.zeros(N, dtype=np.float64)
    dR = np.zeros(N, dtype=np.float64)

    # Extra metrics (paired deltas)
    iae_h_base = np.zeros(N, dtype=np.float64)
    iae_h_rl = np.zeros(N, dtype=np.float64)
    iae_rho_base = np.zeros(N, dtype=np.float64)
    iae_rho_rl = np.zeros(N, dtype=np.float64)
    ts_rho_base = np.zeros(N)
    ts_rho_rl = np.zeros(N)
    sat_uc_base = np.zeros(N)
    sat_uc_rl = np.zeros(N)
    # production 还可加 ts_h, sat_uw

    # Prepare base scales
    base_scales = {spec.name: 1.0 for spec in mode.action_specs}

    # For trajectory plots if N <= 4
    traj_pairs: List[Tuple[Trajectory, Trajectory, EpisodeParams, Dict[str, float], Dict[str, float]]] = []

    for i, p in enumerate(scenarios):

        tm_hat = tau_mix_hat_list[i]
        td_hat = tau_delay_list[i]
        wv_hat = wv_max_flow_list[i]
        cv_hat = cv_max_flow_list[i]
        wv_seed = wv_noise_seed_list[i]
        cv_seed = cv_noise_seed_list[i]

        p_hat = deepcopy(p)
        p_hat.water_valve_max_flow = float(wv_hat)
        p_hat.cement_valve_max_flow = float(cv_hat)

        obs = mode.build_context(p_hat)
        a_det = act_deterministic(agent, obs)
        rl_scales = action_to_scales(a_det, mode.action_specs)
        for j, name in enumerate(scale_names):
            scales_mat[i, j] = float(rl_scales.get(name, 1.0))

        base_params = mode.compute_base_params(p_hat)

        if N <= 4:
            traj_b, extra_b = simulate_episode_with_uncertainty_full(
                p_hat, base_scales,
                base_params=base_params,
                tau_mix_hat=tm_hat,
                tau_delay=td_hat,
                wv_max_flow=wv_hat,
                cv_max_flow=cv_hat,
                wv_noise_seed=wv_seed,
                cv_noise_seed=cv_seed,
            )
            traj_r, extra_r = simulate_episode_with_uncertainty_full(
                p_hat, rl_scales,
                base_params=base_params,
                tau_mix_hat=tm_hat,
                tau_delay=td_hat,
                wv_max_flow=wv_hat,
                cv_max_flow=cv_hat,
                wv_noise_seed=wv_seed,
                cv_noise_seed=cv_seed,
            )

            _save_full_traj_csv(
                os.path.join(outdir, f"{mode.name}_traj_full_{i+1:02d}_base.csv"),
                traj_b, extra_b, p=p, tag="base",
            )
            _save_full_traj_csv(
                os.path.join(outdir, f"{mode.name}_traj_full_{i+1:02d}_rl.csv"),
                traj_r, extra_r, p=p, tag="rl",
            )
        else:
            traj_b = simulate_episode_with_uncertainty(
                p_hat, base_scales,
                base_params=base_params,
                tau_mix_hat=tm_hat,
                tau_delay=td_hat,
                wv_max_flow=wv_hat,
                cv_max_flow=cv_hat,
                wv_noise_seed=wv_seed,
                cv_noise_seed=cv_seed,
            )
            traj_r = simulate_episode_with_uncertainty(
                p_hat, rl_scales,
                base_params=base_params,
                tau_mix_hat=tm_hat,
                tau_delay=td_hat,
                wv_max_flow=wv_hat,
                cv_max_flow=cv_hat,
                wv_noise_seed=wv_seed,
                cv_noise_seed=cv_seed,
            )

        if p.mode == "premix":
            Rb = float(premix_reward(traj_b.rho, p))
            Rr = float(premix_reward(traj_r.rho, p))
        else:
            Rb = float(production_reward(traj_b.rho, traj_b.h, p))
            Rr = float(production_reward(traj_r.rho, traj_r.h, p))

        R_base[i] = Rb
        R_rl[i] = Rr
        dR[i] = Rr - Rb

        mb = compute_basic_metrics(traj_b, p)
        mr = compute_basic_metrics(traj_r, p)
        iae_h_base[i] = mb["IAE_h"]; iae_h_rl[i] = mr["IAE_h"]
        iae_rho_base[i] = mb["IAE_rho"]; iae_rho_rl[i] = mr["IAE_rho"]
        ts_rho_base[i] = mb["Ts_rho"]
        ts_rho_rl[i] = mr["Ts_rho"]
        sat_uc_base[i] = mb["sat_uc"]
        sat_uc_rl[i] = mr["sat_uc"]

        if N <= 4:
            params_base = apply_scales_to_params(base_params, base_scales)
            params_rl = apply_scales_to_params(base_params, rl_scales)
            traj_pairs.append((traj_b, traj_r, p, params_base, params_rl))

        if (i + 1) % max(1, (N // 10)) == 0:
            print(f"Progress: {i+1}/{N}")

    # Save per-scenario trajectory plots if small N
    if N <= 4:
        for idx, (tb, tr, p, pb, pr) in enumerate(traj_pairs, start=1):
            out = os.path.join(outdir, f"{mode.name}_traj_compare_{idx:02d}.png")
            plot_trajectory_compare(
                out, tb, tr, p,
                title_prefix=f"{mode.name}",
                params_base=pb,
                params_rl=pr,
            )
            print(f"Saved trajectory compare: {out}")

    # Always print paired summary
    rng_ci = np.random.default_rng(args.seed + 999)
    mean_d = float(np.mean(dR))
    med_d = float(np.median(dR))
    ci_mean = _bootstrap_ci(dR, rng_ci, stat="mean", B=3000, alpha=0.05)
    ci_med = _bootstrap_ci(dR, rng_ci, stat="median", B=3000, alpha=0.05)
    method, pval = _paired_pvalue_wilcoxon(dR)

    print("\n=== Paired comparison (RL - Base) ===")
    print(f"N = {N}")
    print(f"Return: base mean={float(np.mean(R_base)):.6g}, RL mean={float(np.mean(R_rl)):.6g}")
    print(f"ΔReturn mean={mean_d:.6g}  95%CI[{ci_mean[0]:.6g},{ci_mean[1]:.6g}]")
    print(f"ΔReturn median={med_d:.6g}  95%CI[{ci_med[0]:.6g},{ci_med[1]:.6g}]")
    print(f"Paired p-value ({method}) = {pval:.3g}")
    print(f"Win-rate (Δ>0) = {float(np.mean(dR > 0))*100:.1f}%")

    # Paired deltas of interpretable metrics
    d_iae_h = iae_h_rl - iae_h_base
    d_iae_rho = iae_rho_rl - iae_rho_base
    print("\n--- Interpretable metrics (RL - Base) ---")
    print(f"ΔIAE_h   mean={float(np.mean(d_iae_h)):.6g}, median={float(np.median(d_iae_h)):.6g}")
    print(f"ΔIAE_rho mean={float(np.mean(d_iae_rho)):.6g}, median={float(np.median(d_iae_rho)):.6g}")

    # If N>4: statistical plots & bucket analysis
    if N > 4:
        # Save arrays to npz for reproducibility
        npz_path = os.path.join(outdir, f"{mode.name}_compare_results_seed{args.seed}_N{N}.npz")
        np.savez(
            npz_path,
            R_base=R_base,
            R_rl=R_rl,
            dR=dR,
            tau_mix_hat=np.array(tau_mix_hat_list, dtype=np.float64),
            tau_delay=np.array(tau_delay_list, dtype=np.float64),
            dQs=np.array(dQs_list, dtype=np.float64),
            iae_h_base=iae_h_base,
            iae_h_rl=iae_h_rl,
            iae_rho_base=iae_rho_base,
            iae_rho_rl=iae_rho_rl,
            scales_mat=scales_mat,
            scale_names=np.array(scale_names)
        )
        print(f"Saved raw results: {npz_path}")

        # Plots
        plot_hist(
            os.path.join(outdir, f"{mode.name}_delta_return_hist.png"),
            dR,
            title=f"{mode.name}: ΔReturn histogram (RL - Base)",
            xlabel="ΔReturn",
        )

        plot_scatter(
            os.path.join(outdir, f"{mode.name}_scatter_delta_vs_tau_mix.png"),
            np.array(tau_mix_hat_list, dtype=np.float64),
            dR,
            title=f"{mode.name}: ΔReturn vs tau_mix_hat",
            xlabel="tau_mix_hat (s)",
            ylabel="ΔReturn",
        )

        plot_scatter(
            os.path.join(outdir, f"{mode.name}_scatter_delta_vs_dQs.png"),
            np.array(dQs_list, dtype=np.float64),
            dR,
            title=f"{mode.name}: ΔReturn vs |ΔQs|",
            xlabel="|ΔQs|",
            ylabel="ΔReturn",
        )

        plot_bucket_bars(
            os.path.join(outdir, f"{mode.name}_bucket_tau_mix.png"),
            np.array(tau_mix_hat_list, dtype=np.float64),
            dR,
            title=f"{mode.name}: mean ΔReturn by tau_mix_hat (quantile bins)",
            xlabel="tau_mix_hat bin",
            nbins=5,
            seed=args.seed + 7,
        )

        plot_bucket_bars(
            os.path.join(outdir, f"{mode.name}_bucket_dQs.png"),
            np.array(dQs_list, dtype=np.float64),
            dR,
            title=f"{mode.name}: mean ΔReturn by |ΔQs| (quantile bins)",
            xlabel="|ΔQs| bin",
            nbins=5,
            seed=args.seed + 8,
        )

        # Also: paired IAE scatter/hist
        plot_hist(
            os.path.join(outdir, f"{mode.name}_delta_IAE_rho_hist.png"),
            d_iae_rho,
            title=f"{mode.name}: ΔIAE_rho histogram (RL - Base)",
            xlabel="ΔIAE_rho (lower is better)",
        )
        plot_scatter(
            os.path.join(outdir, f"{mode.name}_scatter_deltaIAE_rho_vs_tau_mix.png"),
            np.array(tau_mix_hat_list, dtype=np.float64),
            d_iae_rho,
            title=f"{mode.name}: ΔIAE_rho vs tau_mix_hat",
            xlabel="tau_mix_hat (s)",
            ylabel="ΔIAE_rho",
        )

        plot_scales_boxplot(
            os.path.join(outdir, f"{mode.name}_scales_boxplot.png"),
            scales_mat,
            scale_names,
            title=f"{mode.name}: RL scales distribution",
        )

        x_tau = np.array(tau_mix_hat_list, dtype=np.float64)
        x_dqs = np.array(dQs_list, dtype=np.float64)

        for name in scale_names:
            y = scales_mat[:, scale_names.index(name)]
            plot_scatter(
                os.path.join(outdir, f"{mode.name}_scatter_{name}_vs_tau_mix.png"),
                x_tau, y,
                title=f"{mode.name}: {name} vs tau_mix_hat",
                xlabel="tau_mix_hat (s)",
                ylabel=f"{name} scale",
            )
            plot_scatter(
                os.path.join(outdir, f"{mode.name}_scatter_{name}_vs_dQs.png"),
                x_dqs, y,
                title=f"{mode.name}: {name} vs |ΔQs|",
                xlabel="|ΔQs|",
                ylabel=f"{name} scale",
            )

        print(f"Saved plots to: {outdir}")

    # Save a short text report
    report_path = os.path.join(outdir, f"{mode.name}_compare_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"mode: {mode.name}\n")
        f.write(f"N: {N}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"dt: {args.dt}\n\n")
        f.write("Paired comparison (RL - Base)\n")
        f.write(f"Return base mean: {float(np.mean(R_base))}\n")
        f.write(f"Return RL mean: {float(np.mean(R_rl))}\n")
        f.write(f"Delta mean: {mean_d}\n")
        f.write(f"Delta mean 95%CI: [{ci_mean[0]}, {ci_mean[1]}]\n")
        f.write(f"Delta median: {med_d}\n")
        f.write(f"Delta median 95%CI: [{ci_med[0]}, {ci_med[1]}]\n")
        f.write(f"p-value ({method}): {pval}\n")
        f.write(f"win-rate (Δ>0): {float(np.mean(dR>0))}\n\n")
        f.write("Interpretable metrics (RL - Base)\n")
        f.write(f"ΔIAE_h mean: {float(np.mean(d_iae_h))}, median: {float(np.median(d_iae_h))}\n")
        f.write(f"ΔIAE_rho mean: {float(np.mean(d_iae_rho))}, median: {float(np.median(d_iae_rho))}\n")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()


# python evaluate_compare_clean_fixed.py --ckpt ./checkpoints/seed1/production_best.pt --N 2 --seed 12
# python evaluate_compare_clean_fixed.py --ckpt ./checkpoints/seed1/premix_best.pt --N 2 --seed 12

# 1) 输入（命令行参数 / CLI）
#
# 脚本支持这些参数：
#
# seed_sweep
#
# --ckpt：要评估的模型 checkpoint（*_best.pt）
#
# 默认：./checkpoints/production_best.pt
#
# --N：Monte Carlo 场景数（采样多少个 episode 做对比）
#
# 默认：1000
#
# --seed：随机种子（控制场景采样、uncertainty 采样、以及噪声种子生成）
#
# 默认：0
#
# --device：cuda 或 cpu
#
# 默认：cuda
#
# --dt：覆盖仿真步长（每个 episode 的 dt）
#
# 默认：0.5
#
# --outdir：输出目录名（注意：会被自动“挂到 ckpt 对应 seed 目录下”）
#
# 默认：compare
#
# 运行例子（文件末尾也写了）：
#
# seed_sweep
#
# python evaluate_compare_clean_fixed.py --ckpt ./checkpoints/seed1/production_best.pt --N 1000 --seed 12
#
# 2) 隐式输入（它还依赖什么）
#
# 除了 CLI 参数，这个脚本还 import 并调用 你的工程代码（这些也是输入的一部分）：
#
# seed_sweep
#
# modes.py：提供 ModeSpec、EpisodeParams、采样器、compute_base_params()、reward 等
#
# PPO_bandit.py：提供 PPOBanditAgent/PPOConfig
#
# sim_config.py / sim_env.py / sim_model.py：仿真环境与状态
#
# tune_baseline_v3.py：用到 opening_to_flow
#
# 以及 *.pt checkpoint 本身（需要包含字段：mode/update/state_dict/obs_dim/act_dim/config）
#
# 3) 输出（会生成哪些文件/打印什么）
# A. 输出目录在哪里？
#
# 它会根据 --ckpt 自动推导输出目录（重点！）：
#
# 如果 ckpt 路径里包含 seedX 目录：输出到 .../seedX/compare/
#
# 否则输出到 ckpt 同级目录/compare/
#
# seed_sweep
#
# 比如：
#
# ckpt=./checkpoints/seed1/production_best.pt
#
# outdir 默认 compare
#
# 则输出目录是：./checkpoints/seed1/compare/
#
# B. 当 N <= 4 时（小样本“查 case”模式）
#
# 会保存每个场景的完整轨迹 CSV（含流量波动前/后） + 对比图：
#
# seed_sweep
#
# CSV（每个场景两份：base / rl）：
#
# production_traj_full_01_base.csv
#
# production_traj_full_01_rl.csv
#
# ...
#
# CSV 列包括（核心）：t, h, rho, u_w, u_c, Q_s, Q_w_nom, Q_w, Q_c_nom, Q_c
# 其中：
#
# Q_*_nom：按开度通过 opening_to_flow()算的“无噪声”流量（波动前）
#
# Q_*：仿真器中的 current_flow（含你注入的随机波动，波动后）
#
# seed_sweep
#
# 对比图：
#
# production_traj_compare_01.png
#
# premix：只画 rho
#
# production：画 h、rho、u_w、u_c 4 行，并在图中标注 base 参数和 RL scale 后参数
#
# seed_sweep
#
# C. 当 N > 4 时（统计评估模式）
#
# 会输出：
#
# 原始结果 NPZ（用于复现实验，不用再跑仿真）：
#
# <mode>_compare_results_seed{seed}_N{N}.npz
# 里面包含：R_base, R_rl, dR, tau_mix_hat, tau_delay, dQs, iae_* , scales_mat, scale_names ...
#
# seed_sweep
#
# 一堆统计图 PNG（直方图/散点图/bucket 图/scale 分布等），例如：
#
# seed_sweep
#
# <mode>_delta_return_hist.png
#
# <mode>_scatter_delta_vs_tau_mix.png
#
# <mode>_bucket_tau_mix.png
#
# <mode>_scales_boxplot.png
#
# 以及每个 scale 对 tau_mix_hat、|ΔQs| 的散点图：
# <mode>_scatter_s_c_p_vs_tau_mix.png 等
#
# 文本报告：
#
# <mode>_compare_report.txt（写明 ckpt、N、seed、ΔReturn 均值/CI、p-value、ΔIAE 等）
#
# seed_sweep
#
# 控制台打印：
#
# 加载信息、进度条（每 10% 打一次）
#
# 统计摘要：base/RL mean，ΔReturn 的 bootstrap CI、Wilcoxon/sign test p-value、win-rate
#
# 以及 ΔIAE_h、ΔIAE_rho 的均值/中位数
#
# seed_sweep