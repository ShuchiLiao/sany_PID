"""
evaluate.py

Evaluation / comparison entrypoint for the sany_PID RL refactor.

This file is the functional successor of evaluate_compare.py.  It performs:
- deterministic RL checkpoint loading;
- all-ones baseline evaluation;
- fair paired baseline-vs-RL Monte Carlo comparison with shared scenarios and
  shared sampled uncertainties;
- bootstrap CI, Wilcoxon/sign-test fallback;
- delta histogram/scatter/bucket plots, scale plots;
- optional full trajectory CSV and trajectory comparison PNG for small N.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parents[1]
for _p in (_SCRIPTS / "core", _SCRIPTS / "PID_control", _SCRIPTS / "rl", _SCRIPTS):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

from modes import ModeSpec, action_to_scales, make_premix_mode, make_production_mode, premix_reward, production_reward  # noqa: E402
from PPO_bandit import PPOBanditAgent, PPOConfig  # noqa: E402
from rollout import (  # noqa: E402
    sample_episode_uncertainty,
    save_trajectory_csv,
    simulate_episode,
    simulate_episode_with_uncertainty,
    trajectory_reward,
)
from metrics import (  # noqa: E402
    aggregate_metric_dicts,
    compare_metric_rows,
    compute_basic_metrics,
    paired_delta_summary,
    save_json,
    summarize_rewards,
    write_rows_csv,
)
from plotting import (  # noqa: E402
    ensure_dir,
    plot_compare_outputs,
    plot_eval_rewards,
    plot_trajectory_compare,
    save_metrics_npz,
)


# -----------------------------------------------------------------------------
# shared construction/loading helpers
# -----------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_mode(
    mode_name: str,
    *,
    premix_duration: float = 400.0,
    premix_hold: float = 10.0,
    production_duration: float = 600.0,
) -> ModeSpec:
    if mode_name == "premix":
        return make_premix_mode(duration_default=float(premix_duration), hold_default=float(premix_hold))
    if mode_name == "production":
        return make_production_mode(duration_default=float(production_duration))
    raise ValueError(f"Unknown mode: {mode_name}")


def mode_names_from_arg(mode: str) -> List[str]:
    return ["premix", "production"] if mode == "both" else [str(mode)]


def make_agent_for_mode(mode: ModeSpec, device: str = "cpu", seed: int = 0) -> PPOBanditAgent:
    rng = np.random.default_rng(int(seed))
    obs = mode.build_context(mode.sample_episode(rng))
    cfg = PPOConfig(obs_dim=int(obs.shape[0]), act_dim=len(mode.action_specs))
    return PPOBanditAgent(cfg, device=device)


def _load_state_into_agent(agent: PPOBanditAgent, ckpt: Any) -> None:
    if isinstance(ckpt, Mapping):
        for key in ("state_dict", "model_state_dict", "agent_state_dict"):
            if key in ckpt:
                agent.load_state_dict(ckpt[key])
                return
    # Fallback: checkpoint itself may be a plain state_dict.
    agent.load_state_dict(ckpt)


def load_agent_from_checkpoint(mode: ModeSpec, ckpt_path: str, device: str = "cpu") -> PPOBanditAgent:
    ckpt = torch.load(ckpt_path, map_location=device)
    obs_dim = 0
    act_dim = len(mode.action_specs)
    cfg_dict: Dict[str, Any] = {}
    if isinstance(ckpt, Mapping):
        obs_dim = int(ckpt.get("obs_dim", 0) or 0)
        act_dim = int(ckpt.get("act_dim", act_dim) or act_dim)
        cfg_dict = dict(ckpt.get("config", {}) or {})
    if obs_dim <= 0:
        rng = np.random.default_rng(0)
        obs_dim = int(mode.build_context(mode.sample_episode(rng)).shape[0])
    cfg_dict.update({"obs_dim": obs_dim, "act_dim": act_dim})
    try:
        cfg = PPOConfig(**cfg_dict)
    except TypeError:
        cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = PPOBanditAgent(cfg, device=device)
    _load_state_into_agent(agent, ckpt)
    return agent


def baseline_scales(mode: ModeSpec) -> Dict[str, float]:
    return {spec.name: 1.0 for spec in mode.action_specs}


@torch.no_grad()
def deterministic_action(agent: PPOBanditAgent, obs: np.ndarray) -> np.ndarray:
    """Best-effort deterministic action extraction across possible PPOBandit variants."""
    if hasattr(agent, "act_deterministic"):
        out = agent.act_deterministic(obs)
        return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)
    if hasattr(agent, "deterministic_action"):
        out = agent.deterministic_action(obs)
        return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)
    if hasattr(agent, "mean_action"):
        out = agent.mean_action(obs)
        return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)
    # Fallback to stochastic act; this preserves runnable behavior even if the
    # old PPO class did not expose a deterministic method.
    out = agent.act(obs)
    return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)


@torch.no_grad()
def agent_scales(agent: PPOBanditAgent, mode: ModeSpec, obs: np.ndarray) -> Dict[str, float]:
    a = deterministic_action(agent, obs)
    return action_to_scales(a, mode.action_specs)


def reward_from_traj(mode: ModeSpec, traj: Any, p: Any) -> float:
    if bool(getattr(traj, "failed", False)):
        return -10.0
    if getattr(p, "mode", mode.name) == "premix":
        return float(premix_reward(traj.rho, p))
    return float(production_reward(traj.rho, traj.h, p))


# -----------------------------------------------------------------------------
# Evaluation helpers used by train.py and standalone evaluate.py
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_agent_with_metrics(
    mode: ModeSpec,
    agent: PPOBanditAgent,
    rng: np.random.Generator,
    *,
    override_dt: float = 0.5,
    episodes: int = 200,
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    rews: List[float] = []
    metric_rows: List[Dict[str, float]] = []
    for _ in range(int(episodes)):
        p = mode.sample_episode(rng)
        p.dt = float(override_dt)
        obs = mode.build_context(p)
        scales = agent_scales(agent, mode, obs)
        base_params = mode.compute_base_params(p)
        traj = simulate_episode(p, scales, rng, return_traj=True, base_params=base_params, full=False)
        rews.append(reward_from_traj(mode, traj, p))
        metric_rows.append(compute_basic_metrics(traj, p))
    rews_np = np.asarray(rews, dtype=np.float32)
    return summarize_rewards(rews_np), aggregate_metric_dicts(metric_rows), rews_np


def evaluate_baseline_with_metrics(
    mode: ModeSpec,
    rng: np.random.Generator,
    *,
    override_dt: float = 0.5,
    episodes: int = 200,
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    rews: List[float] = []
    metric_rows: List[Dict[str, float]] = []
    scales = baseline_scales(mode)
    for _ in range(int(episodes)):
        p = mode.sample_episode(rng)
        p.dt = float(override_dt)
        base_params = mode.compute_base_params(p)
        traj = simulate_episode(p, scales, rng, return_traj=True, base_params=base_params, full=False)
        rews.append(reward_from_traj(mode, traj, p))
        metric_rows.append(compute_basic_metrics(traj, p))
    rews_np = np.asarray(rews, dtype=np.float32)
    return summarize_rewards(rews_np), aggregate_metric_dicts(metric_rows), rews_np


# -----------------------------------------------------------------------------
# Paired compare: main successor of evaluate_compare.py
# -----------------------------------------------------------------------------


def _scenario_row(i: int, p: Any, uncertainty: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "i": int(i),
        "h0": float(getattr(p, "h0", np.nan)),
        "h_sp": float(getattr(p, "h_sp", np.nan)),
        "rho0": float(getattr(p, "rho0", np.nan)),
        "rho_sp": float(getattr(p, "rho_sp", np.nan)),
        "qs": float(getattr(p, "qs", 0.0)),
        "tau_mix_nominal": float(getattr(p, "tau_mix", np.nan)),
        "tau_delay_nominal": float(getattr(p, "tau_delay", np.nan)),
        "tau_mix_hat": float(uncertainty.get("tau_mix_hat", np.nan)),
        "rho_obs_delay": float(uncertainty.get("rho_obs_delay", np.nan)),
        "water_valve_max_flow": float(uncertainty.get("water_valve_max_flow", np.nan)),
        "cement_valve_max_flow": float(uncertainty.get("cement_valve_max_flow", np.nan)),
    }


def run_paired_comparison(
    mode: ModeSpec,
    agent: PPOBanditAgent,
    *,
    N: int,
    seed: int,
    dt: float,
    out_dir: str,
    bootstrap_B: int = 2000,
    save_full_if_n_le: int = 4,
    make_plots: bool = True,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    rng = np.random.default_rng(int(seed))
    b_scales = baseline_scales(mode)

    baseline_rewards: List[float] = []
    rl_rewards: List[float] = []
    baseline_metric_rows: List[Dict[str, float]] = []
    rl_metric_rows: List[Dict[str, float]] = []
    scenario_rows: List[Dict[str, float]] = []
    scale_rows: List[Dict[str, float]] = []
    scale_series: Dict[str, List[float]] = {spec.name: [] for spec in mode.action_specs}

    save_full = int(N) <= int(save_full_if_n_le)
    traj_dir = ensure_dir(os.path.join(out_dir, "trajectories")) if save_full else ""

    for i in range(int(N)):
        p = mode.sample_episode(rng)
        p.dt = float(dt)
        uncertainty = sample_episode_uncertainty(p, rng)
        base_params = mode.compute_base_params(p)
        obs = mode.build_context(p)


        r_scales = agent_scales(agent, mode, obs)

        traj_b = simulate_episode_with_uncertainty(
            p,
            b_scales,
            uncertainty,
            base_params=base_params,
            return_traj=True,
            full=save_full,
            catch_errors=True,
        )
        traj_r = simulate_episode_with_uncertainty(
            p,
            r_scales,
            uncertainty,
            base_params=base_params,
            return_traj=True,
            full=save_full,
            catch_errors=True,
        )

        Rb = reward_from_traj(mode, traj_b, p)
        Rr = reward_from_traj(mode, traj_r, p)
        baseline_rewards.append(float(Rb))
        rl_rewards.append(float(Rr))
        baseline_metric_rows.append(compute_basic_metrics(traj_b, p))
        rl_metric_rows.append(compute_basic_metrics(traj_r, p))

        srow = _scenario_row(i, p, uncertainty)
        srow.update({"baseline_reward": float(Rb), "rl_reward": float(Rr), "delta_reward": float(Rr - Rb)})
        scenario_rows.append(srow)

        scale_row: Dict[str, float] = {"i": int(i)}
        for name in scale_series.keys():
            v = float(r_scales.get(name, np.nan))
            scale_series[name].append(v)
            scale_row[name] = v
        scale_rows.append(scale_row)

        if save_full:
            prefix = os.path.join(traj_dir, f"{mode.name}_case{i:03d}")
            save_trajectory_csv(prefix + "_baseline.csv", traj_b)
            save_trajectory_csv(prefix + "_rl.csv", traj_r)
            if make_plots:
                plot_trajectory_compare(
                    prefix + "_compare.png",
                    traj_b,
                    traj_r,
                    rho_sp=float(getattr(p, "rho_sp", np.nan)),
                    h_sp=(float(getattr(p, "h_sp", np.nan)) if mode.name == "production" else None),
                    title=f"{mode.name} paired case {i}",
                )

    baseline_arr = np.asarray(baseline_rewards, dtype=np.float64)
    rl_arr = np.asarray(rl_rewards, dtype=np.float64)
    delta_arr = rl_arr - baseline_arr

    arrays = {
        "baseline_rewards": baseline_arr,
        "rl_rewards": rl_arr,
        "delta_rewards": delta_arr,
        "tau_mix_hat": np.asarray([r["tau_mix_hat"] for r in scenario_rows], dtype=np.float64),
        "rho_obs_delay": np.asarray([r["rho_obs_delay"] for r in scenario_rows], dtype=np.float64),
        "qs": np.asarray([r["qs"] for r in scenario_rows], dtype=np.float64),
        "rho_sp": np.asarray([r["rho_sp"] for r in scenario_rows], dtype=np.float64),
        "h_sp": np.asarray([r["h_sp"] for r in scenario_rows], dtype=np.float64),
    }

    summary: Dict[str, Any] = {
        "mode": mode.name,
        "N": int(N),
        "seed": int(seed),
        "dt": float(dt),
        "baseline_return": summarize_rewards(baseline_arr, prefix="baseline_R"),
        "rl_return": summarize_rewards(rl_arr, prefix="rl_R"),
        "paired_return": paired_delta_summary(baseline_arr, rl_arr, B=int(bootstrap_B), seed=int(seed), prefix="return"),
        "baseline_control": aggregate_metric_dicts(baseline_metric_rows),
        "rl_control": aggregate_metric_dicts(rl_metric_rows),
        "paired_control_delta": compare_metric_rows(baseline_metric_rows, rl_metric_rows, B=int(bootstrap_B), seed=int(seed)),
    }

    scenario_csv = write_rows_csv(os.path.join(out_dir, f"{mode.name}_paired_cases.csv"), scenario_rows)
    scale_csv = write_rows_csv(os.path.join(out_dir, f"{mode.name}_rl_scales.csv"), scale_rows)
    npz_path = save_metrics_npz(
        os.path.join(out_dir, f"{mode.name}_paired_compare.npz"),
        **arrays,
        **{f"scale_{k}": np.asarray(v, dtype=np.float64) for k, v in scale_series.items()},
    )
    plot_paths: Dict[str, str] = {}
    if make_plots:
        plot_eval_rewards(mode.name, out_dir, baseline_arr, label="baseline")
        plot_eval_rewards(mode.name, out_dir, rl_arr, label="rl")
        plot_paths.update(plot_compare_outputs(out_dir, mode.name, arrays, scale_series))

    summary.update({"scenario_csv": scenario_csv, "scale_csv": scale_csv, "npz": npz_path, "plots": plot_paths})

    summary_path = save_json(os.path.join(out_dir, f"{mode.name}_paired_compare_summary.json"), summary)
    report_path = write_compare_report(os.path.join(out_dir, f"{mode.name}_paired_compare_report.txt"), summary)
    summary["summary_path"] = summary_path
    summary["report_path"] = report_path
    return summary


def write_compare_report(path: str, summary: Mapping[str, Any]) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    pr = summary.get("paired_return", {})
    br = summary.get("baseline_return", {})
    rr = summary.get("rl_return", {})
    lines = [
        f"mode: {summary.get('mode')}",
        f"N: {summary.get('N')}",
        f"seed: {summary.get('seed')}",
        f"dt: {summary.get('dt')}",
        "",
        "Return summary",
        f"  baseline mean: {br.get('baseline_R_mean', float('nan')):.6g}",
        f"  RL mean      : {rr.get('rl_R_mean', float('nan')):.6g}",
        f"  delta mean   : {pr.get('return_delta_mean', float('nan')):.6g}",
        f"  delta median : {pr.get('return_delta_median', float('nan')):.6g}",
        f"  delta 95% CI : [{pr.get('return_delta_ci_low', float('nan')):.6g}, {pr.get('return_delta_ci_high', float('nan')):.6g}]",
        f"  win rate     : {pr.get('return_win_rate', float('nan')):.4f}",
        f"  sign p       : {pr.get('return_sign_p', float('nan')):.6g}",
        f"  Wilcoxon p   : {pr.get('return_wilcoxon_p', float('nan')):.6g}",
        "",
        f"scenario_csv: {summary.get('scenario_csv')}",
        f"scale_csv: {summary.get('scale_csv')}",
        f"npz: {summary.get('npz')}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["premix", "production", "both"], default="both")
    ap.add_argument("--ckpt", type=str, default=None, help="Specific checkpoint path; only valid with a single --mode")
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Use <ckpt_dir>/<mode>_best.pt when --ckpt is not set")
    ap.add_argument("--out_dir", type=str, default="./outputs/evaluate_compare")
    ap.add_argument("--N", type=int, default=None, help="Number of paired scenarios; alias of --episodes")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--premix_duration", type=float, default=400.0)
    ap.add_argument("--premix_hold", type=float, default=10.0)
    ap.add_argument("--production_duration", type=float, default=600.0)
    ap.add_argument("--bootstrap_B", type=int, default=2000)
    ap.add_argument("--save_full_if_n_le", type=int, default=4)
    ap.add_argument("--no_plots", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    N = int(args.N if args.N is not None else args.episodes)
    modes = mode_names_from_arg(args.mode)

    if args.ckpt is not None and len(modes) > 1:
        raise ValueError("--ckpt can only be used with --mode premix or --mode production. For --mode both, use --ckpt_dir.")

    all_reports: List[Dict[str, Any]] = []
    for mode_name in modes:
        mode = make_mode(
            mode_name,
            premix_duration=float(args.premix_duration),
            premix_hold=float(args.premix_hold),
            production_duration=float(args.production_duration),
        )
        ckpt_path = args.ckpt or os.path.join(args.ckpt_dir, f"{mode.name}_best.pt")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        agent = load_agent_from_checkpoint(mode, ckpt_path, device=args.device)

        out_dir = ensure_dir(os.path.join(args.out_dir, mode.name))
        report = run_paired_comparison(
            mode,
            agent,
            N=N,
            seed=int(args.seed),
            dt=float(args.dt),
            out_dir=out_dir,
            bootstrap_B=int(args.bootstrap_B),
            save_full_if_n_le=int(args.save_full_if_n_le),
            make_plots=not bool(args.no_plots),
        )
        report["checkpoint"] = ckpt_path
        all_reports.append(report)
        pr = report.get("paired_return", {})
        br = report.get("baseline_return", {})
        rr = report.get("rl_return", {})

        print(
            f"[{mode.name}] "
            f"baseline_R_mean={br.get('baseline_R_mean', float('nan')):.6g}, "
            f"candidate_R_mean={rr.get('rl_R_mean', float('nan')):.6g}, "
            f"delta_mean={pr.get('return_delta_mean', float('nan')):.6g}, "
            f"win_rate={pr.get('return_win_rate', float('nan')):.3f}, "
            f"summary={report.get('summary_path')}"
        )

    ensure_dir(args.out_dir)
    save_json(os.path.join(args.out_dir, "paired_compare_summary_all.json"), all_reports)


if __name__ == "__main__":
    main()
