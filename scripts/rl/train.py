"""
train.py

Training entrypoint for the sany_PID RL refactor.

This file is the functional successor of train_modes.py plus seed_sweep.py:
- PPO-bandit training for premix / production / both;
- multiprocessing rollout collection;
- checkpoint saving and best-checkpoint selection;
- eval-over-training metrics and plots;
- final best checkpoint evaluation;
- multi-seed execution via --seeds;
- seed-sweep summary aggregation via --summarize_only.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parents[1]
for _p in (_SCRIPTS / "core", _SCRIPTS / "PID_control", _SCRIPTS / "rl", _SCRIPTS):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

from modes import ModeSpec, action_to_scales, make_premix_mode, make_production_mode  # noqa: E402
from PPO_bandit import PPOBanditAgent, PPOConfig, RolloutBuffer  # noqa: E402
from rollout import rollout_reward_worker, simulate_episode  # noqa: E402
from evaluate import evaluate_agent_with_metrics, make_agent_for_mode, set_global_seed  # noqa: E402
from metrics import load_seed_summary_table, save_json, summarize_seed_table, write_rows_csv  # noqa: E402
from plotting import (  # noqa: E402
    ensure_dir,
    plot_ctrl_metrics_curves,
    plot_eval_rewards,
    plot_seed_sweep_boxplot,
    plot_training_curves,
    save_metrics_npz,
)


@dataclass
class TrainConfig:
    device: str = "cpu"
    seed: int = 0
    num_workers: int = 0
    mp_start: str = "spawn"
    mp_chunksize: int = 8
    batch_episodes: int = 256
    updates: int = 2000
    ckpt_dir: str = "./checkpoints"
    save_every: int = 50
    eval_episodes: int = 200
    eval_every: int = 200
    best_metric: str = "ema_return"  # "ema_return" or "mean_return"
    plots_dirname: str = "plots"


def save_checkpoint(path: str, mode: ModeSpec, agent: PPOBanditAgent, update: int) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "mode": mode.name,
            "update": int(update),
            "state_dict": agent.state_dict(),
            "obs_dim": int(agent.cfg.obs_dim),
            "act_dim": int(agent.cfg.act_dim),
            "config": dict(agent.cfg.__dict__),
        },
        path,
    )
    return path


def make_mode_from_name(
    mode_name: str,
    *,
    premix_duration: float,
    premix_hold: float,
    production_duration: float,
) -> ModeSpec:
    if mode_name == "premix":
        return make_premix_mode(duration_default=float(premix_duration), hold_default=float(premix_hold))
    if mode_name == "production":
        return make_production_mode(duration_default=float(production_duration))
    raise ValueError(f"Unknown mode: {mode_name}")


def collect_batch(
    mode: ModeSpec,
    agent: PPOBanditAgent,
    rng: np.random.Generator,
    cfg: TrainConfig,
    *,
    override_dt: float = 0.5,
    pool: Optional[mp.pool.Pool] = None,
) -> RolloutBuffer:
    """Collect one contextual-bandit PPO batch."""
    obs_dim = int(mode.build_context(mode.sample_episode(rng)).shape[0])
    act_dim = len(mode.action_specs)
    buf = RolloutBuffer(obs_dim=obs_dim, act_dim=act_dim)

    ps = []
    obs_list = []
    for _ in range(int(cfg.batch_episodes)):
        p = mode.sample_episode(rng)
        p.dt = float(override_dt)
        ps.append(p)
        obs_list.append(mode.build_context(p))

    obs_batch = np.stack(obs_list, axis=0).astype(np.float32, copy=False)

    if hasattr(agent, "act_batch"):
        a_batch, logp_batch, v_batch = agent.act_batch(obs_batch)
    else:
        a_list, lp_list, v_list = [], [], []
        for i in range(obs_batch.shape[0]):
            a, lp, v = agent.act(obs_batch[i])
            a_list.append(a)
            lp_list.append(lp)
            v_list.append(v)
        a_batch = np.stack(a_list, axis=0)
        logp_batch = np.asarray(lp_list, dtype=np.float32)
        v_batch = np.asarray(v_list, dtype=np.float32)

    scales_list: List[Dict[str, float]] = []
    base_params_list: List[Dict[str, float]] = []
    for i, p in enumerate(ps):
        scales_list.append(action_to_scales(a_batch[i], mode.action_specs))
        base_params_list.append(mode.compute_base_params(p))

    seeds = rng.integers(0, 2**31 - 1, size=obs_batch.shape[0], dtype=np.int64)
    worker_args = [
        (ps[i], scales_list[i], base_params_list[i], float(override_dt), int(seeds[i]))
        for i in range(obs_batch.shape[0])
    ]

    if pool is None:
        rewards = [rollout_reward_worker(item) for item in worker_args]
    else:
        rewards = pool.map(rollout_reward_worker, worker_args, chunksize=int(cfg.mp_chunksize))
    rewards = np.asarray(rewards, dtype=np.float32)

    for i in range(obs_batch.shape[0]):
        buf.add(obs_batch[i], a_batch[i], float(logp_batch[i]), float(v_batch[i]), float(rewards[i]))
    return buf


def train_one_mode(
    mode: ModeSpec,
    agent: PPOBanditAgent,
    cfg: TrainConfig,
    *,
    override_dt: float = 0.5,
) -> Dict[str, Any]:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    plots_dir = ensure_dir(os.path.join(cfg.ckpt_dir, cfg.plots_dirname))

    seed = int(cfg.seed)
    set_global_seed(seed)
    rng_train = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 1234)

    ema_return: Optional[float] = None
    beta = 0.98
    returns_mean: List[float] = []
    returns_ema: List[float] = []
    loss_pi: List[float] = []
    loss_v: List[float] = []
    entropy: List[float] = []
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

    pool: Optional[mp.pool.Pool] = None
    if int(cfg.num_workers) > 1:
        ctx = mp.get_context(str(cfg.mp_start))
        pool = ctx.Pool(processes=int(cfg.num_workers), maxtasksperchild=200)

    try:
        for u in range(1, int(cfg.updates) + 1):
            buf = collect_batch(mode, agent, rng_train, cfg, override_dt=override_dt, pool=pool)
            rewards = np.asarray(buf._rew, dtype=np.float32)
            avg_R = float(np.nanmean(rewards))
            ema_return = avg_R if ema_return is None else beta * ema_return + (1.0 - beta) * avg_R

            losses = agent.update(buf)
            returns_mean.append(avg_R)
            returns_ema.append(float(ema_return))
            loss_pi.append(float(losses.get("policy", losses.get("loss_pi", np.nan))))
            loss_v.append(float(losses.get("value", losses.get("loss_v", np.nan))))
            entropy.append(float(losses.get("entropy", np.nan)))

            if u == 1 or u % 10 == 0 or u == int(cfg.updates):
                print(
                    f"[{mode.name}] seed={seed} update={u:05d} "
                    f"R_mean={avg_R:+.4f} R_ema={float(ema_return):+.4f} "
                    f"loss_pi={loss_pi[-1]:.4f} loss_v={loss_v[-1]:.4f} entropy={entropy[-1]:.4f}"
                )

            if int(cfg.save_every) > 0 and u % int(cfg.save_every) == 0:
                ckpt_path = os.path.join(cfg.ckpt_dir, f"{mode.name}_ppo_bandit_{u:05d}.pt")
                save_checkpoint(ckpt_path, mode, agent, u)
                print(f"Saved checkpoint: {ckpt_path}")

            do_eval = int(cfg.eval_every) > 0 and u % int(cfg.eval_every) == 0
            if do_eval:
                ret, ctrl, _rews = evaluate_agent_with_metrics(
                    mode,
                    agent,
                    rng_eval,
                    override_dt=override_dt,
                    episodes=int(cfg.eval_episodes),
                )
                eval_updates.append(int(u))
                eval_R_mean.append(float(ret.get("R_mean", np.nan)))
                eval_IAE_rho.append(float(ctrl.get("IAE_rho", np.nan)))
                eval_OS_rho.append(float(ctrl.get("overshoot_rho", np.nan)))
                eval_TV_uc.append(float(ctrl.get("TV_uc", np.nan)))
                if mode.name == "production":
                    eval_IAE_h.append(float(ctrl.get("IAE_h", np.nan)))
                    eval_OS_h.append(float(ctrl.get("overshoot_h", np.nan)))
                    eval_TV_uw.append(float(ctrl.get("TV_uw", np.nan)))

                score = float(ema_return) if cfg.best_metric == "ema_return" else float(ret.get("R_mean", -1e18))
                if score > best_score:
                    best_score = score
                    save_checkpoint(best_path, mode, agent, u)
                    print(f"[BEST] update={u:05d} score={best_score:+.4f} -> {best_path}")

        if not os.path.exists(best_path):
            best_score = float(returns_ema[-1]) if returns_ema else -1e18
            save_checkpoint(best_path, mode, agent, int(cfg.updates))
            print(f"[BEST] final score={best_score:+.4f} -> {best_path}")

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    train_plot_paths = plot_training_curves(mode.name, plots_dir, returns_mean, returns_ema, loss_pi, loss_v, entropy)
    ckpt = torch.load(best_path, map_location="cpu")
    agent.load_state_dict(ckpt["state_dict"])

    final_ret, final_ctrl, final_rewards = evaluate_agent_with_metrics(
        mode,
        agent,
        rng_eval,
        override_dt=override_dt,
        episodes=int(cfg.eval_episodes),
    )
    eval_plot_path = plot_eval_rewards(mode.name, plots_dir, final_rewards)

    ctrl_plot_paths = plot_ctrl_metrics_curves(
        mode.name,
        plots_dir,
        eval_updates,
        {
            "R_mean": eval_R_mean,
            "IAE_rho": eval_IAE_rho,
            "OS_rho": eval_OS_rho,
            "TV_uc": eval_TV_uc,
            **({"IAE_h": eval_IAE_h, "OS_h": eval_OS_h, "TV_uw": eval_TV_uw} if mode.name == "production" else {}),
        },
    )

    metrics_npz_path = os.path.join(plots_dir, f"{mode.name}_seed{seed}_metrics_series.npz")
    save_metrics_npz(
        metrics_npz_path,
        train_returns_mean=returns_mean,
        train_returns_ema=returns_ema,
        train_loss_pi=loss_pi,
        train_loss_v=loss_v,
        train_entropy=entropy,
        eval_updates=eval_updates,
        eval_R_mean=eval_R_mean,
        eval_IAE_rho=eval_IAE_rho,
        eval_OS_rho=eval_OS_rho,
        eval_TV_uc=eval_TV_uc,
        eval_IAE_h=(eval_IAE_h if mode.name == "production" else []),
        eval_OS_h=(eval_OS_h if mode.name == "production" else []),
        eval_TV_uw=(eval_TV_uw if mode.name == "production" else []),
        best_eval_rewards=final_rewards,
    )

    summary = {
        "mode": mode.name,
        "seed": seed,
        "updates": int(cfg.updates),
        "batch_episodes": int(cfg.batch_episodes),
        "eval_episodes": int(cfg.eval_episodes),
        "best_path": best_path,
        "best_update": int(ckpt.get("update", -1)),
        "best_score": float(best_score),
        "eval_metrics": final_ret,
        "eval_control_metrics": final_ctrl,
        "train_plots": train_plot_paths,
        "eval_plot": eval_plot_path,
        "ctrl_plots": ctrl_plot_paths,
        "metrics_npz": metrics_npz_path,
    }
    summary_path = os.path.join(plots_dir, f"{mode.name}_seed{seed}_summary.json")
    save_json(summary_path, summary)
    summary["summary_path"] = summary_path
    print(f"Saved summary: {summary_path}")
    return summary


# -----------------------------------------------------------------------------
# Seed-sweep summarization
# -----------------------------------------------------------------------------


def summarize_seed_runs(root: str, *, pattern: str, metric: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
    out_dir = ensure_dir(out_dir or os.path.join(root, "seed_sweep_summary"))
    rows = load_seed_summary_table(root, pattern=pattern)
    table_csv = write_rows_csv(os.path.join(out_dir, "seed_sweep_table.csv"), rows)
    summary = summarize_seed_table(rows, metric=metric)
    summary_json = save_json(os.path.join(out_dir, "seed_sweep_summary.json"), summary)
    boxplot = plot_seed_sweep_boxplot(
        os.path.join(out_dir, "seed_sweep_boxplot.png"),
        rows,
        metric=metric,
        title=f"seed sweep: {metric}",
    )
    result = {"root": root, "pattern": pattern, "metric": metric, "n_files": len(rows), "table_csv": table_csv, "summary_json": summary_json, "boxplot": boxplot, "summary": summary}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1, help="Single seed; ignored when --seeds is supplied")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="Run multiple seeds, replacing old seed_sweep.py run mode")
    ap.add_argument("--mode", choices=["premix", "production", "both"], default="both")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--batch_episodes", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_episodes", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--best_metric", choices=["ema_return", "mean_return"], default="ema_return")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--mp_start", type=str, default="spawn")
    ap.add_argument("--mp_chunksize", type=int, default=8)
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    ap.add_argument("--plots_dirname", type=str, default="plots")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--premix_duration", type=float, default=400.0)
    ap.add_argument("--premix_hold", type=float, default=10.0)
    ap.add_argument("--production_duration", type=float, default=600.0)
    ap.add_argument("--continue_on_fail", action="store_true", help="Continue multi-seed sweep even if one run fails")

    # seed_sweep.py summary mode replacement
    ap.add_argument("--summarize_only", action="store_true", help="Do not train; summarize existing *_summary.json files")
    ap.add_argument("--summary_root", type=str, default=None)
    ap.add_argument("--summary_pattern", type=str, default="**/*_summary.json")
    ap.add_argument("--summary_metric", type=str, default="eval_metrics.R_mean")
    ap.add_argument("--summary_out_dir", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.summarize_only:
        summarize_seed_runs(
            args.summary_root or args.ckpt_dir,
            pattern=args.summary_pattern,
            metric=args.summary_metric,
            out_dir=args.summary_out_dir,
        )
        return

    seeds = list(args.seeds) if args.seeds else [int(args.seed)]
    mode_names = ["premix", "production"] if args.mode == "both" else [args.mode]
    all_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for seed in seeds:
        set_global_seed(int(seed))
        seed_ckpt_root = args.ckpt_dir if len(seeds) == 1 else os.path.join(args.ckpt_dir, f"seed{seed}")
        for mode_name in mode_names:
            try:
                mode = make_mode_from_name(
                    mode_name,
                    premix_duration=float(args.premix_duration),
                    premix_hold=float(args.premix_hold),
                    production_duration=float(args.production_duration),
                )
                agent = make_agent_for_mode(mode, device=args.device, seed=int(seed))
                mode_ckpt_dir = seed_ckpt_root if len(mode_names) == 1 else os.path.join(seed_ckpt_root, mode.name)
                cfg = TrainConfig(
                    device=args.device,
                    seed=int(seed),
                    num_workers=int(args.num_workers),
                    mp_start=str(args.mp_start),
                    mp_chunksize=int(args.mp_chunksize),
                    batch_episodes=int(args.batch_episodes),
                    updates=int(args.updates),
                    ckpt_dir=mode_ckpt_dir,
                    save_every=int(args.save_every),
                    eval_episodes=int(args.eval_episodes),
                    eval_every=int(args.eval_every),
                    best_metric=str(args.best_metric),
                    plots_dirname=str(args.plots_dirname),
                )
                summary = train_one_mode(mode, agent, cfg, override_dt=float(args.dt))
                all_summaries.append(summary)
            except Exception as e:
                failure = {"seed": int(seed), "mode": mode_name, "error": str(e), "traceback": traceback.format_exc()}
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False, indent=2))
                if not args.continue_on_fail:
                    raise

    ensure_dir(args.ckpt_dir)
    save_json(os.path.join(args.ckpt_dir, "train_summary_all.json"), all_summaries)
    if failures:
        save_json(os.path.join(args.ckpt_dir, "train_failures.json"), failures)

    if len(seeds) > 1:
        summarize_seed_runs(args.ckpt_dir, pattern="**/*_summary.json", metric=args.summary_metric)


if __name__ == "__main__":
    main()
