#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figs_2_6.py

Goal
----
Generate paper-ready Figure 2 ~ Figure 6 for either premix or production, by
1) Loading *saved* training/eval metric series from train_modes.py:
   {ckpt_root}/**/plots/{mode}_seed{seed}_metrics_series.npz
2) Loading *saved* large-sample paired comparison results from evaluate_compare.py:
   {compare_root}/**/{mode}_compare_results_seed{seed}_N{N}.npz

and optionally
3) (Re)running a few *typical* scenarios (Figure 3) using the saved best model.

This script is designed to work with your current repo layout and the exact keys
saved by train_modes.py and evaluate_compare.py (see the code you uploaded).

Usage examples
--------------
# premix figs
python make_figs_2_6.py --mode premix --ckpt_root ./checkpoints --compare_root ./checkpoints/compare --outdir ./paper_figs

# production figs
python make_figs_2_6.py --mode production --ckpt_root ./checkpoints --compare_root ./checkpoints/compare --outdir ./paper_figs

Notes
-----
- Figure 2: multi-seed training curves (mean±std + per-seed thin lines)
- Figure 3: typical scenarios trajectory comparisons (Base vs RL), selected by
  extreme tau_mix / tau_delay, and also Qs (production)
- Figure 4: large-sample paired performance plots (hist/CDF/buckets/scatter)
- Figure 5: robustness experiments (3 classes) run here and plotted
- Figure 6: scale distribution + scale-context heatmaps (multiple heatmaps)

"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: discovery & io
# -----------------------------

def _find_files_recursive(root: str, pattern: str) -> List[str]:
    root = os.path.abspath(root)
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))

def discover_metrics_series_npz(ckpt_root: str, mode: str) -> Dict[int, str]:
    """
    Find all: **/plots/{mode}_seed{seed}_metrics_series.npz
    Return {seed -> path}.
    """
    pat = f"{mode}_seed*_metrics_series.npz"
    paths = _find_files_recursive(ckpt_root, pat)
    out: Dict[int, str] = {}
    for p in paths:
        m = re.search(rf"{re.escape(mode)}_seed(\d+)_metrics_series\.npz$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items(), key=lambda kv: kv[0]))

def discover_summary_json(ckpt_root: str, mode: str) -> Dict[int, str]:
    pat = f"{mode}_seed*_summary.json"
    paths = _find_files_recursive(ckpt_root, pat)
    out: Dict[int, str] = {}
    for p in paths:
        m = re.search(rf"{re.escape(mode)}_seed(\d+)_summary\.json$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items(), key=lambda kv: kv[0]))

def discover_compare_npz(compare_root: str, mode: str, N: Optional[int] = None) -> Dict[int, str]:
    """
    Find compare results under:
      checkpoints/seed{train_seed}/compare/{mode}_compare_results_seed{eval_seed}_N{N}.npz

    IMPORTANT:
      The "seed" in filename is often the *evaluation rng seed* (e.g., 777),
      NOT the training seed. Therefore we key results by the *training seed*
      parsed from the directory name (seed1/seed2/...).

    Return {train_seed -> path}. If N is None, pick the largest N for each train_seed.
    """
    pat = f"{mode}_compare_results_seed*_N*.npz"
    paths = _find_files_recursive(compare_root, pat)

    by_train_seed: Dict[int, List[Tuple[int, str]]] = {}

    for p in paths:
        bn = os.path.basename(p)

        # Parse N from filename
        mN = re.search(rf"{re.escape(mode)}_compare_results_seed\d+_N(\d+)\.npz$", bn)
        if not mN:
            continue
        n = int(mN.group(1))
        if (N is not None) and (n != int(N)):
            continue

        # Parse training seed from path segment ".../seed{train_seed}/compare/..."
        # Works for both Windows and Linux paths
        mS = re.search(r"[\\/](seed)(\d+)[\\/]", p)
        if not mS:
            # fallback: try ".../seed_1/..." style
            mS = re.search(r"[\\/](seed_)(\d+)[\\/]", p)
        if not mS:
            # If still not found, skip (or you could key by full path)
            continue

        train_seed = int(mS.group(2))
        by_train_seed.setdefault(train_seed, []).append((n, p))

    out: Dict[int, str] = {}
    for train_seed, items in by_train_seed.items():
        items = sorted(items, key=lambda t: t[0])
        out[train_seed] = items[-1][1]  # largest N
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def mkdirp(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def _load_npz(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}

def _safe_get(d: Dict[str, np.ndarray], k: str, default=None):
    return d[k] if k in d else default


# -----------------------------
# Figure 2: multi-seed training curves
# -----------------------------

def fig2_training_curves(mode: str, metrics_paths: Dict[int, str], outdir: str) -> None:
    """
    Create multi-seed aggregate plots:
    - Train: returns_ema / returns_mean
    - Loss: policy/value/entropy
    - Eval-over-training: R_mean, IAE_rho, OS_rho, TV_uc (+ h metrics for production)
    """
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics_series.npz found for mode={mode}")

    # Load all series
    seeds = sorted(metrics_paths.keys())
    series = {}
    for s in seeds:
        series[s] = _load_npz(metrics_paths[s])

    # Determine aligned length for per-update training curves
    def stack_key(key: str) -> Tuple[np.ndarray, np.ndarray]:
        arrs = []
        minlen = None
        for s in seeds:
            a = np.asarray(_safe_get(series[s], key, []), dtype=np.float64)
            if a.size == 0:
                continue
            minlen = a.size if (minlen is None) else min(minlen, a.size)
            arrs.append(a)
        if not arrs or minlen is None:
            return np.empty((0, 0)), np.array([])
        A = np.stack([a[:minlen] for a in arrs], axis=0)  # [S, T]
        x = np.arange(1, minlen + 1)
        return A, x

    # ---- 2.1 returns ----
    A_ema, x = stack_key("train_returns_ema")
    A_mean, _ = stack_key("train_returns_mean")

    if A_ema.size > 0:
        plt.figure()
        # per-seed thin lines
        for i, s in enumerate(seeds):
            if "train_returns_ema" in series[s]:
                a = np.asarray(series[s]["train_returns_ema"], dtype=np.float64)[:len(x)]
                plt.plot(x, a, linewidth=0.8, alpha=0.25, label=(f"seed{s}" if i < 1 else None))
        # mean±std
        mu = np.mean(A_ema, axis=0)
        sd = np.std(A_ema, axis=0)
        plt.plot(x, mu, linewidth=2.0, label="mean(ema)")
        plt.fill_between(x, mu - sd, mu + sd, alpha=0.2, label="±1 std")
        plt.xlabel("update")
        plt.ylabel("return (EMA)")
        plt.title(f"Figure 2 ({mode}): Training return EMA across seeds")
        plt.grid(True, alpha=0.2)
        plt.legend(loc="best")
        savefig(os.path.join(outdir, f"{mode}_Fig2_train_return_ema_multiseed.png"))

    if A_mean.size > 0:
        plt.figure()
        mu = np.mean(A_mean, axis=0)
        sd = np.std(A_mean, axis=0)
        plt.plot(x, mu, linewidth=2.0, label="mean(return)")
        plt.fill_between(x, mu - sd, mu + sd, alpha=0.2, label="±1 std")
        plt.xlabel("update")
        plt.ylabel("return (mean)")
        plt.title(f"Figure 2 ({mode}): Training return mean across seeds")
        plt.grid(True, alpha=0.2)
        plt.legend(loc="best")
        savefig(os.path.join(outdir, f"{mode}_Fig2_train_return_mean_multiseed.png"))

    # ---- 2.2 losses ----
    for key, ylabel in [
        ("train_loss_pi", "policy loss"),
        ("train_loss_v", "value loss"),
        ("train_entropy", "entropy"),
    ]:
        A, x = stack_key(key)
        if A.size == 0:
            continue
        plt.figure()
        mu = np.mean(A, axis=0)
        sd = np.std(A, axis=0)
        plt.plot(x, mu, linewidth=2.0, label=f"mean({key})")
        plt.fill_between(x, mu - sd, mu + sd, alpha=0.2, label="±1 std")
        plt.xlabel("update")
        plt.ylabel(ylabel)
        plt.title(f"Figure 2 ({mode}): {key} across seeds")
        plt.grid(True, alpha=0.2)
        plt.legend(loc="best")
        savefig(os.path.join(outdir, f"{mode}_Fig2_{key}_multiseed.png"))

    # ---- 2.3 eval-over-training (ctrl metrics) ----
    # Note: eval series are at sparse "eval_updates" indices; align by intersection.
    # We'll first collect updates for each seed, then reindex onto common set.
    eval_keys_common = ["eval_R_mean", "eval_IAE_rho", "eval_OS_rho", "eval_TV_uc"]
    if mode == "production":
        eval_keys_common += ["eval_IAE_h", "eval_OS_h", "eval_TV_uw"]

    # find common update points
    upd_lists = []
    for s in seeds:
        u = np.asarray(_safe_get(series[s], "eval_updates", []), dtype=np.int64)
        if u.size > 0:
            upd_lists.append(u)
    if upd_lists:
        common = set(upd_lists[0].tolist())
        for u in upd_lists[1:]:
            common &= set(u.tolist())
        common = np.array(sorted(common), dtype=np.int64)

        def eval_stack(key: str) -> Tuple[np.ndarray, np.ndarray]:
            arrs = []
            for s in seeds:
                u = np.asarray(_safe_get(series[s], "eval_updates", []), dtype=np.int64)
                y = np.asarray(_safe_get(series[s], key, []), dtype=np.float64)
                if u.size == 0 or y.size == 0:
                    continue
                # map u->y
                mp = {int(uu): float(y[i]) for i, uu in enumerate(u.tolist()) if i < len(y)}
                arrs.append(np.array([mp[int(c)] for c in common], dtype=np.float64))
            if not arrs:
                return np.empty((0, 0)), np.array([])
            return np.stack(arrs, axis=0), common

        for key in eval_keys_common:
            A, x_eval = eval_stack(key)
            if A.size == 0:
                continue
            plt.figure()
            mu = np.mean(A, axis=0)
            sd = np.std(A, axis=0)
            plt.plot(x_eval, mu, linewidth=2.0, label=f"mean({key})")
            plt.fill_between(x_eval, mu - sd, mu + sd, alpha=0.2, label="±1 std")
            plt.xlabel("update")
            plt.ylabel(key.replace("eval_", ""))
            plt.title(f"Figure 2 ({mode}): Eval {key.replace('eval_', '')} over training")
            plt.grid(True, alpha=0.2)
            plt.legend(loc="best")
            savefig(os.path.join(outdir, f"{mode}_Fig2_eval_{key}.png"))


# -----------------------------
# Figure 4: large-sample paired comparison (from evaluate_compare npz)
# -----------------------------

def fig4_large_sample(mode: str, compare_paths: Dict[int, str], outdir: str) -> None:
    """
    Aggregate across seeds:
    - ΔReturn histogram + CDF
    - ΔIAE_rho histogram (+ ΔIAE_h for production)
    - mean ΔReturn by tau_mix_hat quantile bins, by |ΔQs| quantile bins (if provided)
    """
    if not compare_paths:
        raise FileNotFoundError(f"No compare_results_seed*_N*.npz found for mode={mode}")

    seeds = sorted(compare_paths.keys())
    all_dR = []
    all_tau = []
    all_td = []
    all_dQs = []
    all_dIAE_rho = []
    all_dIAE_h = []

    for s in seeds:
        z = _load_npz(compare_paths[s])
        dR = np.asarray(z["dR"], dtype=np.float64)
        all_dR.append(dR)
        all_tau.append(np.asarray(z.get("tau_mix_hat", np.full_like(dR, np.nan)), dtype=np.float64))
        all_td.append(np.asarray(z.get("tau_delay", np.full_like(dR, np.nan)), dtype=np.float64))
        all_dQs.append(np.asarray(z.get("dQs", np.full_like(dR, np.nan)), dtype=np.float64))
        d_iae_rho = np.asarray(z["iae_rho_rl"] - z["iae_rho_base"], dtype=np.float64)
        all_dIAE_rho.append(d_iae_rho)
        if mode == "production" and ("iae_h_rl" in z and "iae_h_base" in z):
            all_dIAE_h.append(np.asarray(z["iae_h_rl"] - z["iae_h_base"], dtype=np.float64))

    dR = np.concatenate(all_dR, axis=0)
    tau = np.concatenate(all_tau, axis=0)
    td = np.concatenate(all_td, axis=0)
    dQs = np.concatenate(all_dQs, axis=0)
    dIAE_rho = np.concatenate(all_dIAE_rho, axis=0)
    dIAE_h = np.concatenate(all_dIAE_h, axis=0) if all_dIAE_h else None

    # ---- 4.1 ΔReturn hist ----
    plt.figure()
    plt.hist(dR, bins=60)
    plt.title(f"Figure 4 ({mode}): ΔReturn histogram (RL - Base), pooled seeds")
    plt.xlabel("ΔReturn")
    plt.ylabel("count")
    plt.grid(True, alpha=0.2)
    savefig(os.path.join(outdir, f"{mode}_Fig4_delta_return_hist_pooled.png"))

    # ---- 4.2 ΔReturn CDF ----
    xs = np.sort(dR)
    ys = np.linspace(0, 1, xs.size, endpoint=False)
    plt.figure()
    plt.plot(xs, ys, linewidth=2.0)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.title(f"Figure 4 ({mode}): ΔReturn empirical CDF, pooled seeds")
    plt.xlabel("ΔReturn")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.2)
    savefig(os.path.join(outdir, f"{mode}_Fig4_delta_return_cdf_pooled.png"))

    # ---- 4.3 ΔIAE_rho hist ----
    plt.figure()
    plt.hist(dIAE_rho, bins=60)
    plt.title(f"Figure 4 ({mode}): ΔIAE_rho histogram (RL - Base), pooled seeds")
    plt.xlabel("ΔIAE_rho  (lower is better)")
    plt.ylabel("count")
    plt.grid(True, alpha=0.2)
    savefig(os.path.join(outdir, f"{mode}_Fig4_delta_IAE_rho_hist_pooled.png"))

    if dIAE_h is not None:
        plt.figure()
        plt.hist(dIAE_h, bins=60)
        plt.title(f"Figure 4 ({mode}): ΔIAE_h histogram (RL - Base), pooled seeds")
        plt.xlabel("ΔIAE_h  (lower is better)")
        plt.ylabel("count")
        plt.grid(True, alpha=0.2)
        savefig(os.path.join(outdir, f"{mode}_Fig4_delta_IAE_h_hist_pooled.png"))

    # ---- 4.4 bucket bars (tau_mix_hat, |ΔQs|) ----
    def bucket_bar(x: np.ndarray, y: np.ndarray, name: str, nbins: int = 5):
        msk = np.isfinite(x) & np.isfinite(y)
        x = x[msk]; y = y[msk]
        if x.size < 10:
            return
        qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
        # de-dup edges
        qs = np.unique(qs)
        if qs.size <= 2:
            return
        # assign bin
        bins = np.digitize(x, qs[1:-1], right=False)
        means = []
        labels = []
        for b in range(qs.size - 1):
            yy = y[bins == b]
            means.append(float(np.mean(yy)) if yy.size else float("nan"))
            labels.append(f"[{qs[b]:.2g},{qs[b+1]:.2g}]")
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(means)), means)
        plt.xticks(np.arange(len(means)), labels, rotation=20, ha="right")
        plt.title(f"Figure 4 ({mode}): mean ΔReturn by {name} quantile bins (pooled)")
        plt.ylabel("mean ΔReturn")
        plt.grid(True, axis="y", alpha=0.2)
        savefig(os.path.join(outdir, f"{mode}_Fig4_bucket_mean_dR_by_{name}.png"))

    bucket_bar(tau, dR, "tau_mix_hat")
    bucket_bar(np.abs(dQs), dR, "abs_dQs")


# -----------------------------
# Figure 6: scales distribution + scale-context heatmaps
# -----------------------------

def _bin2d_mean(x: np.ndarray, y: np.ndarray, z: np.ndarray, nx: int, ny: int):
    """
    Return grid means with shape [ny, nx] and x_edges, y_edges.
    """
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[m]; y = y[m]; z = z[m]
    if x.size < 10:
        return None

    x_edges = np.quantile(x, np.linspace(0, 1, nx + 1))
    y_edges = np.quantile(y, np.linspace(0, 1, ny + 1))
    # ensure monotonic
    x_edges = np.unique(x_edges)
    y_edges = np.unique(y_edges)
    if x_edges.size <= 2 or y_edges.size <= 2:
        return None

    # map to bins
    xi = np.digitize(x, x_edges[1:-1], right=False)
    yi = np.digitize(y, y_edges[1:-1], right=False)
    grid = np.full((y_edges.size - 1, x_edges.size - 1), np.nan, dtype=np.float64)
    for j in range(grid.shape[0]):
        for i in range(grid.shape[1]):
            m2 = (xi == i) & (yi == j)
            if np.any(m2):
                grid[j, i] = float(np.mean(z[m2]))
    return grid, x_edges, y_edges

def fig6_scale_heatmaps(mode: str, compare_paths: Dict[int, str], outdir: str, nx: int = 6, ny: int = 6) -> None:
    """
    Produce several 2D heatmaps of mean scale values across context variable pairs.
    Uses evaluate_compare saved arrays:
      - tau_mix_hat, tau_delay, dQs
      - scales_mat [N, D], scale_names [D]
    We pool across seeds to increase sample size.

    For premix: still plot tau_mix_hat vs tau_delay; optionally vs |ΔQs| if present.
    For production: plot all three pairs:
      (tau_mix_hat, tau_delay), (tau_mix_hat, |ΔQs|), (tau_delay, |ΔQs|).
    """
    if not compare_paths:
        raise FileNotFoundError(f"No compare_results npz found for mode={mode}")

    # concat
    tau_all, td_all, dQs_all, S_all = [], [], [], []
    scale_names = None
    for s, p in sorted(compare_paths.items()):
        z = _load_npz(p)
        tau_all.append(np.asarray(z.get("tau_mix_hat", []), dtype=np.float64))
        td_all.append(np.asarray(z.get("tau_delay", []), dtype=np.float64))
        dQs_all.append(np.asarray(z.get("dQs", []), dtype=np.float64))
        S_all.append(np.asarray(z.get("scales_mat", []), dtype=np.float64))
        if scale_names is None and "scale_names" in z:
            scale_names = [str(x) for x in z["scale_names"].tolist()]

    tau = np.concatenate(tau_all, axis=0)
    td = np.concatenate(td_all, axis=0)
    dQs = np.concatenate(dQs_all, axis=0)
    S = np.concatenate(S_all, axis=0)

    if scale_names is None:
        scale_names = [f"s{i}" for i in range(S.shape[1])]

    # 6.1 distribution (pooled)
    plt.figure(figsize=(10, 4))
    plt.boxplot([S[:, j] for j in range(S.shape[1])], tick_labels=scale_names, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.title(f"Figure 6 ({mode}): scale distributions (pooled seeds)")
    plt.ylabel("scale value")
    plt.grid(True, axis="y", alpha=0.2)
    savefig(os.path.join(outdir, f"{mode}_Fig6_scales_boxplot_pooled.png"))

    # 6.2 heatmaps
    pairs = [("tau_mix_hat", tau, "tau_delay", td)]
    if np.isfinite(dQs).any():
        pairs += [("tau_mix_hat", tau, "abs_dQs", np.abs(dQs)),
                  ("tau_delay", td, "abs_dQs", np.abs(dQs))]

    heat_dir = mkdirp(os.path.join(outdir, f"{mode}_Fig6_heatmaps"))
    for j, name in enumerate(scale_names):
        zj = S[:, j]
        for (xn, x, yn, y) in pairs:
            out = _bin2d_mean(x, y, zj, nx=nx, ny=ny)
            if out is None:
                continue
            grid, x_edges, y_edges = out
            plt.figure(figsize=(6, 5))
            im = plt.imshow(grid, origin="lower", aspect="auto")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"{mode} Figure 6: mean({name}) over {xn}×{yn}")
            plt.xlabel(xn + " (quantile bins)")
            plt.ylabel(yn + " (quantile bins)")
            plt.xticks(np.arange(grid.shape[1]), [f"{x_edges[i]:.2g}" for i in range(grid.shape[1])], rotation=45, ha="right")
            plt.yticks(np.arange(grid.shape[0]), [f"{y_edges[i]:.2g}" for i in range(grid.shape[0])])
            plt.grid(False)
            savefig(os.path.join(heat_dir, f"{mode}_Fig6_heat_{name}_vs_{xn}_x_{yn}.png"))


# -----------------------------
# Figure 3 & 5 require simulation using best model
# -----------------------------

@dataclass
class LoadedAgent:
    agent: object
    mode_spec: object

def _import_repo_symbols():
    """
    Import your repo modules (modes.py, train_modes.py, PPO_bandit.py).
    Assumes this script is run from the repo root or that PYTHONPATH includes it.
    """
    import importlib
    modes = importlib.import_module("modes")
    train_modes = importlib.import_module("train_modes")
    PPO_bandit = importlib.import_module("PPO_bandit")
    return modes, train_modes, PPO_bandit

def _load_best_checkpoint_from_summary(ckpt_root: str, mode: str) -> Tuple[int, str]:
    """
    Choose one 'best' seed based on summary.json eval_metrics.R_mean.
    Return (seed, best_path).
    """
    sum_paths = discover_summary_json(ckpt_root, mode)
    if not sum_paths:
        # fallback to default ckpt name (may be overwritten by last run)
        default = os.path.join(ckpt_root, f"{mode}_best.pt")
        if os.path.exists(default):
            return 0, default
        raise FileNotFoundError(f"No summary.json found for mode={mode} under {ckpt_root}")

    best_seed = None
    best_score = -1e18
    best_path = None
    for seed, p in sum_paths.items():
        try:
            with open(p, "r", encoding="utf-8") as f:
                s = json.load(f)
            score = float(s.get("eval_metrics", {}).get("R_mean", -1e18))
            bp = str(s.get("best_path", ""))
            if score > best_score and bp and os.path.exists(bp):
                best_score = score
                best_seed = seed
                best_path = bp
        except Exception:
            continue
    if best_path is None:
        # fallback: choose first summary and its best_path even if not exists
        seed = sorted(sum_paths.keys())[0]
        with open(sum_paths[seed], "r", encoding="utf-8") as f:
            s = json.load(f)
        bp = str(s.get("best_path", ""))
        if not bp:
            raise FileNotFoundError("summary.json exists but missing best_path")
        return seed, bp
    return int(best_seed), str(best_path)

def _load_agent_from_ckpt(ckpt_path: str):
    modes, train_modes, PPO_bandit = _import_repo_symbols()
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    obs_dim = int(ckpt.get("obs_dim", cfg_dict.get("obs_dim", 0)))
    act_dim = int(ckpt.get("act_dim", cfg_dict.get("act_dim", 0)))

    # PPOConfig lives in PPO_bandit.py
    PPOConfig = getattr(PPO_bandit, "PPOConfig")
    PPOBanditAgent = getattr(PPO_bandit, "PPOBanditAgent")

    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent = PPOBanditAgent(cfg, device="cpu")
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()

    # build mode spec
    if ckpt.get("mode", "") == "premix":
        mode_spec = modes.make_premix_mode(duration_default=400.0, hold_default=10.0)
    else:
        mode_spec = modes.make_production_mode(duration_default=600.0)
    return agent, mode_spec, modes, train_modes

def _scenario_from_template(mode_spec, rng, overrides: Dict[str, float]):
    """
    Sample a baseline scenario then override selected fields.
    """
    p = mode_spec.sample_episode(rng)
    for k, v in overrides.items():
        setattr(p, k, float(v))
    return p

def fig3_typical_trajectories(mode: str, ckpt_root: str, outdir: str) -> None:
    """
    Produce 4 (premix) or 6 (production) typical scenario trajectory comparisons:
      premix: tau_mix low/high × tau_delay low/high (4 cases)
      production: above + Qs low/high (2 additional, keeping mid tau_mix/tau_delay)
    Output multiple pngs into {outdir}/{mode}_Fig3_typical/
    """
    seed_best, ckpt_path = _load_best_checkpoint_from_summary(ckpt_root, mode)
    agent, mode_spec, modes, train_modes = _load_agent_from_ckpt(ckpt_path)

    rng = np.random.default_rng(12345 + (seed_best if seed_best is not None else 0))
    outd = mkdirp(os.path.join(outdir, f"{mode}_Fig3_typical"))

    # pick extreme values based on modes.py ranges (use conservative endpoints)
    # You can adjust these numbers to match your paper setup.
    if mode == "premix":
        tm_lo, tm_hi = 10.0, 60.0
        td_lo, td_hi = 0.0, 10.0
        cases = [
            ("tau_mix_low__tau_delay_low",  dict(tau_mix=tm_lo, tau_delay=td_lo)),
            ("tau_mix_low__tau_delay_high", dict(tau_mix=tm_lo, tau_delay=td_hi)),
            ("tau_mix_high__tau_delay_low", dict(tau_mix=tm_hi, tau_delay=td_lo)),
            ("tau_mix_high__tau_delay_high",dict(tau_mix=tm_hi, tau_delay=td_hi)),
        ]
    else:
        tm_lo, tm_hi = 10.0, 60.0
        td_lo, td_hi = 0.0, 10.0
        qs_lo, qs_hi = 0.4/60.0, 1.5/60.0
        qs_mid = 0.8/60.0
        cases = [
            ("tau_mix_low__tau_delay_low",  dict(tau_mix=tm_lo, tau_delay=td_lo, qs=qs_mid)),
            ("tau_mix_low__tau_delay_high", dict(tau_mix=tm_lo, tau_delay=td_hi, qs=qs_mid)),
            ("tau_mix_high__tau_delay_low", dict(tau_mix=tm_hi, tau_delay=td_lo, qs=qs_mid)),
            ("tau_mix_high__tau_delay_high",dict(tau_mix=tm_hi, tau_delay=td_hi, qs=qs_mid)),
            ("Qs_low",  dict(tau_mix=30.0, tau_delay=5.0, qs=qs_lo)),
            ("Qs_high", dict(tau_mix=30.0, tau_delay=5.0, qs=qs_hi)),
        ]

    # we will use the same plotting helper from evaluate_compare if available; otherwise do minimal plot.
    try:
        from evaluate_compare import plot_trajectory_compare, apply_scales_to_params  # type: ignore
        have_eval_plot = True
    except Exception:
        have_eval_plot = False
        apply_scales_to_params = None

    # action mapping function
    action_to_scales = getattr(modes, "action_to_scales")

    for i, (tag, ov) in enumerate(cases, start=1):
        p = _scenario_from_template(mode_spec, rng, ov)
        p.dt = 0.5
        obs = mode_spec.build_context(p)

        if hasattr(agent, "act_deterministic"):
            a, _, _ = agent.act_deterministic(obs)
        else:
            a, _, _ = agent.act(obs)
        rl_scales = action_to_scales(a, mode_spec.action_specs)
        base_scales = {spec.name: 1.0 for spec in mode_spec.action_specs}
        base_params = mode_spec.compute_base_params(p)

        traj_base = train_modes.simulate_episode(p, base_scales, rng, return_traj=True, base_params=base_params)
        traj_rl = train_modes.simulate_episode(p, rl_scales, rng, return_traj=True, base_params=base_params)

        out_path = os.path.join(outd, f"{mode}_Fig3_{i:02d}_{tag}.png")
        if have_eval_plot:
            params_base = apply_scales_to_params(base_params, base_scales)
            params_rl = apply_scales_to_params(base_params, rl_scales)
            plot_trajectory_compare(
                out_path,
                traj_base,
                traj_rl,
                p,
                title_prefix=f"{mode} (best seed {seed_best})",
                params_base=params_base,
                params_rl=params_rl,
            )
        else:
            # minimal plot
            if mode == "premix":
                plt.figure(figsize=(7, 3))
                plt.plot(traj_base.t, traj_base.rho, label="base")
                plt.plot(traj_rl.t, traj_rl.rho, label="RL")
                plt.axhline(float(p.rho_sp), linestyle="--", linewidth=1)
                plt.xlabel("t (s)")
                plt.ylabel("rho (kg/m^3)")
                plt.title(f"{mode} typical: {tag}")
                plt.grid(True, alpha=0.2)
                plt.legend(loc="best")
                savefig(out_path)
            else:
                fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
                axes[0].plot(traj_base.t, traj_base.h, label="base")
                axes[0].plot(traj_rl.t, traj_rl.h, label="RL")
                axes[0].axhline(float(p.h_sp), linestyle="--", linewidth=1)
                axes[0].set_ylabel("h (m)")
                axes[0].grid(True, alpha=0.2)
                axes[0].legend(loc="best")
                axes[1].plot(traj_base.t, traj_base.rho, label="base")
                axes[1].plot(traj_rl.t, traj_rl.rho, label="RL")
                axes[1].axhline(float(p.rho_sp), linestyle="--", linewidth=1)
                axes[1].set_ylabel("rho (kg/m^3)")
                axes[1].set_xlabel("t (s)")
                axes[1].grid(True, alpha=0.2)
                axes[1].legend(loc="best")
                fig.suptitle(f"{mode} typical: {tag}")
                fig.tight_layout()
                fig.savefig(out_path, dpi=220, bbox_inches="tight")
                plt.close(fig)

    print(f"[Fig3] Saved typical trajectories to: {outd}")

def fig5_robustness(mode: str, ckpt_root: str, outdir: str, M: int = 600, levels: int = 6) -> None:
    """
    Three robustness experiments (Figure 5), each produces a curve of mean ΔReturn vs perturbation level:
      A) tau uncertainty: tau_mix_hat and tau_delay perturbed (±%)
      B) valve max_flow uncertainty: wv_max_flow/cv_max_flow perturbed (±%)
      C) production only: demand disturbance |ΔQs| injected (step disturbance amplitude)

    We run M random scenarios per level (uses mode.sample_episode), evaluate Base vs RL using best model,
    and plot ΔReturn mean±std across scenarios.
    """
    seed_best, ckpt_path = _load_best_checkpoint_from_summary(ckpt_root, mode)
    agent, mode_spec, modes, train_modes = _load_agent_from_ckpt(ckpt_path)
    action_to_scales = getattr(modes, "action_to_scales")

    rng = np.random.default_rng(20240229 + (seed_best if seed_best is not None else 0))
    outd = mkdirp(os.path.join(outdir, f"{mode}_Fig5_robustness"))

    # baseline scales
    base_scales = {spec.name: 1.0 for spec in mode_spec.action_specs}

    # perturbation levels
    frac_levels = np.linspace(0.0, 0.5, levels)  # 0% .. 50%
    dQs_levels = np.linspace(0.0, 0.6/60.0, levels)  # 0 .. 0.6/60 (tune as needed)

    def eval_one(p, override: Dict[str, float]) -> float:
        # override a subset of episode parameters before building obs/base
        p2 = p
        for k, v in override.items():
            setattr(p2, k, float(v))
        p2.dt = 0.5

        # RL scales depend on obs (context)
        obs = mode_spec.build_context(p2)
        if hasattr(agent, "act_deterministic"):
            a, _, _ = agent.act_deterministic(obs)
        else:
            a, _, _ = agent.act(obs)
        rl_scales = action_to_scales(a, mode_spec.action_specs)

        base_params = mode_spec.compute_base_params(p2)
        Rb = float(train_modes.simulate_episode(p2, base_scales, rng, return_traj=False, base_params=base_params))
        Rr = float(train_modes.simulate_episode(p2, rl_scales, rng, return_traj=False, base_params=base_params))
        return Rr - Rb

    def run_curve(kind: str):
        xs = []
        mu = []
        sd = []
        for li in range(levels):
            if kind == "tau_uncertainty":
                frac = float(frac_levels[li])
                deltas = []
                for _ in range(M):
                    p = mode_spec.sample_episode(rng)
                    # perturb tau_mix and tau_delay multiplicatively
                    tm = float(p.tau_mix) * (1.0 + rng.normal(0.0, frac))
                    td = float(p.tau_delay) * (1.0 + rng.normal(0.0, frac))
                    tm = float(max(1e-6, tm))
                    td = float(max(0.0, td))
                    deltas.append(eval_one(p, {"tau_mix": tm, "tau_delay": td}))
                xs.append(frac)
            elif kind == "maxflow_uncertainty":
                frac = float(frac_levels[li])
                deltas = []
                for _ in range(M):
                    p = mode_spec.sample_episode(rng)
                    # perturb valve max flows multiplicatively
                    wv = float(p.water_valve_max_flow) * (1.0 + rng.normal(0.0, frac))
                    cv = float(p.cement_valve_max_flow) * (1.0 + rng.normal(0.0, frac))
                    wv = float(max(1e-9, wv))
                    cv = float(max(1e-9, cv))
                    deltas.append(eval_one(p, {"water_valve_max_flow": wv, "cement_valve_max_flow": cv}))
                xs.append(frac)
            elif kind == "dQs_disturbance":
                if mode != "production":
                    return
                amp = float(dQs_levels[li])
                deltas = []
                for _ in range(M):
                    p = mode_spec.sample_episode(rng)
                    # your evaluate_compare measures dQs as |ΔQs|; here we emulate by setting qs then expecting env step to use it
                    # NOTE: if your simulator uses p.qs directly (env.Q_s fixed), this creates different operating points, not a disturbance.
                    # If you have a proper dQs disturbance hook, replace this with that hook.
                    q0 = float(p.qs)
                    p.qs = float(max(1e-9, q0 + (amp if rng.random() < 0.5 else -amp)))
                    deltas.append(eval_one(p, {}))
                xs.append(amp)
            else:
                raise ValueError(kind)

            deltas_np = np.asarray(deltas, dtype=np.float64)
            mu.append(float(np.mean(deltas_np)))
            sd.append(float(np.std(deltas_np)))

        # plot
        plt.figure()
        plt.plot(xs, mu, linewidth=2.0, label="mean ΔReturn")
        plt.fill_between(xs, np.array(mu) - np.array(sd), np.array(mu) + np.array(sd), alpha=0.2, label="±1 std")
        plt.axhline(0.0, linestyle="--", linewidth=1.0)
        plt.grid(True, alpha=0.2)
        if kind == "dQs_disturbance":
            plt.xlabel("|ΔQs| amplitude")
        else:
            plt.xlabel("perturbation std (fraction)")
        plt.ylabel("ΔReturn (RL - Base)")
        plt.title(f"Figure 5 ({mode}): robustness curve - {kind}")
        plt.legend(loc="best")
        savefig(os.path.join(outd, f"{mode}_Fig5_{kind}.png"))

    run_curve("tau_uncertainty")
    run_curve("maxflow_uncertainty")
    if mode == "production":
        run_curve("dQs_disturbance")

    print(f"[Fig5] Saved robustness plots to: {outd}")


# -----------------------------
# Orchestration
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["premix", "production"])
    ap.add_argument("--ckpt_root", type=str, default="./checkpoints")
    ap.add_argument("--compare_root", type=str, default="./checkpoints/compare")
    ap.add_argument("--outdir", type=str, default="./paper_figs")
    ap.add_argument("--no_fig3", action="store_true")
    ap.add_argument("--no_fig5", action="store_true")
    ap.add_argument("--robust_M", type=int, default=600)
    ap.add_argument("--heat_nx", type=int, default=6)
    ap.add_argument("--heat_ny", type=int, default=6)
    args = ap.parse_args()

    mode = args.mode
    outdir_name = f"{args.outdir}_{mode}"
    outdir = mkdirp(os.path.abspath(outdir_name))

    # Discover saved data
    metrics_paths = discover_metrics_series_npz(args.ckpt_root, mode)
    compare_paths = discover_compare_npz(args.compare_root, mode, N=None)

    print(f"[discover] mode={mode}")
    print(f"  metrics_series: {len(metrics_paths)} seeds -> {list(metrics_paths.keys())}")
    print(f"  compare_npz    : {len(compare_paths)} seeds -> {list(compare_paths.keys())}")

    # Figure 2
    fig2_training_curves(mode, metrics_paths, outdir)

    # Figure 4
    fig4_large_sample(mode, compare_paths, outdir)

    # Figure 6
    fig6_scale_heatmaps(mode, compare_paths, outdir, nx=int(args.heat_nx), ny=int(args.heat_ny))

    # Figure 3 (typical scenarios)
    if not args.no_fig3:
        fig3_typical_trajectories(mode, args.ckpt_root, outdir)

    # Figure 5 (robustness)
    if not args.no_fig5:
        fig5_robustness(mode, args.ckpt_root, outdir, M=int(args.robust_M))

    print(f"All done. Outputs in: {outdir}")

if __name__ == "__main__":
    main()

# python make_figs.py --mode premix --ckpt_root ./checkpoints --compare_root ./checkpoints --outdir ./paper_figs
# python make_figs.py --mode production --ckpt_root ./checkpoints --compare_root ./checkpoints --outdir ./paper_figs