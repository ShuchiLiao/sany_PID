"""
plotting.py

All plotting and artifact-saving helpers for the sany_PID RL refactor.

This file keeps visualization responsibilities out of train.py/evaluate.py while
preserving the old outputs: training curves, eval metric curves, paired-comparison
histograms/scatters/bucket plots, scale plots, trajectory comparison, and
seed-sweep boxplots.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# -----------------------------------------------------------------------------
# generic helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def save_metrics_npz(path: str, **arrays: Any) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    pack: Dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        if v is None:
            continue
        try:
            pack[k] = np.asarray(v)
        except Exception:
            pack[k] = np.asarray([v], dtype=object)
    np.savez_compressed(path, **pack)
    return path


def _arr(x: Iterable[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=np.float64)


def plot_one_series(
    out_png: str,
    y: Iterable[float],
    *,
    x: Optional[Iterable[float]] = None,
    xlabel: str = "index",
    ylabel: str = "value",
    title: str = "",
) -> str:
    ensure_dir(os.path.dirname(out_png) or ".")
    y_arr = _arr(y)
    x_arr = _arr(x) if x is not None else np.arange(y_arr.size)
    fig = plt.figure(figsize=(7, 4))
    plt.plot(x_arr, y_arr, linewidth=1.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


# -----------------------------------------------------------------------------
# training outputs
# -----------------------------------------------------------------------------


def plot_training_curves(
    mode_name: str,
    plots_dir: str,
    returns_mean: Sequence[float],
    returns_ema: Sequence[float],
    loss_pi: Sequence[float],
    loss_v: Sequence[float],
    entropy: Sequence[float],
) -> List[str]:
    ensure_dir(plots_dir)
    outs: List[str] = []
    series = {
        "train_R_mean": (returns_mean, "episode return mean"),
        "train_R_ema": (returns_ema, "episode return EMA"),
        "train_loss_pi": (loss_pi, "policy loss"),
        "train_loss_v": (loss_v, "value loss"),
        "train_entropy": (entropy, "entropy"),
    }
    for suffix, (values, ylabel) in series.items():
        outs.append(
            plot_one_series(
                os.path.join(plots_dir, f"{mode_name}_{suffix}.png"),
                values,
                xlabel="update",
                ylabel=ylabel,
                title=f"{mode_name}: {ylabel}",
            )
        )
    return outs


def plot_ctrl_metrics_curves(
    mode_name: str,
    plots_dir: str,
    eval_updates: Sequence[int],
    series: Mapping[str, Sequence[float]],
) -> Dict[str, str]:
    ensure_dir(plots_dir)
    outs: Dict[str, str] = {}
    for name, values in series.items():
        if values is None or len(values) == 0:
            continue
        outs[name] = plot_one_series(
            os.path.join(plots_dir, f"{mode_name}_eval_{name}.png"),
            values,
            x=eval_updates,
            xlabel="update",
            ylabel=name,
            title=f"{mode_name}: eval {name}",
        )
    return outs


def plot_eval_rewards(mode_name: str, plots_dir: str, rewards: Iterable[float], label: str = "best_eval") -> str:
    ensure_dir(plots_dir)
    rewards = _arr(rewards)
    out_png = os.path.join(plots_dir, f"{mode_name}_{label}_rewards_hist.png")
    fig = plt.figure(figsize=(6, 4))
    plt.hist(rewards[np.isfinite(rewards)], bins=30)
    plt.xlabel("episode reward")
    plt.ylabel("count")
    plt.title(f"{mode_name}: {label} reward distribution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


# -----------------------------------------------------------------------------
# trajectory outputs
# -----------------------------------------------------------------------------


def plot_trajectory(out_png: str, traj: Any, *, rho_sp: float, h_sp: Optional[float] = None, title: str = "") -> str:
    ensure_dir(os.path.dirname(out_png) or ".")
    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax1.plot(traj.t, traj.rho, linewidth=1.8, label="rho")
    ax1.axhline(float(rho_sp), linestyle="--", linewidth=1.2, label="rho_sp")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("density")
    ax1.legend(loc="best")
    if h_sp is not None and getattr(traj, "h", None) is not None:
        ax2 = ax1.twinx()
        ax2.plot(traj.t, traj.h, linewidth=1.2, label="h")
        ax2.axhline(float(h_sp), linestyle=":", linewidth=1.2, label="h_sp")
        ax2.set_ylabel("level")
    if title:
        plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_trajectory_compare(
    out_png: str,
    traj_base: Any,
    traj_rl: Any,
    *,
    rho_sp: float,
    h_sp: Optional[float] = None,
    title: str = "",
) -> str:
    ensure_dir(os.path.dirname(out_png) or ".")
    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax1.plot(traj_base.t, traj_base.rho, linewidth=1.8, label="baseline rho")
    ax1.plot(traj_rl.t, traj_rl.rho, linewidth=1.8, label="RL rho")
    ax1.axhline(float(rho_sp), linestyle="--", linewidth=1.2, label="rho_sp")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("density")
    ax1.legend(loc="best")
    if h_sp is not None and getattr(traj_base, "h", None) is not None and getattr(traj_rl, "h", None) is not None:
        ax2 = ax1.twinx()
        ax2.plot(traj_base.t, traj_base.h, linewidth=1.0, label="baseline h")
        ax2.plot(traj_rl.t, traj_rl.h, linewidth=1.0, label="RL h")
        ax2.axhline(float(h_sp), linestyle=":", linewidth=1.2, label="h_sp")
        ax2.set_ylabel("level")
    if title:
        plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


# -----------------------------------------------------------------------------
# paired-comparison plots: formerly evaluate_compare.py outputs
# -----------------------------------------------------------------------------


def plot_delta_hist(out_png: str, delta: Sequence[float], *, title: str = "paired return delta") -> str:
    ensure_dir(os.path.dirname(out_png) or ".")
    d = _arr(delta)
    d = d[np.isfinite(d)]
    fig = plt.figure(figsize=(6, 4))
    plt.hist(d, bins=30)
    plt.axvline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("RL - baseline")
    plt.ylabel("count")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_delta_scatter(
    out_png: str,
    x: Sequence[float],
    delta: Sequence[float],
    *,
    xlabel: str,
    ylabel: str = "return delta (RL - baseline)",
    title: str = "",
) -> str:
    ensure_dir(os.path.dirname(out_png) or ".")
    x_arr = _arr(x)
    d = _arr(delta)
    m = np.isfinite(x_arr) & np.isfinite(d)
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(x_arr[m], d[m], s=18, alpha=0.75)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_bucket_bar(
    out_png: str,
    x: Sequence[float],
    delta: Sequence[float],
    *,
    xlabel: str = "bucket variable",
    title: str = "bucketed paired delta",
    bins: int = 4,
) -> str:
    ensure_dir(os.path.dirname(out_png) or ".")
    x_arr = _arr(x)
    d = _arr(delta)
    m = np.isfinite(x_arr) & np.isfinite(d)
    x_arr = x_arr[m]
    d = d[m]
    if x_arr.size == 0:
        return plot_delta_hist(out_png, [], title=title)
    qs = np.nanpercentile(x_arr, np.linspace(0, 100, int(bins) + 1))
    # avoid duplicate edges
    qs = np.unique(qs)
    labels: List[str] = []
    means: List[float] = []
    errs: List[float] = []
    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i + 1]
        if i == len(qs) - 2:
            idx = (x_arr >= lo) & (x_arr <= hi)
        else:
            idx = (x_arr >= lo) & (x_arr < hi)
        vals = d[idx]
        if vals.size == 0:
            continue
        labels.append(f"{lo:.3g}-{hi:.3g}")
        means.append(float(np.nanmean(vals)))
        errs.append(float(np.nanstd(vals) / max(np.sqrt(vals.size), 1.0)))

    fig = plt.figure(figsize=(7, 4))
    xpos = np.arange(len(means))
    plt.bar(xpos, means, yerr=errs, capsize=3)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xticks(xpos, labels, rotation=25, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel("mean return delta")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_scale_boxplot(out_png: str, scales: Mapping[str, Sequence[float]], *, title: str = "RL scale distribution") -> str:
    ensure_dir(os.path.dirname(out_png) or ".")
    names = [k for k, v in scales.items() if len(v) > 0]
    vals = [_arr(scales[k]) for k in names]
    fig = plt.figure(figsize=(max(7, 0.7 * len(names)), 4))
    if vals:
        plt.boxplot(vals, labels=names, showfliers=False)
        plt.xticks(rotation=35, ha="right")
    plt.ylabel("scale")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_scales_scatter(
    out_png: str,
    x: Sequence[float],
    scales: Mapping[str, Sequence[float]],
    *,
    xlabel: str,
    title: str = "scale vs scenario",
) -> str:
    ensure_dir(os.path.dirname(out_png) or ".")
    x_arr = _arr(x)
    fig = plt.figure(figsize=(7, 4))
    for name, values in scales.items():
        y = _arr(values)
        n = min(x_arr.size, y.size)
        if n == 0:
            continue
        m = np.isfinite(x_arr[:n]) & np.isfinite(y[:n])
        plt.scatter(x_arr[:n][m], y[:n][m], s=14, alpha=0.7, label=name)
    plt.xlabel(xlabel)
    plt.ylabel("scale")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_compare_outputs(
    out_dir: str,
    mode_name: str,
    arrays: Mapping[str, Any],
    scales: Mapping[str, Sequence[float]],
) -> Dict[str, str]:
    ensure_dir(out_dir)
    baseline = np.asarray(arrays.get("baseline_rewards", []), dtype=np.float64)
    rl = np.asarray(arrays.get("rl_rewards", []), dtype=np.float64)
    delta = rl - baseline
    tau_mix = np.asarray(arrays.get("tau_mix_hat", []), dtype=np.float64)
    tau_delay = np.asarray(arrays.get("rho_obs_delay", []), dtype=np.float64)
    qs = np.asarray(arrays.get("qs", []), dtype=np.float64)

    outs = {
        "delta_hist": plot_delta_hist(os.path.join(out_dir, f"{mode_name}_delta_return_hist.png"), delta, title=f"{mode_name}: return delta"),
        "delta_vs_tau_mix": plot_delta_scatter(
            os.path.join(out_dir, f"{mode_name}_delta_vs_tau_mix.png"),
            tau_mix,
            delta,
            xlabel="tau_mix_hat (s)",
            title=f"{mode_name}: delta vs tau_mix",
        ),
        "delta_vs_delay": plot_delta_scatter(
            os.path.join(out_dir, f"{mode_name}_delta_vs_delay.png"),
            tau_delay,
            delta,
            xlabel="rho_obs_delay (s)",
            title=f"{mode_name}: delta vs delay",
        ),
        "scale_boxplot": plot_scale_boxplot(os.path.join(out_dir, f"{mode_name}_scale_boxplot.png"), scales),
        "scale_vs_tau_mix": plot_scales_scatter(
            os.path.join(out_dir, f"{mode_name}_scales_vs_tau_mix.png"),
            tau_mix,
            scales,
            xlabel="tau_mix_hat (s)",
            title=f"{mode_name}: scales vs tau_mix",
        ),
    }
    if qs.size and np.nanmax(np.abs(qs)) > 0:
        outs["delta_vs_qs"] = plot_delta_scatter(
            os.path.join(out_dir, f"{mode_name}_delta_vs_qs.png"),
            qs,
            delta,
            xlabel="Qs (m3/s)",
            title=f"{mode_name}: delta vs Qs",
        )
        outs["delta_qs_bucket"] = plot_bucket_bar(
            os.path.join(out_dir, f"{mode_name}_delta_qs_bucket.png"),
            qs,
            delta,
            xlabel="Qs bucket",
            title=f"{mode_name}: delta by Qs bucket",
        )
    else:
        outs["delta_tau_bucket"] = plot_bucket_bar(
            os.path.join(out_dir, f"{mode_name}_delta_tau_bucket.png"),
            tau_mix,
            delta,
            xlabel="tau_mix_hat bucket",
            title=f"{mode_name}: delta by tau_mix bucket",
        )
    return outs


# -----------------------------------------------------------------------------
# seed-sweep plots
# -----------------------------------------------------------------------------


def plot_seed_sweep_boxplot(
    out_png: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str = "eval_metrics.R_mean",
    title: str = "seed sweep summary",
) -> str:
    ensure_dir(os.path.dirname(out_png) or ".")
    by_mode: Dict[str, List[float]] = {}
    for row in rows:
        mode = str(row.get("mode", "unknown"))
        try:
            v = float(row.get(metric, np.nan))
        except Exception:
            v = float("nan")
        if np.isfinite(v):
            by_mode.setdefault(mode, []).append(v)
    names = sorted(by_mode.keys())
    vals = [np.asarray(by_mode[n], dtype=np.float64) for n in names]
    fig = plt.figure(figsize=(6, 4))
    if vals:
        plt.boxplot(vals, labels=names, showfliers=True)
    plt.ylabel(metric)
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png
