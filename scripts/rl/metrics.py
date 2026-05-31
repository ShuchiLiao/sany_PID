"""
metrics.py

Shared metrics/statistics layer for the sany_PID RL refactor.

This module combines the scalar control metrics that were formerly embedded in
train_modes.py with paired-comparison statistics and seed-sweep aggregation that
were formerly embedded in evaluate_compare.py / seed_sweep.py.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parents[1]
for _p in (_SCRIPTS / "core", _SCRIPTS / "PID_control", _SCRIPTS / "rl", _SCRIPTS):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

from modes import EpisodeParams, Trajectory  # noqa: E402


# -----------------------------------------------------------------------------
# Basic scalar control metrics
# -----------------------------------------------------------------------------


def safe_array(x: Any) -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def total_variation(x: Any) -> float:
    arr = safe_array(x)
    if arr.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(arr))))


def settling_time(t: np.ndarray, y: np.ndarray, sp: float, band: float) -> float:
    """First time after which y stays within absolute band; NaN if not settled."""
    t = safe_array(t)
    y = safe_array(y)
    if t.size == 0 or y.size == 0:
        return float("nan")
    ok = np.abs(y - float(sp)) <= float(band)
    if not np.any(ok):
        return float("nan")
    suffix_ok = np.flip(np.cumprod(np.flip(ok.astype(np.int8)))).astype(bool)
    idx = np.where(suffix_ok)[0]
    return float(t[int(idx[0])]) if idx.size else float("nan")


def compute_basic_metrics(traj: Trajectory, p: EpisodeParams) -> Dict[str, float]:
    t = safe_array(traj.t)
    rho = safe_array(traj.rho)
    h = safe_array(traj.h)
    dt = float(getattr(p, "dt", 1.0))
    rho_sp = float(getattr(p, "rho_sp", 0.0))
    h_sp = float(getattr(p, "h_sp", 0.0))

    rho_err = rho - rho_sp
    out: Dict[str, float] = {
        "IAE_rho": float(np.sum(np.abs(rho_err)) * dt) if rho.size else float("nan"),
        "MAE_rho": float(np.mean(np.abs(rho_err))) if rho.size else float("nan"),
        "RMSE_rho": float(math.sqrt(np.mean(rho_err**2))) if rho.size else float("nan"),
        "overshoot_rho": float(np.max(rho) - rho_sp) if rho.size else float("nan"),
        "undershoot_rho": float(rho_sp - np.min(rho)) if rho.size else float("nan"),
        "max_abs_err_rho": float(np.max(np.abs(rho_err))) if rho.size else float("nan"),
        "settling_time_rho_2pct": settling_time(t, rho, rho_sp, 0.02 * max(abs(rho_sp), 1.0)),
        "TV_uc": total_variation(getattr(traj, "u_c", None)),
        "TV_uw": total_variation(getattr(traj, "u_w", None)),
        "failed": float(bool(getattr(traj, "failed", False))),
    }

    if getattr(p, "mode", "premix") == "production" and h.size:
        h_err = h - h_sp
        out.update(
            {
                "IAE_h": float(np.sum(np.abs(h_err)) * dt),
                "MAE_h": float(np.mean(np.abs(h_err))),
                "RMSE_h": float(math.sqrt(np.mean(h_err**2))),
                "overshoot_h": float(np.max(h) - h_sp),
                "undershoot_h": float(h_sp - np.min(h)),
                "max_abs_err_h": float(np.max(np.abs(h_err))),
                "settling_time_h_2pct": settling_time(t, h, h_sp, 0.02 * max(abs(h_sp), 1e-6)),
            }
        )
    return out


def aggregate_metric_dicts(rows: Iterable[Mapping[str, float]], prefix: str = "") -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for row in rows:
        for k, v in row.items():
            try:
                fv = float(v)
            except Exception:
                continue
            acc.setdefault(k, []).append(fv)

    out: Dict[str, float] = {}
    for k, values in acc.items():
        arr = np.asarray(values, dtype=np.float64)
        out[prefix + k] = float(np.nanmean(arr)) if arr.size else float("nan")
    return out


def summarize_rewards(rewards: Iterable[float], prefix: str = "R") -> Dict[str, float]:
    arr = np.asarray(list(rewards), dtype=np.float64)
    if arr.size == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan"), f"{prefix}_p10": float("nan"), f"{prefix}_p50": float("nan"), f"{prefix}_p90": float("nan")}
    return {
        f"{prefix}_mean": float(np.nanmean(arr)),
        f"{prefix}_std": float(np.nanstd(arr)),
        f"{prefix}_p10": float(np.nanpercentile(arr, 10)),
        f"{prefix}_p50": float(np.nanpercentile(arr, 50)),
        f"{prefix}_p90": float(np.nanpercentile(arr, 90)),
    }


# -----------------------------------------------------------------------------
# Paired comparison statistics: formerly evaluate_compare.py responsibilities
# -----------------------------------------------------------------------------


def bootstrap_ci(
    x: Sequence[float],
    *,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    stat: str = "mean",
) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or B <= 0:
        v = float(np.nanmean(arr) if stat == "mean" else np.nanmedian(arr))
        return v, v
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        vals[b] = np.nanmean(sample) if stat == "mean" else np.nanmedian(sample)
    lo = float(np.nanpercentile(vals, 100 * alpha / 2.0))
    hi = float(np.nanpercentile(vals, 100 * (1.0 - alpha / 2.0)))
    return lo, hi


def sign_test_pvalue(delta: Sequence[float]) -> float:
    """Two-sided exact sign-test p-value for non-zero paired deltas."""
    arr = np.asarray(delta, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    pos = int(np.sum(arr > 0))
    neg = int(np.sum(arr < 0))
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    # exact two-sided p = 2 * P[Bin(n, 0.5) <= k], capped at 1.
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    return float(min(1.0, 2.0 * prob))


def paired_pvalue(delta: Sequence[float]) -> Dict[str, float]:
    """Return Wilcoxon if SciPy is available, plus sign-test fallback always."""
    arr = np.asarray(delta, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out = {"sign_p": sign_test_pvalue(arr), "wilcoxon_p": float("nan")}
    if arr.size == 0:
        return out
    try:
        from scipy.stats import wilcoxon  # type: ignore

        nonzero = arr[np.abs(arr) > 1e-12]
        if nonzero.size > 0:
            out["wilcoxon_p"] = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    except Exception:
        pass
    return out


def paired_delta_summary(
    baseline: Sequence[float],
    rl: Sequence[float],
    *,
    B: int = 2000,
    seed: int = 0,
    prefix: str = "return",
) -> Dict[str, float]:
    b = np.asarray(baseline, dtype=np.float64)
    r = np.asarray(rl, dtype=np.float64)
    n = int(min(b.size, r.size))
    b = b[:n]
    r = r[:n]
    d = r - b
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {
            f"{prefix}_delta_mean": float("nan"),
            f"{prefix}_delta_median": float("nan"),
            f"{prefix}_win_rate": float("nan"),
            f"{prefix}_delta_ci_low": float("nan"),
            f"{prefix}_delta_ci_high": float("nan"),
            f"{prefix}_sign_p": float("nan"),
            f"{prefix}_wilcoxon_p": float("nan"),
        }
    lo, hi = bootstrap_ci(d, B=B, seed=seed, stat="mean")
    pv = paired_pvalue(d)
    return {
        f"{prefix}_delta_mean": float(np.nanmean(d)),
        f"{prefix}_delta_std": float(np.nanstd(d)),
        f"{prefix}_delta_median": float(np.nanmedian(d)),
        f"{prefix}_win_rate": float(np.mean(d > 0)),
        f"{prefix}_delta_ci_low": float(lo),
        f"{prefix}_delta_ci_high": float(hi),
        f"{prefix}_sign_p": float(pv["sign_p"]),
        f"{prefix}_wilcoxon_p": float(pv["wilcoxon_p"]),
    }


def compare_metric_rows(
    baseline_rows: Sequence[Mapping[str, float]],
    rl_rows: Sequence[Mapping[str, float]],
    *,
    B: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    keys = sorted(set().union(*[r.keys() for r in baseline_rows], *[r.keys() for r in rl_rows])) if baseline_rows or rl_rows else []
    out: Dict[str, float] = {}
    for k in keys:
        b = [float(r.get(k, np.nan)) for r in baseline_rows]
        r = [float(r.get(k, np.nan)) for r in rl_rows]
        s = paired_delta_summary(b, r, B=B, seed=seed, prefix=k)
        out.update(s)
    return out


# -----------------------------------------------------------------------------
# Seed-sweep aggregation: formerly seed_sweep.py responsibilities
# -----------------------------------------------------------------------------


def _nested_get(d: Mapping[str, Any], path: str, default: float = float("nan")) -> float:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return float(default)
    try:
        return float(cur)
    except Exception:
        return float(default)


def find_summary_files(root: str, pattern: str = "**/*_summary.json") -> List[str]:
    return sorted(glob.glob(os.path.join(root, pattern), recursive=True))


def load_seed_summary_table(
    root: str,
    pattern: str = "**/*_summary.json",
    metric_paths: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if metric_paths is None:
        metric_paths = [
            "eval_metrics.R_mean",
            "eval_metrics.R_std",
            "eval_control_metrics.IAE_rho",
            "eval_control_metrics.MAE_rho",
            "eval_control_metrics.overshoot_rho",
            "eval_control_metrics.TV_uc",
            "eval_control_metrics.IAE_h",
            "eval_control_metrics.TV_uw",
            "best_score",
            "best_update",
        ]

    rows: List[Dict[str, Any]] = []
    for path in find_summary_files(root, pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            continue
        row: Dict[str, Any] = {
            "summary_path": path,
            "mode": js.get("mode", ""),
            "seed": js.get("seed", ""),
            "best_path": js.get("best_path", ""),
        }
        for mp in metric_paths:
            row[mp] = _nested_get(js, mp)
        rows.append(row)
    return rows


def summarize_seed_table(rows: Sequence[Mapping[str, Any]], metric: str = "eval_metrics.R_mean") -> Dict[str, Dict[str, float]]:
    by_mode: Dict[str, List[float]] = {}
    for row in rows:
        mode = str(row.get("mode", "unknown"))
        try:
            v = float(row.get(metric, float("nan")))
        except Exception:
            v = float("nan")
        by_mode.setdefault(mode, []).append(v)
    out: Dict[str, Dict[str, float]] = {}
    for mode, vals in by_mode.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        out[mode] = {
            "n": float(arr.size),
            "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
            "std": float(np.nanstd(arr)) if arr.size else float("nan"),
            "min": float(np.nanmin(arr)) if arr.size else float("nan"),
            "max": float(np.nanmax(arr)) if arr.size else float("nan"),
        }
    return out


def write_rows_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return path
    fieldnames = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))
    return path


def save_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path
