#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reviewer-requested lightweight experiments.

This script runs two small supplementary experiments for the manuscript revision:
1) Single-policy-inference timing test.
2) Out-of-training-range generalization test.

It does not retrain any model. It loads trained checkpoints and reuses the
existing paired baseline-vs-RL simulation pipeline.

Run from the repository root, for example:
    python -m scripts.rl.reviewer_lightweight_experiments --mode both --premix_ckpt outputs/premix_train/seed3/premix_best.pt --production_ckpt outputs/production_train/seed3/production_best.pt --N 1000 --device cuda --dt 0.5 --premix_duration 120 --premix_hold 10 --production_duration 240 --timing_repeats 10000 --out_dir outputs/reviewer_lightweight
    python -m scripts.rl.reviewer_lightweight_experiments --mode production --ckpt outputs/production_train/seed3/production_best.pt --N 1000 --device cpu --dt 0.5 --production_duration 240 --out_dir outputs/reviewer_lightweight/production_seed3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parents[1]
for _p in (_SCRIPTS / "core", _SCRIPTS / "PID_control", _SCRIPTS / "rl", _SCRIPTS):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate import agent_scales, baseline_scales, load_agent_from_checkpoint, make_mode, reward_from_traj, set_global_seed  # noqa: E402
from metrics import compute_basic_metrics  # noqa: E402
from rollout import simulate_episode_with_uncertainty  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight reviewer experiments without retraining.")
    parser.add_argument("--mode", choices=["premix", "production", "both"], default="both")
    parser.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path, only valid when --mode is premix or production.")
    parser.add_argument("--premix_ckpt", type=str, default="outputs/premix_train/seed1/premix_best.pt")
    parser.add_argument("--production_ckpt", type=str, default="outputs/production_train/seed1/production_best.pt")
    parser.add_argument("--out_dir", type=str, default="outputs/reviewer_lightweight")
    parser.add_argument("--N", type=int, default=1000, help="Episodes per out-of-range test case.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--premix_duration", type=float, default=120.0)
    parser.add_argument("--premix_hold", type=float, default=10.0)
    parser.add_argument("--production_duration", type=float, default=240.0)
    parser.add_argument("--timing_repeats", type=int, default=10000)
    parser.add_argument("--timing_warmup", type=int, default=500)
    parser.add_argument("--include_combined_extreme", action="store_true", help="Also test combined extreme tau, delay, and high-q cases.")
    parser.add_argument("--no_timing", action="store_true")
    parser.add_argument("--no_ood", action="store_true")
    return parser.parse_args(argv)


def write_json(path: Path, data: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=float)
    return str(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def finite_array(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def stats(values: Iterable[Any], prefix: str) -> Dict[str, float]:
    arr = finite_array(values)
    if arr.size == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_median": float("nan"), f"{prefix}_p25": float("nan"), f"{prefix}_p75": float("nan")}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
    }


def mode_names(mode: str) -> List[str]:
    return ["premix", "production"] if mode == "both" else [mode]


def checkpoint_for_mode(args: argparse.Namespace, mode_name: str) -> str:
    if args.ckpt is not None:
        if args.mode == "both":
            raise ValueError("--ckpt can only be used when --mode is premix or production. Use --premix_ckpt and --production_ckpt for --mode both.")
        return args.ckpt
    return args.premix_ckpt if mode_name == "premix" else args.production_ckpt


def make_mode_from_args(mode_name: str, args: argparse.Namespace):
    return make_mode(mode_name, premix_duration=args.premix_duration, premix_hold=args.premix_hold, production_duration=args.production_duration)


def sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_single_inference(mode, agent, *, repeats: int, warmup: int, seed: int, device: str) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    contexts = []
    for _ in range(max(1, int(repeats))):
        p = mode.sample_episode(rng)
        contexts.append(mode.build_context(p))

    agent.eval()
    with torch.no_grad():
        for i in range(int(warmup)):
            _ = agent_scales(agent, mode, contexts[i % len(contexts)])
        sync_if_cuda(device)

        elapsed_ms: List[float] = []
        for i in range(int(repeats)):
            t0 = time.perf_counter_ns()
            _ = agent_scales(agent, mode, contexts[i])
            sync_if_cuda(device)
            elapsed_ms.append((time.perf_counter_ns() - t0) / 1.0e6)

    arr = np.asarray(elapsed_ms, dtype=np.float64)
    return {
        "mode": mode.name,
        "device": str(device),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "note": "Timing includes deterministic policy action extraction and PID scale mapping, but excludes closed-loop simulation.",
    }


def make_fixed_uncertainty(p: Any, rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "tau_mix_hat": float(p.tau_mix),
        "rho_obs_delay": float(p.tau_delay),
        "water_valve_max_flow": float(p.water_valve_max_flow),
        "cement_valve_max_flow": float(p.cement_valve_max_flow),
        "water_flow_seed": int(rng.integers(0, 2**31 - 1)),
        "cement_flow_seed": int(rng.integers(0, 2**31 - 1)),
    }


def ood_case_names(mode_name: str, include_combined: bool) -> List[str]:
    cases = ["tau_mix_100_150s", "delay_20_30s"]
    if mode_name == "production":
        cases.append("qout_1p5_2p0_m3min")
    if include_combined:
        cases.append("combined_extreme")
    return cases


def sample_ood_episode(mode, case_name: str, rng: np.random.Generator, dt: float):
    p = mode.sample_episode(rng)
    p.dt = float(dt)

    if case_name == "tau_mix_100_150s":
        p.tau_mix = float(rng.uniform(100.0, 150.0))
    elif case_name == "delay_20_30s":
        p.tau_delay = float(rng.uniform(20.0, 30.0))
    elif case_name == "qout_1p5_2p0_m3min":
        if mode.name != "production":
            raise ValueError("qout_1p5_2p0_m3min is only valid for production mode.")
        p.qs = float(rng.uniform(1.5 / 60.0, 2.0 / 60.0))
    elif case_name == "combined_extreme":
        p.tau_mix = float(rng.uniform(100.0, 150.0))
        p.tau_delay = float(rng.uniform(20.0, 30.0))
        if mode.name == "production":
            p.qs = float(rng.uniform(1.5 / 60.0, 2.0 / 60.0))
    else:
        raise ValueError(f"Unknown OOD case: {case_name}")

    return p


def paired_ood_one_case(mode, agent, *, case_name: str, N: int, seed: int, dt: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    b_scales = baseline_scales(mode)
    rows: List[Dict[str, Any]] = []

    for i in range(int(N)):
        p = sample_ood_episode(mode, case_name, rng, dt)
        uncertainty = make_fixed_uncertainty(p, rng)
        base_params = mode.compute_base_params(p)
        obs = mode.build_context(p)
        r_scales = agent_scales(agent, mode, obs)

        traj_b = simulate_episode_with_uncertainty(p, b_scales, uncertainty, base_params=base_params, return_traj=True, full=False, catch_errors=True)
        traj_r = simulate_episode_with_uncertainty(p, r_scales, uncertainty, base_params=base_params, return_traj=True, full=False, catch_errors=True)

        mb = compute_basic_metrics(traj_b, p)
        mr = compute_basic_metrics(traj_r, p)
        Rb = reward_from_traj(mode, traj_b, p)
        Rr = reward_from_traj(mode, traj_r, p)

        row: Dict[str, Any] = {
            "mode": mode.name,
            "case": case_name,
            "i": int(i),
            "tau_mix_s": float(p.tau_mix),
            "delay_s": float(p.tau_delay),
            "qout_m3min": float(getattr(p, "qs", 0.0)) * 60.0,
            "rho_sp": float(getattr(p, "rho_sp", np.nan)),
            "h_sp": float(getattr(p, "h_sp", np.nan)),
            "baseline_return": float(Rb),
            "rl_return": float(Rr),
            "delta_return": float(Rr - Rb),
        }

        for k, v in mb.items():
            row[f"baseline_{k}"] = float(v)
        for k, v in mr.items():
            row[f"rl_{k}"] = float(v)
        for k in sorted(set(mb.keys()) & set(mr.keys())):
            row[f"delta_{k}"] = float(mb[k]) - float(mr[k]) if k.startswith("IAE") else float(mr[k]) - float(mb[k])

        row["improve_IAE_rho"] = float(mb.get("IAE_rho", np.nan)) - float(mr.get("IAE_rho", np.nan))
        row["rel_improve_IAE_rho_pct"] = 100.0 * row["improve_IAE_rho"] / max(abs(float(mb.get("IAE_rho", np.nan))), 1e-12)
        if mode.name == "production":
            row["improve_IAE_h"] = float(mb.get("IAE_h", np.nan)) - float(mr.get("IAE_h", np.nan))
            row["rel_improve_IAE_h_pct"] = 100.0 * row["improve_IAE_h"] / max(abs(float(mb.get("IAE_h", np.nan))), 1e-12)

        for name, value in r_scales.items():
            row[f"scale_{name}"] = float(value)

        rows.append(row)

    summary: Dict[str, Any] = {"mode": mode.name, "case": case_name, "N": int(N), "seed": int(seed), "dt": float(dt)}
    summary.update(stats((r["baseline_return"] for r in rows), "baseline_return"))
    summary.update(stats((r["rl_return"] for r in rows), "rl_return"))
    summary.update(stats((r["delta_return"] for r in rows), "delta_return"))
    summary.update(stats((r["baseline_IAE_rho"] for r in rows), "baseline_IAE_rho"))
    summary.update(stats((r["rl_IAE_rho"] for r in rows), "rl_IAE_rho"))
    summary.update(stats((r["improve_IAE_rho"] for r in rows), "improve_IAE_rho"))
    summary.update(stats((r["rel_improve_IAE_rho_pct"] for r in rows), "rel_improve_IAE_rho_pct"))
    summary["rho_IAE_win_rate"] = float(np.mean(finite_array(r["improve_IAE_rho"] for r in rows) > 0.0))
    summary["baseline_failed_rate"] = float(np.mean(finite_array(r.get("baseline_failed", np.nan) for r in rows) > 0.5))
    summary["rl_failed_rate"] = float(np.mean(finite_array(r.get("rl_failed", np.nan) for r in rows) > 0.5))

    if mode.name == "production":
        summary.update(stats((r["baseline_IAE_h"] for r in rows), "baseline_IAE_h"))
        summary.update(stats((r["rl_IAE_h"] for r in rows), "rl_IAE_h"))
        summary.update(stats((r["improve_IAE_h"] for r in rows), "improve_IAE_h"))
        summary.update(stats((r["rel_improve_IAE_h_pct"] for r in rows), "rel_improve_IAE_h_pct"))
        summary["h_IAE_win_rate"] = float(np.mean(finite_array(r["improve_IAE_h"] for r in rows) > 0.0))

    return rows, summary


def run_for_mode(mode_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    mode = make_mode_from_args(mode_name, args)
    ckpt = checkpoint_for_mode(args, mode_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found for {mode_name}: {ckpt}")

    out_dir = Path(args.out_dir) / mode_name
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = load_agent_from_checkpoint(mode, ckpt, device=args.device)
    agent.eval()

    result: Dict[str, Any] = {"mode": mode_name, "ckpt": ckpt, "out_dir": str(out_dir)}

    if not args.no_timing:
        timing = benchmark_single_inference(mode, agent, repeats=args.timing_repeats, warmup=args.timing_warmup, seed=args.seed + 100, device=args.device)
        result["timing"] = timing
        write_csv(out_dir / f"{mode_name}_single_inference_timing.csv", [timing])
        write_json(out_dir / f"{mode_name}_single_inference_timing.json", timing)
        print(f"[{mode_name}] inference timing: mean={timing['mean_ms']:.6f} ms, p99={timing['p99_ms']:.6f} ms")

    if not args.no_ood:
        all_rows: List[Dict[str, Any]] = []
        summaries: List[Dict[str, Any]] = []
        for j, case_name in enumerate(ood_case_names(mode_name, args.include_combined_extreme)):
            rows, summary = paired_ood_one_case(mode, agent, case_name=case_name, N=args.N, seed=args.seed + 1000 + j, dt=args.dt)
            all_rows.extend(rows)
            summaries.append(summary)
            if mode_name == "production":
                print(
                    f"[{mode_name}:{case_name}] "
                    f"rho IAE median improvement={summary['improve_IAE_rho_median']:.6g}, "
                    f"rho win_rate={summary['rho_IAE_win_rate']:.3f}; "
                    f"h IAE median improvement={summary['improve_IAE_h_median']:.6g}, "
                    f"h win_rate={summary['h_IAE_win_rate']:.3f}"
                )
            else:
                print(
                    f"[{mode_name}:{case_name}] "
                    f"rho IAE median improvement={summary['improve_IAE_rho_median']:.6g}, "
                    f"rho win_rate={summary['rho_IAE_win_rate']:.3f}"
                )

        result["ood_summaries"] = summaries
        write_csv(out_dir / f"{mode_name}_ood_cases.csv", all_rows)
        write_csv(out_dir / f"{mode_name}_ood_summary.csv", summaries)
        write_json(out_dir / f"{mode_name}_ood_summary.json", {"mode": mode_name, "summaries": summaries})

    write_json(out_dir / f"{mode_name}_reviewer_lightweight_result.json", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    set_global_seed(args.seed)

    results: Dict[str, Any] = {}
    for mode_name in mode_names(args.mode):
        results[mode_name] = run_for_mode(mode_name, args)

    out_dir = Path(args.out_dir)
    write_json(out_dir / "reviewer_lightweight_all_results.json", results)
    return results


if __name__ == "__main__":
    main()