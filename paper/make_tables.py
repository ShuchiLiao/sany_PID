
"""
make_tables_1_3.py

Generate Table 1-3 for the paper, by loading existing saved artifacts:
- train_modes.py outputs (metrics_series.npz) for each training seed folder
- evaluate_compare.py outputs (premix/production_compare_results_seed*_N*.npz) under each seed/**/compare/

Tables:
Table 1: Simulation environment & parameter ranges (train/test distributions, noise model, constraints)
Table 2: Overall comparison vs baseline (mean±std across seeds, win-rate, paired deltas)
Table 3: Ablation study (A1–A4) [template; can auto-load if ablation npz exist]

Usage:
  python make_tables_1_3.py --mode premix --ckpt_root ./checkpoints --compare_root ./checkpoints --outdir ./paper_tables
  python make_tables_1_3.py --mode production --ckpt_root ./checkpoints --compare_root ./checkpoints --outdir ./paper_tables

Notes:
- This script is designed to run inside your repo (so it can import your local modes.py/train_modes.py if needed).
- For Table 1, we parse ranges/noise defaults from *the source code* (modes.py/train_modes.py) so it stays synced.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# File discovery helpers
# -----------------------------

def _find_files_recursive(root: str, pattern: str) -> List[str]:
    import fnmatch
    out: List[str] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fnmatch.fnmatch(fn, pattern):
                out.append(os.path.join(dp, fn))
    return sorted(out)

def discover_metrics_series_npz(ckpt_root: str, mode: str) -> Dict[int, str]:
    """
    Find all: **/seed{seed}/**/{mode}_metrics_series.npz  (and a couple of tolerated names)
    Return {seed -> path}.
    """
    # tolerate a few naming conventions
    pats = [
        f"{mode}_metrics_series.npz",
        f"{mode}_metrics.npz",
        f"metrics_series_{mode}.npz",
        f"{mode}_train_metrics.npz",
    ]
    paths: List[str] = []
    for pat in pats:
        paths += _find_files_recursive(ckpt_root, pat)

    by_seed: Dict[int, str] = {}
    for p in paths:
        # expect .../seed{seed}/...
        m = re.search(r"[\\/](?:seed)(\d+)[\\/]", p)
        if not m:
            continue
        s = int(m.group(1))
        # keep the latest (lexicographically) if duplicates
        if (s not in by_seed) or (p > by_seed[s]):
            by_seed[s] = p
    return dict(sorted(by_seed.items(), key=lambda kv: kv[0]))

def discover_compare_npz(compare_root: str, mode: str, N: Optional[int] = None) -> Dict[int, str]:
    """
    Find all: **/{mode}_compare_results_seed{seed}_N{N}.npz
    Return {seed_training -> path}. NOTE: this is "args.seed" in evaluate_compare; many people use 777.
    We group by the *training seed folder* if it exists; otherwise by the compare-seed itself.

    Your layout:
      ./checkpoints/seed1/compare/{mode}_compare_results_seed777_N5000.npz
      ./checkpoints/seed2/compare/{mode}_compare_results_seed777_N5000.npz
      ...
    """
    pat = f"{mode}_compare_results_seed*_N*.npz"
    paths = _find_files_recursive(compare_root, pat)

    out: Dict[int, List[Tuple[int, str]]] = {}  # train_seed -> [(N, path)]
    for p in paths:
        bn = os.path.basename(p)
        m = re.search(rf"{re.escape(mode)}_compare_results_seed(\d+)_N(\d+)\.npz$", bn)
        if not m:
            continue
        compare_seed = int(m.group(1))
        n = int(m.group(2))
        if (N is not None) and (n != int(N)):
            continue

        # training seed folder if present
        m2 = re.search(r"[\\/](?:seed)(\d+)[\\/]", p)
        train_seed = int(m2.group(1)) if m2 else compare_seed
        out.setdefault(train_seed, []).append((n, p))

    picked: Dict[int, str] = {}
    for s, items in out.items():
        items = sorted(items, key=lambda t: t[0])
        picked[s] = items[-1][1]  # largest N
    return dict(sorted(picked.items(), key=lambda kv: kv[0]))

def _load_npz(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


# -----------------------------
# Table 1 parsing from source
# -----------------------------

@dataclass
class RangeSpec:
    name: str
    train_dist: str
    test_dist: str
    unit: str = ""
    notes: str = ""

def _extract_uniform_ranges_from_modes_py(modes_py_path: str, mode: str) -> List[RangeSpec]:
    """
    Very lightweight parser:
    - locate the sampler() of make_premix_mode / make_production_mode
    - extract lines like: x = float(rng.uniform(a, b))
    """
    txt = open(modes_py_path, "r", encoding="utf-8").read().splitlines()

    if mode == "premix":
        # between "def make_premix_mode" and "def make_production_mode"
        start = next(i for i,l in enumerate(txt) if l.strip().startswith("def make_premix_mode"))
        end = next(i for i,l in enumerate(txt) if l.strip().startswith("def make_production_mode"))
    else:
        start = next(i for i,l in enumerate(txt) if l.strip().startswith("def make_production_mode"))
        end = len(txt)

    block = txt[start:end]
    specs: List[RangeSpec] = []

    # map variable -> (a,b)
    for l in block:
        m = re.search(r"(\w+)\s*=\s*float\(rng\.uniform\(([^,]+),\s*([^)]+)\)\)", l.replace(" ", ""))
        if not m:
            continue
        var = m.group(1)
        a = m.group(2)
        b = m.group(3)
        specs.append(RangeSpec(name=var, train_dist=f"Uniform[{a},{b}]", test_dist="Same as train", unit="", notes="from modes.py sampler"))

    # add constant ones we care about that may not be uniform
    if mode == "premix":
        specs.append(RangeSpec("qs", "Fixed 0.0", "Fixed 0.0", unit="m3/s", notes="premix"))
    return specs

def _extract_noise_model_from_train_modes_py(train_modes_path: str) -> pd.DataFrame:
    """
    Parse key uncertainty/noise settings in simulate_episode():
    - tau_mix_hat ~ Normal(p.tau_mix, 0.25 p.tau_mix), clipped [5,100]
    - tau_delay ~ Normal(p.tau_delay, 0.25 p.tau_delay), clipped [0,20]
    - max_flow jitter: water 10%, cement 10%, clipped ranges
    - flow noise: flow_noise_tau, flow_noise_std, per-episode seed
    """
    txt = open(train_modes_path, "r", encoding="utf-8").read().splitlines()
    rows = []

    def grab(pattern: str) -> Optional[str]:
        for l in txt:
            if pattern in l.replace(" ", ""):
                return l.strip()
        return None

    # Hard-code the summary (robust to minor formatting changes) by detecting the key lines
    line_tau_mix = next((l.strip() for l in txt if "tau_mix" in l and "rng.normal" in l and "clip" in l), "")
    line_tau_delay = next((l.strip() for l in txt if "tau_delay" in l and "rng.normal" in l and "clip" in l), "")
    line_wv = next((l.strip() for l in txt if "wv_max_flow" in l and "rng.normal" in l and "clip" in l), "")
    line_cv = next((l.strip() for l in txt if "cv_max_flow" in l and "rng.normal" in l and "clip" in l), "")

    rows.append({"Item": "tau_mix uncertainty", "Model": line_tau_mix or "Not found", "Notes": "Normal jitter (~25%), clipped"})
    rows.append({"Item": "tau_delay uncertainty", "Model": line_tau_delay or "Not found", "Notes": "Normal jitter (~25%), clipped"})
    rows.append({"Item": "water valve max_flow jitter", "Model": line_wv or "Not found", "Notes": "Normal jitter (~10%), clipped"})
    rows.append({"Item": "cement valve max_flow jitter", "Model": line_cv or "Not found", "Notes": "Normal jitter (~10%), clipped"})

    # flow noise lines
    flow_tau = next((l.strip() for l in txt if "flow_noise_tau" in l), "")
    flow_std = next((l.strip() for l in txt if "flow_noise_std" in l), "")
    rows.append({"Item": "flow noise (AR-like)", "Model": f"{flow_tau} ; {flow_std}".strip(" ;"), "Notes": "per-episode independent seeds"})

    return pd.DataFrame(rows)

def build_table1(mode: str, repo_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - df_ranges: parameter ranges
      - df_noise: noise/uncertainty model summary
    """
    modes_path = os.path.join(repo_root, "modes.py")
    train_modes_path = os.path.join(repo_root, "train_modes.py")

    if not os.path.isfile(modes_path):
        # fallback: maybe running from scripts/
        modes_path = os.path.join(os.path.dirname(__file__), "modes.py")
    if not os.path.isfile(train_modes_path):
        train_modes_path = os.path.join(os.path.dirname(__file__), "train_modes.py")

    specs = _extract_uniform_ranges_from_modes_py(modes_path, mode)
    df_ranges = pd.DataFrame([s.__dict__ for s in specs])

    df_noise = _extract_noise_model_from_train_modes_py(train_modes_path)
    return df_ranges, df_noise


# -----------------------------
# Table 2: overall comparison
# -----------------------------

def _seed_summary_from_compare_npz(npz: Dict[str, np.ndarray]) -> Dict[str, float]:
    d = {}
    for k in ["R_base","R_rl","dR","iae_h_base","iae_h_rl","iae_rho_base","iae_rho_rl"]:
        if k in npz:
            a = np.asarray(npz[k], dtype=np.float64)
            d[k+"_mean"] = float(np.mean(a))
            d[k+"_std"] = float(np.std(a))
    if "dR" in npz:
        d["win_rate"] = float(np.mean(np.asarray(npz["dR"]) > 0.0))
    return d

def build_table2(mode: str, compare_paths: Dict[int, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-seed table + aggregated table (mean±std across seeds).
    """
    rows = []
    for seed, path in compare_paths.items():
        z = _load_npz(path)
        s = _seed_summary_from_compare_npz(z)
        s["train_seed"] = seed
        s["npz_path"] = path
        rows.append(s)
    df_seed = pd.DataFrame(rows).sort_values("train_seed")

    # aggregate across training seeds: take mean over seeds of per-seed means
    # (this avoids overweighting a seed if N differs; here N is typically fixed 5000)
    metrics = [c for c in df_seed.columns if c.endswith("_mean") or c in ["win_rate"]]
    agg = []
    for m in metrics:
        x = pd.to_numeric(df_seed[m], errors="coerce").dropna().values
        if x.size == 0:
            continue
        agg.append({
            "metric": m,
            "across_seeds_mean": float(np.mean(x)),
            "across_seeds_std": float(np.std(x)),
        })
    df_agg = pd.DataFrame(agg).sort_values("metric")
    return df_seed, df_agg


# -----------------------------
# Table 3: ablation (template)
# -----------------------------

_ABLATION_DEFAULT = [
    ("A0 (Full)", "Full method (baseline + RL tuning + feedforward + uncertainty)"),
    ("A1", "Remove feedforward learning (fix s_ff_w=s_ff_c=1)"),
    ("A2", "Remove max-flow uncertainty (use wv/cv max_flow without jitter)"),
    ("A3", "Remove tau uncertainty (use tau_mix=tau_delay without jitter)"),
    ("A4", "RL only, no base parameters (start from naive PID gains)"),
]

def build_table3(mode: str, ablation_root: str, N: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If ablation_root contains compare npz for each ablation id, load them.
    Expected layout (suggested):
      ./checkpoints/ablation/A1/seed1/compare/production_compare_results_seed777_N5000.npz
    or:
      ./checkpoints/ablation/A1/**/compare/{mode}_compare_results_seed*_N*.npz

    Returns:
      - df_ablation: rows for each ablation id
      - df_missing: which ones were not found
    """
    rows = []
    missing = []

    for ab_id, desc in _ABLATION_DEFAULT:
        if ab_id.startswith("A0"):
            sub = ""  # allow root itself
            search_root = ablation_root
        else:
            search_root = os.path.join(ablation_root, ab_id)

        found = discover_compare_npz(search_root, mode, N=N)
        if not found:
            missing.append({"ablation": ab_id, "desc": desc, "search_root": search_root})
            continue

        # aggregate across seeds in that ablation folder
        df_seed, df_agg = build_table2(mode, found)
        # pick a compact set of headline metrics
        headline = {}
        for key in ["dR_mean", "iae_rho_rl_mean", "iae_rho_base_mean", "win_rate"]:
            # find across-seeds value
            subdf = df_agg[df_agg["metric"].str.contains(key)]
            if subdf.empty:
                continue
            headline[key+"_across_seeds_mean"] = float(subdf["across_seeds_mean"].iloc[0])
            headline[key+"_across_seeds_std"] = float(subdf["across_seeds_std"].iloc[0])

        headline.update({"ablation": ab_id, "desc": desc, "n_seeds": int(df_seed.shape[0])})
        rows.append(headline)

    return pd.DataFrame(rows), pd.DataFrame(missing)


# -----------------------------
# Save helpers
# -----------------------------

def _save_all_formats(df: pd.DataFrame, outdir: str, name: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    # 1) CSV 永远可用
    df.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)

    # 2) XLSX（可选：需要 openpyxl）
    try:
        df.to_excel(os.path.join(outdir, f"{name}.xlsx"), index=False)
    except Exception as e:
        print(f"[warn] skip xlsx for {name}: {e}")

    # 3) LaTeX（可选）
    try:
        latex = df.to_latex(index=False, escape=True)
        with open(os.path.join(outdir, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(latex)
    except Exception as e:
        print(f"[warn] skip tex for {name}: {e}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["premix","production"], required=True)
    ap.add_argument("--ckpt_root", type=str, default="./checkpoints")
    ap.add_argument("--compare_root", type=str, default="./checkpoints")
    ap.add_argument("--outdir", type=str, default="./paper_tables")
    ap.add_argument("--N", type=int, default=None, help="N for compare npz (e.g., 5000). If None, pick largest N per seed.")
    ap.add_argument("--repo_root", type=str, default=".", help="Path containing modes.py/train_modes.py (usually repo root).")
    ap.add_argument("--ablation_root", type=str, default="./checkpoints/ablation")
    args = ap.parse_args()

    mode = args.mode

    # ---- Table 1 ----
    df_ranges, df_noise = build_table1(mode, repo_root=args.repo_root)
    _save_all_formats(df_ranges, args.outdir, f"{mode}_Table1_param_ranges")
    _save_all_formats(df_noise, args.outdir, f"{mode}_Table1_noise_model")

    # ---- Table 2 ----
    compare_paths = discover_compare_npz(args.compare_root, mode, N=args.N)
    if not compare_paths:
        raise FileNotFoundError(
            f"No compare npz found for mode={mode}. "
            f"Expected under {args.compare_root}/**/compare/{mode}_compare_results_seed*_N*.npz"
        )
    df_seed, df_agg = build_table2(mode, compare_paths)
    _save_all_formats(df_seed, args.outdir, f"{mode}_Table2_compare_per_seed")
    _save_all_formats(df_agg, args.outdir, f"{mode}_Table2_compare_agg")

    # ---- Table 3 ----
    df_ab, df_miss = build_table3(mode, args.ablation_root, N=args.N)
    _save_all_formats(df_ab, args.outdir, f"{mode}_Table3_ablation")
    _save_all_formats(df_miss, args.outdir, f"{mode}_Table3_ablation_missing")

    print(f"[done] wrote tables to: {os.path.abspath(args.outdir)}")
    print(f"  - {mode}_Table1_param_ranges.(csv/xlsx/tex)")
    print(f"  - {mode}_Table1_noise_model.(csv/xlsx/tex)")
    print(f"  - {mode}_Table2_compare_per_seed.(csv/xlsx/tex)")
    print(f"  - {mode}_Table2_compare_agg.(csv/xlsx/tex)")
    print(f"  - {mode}_Table3_ablation.(csv/xlsx/tex)")
    print(f"  - {mode}_Table3_ablation_missing.(csv/xlsx/tex)")

if __name__ == "__main__":
    main()
#python make_tables.py --mode premix --ckpt_root ./checkpoints --compare_root ./checkpoints --outdir ./paper_tables --repo_root .
#python make_tables.py --mode production --ckpt_root ./checkpoints --compare_root ./checkpoints --outdir ./paper_tables --repo_root .
