import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Part 1: run many seeds
# -----------------------------
def run_one(seed: int, train_py: str, extra_args: List[str]) -> int:
    """
    Run: python train_modes.py --seed <seed> <extra_args...>

    Minimal enhancement:
    - If user didn't provide --ckpt_dir, inject a per-seed ckpt_dir:
        --ckpt_dir ./checkpoints/seed{seed}
      so that <mode>_best.pt won't be overwritten across seeds.
    """
    # Respect user override if they already passed --ckpt_dir
    if "--ckpt_dir" in extra_args:
        ckpt_inject: List[str] = []
    else:
        ckpt_inject = ["--ckpt_dir", f"./checkpoints/seed{seed}"]

    cmd = [sys.executable, train_py, "--seed", str(seed)] + ckpt_inject + extra_args
    print("\n=== RUN:", " ".join(cmd))

    # 2) 强制 matplotlib 使用非交互后端，避免 Tkinter/Tcl 崩溃
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")  # 关键：禁用 TkAgg/tkinter

    p = subprocess.run(cmd, env=env)
    return p.returncode
    #如果想手动指定 ckpt_dir（仍然支持）只要passthrough 里自己带了 --ckpt_dir ...，上面代码会自动尊重你的参数，不会再注入。

def run_many_seeds(
    train_py: str,
    seeds: List[int],
    extra_args: List[str],
    continue_on_fail: bool = False,
) -> None:
    train_path = Path(train_py)
    if not train_path.exists():
        raise FileNotFoundError(f"train_py not found: {train_path}")

    failed: List[Tuple[int, int]] = []
    for s in seeds:
        rc = run_one(int(s), train_py, extra_args)
        if rc != 0:
            failed.append((s, rc))
            if not continue_on_fail:
                raise RuntimeError(f"Stopped due to failure: seed={s}, rc={rc}")

    if failed:
        print("\nSome runs failed:")
        for s, rc in failed:
            print(f"  seed={s}, rc={rc}")
        raise RuntimeError("Some runs failed (see above).")

    print("\nAll seeds finished.")


# -----------------------------
# Part 2: summarize & boxplot
# -----------------------------
def find_summaries(root: Path) -> List[Path]:
    return sorted(root.rglob("*_summary.json"))


def load_summaries(paths: List[Path]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            obj["_path"] = str(p)
            rows.append(obj)
        except Exception as e:
            print(f"Skip bad json: {p} ({e})")
    return rows


def extract_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compatible with your earlier two variants:
      - summary["eval_metrics"] = {...}
      - summary["eval_return"]  = {...}
    """
    m = row.get("eval_metrics", None)
    if m is None:
        m = row.get("eval_return", None)
    if m is None:
        m = row.get("eval", None)
    if not isinstance(m, dict):
        return {}
    return m


def group_by_mode(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        mode = r.get("mode", "unknown")
        g.setdefault(mode, []).append(r)
    return g


def _collect_metric_values(rows: List[Dict[str, Any]], metric_key: str) -> List[float]:
    """
    Collect metric values from each summary row. Tries a few common aliases.
    """
    aliases = {
        "return_mean": ["return_mean", "R_mean", "Rmean", "mean_return"],
        "IAE_rho": ["IAE_rho"],
        "IAE_h": ["IAE_h"],
        "overshoot_rho": ["overshoot_rho"],
        "overshoot_h": ["overshoot_h"],
        "TV_uc": ["TV_uc"],
        "TV_uw": ["TV_uw"],
        "Ts_rho": ["Ts_rho"],
        "sat_uc": ["sat_uc"],
        "sat_uw": ["sat_uw"],
    }

    keys_to_try = aliases.get(metric_key, [metric_key])

    vals: List[float] = []
    for r in rows:
        m = extract_metrics(r)
        v = None
        for k in keys_to_try:
            if k in m:
                v = m[k]
                break
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            pass
    return vals


def boxplot_by_mode(
    out_png: Path,
    grouped: Dict[str, List[Dict[str, Any]]],
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    labels = []
    data = []

    for mode, rows in grouped.items():
        vals = _collect_metric_values(rows, metric_key)
        if len(vals) == 0:
            continue
        labels.append(f"{mode}\n(n={len(vals)})")
        data.append(vals)

    if not data:
        print(f"[WARN] No data for metric '{metric_key}', skip: {out_png}")
        return

    fig = plt.figure(figsize=(8, 4))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def summarize_boxplots(root: Path, outdir: Path) -> None:
    paths = find_summaries(root)
    if not paths:
        raise FileNotFoundError(f"No *_summary.json under: {root}")

    rows = load_summaries(paths)
    grouped = group_by_mode(rows)

    metrics_to_plot: List[Tuple[str, str, str]] = [
        ("return_mean", "Return (mean) across seeds", "Return mean"),
        ("IAE_rho", "IAE_rho across seeds", "IAE_rho"),
        ("IAE_h", "IAE_h across seeds", "IAE_h"),
        ("overshoot_rho", "Overshoot_rho across seeds", "Overshoot_rho"),
        ("overshoot_h", "Overshoot_h across seeds", "Overshoot_h"),
        ("TV_uc", "TV_uc across seeds", "TV_uc"),
        ("TV_uw", "TV_uw across seeds", "TV_uw"),
        ("Ts_rho", "Settling time Ts_rho across seeds", "Ts_rho (s)"),
        ("sat_uc", "Saturation ratio (cement) across seeds", "sat_uc"),
        ("sat_uw", "Saturation ratio (water) across seeds", "sat_uw"),
    ]

    outdir.mkdir(parents=True, exist_ok=True)
    for key, title, ylabel in metrics_to_plot:
        boxplot_by_mode(outdir / f"box_{key}.png", grouped, key, title, ylabel)

    # Optional: print a small table summary
    print("\n=== Summary (mean±std) ===")
    for key, _, _ in metrics_to_plot:
        for mode, rows_m in grouped.items():
            vals = _collect_metric_values(rows_m, key)
            if not vals:
                continue
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            print(f"{mode:10s}  {key:12s}  {mu:.6g} ± {sd:.6g}   (n={len(vals)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_py", type=str, default="train_modes.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--root", type=str, default="./checkpoints", help="Where *_summary.json are located (recursive)")
    ap.add_argument("--outdir", type=str, default="./paper_figs", help="Where to write boxplots")

    ap.add_argument("--run", action="store_true", help="Run training for all seeds")
    ap.add_argument("--summarize", action="store_true", help="Summarize and draw boxplots")
    ap.add_argument("--continue_on_fail", action="store_true")

    # passthrough extra args for train_modes.py
    ap.add_argument("--passthrough", nargs=argparse.REMAINDER, default=[],
                    help="Extra args forwarded to train_modes.py (prefix with --passthrough -- <args...>)")

    args = ap.parse_args()

    extra_args = args.passthrough
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    # If neither specified: do both
    do_run = args.run or (not args.run and not args.summarize)
    do_sum = args.summarize or (not args.run and not args.summarize)

    if do_run:
        run_many_seeds(args.train_py, args.seeds, extra_args, continue_on_fail=args.continue_on_fail)

    if do_sum:
        summarize_boxplots(Path(args.root), Path(args.outdir))


if __name__ == "__main__":
    main()

#
# # 跑 8 个 seed，再自动汇总画 boxplot（默认两步都做）
# python seed_sweep.py --train_py train_modes.py --seeds 1 2 3 4 5 6 7 8 --root ./checkpoints --outdir ./paper_figs
#
# # 只跑训练
# python seed_sweep.py --run --train_py train_modes.py --seeds 1 2 3 4 5
#
# # 只汇总画图（你已经跑完了）
# python seed_sweep.py --summarize --root ./checkpoints --outdir ./paper_figs
#
# # 把额外参数透传给 train_modes.py（比如只训练 production）
# python seed_sweep.py --seeds 1 2 3 4 5 --passthrough -- --mode production



# seed_sweep
#
# 输入（命令行参数）
#
# seed_sweep.py 支持两类动作：跑多 seed 训练、汇总画图（也可以两步都做；默认就是两步都做）。
#
# seed_sweep
#
# 关键参数
#
# --train_py：要运行的训练脚本路径，默认 train_modes.py
#
# --seeds：要跑的随机种子列表（如 1 2 3 4 5）
#
# --run：只执行训练（不汇总）
#
# --summarize：只执行汇总画图（不训练）
#
# --continue_on_fail：某个 seed 失败时是否继续（否则直接停止并报错）
#
# --passthrough -- <args...>：把后面的参数原样透传给 train_modes.py（比如 --mode production --updates 500 ...）
#
# seed_sweep
#
# 汇总相关参数
#
# --root：递归搜索 *_summary.json 的根目录（默认 ./checkpoints）
#
# --outdir：boxplot 图片输出目录（默认 ./paper_figs）
#
# seed_sweep
#
# 一个非常重要的“隐式输入”
#
# 如果没有在 passthrough 里指定 --ckpt_dir，脚本会自动注入：
#
# --ckpt_dir ./checkpoints/seed{seed}
# 避免不同 seed 互相覆盖训练输出。
#
# seed_sweep
#
# 输出（文件 + 控制台）
# 1) 训练阶段输出（由 train_modes.py 产生）
#
# seed_sweep.py 本身不定义训练输出格式，它只是多次调用 train_modes.py。
# 但它会确保每个 seed 的 ckpt_dir 不同（除非你手动覆盖）。
#
# seed_sweep
#
# 典型目录会像：
#
# ./checkpoints/seed1/...
#
# ./checkpoints/seed2/...
#
# ...
#
# 里面通常应包含 *_best.pt、训练曲线、以及后续汇总所需的 *_summary.json（注意：汇总阶段就是靠找这个文件）。
#
# seed_sweep
#
# 2) 汇总阶段输出（由 seed_sweep.py 产生）
#
# 它会在 --outdir 下写一组 boxplot 图片：
#
# seed_sweep
#
# box_return_mean.png
#
# box_IAE_rho.png
#
# box_IAE_h.png
#
# box_overshoot_rho.png
#
# box_overshoot_h.png
#
# box_TV_uc.png
#
# box_TV_uw.png
#
# box_Ts_rho.png
#
# box_sat_uc.png
#
# box_sat_uw.png
#
# 并在控制台打印每个 mode 的 mean ± std（按上述这些指标逐项打印）。
#
# seed_sweep