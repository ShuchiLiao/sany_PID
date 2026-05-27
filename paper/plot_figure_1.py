import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "font.sans-serif": ["SimHei"],   # 中文黑体
    "axes.unicode_minus": False,
    "font.size": 24,                 # 全局字号
    "axes.labelsize": 24,
    "legend.fontsize": 24,
})


# =========================
# 常量：IAE归一化尺度
# =========================
RHO_SCALE = 1000.0
H_SCALE = 1.0


def _load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _as_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)


def _find_first_key(d: dict, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def load_seed_metrics(seed_dir: str, s, mode: str):
    """
    读取单个seed的指标文件，并尽量兼容不同key命名。
    返回: (data_dict, file_path)

    data_dict 可能包含:
      - train_updates, train_returns_ema
      - eval_updates, eval_IAE_rho
      - eval_updates, eval_IAE_h
    """
    path = os.path.join(seed_dir, "plots", f"{mode}_seed{s}_metrics_series.npz")
    if not os.path.exists(path):
        return None, path

    d = _load_npz(path)

    # 训练曲线
    k_upd = _find_first_key(d, ["train_updates", "updates", "train_update", "update"])
    k_ema = _find_first_key(d, ["train_returns_ema", "returns_ema", "train_R_ema", "R_ema"])

    # 评估曲线
    k_eupd = _find_first_key(d, ["eval_updates", "eval_update", "eval_steps"])
    k_eiae_rho = _find_first_key(d, ["eval_IAE_rho", "eval_IAE_density", "eval_IAE"])
    k_eiae_h = _find_first_key(d, ["eval_IAE_h", "eval_IAE_level"])

    out = {}

    if k_ema is not None:
        ema = _as_1d(d[k_ema]).astype(float)
        out["train_returns_ema"] = ema

        if k_upd is not None:
            out["train_updates"] = _as_1d(d[k_upd]).astype(int)
        else:
            # npz中没存update：用长度生成（update总数固定时最合适）
            out["train_updates"] = np.arange(1, len(ema) + 1, dtype=int)

    if k_eupd is not None:
        out["eval_updates"] = _as_1d(d[k_eupd]).astype(int)

    if k_eupd is not None and k_eiae_rho is not None:
        out["eval_IAE_rho"] = _as_1d(d[k_eiae_rho]).astype(float)

    if k_eupd is not None and k_eiae_h is not None:
        out["eval_IAE_h"] = _as_1d(d[k_eiae_h]).astype(float)

    return out, path


def align_by_updates(curves):
    """
    curves: list of (updates:int[...], values:[...])
    使用“共同update集合交集”进行对齐，避免插值引入假设。
    返回 common_updates, stacked_values (n_seeds, n_points)
    """
    if len(curves) == 0:
        return None, None

    common = set(curves[0][0].tolist())
    for u, _ in curves[1:]:
        common &= set(u.tolist())

    common_updates = np.array(sorted(common), dtype=int)
    if common_updates.size == 0:
        return None, None

    stacked = []
    for u, v in curves:
        pos = {int(uu): i for i, uu in enumerate(u.tolist())}
        vv = np.array([v[pos[int(uu)]] for uu in common_updates], dtype=float)
        stacked.append(vv)

    return common_updates, np.vstack(stacked)


def plot_mean_band(
    x,
    Y,
    xlabel,
    ylabel,
    out_path,
    *,
    band: str = "quantile",          # "quantile" or "std"
    q_lo: float = 0.25,
    q_hi: float = 0.75,
    line_color="#2F6FED",
    band_color="#A8C5FF",
):
    """
    x: (n_points,)
    Y: (n_seeds, n_points)

    band:
      - "quantile": [q_lo, q_hi] 分位带（默认25-75）
      - "std": 均值±1σ带
    """
    x = np.asarray(x)
    Y = np.asarray(Y, dtype=float)

    mean = np.nanmean(Y, axis=0)

    if band.lower() in ("quantile", "q", "iqr"):
        lo = np.nanquantile(Y, q_lo, axis=0)
        hi = np.nanquantile(Y, q_hi, axis=0)
        band_label = f"{int(q_lo*100)}–{int(q_hi*100)}分位带"
    elif band.lower() in ("std", "sigma", "stdev"):
        std = np.nanstd(Y, axis=0, ddof=0)
        lo = mean - std
        hi = mean + std
        band_label = "±1σ带"
    else:
        raise ValueError(f"Unknown band type: {band}. Use 'quantile' or 'std'.")

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(x, mean, linewidth=2.6, color=line_color, label="均值")
    plt.fill_between(x, lo, hi, color=band_color, alpha=0.55, label=band_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _style_for_mode(mode: str):
    """
    为不同mode返回不同配色（训练EMA曲线要求不同颜色）
    你可以按论文风格自行改这里。
    """
    if mode == "premix":
        # 蓝系
        return {
            "train_line": "#2F6FED",
            "train_band": "#A8C5FF",
            "rho_line":   "#F08A24",
            "rho_band":   "#FFD2A8",
            "h_line":     "#24A148",
            "h_band":     "#BFEBC9",
        }
    else:
        # production：紫系（训练EMA颜色与premix明显区分）
        return {
            "train_line": "#7A4CE0",
            "train_band": "#D6C7FF",
            "rho_line":   "#F08A24",
            "rho_band":   "#FFD2A8",
            "h_line":     "#24A148",
            "h_band":     "#BFEBC9",
        }


def make_figures(
    root="./checkpoints",
    seeds=range(1, 9),
    mode="premix",
    outdir="./paper_figs",
    *,
    band: str = "quantile",   # "quantile" or "std"
):
    """
    生成：
      - Figure 1：train_returns_ema（均值+带）
      - Figure 2：eval_IAE_*（均值+带）；production时额外画IAE_h（单独一张）

    band:
      - "quantile": 25–75分位带（默认）
      - "std": 均值±1σ带
    """
    os.makedirs(outdir, exist_ok=True)
    colors = _style_for_mode(mode)

    train_curves = []
    eval_rho_curves = []
    eval_h_curves = []
    missing = []

    for s in seeds:
        seed_dir = os.path.join(root, f"seed{s}")
        data, path = load_seed_metrics(seed_dir, s, mode)

        if data is None:
            missing.append(path)
            continue

        # Figure 1：训练回报EMA（要求 premix/production 不同颜色：在 _style_for_mode 中控制）
        if "train_updates" in data and "train_returns_ema" in data:
            train_curves.append((data["train_updates"], data["train_returns_ema"]))

        # Figure 2：评估 IAE（密度）——按要求归一化：/1000
        if "eval_updates" in data and "eval_IAE_rho" in data:
            y = data["eval_IAE_rho"] / float(RHO_SCALE)
            eval_rho_curves.append((data["eval_updates"], y))

        # Figure 2：评估 IAE（液位）——按要求归一化：/1
        if "eval_updates" in data and "eval_IAE_h" in data:
            y = data["eval_IAE_h"] / float(H_SCALE)
            eval_h_curves.append((data["eval_updates"], y))

    if missing:
        print("缺少以下文件（已跳过对应seed）：")
        for p in missing:
            print(" -", p)

    # Figure 1：训练回报EMA
    if len(train_curves) >= 2:
        x, Y = align_by_updates(train_curves)
        if x is None:
            print("Figure 1：无法对齐 train_updates（各seed共同update为空）。")
        else:
            out_path = os.path.join(outdir, f"Figure1_{'预混' if mode=='premix' else '生产'}_训练回报EMA.png")
            plot_mean_band(
                x, Y,
                xlabel="更新步",
                ylabel="训练回报（EMA）",
                out_path=out_path,
                band=band,
                line_color=colors["train_line"],
                band_color=colors["train_band"],
            )
            print("已保存：", out_path)
    else:
        print("Figure 1：可用seed不足（至少需要2个）。")

    # Figure 2：评估IAE（密度）——已归一化：IAE_rho / 1000
    if len(eval_rho_curves) >= 2:
        x, Y = align_by_updates(eval_rho_curves)
        if x is None:
            print("Figure 2（密度）：无法对齐 eval_updates（各seed共同update为空）。")
        else:
            out_path = os.path.join(outdir, f"Figure2_{'预混' if mode=='premix' else '生产'}_评估IAE_密度.png")
            plot_mean_band(
                x, Y,
                xlabel="更新步",
                ylabel="IAE",
                out_path=out_path,
                band=band,
                line_color=colors["rho_line"],
                band_color=colors["rho_band"],
            )
            print("已保存：", out_path)
    else:
        print("Figure 2（密度）：可用seed不足或未记录 eval_IAE_rho。")

    # Figure 2：评估IAE（液位，仅production需要且存在）——已归一化：IAE_h / 1
    if mode == "production":
        if len(eval_h_curves) >= 2:
            x, Y = align_by_updates(eval_h_curves)
            if x is None:
                print("Figure 2（液位）：无法对齐 eval_updates（各seed共同update为空）。")
            else:
                out_path = os.path.join(outdir, "Figure2_生产_评估IAE_液位.png")
                plot_mean_band(
                    x, Y,
                    xlabel="更新步",
                    ylabel="IAE",
                    out_path=out_path,
                    band=band,
                    line_color=colors["h_line"],
                    band_color=colors["h_band"],
                )
                print("已保存：", out_path)
        else:
            print("Figure 2（液位）：可用seed不足或未记录 eval_IAE_h。")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./checkpoints")
    ap.add_argument("--outdir", type=str, default="./paper_figs")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--band", type=str, default="quantile", choices=["quantile", "std"],
                    help="band type: 'quantile' for 25–75 percentile band, 'std' for mean±1σ band")
    ap.add_argument("--mode", type=str, default="both", choices=["premix", "production", "both"],
                    help="which mode to plot")
    return ap.parse_args()


if __name__ == "__main__":
    # CLI用法示例：
    #

    #   python plot_figure_1.py --mode production --band std --seeds 1 2 3 4
    args = _parse_args()

    seeds = args.seeds
    if args.mode in ("premix", "both"):
        make_figures(root=args.root, seeds=seeds, mode="premix", outdir=args.outdir, band=args.band)
    if args.mode in ("production", "both"):
        make_figures(root=args.root, seeds=seeds, mode="production", outdir=args.outdir, band=args.band)
