# plot_figure_2.py（仿真与作图分离版）
# 生成论文所需：典型工况轨迹对比图 + 全体情景统计图/表
#
# 用法示例：
# 1) 先运行仿真并缓存（按 seed 存储）
#   预混：
#     python plot_figure_2.py --mode premix --ckpt ./checkpoints/seed1/premix_best.pt --N 5000 --seed 123 --run
#   生产：
#     python plot_figure_2.py --mode production --ckpt ./checkpoints/seed1/production_best.pt --N 5000 --seed 123 --run
#
# 2) 再只作图（直接读取缓存，不重复仿真）
#   预混：
#     python plot_figure_2.py --mode premix --ckpt ./checkpoints/seed1/premix_best.pt --N 5000 --seed 123 --plot
#   生产：
#     python plot_figure_2.py --mode production --ckpt ./checkpoints/seed1/production_best.pt --N 5000 --seed 123 --plot

from __future__ import annotations

import os
import math
import argparse
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

from modes import (
    EpisodeParams,
    ModeSpec,
    Trajectory,
    make_premix_mode,
    make_production_mode,
    action_to_scales,
)
from PPO_bandit import PPOBanditAgent, PPOConfig
from sim_config import PlantParams, ValveParams, SimulationConfig, PhysicalConstraintError
from sim_env import CementingSimEnv
from sim_model import SlurryState


# ==========
# 全局画图风格
# ==========
plt.rcParams.update({
    "font.sans-serif": ["SimHei"],   # 中文黑体
    "axes.unicode_minus": False,
    "font.size": 24,
    "axes.labelsize": 24,
    "legend.fontsize": 22,
})

# 统一标签
LABEL_BASE = r"$\lambda$整定"
LABEL_RL = "RL整定"

# 液位百分比：相对最大液位高度 H_MAX
H_MAX = 2.0

# 典型工况：统一目标
RHO_SP_FIXED = 1650.0         # kg/m3
H_SP_FIXED_PCT = 50.0         # %
H_SP_FIXED = (H_SP_FIXED_PCT / 100.0) * H_MAX

# 统一数值显示
DECIMALS = 2


def _fmt_num(x: float) -> str:
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}"
    return f"{x:.{DECIMALS}f}"


def _axis_plain_formatter(decimals: int = DECIMALS) -> FuncFormatter:
    def _f(x, pos):
        x = float(x)
        if abs(x) >= 1000:
            return f"{int(round(x))}"
        if abs(x - round(x)) < 1e-9:
            return f"{int(round(x))}"
        return f"{x:.{decimals}f}"
    return FuncFormatter(_f)


def _apply_plain_ticks(ax, apply_x: bool = True, apply_y: bool = True):
    if apply_x:
        ax.xaxis.set_major_formatter(_axis_plain_formatter())
    if apply_y:
        ax.yaxis.set_major_formatter(_axis_plain_formatter())


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _ckpt_stem(path: str) -> str:
    b = os.path.basename(path)
    for suf in [".pt", ".pth", ".ckpt"]:
        if b.endswith(suf):
            b = b[:-len(suf)]
            break
    return b


def _result_path(mode: str, ckpt: str, N: int, seed: int) -> str:
    stem = _ckpt_stem(ckpt)
    d = _ensure_dir(f"./paper_cache_{mode}")
    return os.path.join(d, f"seed{seed}_{stem}_N{N}.npz")


@torch.no_grad()
def act_deterministic(agent: PPOBanditAgent, obs: np.ndarray) -> np.ndarray:
    if hasattr(agent, "act_deterministic"):
        a, _, _ = agent.act_deterministic(obs)
        return np.asarray(a, dtype=np.float32)
    a, _, _ = agent.act(obs)
    return np.asarray(a, dtype=np.float32)


def _make_sim_config_filtered(**kwargs):
    # 兼容不同版本 SimulationConfig 字段
    try:
        fields = {f.name for f in getattr(SimulationConfig, "__dataclass_fields__", {}).values()}
        if fields:
            kwargs = {k: v for k, v in kwargs.items() if k in fields}
    except Exception:
        pass
    return SimulationConfig(**kwargs)


def load_agent_from_ckpt(ckpt_path: str, device: str = "cpu") -> tuple[str, PPOBanditAgent, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mode_name = str(ckpt.get("mode", ""))
    cfg_dict = dict(ckpt.get("config", {}))
    if "obs_dim" not in cfg_dict:
        cfg_dict["obs_dim"] = int(ckpt.get("obs_dim", 0))
    if "act_dim" not in cfg_dict:
        cfg_dict["act_dim"] = int(ckpt.get("act_dim", 0))

    cfg = PPOConfig(**cfg_dict)
    agent = PPOBanditAgent(cfg, device=device)
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return mode_name, agent, ckpt


# ==========
# 仿真（成对公平对比）
# ==========

def simulate_episode_with_uncertainty(
    p: EpisodeParams,
    scales: dict[str, float],
    *,
    base_params: dict[str, float],
    tau_mix_hat: float,
    tau_delay: float,
    wv_max_flow: float,
    cv_max_flow: float,
    wv_noise_seed: int,
    cv_noise_seed: int,
) -> Trajectory:
    # PID 参数（位置式PID：基线×缩放）
    Kp_w = base_params.get("Kp_w", 0.0) * scales.get("s_w_p", 1.0)
    Ki_w = base_params.get("Ki_w", 0.0) * scales.get("s_w_i", 1.0)
    Kd_w = base_params.get("Kd_w", 0.0) * scales.get("s_w_d", 1.0)

    Kp_c = base_params.get("Kp_c", 0.0) * scales.get("s_c_p", 1.0)
    Ki_c = base_params.get("Ki_c", 0.0) * scales.get("s_c_i", 1.0)
    Kd_c = base_params.get("Kd_c", 0.0) * scales.get("s_c_d", 1.0)

    ff_w = base_params.get("ff_w", 0.0)
    ff_c = base_params.get("ff_c", 0.0)
    kff  = base_params.get("kff", 0.0)

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

    qs = float(p.qs) if getattr(p, "mode", "") == "production" else 0.0

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
        control_mode=("siso-density" if getattr(p, "mode", "") == "premix" else "mimo"),

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

    if ff_w > 0.0 and hasattr(cfg, "water_opening_ff"):
        cfg.water_opening_ff = float(ff_w)
    if ff_c > 0.0 and hasattr(cfg, "cement_opening_ff"):
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

    n_steps = int(math.floor(float(p.duration_s) / float(p.dt))) + 1
    t_hist  = np.zeros(n_steps, dtype=np.float32)
    h_hist  = np.zeros(n_steps, dtype=np.float32)
    rho_hist= np.zeros(n_steps, dtype=np.float32)
    uw_hist = np.zeros(n_steps, dtype=np.float32)
    uc_hist = np.zeros(n_steps, dtype=np.float32)

    t_hist[0]   = 0.0
    h_hist[0]   = float(getattr(env.state, "h", p.h0))
    rho_hist[0] = float(getattr(env.state, "rho_out", p.rho0))
    uw_hist[0]  = float(getattr(env.water_valve, "current_opening", 0.0))
    uc_hist[0]  = float(getattr(env.cement_valve, "current_opening", 0.0))

    for k in range(1, n_steps):
        t = k * float(p.dt)
        if getattr(p, "mode", "") == "production":
            env.Q_s = float(p.qs)

        try:
            env.step(None, float(t))
        except PhysicalConstraintError:
            # 约束违规：后续保持不变
            t_hist[k:]   = float(t)
            h_hist[k:]   = h_hist[k - 1]
            rho_hist[k:] = rho_hist[k - 1]
            uw_hist[k:]  = uw_hist[k - 1]
            uc_hist[k:]  = uc_hist[k - 1]
            break

        t_hist[k]   = float(t)
        h_hist[k]   = float(getattr(env.state, "h", h_hist[k - 1]))
        rho_hist[k] = float(getattr(env.state, "rho_out", rho_hist[k - 1]))
        uw_hist[k]  = float(getattr(env.water_valve, "current_opening", uw_hist[k - 1]))
        uc_hist[k]  = float(getattr(env.cement_valve, "current_opening", uc_hist[k - 1]))

    return Trajectory(t=t_hist, rho=rho_hist, h=h_hist, u_w=uw_hist, u_c=uc_hist)


def compute_iae(y: np.ndarray, sp: float, dt: float) -> float:
    return float(np.sum(np.abs(np.asarray(y, dtype=np.float64) - float(sp))) * float(dt))


# ==========
# 画图：典型轨迹对比（legend 显示 PID 参数）
# ==========

def plot_typical_compare_premix(out_png: str, tb: Trajectory, tr: Trajectory, rho_sp: float, *, label_base: str, label_rl: str):
    fig = plt.figure(figsize=(8, 6), dpi=600, constrained_layout=True)
    ax = plt.gca()

    ax.plot(tb.t, tb.rho, lw=2.2, color="#2F6FED", label=label_base)
    ax.plot(tr.t, tr.rho, lw=2.2, color="#F08A24", label=label_rl)
    ax.axhline(float(rho_sp), ls="--", lw=1.2, color="black", label="目标值")

    ymin = float(np.nanmin(np.r_[tb.rho, tr.rho]))
    ymax = float(np.nanmax(np.r_[tb.rho, tr.rho]))
    yspan = max(1e-12, ymax - ymin)
    pad = 0.10 * yspan  # premix 上下各留10%
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("时间 / s")
    ax.set_ylabel("出口密度 / (kg·m$^{-3}$)")
    ax.legend(fontsize=20, frameon=True, loc="best")

    _apply_plain_ticks(ax)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_typical_compare_production_h(out_png: str, tb: Trajectory, tr: Trajectory, h_sp: float, *, label_base: str, label_rl: str):
    fig = plt.figure(figsize=(8, 6), dpi=600, constrained_layout=True)
    ax = plt.gca()

    # 液位百分比：相对 H_MAX
    h0 = 100.0 * np.asarray(tb.h, dtype=np.float64) / float(H_MAX)
    h1 = 100.0 * np.asarray(tr.h, dtype=np.float64) / float(H_MAX)
    hsp_pct = 100.0 * float(h_sp) / float(H_MAX)

    ax.axhline(hsp_pct, ls="--", lw=1.2, color="black", label="目标值")
    ax.plot(tb.t, h0, lw=2.2, color="#24A148", label=label_base)  # 绿色（液位-基线）
    ax.plot(tr.t, h1, lw=2.2, color="#D1495B", label=label_rl)  # 暖色（液位-RL）

    ymin = float(np.nanmin(np.r_[h0, h1]))
    ymax = float(np.nanmax(np.r_[h0, h1]))
    yspan = max(1e-12, ymax - ymin)
    pad = 0.90 * yspan  # production 上下各留50%
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("时间 / s")
    ax.set_ylabel("液位 / %")
    ax.legend(fontsize=20, frameon=True, loc="best")

    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    _apply_plain_ticks(ax, apply_x=True, apply_y=True)

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_typical_compare_production_rho(out_png: str, tb: Trajectory, tr: Trajectory, rho_sp: float, *, label_base: str, label_rl: str):
    fig = plt.figure(figsize=(8, 6), dpi=600, constrained_layout=True)
    ax = plt.gca()

    ax.plot(tb.t, tb.rho, lw=2.2, color="#2F6FED", label=label_base)
    ax.plot(tr.t, tr.rho, lw=2.2, color="#F08A24", label=label_rl)
    ax.axhline(float(rho_sp), ls="--", lw=1.2, color="black", label="目标值")

    ymin = float(np.nanmin(np.r_[tb.rho, tr.rho]))
    ymax = float(np.nanmax(np.r_[tb.rho, tr.rho]))
    yspan = max(1e-12, ymax - ymin)
    pad = 0.90 * yspan  # production 上下各留50%
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("时间 / s")
    ax.set_ylabel("出口密度 / (kg·m$^{-3}$)")
    ax.legend(fontsize=20, frameon=True, loc="best")

    ax.yaxis.set_major_locator(MultipleLocator(20.0))
    _apply_plain_ticks(ax)

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ==========
# 分布对比（小提琴 + 箱线叠加）
# ==========

def _set_ylim_centered(ax, data: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, pad_ratio: float = 0.08):
    data = np.asarray(data, dtype=np.float64)
    lo = float(np.quantile(data, p_low / 100.0))
    hi = float(np.quantile(data, p_high / 100.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(data))
        hi = float(np.max(data))
    span = max(1e-12, hi - lo)
    pad = span * pad_ratio
    ax.set_ylim(lo - pad, hi + pad)


def plot_violin_box(
    out_png: str,
    data_a: np.ndarray,
    data_b: np.ndarray,
    label_a: str,
    label_b: str,
    ylabel: str,
    *,
    violin_colors: tuple[str, str] = ("#A8C5FF", "#FFD2A8"),
    box_colors: tuple[str, str] = ("#2F6FED", "#F08A24"),
):
    data_a = np.asarray(data_a, dtype=np.float64)
    data_b = np.asarray(data_b, dtype=np.float64)
    all_data = np.concatenate([data_a, data_b], axis=0)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()

    parts = ax.violinplot([data_a, data_b], positions=[1, 2], showmeans=False, showmedians=False, showextrema=False)
    colors = list(violin_colors)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("none")
        pc.set_alpha(0.85)

    bp = ax.boxplot([data_a, data_b], positions=[1, 2], widths=0.25, showfliers=False, patch_artist=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(list(box_colors)[i])
        box.set_alpha(0.25)
        box.set_edgecolor("none")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel(ylabel)

    _apply_plain_ticks(ax, apply_x=False, apply_y=True)
    _set_ylim_centered(ax, all_data, p_low=1.0, p_high=99.0, pad_ratio=0.10)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_violin_box_3(
    out_png: str,
    data3: np.ndarray,
    labels3: list[str],
    ylabel: str,
    *,
    hline: float | None = 1.0,
    violin_colors: tuple[str, str, str] = ("#A8C5FF", "#FFD2A8", "#E6D7FF"),
    box_colors: tuple[str, str, str] = ("#2F6FED", "#F08A24", "#7B61FF"),
):
    data3 = np.asarray(data3, dtype=np.float64)
    assert data3.ndim == 2 and data3.shape[1] == 3

    cols = [data3[:, 0], data3[:, 1], data3[:, 2]]
    all_data = np.concatenate(cols, axis=0)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()

    parts = ax.violinplot(cols, positions=[1, 2, 3], showmeans=False, showmedians=False, showextrema=False)
    colors = list(violin_colors)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("none")   # 保持原风格：不描边
        pc.set_alpha(0.85)

    bp = ax.boxplot(cols, positions=[1, 2, 3], widths=0.25, showfliers=False, patch_artist=True)
    bcols = list(box_colors)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(bcols[i])
        box.set_alpha(0.25)
        box.set_edgecolor("none")  # 保持原风格：不描边

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels3)
    ax.set_ylabel(ylabel)

    if hline is not None:
        ax.axhline(float(hline), ls="--", lw=1.2, color="black", alpha=0.8)

    _apply_plain_ticks(ax, apply_x=False, apply_y=True)
    _set_ylim_centered(ax, all_data, p_low=1.0, p_high=99.0, pad_ratio=0.10)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ==========
# 分组箱线图（严格按 nbins 等宽分箱）
# ==========

def plot_grouped_delta_box(
    out_png: str,
    x: np.ndarray,
    delta: np.ndarray,
    xlabel: str,
    *,
    nbins: int = 6,
    x_min: float | None = None,
    x_max: float | None = None,
):
    x = np.asarray(x, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)

    xmin_data = float(np.nanmin(x))
    xmax_data = float(np.nanmax(x))
    xmin = float(x_min) if x_min is not None else xmin_data
    xmax = float(x_max) if x_max is not None else xmax_data
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmin, xmax = xmin_data, xmax_data + 1e-9

    edges = np.linspace(xmin, xmax, nbins + 1, dtype=np.float64)

    groups, labels = [], []
    for i in range(nbins):
        l, r = edges[i], edges[i + 1]
        mask = (x >= l) & (x < r) if i < nbins - 1 else (x >= l) & (x <= r)
        g = delta[mask]
        if g.size == 0:
            g = np.array([np.nan], dtype=np.float64)
        groups.append(g)
        labels.append(f"[{_fmt_num(l)},{_fmt_num(r)}]")

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()

    pos = np.arange(1, nbins + 1, dtype=np.float64)
    ax.boxplot(groups, positions=pos, widths=0.6, showfliers=False)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=18)

    ax.axhline(0.0, ls="--", lw=1.2, color="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ΔIAE")

    _apply_plain_ticks(ax, apply_x=False, apply_y=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def plot_grouped_delta_box_dual(
    out_png: str,
    x: np.ndarray,
    delta_rho: np.ndarray,
    delta_h: np.ndarray,
    xlabel: str,
    *,
    nbins: int = 6,
    x_min: float | None = None,
    x_max: float | None = None,
    label_rho: str = "密度",
    label_h: str = "液位",
    # 颜色：密度沿用蓝/橙体系；液位用绿/暖色体系
    color_rho: str = "#F08A24",
    color_h: str = "#24A148",
):
    """
    一个分箱线图里同时画两套箱体：密度 ΔIAE 与 液位 ΔIAE
    - 每个 bin 画两组箱体：左=密度，右=液位
    - 不改箱体填充，仅通过箱体边框/中位数线/须线颜色区分两指标
    """
    x = np.asarray(x, dtype=np.float64)
    delta_rho = np.asarray(delta_rho, dtype=np.float64)
    delta_h = np.asarray(delta_h, dtype=np.float64)

    xmin_data = float(np.nanmin(x))
    xmax_data = float(np.nanmax(x))
    xmin = float(x_min) if x_min is not None else xmin_data
    xmax = float(x_max) if x_max is not None else xmax_data
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmin, xmax = xmin_data, xmax_data + 1e-9

    edges = np.linspace(xmin, xmax, nbins + 1, dtype=np.float64)

    # 每个 bin 两组：rho / h
    groups_rho, groups_h, labels = [], [], []
    for i in range(nbins):
        l, r = edges[i], edges[i + 1]
        mask = (x >= l) & (x < r) if i < nbins - 1 else (x >= l) & (x <= r)

        g_rho = delta_rho[mask]
        g_h = delta_h[mask]
        if g_rho.size == 0:
            g_rho = np.array([np.nan], dtype=np.float64)
        if g_h.size == 0:
            g_h = np.array([np.nan], dtype=np.float64)

        groups_rho.append(g_rho)
        groups_h.append(g_h)
        labels.append(f"[{_fmt_num(l)},{_fmt_num(r)}]")

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()

    # 位置：每个 bin 一个中心，左右偏移画两组箱体
    centers = np.arange(1, nbins + 1, dtype=np.float64)
    offset = 0.18
    pos_rho = centers - offset
    pos_h = centers + offset

    # 密度箱体（蓝系）
    bp_rho = ax.boxplot(
        groups_rho,
        positions=pos_rho,
        widths=0.28,
        showfliers=False,
        boxprops=dict(color=color_rho, linewidth=1.6),
        whiskerprops=dict(color=color_rho, linewidth=1.2),
        capprops=dict(color=color_rho, linewidth=1.2),
        medianprops=dict(color=color_rho, linewidth=2.2),
    )

    # 液位箱体（绿系）
    bp_h = ax.boxplot(
        groups_h,
        positions=pos_h,
        widths=0.28,
        showfliers=False,
        boxprops=dict(color=color_h, linewidth=1.6),
        whiskerprops=dict(color=color_h, linewidth=1.2),
        capprops=dict(color=color_h, linewidth=1.2),
        medianprops=dict(color=color_h, linewidth=2.2),
    )

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=18)

    ax.axhline(0.0, ls="--", lw=1.2, color="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ΔIAE")

    # 图例：用 dummy 线条做 legend
    ax.plot([], [], color=color_rho, lw=3.0, label=label_rho)
    ax.plot([], [], color=color_h, lw=3.0, label=label_h)
    ax.legend(frameon=False, loc="best")

    _apply_plain_ticks(ax, apply_x=False, apply_y=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# ==========
# Spearman 相关性热力图
# ==========

def _rankdata(a: np.ndarray) -> np.ndarray:
    """average-rank for ties (Spearman)"""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)

    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 5:
        return np.nan
    rx = _rankdata(x[m])
    ry = _rankdata(y[m])
    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)
    denom = (np.sqrt(np.sum(rx**2)) * np.sqrt(np.sum(ry**2)))
    if denom < 1e-12:
        return np.nan
    return float(np.sum(rx * ry) / denom)


def plot_corr_heatmap(out_png: str, ctx_cn: dict[str, np.ndarray], y3: np.ndarray, cmap):
    """
    热力图：工况变量 vs 三参数缩放因子的 Spearman 相关系数
    - 不对 scale 做 log（heatmap 显示的是相关系数 ρ）
    - x轴显示三列 Kp / Ki / Kd，并在左侧加“缩放因子：”
    - 无 title
    """
    y3 = np.asarray(y3, dtype=np.float64)
    assert y3.ndim == 2 and y3.shape[1] == 3

    row_names = list(ctx_cn.keys())
    R = np.full((len(row_names), 3), np.nan, dtype=np.float64)
    for i, k in enumerate(row_names):
        x = np.asarray(ctx_cn[k], dtype=np.float64)
        for j in range(3):
            R[i, j] = _spearman_corr(x, y3[:, j])

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()

    # 高级发散配色（避免默认绿蓝）
    im = ax.imshow(R, aspect="equal", origin="lower", vmin=-1.0, vmax=1.0, cmap=cmap)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Spearman ρ")

    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Kp", "Ki", "Kd"])

    # --- 让格子边界更清晰：给每个 cell 画网格线 ---
    n_rows = len(row_names)
    n_cols = 3

    # 在每个格子的边界处放 minor ticks：-0.5, 0.5, 1.5, ...
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)

    # 画 minor grid 作为“格子边框”
    ax.grid(which="minor", linestyle="-", linewidth=0.8, color="white", alpha=0.9)

    # 不显示 minor tick 的刻度线和标签
    ax.tick_params(which="minor", bottom=False, left=False)

    # “缩放因子：”放在 x 轴左前方
    ax.text(-0.45, -0.01, "缩放因子：", transform=ax.transAxes,
            ha="left", va="top", fontsize=22)

    _apply_plain_ticks(ax, apply_x=False, apply_y=False)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ==========
# 统计表（CSV）
# ==========
def summarize_and_save_csv(out_csv: str, base: np.ndarray, rl: np.ndarray, name: str):
    base = np.asarray(base, dtype=np.float64)
    rl   = np.asarray(rl, dtype=np.float64)
    d    = base - rl  # ΔIAE = λ整定 - RL整定

    def row(x):
        return {
            "均值": float(np.mean(x)),
            "中位数": float(np.median(x)),
            "25分位": float(np.quantile(x, 0.25)),
            "75分位": float(np.quantile(x, 0.75)),
        }

    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["指标", "对象", "均值", "中位数", "25分位", "75分位"])
        for label, arr in [(LABEL_BASE, base), (LABEL_RL, rl), (f"差值({LABEL_BASE}-{LABEL_RL})", d)]:
            r = row(arr)
            w.writerow([name, label, _fmt_num(r["均值"]), _fmt_num(r["中位数"]), _fmt_num(r["25分位"]), _fmt_num(r["75分位"])])


# ==========
# 典型工况选择工具
# ==========
def _pick_idx_near_target(tm: np.ndarray, td: np.ndarray, target_tm: float, target_td: float, mask: np.ndarray | None = None) -> int:
    tm = np.asarray(tm, dtype=np.float64)
    td = np.asarray(td, dtype=np.float64)
    if mask is None:
        mask = np.ones_like(tm, dtype=bool)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        idxs = np.arange(tm.size)

    tm_s = tm[idxs]
    td_s = td[idxs]

    tm_std = float(np.std(tm_s)) + 1e-12
    td_std = float(np.std(td_s)) + 1e-12
    dist = ((tm_s - target_tm) / tm_std) ** 2 + ((td_s - target_td) / td_std) ** 2
    return int(idxs[int(np.argmin(dist))])

def _pick_idx_by_quantile(x: np.ndarray, q: float, mask: np.ndarray | None = None) -> int:
    x = np.asarray(x, dtype=np.float64)
    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        idxs = np.arange(x.size)

    xs = x[idxs]
    target = float(np.quantile(xs, q))
    j = int(np.argmin(np.abs(xs - target)))
    return int(idxs[j])

# ==========
# 运行仿真并缓存
# ==========
def run_and_save(args):
    out_npz = _result_path(args.mode, args.ckpt, int(args.N), int(args.seed))
    print(f"缓存文件：{out_npz}")

    mode: ModeSpec = make_premix_mode(duration_default=400.0, hold_default=10.0) if args.mode == "premix" \
        else make_production_mode(duration_default=600.0)

    _, agent, ckpt = load_agent_from_ckpt(args.ckpt, device=args.device)
    print(f"加载模型：{args.ckpt}")
    print(f"更新步：{ckpt.get('update')}  情景：{args.mode}  N={args.N}")

    scen_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 12345)

    N = int(args.N)
    scenarios: list[EpisodeParams] = []
    tau_mix_hat = np.zeros(N, dtype=np.float64)
    tau_delay   = np.zeros(N, dtype=np.float64)
    Qs_abs_m3min = np.zeros(N, dtype=np.float64)
    wv_max_flow = np.zeros(N, dtype=np.float64)
    cv_max_flow = np.zeros(N, dtype=np.float64)
    wv_seed     = np.zeros(N, dtype=np.int64)
    cv_seed     = np.zeros(N, dtype=np.int64)

    # 便于后续绘图/典型工况选择的情景参数数组
    h_sp = np.zeros(N, dtype=np.float64)
    rho_sp = np.zeros(N, dtype=np.float64)
    h0 = np.zeros(N, dtype=np.float64)
    rho0 = np.zeros(N, dtype=np.float64)
    qs = np.zeros(N, dtype=np.float64)

    for i in range(N):
        p = mode.sample_episode(scen_rng)
        p.dt = float(args.dt)
        p.mode = mode.name
        scenarios.append(p)

        h_sp[i] = float(p.h_sp)
        rho_sp[i] = float(p.rho_sp)
        h0[i] = float(p.h0)
        rho0[i] = float(p.rho0)
        qs[i] = float(getattr(p, "qs", 0.0))

        tm = float(p.tau_mix)
        td = float(p.tau_delay)
        tm_hat = float(np.clip(eval_rng.normal(tm, max(1e-6, 0.25 * tm)), 5.0, 100.0))
        td_hat = float(np.clip(eval_rng.normal(td, max(1e-6, 0.25 * td)), 0.0, 20.0))
        tau_mix_hat[i] = tm_hat
        tau_delay[i]   = td_hat

        Qs_abs_m3min[i] = abs(float(getattr(p, "qs", 0.0))) * 60.0

        wv0 = float(p.water_valve_max_flow)
        cv0 = float(p.cement_valve_max_flow)
        wv_max_flow[i] = float(np.clip(eval_rng.normal(wv0, max(1e-9, 0.25 * wv0)), 1.5/60, 3.0/60))
        cv_max_flow[i] = float(np.clip(eval_rng.normal(cv0, max(1e-9, 0.25 * cv0)), 0.5/60, 2.0/60))
        wv_seed[i] = int(eval_rng.integers(0, 2**31 - 1))
        cv_seed[i] = int(eval_rng.integers(0, 2**31 - 1))

    base_scales = {spec.name: 1.0 for spec in mode.action_specs}

    base_pid_w = np.zeros((N, 3), dtype=np.float64)
    base_pid_c = np.zeros((N, 3), dtype=np.float64)
    rl_pid_w   = np.zeros((N, 3), dtype=np.float64)
    rl_pid_c   = np.zeros((N, 3), dtype=np.float64)

    T = int(math.floor(float(scenarios[0].duration_s) / float(scenarios[0].dt))) + 1
    t_common = np.asarray(np.arange(T, dtype=np.float64) * float(scenarios[0].dt), dtype=np.float64)

    rho_base = np.zeros((N, T), dtype=np.float32)
    rho_rl   = np.zeros((N, T), dtype=np.float32)
    h_base   = np.zeros((N, T), dtype=np.float32)
    h_rl     = np.zeros((N, T), dtype=np.float32)
    uw_base  = np.zeros((N, T), dtype=np.float32)
    uw_rl    = np.zeros((N, T), dtype=np.float32)
    uc_base  = np.zeros((N, T), dtype=np.float32)
    uc_rl    = np.zeros((N, T), dtype=np.float32)

    # 先分配 IAE（归一化后的）
    iae_rho_base = np.zeros(N, dtype=np.float64)
    iae_rho_rl = np.zeros(N, dtype=np.float64)
    iae_h_base = np.full(N, np.nan, dtype=np.float64)
    iae_h_rl = np.full(N, np.nan, dtype=np.float64)
    # --- 归一化尺度（每个episode一个尺度，避免密度/液位量纲差异导致数值悬殊） ---
    rho_scale = 1000
    if args.mode == "production":
        h_scale = 1


    for i, p in enumerate(scenarios):
        p_hat = deepcopy(p)
        p_hat.water_valve_max_flow = float(wv_max_flow[i])
        p_hat.cement_valve_max_flow = float(cv_max_flow[i])

        obs = mode.build_context(p_hat)
        a = act_deterministic(agent, obs)
        rl_scales = action_to_scales(a, mode.action_specs)

        base_params = mode.compute_base_params(p_hat)

        base_pid_w[i, :] = [float(base_params.get("Kp_w", 0.0)), float(base_params.get("Ki_w", 0.0)), float(base_params.get("Kd_w", 0.0))]
        base_pid_c[i, :] = [float(base_params.get("Kp_c", 0.0)), float(base_params.get("Ki_c", 0.0)), float(base_params.get("Kd_c", 0.0))]

        rl_pid_w[i, :] = [base_pid_w[i, 0] * float(rl_scales.get("s_w_p", 1.0)),
                          base_pid_w[i, 1] * float(rl_scales.get("s_w_i", 1.0)),
                          base_pid_w[i, 2] * float(rl_scales.get("s_w_d", 1.0))]
        rl_pid_c[i, :] = [base_pid_c[i, 0] * float(rl_scales.get("s_c_p", 1.0)),
                          base_pid_c[i, 1] * float(rl_scales.get("s_c_i", 1.0)),
                          base_pid_c[i, 2] * float(rl_scales.get("s_c_d", 1.0))]

        tb = simulate_episode_with_uncertainty(
            p_hat, base_scales,
            base_params=base_params,
            tau_mix_hat=float(tau_mix_hat[i]),
            tau_delay=float(tau_delay[i]),
            wv_max_flow=float(wv_max_flow[i]),
            cv_max_flow=float(cv_max_flow[i]),
            wv_noise_seed=int(wv_seed[i]),
            cv_noise_seed=int(cv_seed[i]),
        )
        tr = simulate_episode_with_uncertainty(
            p_hat, rl_scales,
            base_params=base_params,
            tau_mix_hat=float(tau_mix_hat[i]),
            tau_delay=float(tau_delay[i]),
            wv_max_flow=float(wv_max_flow[i]),
            cv_max_flow=float(cv_max_flow[i]),
            wv_noise_seed=int(wv_seed[i]),
            cv_noise_seed=int(cv_seed[i]),
        )

        rho_base[i, :] = np.asarray(tb.rho, dtype=np.float32)
        rho_rl[i, :]   = np.asarray(tr.rho, dtype=np.float32)
        h_base[i, :]   = np.asarray(tb.h, dtype=np.float32)
        h_rl[i, :]     = np.asarray(tr.h, dtype=np.float32)
        uw_base[i, :]  = np.asarray(tb.u_w, dtype=np.float32)
        uw_rl[i, :]    = np.asarray(tr.u_w, dtype=np.float32)
        uc_base[i, :]  = np.asarray(tb.u_c, dtype=np.float32)
        uc_rl[i, :]    = np.asarray(tr.u_c, dtype=np.float32)

        iae_rho_base[i] = compute_iae(tb.rho, p_hat.rho_sp, p_hat.dt) / float(rho_scale)
        iae_rho_rl[i] = compute_iae(tr.rho, p_hat.rho_sp, p_hat.dt) / float(rho_scale)

        if args.mode == "production":
            iae_h_base[i] = compute_iae(tb.h, p_hat.h_sp, p_hat.dt) / float(h_scale)
            iae_h_rl[i] = compute_iae(tr.h, p_hat.h_sp, p_hat.dt) / float(h_scale)

        if (i + 1) % max(1, N // 10) == 0:
            print(f"进度：{i+1}/{N}")

    np.savez_compressed(
        out_npz,
        mode=str(args.mode),
        ckpt=str(args.ckpt),
        N=int(N),
        seed=int(args.seed),
        dt=float(args.dt),
        T=int(T),
        t=t_common,

        tau_mix_hat=tau_mix_hat,
        tau_delay=tau_delay,
        Qs_abs_m3min=Qs_abs_m3min,

        h_sp=h_sp, rho_sp=rho_sp, h0=h0, rho0=rho0, qs=qs,

        base_pid_w=base_pid_w,
        base_pid_c=base_pid_c,
        rl_pid_w=rl_pid_w,
        rl_pid_c=rl_pid_c,

        rho_base=rho_base,
        rho_rl=rho_rl,
        h_base=h_base,
        h_rl=h_rl,
        uw_base=uw_base,
        uw_rl=uw_rl,
        uc_base=uc_base,
        uc_rl=uc_rl,

        iae_rho_base=iae_rho_base,
        iae_rho_rl=iae_rho_rl,
        iae_h_base=iae_h_base,
        iae_h_rl=iae_h_rl,

        wv_max_flow=wv_max_flow,
        cv_max_flow=cv_max_flow,
        wv_seed=wv_seed,
        cv_seed=cv_seed,
    )

    print(f"\n仿真完成，已缓存：{out_npz}\n")


# ==========
# 读取缓存并作图
# ==========

def load_cache(args) -> dict:
    path = _result_path(args.mode, args.ckpt, int(args.N), int(args.seed))
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到缓存文件：{path}\n请先运行 --run 生成缓存。")
    return dict(np.load(path, allow_pickle=False))


def _build_p_hat_from_cache(
    d: dict, mode_name: str, idx: int, *,
    tau_mix_hat: np.ndarray, tau_delay: np.ndarray,
    wv_max_flow: np.ndarray, cv_max_flow: np.ndarray
) -> EpisodeParams:
    """
    从缓存构造“典型工况”的 p_hat（用于 context、base_params 以及统一 setpoint 的重新仿真）
    注意：EpisodeParams 不能空构造，必须把必填字段一次性传入。
    """

    # 初始值保持该典型工况；目标统一为固定 setpoint
    p_hat = EpisodeParams(
        mode=mode_name,
        duration_s=300.0 if mode_name == "premix" else 600.0,
        dt=float(d["dt"]),

        h0=float(d["h0"][idx]),
        rho0=float(d["rho0"][idx]),
        h_sp=float(H_SP_FIXED),
        rho_sp=float(RHO_SP_FIXED),

        # 工况估计量（用于 context 与 base_params）
        tau_mix=float(tau_mix_hat[idx]),
        tau_delay=float(tau_delay[idx]),

        # max_flow（用于 base_params）
        water_valve_max_flow=float(wv_max_flow[idx]),
        cement_valve_max_flow=float(cv_max_flow[idx]),
    )

    # 排量：production 用该典型工况的排量（不强行统一）
    if mode_name == "production" and ("qs" in d):
        # 有的 EpisodeParams 把 qs 设为可选字段；这里用 setattr 更稳
        try:
            setattr(p_hat, "qs", float(d["qs"][idx]))
        except Exception:
            pass

    return p_hat


def plot_from_cache(args):
    d = load_cache(args)
    outdir = _ensure_dir(f"./paper_figs_{args.mode}/seed_{args.seed}")

    tau_mix_hat = d["tau_mix_hat"].astype(np.float64)
    tau_delay = d["tau_delay"].astype(np.float64)

    h_sp = d["h_sp"].astype(np.float64)
    rho_sp = d["rho_sp"].astype(np.float64)

    base_pid_w = d["base_pid_w"].astype(np.float64)
    base_pid_c = d["base_pid_c"].astype(np.float64)
    rl_pid_w   = d["rl_pid_w"].astype(np.float64)
    rl_pid_c   = d["rl_pid_c"].astype(np.float64)

    iae_rho_base = d["iae_rho_base"].astype(np.float64)
    iae_rho_rl   = d["iae_rho_rl"].astype(np.float64)
    iae_h_base   = d["iae_h_base"].astype(np.float64)
    iae_h_rl     = d["iae_h_rl"].astype(np.float64)

    wv_max_flow = d["wv_max_flow"].astype(np.float64)
    cv_max_flow = d["cv_max_flow"].astype(np.float64)
    wv_seed = d["wv_seed"].astype(np.int64)
    cv_seed = d["cv_seed"].astype(np.int64)

    # ==========
    # 1) scale-context 相关性热力图（Spearman）
    # ==========
    scale_c = rl_pid_c / np.clip(base_pid_c, 1e-12, None)
    scale_w = rl_pid_w / np.clip(base_pid_w, 1e-12, None)

    # 工况变量（中文）
    ctx_cn: dict[str, np.ndarray] = {
        "混合时间常数": tau_mix_hat,
        "观测时滞": tau_delay,
        "目标密度": rho_sp,
        "起始密度": d["rho0"].astype(np.float64),
        "起始液位": d["h0"].astype(np.float64),
    }

    if args.mode == "production":
        ctx_cn["目标液位"] = h_sp
        ctx_cn["排量"] = d["qs"].astype(np.float64)

    # premix：不需要 qs/h_sp/Qs_abs，已经满足

    # 绘制 heatmap：premix 仅灰阀；production 灰阀+水阀
    if args.mode == "premix":
        plot_corr_heatmap(os.path.join(outdir, "图_相关性热力图_灰阀.png"), ctx_cn=ctx_cn, y3=scale_c, cmap="RdBu_r")
    if args.mode == "production":
        plot_corr_heatmap(os.path.join(outdir, "图_相关性热力图_灰阀.png"), ctx_cn=ctx_cn, y3=scale_c, cmap="coolwarm")
        plot_corr_heatmap(os.path.join(outdir, "图_相关性热力图_水阀.png"), ctx_cn=ctx_cn, y3=scale_w, cmap="PuOr")

    # ==========
    # 2) 典型工况：统一目标再仿真（基线 vs RL）
    # ==========
    mode: ModeSpec = make_premix_mode(duration_default=400.0, hold_default=10.0) if args.mode == "premix" \
        else make_production_mode(duration_default=600.0)
    _, agent, _ = load_agent_from_ckpt(args.ckpt, device=args.device)
    base_scales = {spec.name: 1.0 for spec in mode.action_specs}

    tm_small = float(np.quantile(tau_mix_hat, 0.10))
    tm_large = float(np.quantile(tau_mix_hat, 0.90))
    td_small = float(np.quantile(tau_delay,   0.10))
    td_large = float(np.quantile(tau_delay,   0.90))

    # 典型工况（production）额外约束：起始状态应接近统一目标，便于对比
    mask_typical = None
    if args.mode == "production" and ("h0" in d) and ("rho0" in d):
        h0_all = d["h0"].astype(np.float64)
        rho0_all = d["rho0"].astype(np.float64)

        # 目标附近：密度±50 kg/m^3，液位±0.1 m
        mask_typical = (np.abs(rho0_all - float(RHO_SP_FIXED)) <= 20.0) & \
                       (np.abs(h0_all - float(H_SP_FIXED)) <= 0.05)

        # 防止 mask 过窄导致选不到样本：样本太少时回退到不加约束
        if int(np.count_nonzero(mask_typical)) < 5:
            mask_typical = None

    idx_tmS_tdS = _pick_idx_near_target(tau_mix_hat, tau_delay, tm_small, td_small, mask=mask_typical)
    idx_tmS_tdL = _pick_idx_near_target(tau_mix_hat, tau_delay, tm_small, td_large, mask=mask_typical)
    idx_tmL_tdS = _pick_idx_near_target(tau_mix_hat, tau_delay, tm_large, td_small, mask=mask_typical)
    idx_tmL_tdL = _pick_idx_near_target(tau_mix_hat, tau_delay, tm_large, td_large, mask=mask_typical)

    pairs = [
        ("小惯性_小时滞", idx_tmS_tdS),
        ("小惯性_大时滞", idx_tmS_tdL),
        ("大惯性_小时滞", idx_tmL_tdS),
        ("大惯性_大时滞", idx_tmL_tdL),
    ]

    def _fmt_hpid(x: float) -> str:
        # production 的液位图：不保留小数（或可改成 .1f）
        return f"{int(round(float(x)))}"

    for tag, idx in pairs:
        p_hat = _build_p_hat_from_cache(
            d, mode_name=args.mode, idx=idx,
            tau_mix_hat=tau_mix_hat, tau_delay=tau_delay,
            wv_max_flow=wv_max_flow, cv_max_flow=cv_max_flow,
        )

        # RL scales
        obs = mode.build_context(p_hat)
        a = act_deterministic(agent, obs)
        rl_scales = action_to_scales(a, mode.action_specs)

        # 基线参数（在统一 setpoint 下重新计算）
        base_params = mode.compute_base_params(p_hat)

        tb = simulate_episode_with_uncertainty(
            p_hat, base_scales,
            base_params=base_params,
            tau_mix_hat=float(tau_mix_hat[idx]),
            tau_delay=float(tau_delay[idx]),
            wv_max_flow=float(wv_max_flow[idx]),
            cv_max_flow=float(cv_max_flow[idx]),
            wv_noise_seed=int(wv_seed[idx]),
            cv_noise_seed=int(cv_seed[idx]),
        )
        tr = simulate_episode_with_uncertainty(
            p_hat, rl_scales,
            base_params=base_params,
            tau_mix_hat=float(tau_mix_hat[idx]),
            tau_delay=float(tau_delay[idx]),
            wv_max_flow=float(wv_max_flow[idx]),
            cv_max_flow=float(cv_max_flow[idx]),
            wv_noise_seed=int(wv_seed[idx]),
            cv_noise_seed=int(cv_seed[idx]),
        )

        if args.mode == "premix":
            # 只显示灰阀 PID
            kp0, ki0, kd0 = float(base_params.get("Kp_c", 0.0)), float(base_params.get("Ki_c", 0.0)), float(base_params.get("Kd_c", 0.0))
            kp1, ki1, kd1 = kp0 * float(rl_scales.get("s_c_p", 1.0)), ki0 * float(rl_scales.get("s_c_i", 1.0)), kd0 * float(rl_scales.get("s_c_d", 1.0))
            label_base = f"{LABEL_BASE}（kp={_fmt_num(kp0)}, ki={_fmt_num(ki0)}, kd={_fmt_num(kd0)}）"
            label_rl   = f"{LABEL_RL}（kp={_fmt_num(kp1)}, ki={_fmt_num(ki1)}, kd={_fmt_num(kd1)}）"

            plot_typical_compare_premix(
                os.path.join(outdir, f"图_典型工况_{tag}_密度.png"),
                tb, tr,
                rho_sp=float(RHO_SP_FIXED),
                label_base=label_base,
                label_rl=label_rl,
            )
        else:
            # production：液位图只显示水 PID；密度图只显示灰 PID
            kp_w0, ki_w0, kd_w0 = float(base_params.get("Kp_w", 0.0)), float(base_params.get("Ki_w", 0.0)), float(base_params.get("Kd_w", 0.0))
            kp_w1, ki_w1, kd_w1 = kp_w0 * float(rl_scales.get("s_w_p", 1.0)), ki_w0 * float(rl_scales.get("s_w_i", 1.0)), kd_w0 * float(rl_scales.get("s_w_d", 1.0))
            label_base_h = f"{LABEL_BASE}（kp={_fmt_hpid(kp_w0)}, ki={_fmt_hpid(ki_w0)}, kd={_fmt_hpid(kd_w0)}）"
            label_rl_h   = f"{LABEL_RL}（kp={_fmt_hpid(kp_w1)}, ki={_fmt_hpid(ki_w1)}, kd={_fmt_hpid(kd_w1)}）"

            kp_c0, ki_c0, kd_c0 = float(base_params.get("Kp_c", 0.0)), float(base_params.get("Ki_c", 0.0)), float(base_params.get("Kd_c", 0.0))
            kp_c1, ki_c1, kd_c1 = kp_c0 * float(rl_scales.get("s_c_p", 1.0)), ki_c0 * float(rl_scales.get("s_c_i", 1.0)), kd_c0 * float(rl_scales.get("s_c_d", 1.0))
            label_base_r = f"{LABEL_BASE}（kp={_fmt_num(kp_c0)}, ki={_fmt_num(ki_c0)}, kd={_fmt_num(kd_c0)}）"
            label_rl_r   = f"{LABEL_RL}（kp={_fmt_num(kp_c1)}, ki={_fmt_num(ki_c1)}, kd={_fmt_num(kd_c1)}）"

            plot_typical_compare_production_h(
                os.path.join(outdir, f"图_典型工况_{tag}_液位.png"),
                tb, tr,
                h_sp=float(H_SP_FIXED),
                label_base=label_base_h,
                label_rl=label_rl_h,
            )
            plot_typical_compare_production_rho(
                os.path.join(outdir, f"图_典型工况_{tag}_密度.png"),
                tb, tr,
                rho_sp=float(RHO_SP_FIXED),
                label_base=label_base_r,
                label_rl=label_rl_r,
            )
    # ==========
    # 2.5) production：外输(Qs) 小/中/大典型工况（20%/50%/80%），统一目标再仿真
    # ==========
    if args.mode == "production":
        Qs_abs_m3min = d["Qs_abs_m3min"].astype(np.float64)  # run_and_save 已存 :contentReference[oaicite:5]{index=5}

        # 起始点接近目标（你要求：±20 kg/m3，±0.05 m）
        h0_all = d["h0"].astype(np.float64)
        rho0_all = d["rho0"].astype(np.float64)
        mask_start = (np.abs(rho0_all - float(RHO_SP_FIXED)) <= 20.0) & \
                     (np.abs(h0_all   - float(H_SP_FIXED))   <= 0.05)

        # 为了避免把 τ_mix/τ_delay 极端样本选进来：取中间段（可选，但建议）
        q_tm_lo, q_tm_hi = np.quantile(tau_mix_hat, [0.40, 0.60])
        q_td_lo, q_td_hi = np.quantile(tau_delay,   [0.40, 0.60])
        mask_mid = (tau_mix_hat >= q_tm_lo) & (tau_mix_hat <= q_tm_hi) & \
                   (tau_delay   >= q_td_lo) & (tau_delay   <= q_td_hi)

        mask_qs = mask_start & mask_mid
        if int(np.count_nonzero(mask_qs)) < 5:
            # 回退：只要求起始点接近目标
            mask_qs = mask_start
        if int(np.count_nonzero(mask_qs)) < 5:
            # 再回退：不加任何约束，保证一定能选到
            mask_qs = None

        idx_qs_20 = _pick_idx_by_quantile(Qs_abs_m3min, 0.20, mask=mask_qs)
        idx_qs_50 = _pick_idx_by_quantile(Qs_abs_m3min, 0.50, mask=mask_qs)
        idx_qs_80 = _pick_idx_by_quantile(Qs_abs_m3min, 0.80, mask=mask_qs)

        qs_triplets = [("小外输(20%)", idx_qs_20), ("中外输(50%)", idx_qs_50), ("大外输(80%)", idx_qs_80)]

        for tag, idx in qs_triplets:
            # 构造该典型工况 p_hat：这里本来就会强制统一 setpoint（见 _build_p_hat_from_cache）:contentReference[oaicite:6]{index=6}
            p_hat = _build_p_hat_from_cache(
                d, mode_name=args.mode, idx=idx,
                tau_mix_hat=tau_mix_hat, tau_delay=tau_delay,
                wv_max_flow=wv_max_flow, cv_max_flow=cv_max_flow,
            )
            # 再显式写一遍，避免未来你改 _build_p_hat_from_cache 破坏一致性
            p_hat.h_sp = float(H_SP_FIXED)
            p_hat.rho_sp = float(RHO_SP_FIXED)

            obs = mode.build_context(p_hat)
            a = act_deterministic(agent, obs)
            rl_scales = action_to_scales(a, mode.action_specs)

            base_params = mode.compute_base_params(p_hat)

            tb = simulate_episode_with_uncertainty(
                p_hat, base_scales,
                base_params=base_params,
                tau_mix_hat=float(tau_mix_hat[idx]),
                tau_delay=float(tau_delay[idx]),
                wv_max_flow=float(wv_max_flow[idx]),
                cv_max_flow=float(cv_max_flow[idx]),
                wv_noise_seed=int(wv_seed[idx]),
                cv_noise_seed=int(cv_seed[idx]),
            )
            tr = simulate_episode_with_uncertainty(
                p_hat, rl_scales,
                base_params=base_params,
                tau_mix_hat=float(tau_mix_hat[idx]),
                tau_delay=float(tau_delay[idx]),
                wv_max_flow=float(wv_max_flow[idx]),
                cv_max_flow=float(cv_max_flow[idx]),
                wv_noise_seed=int(wv_seed[idx]),
                cv_noise_seed=int(cv_seed[idx]),
            )

            # legend：液位图只显示水PID；密度图只显示灰PID（沿用你现有写法）:contentReference[oaicite:7]{index=7}
            kp_w0, ki_w0, kd_w0 = float(base_params.get("Kp_w", 0.0)), float(base_params.get("Ki_w", 0.0)), float(base_params.get("Kd_w", 0.0))
            kp_w1, ki_w1, kd_w1 = kp_w0 * float(rl_scales.get("s_w_p", 1.0)), ki_w0 * float(rl_scales.get("s_w_i", 1.0)), kd_w0 * float(rl_scales.get("s_w_d", 1.0))
            label_base_h = f"{LABEL_BASE}（kp={_fmt_hpid(kp_w0)}, ki={_fmt_hpid(ki_w0)}, kd={_fmt_hpid(kd_w0)}）"
            label_rl_h   = f"{LABEL_RL}（kp={_fmt_hpid(kp_w1)}, ki={_fmt_hpid(ki_w1)}, kd={_fmt_hpid(kd_w1)}）"

            kp_c0, ki_c0, kd_c0 = float(base_params.get("Kp_c", 0.0)), float(base_params.get("Ki_c", 0.0)), float(base_params.get("Kd_c", 0.0))
            kp_c1, ki_c1, kd_c1 = kp_c0 * float(rl_scales.get("s_c_p", 1.0)), ki_c0 * float(rl_scales.get("s_c_i", 1.0)), kd_c0 * float(rl_scales.get("s_c_d", 1.0))
            label_base_r = f"{LABEL_BASE}（kp={_fmt_num(kp_c0)}, ki={_fmt_num(ki_c0)}, kd={_fmt_num(kd_c0)}）"
            label_rl_r   = f"{LABEL_RL}（kp={_fmt_num(kp_c1)}, ki={_fmt_num(ki_c1)}, kd={_fmt_num(kd_c1)}）"

            plot_typical_compare_production_h(
                os.path.join(outdir, f"图_典型工况_{tag}_液位.png"),
                tb, tr,
                h_sp=float(H_SP_FIXED),
                label_base=label_base_h,
                label_rl=label_rl_h,
            )
            plot_typical_compare_production_rho(
                os.path.join(outdir, f"图_典型工况_{tag}_密度.png"),
                tb, tr,
                rho_sp=float(RHO_SP_FIXED),
                label_base=label_base_r,
                label_rl=label_rl_r,
            )

    # ==========
    # 3) 全体情景统计表（CSV）
    # ==========
    summarize_and_save_csv(os.path.join(outdir, "表_IAE_密度_统计.csv"), iae_rho_base, iae_rho_rl, "IAE_密度")
    if args.mode == "production":
        summarize_and_save_csv(os.path.join(outdir, "表_IAE_液位_统计.csv"), iae_h_base, iae_h_rl, "IAE_液位")

    # ==========
    # 4) IAE 分布对比图（小提琴+箱线）
    # ==========
    plot_violin_box(os.path.join(outdir, "图_IAE分布对比_密度.png"), iae_rho_base, iae_rho_rl, LABEL_BASE, LABEL_RL, ylabel="IAE（密度）")
    if args.mode == "production":
        plot_violin_box(
            os.path.join(outdir, "图_IAE分布对比_液位.png"),
            iae_h_base, iae_h_rl,
            LABEL_BASE, LABEL_RL,
            ylabel="IAE（液位）",
            violin_colors=("#BFEBC9", "#F6C1B5"),  # 浅绿 / 浅暖红
            box_colors=("#24A148", "#D1495B"),  # 绿 / 暖红
        )

    # ==========
    # 5) 关键工况分组箱线图（ΔIAE=λ整定-RL整定）
    #    production：把“密度 + 液位”合并到一张图（每个自变量一张），6张 -> 3张
    # ==========
    d_iae_rho = iae_rho_base - iae_rho_rl

    if args.mode != "production":
        # premix：仍只画密度（保持原逻辑）
        plot_grouped_delta_box(
            os.path.join(outdir, "图_分组箱线_ΔIAE密度_按混合惯性.png"),
            tau_mix_hat, d_iae_rho,
            xlabel="混合时间常数 / s",
            nbins=int(args.nbins_tau_mix),
            x_min=0,
        )
        plot_grouped_delta_box(
            os.path.join(outdir, "图_分组箱线_ΔIAE密度_按观测时滞.png"),
            tau_delay, d_iae_rho,
            xlabel="观测时滞 / s",
            nbins=int(args.nbins_delay),
            x_min=0,
        )
    else:
        # production：密度 + 液位 合并
        d_iae_h = iae_h_base - iae_h_rl
        Qs_abs_m3min = d["Qs_abs_m3min"].astype(np.float64)

        plot_grouped_delta_box_dual(
            os.path.join(outdir, "图_分组箱线_ΔIAE_密度+液位_按混合惯性.png"),
            tau_mix_hat, d_iae_rho, d_iae_h,
            xlabel="混合时间常数 / s",
            nbins=int(args.nbins_tau_mix),
            x_min=0,
            label_rho="密度",
            label_h="液位",
        )

        plot_grouped_delta_box_dual(
            os.path.join(outdir, "图_分组箱线_ΔIAE_密度+液位_按观测时滞.png"),
            tau_delay, d_iae_rho, d_iae_h,
            xlabel="观测时滞 / s",
            nbins=int(args.nbins_delay),
            x_min=0,
            label_rho="密度",
            label_h="液位",
        )

        # 注意：别忘记“按排量”——你提醒的这张在这里合并输出一张
        plot_grouped_delta_box_dual(
            os.path.join(outdir, "图_分组箱线_ΔIAE_密度+液位_按排量.png"),
            Qs_abs_m3min, d_iae_rho, d_iae_h,
            xlabel="排量 / (m$^3$/min)",
            nbins=int(args.nbins_qs),
            label_rho="密度",
            label_h="液位",
        )

    # ==========
    # 6) PID 三参数 scale 统计图（violin+box）
    # ==========
    plot_violin_box_3(
        os.path.join(outdir, "图_scale_灰阀_PID.png"),
        scale_c,
        labels3=["Kp", "Ki", "Kd"],
        ylabel="缩放因子",
        hline=1.0,
    )

    if args.mode == "production":
        # 水阀(液位) scale：保持原风格，只换成绿色系填充
        plot_violin_box_3(
            os.path.join(outdir, "图_scale_水阀_PID.png"),
            scale_w,
            labels3=["Kp", "Ki", "Kd"],
            ylabel="缩放因子",
            hline=1.0,
            violin_colors=("#CFEFD6", "#BFEBC9", "#AEE6BF"),  # 浅绿（violin）
            box_colors=("#24A148", "#2FBF71", "#1E8E5A"),  # 绿（box）
        )


# ==========
# 主入口
# ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["premix", "production"])
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dt", type=float, default=0.5)

    # 分箱数：严格按指定 nbins
    ap.add_argument("--nbins_tau_mix", type=int, default=4)
    ap.add_argument("--nbins_delay", type=int, default=4)
    ap.add_argument("--nbins_qs", type=int, default=4)

    ap.add_argument("--run", action="store_true", help="只运行仿真并缓存结果")
    ap.add_argument("--plot", action="store_true", help="只从缓存作图（不仿真）")
    args = ap.parse_args()

    if (not args.run) and (not args.plot):
        # 默认：先 run 再 plot
        args.run = True
        args.plot = True

    if args.run:
        run_and_save(args)

    if args.plot:
        plot_from_cache(args)


if __name__ == "__main__":
    main()
