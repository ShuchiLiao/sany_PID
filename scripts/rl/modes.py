# modes.py
"""
modes.py

定义“运行模式（premix/production）”的可插拔组件：
- EpisodeParams: 每个 episode 的工况参数（上下文）
- Trajectory: 仿真输出的轨迹（rho/h/u 等时间序列）
- ModeSpec: 一个 mode 的完整定义（采样器、状态构造、动作映射、reward）

本项目特点：
- 上下文老虎机：每个 episode 仅在开始时输出一次“缩放因子”
- 缩放因子作用于 base PID/前馈参数，episode 内参数固定不变
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import math
import numpy as np

# --- NEW: base-parameter computation (scenario -> base PID/FF) ---
from copy import deepcopy

# tune_baseline_v3 里提供了“按工况粗算 base PID + FF”的函数
from scripts.PID_control.tune_baseline import (
    tune_baseline_params,          # production base
    tune_premix_base_params,        # premix base
    EngineeringKnobs,
    PremixKnobs,
    BaselineTuningResult,
    PremixTuningResult,
)

# sim_config 里是 PlantParams / SimulationConfig / ValveParams
from scripts.core.sim_config import PlantParams, SimulationConfig, ValveParams

# -----------------------------
# Data containers
# -----------------------------

@dataclass
class EpisodeParams:
    """一个 episode 的工况参数（上下文）。"""
    mode: str  # "premix" or "production"
    duration_s: float  # premix 默认 300, production 默认 600
    dt: float  # 仿真步长（由你 env 决定）

    # 初始与目标
    h0: float
    rho0: float
    h_sp: float
    rho_sp: float

    # 动力学不确定性（预估值）
    tau_mix: float
    tau_delay: float

    # 阀门-流量线性对应参数
    water_valve_max_flow: float
    cement_valve_max_flow: float

    # 出流信息
    qs: float = 0.0

    # premix “达到”判据
    premix_hold_s: float = 10.0  # 默认 10s 持续满足 rho >= 0.95*rho_sp

# -----------------------------
# NEW: Base parameters computed from scenario
# -----------------------------

@dataclass
class BaseParamsPremix:
    """premix（Qs=0）灰阀密度回路 base（PID）。"""
    rho_kp: float
    rho_ki: float
    rho_kd: float


@dataclass
class BaseParamsProduction:
    """production（Qs step）双回路 base（开度域[%] + PID + kff）。"""
    uw_ff: float
    uc_ff: float
    h_kp: float
    h_ki: float
    h_kd: float
    rho_kp: float
    rho_ki: float
    rho_kd: float
    kff: float


def _make_tuning_objects_from_episode(
    p: EpisodeParams,
    *,
    plant_template: Optional[PlantParams] = None,
    sim_template: Optional[SimulationConfig] = None,
    water_valve_params_template:Optional[ValveParams] = None,
    cement_valve_params_template:Optional[ValveParams] = None,
) -> tuple[PlantParams, SimulationConfig, ValveParams, ValveParams]:
    """
    用 EpisodeParams 覆盖/构造 tune_baseline_v3 所需的 PlantParams / SimulationConfig：
    - plant.tau_mix_hat <- p.tau_mix
    - sim.rho_obs_delay <- p.tau_delay
    - sim.rho_sp/h_sp/dt/t_end <- episode 对应值
    """
    plant = deepcopy(plant_template) if plant_template is not None else PlantParams()
    sim = deepcopy(sim_template) if sim_template is not None else SimulationConfig()
    wv_params = deepcopy(water_valve_params_template) if water_valve_params_template is not None else ValveParams()
    cv_params = deepcopy(cement_valve_params_template) if cement_valve_params_template is not None else ValveParams()


    # 工况覆盖（这几项就是 tune_baseline_v3 用到的关键量）
    plant.tau_mix_hat = float(p.tau_mix)

    sim.dt = float(p.dt)
    sim.t_end = float(p.duration_s)
    sim.h_sp = float(p.h_sp)
    sim.rho_sp = float(p.rho_sp)

    # “无法直接读取”的 delay：这里用 episode 的 tau_delay 作为 rho_obs_delay 的估计/设定
    sim.rho_obs_delay = float(p.tau_delay)

    wv_params.max_flow = p.water_valve_max_flow
    cv_params.max_flow = p.cement_valve_max_flow

    return plant, sim, wv_params, cv_params


def compute_base_premix(
    p: EpisodeParams,
    *,
    plant_template: Optional[PlantParams] = None,
    sim_template: Optional[SimulationConfig] = None,
    water_valve_params_template: Optional[ValveParams] = None,
    cement_valve_params_template: Optional[ValveParams] = None,
    apply_premix_knobs: bool = False,
    premix_knobs: Optional[PremixKnobs] = None,
) -> BaseParamsPremix:
    """
    premix base：复用 tune_baseline_v3.tune_premix_base_params() 的逻辑。
    h_level 推荐用 h_sp（也可用 h0）；这里取 max(h0, h_sp) 更稳一点。
    """
    assert p.mode == "premix"
    plant, sim, wv_params, cv_params = _make_tuning_objects_from_episode(p, plant_template=plant_template, sim_template=sim_template,
                                                                         water_valve_params_template=water_valve_params_template,
                                                                         cement_valve_params_template=cement_valve_params_template)
    h_level = float(max(p.h0, p.h_sp))

    res: PremixTuningResult = tune_premix_base_params(
        plant=plant,
        sim=sim,
        cement_valve_params=cv_params,
        h_level=h_level,
        apply_premix_knobs=apply_premix_knobs,
        knobs=premix_knobs,
    )

    return BaseParamsPremix(
        rho_kp=float(res.rho_kp),
        rho_ki=float(res.rho_ki),
        rho_kd=float(res.rho_kd),
    )


def compute_base_production(
    p: EpisodeParams,
    *,
    plant_template: Optional[PlantParams] = None,
    sim_template: Optional[SimulationConfig] = None,
    water_valve_params_template: Optional[ValveParams] = None,
    cement_valve_params_template: Optional[ValveParams] = None,
    apply_engineering_knobs: bool = False,
    eng_knobs: Optional[EngineeringKnobs] = None,
) -> BaseParamsProduction:
    """
    production base：复用 tune_baseline_v3.tune_baseline_params() 的逻辑。
    Qs_step 取“阶跃后的目标排量”更贴近你生产阶段的稳态工作点：优先用 qs1，否则退回 qs0。
    """
    assert p.mode == "production"
    plant, sim, wv_params, cv_params = _make_tuning_objects_from_episode(p, plant_template=plant_template, sim_template=sim_template,
                                                                         water_valve_params_template=water_valve_params_template,
                                                                         cement_valve_params_template=cement_valve_params_template)

    qs_step = float(p.qs)
    if qs_step <= 1e-9:
        qs_step = float(sim.Qs_nominal)  # 兜底

    res: BaselineTuningResult = tune_baseline_params(
        plant=plant,
        sim=sim,
        water_valve_params=wv_params,
        cement_valve_params=cv_params,
        Qs_step=qs_step,
        apply_engineering_knobs=apply_engineering_knobs,
        knobs=eng_knobs,
    )

    return BaseParamsProduction(
        uw_ff=float(res.uw_ff),
        uc_ff=float(res.uc_ff),
        h_kp=float(res.h_kp),
        h_ki=float(res.h_ki),
        h_kd=float(res.h_kd),
        rho_kp=float(res.rho_kp),
        rho_ki=float(res.rho_ki),
        rho_kd=float(res.rho_kd),
        kff=float(res.kff),
    )


@dataclass
class Trajectory:
    """
    仿真输出轨迹（离散序列），用于 reward 计算。
    你在 simulate_episode() 里只要按这些字段填即可。
    """
    t: np.ndarray              # shape (N,)
    rho: np.ndarray            # shape (N,)
    h: np.ndarray              # shape (N,)

    # 控制量（如果没有，可填 None；reward 中默认小权重项也可不用）
    u_w: Optional[np.ndarray] = None  # 水阀控制量/开度/流量增量等
    u_c: Optional[np.ndarray] = None  # 灰阀控制量/开度/流量增量等


# -----------------------------
# Action mapping (policy output -> scaling -> controller params)
# -----------------------------

@dataclass
class ActionDimSpec:
    """
    每个动作维度的缩放映射规格。
    我们让 policy 输出 a in [-1, 1]。
    """
    name: str
    kind: str  # "log10" or "sigmoid"
    alpha: float  # 对数缩放强度（log10: scale=10^(alpha*a)）
    s_min: float
    s_max: float


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def action_to_scales(
    a: np.ndarray,
    specs: Tuple[ActionDimSpec, ...],
) -> Dict[str, float]:
    """
    将 policy 输出 a（每维 [-1,1]）映射到缩放因子 s（有界正数）。
    - log10: s = 10^(alpha*a), 再裁剪到 [s_min, s_max]
    - sigmoid: s = s_min + (s_max-s_min)*sigmoid(alpha*a)
      适用于 Ki，希望能自然逼近 0（s_min 可设很小）
    """
    assert a.shape[0] == len(specs), "action dim mismatch"
    out: Dict[str, float] = {}
    for i, spec in enumerate(specs):
        ai = _clip(float(a[i]), -1.0, 1.0)
        if spec.kind == "log10":
            s = 10.0 ** (spec.alpha * ai)
        elif spec.kind == "sigmoid":
            # s in (0,1) then affine to [s_min, s_max]
            z = spec.alpha * ai
            sig = 1.0 / (1.0 + math.exp(-z))
            s = spec.s_min + (spec.s_max - spec.s_min) * sig
        else:
            raise ValueError(f"Unknown kind: {spec.kind}")
        s = _clip(s, spec.s_min, spec.s_max)
        out[spec.name] = float(s)
    return out


# -----------------------------
# Normalization helpers
# -----------------------------

def _center_scale(
    x: float,
    *,
    x_mid: float,
    x_half: float,
    x_min: float,
    x_max: float,
    clip: float = 3.0,
) -> float:
    """
    先把 x 裁剪到物理范围 [x_min, x_max]，再做居中缩放：
        (x - x_mid) / x_half
    其中 x_mid/x_half 通常来自“集中分布区间”：
        x_mid = (x_c_min + x_c_max)/2
        x_half = (x_c_max - x_c_min)/2
    最后再把归一化值裁剪到 [-clip, clip]，防止极端工况炸掉网络。
    """
    x = float(max(x_min, min(x_max, float(x))))
    y = (x - x_mid) / (x_half + 1e-12)
    y = float(max(-clip, min(clip, y)))
    return y


# ---- 物理范围 & 集中范围 ----
_RHO_MIN, _RHO_MAX = 1000.0, 2500.0
_RHO_C_MIN, _RHO_C_MAX = 1000.0, 2000.0   # 集中分布
_RHO_MID = 0.5 * (_RHO_C_MIN + _RHO_C_MAX)   # 1500
_RHO_HALF = 0.5 * (_RHO_C_MAX - _RHO_C_MIN)  # 500

_H_MIN, _H_MAX = 0.2, 1.8
_H_C_MIN, _H_C_MAX = 0.5, 1.5
_H_MID = 0.5 * (_H_C_MIN + _H_C_MAX)    # 1.0
_H_HALF = 0.5 * (_H_C_MAX - _H_C_MIN)   # 0.5

_QS_MIN, _QS_MAX = 0.2/60.0, 2.0/60.0
_QS_C_MIN, _QS_C_MAX = 0.4/60.0, 1.5/60.0
_QS_MID = 0.5 * (_QS_C_MIN + _QS_C_MAX)   # 0.8/60
_QS_HALF = 0.5 * (_QS_C_MAX - _QS_C_MIN)  # 0.4/60

_TAUM_MIN, _TAUM_MAX = 5.0, 100.0
_TAUM_C_MIN, _TAUM_C_MAX = 10.0, 50.0

_TAUD_MIN, _TAUD_MAX = 0.0, 20.0
_TAUD_C_MIN, _TAUD_C_MAX = 0.0, 10.0

_MAX_FLOW_MIN, _MAX_FLOW_MAX = 0.8/60, 2.8/60
_MAX_FLOW_MID = 0.5 * (_MAX_FLOW_MIN + _MAX_FLOW_MAX)
_MAX_FLOW_HALF = 0.5 * (_MAX_FLOW_MAX - _MAX_FLOW_MIN)

_OPEN_MIN, _OPEN_MAX = 0.0, 100.0
_OPEN_C_MIN, _OPEN_C_MAX = 0.0, 100.0   # 经验“集中区间”，可按你系统再调
_OPEN_MID  = 0.5 * (_OPEN_C_MIN + _OPEN_C_MAX)
_OPEN_HALF = 0.5 * (_OPEN_C_MAX - _OPEN_C_MIN)

def opening_norm(opening_pct: float) -> float:
    return _center_scale(opening_pct, x_mid=_OPEN_MID, x_half=_OPEN_HALF,
                         x_min=_OPEN_MIN, x_max=_OPEN_MAX, clip=3.0)


def rho_norm(rho: float) -> float:
    return _center_scale(rho, x_mid=_RHO_MID, x_half=_RHO_HALF, x_min=_RHO_MIN, x_max=_RHO_MAX, clip=3.0)


def h_norm(h: float) -> float:
    return _center_scale(h, x_mid=_H_MID, x_half=_H_HALF, x_min=_H_MIN, x_max=_H_MAX, clip=3.0)


def qs_norm(qs: float) -> float:
    return _center_scale(qs, x_mid=_QS_MID, x_half=_QS_HALF, x_min=_QS_MIN, x_max=_QS_MAX, clip=3.0)

def max_flow_norm(max_flow: float) -> float:
    return _center_scale(max_flow, x_mid=_MAX_FLOW_MID, x_half=_MAX_FLOW_HALF, x_min=_MAX_FLOW_MIN, x_max=_MAX_FLOW_MAX, clip=3.0)

def tau_mix_norm(tau_mix: float) -> float:
    """
    tau_mix: 用 log 空间做居中缩放，让 10~50s 大致映射到 [-1, 1]。
    """
    tm = float(max(_TAUM_MIN, min(_TAUM_MAX, float(tau_mix))))
    z = math.log(max(tm, 1e-12))

    z_min = math.log(_TAUM_C_MIN)
    z_max = math.log(_TAUM_C_MAX)
    z_mid = 0.5 * (z_min + z_max)
    z_half = 0.5 * (z_max - z_min)

    y = (z - z_mid) / (z_half + 1e-12)
    return float(max(-3.0, min(3.0, y)))


def tau_delay_norm(tau_delay: float) -> float:
    """
    tau_delay: 因为可能为 0，用 log1p(tau) 做居中缩放，让 0~10s 大致映射到 [-1, 1]。
    """
    td = float(max(_TAUD_MIN, min(_TAUD_MAX, float(tau_delay))))
    z = math.log1p(max(td, 0.0))

    z_min = math.log1p(_TAUD_C_MIN)  # 0
    z_max = math.log1p(_TAUD_C_MAX)  # log(11)
    z_mid = 0.5 * (z_min + z_max)
    z_half = 0.5 * (z_max - z_min)

    y = (z - z_mid) / (z_half + 1e-12)
    return float(max(-3.0, min(3.0, y)))


# ---- reward 里仍在用 rho_scale/h_scale：这里改成“固定误差尺度”，避免跟 setpoint 绑定 ----
def rho_scale(_: float, fixed: float = 1000.0) -> float:
    # 用集中分布宽度 1000 作为密度误差归一化尺度（rho 1000~2000）
    return float(fixed)

def h_scale(_: float, fixed: float = 1.0) -> float:
    # 用集中分布宽度 1.0 作为液位误差归一化尺度（h 0.5~1.5）
    return float(fixed)


def build_context_premix(p: EpisodeParams) -> np.ndarray:
    """
    premix 上下文（一次性输入 policy）：
    用“集中分布区间做居中缩放”，并对极端值做裁剪。
    """
    x = np.array([
        h_norm(p.h0),
        rho_norm(p.rho0),
        rho_norm(p.rho_sp),
        tau_mix_norm(p.tau_mix),
        tau_delay_norm(p.tau_delay),
        max_flow_norm(p.cement_valve_max_flow)
    ], dtype=np.float32)
    return x


def build_context_production(p: EpisodeParams) -> np.ndarray:
    """
    production 上下文（一次性输入 policy）：
    """

    if not hasattr(p, "_cached_base_production"):
        p._cached_base_production = compute_base_production(p)
    base = p._cached_base_production

    x = np.array([
        h_norm(p.h0),
        rho_norm(p.rho0),
        h_norm(p.h_sp),
        rho_norm(p.rho_sp),
        qs_norm(p.qs),
        max_flow_norm(p.water_valve_max_flow),
        max_flow_norm(p.cement_valve_max_flow),
        tau_mix_norm(p.tau_mix),
        tau_delay_norm(p.tau_delay),

        opening_norm(base.uw_ff),   # NEW
        opening_norm(base.uc_ff),   # NEW
    ], dtype=np.float32)
    return x


def premix_reward(
    rho: np.ndarray,
    p: EpisodeParams,
    w_e: float = 1.0,
    w_os: float = 0.0,
) -> float:
    """
    premix reward（全时域误差版）：
    - 在整个仿真时间 [0, T] 上，rho(t) 与 rho_sp 的差距越小越好
    - 可选：给超调面积一个很小权重（w_os）
    """
    dt = float(p.dt)
    T = float(p.duration_s)
    rs = rho_scale(p.rho_sp)

    # 全时域 IAE（L1误差积分）
    e_all = float(np.sum(np.abs(rho - float(p.rho_sp))) * dt)

    # 可选：超调面积
    os_area = float(np.sum(np.maximum(0.0, rho - float(p.rho_sp))) * dt)

    # 归一化（让不同 episode 时长 / 不同尺度可比）
    e_term = e_all / (max(T, 1e-6) * rs)
    os_term = os_area / (max(T, 1e-6) * rs)

    return float(-w_e * e_term - w_os * os_term)



def production_reward(
    rho: np.ndarray,
    h: np.ndarray,
    p: EpisodeParams,
    w_rho: float = 0.9,
    w_h: float = 0.1,
    w_tv: float = 0.0,
) -> float:
    """
    与 production_reward(traj,p) 等价，但只需要 rho/h 序列。
    """
    dt = float(p.dt)
    rs = rho_scale(p.rho_sp)
    hs = h_scale(p.h_sp)

    dur = float(p.duration_s)
    e_rho = float(np.sum(np.abs(rho - float(p.rho_sp))) * dt)
    e_h = float(np.sum(np.abs(h - float(p.h_sp))) * dt)
    tv_rho = float(np.sum(np.abs(np.diff(rho))))
    tv_h = float(np.sum(np.abs(np.diff(h))))

    e_rho_n = e_rho / (dur * rs)
    e_h_n = e_h / (dur * hs)
    tv_n = (tv_rho / rs) + (tv_h / hs)

    return float(-w_rho * e_rho_n - w_h * e_h_n - w_tv * tv_n)

# -----------------------------
# ModeSpec and Samplers
# -----------------------------

@dataclass
class ModeSpec:
    """
    一个模式的完整定义：
    - action_specs: 动作维度定义（缩放映射）
    - sample_episode: 采样一个 episode 的上下文
    - build_context: 构造 policy 输入
    - compute_reward: 根据轨迹计算 reward
    - compute_base_params: 根据 EpisodeParams 计算该 episode 的 base 参数（PID/FF/decouple）
    """
    name: str
    action_specs: Tuple[ActionDimSpec, ...]
    sample_episode: Callable[[np.random.Generator], EpisodeParams]
    build_context: Callable[[EpisodeParams], np.ndarray]
    compute_base_params: Callable[[EpisodeParams], Dict[str, float]]


def make_premix_mode(
    *,
    duration_default: float = 300.0,
    hold_default: float = 10.0,
) -> ModeSpec:
    """
    premix：只学习灰阀 PID 三参数缩放。
    让 Ki 更容易学到接近 0：sigmoid 映射到 [0, s_i_max]
    """
    action_specs = (
        ActionDimSpec("s_c_p", kind="log10", alpha=1.0, s_min=0.1, s_max=10.0),   # 约覆盖 ~10x
        ActionDimSpec("s_c_i", kind="sigmoid", alpha=1.0, s_min=0.0, s_max=1.0),  # 可逼近0
        ActionDimSpec("s_c_d", kind="log10", alpha=1.0, s_min=0.1, s_max=10.0),  # 约覆盖 ~10x
    )

    def sampler(rng: np.random.Generator) -> EpisodeParams:
        # 下面的采样范围你可以按你的系统改；这里给一个合理默认模板
        duration = float(duration_default)
        dt = 0.5  # 默认步长占位，你应在 train_modes.py 中统一传/覆盖

        rho_sp = float(rng.uniform(1300.0, 2200.0))  # kg/m3 示例
        rho0 = float(rng.uniform(1000.0, 1100.0))
        h_sp = 1.0
        h0 = float(rng.uniform(0.6, 1.2))  # 预混液位附近

        tau_mix = float(rng.uniform(5.0, 100.0))
        tau_delay = float(rng.uniform(0.0, 20.0))

        hold_s = float(hold_default)

        water_valve_max_flow = float(rng.uniform(1.5/60, 2.5/60))
        cement_valve_max_flow = float(rng.uniform(1.0/60, 1.5/60))


        return EpisodeParams(
            mode="premix",
            duration_s=duration,
            dt=dt,
            h0=h0, rho0=rho0, h_sp=h_sp, rho_sp=rho_sp,
            tau_mix=tau_mix, tau_delay=tau_delay,
            qs=0.0,
            water_valve_max_flow=water_valve_max_flow,
            cement_valve_max_flow=cement_valve_max_flow,
            premix_hold_s=hold_s,
        )

    def compute_base_params(p: EpisodeParams) -> Dict[str, float]:
        plant = PlantParams()
        if hasattr(plant, "tau_mix_hat"):
            plant.tau_mix_hat = float(p.tau_mix)

        cement_valve = ValveParams()

        base = compute_base_premix(
            p=p,
            cement_valve_params_template=cement_valve,
            plant_template=plant,
            sim_template=None,
        )

        return {
            "Kp_c": float(base.rho_kp),
            "Ki_c": float(base.rho_ki),
            "Kd_c": float(base.rho_kd),
            "ff_c": 0.0,
            "Kp_w": 0.0, "Ki_w": 0.0, "Kd_w": 0.0,
            "ff_w": 0.0,
            "kff": 0.0,
        }

    return ModeSpec(
        name="premix",
        action_specs=action_specs,
        sample_episode=sampler,
        build_context=build_context_premix,
        compute_base_params=compute_base_params,   # NEW
    )



def make_production_mode(
    *,
    duration_default: float = 600.0,
) -> ModeSpec:
    """
    production：学习水阀PID(3) + 灰阀PID(3) 。
    """
    action_specs = (
        # water PID
        ActionDimSpec("s_w_p", kind="log10", alpha=1.0, s_min=0.1, s_max=10.0),
        ActionDimSpec("s_w_i", kind="sigmoid", alpha=5.0, s_min=0.0, s_max= 1.0),
        ActionDimSpec("s_w_d", kind="log10", alpha=1.0, s_min=0.1, s_max=10.0),
        # cement PID
        ActionDimSpec("s_c_p", kind="log10", alpha=1.0, s_min=0.1, s_max=10.0),
        ActionDimSpec("s_c_i", kind="sigmoid", alpha=5.0, s_min=0.0, s_max= 1.0),
        ActionDimSpec("s_c_d", kind="log10", alpha=1.0, s_min=0.1, s_max=10.0),
    )

    def sampler(rng: np.random.Generator) -> EpisodeParams:
        duration = float(duration_default)
        dt = 0.5  # 占位，你应在 train_modes.py 中统一传/覆盖

        # 目标与初始（示例范围）
        h_sp = float(rng.uniform(0.8, 1.2))
        rho_sp = float(rng.uniform(1300.0, 2200.0))
        h0 = float(h_sp + rng.uniform(-0.1, 0.1))
        rho0 = float(rho_sp + rng.uniform(-50.0, 50.0))

        tau_mix = float(rng.uniform(5.0, 100.0))
        tau_delay = float(rng.uniform(0.0, 20.0))

        # 排量
        qs = float(rng.uniform(0.3/60, 1.5/60))
        water_valve_max_flow = float(rng.uniform(1.5/60, 2.5/60))
        cement_valve_max_flow = float(rng.uniform(0.8/60, 1.6/60))

        return EpisodeParams(
            mode="production",
            duration_s=duration,
            dt=dt,
            h0=h0, rho0=rho0, h_sp=h_sp, rho_sp=rho_sp,
            tau_mix=tau_mix, tau_delay=tau_delay, qs=qs,
            water_valve_max_flow=water_valve_max_flow,
            cement_valve_max_flow=cement_valve_max_flow,
            premix_hold_s=10.0,
        )

    def compute_base_params(p: EpisodeParams) -> Dict[str, float]:
        plant = PlantParams()
        if hasattr(plant, "tau_mix_hat"):
            plant.tau_mix_hat = float(p.tau_mix)

        water_valve = ValveParams()
        cement_valve = ValveParams()

        if not hasattr(p, "_cached_base_production"):
            p._cached_base_production = compute_base_production(p=p,
                                                                water_valve_params_template=water_valve,
                                                                cement_valve_params_template=cement_valve,
                                                                plant_template=plant,
                                                                sim_template=None,
                                                                )
        base = p._cached_base_production

        return {
            "Kp_w": float(base.h_kp),
            "Ki_w": float(base.h_ki),
            "Kd_w": float(base.h_kd),
            "ff_w": float(base.uw_ff),

            "Kp_c": float(base.rho_kp),
            "Ki_c": float(base.rho_ki),
            "Kd_c": float(base.rho_kd),
            "ff_c": float(base.uc_ff),

            "kff": float(base.kff),
        }

    return ModeSpec(
        name="production",
        action_specs=action_specs,
        sample_episode=sampler,
        build_context=build_context_production,
        compute_base_params=compute_base_params,
    )

