"""
tune_baseline_v3.py

粗整定：生产阶段（Qs: 0 -> Qs_step）双回路 PID + 前馈
====================================================

目标：
- 密度回路优先（rho_sp 附近尽量稳），液位回路次要（允许在安全范围内波动）。
- 不依赖阶跃实验数据；只用“设备常数 + 当前工况”粗算一组参数。
- 输出：
  - 水阀：uw_ff, kp_h, ki_h, kd_h
  - 灰阀：uc_ff, kp_rho, ki_rho, kd_rho
  - 前馈耦合系数：kff （用于“水阀增量 -> 灰阀增量”的快速补偿）

关键思路（写在注释里，便于对照数学推导）：
1) 利用目标密度 rho_sp 推“理论水灰配比” r：
      r = Qc/Qw = (rho_sp - rho_w) / (rho_c - rho_sp)
   则在稳态且忽略压缩/漏失时，满足：
      Qw* + Qc* = Qs_step
      Qc* = r Qw*
   得：
      Qw* = Qs_step / (1 + r)
      Qc* = r * Qw*

2) 阀门开度-流量用 sim_config.ValveParams 里的线性段近似：
      Q = slope * (opening - dead_zone_opening) + offset
   反求：
      opening = dead_zone_opening + (Q - offset) / slope

3) 密度回路粗整定（把“灰阀->密度”近似成 FOPDT，用于 IMC 形式的粗 Kp）：
   - 能预估的：观测延迟 theta ≈ rho_obs_delay，混合常数 tau ≈ tau_mix_hat。
   - 过程增益用局部线性化（围绕 rho_sp, Qs_step）：
        rho ≈ (rho_w Qw + rho_c Qc) / (Qw + Qc)
        且 Qw+Qc ≈ Qs_step（生产阶段以排量为主）
     在 Qw 随水阀扰动时，用前馈让 Qc ≈ r Qw，则 rho 的一阶变化主要来自“前馈不准+延迟”。
     为了给 PID 一个量级，我们取“单独调灰阀”时的局部增益：
        d rho / d Qc |_{rho_sp} ≈ (rho_c - rho_sp) / Qs_step
     而 dQc/duc ≈ slope_c（m^3/s per %），所以
        K_rho = d rho / d uc ≈ (rho_c - rho_sp)/Qs_step * slope_c   [kg/m^3 per %]

   - IMC 粗参数（用于“延迟大 + 想稳”的保守整定）：
        lambda_rho = theta + 0.5*tau
        Kp_rho = (tau / (K_rho * (lambda_rho + theta)))
     积分/微分给很小：
        Ti_rho = beta_i * (theta + tau)
        Ki_rho = Kp_rho / Ti_rho
        Kd_rho = kd_ratio * Kp_rho * tau

4) 液位回路粗整定（把“水阀->液位”看作积分对象）：
      dh/dt = (Qw - Qs_step)/A
   因为我们做了前馈 Qw*≈Qs_step/(1+r)，起始时积分对象对“误差”主要是小偏差。
   线性化增益：
      K_h = d(dh/dt)/d uw ≈ (dQw/duw)/A = slope_w / A   [m/s per %]
   选更大的 lambda_h（更慢更软），降低液位对密度的扰动：
      lambda_h = lambda_ratio_h * (theta + tau)
      Kp_h = 1/(K_h * lambda_h)
      Ti_h = beta_i * lambda_h
      Ki_h = Kp_h / Ti_h
      Kd_h = kd_ratio * Kp_h * lambda_h

注意：
- 这只是“能跑起来”的粗量级，真实设备差异靠后续 RL/自适应再细化。
- 如果你当前验证发现 Ki 仍容易引发超调：把 beta_i 调大（让 Ti 更长）或直接把 Ki 缩放到更小。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 复用你上传的参数类（PlantParams / ValveParams / SimulationConfig）
from scripts.core.sim_config import PlantParams, ValveParams, SimulationConfig


# ----------------------------
# 工具函数：阀门线性段 slope
# ----------------------------

def _ensure_linear_slope(v: ValveParams) -> float:
    """返回阀门线性段斜率 slope [m^3/s per %]，若未提供则按 max_flow 推导。"""
    if v.linear_slope is not None:
        return float(v.linear_slope)
    denom = float(v.opening_max - v.dead_zone_opening)
    if denom <= 0:
        raise ValueError("Invalid valve params: opening_max must be > dead_zone_opening.")
    return float(v.max_flow) / denom


def flow_to_opening(Q: float, v: ValveParams) -> float:
    """将期望流量 Q [m^3/s] 转为阀门开度 [%]（按线性段反求，并裁剪到 [opening_min, opening_max]）。"""
    slope = _ensure_linear_slope(v)
    if slope <= 0:
        # 极端情况下（无有效斜率）只能全关
        return float(v.opening_min)

    opening = float(v.dead_zone_opening) + (float(Q) - float(v.linear_offset)) / slope
    # 裁剪
    opening = max(float(v.opening_min), min(float(v.opening_max), opening))
    return opening


def opening_to_flow(opening: float, v: ValveParams) -> float:
    """将阀门开度 [%] 转为近似流量 Q [m^3/s]（线性段 + 死区裁剪）。"""
    op = float(opening)
    if op < float(v.dead_zone_opening):
        return 0.0
    slope = _ensure_linear_slope(v)
    Q = slope * (op - float(v.dead_zone_opening)) + float(v.linear_offset)
    return Q


# ----------------------------
# 粗整定函数（核心）
# ----------------------------

@dataclass
class BaselineTuningResult:
    """粗整定输出（所有开度都是 [%]）。"""

    # 前馈（稳态）
    uw_ff: float  # 水阀前馈开度 [%]
    uc_ff: float  # 灰阀前馈开度 [%]

    # 水阀 PID（液位）
    h_kp: float
    h_ki: float
    h_kd: float

    # 灰阀 PID（密度）
    rho_kp: float
    rho_ki: float
    rho_kd: float

    # “水阀增量 -> 灰阀增量”耦合前馈系数（开度域）
    kff: float

    # 额外信息（便于你打印/排查）
    Qw_star: float
    Qc_star: float
    ratio_r: float



@dataclass
class EngineeringKnobs:
    """工程保守度“旋钮”（可选）。

    这些参数不是 IMC/DS 等经典整定公式“必需”的一部分，而是为了在真实设备上：
    - 抑制 Ki 造成的超调/振荡（特别是混合慢 + 测量延迟大时）
    - 给 D 项一个很小的“阻尼”以对冲延迟导致的相位滞后
    - 让液位回路比密度回路更慢，减少对配比/密度的扰动

    若 `apply_engineering_knobs=False`，则粗整定会退回到更“教科书”的 IMC/DS 风格默认值，
    不再使用这些额外缩放。
    """

    # 液位回路闭环时间常数放大倍数（>1 更慢）
    lambda_ratio_h: float = 3.0

    # 积分时间放大倍数（>1 => Ti 更长 => Ki 更小）
    beta_i: float = 8.0

    # D 项比例：Kd = kd_ratio * Kp * (tau 或 lambda_h)
    kd_ratio: float = 0.02

    # 额外缩小 Ki（在 beta_i 基础上再乘一个 <1 的缩放）
    ki_scale_h: float = 0.2
    ki_scale_rho: float = 0.2



# ============================================================
# Premix base tuning (Qs = 0): conservative density-loop base
# ============================================================
#
# 预混阶段（通常：先加水到目标液位 h_sp，然后关水阀，开灰阀把密度拉到 rho_sp），
# 不存在“Qs_step>0 的稳态配比工作点”，因此生产阶段那套 (FOPDT@Qs_step) 的量级
# 不再合适。
#
# 这里采用一个保守但可部署的近似：
#   - 罐内体积 V = A*h （取 h 为预混开始时的液位，通常可用 h_sp）
#   - 近似“均匀混合”的密度插值：rho ≈ rho_w + x (rho_c - rho_w)
#   - 固相体积分数 x 的变化由进灰体积流量决定：dx/dt ≈ Qc / V
#   - 灰阀线性段：Qc ≈ slope_c * u_c
#
# 得到：rho 对灰阀开度的“积分型”增益（影响的是 rho 上升速率）：
#   d(rho_dot)/d u_c ≈ (rho_c - rho_w) * slope_c / (A*h)
#
# 因此将对象近似为 “积分对象 + 等效延迟”：
#   G_premix(s) = Δrho(s) / Δu_c(s) ≈ (K_I * e^{-θ s}) / s
#   K_I = (rho_c - rho_w) * slope_c / (A*h)
#
# 用 IMC/λ 风格的保守 PI(D) 量级：
#   Kp = 1 / (K_I * (λ + θ))
#   Ki = Kp / Ti ,  Ti = β_i * λ          (β_i>1 使积分更慢更稳)
#   Kd = kd_ratio * Kp * (λ + θ)          (必须非零：给延迟对象一点阻尼，但保持很小)
#
# 备注：
# - 这是“无阶跃实验”的粗 base；真实设备差异交给后续 RL 缩放或再辨识。
# - Kd 在预混阶段非常容易放大噪声/量化误差，因此强烈建议 kd_ratio 很小（如 0.005~0.02）
#   并在控制器内部对微分项做滤波/或使用 D-on-measurement。
#


@dataclass
class PremixTuningResult:
    """预混阶段（Qs=0）密度回路 base（灰阀）粗整定输出（开度域，单位 [%]）。"""

    # 建议的前馈开度
    uc_ff: float

    # 灰阀密度 PID
    rho_kp: float
    rho_ki: float
    rho_kd: float  # 注意：要求 kd 不能为 0

    # 诊断信息
    K_I: float          # 积分增益 [kg/m^3 per (%·s)]
    theta_eff: float    # 等效延迟 [s]
    lambda_rho: float   # 设计闭环时间常数 [s]
    h_used: float       # 用于计算的液位 [m]


@dataclass
class PremixKnobs:
    """预混阶段的保守度旋钮（无阶跃实验时建议用更保守的默认值）。"""

    # θ_eff = rho_obs_delay + theta_mix_ratio * tau_mix_hat
    theta_mix_ratio: float = 0.8

    # λ = lambda_gamma * θ_eff（建议 5~10）
    lambda_gamma: float = 7.0

    # Ti = beta_i * λ（>1 使积分更慢）
    beta_i: float = 12.0

    # D 项：Kd = Kp * (alpha * theta_eff)
    alpha: float = 0.5

    # 额外缩小 Ki（在 beta_i 基础上再乘一个 <1 的缩放）
    ki_scale_rho: float = 0.2

    # 防止 h 太小导致增益爆炸
    h_min: float = 0.3

    # 保险：Kd 最小值（确保“非零”）
    kd_min: float = 1e-9


def tune_premix_base_params(
    *,
    plant: PlantParams,
    sim: SimulationConfig,
    cement_valve_params: ValveParams,
    h_level: float,
    apply_premix_knobs: bool = False,
    knobs: Optional[PremixKnobs] = None,
) -> PremixTuningResult:
    """不做阶跃实验，直接用“设备常数 + 预混液位 h_level”计算预混阶段密度回路（灰阀）base。

    参数：
    - h_level: 预混开始时的液位（推荐用 h_sp 或当前液位 h0）[m]
    """

    rho_w = float(plant.rho_water)
    rho_c = float(plant.rho_cement)
    A = float(plant.tank_cross_section_area)
    if A <= 0:
        raise ValueError("tank_cross_section_area must be > 0")

    # 预混的目标密度仍然来自 sim.rho_sp（用于 sanity check）
    rho_sp = float(sim.rho_sp)
    if not (rho_w < rho_sp < rho_c):
        raise ValueError(
            f"rho_sp must be between rho_water and rho_cement, got rho_sp={rho_sp}."
        )

    # knobs
    if knobs is None:
        knobs = PremixKnobs()

    # 等效延迟：观测延迟 + 混合常数的一部分
    theta_meas = float(sim.rho_obs_delay)
    tau_mix = float(getattr(plant, "tau_mix_hat", 0.0))
    theta_eff = max(0.0, theta_meas + knobs.theta_mix_ratio * max(0.0, tau_mix))

    # 体积 V = A*h
    h_used = max(float(h_level), float(knobs.h_min))
    V = A * h_used

    # 阀门线性段 slope_c（m^3/s per %）
    slope_c = _ensure_linear_slope(cement_valve_params)
    if slope_c <= 1e-12:
        # 退化：无法从开度产生有效流量
        return PremixTuningResult(
            uc_ff=0.0,
            rho_kp=0.0,
            rho_ki=0.0,
            rho_kd=float(knobs.kd_min),
            K_I=0.0,
            theta_eff=theta_eff,
            lambda_rho=0.0,
            h_used=h_used,
        )

    # 积分增益：K_I = d(rho_dot)/d u_c
    # 单位检查：
    #   (rho_c-rho_w)[kg/m^3] * slope_c[m^3/s/%] / V[m^3] => kg/m^3/s/%
    K_I = (rho_c - rho_w) * slope_c / max(V, 1e-12)

    # 设计闭环时间常数 λ（更保守）
    if apply_premix_knobs:
        lambda_rho = max(1e-6, knobs.lambda_gamma * max(theta_eff, 1e-3))
        Ti = max(1e-6, knobs.beta_i * lambda_rho)
        kp = 1.0 / (K_I * (lambda_rho + max(theta_eff, 0.0)))
        ki = (kp / Ti) * knobs.ki_scale_rho

        # 必须非零的 Kd：给一点阻尼，但保持很小
        kd = max(float(knobs.kd_min), kp * (knobs.alpha * max(theta_eff, 0.0)))

    else:
        # 若不启用旋钮：给一个仍然偏保守的默认
        lambda_rho = max(1e-6, (theta_eff + max(tau_mix, 0.0)))
        Ti = max(1e-6, lambda_rho)
        kp = 1.0 / (K_I * (lambda_rho + max(theta_eff, 0.0))) * 5
        ki = kp / Ti * 1e-3
        kd = max(float(knobs.kd_min), kp * (0.5 * max(theta_eff, 0.0)))


    # 预混默认前馈开度：从 0 开始（
    # 如有经验基线，可替换为一个小值）
    uc_ff = 0.0

    return PremixTuningResult(
        uc_ff=uc_ff,
        rho_kp=kp,
        rho_ki=ki,
        rho_kd=kd,
        K_I=K_I,
        theta_eff=theta_eff,
        lambda_rho=lambda_rho,
        h_used=h_used,
    )



def tune_baseline_params(
    *,
    plant: PlantParams,
    sim: SimulationConfig,
    water_valve_params: ValveParams,
    cement_valve_params: ValveParams,
    Qs_step: float,
    # 是否启用“工程保守度旋钮”
    apply_engineering_knobs: bool = False,
    knobs: Optional[EngineeringKnobs] = None,
) -> BaselineTuningResult:
    """输入当前工况与设备常数，输出“生产阶段”粗整定的一组前馈 + PID 参数。"""

    # --- 0) 读取工况（你说的：能预估的只有 delay & tau_mix；能观测的只有 setpoints 与 Qs_step） ---
    theta = float(sim.rho_obs_delay)  # 观测延迟 [s]
    tau = float(plant.tau_mix_hat)  # 混合时间常数 [s]

    # 旋钮：若启用但未显式传入，则用默认值
    if apply_engineering_knobs and knobs is None:
        knobs = EngineeringKnobs()


    A = float(plant.tank_cross_section_area)
    if A <= 0:
        raise ValueError("tank_cross_section_area must be > 0")

    rho_w = float(plant.rho_water)
    rho_c = float(plant.rho_cement)
    rho_sp = float(sim.rho_sp)

    if not (rho_w < rho_sp < rho_c):
        raise ValueError(
            f"rho_sp must be between rho_water and rho_cement, got rho_sp={rho_sp}."
        )

    Qs = float(Qs_step)
    if Qs <= 1e-12:
        raise ValueError("Qs_step must be > 0 for production-step tuning")

    # --- 1) 理论配比 r 与稳态流量 (Qw*, Qc*) ---
    ratio_r = (rho_sp - rho_w) / (rho_c - rho_sp)  # r = Qc/Qw
    Qw_star = Qs / (1.0 + ratio_r)
    Qc_star = ratio_r * Qw_star

    # --- 2) 前馈开度（稳态） ---
    uw_ff = flow_to_opening(Qw_star, water_valve_params)
    uc_ff = flow_to_opening(Qc_star, cement_valve_params)

    # --- 3) 开度域“水->灰”耦合前馈系数 kff ---
    # 目标：当水阀因液位扰动产生 Δuw 时，灰阀同步给 Δuc ≈ kff * Δuw，
    #      使得 ΔQc ≈ r * ΔQw，从而尽量保持配比。
    slope_w = _ensure_linear_slope(water_valve_params)
    slope_c = _ensure_linear_slope(cement_valve_params)
    if slope_c <= 1e-12:
        kff = 0.0
    else:
        kff = ratio_r * (slope_w / slope_c)

    # --- 4) 密度回路：FOPDT + IMC 粗整定（可选工程旋钮） ---
    # K_rho: [kg/m^3 per %]
    K_rho = (rho_c - rho_sp) / Qs * slope_c
    if abs(K_rho) <= 1e-12:
        # 极端退化：不给 PID 过大
        rho_kp = 0.0
        rho_ki = 0.0
        rho_kd = 0.0
    else:
        if apply_engineering_knobs:
            # 工程版（带旋钮）：更保守，尤其是把 Ki 压小，减少超调/振荡
            assert knobs is not None
            # 过滤因子：延迟越大，越保守（这里采用更激进的 theta+0.5*tau 作为经验）
            lambda_rho = theta + 0.5 * tau
            # IMC-PI 量级：Kc ~ tau / (K*(lambda+theta))
            rho_kp = tau / (K_rho * (lambda_rho + theta))

            Ti_rho = knobs.beta_i * (theta + tau)
            rho_ki = (rho_kp / max(Ti_rho, 1e-12)) * knobs.ki_scale_rho
            rho_kd = knobs.kd_ratio * rho_kp * tau
        else:
            # 学术/教科书版（不带旋钮）：IMC-PID 思路
            # 仅保留一个“设计自由度”lambda_rho，这里用更保守的 (theta + tau) 作为默认
            lambda_rho = theta + tau

            # 常见 IMC-PID（基于一阶+纯滞后近似）的量级关系：
            #   Kc ~ (tau + 0.5*theta) / (K*(lambda + 0.5*theta))
            rho_kp = (tau + 0.5 * theta) / (K_rho * (lambda_rho + 0.5 * theta)) * 2

            Ti_rho = tau + 0.5 * theta
            rho_ki = rho_kp / max(Ti_rho, 1e-12)

            # D 项常用：Td = tau*theta/(2*tau + theta)，theta=0 时退化为 0
            if theta <= 1e-12:
                rho_kd = 0.0
            else:
                Td_rho = (tau * theta) / (2.0 * tau + theta)
                rho_kd = rho_kp * Td_rho * 2

    # --- 5) 液位回路：积分对象（dh/dt = K_h*u） + 慢回路整定（可选工程旋钮） ---
    # K_h: [m/s per %]
    K_h = slope_w / A
    if abs(K_h) <= 1e-12:
        h_kp = 0.0
        h_ki = 0.0
        h_kd = 0.0
    else:
        if apply_engineering_knobs:
            # 工程版：故意让液位回路更慢，避免扰动配比/密度
            assert knobs is not None
            lambda_h = knobs.lambda_ratio_h * (theta + tau)
            h_kp = 1.0 / (K_h * lambda_h)

            Ti_h = knobs.beta_i * lambda_h
            h_ki = (h_kp / max(Ti_h, 1e-12)) * knobs.ki_scale_h
            h_kd = max(1e-9, knobs.kd_ratio * h_kp * lambda_h)
        else:
            # 学术/教科书版：积分对象常用的 IMC/DS 量级，默认选 lambda_h = theta + tau
            lambda_h = theta + tau
            h_kp = 1.0 / (K_h * lambda_h)

            Ti_h = lambda_h
            h_ki = h_kp / max(Ti_h, 1e-12)
            h_kd = max(1e-9, 0.01 * h_kp * lambda_h)


    return BaselineTuningResult(
        uw_ff=uw_ff,
        uc_ff=uc_ff,
        h_kp=h_kp,
        h_ki=h_ki,
        h_kd=h_kd,
        rho_kp=rho_kp,
        rho_ki=rho_ki,
        rho_kd=rho_kd,
        kff=kff,
        Qw_star=Qw_star,
        Qc_star=Qc_star,
        ratio_r=ratio_r,
    )


# ----------------------------
# main：一键运行示例
# ----------------------------

def main() -> None:
    # 1) 读取默认配置（你也可以在这里按“某台设备”的经验 max_flow 改掉）
    plant = PlantParams(tau_mix_hat=56.64)
    sim = SimulationConfig(rho_obs_delay=18.7, rho_sp=2121.48)

    # 2) 两个阀门：你说每台设备都有经验估计的 Qw_max / Qc_max
    #    这里直接复用 ValveParams，并分别设置 max_flow。
    # water_valve = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
    #                                  # slope = max_flow / (opening_max - dead_zone_opening)
    #                                  dead_zone_opening=2, max_flow=1.5 / 60, min_flow=0,
    #                                  actuator_time_constant=2.0, max_opening_rate=50,
    #                                  flow_physical_max=1.5 / 60, flow_physical_min=0.0,
    #                                  initial_valve_opening=0.0)

    # cement_valve = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
    #                                   # slope = max_flow / (opening_max - dead_zone_opening)
    #                                   dead_zone_opening=5, max_flow=1.0 / 60, min_flow=0,
    #                                   actuator_time_constant=2.0, max_opening_rate=50,
    #                                   flow_physical_max=1.0 / 60, flow_physical_min=0.0,
    #                                   initial_valve_opening=0.0)

    water_valve = ValveParams()
    cement_valve = ValveParams()

    # 3) 生产阶段阶跃：0 -> 0.8 m^3/min
    Qs_step = 0.8 / 60

    # 默认：学术/教科书版（不启用旋钮）
    res = tune_baseline_params(
        plant=plant,
        sim=sim,
        water_valve_params=water_valve,
        cement_valve_params=cement_valve,
        Qs_step=Qs_step,
    )

    # 可选：启用工程旋钮（更保守，通常更不容易超调/振荡）
    res_eng = tune_baseline_params(
        plant=plant,
        sim=sim,
        water_valve_params=water_valve,
        cement_valve_params=cement_valve,
        Qs_step=Qs_step,
        apply_engineering_knobs=False,
        knobs=EngineeringKnobs(
            lambda_ratio_h=4.0,
            beta_i=10.0,
            kd_ratio=0.02,
            ki_scale_h=0.1,
            ki_scale_rho=0.2,
        ),
    )

    print("\n=== Baseline tuning (production step) ===")
    print(f"Target rho_sp = {sim.rho_sp:.1f} kg/m^3")
    print(f"Step Qs = {Qs_step:.6f} m^3/s ({Qs_step*60:.3f} m^3/min)")
    print(f"r = Qc/Qw = {res.ratio_r:.6f}")
    print(f"Qw* = {res.Qw_star:.6f} m^3/s,  Qc* = {res.Qc_star:.6f} m^3/s")
    print("\n-- Feedforward openings --")
    print(f"uw_ff = {res.uw_ff:.2f} %")
    print(f"uc_ff = {res.uc_ff:.2f} %")
    print(f"kff (Δuc = kff·Δuw) = {res.kff:.4f}  [-]")

    print("\n-- Water (level) PID --")
    print(f"kp_h  = {res.h_kp:.6g}")
    print(f"ki_h  = {res.h_ki:.6g}")
    print(f"kd_h  = {res.h_kd:.6g}")

    print("\n-- Cement (density) PID --")
    print(f"kp_rho = {res.rho_kp:.6g}")
    print(f"ki_rho = {res.rho_ki:.6g}")
    print(f"kd_rho = {res.rho_kd:.6g}")

    print("\n=== Baseline tuning (engineering knobs enabled) ===")
    print(f"r = Qc/Qw = {res_eng.ratio_r:.6f}")
    print(f"Qw* = {res_eng.Qw_star:.6f} m^3/s,  Qc* = {res_eng.Qc_star:.6f} m^3/s")
    print("\n-- Feedforward openings --")
    print(f"uw_ff = {res_eng.uw_ff:.2f} %")
    print(f"uc_ff = {res_eng.uc_ff:.2f} %")
    print(f"kff (Δuc = kff·Δuw) = {res_eng.kff:.4f}  [-]")

    print("\n-- Water (level) PID --")
    print(f"kp_h  = {res_eng.h_kp:.6g}")
    print(f"ki_h  = {res_eng.h_ki:.6g}")
    print(f"kd_h  = {res_eng.h_kd:.6g}")

    print("\n-- Cement (density) PID --")
    print(f"kp_rho = {res_eng.rho_kp:.6g}")
    print(f"ki_rho = {res_eng.rho_ki:.6g}")
    print(f"kd_rho = {res_eng.rho_kd:.6g}")

    # ----------------------------
    # Premix base (Qs = 0) example
    # ----------------------------
    # 预混通常先把液位加到 h_sp，然后关水阀开始加灰；这里用 h_level = h_sp 作为计算基准。
    h_level = float(getattr(sim, "h_sp", 1.0))
    premix = tune_premix_base_params(
        plant=plant,
        sim=sim,
        cement_valve_params=cement_valve,
        h_level=h_level,
        apply_premix_knobs=False,
        knobs=PremixKnobs(
            theta_mix_ratio=0.8,
            lambda_gamma=7.0,
            beta_i=12.0,
            alpha=0.5,          # 默认 α=0.5
            ki_scale_rho=0.2,
            h_min=0.3,
            kd_min=1e-9,
        ),
    )

    print("\n=== Premix base tuning (Qs = 0) ===")
    print(f"h_level used = {premix.h_used:.3f} m")
    print(f"theta_eff = {premix.theta_eff:.3f} s,  lambda = {premix.lambda_rho:.3f} s")
    print(f"K_I (d rho_dot / d uc) = {premix.K_I:.6g} (kg/m^3)/(s·%)")
    print("\n-- Cement (density) PID [premix base] --")
    print(f"uc_ff_premix = {premix.uc_ff:.2f} %")
    print(f"kp_rho_premix = {premix.rho_kp:.6g}")
    print(f"ki_rho_premix = {premix.rho_ki:.6g}")
    print(f"kd_rho_premix = {premix.rho_kd:.6g}   (non-zero by design)")




if __name__ == "__main__":
    main()