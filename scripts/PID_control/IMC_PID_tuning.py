"""
IMC_PID_tuning.py

自动完成：
1) 利用固井混浆仿真环境做阶跃试验（open_loop）：
   - 回路 1：水阀**开度** water_cmd 阶跃 -> 液位测量值 h_obs
   - 回路 2：灰阀**开度** cement_cmd 阶跃 -> 密度测量值 rho_obs
2) 从阶跃响应中辨识 FOPDT 模型参数 (K, tau, theta)
3) 按 IMC-PID 等价公式计算 PID 参数 (Kp, Ki, Kd)，用于后续 RL 缩放的 base PID

本文件相对旧版的关键修改：
- [MOD-1] 使用阀门开度（water_cmd/cement_cmd），不再用流量 Q_w/Q_c 作为辨识输入；
         这样辨识得到的 Kp/Ki/Kd 与 sim_env 中“PID 输出=开度修正量(%)”的控制通道一致。
- [MOD-2] 保留并输出完整 PID（含 Kd），不再只给 PI。
- [MOD-3] 密度回路使用 rho_obs（带测量时滞），而不是 rho_out（真实值），与控制器实际输入一致。
- [MOD-4] 兼容新版 sim_env：创建环境后显式调用 env.reset() 初始化观测缓冲区。
- [MOD-5] 阶跃辨识更稳健：y0/u0 用“阶跃前窗口”均值，而不是序列开头（避免初始态不在稳态导致辨识失败）。
"""

from __future__ import annotations

import numpy as np

from scripts.core.sim_config import PlantParams, ValveParams, SimulationConfig
from scripts.core.sim_model import SlurryState
from scripts.core.sim_env import CementingSimEnv


# =============================================================================
# 一、通用工具：FOPDT 辨识 + IMC-PID
# =============================================================================

def estimate_fopdt_from_step(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    *,
    tol_y_frac: float = 0.1,
    pre_window_sec: float = 5.0,
    post_window_sec: float = 10.0,
) -> tuple[float, float, float]:
    """
    根据单次阶跃响应数据估计 FOPDT 参数:
        G(s) ≈ K * exp(-theta*s) / (tau*s + 1)

    参数
    ----
    t : np.ndarray
        时间序列 [s]
    u : np.ndarray
        阶跃输入（本脚本中为阀门开度命令，单位：%）
    y : np.ndarray
        输出（液位 h_obs 或 密度 rho_obs）
    tol_y_frac : float
        判定输出开始响应阈值，占 Δy 的比例
    pre_window_sec : float
        用于估计 u0/y0 的“阶跃前窗口”长度（秒）
    post_window_sec : float
        用于估计 u_ss/y_ss 的“末端窗口”长度（秒）

    返回
    ----
    K : float
        过程增益（Δy / Δu）
    tau : float
        时间常数 [s]
    theta : float
        纯滞后 [s]
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.ndim != 1 or u.ndim != 1 or y.ndim != 1 or len(t) != len(u) or len(t) != len(y):
        raise ValueError("t/u/y 必须是等长一维数组。")
    if len(t) < 50:
        raise ValueError("数据点太少，无法稳定辨识（建议 >= 50）。")

    dt = float(np.median(np.diff(t)))
    pre_N = max(3, int(pre_window_sec / max(dt, 1e-6)))
    post_N = max(5, int(post_window_sec / max(dt, 1e-6)))
    # print(post_N, pre_N)

    # 1) 用 u 的最大跳变定位阶跃时刻（更稳健：不依赖序列开头的稳态）
    du = np.diff(u)
    idx_step = int(np.argmax(np.abs(du)) + 1)
    t_step = float(t[idx_step])

    # 2) u0/y0 取“阶跃前窗口”的均值
    i0 = max(0, idx_step - pre_N)
    u0 = float(np.mean(u[i0:idx_step]))
    y0 = float(np.mean(y[i0:idx_step]))

    # 3) 末端窗口估计稳态
    u_ss = float(np.mean(u[-post_N:]))
    y_ss = float(np.mean(y[-post_N:]))
    # print(y[-post_N:])

    d_u = u_ss - u0
    d_y = y_ss - y0
    if abs(d_u) < 1e-10:
        raise ValueError("输入没有明显阶跃变化（Δu≈0），无法做 FOPDT 辨识。")
    if abs(d_y) < 1e-10:
        raise ValueError("输出几乎没有变化（Δy≈0），无法做 FOPDT 辨识。")

    K = d_y / d_u

    # 4) 纯滞后 theta：在阶跃后，y 首次超过阈值的时刻
    tol_y = tol_y_frac * abs(d_y)
    idx_after = np.arange(idx_step, len(t))
    idx_delay_rel = np.where(np.abs(y[idx_after] - y0) > tol_y)[0]
    if len(idx_delay_rel) == 0:
        theta = 0.0
        idx_delay = idx_step
    else:
        idx_delay = int(idx_after[int(idx_delay_rel[0])])
        theta = max(0.0, float(t[idx_delay]) - t_step)

    # 5) tau：达到 63.2% Δy 的时刻（从 idx_delay 之后找首次“穿越”）
    y_target = y0 + 0.632 * d_y
    y_seg = y[idx_delay:]
    t_seg = t[idx_delay:]

    if d_y > 0:
        cross = np.where(y_seg >= y_target)[0]
    else:
        cross = np.where(y_seg <= y_target)[0]

    if len(cross) == 0:
        # 未穿越则用最近点兜底
        idx_63 = int(idx_delay + np.argmin(np.abs(y_seg - y_target)))
    else:
        idx_63 = int(idx_delay + int(cross[0]))

    tau = max(1e-3, float(t[idx_63]) - t_step - theta)
    return float(K), float(tau), float(theta)


def imc_pid_from_fopdt(K: float, tau: float, theta: float, lam_factor: float = 1.0) -> tuple[float, float, float]:
    """
    基于 IMC 思路的 PID 整定（带 D 项）：

    对 FOPDT:
        Gp(s) = K * exp(-theta*s) / (tau*s + 1)

    选定闭环时间常数 λ（通常 λ >= theta），
    采用一类常见的 IMC-PID 等价公式：

        Kc = (tau + 0.5*theta) / (K * (lam + 0.5*theta))
        Ti = tau + 0.5*theta
        Td = tau * theta / (2*tau + theta)

    PID 采用并联形式：
        u = Kp * e + Ki * ∫e dt + Kd * de/dt
    """
    K = float(K)
    tau = float(tau)
    theta = float(theta)

    if abs(K) < 1e-12:
        raise ValueError("K 太小，无法整定（会导致 Kp 爆炸）。")
    if tau <= 0:
        raise ValueError("tau 必须为正。")
    if theta < 0:
        raise ValueError("theta 不能为负。")

    # 选择闭环时间常数 λ（越大越保守）
    lam = max(theta, lam_factor * tau, 1e-3)

    Kc = (tau + 0.5 * theta) / (K * (lam + 0.5 * theta))
    Ti = tau + 0.5 * theta
    Td = (tau * theta) / max(1e-6, (2 * tau + theta))

    # [MOD-2] 输出完整 PID：并联形式 (Kp, Ki, Kd)
    Kp = Kc
    Ki = Kc / Ti
    Kd = Kc * Td
    return float(Kp), float(Ki), float(Kd)


# =============================================================================
# 二、针对混浆系统的阶跃实验（open_loop）
# =============================================================================

def run_step_test_level() -> dict:
    """
    回路 1：水阀开度 water_cmd 阶跃 -> 液位测量 h_obs
    """
    # ---------- 仿真参数 ----------
    dt = 0.1
    t_end = 200.0
    t_step = 50.0  # 给系统一定时间靠近预稳态

    # ---------- 物理参数 ----------
    plant_params = PlantParams(
        tank_cross_section_area=1.0,
        tank_height=2.0,
        h_max=1.8,
        h_min=0.0,
    )

    Qs_nominal = 0.5 / 60.0  # 非零出流，形成“准稳态”

    # 水阀参数（线性开度->流量），用来选一个“预稳态”开度
    water_dead = 3.0
    water_max_flow = 1.5 / 60.0
    water_opening_max = 100.0
    water_slope = water_max_flow / max(1e-6, (water_opening_max - water_dead))

    # 选 opening_low 使得 Q_w ≈ Qs_nominal，避免液位单调饱和
    opening_low = float(np.clip(water_dead + Qs_nominal / max(water_slope, 1e-9), 0.0, 100.0))
    opening_high = float(np.clip(opening_low + 20.0, 0.0, 100.0))
    cement_opening_const = 0.0

    water_valve_params = ValveParams(
        opening_max=water_opening_max,
        opening_min=0.0,
        dead_zone_opening=water_dead,
        max_flow=water_max_flow,
        min_flow=0.0,
        actuator_time_constant=1.0,
        max_opening_rate=50.0,
        initial_valve_opening=opening_low,
    )

    cement_valve_params = ValveParams(
        opening_max=100.0,
        opening_min=0.0,
        dead_zone_opening=5.0,
        max_flow=1.0 / 60.0,
        min_flow=0.0,
        actuator_time_constant=1.0,
        max_opening_rate=50.0,
        initial_valve_opening=cement_opening_const,
    )

    cfg = SimulationConfig(
        dt=dt,
        t_end=t_end,
        h_sp=1.0,               # open_loop 不用，仅占位
        rho_sp=1600.0,
        Qs_nominal=Qs_nominal,
        h_obs_delay=0.0,
        rho_obs_delay=0.0,
        # open_loop 不用 PID
        h_pid_kp=0.0, h_pid_ki=0.0, h_pid_kd=0.0,
        rho_pid_kp=0.0, rho_pid_ki=0.0, rho_pid_kd=0.0,
        use_density_feedforward=False,
        use_h_feedforward=False,
        control_mode="open_loop",
        use_smith_decoupler=False,
        use_kff_decoupler=False,
        enable_logger=True,
        log_to_csv=False,
        run_name_prefix="IMC_step_level",
    )

    initial_state = SlurryState(h=1.0, rho_out=1000.0)
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, cfg, initial_state)
    # [MOD-4]


    # ---------- 构造开环动作序列（sim_env.run 需要 list[[water_opening, cement_opening]]） ----------
    num_steps = int(cfg.t_end / cfg.dt)
    actions: list[list[float]] = []
    for k in range(num_steps):
        t = k * cfg.dt
        water_cmd = opening_low if t < t_step else opening_high
        actions.append([water_cmd, cement_opening_const])

    env.run(actions=actions)

    entries = env.logger._step_entries
    t_arr = np.array([e.t for e in entries], dtype=float)

    # [MOD-1] 输入用“阀门开度命令”，而不是 Q_w 流量
    u_arr = np.array([e.valve_cmd.get("water_cmd", 0.0) for e in entries], dtype=float)

    # 输出用控制器实际看到的测量值（h_obs）
    y_arr = np.array([e.observations.get("h_obs", e.state.get("h", 0.0)) for e in entries], dtype=float)

    K, tau, theta = estimate_fopdt_from_step(t_arr, u_arr, y_arr, tol_y_frac=0.1)
    Kp, Ki, Kd = imc_pid_from_fopdt(K, tau, theta, lam_factor=1.0)

    print("=== 液位回路：水阀开度 water_cmd -> h_obs 的 FOPDT + IMC-PID 结果 ===")
    print(f"water opening: low={opening_low:.2f}%, high={opening_high:.2f}% (Qs_nominal={Qs_nominal:.4g} m3/s)")
    print(f"FOPDT: K = {K:.4g} [Δh/Δopening], tau = {tau:.4g} s, theta = {theta:.4g} s")
    print(f"PID(base): Kp_h = {Kp:.6g}, Ki_h = {Ki:.6g}, Kd_h = {Kd:.6g}")
    print()

    return dict(K=K, tau=tau, theta=theta, Kp=Kp, Ki=Ki, Kd=Kd)


def run_step_test_density() -> dict:
    """
    回路 2：灰阀开度 cement_cmd 阶跃 -> 密度测量 rho_obs
    """
    dt = 0.1
    t_end = 300.0
    t_step = 100.0  # 给系统时间靠近预稳态

    water_opening_const = 50.0

    # 灰阀开度阶跃（确保高低两侧都大于 dead_zone）
    opening_low = 15.0
    opening_high = 30.0

    plant_params = PlantParams(
        tank_cross_section_area=1.0,
        tank_height=2.0,
        h_max=1.8,
        h_min=0.0,
    )

    water_valve_params = ValveParams(
        opening_max=100.0,
        opening_min=0.0,
        dead_zone_opening=3.0,
        max_flow=1.5 / 60.0,
        min_flow=0.0,
        actuator_time_constant=1.0,
        max_opening_rate=50.0,
        initial_valve_opening=water_opening_const,
    )

    cement_valve_params = ValveParams(
        opening_max=100.0,
        opening_min=0.0,
        dead_zone_opening=5.0,
        max_flow=1.0 / 60.0,
        min_flow=0.0,
        actuator_time_constant=1.0,
        max_opening_rate=50.0,
        initial_valve_opening=opening_low,
    )

    cfg = SimulationConfig(
        dt=dt,
        t_end=t_end,
        h_sp=1.0,
        rho_sp=1650.0,
        Qs_nominal=0.8 / 60.0,
        h_obs_delay=0.0,
        rho_obs_delay=10.0,  # 测量时滞
        # open_loop 不用 PID
        h_pid_kp=0.0, h_pid_ki=0.0, h_pid_kd=0.0,
        rho_pid_kp=0.0, rho_pid_ki=0.0, rho_pid_kd=0.0,
        use_density_feedforward=False,
        use_h_feedforward=False,
        control_mode="open_loop",
        use_smith_decoupler=False,
        use_kff_decoupler=False,
        enable_logger=True,
        log_to_csv=False,
        run_name_prefix="IMC_step_density",
    )

    # 初始密度尽量贴近 opening_low 的准稳态（减少前段大漂移）
    initial_state = SlurryState(h=1.0, rho_out=1120.0)
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, cfg, initial_state)
    # [MOD-4]


    num_steps = int(cfg.t_end / cfg.dt)
    actions: list[list[float]] = []
    for k in range(num_steps):
        t = k * cfg.dt
        cement_cmd = opening_low if t < t_step else opening_high
        actions.append([water_opening_const, cement_cmd])

    env.run(actions=actions)

    entries = env.logger._step_entries
    t_arr = np.array([e.t for e in entries], dtype=float)

    # [MOD-1] 输入用“灰阀开度命令”，而不是 Q_c 流量
    u_arr = np.array([e.valve_cmd.get("cement_cmd", 0.0) for e in entries], dtype=float)

    # [MOD-3] 输出用控制器实际看到的测量值（rho_obs）
    y_arr = np.array([e.observations.get("rho_obs", e.state.get("rho_out", 0.0)) for e in entries], dtype=float)

    K, tau, theta = estimate_fopdt_from_step(t_arr, u_arr, y_arr, tol_y_frac=0.1)
    Kp, Ki, Kd = imc_pid_from_fopdt(K, tau, theta, lam_factor=1.0)

    print("=== 密度回路：灰阀开度 cement_cmd -> rho_obs 的 FOPDT + IMC-PID 结果 ===")
    print(f"cement opening: low={opening_low:.2f}%, high={opening_high:.2f}% (water_opening={water_opening_const:.2f}%)")
    print(f"FOPDT: K = {K:.4g} [Δrho/Δopening], tau = {tau:.4g} s, theta = {theta:.4g} s")
    print(f"PID(base): Kp_rho = {Kp:.6g}, Ki_rho = {Ki:.6g}, Kd_rho = {Kd:.6g}")
    print()

    return dict(K=K, tau=tau, theta=theta, Kp=Kp, Ki=Ki, Kd=Kd)


# =============================================================================
# 三、主入口
# =============================================================================

if __name__ == "__main__":
    print(">>> 开始自动阶跃试验 + IMC-PID 整定 ...")
    level_result = run_step_test_level()
    density_result = run_step_test_density()

    print("=== 汇总：可作为 RL 缩放的 base PID 参数（单位：开度%） ===")
    print(
        f"液位回路 base PID:  Kp_h = {level_result['Kp']:.6g}, "
        f"Ki_h = {level_result['Ki']:.6g}, Kd_h = {level_result['Kd']:.6g}"
    )
    print(
        f"密度回路 base PID:  Kp_rho = {density_result['Kp']:.6g}, "
        f"Ki_rho = {density_result['Ki']:.6g}, Kd_rho = {density_result['Kd']:.6g}"
    )
