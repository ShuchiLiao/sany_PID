"""
sim_model.py

固井混浆系统的物理模型层
====================

本文件包含：
1. ValveActuator 类：模拟阀门的控制，包括死区、滞后和流量变化。
2. SlurryState 类：保存仿真过程中的所有状态变量。
3. SlurryTankModel 类：实现固井混浆系统的物理模型（液位、密度、质量等）。
4. RK4Integrator 类：实现 Runge-Kutta 4 数值积分，计算每步的状态变化。

所有计算和状态更新遵循 sim_config.py 中定义的配置。
"""

from __future__ import annotations
from typing import Dict
from dataclasses import dataclass
import numpy as np
import copy
from sim_config import PlantParams, ValveParams, PhysicalConstraintError, MassBalanceError, NumericalError


# ============================================================================ #
# 一、ValveActuator: 模拟阀门控制
# ============================================================================ #

class ValveActuator:
    """
    模拟阀门控制（Valve control with deadzone, rate-limiting, and lag）

    说明：
    ----
    - 模拟阀门执行器的控制；
    - 阀门的控制包括：滞后时间常数、一阶线性流量特性、死区和速率限幅。

    输入与输出：
    ----
    - 输入：控制指令（目标开度），更新后的流量。
    - 输出：当前实际的阀门开度与流量。
    """

    def __init__(self, valve_params: ValveParams) -> None:
        """
        初始化阀门执行器。

        参数：
        ----
        valve_params : ValveParams
            阀门的参数，包括开度范围、死区、流量特性等。
        """
        self.params = valve_params
        self.current_opening = valve_params.initial_valve_opening  # 初始开度
        self.target_opening = valve_params.initial_valve_opening  # 当前目标开度
        self.last_opening = valve_params.initial_valve_opening  # 上一时刻的开度
        self.current_flow = 0.0  # 初始流量
        # ---- flow noise state (OU/AR1) ----
        self._rng = np.random.default_rng(self.params.flow_noise_seed)
        self._flow_noise_state = 0.0
        self.current_flow_raw = 0.0  # 记录“抖动前”的基准流量，便于诊断

    def set_command(self, opening_cmd: float) -> None:
        """
        设置阀门目标开度。

        参数：
        ----
        opening_cmd : float
            目标开度 [%]，可能超过死区范围。
        """
        self.target_opening = opening_cmd

    def update(self, dt: float) -> None:
        """
        更新阀门的实际开度和流量。

        参数：
        ----
        dt : float
            当前仿真时间步长 [s]。
        """
        # 处理滞后：一阶低通滤波（使用时间常数）

        tau = float(self.params.actuator_time_constant)
        if tau <= 1e-9:
            self.current_opening = self.target_opening  # 直接到位
        else:
            lag_factor = np.exp(-dt / tau)
            self.current_opening =  lag_factor * self.current_opening + (1 - lag_factor) * self.target_opening

        # 计算流量（假设线性关系）
        if self.params.linear_slope is None:
            slope = self.params.max_flow / (self.params.opening_max - self.params.dead_zone_opening)
        else:
            slope = self.params.linear_slope

        # 限制开度变化速率
        delta_opening = self.current_opening - self.last_opening
        max_delta = self.params.max_opening_rate * dt
        if abs(delta_opening) > max_delta:
            self.current_opening = self.last_opening + np.sign(delta_opening) * max_delta

        # 计算流量（base）
        if self.current_opening <= self.params.dead_zone_opening:
            base_flow = 0.0
        else:
            base_flow = slope * (self.current_opening - self.params.dead_zone_opening)
            base_flow = float(np.clip(base_flow, self.params.min_flow, self.params.max_flow))

        self.current_flow_raw = base_flow  # 记录抖动前

        # 叠加“流量抖动”（进料量波动）
        Q = base_flow
        p = self.params
        if getattr(p, "flow_noise_enable", False) and getattr(p, "flow_noise_std", 0.0) > 0.0 and base_flow > 0.0:
            tau_n = float(getattr(p, "flow_noise_tau", 0.0))
            std_n = float(p.flow_noise_std)

            if tau_n is not None and tau_n > 1e-9:
                # OU/AR(1): x_{k+1} = a x_k + b N(0,1), stationary std = std_n
                a = float(np.exp(-dt / tau_n))
                b = float(std_n * np.sqrt(max(1.0 - a * a, 0.0)))
                self._flow_noise_state = a * self._flow_noise_state + b * float(self._rng.standard_normal())
                eps = self._flow_noise_state
            else:
                # 白噪声（不推荐，但保留开关）
                eps = std_n * float(self._rng.standard_normal())

            mode = getattr(p, "flow_noise_mode", "mul")
            if mode == "add":
                Q = base_flow + eps
            else:
                # 默认乘性扰动：Q = Q_base * (1 + eps)
                Q = base_flow * (1.0 + eps)

        # 物理限幅 + 非负
        Q = float(np.clip(Q, p.min_flow, p.max_flow))
        self.current_flow = max(0.0, Q)

        self.last_opening = self.current_opening

    def get_flow(self) -> float:
        """
        返回当前流量 [m^3/s]。
        """
        return self.current_flow

    def get_opening(self) -> float:
        """
        返回当前实际阀门开度 [%]。
        """
        return self.current_opening
    def compute_opening_from_flow(self, Q_target: float) -> float:
        """
        根据目标流量 Q_target 估算对应的阀门开度。
        使用与 ValveActuator 相同的线性关系进行近似反算。
        """

        if self.params.linear_slope is None:
            slope = self.params.max_flow / (self.params.opening_max - self.params.dead_zone_opening)
        else:
            slope = self.params.linear_slope

        # 简单线性反算：Q = slope * (opening - dead_zone_opening)
        opening = self.params.dead_zone_opening + Q_target / slope

        # 剪裁到物理开度范围
        opening = max(self.params.opening_min, min(self.params.opening_max, opening))
        return opening

    def compute_flow_from_opening(self, opening: float) -> float:
        """
        根据给定开度 opening [%] 估算对应的流量 Q [m^3/s]（静态阀特性）。
        注意：
        - 这里是“静态映射”，不包含执行器滞后/速率限制（那些在 update() 里体现）。
        - 死区、线性斜率、物理流量限幅与 update() 完全一致，保证正反算一致性。
        """
        # 计算线性斜率（与 update() 规则一致）
        if self.params.linear_slope is None:
            slope = self.params.max_flow / (self.params.opening_max - self.params.dead_zone_opening)
        else:
            slope = self.params.linear_slope

        # 开度先剪裁到物理范围（与 update() 的 opening clip 一致）
        opening = float(np.clip(opening, self.params.opening_min, self.params.opening_max))

        # 死区处理：死区内视为无流量
        if opening <= self.params.dead_zone_opening:
            Q = 0.0
        else:
            Q = slope * (opening - self.params.dead_zone_opening)
            Q = float(np.clip(Q, self.params.min_flow, self.params.max_flow))

        return Q


# ============================================================================ #
# 二、SlurryState: 仿真状态类
# ============================================================================ #

@dataclass
class SlurryState:
    """
    仿真状态类（Simulation state class）

    说明：
    ----
    该类用于存储仿真过程中罐体的各类状态信息，如液位、密度、质量等。
    """

    h: float = 0.0  # 液位 [m]
    rho_out: float = 0.0 # 出口密度 [kg/m^3]
    x: float = 0.0  # 固相质量分数 [-]
    M_w: float = 0.0 # 水相质量 [kg]
    M_c: float = 0.0  # 固相质量 [kg]
    M: float = 0.0  # 总质量 [kg]

    def initial_x(self, plant_params: PlantParams) -> None:
        self.x = (1.0/self.rho_out - 1.0/plant_params.rho_water)/(1.0/plant_params.rho_cement-1.0/plant_params.rho_water)
        self.compute_states(plant_params)

    def update(self,plant_params: PlantParams, h: float, x:float, compute_states:bool, check_states:bool) -> None:
        self.h = h
        self.x = x  # 固相质量分数

        #更新密度
        if compute_states:
            self.compute_states(plant_params) # 出口密度
        if check_states:
            if h <= 0 :
                raise PhysicalConstraintError(f"Invalid liquid level: h={h}. Liquid level must be greater than 0.")
            if x < 0 or x > 1:
                raise PhysicalConstraintError(f"Invalid mass ratio: x={x}.")
            if self.rho_out < plant_params.rho_water:
                raise PhysicalConstraintError(
                    f"Invalid outlet density: rho_out={self.rho_out}. Outlet density must be greater than water density.")
            if self.rho_out > plant_params.rho_cement:
                raise PhysicalConstraintError(
                    f"Invalid outlet density: rho_out={self.rho_out}. Outlet density must be smaller than cement density.")
    def compute_states(self, plant_params: PlantParams) :
        self.rho_out = 1 / (self.x / plant_params.rho_cement + (1 - self.x) / plant_params.rho_water)
        self.M = self.rho_out * self.h * plant_params.tank_cross_section_area
        self.M_w = self.M * (1 - self.x)
        self.M_c = self.M * self.x





# 计算出口密度


# ============================================================================ #
# 三、SlurryTankModel: 物理模型类
# ============================================================================ #

class SlurryTankModel:
    """
    固井混浆系统物理模型（Cementing slurry model）

    说明：
    ----
    本类实现固井混浆系统的动力学模型，主要计算液位变化、质量守恒、混合区密度变化等。
    """

    def __init__(self, plant_params: PlantParams) -> None:
        """
        初始化物理模型。

        参数：
        ----
        plant_params : PlantParams
            系统的物理参数（如容积、密度等）。
        """
        self.params = plant_params

    def compute_derivatives(self, state: SlurryState, Q_w: float, Q_c: float, Q_s: float) :
        """
        计算系统的状态导数（液位、质量、密度的变化）。

        参数：
        ----
        state : SlurryState
            当前状态（液位、密度、质量等）。
        Q_w : float
            水流量 [m^3/s]。
        Q_c : float
            灰流量 [m^3/s]。
        Q_s : float
            出流量 [m^3/s]。

        返回：
        ----
        derivatives : dict
            状态导数字典，包含液位变化率、质量变化率等。
        """

        # 液位变化率
        A = self.params.tank_cross_section_area
        rho_w = self.params.rho_water
        rho_c = self.params.rho_cement

        d_h = (Q_w + Q_c - Q_s) / A  # 液位变化率 [m/s]
        if state.M <= 0:
            raise PhysicalConstraintError( f"h={state.h}. 起始值h不能小于等于0， 否则state.M<0；可以给h一个很小的正数值")
        d_x = ((1-state.x)*rho_c*Q_c - state.x*rho_w*Q_w)/state.M # h不能等于0， 否则state.M为0；可以给h一个很小的数值


        # print( "state.h: ", state.h, "state.x", state.x, "Q_s", Q_s)
        # print("d_h", d_h, "d_x", d_x)
        return (d_h, d_x)

    def check_dt_stability(self, dt: float, state: SlurryState, Q_w: float, Q_c: float) -> None:
        """
        基于 x 动力学线性化推导的时间步长稳定性检查。

        推导要点：
        --------
        在一小步内近似认为 M 不变，则
            dx/dt = (ρ_c Q_c - x(ρ_c Q_c + ρ_w Q_w)) / M ≈ a - b x
        其中 b = (ρ_c Q_c + ρ_w Q_w) / M >= 0

        对齐次部分 dx/dt = -b x，显式方法的稳定性条件近似为：
            dt < 2 / b = 2 M / (ρ_c Q_c + ρ_w Q_w)

        实现细节：
        --------
        - 当总质量 M 很小（如空罐阶段），该约束意义有限，直接跳过；
        - 当 (ρ_c Q_c + ρ_w Q_w) ≈ 0（几乎无进料）时，也跳过。
        """
        rho_w = self.params.rho_water
        rho_c = self.params.rho_cement

        # 当前总质量（由状态代数计算）
        M = state.M  # 等价于 state.M_w + state.M_c

        # 质量太小：多为起步阶段，跳过该约束
        M_min = getattr(self.params, "min_effective_mass", 1e-3)
        if M <= M_min:
            return

        # 线性化系数 b 对应的分母
        denom = rho_c * Q_c + rho_w * Q_w

        # 几乎没有进料，刚性约束很弱，跳过
        if abs(denom) < 1e-6:
            return

        # 由 b 推导的临界步长（对 Euler 稍保守，对 RK4 更保守）
        critical_dt = 2.0 * M / denom  # denom > 0 时成立

        if dt > critical_dt:
            raise NumericalError(
                f"时间步长过大，可能导致 x(t) 数值不稳定：dt={dt:.3e}, 临界值={critical_dt:.3e}, "
                f"M={M:.4f}, Q_w={Q_w:.4e}, Q_c={Q_c:.4e}, denom={denom:.4e}"
            )


    def check_mass_balance(
        self,
        dt: float,
        prev_state: SlurryState,
        curr_state: SlurryState,
        Q_w: float,
        Q_c: float,
        Q_s: float,
    ) -> float:
        """
        检查单步质量守恒（总质量），返回相对误差指标。

        连续质量守恒方程：
        ----------------
            dM/dt = ρ_c Q_c + ρ_w Q_w - ρ_out(t) Q_s

        在 [t_k, t_{k+1}] 上，质量变化为：
            M(t_{k+1}) - M(t_k)
            = ∫ [ρ_c Q_c + ρ_w Q_w - ρ_out(t) Q_s] dt

        假定 Q_w, Q_c, Q_s 在步长内恒定，ρ_out(t) 随 x(t) 平滑变化，
        用梯形公式近似出流项积分：

            ΔM_theory = dt * (ρ_c Q_c + ρ_w Q_w)
                       - dt * Q_s * (ρ_out_prev + ρ_out_curr) / 2

        数值积分后的质量变化为：
            ΔM_num = M_curr - M_prev

        则质量守恒误差为：
            e_M = ΔM_num - ΔM_theory

        相对误差：
            rel_err_M = |e_M| / max(|ΔM_num|, |ΔM_theory|, M_ref)

        当 rel_err_M 长期小于 mass_balance_tol（例如 1e-3~1e-2），
        可认为数值上无明显 mass loss；若突然变大，则需排查。
        """
        rho_w = self.params.rho_water
        rho_c = self.params.rho_cement

        # 1) 步前/步后总质量
        M_prev = prev_state.M  # = prev_state.M_w + prev_state.M_c
        M_curr = curr_state.M  # = curr_state.M_w + curr_state.M_c

        # 2) 步前/步后出口密度
        rho_out_prev = prev_state.rho_out
        rho_out_curr = curr_state.rho_out

        # 3) 理论质量变化（梯形近似）
        m_in = rho_c * Q_c + rho_w * Q_w
        # 出流项用步前/步后 ρ_out 的均值
        rho_out_avg = 0.5 * (rho_out_prev + rho_out_curr)

        dM_theory = dt * m_in - dt * Q_s * rho_out_avg

        # 4) 数值质量变化（由状态重建）
        dM_num = M_curr - M_prev

        # 5) 质量平衡残差与相对误差
        e_M = dM_num - dM_theory
        # denom = max(abs(dM_num), abs(dM_theory), 1e-6)
        # rel_err_M = abs(e_M) / denom
        #
        # if rel_err_M > self.params.mass_balance_tol:
        #     raise MassBalanceError(
        #         f"在仿真时刻 {prev_state.h=}, 质量守恒误差过大: "
        #         f"rel_err={rel_err_M:.3e}, "
        #         f"M_prev={M_prev:.3f}, M_curr={M_curr:.3f}, "
        #         f"dM_num={dM_num:.3f}, dM_theory={dM_theory:.3f}, "
        #         f"Q_w={Q_w:.3e}, Q_c={Q_c:.3e}, Q_s={Q_s:.3e}, dt={dt:.3e}"
        #     )

        return e_M

# ============================================================================ #
# 四、RK4Integrator: Runge-Kutta 4 积分器
# ============================================================================ #

class RK4Integrator:
    """
    Runge-Kutta 4 积分器（RK4 integration）

    说明：
    ----
    该类用于仿真中使用 RK4 方法对状态进行数值积分，更新状态变量。
    """

    def __init__(self, dt: float) -> None:
        """
        初始化 RK4 积分器。

        参数：
        ----
        dt : float
            仿真时间步长 [s]。
        """
        self.dt = dt

    def step(self, model: SlurryTankModel, state: SlurryState, Q_w: float, Q_c: float, Q_s: float) -> SlurryState:
        """
        执行一步 RK4 积分，更新系统状态。

        参数：
        ----
        model : SlurryTankModel
            物理模型实例，需要提供 compute_derivatives(state, Q_w, Q_c)，
            返回包含 'd_h', 'd_x'的字典。
        state : SlurryState
            当前状态（液位、各质量、固相质量分数等）。
        Q_w : float
            水流量 [m^3/s]。
        Q_c : float
            灰流量 [m^3/s]。
        Q_s : float
            出流量 [m^3/s]。
        返回：
        ----
        next_state : SlurryState
            更新后的新状态（已通过 SlurryState.update 计算 rho_out）。
        """
        dt = self.dt
        p = model.params  # PlantParams，用于调用 SlurryState.update

        # 小工具：从 base_state 和一组导数 k，构造中间状态（用于 k2,k3,k4 的计算）
        def make_state(base: SlurryState, k, alpha: float) -> SlurryState:
            h   = base.h   + alpha * k[0]
            x   = base.x   + alpha * k[1]
            s_mid = copy.copy(base)
            s_mid.update(p, h, x, True, False)
            return s_mid

        # k1
        k1 = model.compute_derivatives(state, Q_w, Q_c, Q_s)
        # print("state", state, "k1", k1)


        # k2：在 state + dt/2 * k1 处求导
        s2 = make_state(state, k1, dt * 0.5)
        k2 = model.compute_derivatives(s2, Q_w, Q_c, Q_s)
        # print("s2", s2, "k2", k2)

        # k3：在 state + dt/2 * k2 处求导
        s3 = make_state(state, k2, dt * 0.5)
        k3 = model.compute_derivatives(s3, Q_w, Q_c, Q_s)
        # print("s3", s3, "k3", k3)

        # k4：在 state + dt * k3 处求导
        s4 = make_state(state, k3, dt)
        k4 = model.compute_derivatives(s4, Q_w, Q_c, Q_s)
        # print("s4", s4, "k4", k4)

        # 按 RK4 权重合成新状态
        h_new   = state.h   + dt / 6.0 * (k1[0]   + 2.0 * k2[0]   + 2.0 * k3[0]   + k4[0])
        x_new   = state.x   + dt / 6.0 * (k1[1]   + 2.0 * k2[1]   + 2.0 * k3[1]   + k4[1])


        # 用 SlurryState.update 生成 next_state，并同步更新 rho_out
        next_state = SlurryState()
        next_state.update(p, h_new, x_new, True, True)
        # print("next_state", next_state)

        return next_state
