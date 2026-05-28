"""
PIDcontroller.py

通用 PID 控制器类（位置式 / Positional PID）
==========================================

- 可用于液位/密度等回路；
- **直接输出绝对控制量 u_k**（例如：绝对阀门开度 %），而不是“增量/修正量”。

在标准位置式 PID 基础上加入：
1) 微分滤波（Derivative filtering）与可选 D-on-measurement
2) 抗积分饱和（Anti-windup）：clamp 或 back-calculation

----------------------------------------------------------------------
为什么要“位置式”？
- 位置式 PID 直接给出当前控制输出 u_k（更符合“阀门开度命令”直出）。
- 如果你原先在环境里有 feedforward（例如 base opening），可以通过 step() 的 u_ff 参数叠加。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import math


AntiWindupMode = Literal["none", "clamp", "backcalc"]


@dataclass
class PIDController:
    """
    位置式 PID（Positional PID）

    连续域理想并联 PID：
        u(t) = u_ff + Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt

    这里采用离散实现（采样周期 dt）：
        - P:  Kp * e_k
        - I:  I_k = I_{k-1} + Ki * e_k * dt
        - D:  对误差或测量做差分求导，并可做一阶低通滤波
        - u:  u_k = u_ff + P + I + D
        - 输出饱和：u_sat = clip(u_k, u_min, u_max)

    注意：
    - derivative_on_measurement=True 且提供 measurement 时，D 项使用：
          D = -Kd * d(measurement)/dt
      这样可避免 setpoint 阶跃导致的 derivative kick（微分冲击）。
    - derivative_filter_tau(Tf) 用于给 D 通道加一阶低通滤波，抑制噪声/量化/锯齿放大。

    参数
    ----
    kp, ki, kd : float
        并联形式 PID 的 Kp, Ki, Kd（单位取决于你的 e 与 u 的单位）。
        例如：u 为阀门开度(%)，e 为液位(m)，则 kp 单位为 %/m。

    derivative_filter_tau : float
        微分滤波时间常数 Tf [s]。Tf<=0 表示不滤波（直接用差分导数）。

    derivative_on_measurement : bool
        True 时，若 step() 提供 measurement，则 D 项用 -d(measurement)/dt（D on measurement）。

    anti_windup : "none" | "clamp" | "backcalc"
        抗积分饱和策略：
        - none: 不做 anti-windup（积分可能 windup）
        - clamp: 若输出饱和且误差推动更饱和，则冻结本步积分
        - backcalc: 反算回馈：I += (dt/Tt)*(u_sat - u_unsat)

    aw_tau : float
        backcalc 的回馈时间常数 Tt [s]，越小纠正越强（建议 0.2~2s 起试）。
    """

    kp: float
    ki: float
    kd: float

    # ===== derivative filtering & anti-windup config =====
    derivative_filter_tau: float = 0.0   # Tf [s]
    derivative_on_measurement: bool = False
    anti_windup: AntiWindupMode = "none"
    aw_tau: float = 0.5                  # Tt [s] for back-calculation

    # ===== internal states =====
    _initialized: bool = False
    _prev_error: float = 0.0

    # 积分器状态（注意：这里存的是“积分项对输出的贡献”，单位与 u 相同，例如 %）
    _i_out: float = 0.0

    # D 通道状态：用于滤波与差分
    _prev_d_input: float = 0.0   # 上一步用于求导的信号（error 或 measurement）
    _d_f: float = 0.0            # 滤波后的导数估计（单位：error/s 或 measurement/s）

    def reset(self) -> None:
        """重置 PID 内部状态。"""
        self._initialized = False
        self._prev_error = 0.0
        self._i_out = 0.0
        self._prev_d_input = 0.0
        self._d_f = 0.0

    @staticmethod
    def _clip(x: float, lo: Optional[float], hi: Optional[float]) -> float:
        if lo is not None:
            x = max(lo, x)
        if hi is not None:
            x = min(hi, x)
        return x

    def step(
        self,
        error: float,
        dt: float,
        *, #* 在 Python 函数参数里表示：从这里往后所有参数都必须用关键字传参（keyword-only arguments），
            # 不能再用位置参数。 这段定义等价于：位置参数只能有：error, dt,其余的 measurement, u_ff, u_min, u_max 必须写成 name=value 的形式传入
        measurement: Optional[float] = None,
        u_ff: float = 0.0,
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
    ) -> float:
        """
        计算并返回位置式 PID 的输出 u_k（绝对控制量，例如阀门开度 %）。

        参数
        ----
        error : float
            e_k = setpoint - measurement
        dt : float
            采样周期 [s]
        measurement : Optional[float]
            若 derivative_on_measurement=True，则建议传入测量值，用于 D on measurement。
        u_ff : float
            前馈/基准输出（例如 base opening）。默认 0。
            最终输出：u = u_ff + P + I + D
        u_min, u_max : Optional[float]
            输出饱和上下限（例如阀门开度 0~100）。
            若提供且 anti_windup != "none"，则用于 anti-windup 判断与回馈。
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive in PIDController.step().")

        # -------------------------
        # 初始化：只建立“上一拍”状态，避免第一步导数乱跳
        # -------------------------
        if not self._initialized:
            self._initialized = True
            self._prev_error = float(error)

            # D 通道的“求导输入”：
            # - 默认对 error 求导：d_input = error
            # - D on measurement：对 measurement 求导，但 D 项要加负号（见后文）
            if self.derivative_on_measurement and (measurement is not None):
                self._prev_d_input = float(measurement)
            else:
                self._prev_d_input = float(error)

            self._d_f = 0.0
            # 初始输出：仅前馈 + P + I（I 初值 0），D 初值 0
            u0 = u_ff + self.kp * error + self._i_out
            return self._clip(float(u0), u_min, u_max)

        # =========================
        # 1) P 项（位置式：直接用 e_k）
        # =========================
        p_out = self.kp * float(error)

        # print("self.kp", self.kp, "p_out", p_out)
        # =========================
        # 2) D 项（可选 D on measurement + 可选滤波）
        # =========================
        if self.derivative_on_measurement and (measurement is not None):
            # D on measurement：对测量求导
            d_input = float(measurement)
            d_sign = -1.0  # 关键：D = -Kd * dy/dt，避免 setpoint 阶跃带来 D kick
        else:
            # D on error：对误差求导
            d_input = float(error)
            d_sign = 1.0

        # 原始差分导数
        d_raw = (d_input - self._prev_d_input) / dt

        # 一阶低通滤波（指数平滑）
        if self.derivative_filter_tau and self.derivative_filter_tau > 0.0:
            Tf = max(1e-9, float(self.derivative_filter_tau))
            alpha = math.exp(-dt / Tf)  # alpha 越接近 1，滤波越强（更平滑更慢）
            d_f_new = alpha * self._d_f + (1.0 - alpha) * d_raw
        else:
            d_f_new = d_raw

        d_out = d_sign * self.kd * d_f_new
        # print("self.kd", self.kd, "d_out", d_out)

        # =========================
        # 3) I 项（先计算候选积分；必要时再做 anti-windup 修正）
        # =========================
        i_candidate = self._i_out + self.ki * float(error) * dt

        # 先按“未饱和”输出组合（用于 anti-windup 判断/回馈）
        u_unsat = float(u_ff) + p_out + i_candidate + d_out
        u_sat = self._clip(u_unsat, u_min, u_max)

        # =========================
        # 4) Anti-windup（只有在提供了限幅时才有意义）
        # =========================
        if (u_min is not None) or (u_max is not None):
            if self.anti_windup == "clamp":
                # 若输出已饱和，且误差仍“推动输出继续朝饱和方向走”，则冻结积分
                pushing_upper = (u_max is not None) and (u_unsat > u_max) and (error > 0.0)
                pushing_lower = (u_min is not None) and (u_unsat < u_min) and (error < 0.0)
                if pushing_upper or pushing_lower:
                    i_candidate = self._i_out  # 冻结：本步不积分
                    # 重新组合输出
                    u_unsat = float(u_ff) + p_out + i_candidate + d_out
                    u_sat = self._clip(u_unsat, u_min, u_max)

            elif self.anti_windup == "backcalc":
                # 反算回馈：用饱和前后差值把积分器“拉回可实现范围”
                Tt = max(1e-6, float(self.aw_tau))
                i_candidate = i_candidate + (dt / Tt) * (u_sat - u_unsat)
                # 回馈后再算一次输出（更一致）
                u_unsat2 = float(u_ff) + p_out + i_candidate + d_out
                u_sat = self._clip(u_unsat2, u_min, u_max)

            # "none"：不做处理
        # print("self.ki", self.ki, "i_candidate", i_candidate, "u_unsat", u_unsat, "u_sat", u_sat)
        # =========================
        # 5) 提交状态
        # =========================
        self._prev_error = float(error)
        self._prev_d_input = float(d_input)
        self._d_f = float(d_f_new)
        self._i_out = float(i_candidate)

        return float(u_sat)
