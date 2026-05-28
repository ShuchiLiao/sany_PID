"""
sim_env.py

仿真环境层与测试可视化层
====================

本文件包含：
1. `CementingSimEnv` 类：仿真环境层，封装了固井混浆系统仿真过程的各个环节，包括初始化、步进更新、状态计算等。
"""

# import logging
# from tqdm import tqdm
from scripts.core.sim_model import SlurryState, SlurryTankModel, RK4Integrator, ValveActuator
from scripts.core.sim_config import PlantParams, ValveParams, SimulationConfig, SimulationLogger, MassBalanceError, StepLogEntry
from scripts.PID_control.PIDcontroller import PIDController
from collections import deque


class CementingSimEnv:
    """
    Cementing Simulation Environment 类

    说明：
    ----
    本类实现固井混浆系统的仿真环境，管理仿真状态、控制动作、仿真步骤和日志记录。
    """

    def __init__(self, plant_params: PlantParams,
                 water_valve_params: ValveParams,cement_valve_params: ValveParams,
                 config: SimulationConfig,
                 initial_slurry_state: SlurryState) -> None:
        """
        初始化仿真环境。

        参数：
        ----
        plant_params : PlantParams
            系统物理参数（如容积、密度等）。
        valve_params : ValveParams
            阀门执行器参数（死区、滞后、速率限幅等）。
        config : SimulationConfig
            仿真配置（时间步长、控制模式等）。
        """
        self.plant_params = plant_params
        self.config = config

        # 初始化物理模型和控制执行器
        self.model = SlurryTankModel(plant_params=self.plant_params)
        self.water_valve = ValveActuator(valve_params=water_valve_params)
        self.cement_valve = ValveActuator(valve_params=cement_valve_params)


        self.total_mass_balance_error = 0.0

        # 初始化状态
        self.state = SlurryState()

        # RK4 积分器
        self.integrator = RK4Integrator(dt=self.config.dt)

        # 日志记录器
        self.logger = None
        if self.config.enable_logger:
            self.logger = SimulationLogger(
                run_dir=self.config.make_run_dir(),
                level=self.config.log_level,
            )

        # === 观测时滞相关（以步数表示） ==================================
        self.h_delay_steps = int(round(self.config.h_obs_delay / self.config.dt))
        self.rho_delay_steps = int(round(self.config.rho_obs_delay / self.config.dt))

        # 用简单 list 作为时序缓冲区
        self._h_buffer = deque(maxlen=self.h_delay_steps + 1)
        self._rho_buffer = deque(maxlen=self.rho_delay_steps + 1)
        self._last_h_obs: float = 0.0
        self._last_rho_obs: float = 0.0

        # === PID 控制器实例 ================================================
        # 液位 PID（SISO & MIMO 共用）
        self.h_pid = PIDController(
            kp=self.config.h_pid_kp,
            ki=self.config.h_pid_ki,
            kd=self.config.h_pid_kd,
            derivative_filter_tau=0.,
            derivative_on_measurement=False,
            anti_windup="clamp",
        )

        # 密度 PID（仅在 MIMO 使用）
        self.rho_pid = PIDController(
            kp=self.config.rho_pid_kp,
            ki=self.config.rho_pid_ki,
            kd=self.config.rho_pid_kd,
            derivative_filter_tau=0.,
            derivative_on_measurement=False,
            anti_windup="clamp",
        )

        self.state = initial_slurry_state
        self.state.initial_x(plant_params=self.plant_params)
        self.Q_s = self.config.Qs_nominal
        self.total_Q_s = 0.0

        # 前馈相关
        self.cement_opening_ff = self.config.cement_opening_ff
        self.water_opening_ff = self.config.water_opening_ff
        self.Q_w_bias = 0.0

        self.reset(initial_slurry_state)



    def reset(self, slurry_state: SlurryState):
        """
        重置仿真环境并返回初始状态。

        参数：
        ----
        slurry_state : SlurryState
            初始状态（例如：液位、固相质量分数等）。

        返回：
        ----
        SlurryState : 初始状态
        """
        self.state = slurry_state
        self.state.initial_x(plant_params=self.plant_params)

        # 重置 PID 内部状态
        if hasattr(self, "h_pid"):
            self.h_pid.reset()
        if hasattr(self, "rho_pid"):
            self.rho_pid.reset()

        # 重置观测时滞缓冲区
        # 可选：让初始缓冲区填满，避免前 delay_steps 步 rho_obs/h_obs 一直等于真实值或出现空
        self._h_buffer.clear()
        self._rho_buffer.clear()
        for _ in range(self.h_delay_steps + 1):
            self._h_buffer.append(self.state.h)
        for _ in range(self.rho_delay_steps + 1):
            self._rho_buffer.append(self.state.rho_out)

        self._last_h_obs = self.state.h
        self._last_rho_obs = self.state.rho_out

        self.total_mass_balance_error = 0.0

        self._init_feedforward_openings()
        self._init_SMITH_decoupler()

        if self.config.enable_logger:
            self.logger.log_info(f"仿真重置，初始状态：{self.state}; 控制模式： {self.config.control_mode}; "
                                 f"前置参数：water_opening_bias = {self.water_opening_bias}, "
                                 f"water_opening_ff= {self.water_opening_ff}, "
                                 f"cement_opening_ff= {self.cement_opening_ff}; "
                                 f"解耦参数： G_D1 = {self.GD1}, G_D2 = {self.GD2}.")


    # ------------------------------------------------------------------
    # 内部工具：初始化前馈开度（SISO 偏置 + MIMO 前馈）
    # ------------------------------------------------------------------
    # sim_env.py  (class CementingSimEnv 内新增)
    def _valve_slope(self, valve) -> float:
        """
        返回阀门开度-流量线性段斜率 slope [m^3/s per %]
        与 ValveActuator 内部一致（linear_slope None 则用 max_flow 推导）。
        """
        p = valve.params
        if p.linear_slope is None:
            return p.max_flow / (p.opening_max - p.dead_zone_opening)
        return p.linear_slope

    def _init_feedforward_openings(self):
        """
        初始化：
        - SISO 液位 流量前馈 Q_w_bias；
        - MIMO 模式下的水阀/灰阀前馈基准开度 water_opening_ff / cement_opening_ff。
        """
        # 1) SISO 液位平衡偏置（假设仅水阀补偿出流 self.Q_s，灰阀为 0）

        if self.config.use_h_feedforward:
            self.Q_w_bias = self.Q_s
        else:
            self.Q_w_bias = 0.0

        self.water_opening_bias = self.water_valve.compute_opening_from_flow(self.Q_w_bias)

        # 2) MIMO 模式：基于目标密度的前馈水灰配比
        if self.config.use_density_feedforward:
            if self.config.cement_opening_ff==0 and self.config.water_opening_ff==0:
                rho_sp_ff = self.config.rho_sp
                rho_w = self.plant_params.rho_water
                rho_c = self.plant_params.rho_cement

                # 简单稳态配比：rho = rho_w + x_v (rho_c - rho_w) => x_v = (rho_sp - rho_w)/(rho_c - rho_w)
                x_v = (rho_sp_ff - rho_w) / (rho_c - rho_w) # 固相体积分数

                # 限制在 [0, 1]
                x_v = max(0.0, min(1.0, x_v))

                Q_total = self.Q_s
                self.Q_c_ff = x_v * Q_total
                self.Q_w_ff = (1.0 - x_v) * Q_total
                self.water_opening_ff = self.water_valve.compute_opening_from_flow(self.Q_w_ff)
                self.cement_opening_ff = self.cement_valve.compute_opening_from_flow(self.Q_c_ff)
            else:
                self.water_opening_ff = self.config.water_opening_ff
                self.cement_opening_ff = self.config.cement_opening_ff
                self.Q_c_ff = self.cement_valve.compute_flow_from_opening(self.cement_opening_ff)
                self.Q_w_ff = self.water_valve.compute_flow_from_opening(self.water_opening_ff)
        else:
            # 不使用密度前馈：
            self.Q_w_ff = 0.0
            self.Q_c_ff = 0.0
            self.water_opening_ff = 0.0
            self.cement_opening_ff = 0.0

        # === 新增：把前馈流量换算成前馈开度（PID 将在开度域工作） ===



    def _init_SMITH_decoupler(self) -> None:
        """
        基于 Fan 式 G(s) 分析的 MIMO 解耦初始化。
        步骤：
        1) 用密度设定值 rho_sp 作为工作点密度，反算平衡固相质量分数 x0；
        2) 用解析公式计算静态解耦增益 GD1, GD2；
        """
        rho_w = self.plant_params.rho_water
        rho_c = self.plant_params.rho_cement
        rho_sp = self.config.rho_sp              # 目标密度当作工作点密度

        # 1) 由 rho_sp 反算工作点固相质量分数 x0
        #    1/rho_sp = x0/rho_c + (1-x0)/rho_w
        x0 = (1.0 / rho_sp - 1.0 / rho_w) / (1.0 / rho_c - 1.0 / rho_w)
        eps = 1e-6
        x0 = max(eps, min(1.0 - eps, x0))
        self.x0_working = x0

        # 4) Fan 式静态解耦增益：
        #    GD2 = -G12/G11 = -1
        #    GD1 = -G21/G22 = x0*rho_w / [rho_c*(1-x0)]
        self.GD2 = -1.0
        self.GD1 = (x0 * rho_w) / (rho_c * (1.0 - x0))

    # ------------------------------------------------------------------
    # 内部工具：观测时滞缓冲与读取
    # ------------------------------------------------------------------
    def _get_observations(self) -> tuple[float, float]:
        """
        deque 的 maxlen 固定为 delay_steps+1：
        - 右端 [-1] 永远是“当前”
        - 左端 [0] 永远是“delay_steps 步之前”
        """
        if self.h_delay_steps <= 0:
            h_obs = self._h_buffer[-1]
        else:
            h_obs = self._h_buffer[0]

        if self.rho_delay_steps <= 0:
            rho_obs = self._rho_buffer[-1]
        else:
            rho_obs = self._rho_buffer[0]

        self._last_h_obs = h_obs
        self._last_rho_obs = rho_obs
        return h_obs, rho_obs

    def _push_measurement(self, state: SlurryState) -> None:
        """
        将最新的真实状态 (h, rho_out) 推入观测缓冲区，用于下一步计算带时滞观测值。
        """
        self._h_buffer.append(state.h)
        self._rho_buffer.append(state.rho_out)

    def step(self, action, t: float) -> None:
        """
        执行仿真一步，计算状态变化，并更新当前状态。

        参数：
        ----
        action : [水阀开度， 灰阀开度] 或 None
            - 在 control_mode = "open_loop" 时，需要显式传入 [ water_cmd,cement_cmd]；
            - 在 control_mode = "siso"/"mimo" 时，忽略此参数（可传 None），由内部 PID 计算。
        t : float
            当前仿真时间 [s]。
        """
        # 1) 读取带时滞的观测值（供 PID 使用）
        h_obs, rho_obs = self._get_observations()

        # 未来要在 phase 内动态改变 Q_s/rho_sp，以下uncomment能自动跟随，目前不需要改变Q_s/rho_sp，保持注释状态。
        # self._init_feedforward_openings()
        # self._init_SMITH_decoupler()

        # 2) 根据控制模式计算阀门指令
        if self.config.control_mode == "open_loop":
            if action is None:
                raise ValueError("control_mode='open_loop' 时必须提供 action。")
            water_cmd = float(action[0])
            cement_cmd = float(action[1])
        else:
            # 1) SISO 液位平衡偏置开度 u_b（假设仅靠水阀平衡 Qs_nominal）；
            # 2) MIMO 模式下的水阀/灰阀前馈基准开度（基于 rho_sp 和 Qs_nominal）和解耦。
            if self.config.control_mode == "siso-level":
                e_h = self.config.h_sp - h_obs
                water_cmd = self.h_pid.step(
                    e_h, self.config.dt,
                    u_ff=self.water_opening_bias,
                    u_min=0, u_max=100)
                cement_cmd = 0.0

            elif self.config.control_mode == "siso-density":
                e_rho = self.config.rho_sp - rho_obs
                cement_cmd = self.rho_pid.step(
                    e_rho, self.config.dt,
                    u_ff=self.cement_opening_ff,  # 工作点前馈（开度）
                    u_min=0.0, u_max=100.0
                )
                water_cmd = 0.0

            elif self.config.control_mode == "mimo":
                # 液位 + 密度双回路 PID
                e_h = self.config.h_sp - h_obs
                e_rho = self.config.rho_sp - rho_obs

                # print("e_h: ", e_h, " e_rho: ", e_rho)


                # 1) 位置式 PID：直接输出“绝对开度命令”（内部：u = u_ff + P + I + D）
                uOw = self.h_pid.step(
                    e_h, self.config.dt,
                    u_ff=self.water_opening_ff,  # 工作点前馈（开度）
                    u_min=0.0, u_max=100.0
                )
                uOc = self.rho_pid.step(
                    e_rho, self.config.dt,
                    u_ff=self.cement_opening_ff,  # 工作点前馈（开度）
                    u_min=0.0, u_max=100.0
                )
                # print("uOw: ", uOw, " uOc: ", uOc)

                if getattr(self.config, "use_smith_decoupler", True):

                    # 2) 用阀门非线性映射：开度 -> 流量
                    #    注意：这里的 flow 已经自动考虑死区/饱和（取决于你的 ValveActuator 实现）
                    Qw_cmd0 = self.water_valve.compute_flow_from_opening(uOw)
                    Qc_cmd0 = self.cement_valve.compute_flow_from_opening(uOc)

                    # 3) 转到“相对工作点”的流量增量 dQ（解耦应作用在增量上）
                    #    工作点流量建议直接用你在 _init_feedforward_openings() 里算出的 Q_w_ff / Q_c_ff
                    dQ_w0 = Qw_cmd0 - self.Q_w_ff
                    dQ_c0 = Qc_cmd0 - self.Q_c_ff

                    # 4) 解耦（在“流量增量域”做）
                    dQ_w = dQ_w0 + self.GD2 * dQ_c0
                    dQ_c = self.GD1 * dQ_w0 + dQ_c0

                    # 5) 回到“绝对流量域”并做物理限幅（可选但推荐）
                    Qw_des = self.Q_w_ff + dQ_w
                    Qc_des = self.Q_c_ff + dQ_c

                    # 若你的阀门模型要求流量非负/不超过 max_flow，可做裁剪
                    Qw_des = max(0.0, min(self.water_valve.params.max_flow, Qw_des))
                    Qc_des = max(0.0, min(self.cement_valve.params.max_flow, Qc_des))

                    # 6) 用阀门逆映射：流量 -> 开度，得到最终“绝对开度命令”
                    water_cmd = self.water_valve.compute_opening_from_flow(Qw_des)
                    cement_cmd = self.cement_valve.compute_opening_from_flow(Qc_des)

                elif getattr(self.config, "use_kff_decoupler", True):
                    dUw = uOw - self.water_opening_ff
                    uOc = uOc + self.config.kff * dUw
                    water_cmd = max(self.water_valve.params.opening_min, min(self.water_valve.params.opening_max, uOw))  # 物理限幅
                    cement_cmd = max(self.cement_valve.params.opening_min, min(self.cement_valve.params.opening_max, uOc))
                else:
                    water_cmd = max(self.water_valve.params.opening_min, min(self.water_valve.params.opening_max, uOw))  # 物理限幅
                    cement_cmd = max(self.cement_valve.params.opening_min, min(self.cement_valve.params.opening_max, uOc))
            else:
                raise ValueError(f"未知的控制模式: {self.config.control_mode}")

        # 3) 将阀门指令裁剪到物理范围内

        w_params = self.water_valve.params
        c_params = self.cement_valve.params
        water_cmd = max(w_params.opening_min, min(w_params.opening_max, water_cmd))
        cement_cmd = max(c_params.opening_min, min(c_params.opening_max, cement_cmd))

        # 4) 设置阀门开度并更新执行器动态

        self.water_valve.set_command(water_cmd)
        self.water_valve.update(self.config.dt)  # 更新阀门状态（包括死区、滞后）
        self.cement_valve.set_command(cement_cmd)
        self.cement_valve.update(self.config.dt)  # 更新阀门状态（包括死区、滞后）
        water_opening = self.water_valve.current_opening
        cement_opening = self.cement_valve.current_opening

        # 5) 计算流量
        Q_w = self.water_valve.get_flow()
        Q_c = self.cement_valve.get_flow()
        self.total_Q_s += self.Q_s * self.config.dt

        dt = self.config.dt  # 本地变量，减少属性查找
        do_diag = getattr(self.config, "enable_diagnostics", False)
        # 6) 数值稳定性检查 & RK4 步进
        if do_diag:
            self.model.check_dt_stability(dt, self.state, Q_w, Q_c)

        prev_state = self.state
        new_state = self.integrator.step(self.model, self.state, Q_w, Q_c, self.Q_s)

        # 7) 质量守恒误差
        if do_diag:
            step_mass_balance_error = self.model.check_mass_balance(
                prev_state=prev_state,
                curr_state=new_state,
                Q_w=Q_w,
                Q_c=Q_c,
                Q_s=self.Q_s,
                dt=dt,
            )
            self.total_mass_balance_error += abs(step_mass_balance_error)
        else:
            step_mass_balance_error = 0.0

        # 8) 将最新真实状态推入观测缓冲区，用于下一步的带时滞观测
        self._push_measurement(new_state)

        # 9) 日志与 StepLogEntry 记录
        if getattr(self.config, "enable_logger", False) and (self.logger is not None):
            step_entry = StepLogEntry(
                t=t,
                state={
                    "h": new_state.h,
                    "x": new_state.x,
                    "rho_out": new_state.rho_out,
                    "M_w": new_state.M_w,
                    "M_c": new_state.M_c,
                    "M": new_state.M,
                },
                flows={
                    "Q_w": Q_w,
                    "Q_c": Q_c,
                    "Q_s": self.Q_s,
                    "total_Q_s": self.total_Q_s,
                    # --- optional diagnostics: raw (before noise) ---
                    "Q_w_raw": getattr(self.water_valve, "current_flow_raw", Q_w),
                    "Q_c_raw": getattr(self.cement_valve, "current_flow_raw", Q_c),
                },
                valve_cmd={
                    "cement_cmd": cement_cmd,
                    "water_cmd": water_cmd,
                },
                valve_actual={
                    "cement_opening": cement_opening,
                    "water_opening": water_opening,
                },
                mass_balance_error=step_mass_balance_error,
                observations={
                    # 这里记录的是带时滞的“测量值”
                    "h_obs": h_obs,
                    "rho_obs": rho_obs,
                    # 保留原字段名以兼容旧代码
                    "rho_out_obs": rho_obs,
                },
                setpoints={
                    "h_sp": self.config.h_sp,
                    "rho_sp": self.config.rho_sp,
                },
                extra={
                    "_last_h_obs": self._last_h_obs,
                    "_last_rho_obs": self._last_rho_obs,
                    "total_mass_balance_error": self.total_mass_balance_error,
                },
            )

            self.logger.log_info(f"仿真步进，时间 {t}, 新状态：{new_state}")
            self.logger.log_step(step_entry)
            # print("valve_comand", step_entry.valve_cmd, "valve_actual", step_entry.valve_actual)

        # 10) 更新内部状态
        self.state = new_state


    def run(self, actions=None) -> None:
        """
        执行仿真。

        参数：
        ----
        actions : list[[water_opening, cement_opening]] 或 None
            - 对于 open_loop 模式，必须提供动作序列；
            - 对于 siso/mimo 模式，可为 None（由环境内部 PID 自动计算）。
        """
        num_steps = int(self.config.t_end / self.config.dt)
        iterator = range(num_steps)

        for k in iterator:
            t = k * self.config.dt

            if self.config.control_mode == "open_loop":
                if actions is None:
                    raise ValueError("open_loop 模式下 Scenario.run 必须提供 actions。")
                action_k = actions[k]
            else:
                action_k = None  # 由环境内部 PID 计算

            self.step(action_k, t)
        if self.config.enable_logger:
            if self.config.log_to_csv:
                self.logger.flush_to_csv()
            self.logger.log_info("仿真结束")








# if __name__ == "__main__":
#     示例 1
#     # 初始化参数和配置
#     plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
#                                h_max=1.8, h_min=0, rho_cement=3150.0,
#                                rho_min_physical=1000, rho_max_physical=3000,
#                                Qs_nominal=0.0/60, mass_balance_tol=1e-3)
#
#     water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
#                                      dead_zone_opening=5, max_flow=0.75/60, min_flow=0,
#                                      actuator_time_constant=2.0, max_opening_rate=50,
#                                      initial_valve_opening=0.0)
#
#     cement_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
#                                      dead_zone_opening=8, max_flow=0.5/60, min_flow=0,
#                                      actuator_time_constant=3.0, max_opening_rate=50,
#                                       initial_valve_opening=0.0)
#
#     simulation_config = SimulationConfig(dt=0.1, t_end=100, record_interval_steps=10,
#                                          control_mode="open_loop",
#                                          log_to_file=True,log_level=logging.DEBUG,
#                                          show_progress_bar=True,run_name_prefix="run")
#
#     # 初始化仿真环境
#     env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config)
#
#
    # initial_slurrystate = SlurryState(h=0.01, rho_out=1000.0) # h的起始值不要设置为0（奇点），给一个很小的数值。
    # initial_slurrystate.initial_x(plant_params=plant_params)
#     initial_slurrystate.compute_states(plant_params)
#
#     # 初始化并运行仿真场景
#     scenario1 = Scenario(env, initial_slurrystate)
#     actions1 = scenario1.create_actions(mode="constant", cement_start=0.0, water_start=30.0)
#     # scenario1.run(actions=actions1)
#
#     plant_params2 = PlantParams(Qs_nominal=0.5/60, mass_balance_tol=1e-3)
#     water_valve_params2 = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
#                                      dead_zone_opening=5, max_flow=0.75/60, min_flow=0,
#                                      actuator_time_constant=2.0, max_opening_rate=50,
#                                      initial_valve_opening=0.0)
#
#     cement_valve_params2 = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
#                                      dead_zone_opening=8, max_flow=0.5/60, min_flow=0,
#                                      actuator_time_constant=3.0, max_opening_rate=50,
#                                       initial_valve_opening=0.0)
#     env2 = CementingSimEnv(plant_params2, water_valve_params2, cement_valve_params2, simulation_config)
#
     # initial_slurrystate2 = SlurryState(h=0.5, rho_out=1600.0)
     # initial_slurrystate2.initial_x(plant_params)
#     initial_slurrystate2.compute_states(plant_params)
#     scenario2 = Scenario(env2, initial_slurrystate2)
#     actions2 = scenario2.create_actions(mode="constant")
#     scenario2.run(actions=actions2)