"""
sim_config.py

参数配置 & 错误诊断层
====================

本文件负责：
1. 存放固井混浆系统仿真所需的所有“配置与参数类”（不做具体数值计算）；
2. 定义仿真过程中可能出现的异常类型（物理错误 / 质量守恒错误 / 数值错误）；
3. 提供一个统一的日志与诊断工具，用于：
   - 运行时在 console 与日志文件中记录关键信息；
   - 按步记录仿真轨迹（便于后续保存为 CSV、作图与排查问题）。
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# 一、参数 & 配置层
# ============================================================================


@dataclass
class PlantParams:
    """
    系统物理参数（System physical parameters）

    说明：
    ----
    本类仅存放“固井混浆系统”的几何、物性与混合相关的**常数参数**，
    不包含任何动态计算逻辑。

    典型用途：
    --------
    - 供物理模型 (SlurryTankModel) 使用，用于计算液位/体积变化、质量守恒等；
    - 供数值检查（如质量守恒误差阈值）与物理约束（如密度上下限）使用。
    """

    # === 1) 几何参数（Geometry） =========================================
    tank_cross_section_area: float = 1.0
    """
    罐体截面积 [m^2]
    - 假设罐为截面积恒定的立式圆柱或近似矩形槽；
    - 液位 h 与体积 V 的关系通常为 V = A * h。
    """

    tank_height: float = 2.0
    """
    罐体总高度 [m]
    """

    h_min: float = 0.01
    """
    最小有效液位 [m]
    - 低于此液位可能导致吸空、搅拌失效等；
    - 可用于物理约束和报警。
    """

    h_max: float = 1.8
    """
    最大有效液位 [m]
    - 高于此液位可能导致溢流风险；
    - 可用于物理约束和报警。
    """

    # === 2) 介质物性（Fluid / solid properties） ==========================
    rho_water: float = 1000.0
    """
    清水密度 [kg/m^3]
    - 常温常压下约为 1000 kg/m^3。
    """

    rho_cement: float = 3150.0
    """
    纯水泥粉（固相）密度 [kg/m^3]
    - 具体数值可根据水泥品种标定。
    """

    rho_min_physical: float = 1000.0
    """
    物理上允许的最小密度 [kg/m^3]
    - 用于物理约束检查，防止数值发散到不合理密度。
    """

    rho_max_physical: float = 3150.0
    """
    物理上允许的最大密度 [kg/m^3]
    - 用于物理约束检查，防止数值发散到不合理密度。
    """

    tau_mix_hat: float = 10.0
    """
    混合时间常数
    """

    # === 4) 质量守恒检查（Mass balance check） ==========================
    mass_balance_tol: float = 1e-6
    """
    质量守恒误差容差（relative or absolute tolerance）[- 或 kg/s]
    """

    # === 液位单位换算（m <-> %） ==========================================
    def h_m_to_percent(self, h_m: float, *, clamp: bool = True) -> float:
        """
        将液位从 [m] 转为 [%]，其中 100% 对应 tank_height。
        """
        if self.tank_height <= 0:
            raise ValueError("tank_height must be > 0 to convert level to percent.")
        pct = 100.0 * float(h_m) / float(self.tank_height)
        if clamp:
            pct = max(0.0, min(100.0, pct))
        return pct

    def h_percent_to_m(self, h_pct: float, *, clamp: bool = True) -> float:
        """
        将液位从 [%] 转为 [m]，其中 100% 对应 tank_height。
        """
        if self.tank_height <= 0:
            raise ValueError("tank_height must be > 0 to convert percent level to meters.")
        hp = float(h_pct)
        if clamp:
            hp = max(0.0, min(100.0, hp))
        return (hp / 100.0) * float(self.tank_height)



@dataclass
class ValveParams:
    """
    阀门与执行器参数（Valve & actuator parameters）

    说明：
    ----
    本类描述**单个阀门**或一类阀门的静态与动态特性，
    用于后续 ValveActuator 执行器模型。

    包含信息：
    --------
    - 开度范围与死区（0～100%）；
    - 开度–流量近似一阶线性关系；
    - 一阶执行器滞后时间常数；
    - 开度变化速率限幅；
    - 流量物理上下限。
    """

    # === 1) 结构参数（Structure / range） ================================
    opening_min: float = 0.0
    """
    最小阀门开度 [%]
    - 一般为 0（全关），但也可能存在安全最小开度。
    """

    opening_max: float = 100.0
    """
    最大阀门开度 [%], 对应max_flow
    - 一般为 100（全开）。
    """

    dead_zone_opening: float = 5.0
    """
    阀门死区上限 [%]
    - 当开度 < dead_zone_opening 时，视为无流量；
    - 死区用于模拟小开度下实际流量近似为 0 的工况。
    """

    # === 2) 流量特性（Opening -> flow characteristic） ====================
    max_flow: float = 1.5/60
    """
    阀门在最大开度时(opening_max)的体积流量 [m^3/s]
    - 通常来自标定或厂家数据；
    - 开度–流量线性关系可根据 max_flow 推导：
      Q = slope * (opening - dead_zone_opening)，并在 [Q_min, Q_max] 内裁剪。
    """

    min_flow: float = 0.0
    """
    阀门允许的最小体积流量 [m^3/s]
    - 通常为 0；
    - 与 max_flow 一起用于防止数值超界。
    """

    linear_slope: Optional[float] = None
    """
    开度–流量线性段的斜率 [m^3/s per %]
    - 若为 None，则在初始化或运行时根据 max_flow 自动推导：
      slope = max_flow / (opening_max - dead_zone_opening)
    """

    linear_offset: float = 0.0
    """
    开度–流量线性段的偏置项 [m^3/s]
    - 通常为 0；
    - 若需要拟合非零截距，可以调整此参数。
    """

    # === 3) 动态特性（Actuator dynamics） ================================
    actuator_time_constant: float = 0.5
    """
    阀门执行器一阶滞后时间常数 T_a [s]
    - 描述实际开度响应控制指令的“慵懒程度”；
    - 时间常数越大，响应越慢。
    """

    max_opening_rate: float = 50.0
    """
    每秒最大开度变化速率 [%/s]
    - 用于模拟执行器电机/液压系统的速度限制；
    - 例如 max_opening_rate = 50 [%/s]，则 1 秒最多能从 0 开到 50%。
    """

    initial_valve_opening: float = 0.0

    # === 4) 流量抖动（Flow jitter / disturbance） ==============================
    flow_noise_enable: bool = False
    """
    是否开启“流量抖动”模拟（进料量随机波动）
    """

    flow_noise_mode: str = "mul"
    """
    抖动叠加方式：
    - "mul": 乘性扰动，Q = Q_base * (1 + eps)，eps 为相对扰动（推荐）
    - "add": 加性扰动，Q = Q_base + eps，eps 为绝对扰动[m^3/s]
    """

    flow_noise_std: float = 0.03
    """
    抖动强度：
    - mode="mul" 时：eps 的标准差（相对量），例如 0.01=1% 抖动
    - mode="add" 时：eps 的标准差（绝对量），单位 m^3/s
    """

    flow_noise_tau: float = 1.0
    """
    抖动相关时间常数 [s]（OU/AR(1) 相关噪声）
    - tau 越大：抖动变化越慢（更像“供料波动”）
    - tau<=0：退化为每步独立白噪声（一般不推荐）
    """

    flow_noise_seed: Optional[int] = 0
    """
    随机种子（保证可复现）
    - 水阀和灰阀建议给不同 seed
    """

    def __post_init__(self) -> None:
        """
        初始化后检查参数自洽性：
        - 开度区间是否合理；
        - 流量上下限与 max_flow 的关系是否合理。
        """
        if self.opening_max <= self.opening_min:
            raise ValueError(
                f"Invalid opening range: [{self.opening_min}, {self.opening_max}]"
            )
        if not (self.opening_min <= self.dead_zone_opening <= self.opening_max):
            raise ValueError(
                f"dead_zone_opening={self.dead_zone_opening} is outside "
                f"[{self.opening_min}, {self.opening_max}]"
            )
        if self.max_flow < 0.0:
            raise ValueError("max_flow must be >= 0")



@dataclass
class SimulationConfig:
    """
    仿真配置（Simulation settings）

    说明：
    ----
    本类只描述“如何运行仿真”，即数值积分和输出配置，
    与具体物理参数（PlantParams）和阀门特性（ValveParams）解耦。

    主要包括：
    --------
    - 时间步长与总时长；
    - 输出采样间隔；
    - 日志和结果保存配置。
    """

    # === 1) 时间相关（Time settings） ====================================
    dt: float = 0.05
    """
    仿真时间步长 [s]
    - RK4 积分器将使用此 dt 作为每一步的时间间隔；
    - dt 过大可能导致数值不稳定，过小则计算量大。
    """

    t_end: float = 3000.0
    """
    仿真总时长 [s]
    - 仿真将从 t=0 运行到 t=t_end；
    - 具体时间序列通常由上层环境生成。
    """

    record_interval_steps: int = 1
    """
    记录仿真轨迹的步长间隔 [步]
    - 例如 record_interval_steps = 10 表示每 10 步写入一次日志；
    - 值越大，CSV 与图像中的点越少，文件越小。
    """

    h_sp: float = 1.0
    """
    目标液位 [m]
    """



    rho_sp: float = 1650.0
    """
    目标密度 [kg/m^3]
    """

    # 出流设置（Outlet / slurry pump settings） ====================
    # Qs_fixed: bool = True
    # #Qs在仿真过程中是否变化


    Qs_nominal: float = 0.5/60
    """
    出口体积流量 Q_s 的标称值 [m^3/s]
    - 在大多数情况下作为常数使用；
    - 若测试需要，也可以在环境中按场景修改。
    """
    #
    # Qs_series:list[float] = None
    # #出流量序列
    # === 2.5) 测量时滞相关（Measurement delay） ==========================
    h_obs_delay: float = 0.0
    """
    液位测量时滞 [s]
    - 代表液位传感器从真实液位到输出测量值的延迟时间；
    - 一般可以近似为 0（液位计响应快）。
    """

    rho_obs_delay: float = 10.0
    """
    密度测量时滞 [s]
    - 代表 rho_out 到密度计观测值 rho_obs 的延迟时间；
    - 例如 rho_obs(t) ≈ rho_out(t - rho_obs_delay) + 噪声。
    """

    # === 2.6) PID 控制参数（PID gains） ==================================
    # 液位 PID（SISO & MIMO 共用）
    h_pid_kp: float = 5.0
    """
    液位回路比例增益 Kp
    - 数值仅作为默认值，具体可在仿真脚本中根据对象特性调节。
    """

    h_pid_ki: float = 1.0e-3
    """
    液位回路积分增益 Ki
    """

    h_pid_kd: float = 0.0
    """
    液位回路微分增益 Kd
    - 如不需要 D，可保持 0.
    """

    # 密度 PID（仅在 MIMO 使用）
    rho_pid_kp: float = 1.0e-3
    """
    密度回路比例增益 Kp
    - 数值量级通常远小于液位回路（因为密度单位和动态不同）。
    """

    rho_pid_ki: float = 1.0e-4
    """
    密度回路积分增益 Ki
    """

    rho_pid_kd: float = 0.0
    """
    密度回路微分增益 Kd
    """

    # === 2.7) 密度前馈相关（Density feed-forward settings） ==============
    use_density_feedforward: bool = False
    """
    是否启用基于目标密度的前馈（MIMO 模式），自动计算
    - True: 使用 rho_sp（或 rho_sp_for_ff）计算理论水灰配比与阀门基准开度；
    - False: 仅根据 Qs_nominal 计算“总进流=出流”的平衡开度，不做密度前馈。
    """
    cement_opening_ff: float = 0.0
    water_opening_ff: float = 0.0

    use_h_feedforward: bool = True
    """
    是否启用基于目标液位的前馈（MIMO 模式）
    - True: 使用 h_sp（或 h_sp_for_ff）计算水阀基准开度；
    - False: 不做前馈。
    """
    use_kff_decoupler: bool = False
    """
    - True: 耦合前馈；
    - False: 不做耦合前馈。
    """

    kff: float = 0.4
    "耦合前馈系数"

    # === 3) 控制模式（Control mode） ====================================
    control_mode: str = "open_loop"
    """
    当前仿真使用的控制模式：
    - open_loop: 开环测试；
    - siso:  单回路 PID（液位）；
    - siso-density: 单回路 PID（密度）
    - mimo:  双回路 PID（液位 + 密度）。
    """
    use_smith_decoupler: bool = True   # True: Fan 解耦; False: 无解耦

    # === 4) 输出与日志（Output & logging） ================================
    runs_root_dir: Path = field(
        default_factory=lambda: Path("./sim_runs")
    )
    """
    仿真结果保存的根目录（root path）
    - 每次仿真可在此目录下创建一个子目录（例如以时间戳命名）；
    - 并在子目录内保存 CSV、图像和日志文件。
    """

    enable_diagnostics: bool = False
    """
    检查dt稳定性和质量守恒。
    """
    enable_logger: bool = True
    """
    是否输出logger日志。
    - 若为 True，则 需要新建SimulationLogger，然后会在对应目录下生成 .log 文件；
    - 若为 False，则仅在 console 输出。
    """

    log_to_csv: bool = True
    """
    是否保存log为csv
    """


    log_level: int = logging.INFO
    """
    日志级别（logging level）
    - 典型值：logging.ERROR / logging.WARNING / logging.INFO / logging.DEBUG；
    - 日志记录越详细，文件越大，但便于排查问题。
    """

    run_name_prefix: str = "run"
    """
    仿真运行名称前缀
    - 实际运行名通常为 f"{run_name_prefix}_{时间戳}"；
    - 用于在 runs_root_dir 下区分不同试验。
    """

    def make_run_dir(self) -> Path:
        """
        根据当前时间戳和 run_name_prefix 创建一个“本次仿真”的输出目录。

        返回：
        ----
        Path: 创建好的目录路径。若已存在，则直接返回该路径。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.run_name_prefix}_{timestamp}"
        run_dir = self.runs_root_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir



# ============================================================================
# 二、错误与诊断层：异常类型定义
# ============================================================================


class SimulationError(Exception):
    """
    仿真通用异常基类（Base class for simulation-related errors）

    所有仿真过程中可能抛出的异常（物理约束错误、质量守恒错误、数值错误）均可继承自本类。

    统一特性：
    --------
    - message: 人类可读的错误信息；
    - context: 可选的上下文信息（如时间 t、状态摘要、控制命令等），便于日志记录。
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def format_detailed_message(self) -> str:
        """
        将错误信息和上下文整合为一个详细的字符串，用于日志输出。

        返回：
        ----
        str: 适合写入日志文件和 console 的详细错误描述。
        """
        if not self.context:
            return self.message
        ctx_parts = [f"{k}={v!r}" for k, v in self.context.items()]
        ctx_str = ", ".join(ctx_parts)
        return f"{self.message} | context: {ctx_str}"


class PhysicalConstraintError(SimulationError):
    """
    物理不允许错误（Physical constraint violation）

    典型触发条件：
    ------------
    - 液位 < 0 或 > 罐体上限；
    - 某个质量变量为负；
    - 密度超出物理合理范围；
    - tau_mix 或其它物理参数出现负值或 NaN。
    """

    pass


class MassBalanceError(SimulationError):
    """
    质量守恒错误（Mass balance violation）

    典型触发条件：
    ------------
    - SlurryTankModel.check_mass_balance(...) 发现质量守恒误差超过 PlantParams.mass_balance_tol；
    - 连续多步误差偏大，表明模型或积分器存在系统性错误。
    """

    pass


class NumericalError(SimulationError):
    """
    数值计算错误（Numerical instability or failure）

    典型触发条件：
    ------------
    - 积分过程中产生 NaN / Inf；
    - 单步状态变化异常巨大（疑似发散）；
    - 线性代数运算失败（如矩阵不可逆等）。
    """

    pass


# ============================================================================
# 三、日志 & 诊断工具
# ============================================================================


@dataclass
class StepLogEntry:
    """
    单步仿真记录（Per-step simulation record）

    说明：
    ----
    - 用于统一收集“某一时间步”的所有重要信息；
    - 后续可由 SimulationLogger 将一系列 StepLogEntry 写入 CSV，
      并在测试模块中据此绘图分析。

    字段设计：
    --------
    - t: 当前时间 [s]；
    - state: 状态变量（例如：h, M_w, M_c,rho_out, ...）；
    - flows: 进出流量（Q_w, Q_c, Q_s 等）；
    - valve_cmd: 阀门指令开度（控制器输出）；
    - valve_actual: 阀门实际开度（考虑执行器滞后与限幅）；
    - mass_balance_error: 本步质量守恒误差；
    - observations: 观测值（h_obs, rho_obs 等）；
    - setpoints: 设定值（h_sp, rho_sp 等）；
    - extra: 预留字段，可放任何辅助信息（例如控制模式、奖励、episode id 等）。
    """

    t: float
    state: Dict[str, Any] = field(default_factory=dict)
    flows: Dict[str, Any] = field(default_factory=dict)
    valve_cmd: Dict[str, Any] = field(default_factory=dict)
    valve_actual: Dict[str, Any] = field(default_factory=dict)
    mass_balance_error: Optional[float] = None
    observations: Dict[str, Any] = field(default_factory=dict)
    setpoints: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        将嵌套字典展开为“扁平字典”，便于写 CSV。

        例如：
        - state.h -> 'state.h'
        - flows.Qw -> 'flows.Qw'
        等。
        """
        base = {"t": self.t, "mass_balance_error": self.mass_balance_error}

        def expand(prefix: str, d: Dict[str, Any], out: Dict[str, Any]) -> None:
            for k, v in d.items():
                out[f"{prefix}.{k}"] = v

        expand("state", self.state, base)
        expand("flows", self.flows, base)
        expand("valve_cmd", self.valve_cmd, base)
        expand("valve_actual", self.valve_actual, base)
        expand("obs", self.observations, base)
        expand("sp", self.setpoints, base)
        expand("extra", self.extra, base)
        return base


class SimulationLogger:
    """
    仿真日志记录器（Simulation logger）

    说明：
    ----
    统一管理：
    - Python logging（console + file）；
    - 每步 StepLogEntry 的收集与导出（CSV）；
    - 异常的详细记录。

    """

    def __init__(
        self,
        run_dir: Path,
        level: int = logging.INFO,
        logger_name: str = "cementing_sim",
    ) -> None:
        """
        初始化日志系统。

        参数：
        ----
        run_dir : Path
            本次仿真输出目录（由 SimulationConfig.make_run_dir() 创建）；
        level : int
            日志级别（logging.INFO / logging.DEBUG 等）；
        logger_name : str
            Python logger 的名称，便于在其他模块中复用相同 logger。
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # 避免重复输出到 root logger

        # 为避免重复添加 handler，先清空现有 handler
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Console handler（打印到终端）
        # console_handler = logging.StreamHandler()
        # console_fmt = logging.Formatter(
        #     fmt="%(asctime)s - %(levelname)s - %(message)s",
        #     datefmt="%H:%M:%S",
        # )
        # console_handler.setFormatter(console_fmt)
        # console_handler.setLevel(level)
        # self.logger.addHandler(console_handler)

        # File handler（写入文件）
        log_file = self.run_dir / "simulation.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_fmt = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)

        # 存储每步轨迹
        self._step_entries: List[StepLogEntry] = []

    # ------------------------------------------------------------------ #
    # 日志基础方法（信息 / 警告 / 错误）
    # ------------------------------------------------------------------ #

    def log_debug(self, msg: str) -> None:
        """记录调试信息（debug level）。"""
        self.logger.debug(msg)

    def log_info(self, msg: str) -> None:
        """记录一般信息（info level）。"""
        self.logger.info(msg)

    def log_warning(self, msg: str) -> None:
        """记录警告信息（warning level）。"""
        self.logger.warning(msg)

    def log_error(self, msg: str) -> None:
        """记录错误信息（error level）。"""
        self.logger.error(msg)

    def log_exception(self, exc: Exception) -> None:
        """
        记录异常的详细信息（含 traceback）。

        - 对 SimulationError 子类，会调用 format_detailed_message()；
        - 其他异常则使用默认的 str(exc)。
        """
        if isinstance(exc, SimulationError):
            msg = exc.format_detailed_message()
        else:
            msg = str(exc)
        self.logger.exception(msg)

    # ------------------------------------------------------------------ #
    # 步进记录（Step log entries）
    # ------------------------------------------------------------------ #

    def log_step(self, entry: StepLogEntry) -> None:
        """
        记录单步仿真信息。

        参数：
        ----
        entry : StepLogEntry
            该时间步的完整记录。
        """
        self._step_entries.append(entry)

    def flush_to_csv(self, filename: str = "trajectory.csv") -> Path:
        """
        将所有 StepLogEntry 写入 CSV 文件。

        参数：
        ----
        filename : str
            保存的文件名（位于 run_dir 下）。

        返回：
        ----
        Path: 写入的 CSV 文件路径。
        """
        if not self._step_entries:
            self.logger.warning("No step entries to write; trajectory.csv will not be created.")
            return self.run_dir / filename

        csv_path = self.run_dir / filename

        # 将第一条记录作为列名模板
        first_flat = self._step_entries[0].to_flat_dict()
        fieldnames = list(first_flat.keys())

        # 写入 CSV
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._step_entries:
                row = entry.to_flat_dict()
                # 确保所有字段都有值（若后续新增字段，可在这里补缺省）
                for fn in fieldnames:
                    row.setdefault(fn, "")
                writer.writerow(row)

        self.logger.info(f"Trajectory saved to: {csv_path}")
        return csv_path

# 示例
# 1) plant = PlantParams(...)
plant = PlantParams(
    tank_cross_section_area=1.0,
    tank_height=2.0,
    h_min=0.01,
    h_max=1.8,
    rho_water=1000.0,
    rho_cement=3150.0,
    rho_min_physical=1000.0,
    rho_max_physical=3150.0,
    tau_mix_hat=10.0,
    mass_balance_tol=1e-6,
)

# 2) valve = ValveParams(...)
valve = ValveParams(
    opening_min=0.0,
    opening_max=100.0,
    dead_zone_opening=5.0,
    max_flow=0.0125,              # 0.75/60
    min_flow=0.0,
    linear_slope=None,            # 运行时用 max_flow/(opening_max-dead_zone_opening) 推导
    linear_offset=0.0,
    actuator_time_constant=1.0,
    max_opening_rate=50.0,
    initial_valve_opening=0.0,
)

# 3) config = SimulationConfig(...)
config = SimulationConfig(
    dt=0.05,
    t_end=3000.0,
    record_interval_steps=1,
    h_sp=1.0,
    rho_sp=1650.0,
    Qs_nominal=0.008333333333333333,  # 0.5/60
    h_obs_delay=0.0,
    rho_obs_delay=10.0,
    h_pid_kp=5.0,
    h_pid_ki=1.0e-3,
    h_pid_kd=0.0,
    rho_pid_kp=1.0e-3,
    rho_pid_ki=1.0e-4,
    rho_pid_kd=0.0,
    use_density_feedforward=False,
    cement_opening_ff=0.0,
    water_opening_ff=0.0,
    use_h_feedforward=True,
    use_kff_decoupler=False,
    kff=0.4,
    control_mode="open_loop",
    use_smith_decoupler=True,
    runs_root_dir=Path("./sim_runs"),
    enable_logger=True,
    log_to_csv=True,
    log_level=logging.INFO,
    run_name_prefix="run",
)
