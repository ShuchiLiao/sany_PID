from scripts.core.sim_config import PlantParams, ValveParams, SimulationConfig
from scripts.core.sim_model import SlurryState
from scripts.core.sim_env import CementingSimEnv
import logging

# from tqdm import tqdm

# 经过pid_fit_and_tune.py整定后的参数如下：
H_KP = 784.3  # 3829.53
H_KI = 20.24  # 964.822
H_KD = 760.0  # 7600.01
RHO_KP = 0.046  # 0.0573077
RHO_KI = 1.607e-4  # 0.000577 # 0.0052796
RHO_KD = 0.225  # 0.0


def create_actions(
        t_start,
        t_end,
        dt,
        mode: str = "constant",
        water_start: float = 40.0,
        water_end: float = 50.0,
        cement_start: float = 50.0,
        cement_end: float = 50.0,
        change_step: int | None = None,
):
    """
    生成一个时间序列的阀门开度动作。

    返回：
    ----
    actions : list[list[float, float]]
        长度为 num_steps 的列表，
        actions[t] = [water_valve_opening, cement_valve_opening]
    """
    num_steps = int((t_end - t_start) / dt)
    actions: list[list[float]] = []

    if mode == "constant":
        # 全程恒定开度
        for _ in range(num_steps):
            actions.append([water_start, cement_start])

    elif mode == "ramp":
        # 从 start 线性变化到 end
        if num_steps <= 1:
            return [[water_start, cement_start]]
        for i in range(num_steps):
            alpha = i / (num_steps - 1)  # 从 0 到 1
            cement = cement_start + (cement_end - cement_start) * alpha
            water = water_start + (water_end - water_start) * alpha
            actions.append([water, cement])

    elif mode == "step":
        # 简单两段阶梯：前半段用 start，后半段用 end
        if change_step is None:
            change_step = num_steps // 2  # 默认在中间切换

        for i in range(num_steps):
            if i < change_step:
                actions.append([water_start, cement_start])
            else:
                actions.append([water_end, cement_end])

    else:
        raise ValueError(f"未知的动作模式: {mode}")

    return actions


def example_1():
    # 示例： 仿真水阀开启后液位上升， 无出流
    t_start = 0
    t_end = 20.0
    dt = 0.1
    plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
                               h_max=1.8, h_min=0)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                     dead_zone_opening=5, max_flow=0.75 / 60, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    cement_valve_params = ValveParams()

    simulation_config1 = SimulationConfig(dt=dt, t_end=t_end, Qs_nominal=0.0 / 60, control_mode="open_loop",
                                          enable_logger=True, log_level=logging.INFO, log_to_csv=True,
                                          run_name_prefix="example_1_1")
    simulation_config2 = SimulationConfig(dt=dt, t_end=t_end, control_mode="open_loop",
                                          enable_logger=True, log_level=logging.INFO, log_to_csv=True,
                                          run_name_prefix="example_1_2")

    # 初始化仿真环境

    initial_slurrystate = SlurryState(h=0.01, rho_out=1000.0)  # h的起始值不要设置为0（奇点），给一个很小的数值。s

    env1 = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config1,
                           initial_slurrystate)
    env2 = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config2,
                           initial_slurrystate)

    # 第一个示例
    actions1 = create_actions(t_start=t_start, t_end=t_end, dt=dt, mode="constant", cement_start=0.0, water_start=50.0)
    env1.run(actions=actions1)

    # 第二个示例

    actions2 = create_actions(t_start=t_start, t_end=t_end, dt=dt, mode="ramp", cement_start=0.0, cement_end=0,
                              water_start=0, water_end=50.0)
    env2.run(actions=actions2)


def example_2():
    h_target = 0.5
    Qs = 0.5 / 60
    # 示例： 仿真水阀开启后液位上升， 上升到h_sp后, 有出流
    plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
                               h_max=1.8, h_min=0)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                     dead_zone_opening=5, max_flow=0.75 / 60, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    cement_valve_params = ValveParams()

    simulation_config = SimulationConfig(dt=0.1, t_end=200, h_sp=h_target, Qs_nominal=Qs, control_mode="open_loop",
                                         enable_logger=True, log_level=logging.INFO, log_to_csv=True,
                                         run_name_prefix="example_2")

    # 初始化仿真环境
    initial_slurrystate = SlurryState(h=0.01, rho_out=1000.0)  # h的起始值不要设置为0（奇点），给一个很小的数值。
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config, initial_slurrystate)

    dt = env.config.dt
    num_steps = int(env.config.t_end / dt)

    iterator = range(num_steps)

    # 根据液位决定动作
    k = 0
    actions1 = create_actions(0, 200, dt, mode="constant",
                              water_start=50.0, water_end=50.0,
                              cement_start=0.0, cement_end=0.0, )
    while env.state.h < h_target:
        t = k * dt
        action_k = actions1[k]  # water=50, cement=0
        env.step(action_k, t)
        k = k + 1
    print(k)
    env.Q_s = Qs
    i = 0
    actions2 = create_actions(k * dt, 200, dt, mode="ramp",
                              water_start=50.0, water_end=30.0,
                              cement_start=0.0, cement_end=0.0,
                              )
    while k < num_steps - 1:
        t = k * dt
        action_i = actions2[i]  # water=50, cement=0
        env.step(action_i, t)
        k += 1
        i += 1
    print(k)

    env.logger.flush_to_csv()
    env.logger.log_info("仿真结束")


def example_3():
    h_target = 0.5
    rho_target = 1650
    Qs = 0.3 / 60
    # 示例： 仿真水阀和灰阀同时开启后液位上升， 上升到h_sp后, 有出流
    plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
                               h_max=1.8, h_min=0)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                     # slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=5, max_flow=0.75 / 60, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    cement_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                      # slope = max_flow / (opening_max - dead_zone_opening)
                                      dead_zone_opening=5, max_flow=0.4 / 60, min_flow=0,
                                      actuator_time_constant=2.0, max_opening_rate=50,
                                      initial_valve_opening=0.0)

    simulation_config = SimulationConfig(dt=0.1, t_end=200, h_sp=h_target, rho_sp=rho_target, Qs_nominal=0.0,
                                         h_obs_delay=0.0, rho_obs_delay=4.0,
                                         h_pid_kp=5.0, h_pid_ki=1.0, h_pid_kd=1.0,
                                         rho_pid_kp=5.0, rho_pid_ki=1.0, rho_pid_kd=1.0,
                                         use_density_feedforward=True, use_h_feedforward=True,
                                         control_mode="open_loop", use_smith_decoupler=True,
                                         enable_logger=True, log_level=logging.INFO, log_to_csv=True,
                                         run_name_prefix="example_3")

    # 初始化仿真环境
    initial_slurrystate = SlurryState(h=0.01, rho_out=1000.0)  # h的起始值不要设置为0（奇点），给一个很小的数值。
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config, initial_slurrystate)

    dt = env.config.dt
    num_steps = int(env.config.t_end / dt)

    iterator = range(num_steps)

    # 根据液位决定动作
    k = 0
    actions1 = create_actions(0, 200, dt, mode="constant",
                              water_start=40.0, water_end=40.0,
                              cement_start=20.0, cement_end=20.0, )
    while env.state.h < h_target:
        t = k * dt
        action_k = actions1[k]  # water=50, cement=0
        env.step(action_k, t)
        k = k + 1
    print(k)
    env.Q_s = Qs
    i = 0
    actions2 = create_actions(k * dt, 200, dt, mode="ramp",
                              water_start=40.0, water_end=50.0,
                              cement_start=20.0, cement_end=30.0,
                              )
    while k < num_steps - 1:
        t = k * dt
        action_i = actions2[i]  # water=50, cement=0
        env.step(action_i, t)
        k += 1
        i += 1
    print(k)

    env.logger.flush_to_csv()
    env.logger.log_info("仿真结束")


def PID_SISO_prelevel_example():
    # 示例： 单独控制液位到目标液位
    h_target = 1.0
    rho_target = 1650
    Qs = 0.0 / 60
    h_pid_kp = H_KP
    h_pid_ki = H_KI
    h_pid_kd = H_KD
    plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
                               h_max=1.8, h_min=0)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                     # slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=5, max_flow=0.75 / 60, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    cement_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                      # slope = max_flow / (opening_max - dead_zone_opening)
                                      dead_zone_opening=5, max_flow=0.4 / 60, min_flow=0,
                                      actuator_time_constant=2.0, max_opening_rate=50,
                                      initial_valve_opening=0.0)

    simulation_config = SimulationConfig(dt=0.1, t_end=200, h_sp=h_target, rho_sp=rho_target, Qs_nominal=Qs,
                                         h_obs_delay=0.0, rho_obs_delay=4.0,
                                         h_pid_kp=h_pid_kp, h_pid_ki=h_pid_ki, h_pid_kd=h_pid_kd,
                                         rho_pid_kp=5.0, rho_pid_ki=1.0, rho_pid_kd=1.0,
                                         use_density_feedforward=True, use_h_feedforward=True,
                                         control_mode="siso-level", use_smith_decoupler=True,
                                         enable_logger=True, log_level=logging.INFO, log_to_csv=True,

                                         run_name_prefix=f"PID_SISO_prelevel_example_{h_pid_kp}_{h_pid_ki}_{h_pid_kd}")

    # PID 参数（先用一个保守的初值，后面可以通过实验调整）
    # cfg.h_pid_kp = 5.0
    # cfg.h_pid_ki = 0.1
    # cfg.h_pid_kd = 0.0

    # 初始化仿真环境
    initial_slurrystate = SlurryState(h=0.01, rho_out=1000.0)  # h的起始值不要设置为0（奇点），给一个很小的数值。
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config, initial_slurrystate)

    dt = env.config.dt
    num_steps = int(env.config.t_end / dt)
    for k in range(num_steps):
        t = k * simulation_config.dt
        # 在 t=300s 时把目标从 0.5 改为 0.8，形成 setpoint step
        # if t < 300.0:
        #     simulation_config.h_sp = 0.5
        # else:
        #     simulation_config.h_sp = 0.8
        env.step(action=None, t=t)  # SISO / MIMO 模式下内部自己算阀门指令
    # 6) 输出结果（CSV + log）
    env.logger.flush_to_csv()
    print(f"仿真结束，结果保存在：{env.logger.run_dir}")


def PID_SISO_production_example():
    # 示例： 控制液位稳定在目标液位
    h_target = 0.5
    rho_target = 1650
    Qs = 0.5 / 60
    h_pid_kp = H_KP
    h_pid_ki = H_KI
    h_pid_kd = H_KD
    plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
                               h_max=1.8, h_min=0)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                     # slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=5, max_flow=0.75 / 60, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    cement_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                      # slope = max_flow / (opening_max - dead_zone_opening)
                                      dead_zone_opening=5, max_flow=0.4 / 60, min_flow=0,
                                      actuator_time_constant=2.0, max_opening_rate=50,
                                      initial_valve_opening=0.0)

    simulation_config = SimulationConfig(dt=0.1, t_end=200, h_sp=h_target, rho_sp=rho_target, Qs_nominal=Qs,
                                         h_obs_delay=0.0, rho_obs_delay=4.0,
                                         h_pid_kp=h_pid_kp, h_pid_ki=h_pid_ki, h_pid_kd=h_pid_kd,
                                         rho_pid_kp=5.0, rho_pid_ki=1.0, rho_pid_kd=1.0,
                                         use_density_feedforward=True, use_h_feedforward=True,
                                         control_mode="siso-level", use_smith_decoupler=True,
                                         enable_logger=True, log_level=logging.INFO, log_to_csv=True,

                                         run_name_prefix=f"PID_SISO_production_example_{h_pid_kp}_{h_pid_ki}_{h_pid_kd}")

    # PID 参数（先用一个保守的初值，后面可以通过实验调整）
    # cfg.h_pid_kp = 5.0
    # cfg.h_pid_ki = 0.1
    # cfg.h_pid_kd = 0.0

    # 初始化仿真环境
    initial_slurrystate = SlurryState(h=0.5, rho_out=1000.0)  # h的起始值不要设置为0（奇点），给一个很小的数值。
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config, initial_slurrystate)

    dt = env.config.dt
    num_steps = int(env.config.t_end / dt)
    for k in range(num_steps):
        t = k * simulation_config.dt
        # 在 t=300s 时把目标从 0.5 改为 0.8，形成 setpoint step
        # if t < 300.0:
        #     simulation_config.h_sp = 0.5
        # else:
        #     simulation_config.h_sp = 0.8
        env.step(action=None, t=t)  # SISO / MIMO 模式下内部自己算阀门指令
    # 6) 输出结果（CSV + log）
    env.logger.flush_to_csv()
    print(f"仿真结束，结果保存在：{env.logger.run_dir}")


def PID_MIMO_premix_example():
    # 示例： 仿真同时控制水阀灰阀到目标密度后， 无出流
    h_pid_kp = H_KP
    h_pid_ki = H_KI
    h_pid_kd = H_KD
    rho_pid_kp = RHO_KP
    rho_pid_ki = RHO_KI
    rho_pid_kd = RHO_KD
    Qs = 0.0 / 60
    h_sp = 1.2
    rho_sp = 1650.0

    plant_params = PlantParams(tank_cross_section_area=1.5, tank_height=2.0,
                               h_max=1.8, h_min=0)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                     # slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=2, max_flow=1.5 / 60, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    cement_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                      # slope = max_flow / (opening_max - dead_zone_opening)
                                      dead_zone_opening=5, max_flow=1.0 / 60, min_flow=0,
                                      actuator_time_constant=2.0, max_opening_rate=50,
                                      initial_valve_opening=0.0)

    simulation_config = SimulationConfig(dt=0.1, t_end=600, h_sp=h_sp, rho_sp=rho_sp, Qs_nominal=Qs,
                                         h_obs_delay=0.0, rho_obs_delay=10.0,
                                         h_pid_kp=h_pid_kp, h_pid_ki=h_pid_ki, h_pid_kd=h_pid_kd,
                                         rho_pid_kp=rho_pid_kp, rho_pid_ki=rho_pid_ki, rho_pid_kd=rho_pid_kd,
                                         use_density_feedforward=False, use_h_feedforward=False,
                                         control_mode="mimo", use_smith_decoupler=False,
                                         enable_logger=True, log_level=logging.INFO, log_to_csv=True,

                                         run_name_prefix=f"PID_MIMO_premix_example_{h_pid_kp}_{h_pid_ki}_{h_pid_kd}"
                                                         f"_{rho_pid_kp}_{rho_pid_ki}_{rho_pid_kd}")
    # 2) 初始化液位状态（记得不要 h=0，避免奇点）
    initial_slurrystate = SlurryState(h=0.1, rho_out=1000.0)  # h的起始值不要设置为0（奇点），给一个很小的数值。
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config, initial_slurrystate)

    dt = env.config.dt
    num_steps = int(env.config.t_end / dt)

    # 根据液位决定动作
    for k in range(num_steps):
        t = k * simulation_config.dt
        # 在 t=300s 时把目标从 0.5 改为 0.8，形成 setpoint step
        # if t < 300.0:
        #     simulation_config.h_sp = 0.5
        # else:
        #     simulation_config.h_sp = 0.8
        env.step(action=None, t=t)  # SISO / MIMO 模式下内部自己算阀门指令
    # 6) 输出结果（CSV + log）
    env.logger.flush_to_csv()
    env.logger.log_info("仿真结束")
    print(f"仿真结束，结果保存在：{env.logger.run_dir}")


def PID_SISO_premix_example():
    # 示例： 仿真控制灰阀到目标密度后， 无出流
    h_pid_kp = H_KP
    h_pid_ki = H_KI
    h_pid_kd = H_KD
    rho_pid_kp = 0.0768 * 5
    rho_pid_ki = 0  # 0.0027
    rho_pid_kd = 0.6916  # 0.0353
    Qs = 0.0 / 60
    h_sp = 1.0
    rho_sp = 1650.0

    plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
                               h_max=1.8, h_min=0)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                     # slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=2, max_flow=1.5 / 60, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    cement_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                      # slope = max_flow / (opening_max - dead_zone_opening)
                                      dead_zone_opening=5, max_flow=1.0 / 60, min_flow=0,
                                      actuator_time_constant=2.0, max_opening_rate=50,
                                      initial_valve_opening=0.0)

    simulation_config = SimulationConfig(dt=0.1, t_end=600, h_sp=h_sp, rho_sp=rho_sp, Qs_nominal=Qs,
                                         h_obs_delay=0.0, rho_obs_delay=10.0,
                                         h_pid_kp=h_pid_kp, h_pid_ki=h_pid_ki, h_pid_kd=h_pid_kd,
                                         rho_pid_kp=rho_pid_kp, rho_pid_ki=rho_pid_ki, rho_pid_kd=rho_pid_kd,
                                         use_density_feedforward=False, use_h_feedforward=False,
                                         control_mode="siso-density", use_smith_decoupler=False,
                                         enable_logger=True, log_level=logging.INFO, log_to_csv=True,

                                         run_name_prefix=f"PID_SISO_premix_example"
                                                         f"_{rho_pid_kp}_{rho_pid_ki}_{rho_pid_kd}")
    # 2) 初始化液位状态（记得不要 h=0，避免奇点）
    initial_slurrystate = SlurryState(h=1.0, rho_out=1000.0)  # h的起始值不要设置为0（奇点），给一个很小的数值。
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config, initial_slurrystate)

    dt = env.config.dt
    num_steps = int(env.config.t_end / dt)

    # 根据液位决定动作
    for k in range(num_steps):
        t = k * simulation_config.dt
        # 在 t=300s 时把目标从 0.5 改为 0.8，形成 setpoint step
        # if t < 300.0:
        #     simulation_config.h_sp = 0.5
        # else:
        #     simulation_config.h_sp = 0.8
        env.step(action=None, t=t)  # SISO / MIMO 模式下内部自己算阀门指令
    # 6) 输出结果（CSV + log）
    env.logger.flush_to_csv()
    env.logger.log_info("仿真结束")
    print(f"仿真结束，结果保存在：{env.logger.run_dir}")


def PID_MIMO_production_example():
    # 示例： 仿真同时控制灰阀水阀到目标密度后， 有出流
    # h_pid_kp = 29.92*0.564
    # h_pid_ki = 0.305*0.00793
    # h_pid_kd = 29.34*8.46322
    # rho_pid_kp = 0.03888*0.1894
    # rho_pid_ki = 0.0004076*0.005086
    # rho_pid_kd = 0.10006044*0.482758
    h_pid_kp = 29.92
    h_pid_ki = 0.305
    h_pid_kd = 29.34
    rho_pid_kp = 0.03888
    rho_pid_ki = 0.0004076
    rho_pid_kd = 0.10006044
    Qs = 0.01577868815244373
    h_sp = 1.1289495310172282
    rho_sp = 1731.989131427049

    kff = 0.97377
    ff_w = 17.7665 * 1.2
    ff_c = 12.4107 * 1.2

    plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
                               h_max=1.8, h_min=0, tau_mix_hat=92.735365)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                     # slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=5, max_flow=0.03237921381624219, min_flow=0,
                                     actuator_time_constant=0.5, max_opening_rate=50,
                                     initial_valve_opening=0.0,
                                     flow_noise_enable=False, flow_noise_mode="mul",
                                     flow_noise_tau=1,
                                     flow_noise_std=0.01,
                                     flow_noise_seed=8)

    cement_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                      # slope = max_flow / (opening_max - dead_zone_opening)
                                      dead_zone_opening=5, max_flow=0.025746897546415404, min_flow=0,
                                      actuator_time_constant=0.5, max_opening_rate=50,
                                      initial_valve_opening=0.0,
                                      flow_noise_enable=False, flow_noise_mode="mul",
                                      flow_noise_tau=1,
                                      flow_noise_std=0.1,
                                      flow_noise_seed=78, )

    simulation_config = SimulationConfig(dt=0.5, t_end=600, h_sp=h_sp, rho_sp=rho_sp, Qs_nominal=Qs,
                                         h_obs_delay=0.0, rho_obs_delay=5.3226,
                                         h_pid_kp=h_pid_kp, h_pid_ki=h_pid_ki, h_pid_kd=h_pid_kd,
                                         rho_pid_kp=rho_pid_kp, rho_pid_ki=rho_pid_ki, rho_pid_kd=rho_pid_kd,
                                         cement_opening_ff=ff_c, water_opening_ff=ff_w, kff=kff,
                                         use_h_feedforward=False,
                                         use_density_feedforward=True,
                                         control_mode="mimo",
                                         use_smith_decoupler=False,
                                         use_kff_decoupler=True,
                                         enable_logger=True, log_level=logging.INFO, log_to_csv=True,
                                         run_name_prefix=f"PID_MIMO_production_example_{h_pid_kp}_{h_pid_ki}_{h_pid_kd}"
                                                         f"_{rho_pid_kp}_{rho_pid_ki}_{rho_pid_kd}")
    # 2) 初始化液位状态（记得不要 h=0，避免奇点）
    initial_slurrystate = SlurryState(h=1.075424114945089, rho_out=1762.1771892988797)  # h的起始值不要设置为0（奇点），给一个很小的数值。
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config, initial_slurrystate)

    dt = env.config.dt
    num_steps = int(env.config.t_end / dt)

    # 根据液位决定动作
    for k in range(num_steps):
        t = k * simulation_config.dt
        # 在 t=300s 时把目标从 0.5 改为 0.8，形成 setpoint step
        # if t < 300.0:
        #     simulation_config.h_sp = 0.5
        # else:
        #     simulation_config.h_sp = 0.8
        env.step(action=None, t=t)  # SISO / MIMO 模式下内部自己算阀门指令
    # 6) 输出结果（CSV + log）
    env.logger.flush_to_csv()
    env.logger.log_info("仿真结束")
    print(f"仿真结束，结果保存在：{env.logger.run_dir}")


def PID_MIMO_full_example():
    # 示例： 仿真同时控制灰阀水阀到目标密度后， 有出流
    h_pid_kp = 29.92 * 0.164
    h_pid_ki = 0.305 * 0.00793
    h_pid_kd = 29.34 * 8.46322
    rho_pid_kp = 0.03888 * 0.1894
    rho_pid_ki = 0.0004076 * 0.005086
    rho_pid_kd = 0.10006044 * 0.482758
    Qs = 0.01577868815244373
    h_sp = 1.1289495310172282
    rho_sp = 1731.989131427049

    kff = 0.97377
    ff_w = 17.7665
    ff_c = 12.4107
    plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
                               h_max=1.8, h_min=0, tau_mix_hat=92.735365)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                     # slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=5, max_flow=0.03237921381624219, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    cement_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None,
                                      # slope = max_flow / (opening_max - dead_zone_opening)
                                      dead_zone_opening=5, max_flow=0.025746897546415404, min_flow=0,
                                      actuator_time_constant=2.0, max_opening_rate=50,
                                      initial_valve_opening=0.0)

    simulation_config = SimulationConfig(dt=0.1, t_end=600, h_sp=h_sp, rho_sp=rho_sp, Qs_nominal=0.015778688152443737,
                                         h_obs_delay=0.0, rho_obs_delay=5.3226,
                                         h_pid_kp=h_pid_kp, h_pid_ki=h_pid_ki, h_pid_kd=h_pid_kd,
                                         rho_pid_kp=rho_pid_kp, rho_pid_ki=rho_pid_ki, rho_pid_kd=rho_pid_kd,
                                         use_density_feedforward=False, use_h_feedforward=False,
                                         control_mode="mimo", use_smith_decoupler=False,
                                         enable_logger=True, log_level=logging.INFO, log_to_csv=True,
                                         run_name_prefix=f"PID_MIMO_full_example_{h_pid_kp}_{h_pid_ki}_{h_pid_kd}"
                                                         f"_{rho_pid_kp}_{rho_pid_ki}_{rho_pid_kd}")
    # 2) 初始化液位状态（记得不要 h=0，避免奇点）
    initial_slurrystate = SlurryState(h=1.075424114945089, rho_out=1762.1771892988797)  # h的起始值不要设置为0（奇点），给一个很小的数值。
    env = CementingSimEnv(plant_params, water_valve_params, cement_valve_params, simulation_config, initial_slurrystate)

    dt = env.config.dt
    num_steps = int(env.config.t_end / dt)

    Qs_fixed = False
    # 根据液位决定动作
    for k in range(num_steps):
        t = k * simulation_config.dt
        # 在 t=300s 时把目标从 0.5 改为 0.8，形成 setpoint step
        # if t < 300.0:
        #     simulation_config.h_sp = 0.5
        # else:
        #     simulation_config.h_sp = 0.8
        env.step(action=None, t=t)  # SISO / MIMO 模式下内部自己算阀门指令
        if env.state.h >= env.config.h_sp and not Qs_fixed:
            env.Q_s = Qs
            Qs_fixed = True
    # 6) 输出结果（CSV + log）
    env.logger.flush_to_csv()
    env.logger.log_info("仿真结束")
    print(f"仿真结束，结果保存在：{env.logger.run_dir}")


if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    PID_SISO_prelevel_example()
    PID_SISO_production_example()
    PID_MIMO_premix_example()
    PID_SISO_premix_example()
    PID_MIMO_production_example()
    pass