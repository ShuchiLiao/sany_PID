from scripts.core.sim_config import PlantParams, ValveParams, SimulationConfig
from scripts.core.sim_model import SlurryState
from scripts.core.sim_env import CementingSimEnv
import logging
#from tqdm import tqdm

#经过pid_fit_and_tune.py整定后的参数如下：
H_KP = 784.3 # 3829.53
H_KI = 20.24 # 964.822
H_KD = 760.0 # 7600.01
RHO_KP = 0.046 # 0.0573077
RHO_KI = 1.607e-4 #0.000577 # 0.0052796
RHO_KD = 0.225 # 0.0

def PID_MIMO_premix_example():
    # 示例： 仿真同时控制水阀灰阀到目标密度后， 无出流
    h_pid_kp = H_KP
    h_pid_ki = H_KI
    h_pid_kd = H_KD
    rho_pid_kp = RHO_KP
    rho_pid_ki = RHO_KI
    rho_pid_kd = RHO_KD
    Qs = 0.0/60
    h_sp = 1.2
    rho_sp = 1650.0



    plant_params = PlantParams(tank_cross_section_area=1.5, tank_height=2.0,
                               h_max=1.8, h_min=0)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None, #slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=2, max_flow=1.5/60, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    cement_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None, #slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=5, max_flow=1.0/60, min_flow=0,
                                     actuator_time_constant=2.0, max_opening_rate=50,
                                     initial_valve_opening=0.0)

    simulation_config = SimulationConfig(dt=0.1, t_end=600, h_sp=h_sp, rho_sp=rho_sp, Qs_nominal=Qs,
                                         h_obs_delay=0.0, rho_obs_delay=10.0,
                                         h_pid_kp=h_pid_kp, h_pid_ki=h_pid_ki, h_pid_kd=h_pid_kd,
                                         rho_pid_kp=rho_pid_kp, rho_pid_ki=rho_pid_ki, rho_pid_kd=rho_pid_kd,
                                         use_density_feedforward=False,use_h_feedforward=False,
                                         control_mode="mimo", use_smith_decoupler=False,
                                         enable_logger=True,log_level=logging.INFO, log_to_csv=True,
                                         
                                         run_name_prefix=f"PID_MIMO_premix_example_{h_pid_kp}_{h_pid_ki}_{h_pid_kd}"
                                                         f"_{rho_pid_kp}_{rho_pid_ki}_{rho_pid_kd}")
    # 2) 初始化液位状态（记得不要 h=0，避免奇点）
    initial_slurrystate = SlurryState(h=0.1, rho_out=1000.0) # h的起始值不要设置为0（奇点），给一个很小的数值。
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
        env.step(action=None, t=t)   # SISO / MIMO 模式下内部自己算阀门指令
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
    ff_w = 17.7665*1.2
    ff_c = 12.4107*1.2

    plant_params = PlantParams(tank_cross_section_area=1.0, tank_height=2.0,
                               h_max=1.8, h_min=0, tau_mix_hat=92.735365)

    water_valve_params = ValveParams(opening_max=100, opening_min=0, linear_slope=None, #slope = max_flow / (opening_max - dead_zone_opening)
                                     dead_zone_opening=5, max_flow=0.03237921381624219, min_flow=0,
                                     actuator_time_constant=0.5, max_opening_rate=50,
                                     initial_valve_opening=0.0,
                                     flow_noise_enable=False, flow_noise_mode="mul",
                                     flow_noise_tau=1,
                                     flow_noise_std=0.01,
                                     flow_noise_seed=8 )

    cement_valve_params = ValveParams( opening_max=100, opening_min=0, linear_slope=None, #slope = max_flow / (opening_max - dead_zone_opening)
                                        dead_zone_opening=5, max_flow=0.025746897546415404, min_flow=0,
                                        actuator_time_constant=0.5, max_opening_rate=50,
                                        initial_valve_opening=0.0,
                                        flow_noise_enable=False, flow_noise_mode="mul",
                                        flow_noise_tau=1,
                                        flow_noise_std=0.1,
                                        flow_noise_seed=78,)


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
    initial_slurrystate = SlurryState(h=1.075424114945089, rho_out=1762.1771892988797) # h的起始值不要设置为0（奇点），给一个很小的数值。
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

if __name__ == "__main__":
    PID_MIMO_premix_example()
    PID_MIMO_production_example()
    pass