"""
rollout.py

Shared simulation-rollout layer for the sany_PID RL refactor.

This file intentionally absorbs the simulation-building / episode-running code
that was previously mixed into train_modes.py and evaluate_compare.py.

Design goals
------------
1. train.py and evaluate.py call the same rollout implementation.
2. Paired evaluation can reuse exactly the same sampled uncertainty for
   baseline and RL, including tau_mix_hat, rho-delay, valve max-flow and flow
   noise seeds.
3. Fast training rollouts can return only scalar reward, while evaluation can
   request full trajectories and additional flow/opening histories.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np

_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parents[1]
for _p in (_SCRIPTS / "core", _SCRIPTS / "PID_control", _SCRIPTS / "rl", _SCRIPTS):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

from modes import EpisodeParams, Trajectory, premix_reward, production_reward  # noqa: E402
from scripts.core.sim_config import PlantParams, PhysicalConstraintError, SimulationConfig, ValveParams  # noqa: E402
from scripts.core.sim_env import CementingSimEnv  # noqa: E402
from scripts.core.sim_model import SlurryState  # noqa: E402


# -----------------------------------------------------------------------------
# Crash logging / generic helpers
# -----------------------------------------------------------------------------


def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _dump_jsonl(tag: str, payload: Mapping[str, Any], out_dir: str = "crash_logs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")


def clone_episode(p: EpisodeParams) -> EpisodeParams:
    """Deep-copy an EpisodeParams object so paired runs cannot mutate each other."""
    return deepcopy(p)


# -----------------------------------------------------------------------------
# Episode uncertainty: this is the key to fair paired baseline-vs-RL comparison
# -----------------------------------------------------------------------------


def sample_episode_uncertainty(p: EpisodeParams, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """
    Sample the same per-episode uncertainty that the old train_modes.py rollout
    used inline.

    Returning a dict makes it easy for evaluate.py to reuse exactly the same
    uncertainty for baseline and RL in paired Monte Carlo comparison.
    """
    if rng is None:
        rng = np.random.default_rng()

    tau_mix_nom = float(getattr(p, "tau_mix", 20.0))
    tau_delay_nom = float(getattr(p, "tau_delay", 0.0))
    wv_nom = float(getattr(p, "water_valve_max_flow", 2.0 / 60.0))
    cv_nom = float(getattr(p, "cement_valve_max_flow", 1.0 / 60.0))

    return {
        "tau_mix_hat": float(np.clip(rng.normal(tau_mix_nom, max(tau_mix_nom * 0.25, 1e-9)), 5.0, 100.0)),
        "rho_obs_delay": float(np.clip(rng.normal(tau_delay_nom, max(tau_delay_nom * 0.25, 1e-9)), 0.0, 20.0)),
        "water_valve_max_flow": float(np.clip(rng.normal(wv_nom, max(wv_nom * 0.1, 1e-12)), 1.5 / 60.0, 3.0 / 60.0)),
        "cement_valve_max_flow": float(np.clip(rng.normal(cv_nom, max(cv_nom * 0.1, 1e-12)), 0.5 / 60.0, 2.0 / 60.0)),
        "water_flow_seed": int(rng.integers(0, 2**31 - 1)),
        "cement_flow_seed": int(rng.integers(0, 2**31 - 1)),
    }


def uncertainty_from_episode_nominal(p: EpisodeParams, seed: int = 0) -> Dict[str, Any]:
    """Deterministic helper useful for tiny validations and reproducibility checks."""
    return sample_episode_uncertainty(p, np.random.default_rng(int(seed)))


# -----------------------------------------------------------------------------
# Environment construction
# -----------------------------------------------------------------------------


def apply_scales_to_base_params(base_params: Mapping[str, float], scales: Mapping[str, float]) -> Dict[str, float]:
    """Apply policy scale factors to baseline PID/feedforward parameters."""
    return {
        "Kp_w": float(base_params.get("Kp_w", 0.0)) * float(scales.get("s_w_p", 1.0)),
        "Ki_w": float(base_params.get("Ki_w", 0.0)) * float(scales.get("s_w_i", 1.0)),
        "Kd_w": float(base_params.get("Kd_w", 0.0)) * float(scales.get("s_w_d", 1.0)),
        "Kp_c": float(base_params.get("Kp_c", 0.0)) * float(scales.get("s_c_p", 1.0)),
        "Ki_c": float(base_params.get("Ki_c", 0.0)) * float(scales.get("s_c_i", 1.0)),
        "Kd_c": float(base_params.get("Kd_c", 0.0)) * float(scales.get("s_c_d", 1.0)),
        "ff_w": float(base_params.get("ff_w", 0.0)),
        "ff_c": float(base_params.get("ff_c", 0.0)),
        "kff": float(base_params.get("kff", 0.0)),
    }


def build_episode_env(
    p: EpisodeParams,
    scales: Mapping[str, float],
    rng: Optional[np.random.Generator] = None,
    *,
    base_params: Mapping[str, float],
    uncertainty: Optional[Mapping[str, Any]] = None,
) -> Tuple[CementingSimEnv, SimulationConfig, PlantParams, ValveParams, ValveParams, Dict[str, Any]]:
    """
    Construct one configured CementingSimEnv.

    Parameters
    ----------
    p:
        Episode scenario sampled by ModeSpec.sample_episode().
    scales:
        PID scale factors from policy action or all-ones baseline.
    base_params:
        Episode-specific PID/feedforward base parameters from ModeSpec.
    uncertainty:
        Optional dict from sample_episode_uncertainty(). If supplied, all random
        plant/valve disturbances are reused exactly. This is required for paired
        evaluation.
    """
    if rng is None:
        rng = np.random.default_rng()
    if uncertainty is None:
        uncertainty = sample_episode_uncertainty(p, rng)

    scaled = apply_scales_to_base_params(base_params, scales)

    plant = PlantParams()
    if hasattr(plant, "tau_mix_hat"):
        plant.tau_mix_hat = float(uncertainty.get("tau_mix_hat", getattr(p, "tau_mix", 20.0)))

    water_valve_params = ValveParams(
        max_flow=float(uncertainty.get("water_valve_max_flow", getattr(p, "water_valve_max_flow", 2.0 / 60.0))),
        flow_noise_enable=True,
        flow_noise_mode="mul",
        flow_noise_tau=1,
        flow_noise_std=0.01,
        flow_noise_seed=int(uncertainty.get("water_flow_seed", rng.integers(0, 2**31 - 1))),
    )
    cement_valve_params = ValveParams(
        max_flow=float(uncertainty.get("cement_valve_max_flow", getattr(p, "cement_valve_max_flow", 1.0 / 60.0))),
        flow_noise_enable=True,
        flow_noise_mode="mul",
        flow_noise_tau=1,
        flow_noise_std=0.1,
        flow_noise_seed=int(uncertainty.get("cement_flow_seed", rng.integers(0, 2**31 - 1))),
    )

    qs = float(getattr(p, "qs", 0.0)) if getattr(p, "mode", "premix") == "production" else 0.0

    cfg = SimulationConfig(
        dt=float(p.dt),
        t_end=float(p.duration_s),
        h_sp=float(p.h_sp),
        rho_sp=float(p.rho_sp),
        Qs_nominal=float(qs),
        h_obs_delay=0.0,
        rho_obs_delay=float(uncertainty.get("rho_obs_delay", getattr(p, "tau_delay", 0.0))),
        enable_logger=False,
        log_to_csv=False,
        control_mode=("siso-density" if getattr(p, "mode", "premix") == "premix" else "mimo"),
        h_pid_kp=float(scaled["Kp_w"]),
        h_pid_ki=float(scaled["Ki_w"]),
        h_pid_kd=float(scaled["Kd_w"]),
        rho_pid_kp=float(scaled["Kp_c"]),
        rho_pid_ki=float(scaled["Ki_c"]),
        rho_pid_kd=float(scaled["Kd_c"]),
        use_h_feedforward=True,
        use_density_feedforward=True,
        use_kff_decoupler=False,
        kff=float(scaled["kff"]),
        use_smith_decoupler=False,
    )

    if float(scaled.get("ff_w", 0.0)) > 0.0:
        cfg.water_opening_ff = float(scaled["ff_w"])
    if float(scaled.get("ff_c", 0.0)) > 0.0:
        cfg.cement_opening_ff = float(scaled["ff_c"])

    init_state = SlurryState()
    if hasattr(init_state, "h"):
        init_state.h = float(p.h0)
    if hasattr(init_state, "rho_out"):
        init_state.rho_out = float(p.rho0)

    env = CementingSimEnv(
        plant_params=plant,
        water_valve_params=water_valve_params,
        cement_valve_params=cement_valve_params,
        config=cfg,
        initial_slurry_state=init_state,
    )
    if getattr(p, "mode", "premix") == "production":
        env.Q_s = float(getattr(p, "qs", qs))

    episode_meta: Dict[str, Any] = {
        "mode": str(getattr(p, "mode", "")),
        "dt": float(p.dt),
        "T": float(p.duration_s),
        "h0": float(p.h0),
        "h_sp": float(p.h_sp),
        "rho0": float(p.rho0),
        "rho_sp": float(p.rho_sp),
        "qs": float(getattr(p, "qs", 0.0)),
        "tau_mix_nominal": float(getattr(p, "tau_mix", float("nan"))),
        "tau_delay_nominal": float(getattr(p, "tau_delay", float("nan"))),
        "water_valve_max_flow_nominal": float(getattr(p, "water_valve_max_flow", float("nan"))),
        "cement_valve_max_flow_nominal": float(getattr(p, "cement_valve_max_flow", float("nan"))),
        "uncertainty": dict(uncertainty),
        "base_params": dict(base_params),
        "scaled_params": dict(scaled),
        "scales": dict(scales),
        "cfg": {
            "Qs_nominal": float(cfg.Qs_nominal),
            "control_mode": str(cfg.control_mode),
            "h_sp": float(cfg.h_sp),
            "rho_sp": float(cfg.rho_sp),
            "rho_obs_delay": float(cfg.rho_obs_delay),
            "use_density_feedforward": bool(cfg.use_density_feedforward),
            "use_kff_decoupler": bool(cfg.use_kff_decoupler),
            "kff": float(cfg.kff),
            "water_opening_ff_cfg": float(getattr(cfg, "water_opening_ff", 0.0)),
            "cement_opening_ff_cfg": float(getattr(cfg, "cement_opening_ff", 0.0)),
        },
        "plant": {
            "A": float(getattr(plant, "tank_cross_section_area", float("nan"))),
            "h_min": float(getattr(plant, "h_min", 0.0)),
            "tau_mix_hat": float(getattr(plant, "tau_mix_hat", float("nan"))),
        },
        "valve_params": {
            "wv_max_flow": float(water_valve_params.max_flow),
            "cv_max_flow": float(cement_valve_params.max_flow),
            "dead_zone": float(getattr(water_valve_params, "dead_zone_opening", 5.0)),
        },
    }

    return env, cfg, plant, water_valve_params, cement_valve_params, episode_meta


# -----------------------------------------------------------------------------
# Simulation loops
# -----------------------------------------------------------------------------


def _reward_from_arrays(p: EpisodeParams, rho_hist: np.ndarray, h_hist: Optional[np.ndarray]) -> float:
    if getattr(p, "mode", "premix") == "premix":
        return float(premix_reward(rho_hist, p))
    return float(production_reward(rho_hist, h_hist, p))


def _make_crash_payload(
    p: EpisodeParams,
    meta: Mapping[str, Any],
    env: CementingSimEnv,
    k: int,
    t: float,
    err: BaseException,
) -> Dict[str, Any]:
    return {
        **dict(meta),
        "k": int(k),
        "t": float(t),
        "err": str(err),
        "traceback": traceback.format_exc(),
        "state_before_or_after_step": {
            "h": _as_float(getattr(env.state, "h", float("nan"))),
            "rho_out": _as_float(getattr(env.state, "rho_out", float("nan"))),
            "x": _as_float(getattr(env.state, "x", float("nan"))),
            "M": _as_float(getattr(env.state, "M", float("nan"))),
        },
        "Q": {
            "Q_s": _as_float(getattr(env, "Q_s", float("nan"))),
            "Q_w": _as_float(getattr(env.water_valve, "current_flow", float("nan"))),
            "Q_c": _as_float(getattr(env.cement_valve, "current_flow", float("nan"))),
        },
        "valves": {
            "water_target_opening": _as_float(getattr(env.water_valve, "target_opening", float("nan"))),
            "water_current_opening": _as_float(getattr(env.water_valve, "current_opening", float("nan"))),
            "cement_target_opening": _as_float(getattr(env.cement_valve, "target_opening", float("nan"))),
            "cement_current_opening": _as_float(getattr(env.cement_valve, "current_opening", float("nan"))),
        },
        "episode": {
            "mode": getattr(p, "mode", ""),
            "rho_sp": _as_float(getattr(p, "rho_sp", float("nan"))),
            "h_sp": _as_float(getattr(p, "h_sp", float("nan"))),
            "qs": _as_float(getattr(p, "qs", float("nan"))),
        },
    }


def simulate_episode(
    p: EpisodeParams,
    scales: Mapping[str, float],
    rng: Optional[np.random.Generator] = None,
    return_traj: bool = False,
    *,
    base_params: Mapping[str, float],
    uncertainty: Optional[Mapping[str, Any]] = None,
    full: bool = False,
    catch_errors: bool = True,
) -> Union[Trajectory, float]:
    """
    Run one episode.

    Parameters are kept compatible with the old train_modes.py function.  The
    optional `uncertainty` argument is the new piece needed by evaluate.py for
    paired comparison.
    """
    if rng is None:
        rng = np.random.default_rng()

    env, _cfg, _plant, _wv, _cv, meta = build_episode_env(
        p, scales, rng, base_params=base_params, uncertainty=uncertainty
    )

    dt = float(p.dt)
    T = float(p.duration_s)
    n_steps = int(math.floor(T / dt)) + 1

    want_hist = bool(return_traj or full)
    rho_hist = np.empty(n_steps, dtype=np.float32)
    h_hist = np.empty(n_steps, dtype=np.float32)
    rho_hist[0] = _as_float(getattr(env.state, "rho_out", p.rho0), p.rho0)
    h_hist[0] = _as_float(getattr(env.state, "h", p.h0), p.h0)

    if want_hist:
        t_hist = np.zeros(n_steps, dtype=np.float32)
        uw_hist = np.zeros(n_steps, dtype=np.float32)
        uc_hist = np.zeros(n_steps, dtype=np.float32)
        q_w_hist = np.zeros(n_steps, dtype=np.float32)
        q_c_hist = np.zeros(n_steps, dtype=np.float32)
        q_s_hist = np.zeros(n_steps, dtype=np.float32)
        uw_target_hist = np.zeros(n_steps, dtype=np.float32)
        uc_target_hist = np.zeros(n_steps, dtype=np.float32)
        uw_hist[0] = _as_float(getattr(env.water_valve, "current_opening", 0.0), 0.0)
        uc_hist[0] = _as_float(getattr(env.cement_valve, "current_opening", 0.0), 0.0)
        q_w_hist[0] = _as_float(getattr(env.water_valve, "current_flow", 0.0), 0.0)
        q_c_hist[0] = _as_float(getattr(env.cement_valve, "current_flow", 0.0), 0.0)
        q_s_hist[0] = _as_float(getattr(env, "Q_s", 0.0), 0.0)
        uw_target_hist[0] = _as_float(getattr(env.water_valve, "target_opening", uw_hist[0]), uw_hist[0])
        uc_target_hist[0] = _as_float(getattr(env.cement_valve, "target_opening", uc_hist[0]), uc_hist[0])

    failed = False
    fail_payload: Optional[Dict[str, Any]] = None

    for k in range(1, n_steps):
        t = k * dt
        try:
            env.step(None, t)
        except PhysicalConstraintError as e:
            failed = True
            fail_payload = _make_crash_payload(p, meta, env, k, t, e)
            _dump_jsonl("physical_constraint", fail_payload)
            if not catch_errors:
                raise
            # Fill remaining steps with last valid values.
            rho_hist[k:] = rho_hist[k - 1]
            h_hist[k:] = h_hist[k - 1]
            if want_hist:
                t_hist[k:] = np.arange(k, n_steps, dtype=np.float32) * dt
                uw_hist[k:] = uw_hist[k - 1]
                uc_hist[k:] = uc_hist[k - 1]
                q_w_hist[k:] = q_w_hist[k - 1]
                q_c_hist[k:] = q_c_hist[k - 1]
                q_s_hist[k:] = q_s_hist[k - 1]
                uw_target_hist[k:] = uw_target_hist[k - 1]
                uc_target_hist[k:] = uc_target_hist[k - 1]
            break

        rho_hist[k] = _as_float(getattr(env.state, "rho_out", rho_hist[k - 1]), rho_hist[k - 1])
        h_hist[k] = _as_float(getattr(env.state, "h", h_hist[k - 1]), h_hist[k - 1])
        if want_hist:
            t_hist[k] = float(t)
            uw_hist[k] = _as_float(getattr(env.water_valve, "current_opening", uw_hist[k - 1]), uw_hist[k - 1])
            uc_hist[k] = _as_float(getattr(env.cement_valve, "current_opening", uc_hist[k - 1]), uc_hist[k - 1])
            q_w_hist[k] = _as_float(getattr(env.water_valve, "current_flow", q_w_hist[k - 1]), q_w_hist[k - 1])
            q_c_hist[k] = _as_float(getattr(env.cement_valve, "current_flow", q_c_hist[k - 1]), q_c_hist[k - 1])
            q_s_hist[k] = _as_float(getattr(env, "Q_s", q_s_hist[k - 1]), q_s_hist[k - 1])
            uw_target_hist[k] = _as_float(getattr(env.water_valve, "target_opening", uw_hist[k]), uw_hist[k])
            uc_target_hist[k] = _as_float(getattr(env.cement_valve, "target_opening", uc_hist[k]), uc_hist[k])

    if not want_hist:
        if failed:
            return -10.0
        return _reward_from_arrays(p, rho_hist, h_hist if getattr(p, "mode", "premix") == "production" else None)

    traj = Trajectory(t=t_hist, rho=rho_hist, h=h_hist, u_w=uw_hist, u_c=uc_hist)
    # Dynamic attrs; dataclass has no slots in modes.py.
    setattr(traj, "q_w", q_w_hist)
    setattr(traj, "q_c", q_c_hist)
    setattr(traj, "q_s", q_s_hist)
    setattr(traj, "u_w_target", uw_target_hist)
    setattr(traj, "u_c_target", uc_target_hist)
    setattr(traj, "meta", dict(meta))
    setattr(traj, "failed", bool(failed))
    if fail_payload is not None:
        setattr(traj, "failure", fail_payload)
    return traj


def simulate_episode_with_uncertainty(
    p: EpisodeParams,
    scales: Mapping[str, float],
    uncertainty: Mapping[str, Any],
    *,
    base_params: Mapping[str, float],
    return_traj: bool = False,
    full: bool = False,
    catch_errors: bool = True,
) -> Union[Trajectory, float]:
    """Explicit paired-evaluation interface."""
    return simulate_episode(
        clone_episode(p),
        dict(scales),
        np.random.default_rng(0),
        return_traj=return_traj,
        base_params=dict(base_params),
        uncertainty=dict(uncertainty),
        full=full,
        catch_errors=catch_errors,
    )


def trajectory_reward(traj: Trajectory, p: EpisodeParams) -> float:
    if bool(getattr(traj, "failed", False)):
        return -10.0
    return _reward_from_arrays(p, np.asarray(traj.rho), np.asarray(traj.h))


# -----------------------------------------------------------------------------
# Multiprocessing workers and CSV artifact helpers
# -----------------------------------------------------------------------------


def rollout_reward_worker(args: Tuple[EpisodeParams, Mapping[str, float], Mapping[str, float], float, int]) -> float:
    """Multiprocessing-safe worker used by train.py."""
    p, scales, base_params, override_dt, seed = args
    rrng = np.random.default_rng(int(seed))
    pp = clone_episode(p)
    pp.dt = float(override_dt)
    return float(simulate_episode(pp, dict(scales), rrng, return_traj=False, base_params=dict(base_params)))


def trajectory_to_rows(traj: Trajectory) -> Iterable[Dict[str, float]]:
    q_w = getattr(traj, "q_w", None)
    q_c = getattr(traj, "q_c", None)
    q_s = getattr(traj, "q_s", None)
    uw_t = getattr(traj, "u_w_target", None)
    uc_t = getattr(traj, "u_c_target", None)
    n = len(traj.t)
    for i in range(n):
        yield {
            "t": _as_float(traj.t[i]),
            "rho": _as_float(traj.rho[i]),
            "h": _as_float(traj.h[i]) if getattr(traj, "h", None) is not None else float("nan"),
            "u_w": _as_float(traj.u_w[i]) if getattr(traj, "u_w", None) is not None else float("nan"),
            "u_c": _as_float(traj.u_c[i]) if getattr(traj, "u_c", None) is not None else float("nan"),
            "q_w": _as_float(q_w[i]) if q_w is not None else float("nan"),
            "q_c": _as_float(q_c[i]) if q_c is not None else float("nan"),
            "q_s": _as_float(q_s[i]) if q_s is not None else float("nan"),
            "u_w_target": _as_float(uw_t[i]) if uw_t is not None else float("nan"),
            "u_c_target": _as_float(uc_t[i]) if uc_t is not None else float("nan"),
        }


def save_trajectory_csv(path: str, traj: Trajectory) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = list(trajectory_to_rows(traj))
    fieldnames = ["t", "rho", "h", "u_w", "u_c", "q_w", "q_c", "q_s", "u_w_target", "u_c_target"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path
