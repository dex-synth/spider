"""Explicit H2R adaptation settings, independent of native dataset paths."""

import math
from dataclasses import asdict, dataclass

import yaml


@dataclass
class IKSettings:
    """Native fast-IK costs and iteration counts; no cropping or filtering."""

    dt: float = 0.005
    wrist_pos_cost: float = 0.3
    wrist_ori_cost: float = 3.0
    finger_pos_cost: float = 10.0
    posture_cost: float = 0.01
    wrist_init_steps: int = 200
    finger_init_steps: int = 300


@dataclass
class MPCSettings:
    """Small development recipe; native sampling and reward equations are retained."""

    num_samples: int = 128
    max_num_iterations: int = 2
    horizon: float = 0.4
    knot_dt: float = 0.1
    physics_substeps: int = 8
    nconmax_per_env: int = 256
    njmax_per_env: int = 1024
    temperature: float = 0.3
    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    final_noise_scale: float = 0.1
    joint_noise_scale: float = 0.15
    pos_noise_scale: float = 0.03
    rot_noise_scale: float = 0.03
    base_pos_rew_scale: float = 1.0
    base_rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.003
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    vel_rew_scale: float = 0.0001
    terminal_rew_scale: float = 1.0


def settings(path=None):
    """Reject unknown options and invalid numeric budgets before preparation."""
    raw = yaml.safe_load(path.read_text()) if path else {}
    if not isinstance(raw, dict) or set(raw) - {"ik", "mpc"}:
        raise ValueError("Settings must contain only ik and mpc mappings")
    result = {}
    for name, cls in (("ik", IKSettings), ("mpc", MPCSettings)):
        defaults = asdict(cls())
        values = asdict(cls(**raw.get(name, {})))
        for key, value in values.items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
                or (type(defaults[key]) is int and type(value) is not int)
            ):
                raise ValueError(
                    f"Expected positive {type(defaults[key]).__name__}: {name}.{key}"
                )
        result[name] = values
    if result["mpc"]["num_samples"] < 100:
        raise ValueError("Native exploit sampling requires at least 100 samples")
    return result


def validate_clock(options, task):
    """Reject incompatible control knots before starting IK or allocating a GPU."""
    dt = task["dt_s"] * task["control_ticks"]
    steps = {}
    for name in ("horizon", "knot_dt"):
        ratio = options[name] / dt
        steps[name] = round(ratio)
        if steps[name] < 1 or not math.isclose(ratio, steps[name], abs_tol=1e-9):
            raise ValueError(f"mpc.{name} must be a positive multiple of control_dt_s")
    if steps["horizon"] % steps["knot_dt"]:
        raise ValueError("MPC horizon must contain an integer number of knots")
