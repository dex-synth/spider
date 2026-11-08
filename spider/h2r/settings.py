"""Explicit H2R adaptation settings, independent of native dataset paths."""

import math
from dataclasses import asdict, dataclass

import yaml


@dataclass
class IKSettings:
    """Paper IK defaults from upstream preprocess/ik.py."""

    sim_dt: float = 0.005
    ref_dt: float = 0.02
    max_num_initial_guess: int = 8
    average_frame_size: int = 3
    finger_solimp_width: float = 0.01
    wrist_solimp_width: float = 10.0
    wrist_torque_scale: float = 10.0
    object_solimp_width: float = 0.001
    seed: int = 0


@dataclass
class MPCSettings:
    """Ordinary runner defaults from upstream examples/config/default.yaml."""

    num_samples: int = 1024
    max_num_iterations: int = 32
    horizon: float = 1.6
    knot_dt: float = 0.4
    sim_dt: float = 0.01
    ref_dt: float = 0.02
    ctrl_dt: float = 0.4
    physics_substeps: int = 1
    improvement_threshold: float = 0.01
    improvement_check_steps: int = 1
    use_torch_compile: bool = True
    exploit_ratio: float = 0.01
    exploit_noise_scale: float = 0.01
    object_pos_threshold: float = 0.1
    object_rot_threshold: float = 0.3
    nconmax_per_env: int = 256
    njmax_per_env: int = 1024
    temperature: float = 0.1
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
            if type(defaults[key]) is bool:
                if type(value) is not bool:
                    raise ValueError(f"Expected bool: {name}.{key}")
                continue
            minimum = 0 if key in {"seed", "improvement_threshold"} else 1e-15
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < minimum
                or (type(defaults[key]) is int and type(value) is not int)
            ):
                raise ValueError(
                    f"Expected positive {type(defaults[key]).__name__}: {name}.{key}"
                )
        result[name] = values
    if result["mpc"]["num_samples"] < 100:
        raise ValueError("Native exploit sampling requires at least 100 samples")
    if result["ik"]["ref_dt"] != result["mpc"]["ref_dt"]:
        raise ValueError("IK and MPC reference periods must match")
    return result


def validate_clock(options, task):
    """Validate native grids; evaluation publication is an explicit hold mapping."""
    steps = {}
    for name in ("horizon", "knot_dt", "ctrl_dt", "ref_dt"):
        ratio = options[name] / options["sim_dt"]
        steps[name] = round(ratio)
        if steps[name] < 1 or not math.isclose(ratio, steps[name], abs_tol=1e-9):
            raise ValueError(f"mpc.{name} must be a positive multiple of sim_dt")
    if steps["horizon"] % steps["knot_dt"]:
        raise ValueError("MPC horizon must contain an integer number of knots")
    if steps["ctrl_dt"] > steps["horizon"]:
        raise ValueError("MPC execution interval cannot exceed the horizon")


UPSTREAM_REVISION = "44717007de41cbef7565dff7ff9f4453557a2d3d"
PIPELINE = {
    "upstream_revision": UPSTREAM_REVISION,
    "ik_entry": "spider/preprocess/ik.py",
    "mpc_entry": "examples/run_mjwp.py",
    "config_source": "examples/config/default.yaml + Config.use_torch_compile",
    "dataset_override": None,
    "recipe": "paper-ik-ordinary-mpc-generic-h2r",
}
