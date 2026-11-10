"""Explicit H2R adaptation settings, independent of native dataset paths."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class IKSettings:
    """Defaults of upstream preprocess/ik_fast.py."""

    sim_dt: float = 0.005
    ref_dt: float = 0.02
    wrist_pos_cost: float = 0.3
    wrist_ori_cost: float = 3.0
    finger_pos_cost: float = 10.0
    posture_cost: float = 0.01
    wrist_init_steps: int = 200
    finger_init_steps: int = 300
    average_frame_size: int = 3


@dataclass
class MPCSettings:
    """Fast runner base defaults from upstream examples/config/default.yaml."""

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
    nconmax_per_env: int = 120
    njmax_per_env: int = 350
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

    max_local_retries: int = 3
    max_revert_depth: int = 3
    sanity_check_seconds: float = 3.0
    sanity_check_pos_thresh: float = 0.01
    sanity_check_rot_thresh: float = 0.3
    num_dr: int = 1
    num_dyn: int = 1
    terminate_resample: bool = False
    contact_guidance: bool = False
    gibbs_sampling: bool = False


CONFIG_DIR = Path(__file__).with_name("config")
DATASET_RECIPES = {
    "arctic": "arcticv2",
    "arcticv2": "arcticv2",
    "oakink2": "oakinkv2",
    "oakinkv2": "oakinkv2",
    "oakink": "oakink",
    "gigahand": "gigahand",
}


def recipe(dataset):
    """Select only the matching upstream fast override, never borrow a recipe."""
    name = DATASET_RECIPES.get(dataset)
    return f"override/{name}_fast.yaml" if name else None


def settings(path=None, *, dataset=None, ref_dt=None):
    """Resolve pinned YAMLs, then explicit overrides; retain dataset reference rate."""
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    if not isinstance(raw, dict) or set(raw) - {"ik", "mpc"}:
        raise ValueError("Settings must contain only ik and mpc mappings")
    native = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text())
    if recipe(dataset):
        native.update(yaml.safe_load((CONFIG_DIR / recipe(dataset)).read_text()))
    result = {}
    for name, cls in (("ik", IKSettings), ("mpc", MPCSettings)):
        defaults = asdict(cls())
        values = defaults.copy()
        if name == "mpc":
            values.update({k: v for k, v in native.items() if k in defaults})
        if ref_dt is not None:
            values["ref_dt"] = ref_dt
        overrides = raw.get(name, {})
        if not isinstance(overrides, dict) or set(overrides) - defaults.keys():
            raise ValueError(f"Unknown {name} options")
        values.update(overrides)
        for key, value in values.items():
            if type(defaults[key]) is bool:
                if type(value) is not bool:
                    raise ValueError(f"Expected bool: {name}.{key}")
                continue
            minimum = (
                0
                if key
                in {"max_local_retries", "max_revert_depth", "sanity_check_seconds"}
                else 1e-15
            )
            if key == "improvement_threshold":
                minimum = -math.inf  # Fast optimizer does not use this field.
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < minimum
                or (type(defaults[key]) is int and type(value) is not int)
            ):
                raise ValueError(f"Invalid {name}.{key}: {value!r}")
        result[name] = values
    mpc = result["mpc"]
    if mpc["num_samples"] < 100:
        raise ValueError("Native exploit sampling requires at least 100 samples")
    if mpc["num_dr"] != 1 or mpc["num_dyn"] != 1:
        raise ValueError("H2R supports the nominal num_dr=num_dyn=1 recipe only")
    if mpc["contact_guidance"] or mpc["gibbs_sampling"]:
        raise ValueError("Upstream fast does not support contact guidance or Gibbs")
    if result["ik"]["ref_dt"] != mpc["ref_dt"]:
        raise ValueError("IK and MPC reference periods must match")
    return result


def validate_clock(options, task):
    """Native action grids must be integral; dataset sampling need not be."""
    steps = {}
    for name in ("horizon", "knot_dt", "ctrl_dt"):
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
    "ik_entry": "spider/preprocess/ik_fast.py",
    "mpc_entry": "examples/run_mjwp_fast.py",
    "config_source": "examples/config/default.yaml + dataset fast override + Config",
    "recipe": "fast-mink-best-sample-h2r",
    "rotation_processing": "sign-continuous-mean-nlerp-v1",
    "num_dr_1": "nominal; upstream single-world nonzero offsets intentionally disabled",
}
