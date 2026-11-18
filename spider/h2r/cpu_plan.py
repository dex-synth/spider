"""Plan once with upstream's CPU workflow on the task scene; seal the controls.

The loop, the recovery (planning a frame again, going back when the object slips)
and the pass verdict are upstream's `examples/run_cpu.py` and
`spider/postprocess/evaluate_cpu.py`, unchanged. This entry supplies the private
world of the benchmark task (`cpu_scene`, `cpu_world`) and seals the position
target of every control frame for an offline replay. No GPU is needed.
"""

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .artifacts import INITIAL_POSE_ONLY, REFERENCE_V2D, arrays, load_prepared
from .h2r_io import load_task
from .h2r_io.files import read_json, sha256, write_json
from .plan import seal
from .replay import control_count

REPO = Path(__file__).resolve().parents[2]
# Upstream's recipe, at its upstream path, and the H2R adaptation next to this file.
RECIPE = REPO / "examples" / "config" / "cpu.yaml"
ADAPTATION = Path(__file__).with_name("config") / "h2r_cpu.yaml"
UPSTREAM_REVISION = "3d132978dc73c7087dacb4455baf9b1d5a060f34"
CONTROLS_FILE = "controls.npz"
# Keys of upstream's recipe that are no algorithm settings: the task, paths, the
# device, the viewer and the output. They belong to the benchmark or to this entry.
NOT_SETTINGS = {
    "defaults",
    "hydra",
    "recovery",
    "simulator",
    "device",
    "dataset_dir",
    "dataset_name",
    "robot_type",
    "embodiment_type",
    "task",
    "data_id",
    "model_path",
    "data_path",
    "output_dir",
    "seed",
    "num_threads",
    "sim_dt",
    "max_sim_steps",
    "show_viewer",
    "viewer",
    "autostart",
    "wait_on_finish",
    "save_video",
    "save_info",
    "num_trace_uniform_samples",
    "num_trace_topk_samples",
}
# A sealed plan must replay to the planned states; see `verify_replay`.
REPLAY_TOLERANCE = 1e-9


def runner():
    """`examples/run_cpu.py` as a module (it is a script, not a package)."""
    if "run_cpu" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "run_cpu", REPO / "examples" / "run_cpu.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["run_cpu"] = module  # dataclasses look their module up
        spec.loader.exec_module(module)
    return sys.modules["run_cpu"]


def _update(values, overrides, name):
    """Apply explicit overrides to known keys of the same kind only."""
    if not isinstance(overrides, dict) or set(overrides) - values.keys():
        unknown = sorted(set(overrides) - values.keys()) if overrides else overrides
        raise ValueError(f"Unknown {name} options: {unknown}")
    for key, value in overrides.items():
        if isinstance(values[key], dict):
            _update(values[key], value, f"{name}.{key}")
        elif type(value) is bool or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid {name}.{key}: {value!r}")
        elif isinstance(values[key], int) and type(value) is not int:
            raise ValueError(f"Expected an integer: {name}.{key}")
        else:
            values[key] = value


def settings(path=None):
    """Upstream's recipe, the H2R adaptation, then explicit overrides.

    Returns `mpc` (the algorithm fields of upstream's `cpu.yaml`), `recovery`
    (its recovery section) and `h2r` (the adaptation). `path` may override keys
    of these three sections; a lower budget is for debugging, not a recipe.
    """
    recipe = yaml.safe_load(RECIPE.read_text())
    result = {
        "mpc": {k: v for k, v in recipe.items() if k not in NOT_SETTINGS},
        "recovery": dict(recipe["recovery"]),
        "h2r": yaml.safe_load(ADAPTATION.read_text()),
    }
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    if not isinstance(raw, dict) or set(raw) - result.keys():
        raise ValueError("Settings must contain only mpc, recovery and h2r mappings")
    for name, overrides in raw.items():
        _update(result[name], overrides, name)
    adaptation = result["h2r"]
    if (
        adaptation["frame_dt"] <= 0
        or adaptation["substeps"] < 1
        or not 0 <= adaptation["action_ema"] < 1
        or min(adaptation["action_clip"].values()) <= 0
    ):
        raise ValueError(f"Invalid H2R adaptation: {adaptation}")
    return result


def planner_config(options, output, seed):
    """Upstream's `Config` and `Recovery` for a run without viewer or video."""
    values = copy.deepcopy(yaml.safe_load(RECIPE.read_text()))
    values.update(options["mpc"])
    values.update(
        recovery=dict(options["recovery"]),
        simulator="cpu",
        device="cpu",
        model_path="",
        data_path="",
        output_dir=str(output),
        seed=seed,
        show_viewer=False,
        save_video=False,
        save_info=True,
    )
    from omegaconf import OmegaConf

    config, recovery, _, _ = runner().parse(OmegaConf.create(values))
    return config, recovery


def verify_replay(model, layout, initial_qpos, targets, substeps, recorded):
    """Replay the sealed targets in one private world; return the largest deviation.

    The evaluation receives nothing but these targets. If holding each for one
    control frame from the initial state does not reproduce the states the planner
    recorded, the filter state or the bookkeeping of a return is off.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    worst = 0.0
    for target, expected in zip(targets, recorded, strict=True):
        data.ctrl[layout.actuators] = target
        mujoco.mj_step(model, data, nstep=substeps)
        worst = max(worst, float(np.abs(data.qpos - expected).max()))
    return worst


def plan(prepared, task_path, output, seed, threads=0, config=None):
    """Run upstream's receding-horizon pass in the private world and seal it."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Planning seed must be a nonnegative integer")
    if type(threads) is not int or threads < 0:
        raise ValueError("Threads must be a nonnegative integer (0: one per CPU)")
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    task = load_task(task_path)
    if output.is_relative_to(prepared) or output.is_relative_to(task.path):
        raise ValueError("Plan output must be outside its inputs")
    binding = load_prepared(prepared, task, REFERENCE_V2D)
    options = settings(config)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump({"recipe": "cpu", **options}, sort_keys=False)
    )
    adaptation = options["h2r"]
    control_dt, substeps = float(adaptation["frame_dt"]), int(adaptation["substeps"])
    names = task.robot.semantics["joint_names"]
    sealing = {
        "backend": "mujoco-cpu",
        "controls_file": CONTROLS_FILE,
        "copied": ("initial_robot.json", "initialization.json"),
    }
    if not read_json(prepared / "initialization.json")["accepted"]:
        # The initial pose penetrates the scene: a failure of the method. The plan
        # proposes that pose so the evaluation records the failure, and holds no
        # controls, since the evaluation rejects such a start.
        print("Initial pose fails SPIDER's pre-check: no planning", flush=True)
        return seal(
            output,
            task,
            prepared,
            binding,
            seed,
            control_dt,
            np.zeros((0, len(names))),
            {"deployment": INITIAL_POSE_ONLY, "recipe": "cpu"},
            **sealing,
        )
    from . import cpu_world
    from .cpu_scene import private_model, tables

    model, layout, damping_change = private_model(task, control_dt / substeps)
    scene, initial_qpos = tables(
        model, layout, task, arrays(prepared / "trajectory_kinematic.npz"), adaptation
    )
    np.savez_compressed(output / "scene_tables.npz", **scene)
    planner, recovery = planner_config(options, output, seed)
    env = cpu_world.build_env(planner, model, scene, initial_qpos, threads)
    trajectory = runner().track(
        planner, recovery, cpu_world, env, cpu_world.reference_data(planner, env)
    )
    # `track` has written upstream's own outputs: trajectory_cpu.npz, metrics.json.
    metrics = read_json(output / "metrics.json")
    count = control_count(task.duration_s, control_dt)
    executed = len(trajectory["qpos"])
    if int(trajectory["stopped_early"]) or executed != count:
        # Upstream stops a run that has lost the object with no tries left. The
        # motion up to there is a diagnostic, never a plan for the whole task.
        write_json(
            output / "failure.json",
            {
                "reason": "upstream-cpu-stopped-early",
                "executed_controls": executed,
                "required_controls": count,
                "task_id": task.task_id,
                "task_duration_s": task.duration_s,
                "backtracks": int(metrics["backtracks"]),
            },
        )
        raise RuntimeError(
            "Upstream CPU planning stopped early; full task denominator is unchanged"
        )
    targets = np.asarray(trajectory["joint_target"], dtype=float)
    deviation = verify_replay(
        model, layout, initial_qpos, targets, substeps, trajectory["qpos"]
    )
    if not deviation <= REPLAY_TOLERANCE:
        write_json(
            output / "failure.json",
            {"reason": "sealed-controls-do-not-replay", "max_abs_qpos": deviation},
        )
        raise RuntimeError(
            f"Sealed targets do not reproduce the planned states ({deviation:.3e})"
        )
    record = {
        "deployment": "offline-control-replay",
        "recipe": "cpu",
        "clock": {
            "replan_dt_s": planner.ctrl_dt,
            "private_physics_dt_s": control_dt / substeps,
            "physics_substeps": substeps,
            "reference_rows": "end of each control frame; initial state apart",
        },
        "planner": {
            "upstream_revision": UPSTREAM_REVISION,
            "mujoco_version": mujoco.__version__,
            "threads": env.batch.nthread,
            "actuator_damping_reset_from_task": damping_change,
        },
        # Upstream's own verdict in the private world: recorded, never the score.
        "native": {
            "success": int(metrics["success"]),
            "success_object": int(metrics["success_object"]),
            "frac_ok": metrics["frac_ok"],
            "obj_pos_cm": metrics["obj_pos_cm"],
            "obj_rot_deg": metrics["obj_rot_deg"],
            "backtracks": int(metrics["backtracks"]),
            "metrics_sha256": sha256(output / "metrics.json"),
        },
        "replay_check_max_abs_qpos": deviation,
        "plan_elapsed_s": time.perf_counter() - started,
    }
    return seal(
        output, task, prepared, binding, seed, control_dt, targets, record, **sealing
    )


def main():
    """Create one complete plan for the selected sequence and planning seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "task", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads", type=int, default=0, help="batch threads (0: one per CPU)"
    )
    parser.add_argument("--config", type=Path, help="overrides of mpc, recovery, h2r")
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        result = plan(
            args.prepared, args.task, args.output, args.seed, args.threads, args.config
        )
        manifest = read_json(result / "plan.json")
        print(json.dumps({k: manifest.get(k) for k in ("deployment", "native")}))
        print(result)
    except Exception as error:
        if not existed and args.output.is_dir():
            write_json(
                args.output / "error.json",
                {"type": type(error).__name__, "detail": str(error)},
            )
        raise


if __name__ == "__main__":
    main()
