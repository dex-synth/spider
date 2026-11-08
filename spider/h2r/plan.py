"""Plan once in SPIDER's private MJWarp world; seal controls for repeated eval."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .bundle import digest, identity, load_prepared, load_task, write_json
from .fast_loop import run_fast
from .reference import hold_indices


def plan(prepared, task_path, output, device, seed):
    """Run one native receding-horizon pass, without an evaluation connection."""
    from .controller import Controller

    if type(seed) is not int or seed < 0:
        raise ValueError("Planning seed must be a nonnegative integer")
    prepared, task_path, output = map(Path.resolve, (prepared, task_path, output))
    if output.is_relative_to(prepared) or output.is_relative_to(task_path):
        raise ValueError("Plan output must be outside its inputs")
    task = load_task(task_path)
    binding = load_prepared(prepared, task)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    controller = Controller(prepared, task_path, device, seed)
    native_dt = controller.config.sim_dt
    count = int(np.ceil(task["end_tick"] * task["dt_s"] / native_dt))
    with (output / "planning.jsonl").open("w") as log:

        def report(diagnostic):
            log.write(json.dumps(diagnostic, allow_nan=False) + "\n")
            log.flush()
            print(
                f"Plan step {diagnostic['native_step']}/{count}; "
                f"iterations {diagnostic['opt_steps']}; "
                f"success {diagnostic['succeeded']}; "
                f"boost {diagnostic['boost_level']}",
                flush=True,
            )

        complete = run_fast(controller, count, report)
    segments = controller.segments
    steps = np.concatenate(
        [part["step"] + np.arange(len(part["ctrl"])) for part in segments]
    )
    qpos, qvel, ctrl = [
        np.concatenate([part[key] for part in segments])
        for key in ("qpos", "qvel", "ctrl")
    ]
    # Native unrecoverable failure still replays its last best candidate after
    # truncation. Preserve that diagnostic, but never seal it as a complete task.
    if (
        not complete
        or len(steps) < count
        or not np.array_equal(steps[:count], np.arange(count))
    ):
        np.savez_compressed(
            output / "trajectory_failed.npz",
            native_steps=steps,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
        )
        write_json(
            output / "failure.json",
            {
                "reason": "upstream-fast-unrecoverable",
                "planned_steps": steps.tolist(),
                "required_native_steps": count,
                "task_id": task["task_id"],
                "task_end_tick": task["end_tick"],
            },
        )
        raise RuntimeError(
            "Upstream fast planning failed; full task denominator is unchanged"
        )
    qpos, qvel, ctrl = qpos[:count], qvel[:count], ctrl[:count]
    targets = np.asarray(
        [list(controller.layout.targets(value).values()) for value in ctrl]
    )
    native_times = np.arange(count) * native_dt
    ticks = np.arange(0, task["end_tick"], task["control_ticks"], dtype=np.int64)
    times = ticks * task["dt_s"]
    indices = hold_indices(native_times, times)
    np.savez_compressed(
        output / "trajectory_mjwp.npz",
        ticks=ticks,
        time_s=times,
        joint_targets=targets[indices],
        qpos=qpos[indices],
        qvel=qvel[indices],
        ctrl=ctrl[indices],
        native_time_s=native_times,
        native_joint_targets=targets,
        native_qpos=qpos,
        native_state_time_s=native_times + native_dt,
        native_qvel=qvel,
        native_ctrl=ctrl,
        source_time_s=native_times[indices],
    )
    for name in ("initial_robot.json", "resolved_config.yaml"):
        shutil.copy2(prepared / name, output / name)
    repo = Path(__file__).resolve().parents[2]
    manifest = {
        "format": "spider-h2r-plan-v4",
        "deployment": "offline-control-replay",
        "planning_backend": "mjwarp",
        "planning_seed": seed,
        "method_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "method_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, text=True
            )
        ),
        "preparation_binding_sha256": digest(prepared / "h2r_binding.json"),
        "preparation": binding,
        "joint_names": controller.layout.names,
        "joint_units": task["semantics"]["joint_units"],
        "control_ticks": task["control_ticks"],
        "end_tick": task["end_tick"],
        "dt_s": task["dt_s"],
        "controls": len(ticks),
        "native_controls": count,
        "clock_mapping": {
            "native_action_dt_s": native_dt,
            "replan_dt_s": controller.config.ctrl_dt,
            "private_physics_dt_s": native_dt / controller.config.physics_substeps,
            "publication_dt_s": task["dt_s"] * task["control_ticks"],
            "mapping": "zero-order-hold-at-common-ticks",
            "planning_state_times": "source_time_s + native_action_dt_s (post-step diagnostic)",
        },
        "plan_elapsed_s": time.perf_counter() - started,
        **{
            k: task[k]
            for k in ("task_id", "bundle_id", "scene_id", "robot_manifest_sha256")
        },
        "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    manifest["plan_id"] = identity(manifest)
    write_json(output / "plan.json", manifest)
    return output


def main():
    """Create one complete plan for the selected sequence and planning seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "task", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        print(plan(args.prepared, args.task, args.output, args.device, args.seed))
    except Exception as error:
        if not existed and args.output.is_dir():
            write_json(
                args.output / "error.json",
                {"type": type(error).__name__, "detail": str(error)},
            )
        raise


if __name__ == "__main__":
    main()
