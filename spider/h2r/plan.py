"""Plan once in SPIDER's private MJWarp world; seal controls for repeated eval."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .artifacts import PLAN_FORMAT, load_prepared, task_identity
from .fast_loop import run_fast
from .h2r_io import load_task
from .h2r_io.files import content_hash, sha256, write_json
from .replay import control_count


def plan(prepared, task_path, output, device, seed):
    """Run one native receding-horizon pass, without an evaluation connection."""
    from .controller import Controller

    if type(seed) is not int or seed < 0:
        raise ValueError("Planning seed must be a nonnegative integer")
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    task = load_task(task_path)
    if output.is_relative_to(prepared) or output.is_relative_to(task.path):
        raise ValueError("Plan output must be outside its inputs")
    binding = load_prepared(prepared, task)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    controller = Controller(prepared, task, device, seed)
    # The plan is published on SPIDER's native action clock, not on evaluation ticks.
    control_dt = float(controller.config.sim_dt)
    count = control_count(task.duration_s, control_dt)
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
                "task_id": task.task_id,
                "task_duration_s": task.duration_s,
            },
        )
        raise RuntimeError(
            "Upstream fast planning failed; full task denominator is unchanged"
        )
    qpos, qvel, ctrl = qpos[:count], qvel[:count], ctrl[:count]
    targets = np.asarray(
        [list(controller.layout.targets(value).values()) for value in ctrl]
    )
    times = np.arange(count) * control_dt
    np.savez_compressed(
        output / "trajectory_mjwp.npz",
        time_s=times,
        joint_targets=targets,
        ctrl=ctrl,
        # Private planning states after each control: diagnostics, never replayed.
        state_time_s=times + control_dt,
        qpos=qpos,
        qvel=qvel,
    )
    for name in ("initial_robot.json", "resolved_config.yaml"):
        shutil.copy2(prepared / name, output / name)
    repo = Path(__file__).resolve().parents[2]
    manifest = {
        "format": PLAN_FORMAT,
        **task_identity(task),
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
        "preparation_binding_sha256": sha256(prepared / "h2r_binding.json"),
        "preparation": binding,
        "joint_names": controller.layout.names,
        "joint_units": task.robot.semantics["joint_units"],
        "control_dt_s": control_dt,
        "controls": count,
        "clock": {
            "replan_dt_s": controller.config.ctrl_dt,
            "private_physics_dt_s": control_dt / controller.config.physics_substeps,
            "planning_state_times": "time_s + control_dt_s (post-step diagnostic)",
        },
        "plan_elapsed_s": time.perf_counter() - started,
        "files": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    manifest["plan_id"] = content_hash(manifest)
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
