"""Plan once in SPIDER's private MJWarp world; seal controls for repeated eval."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from .bundle import digest, identity, load_prepared, load_task, write_json


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
    controller = Controller(prepared, task, device, seed)
    # Match the common initial proposal: reference object pose, robot IK, zero
    # velocities and no warm-up. No reference states are imposed after this reset.
    controller.data.qpos[:] = controller.source["qpos"][0]
    controller.data.qvel[:] = 0
    controller.data.ctrl[:] = controller.layout.controls(controller.data.qpos)
    controller.data.time = 0
    mujoco.mj_forward(controller.env.model_cpu, controller.data)
    rows = []
    with (output / "planning.jsonl").open("w") as log:
        for tick in range(0, task["end_tick"], task["control_ticks"]):
            targets, diagnostic = controller.optimize_current(tick, tick * task["dt_s"])
            ctrl = controller.controls[0].detach().cpu().numpy().copy()
            rows.append(
                (
                    tick,
                    list(targets.values()),
                    controller.data.qpos.copy(),
                    controller.data.qvel.copy(),
                    ctrl,
                )
            )
            log.write(json.dumps(diagnostic, allow_nan=False) + "\n")
            log.flush()
            if tick + task["control_ticks"] < task["end_tick"]:
                controller.advance()
            if len(rows) % 100 == 0:
                print(
                    f"Plan control {len(rows)}; tick {tick}/{task['end_tick']}",
                    flush=True,
                )
    ticks, targets, qpos, qvel, ctrl = map(np.asarray, zip(*rows, strict=True))
    np.savez_compressed(
        output / "trajectory_mjwp.npz",
        ticks=ticks,
        time_s=ticks * task["dt_s"],
        joint_targets=targets,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
    )
    for name in ("initial_robot.json", "resolved_config.yaml"):
        shutil.copy2(prepared / name, output / name)
    repo = Path(__file__).resolve().parents[2]
    manifest = {
        "format": "spider-h2r-plan-v1",
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
        "controls": len(rows),
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
    print(plan(args.prepared, args.task, args.output, args.device, args.seed))


if __name__ == "__main__":
    main()
