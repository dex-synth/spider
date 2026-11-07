"""Audit nominal eval state synchronization and probe native MPC on saved frames."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

from .bundle import load_prepared, load_task, read_json, write_json
from .controller import Controller
from .model import Layout


def frames(path, task):
    """Read the actual pre-control states from a nominal physical recording."""
    result = {}
    with path.open() as stream:
        for line in stream:
            frame = json.loads(line)
            tick = frame["tick"]
            if tick % task["control_ticks"] or tick >= task["end_tick"]:
                continue
            frame["bodies"] = {
                body: frame["object_bodies"][part]
                for part, body in task["object"]["parts"].items()
            }
            result[tick] = frame
    return result


def check(prepared, task_path, run, controller_log, episode, device, steps):
    """Compare saved planner initial states, then inspect GPU state before search."""
    task = load_task(task_path)
    binding = load_prepared(prepared, task)
    metadata = read_json(run / "run.json")
    if metadata["task_id"] != task["task_id"] or metadata["mode"] != "sim":
        raise ValueError(
            "Requires a matching nominal run; noisy observations are not truth"
        )
    observations = frames(run / episode / "physical.jsonl", task)
    model = mujoco.MjModel.from_xml_path(str(prepared / binding["planning_scene"]))
    layout, data = Layout(model, task), mujoco.MjData(model)
    names = layout.names
    count, max_error = 0, 0.0
    with controller_log.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["episode_id"] != episode:
                continue
            layout.write_observation(data, observations[row["tick"]], names)
            for field in ("qpos", "qvel"):
                error = float(
                    np.max(np.abs(getattr(data, field) - row[f"planning_{field}"]))
                )
                max_error = max(max_error, error)
                np.testing.assert_allclose(
                    getattr(data, field), row[f"planning_{field}"], atol=1e-9, rtol=0
                )
            count += 1
    if not count or steps < 1 or len(observations) < steps:
        raise ValueError("Insufficient controller or physical frames")
    controller = Controller(prepared, task, device, seed=17)
    optimize = controller.optimize
    previous = None
    gpu_error = 0.0

    def audit(config, env, controls, reference):
        nonlocal gpu_error
        # This callback runs immediately before the unchanged native optimizer.
        for field in ("qpos", "qvel", "ctrl"):
            actual = wp.to_torch(getattr(env.data_wp, field)).cpu().numpy()
            expected = np.broadcast_to(getattr(controller.data, field), actual.shape)
            gpu_error = max(gpu_error, float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            wp.to_torch(env.data_wp.time).cpu(), controller.data.time, atol=1e-6
        )
        if previous is not None:
            torch.testing.assert_close(controls[:-1], previous[1:], rtol=0, atol=0)
        return optimize(config, env, controls, reference)

    controller.optimize = audit
    diagnostics = []
    for frame in list(observations.values())[:steps]:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(controller.device_index)
        targets, diagnostic = controller.control(frame, names)
        if set(targets) != set(names) or not np.isfinite(list(targets.values())).all():
            raise AssertionError("Invalid controller output")
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.cuda.get_rng_state(controller.device_index), cuda_rng, rtol=0, atol=0
        )
        previous = controller.controls.clone()
        diagnostics.append(
            {"tick": frame["tick"], "compute_s": diagnostic["compute_s"]}
        )
        try:
            controller.control(frame, names)
        except ValueError:
            pass
        else:
            raise AssertionError("Repeated tick was accepted")
    return {
        "task_id": task["task_id"],
        "saved_frames_checked": count,
        "saved_state_max_error": max_error,
        "gpu_state_max_error": gpu_error,
        "probe_steps": diagnostics,
        "rng_preserved": True,
        "rolling_plan_checked": steps > 1,
        "stale_tick_rejected": True,
    }


def main():
    """Run diagnostics without altering the recorded evaluation or its scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "task", "run", "controller-log", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--episode", default="seed-42")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    result = check(
        args.prepared,
        args.task,
        args.run,
        args.controller_log,
        args.episode,
        args.device,
        args.steps,
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
