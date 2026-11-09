"""Audit a v4 offline plan against evaluator-owned control and rollout records."""

import argparse
import json
from pathlib import Path

import numpy as np

from .bundle import arrays, load_task, read_json, write_json
from .replay import load_plan


def json_lines(path):
    """Read finalized records; an incomplete JSON line is an error."""
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def target_vector(targets, names):
    """Resolve the complete named command in task joint order."""
    if not isinstance(targets, dict) or set(targets) != set(names):
        raise ValueError("Joint targets must name every task joint exactly once")
    values = np.asarray([targets[name] for name in names], dtype=float)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("Invalid joint targets")
    return values


def check_episode(directory, task, data, initial, commands, result):
    """Check the recorded prefix without treating an early failure as completion."""
    names = task["semantics"]["joint_names"]
    for index, command in enumerate(commands):
        tick = command["tick"]
        if (
            type(tick) is not int
            or index >= len(data["ticks"])
            or tick != data["ticks"][index]
            or tick > result["ended_tick"]
        ):
            raise ValueError("Missing, repeated or out-of-order control tick")
        if not np.array_equal(
            target_vector(command["joint_targets"], names), data["joint_targets"][index]
        ):
            raise ValueError(f"Submitted command differs from plan at tick {tick}")

    rows = json_lines(directory / "physical.jsonl")
    last_tick = len(rows) - 1 if rows else None
    if result["last_valid_tick"] != last_tick or len(rows) > task["end_tick"] + 1:
        raise ValueError("Physical recording and termination coverage differ")
    if result["end_tick"] != task["end_tick"]:
        raise ValueError("Termination changed the task endpoint")
    if (
        type(result["ended_tick"]) is not int
        or not 0 <= result["ended_tick"] <= task["end_tick"]
        or last_tick is not None
        and last_tick > result["ended_tick"]
    ):
        raise ValueError(
            "Termination precedes the physical recording or exceeds the task"
        )
    for tick, row in enumerate(rows):
        if (
            type(row["tick"]) is not int
            or row["tick"] != tick
            or not np.isclose(row["time_s"], tick * task["dt_s"], rtol=0, atol=1e-9)
        ):
            raise ValueError("Physical recording is not a contiguous task-time prefix")
        # A target submitted at k first appears in the post-step record k+1.
        index = (tick - 1) // task["control_ticks"]
        if tick and index >= len(commands):
            raise ValueError("Physical record has no corresponding submitted command")
        expected = data["joint_targets"][index] if tick else initial
        if not np.array_equal(row["submitted_target"], expected):
            raise ValueError(f"Recorded submitted target differs at tick {tick}")
    if rows and (
        not np.allclose(rows[0]["robot_q"], initial, rtol=0, atol=1e-6)
        or not np.array_equal(rows[0]["robot_dq"], np.zeros(len(names)))
    ):
        raise ValueError("Physical initial robot state differs from the proposal")
    if (directory / "physical.npz").is_file():
        saved = arrays(directory / "physical.npz")
        for key in ("tick", "time_s", "submitted_target"):
            if not np.array_equal(saved[key], np.asarray([row[key] for row in rows])):
                raise ValueError(f"NPZ and JSONL disagree: {key}")

    full = last_tick == task["end_tick"] and len(commands) == len(data["ticks"])
    if result["complete"] and (not full or result["reason"] != "target_end"):
        raise ValueError("Episode claims completion without full task coverage")
    return {
        "controls_checked": len(commands),
        "physical_frames_checked": len(rows),
        "last_valid_tick": last_tick,
        "control_consistent": True,
        "full_time_coverage": full,
        "complete": result["complete"],
        "reason": result["reason"],
        "coverage": 0.0 if last_tick is None else last_tick / task["end_tick"],
    }


def check(plan, task_path, run, episode=None):
    """Audit all seeds, or one selected seed, in a finalized sim/sim-dr run."""
    plan, run = Path(plan), Path(run)
    task = load_task(task_path)
    binding, data = load_plan(plan, task)
    metadata = read_json(run / "run.json")
    if (
        metadata["format"] != "h2r-rollout-v1"
        or metadata["mode"] not in {"sim", "sim-dr"}
        or metadata["status"] != "finished"
    ):
        raise ValueError("Requires a finalized sim or sim-dr recording")
    for key in ("task_id", "dt_s", "control_ticks", "end_tick"):
        if metadata[key] != task[key]:
            raise ValueError(f"Recording differs from task: {key}")
    recorded_binding = metadata["binding"]
    if (
        recorded_binding["method"] != "spider"
        or recorded_binding["controller"] != binding
    ):
        raise ValueError("Recording is bound to a different SPIDER plan")
    ids = [f"seed-{seed}" for seed in metadata["seeds"]]
    if not ids or len(set(ids)) != len(ids) or set(metadata["episodes"]) != set(ids):
        raise ValueError("Invalid recording episode identities")
    if episode is not None and episode not in ids:
        raise ValueError(f"Episode is absent from recording: {episode}")
    events = json_lines(run / "events.jsonl")
    if not events or events[0]["type"] != "start":
        raise ValueError("Missing evaluator initialization event")
    start, *steps = events
    if (
        start["task_id"] != task["task_id"]
        or start["binding"] != recorded_binding
        or set(start["initial_robot"]) != set(ids)
    ):
        raise ValueError("Initialization event differs from recording")
    commands = {name: [] for name in ids}
    for step in steps:
        if step["type"] != "step" or not set(step["commands"]) <= set(ids):
            raise ValueError("Unexpected evaluator event")
        for name, command in step["commands"].items():
            commands[name].append(command)
    names = task["semantics"]["joint_names"]
    initial = target_vector(read_json(plan / "initial_robot.json"), names)
    reports = {}
    for name in [episode] if episode is not None else ids:
        if not np.array_equal(
            target_vector(start["initial_robot"][name], names), initial
        ):
            raise ValueError(f"Initial proposal differs from plan: {name}")
        result = read_json(run / name / "result.json")
        if result != metadata["episodes"][name]:
            raise ValueError(f"Episode result differs from run manifest: {name}")
        reports[name] = check_episode(
            run / name, task, data, initial, commands[name], result
        )
    return {
        "task_id": task["task_id"],
        "plan_id": binding["plan_id"],
        "mode": metadata["mode"],
        "control_hz": 1 / (task["dt_s"] * task["control_ticks"]),
        "expected_controls": len(data["ticks"]),
        "episodes": reports,
    }


def main():
    """Check offline deployment on CPU without loading an MPC or simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "task", "run", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--episode", help="Episode ID; default checks every seed")
    args = parser.parse_args()
    report = check(args.plan, args.task, args.run, args.episode)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
