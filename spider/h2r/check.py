"""Audit an offline plan against evaluator-owned control and rollout records."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import arrays
from .evaluation import METHOD
from .h2r_io import load_task
from .h2r_io.files import read_json, write_json
from .replay import load_plan

ROLLOUT_FORMAT = "h2r-rollout-v2"


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


def check_episode(directory, clock, names, targets, initial, commands, result):
    """Check the recorded prefix without treating an early failure as completion.

    `clock` holds the run's own `control_ticks`, `end_tick` and recording stride.
    """
    ticks, end_tick, stride = clock
    for index, command in enumerate(commands):
        tick = command["tick"]
        if (
            type(tick) is not int
            or index >= len(targets)
            or tick != index * ticks
            or tick > result["ended_tick"]
        ):
            raise ValueError("Missing, repeated or out-of-order control tick")
        if not np.array_equal(
            target_vector(command["joint_targets"], names), targets[index]
        ):
            raise ValueError(f"Submitted command differs from plan at tick {tick}")

    # The rollout records the physical state on the common recording grid only.
    rows = json_lines(directory / "physical.jsonl")
    recorded = [row["tick"] for row in rows]
    if recorded != [i * stride for i in range(len(rows))]:
        raise ValueError("Physical recording is not a prefix of the recording grid")
    last_recorded = recorded[-1] if rows else None
    last_valid = result["last_valid_tick"]
    if result["last_recorded_tick"] != last_recorded or result["end_tick"] != end_tick:
        raise ValueError("Physical recording and termination coverage differ")
    if (
        type(result["ended_tick"]) is not int
        or not 0 <= result["ended_tick"] <= end_tick
        or (last_valid is None) != (last_recorded is None)
        or last_valid is not None
        and not last_recorded <= last_valid <= result["ended_tick"]
    ):
        raise ValueError(
            "Termination precedes the physical recording or exceeds the task"
        )
    if rows and not np.allclose(rows[0]["robot_q"], initial, rtol=0, atol=1e-6):
        raise ValueError("Physical initial robot state differs from the proposal")
    if (directory / "physical.npz").is_file():
        saved = arrays(directory / "physical.npz")
        for key in ("tick", "robot_q"):
            if not np.array_equal(saved[key], np.asarray([row[key] for row in rows])):
                raise ValueError(f"NPZ and JSONL disagree: {key}")

    full = last_valid == end_tick and len(commands) == len(targets)
    if result["complete"] and (not full or result["reason"] != "target_end"):
        raise ValueError("Episode claims completion without full task coverage")
    return {
        "controls_checked": len(commands),
        "physical_frames_checked": len(rows),
        "last_valid_tick": last_valid,
        "control_consistent": True,
        "full_time_coverage": full,
        "complete": result["complete"],
        "reason": result["reason"],
        "coverage": 0.0 if last_valid is None else last_valid / end_tick,
    }


def check(plan, task_path, run, episode=None):
    """Audit all seeds, or one selected seed, in a finalized sim/sim-dr run."""
    run = Path(run)
    task = load_task(task_path)
    binding, data, initial = load_plan(plan, task)
    metadata = read_json(run / "run.json")
    if (
        metadata["format"] != ROLLOUT_FORMAT
        or metadata["mode"] not in {"sim", "sim-dr"}
        or metadata["status"] != "finished"
    ):
        raise ValueError(
            f"Requires a finalized {ROLLOUT_FORMAT} sim or sim-dr recording"
        )
    if metadata["task_id"] != task.task_id:
        raise ValueError("Recording belongs to another task")
    # The run owns its clock: the physics step is an evaluation setting and the
    # control period is the one this client declared at start.
    dt, ticks, end_tick = (
        metadata[key] for key in ("dt_s", "control_ticks", "end_tick")
    )
    if end_tick != round(task.duration_s / dt) or not math.isclose(
        end_tick * dt, task.duration_s, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("Recording does not end at the task duration")
    if type(ticks) is not int or not math.isclose(
        ticks * dt, binding["control_dt_s"], rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("Recording control period differs from the plan")
    recorded_binding = metadata["binding"]
    if (
        recorded_binding["method"] != METHOD
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
        start["task_id"] != task.task_id
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
    names = task.robot.semantics["joint_names"]
    initial = target_vector(initial, names)
    clock = (ticks, end_tick, metadata["record_stride_ticks"])
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
            run / name,
            clock,
            names,
            data["joint_targets"],
            initial,
            commands[name],
            result,
        )
    return {
        "task_id": task.task_id,
        "plan_id": binding["plan_id"],
        "mode": metadata["mode"],
        "evaluation_dt_s": dt,
        "control_ticks": ticks,
        "control_hz": 1 / (dt * ticks),
        "expected_controls": len(data["joint_targets"]),
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
