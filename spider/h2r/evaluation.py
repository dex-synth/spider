"""Deploy a sealed SPIDER fast plan to the independent evaluator."""

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
from h2r_io import client, load_task
from h2r_io.files import write_json

from .replay import Replay, load_plan

METHOD = "spider"


def start_request(hello, task, binding, initial):
    """Check that the evaluator runs the planned task; declare SPIDER's own period."""
    if hello.get("type") != "hello" or hello.get("protocol") != client.PROTOCOL:
        raise ValueError(f"Expected a {client.PROTOCOL} hello")
    if hello["task_id"] != task.task_id:
        raise ValueError("Evaluator runs a different task than the plan")
    semantics = task.robot.semantics
    units = dict(zip(semantics["joint_names"], semantics["joint_units"], strict=True))
    names = hello["joint_names"]
    if (
        len(names) != len(set(names))
        or set(names) != set(units)
        or [units[name] for name in names] != hello["joint_units"]
        or set(initial) != set(names)
        or not np.isfinite(list(initial.values())).all()
    ):
        raise ValueError("Robot joint names, units or initial state differ")
    ids = [episode["episode_id"] for episode in hello["episodes"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Invalid episode identities")
    # Replay reads no observation, so no body state or contact is requested.
    return client.start_request(
        hello,
        method=METHOD,
        controller=binding,
        initial_robot=dict.fromkeys(ids, initial),
        control_dt_s=binding["control_dt_s"],
    )


def run(stream, task, plan, output):
    """Handle independent episode lifecycles; finished slots never issue controls."""
    binding, data, initial = plan
    hello = client.receive(stream)
    request = start_request(hello, task, binding, initial)
    ticks = client.control_ticks(hello, binding["control_dt_s"])
    client.send(stream, request)
    ids = set(request["initial_robot"])
    replays, finished = {}, {}
    started = time.perf_counter()
    with (output / "controller.jsonl").open("w") as log:
        while True:
            message = client.receive(stream)
            if message["type"] == "error":
                raise RuntimeError(f"Evaluator rejected controller: {message}")
            if message["type"] != "frames":
                raise ValueError("Expected frames")
            frames = message["episodes"]
            if (
                len(frames) != len(ids)
                or {f["episode_id"] for f in frames} != ids
                or message["done"] != all(f["done"] for f in frames)
            ):
                raise ValueError("Invalid episode frame set")
            commands = {}
            for frame in frames:
                name = frame["episode_id"]
                if frame["done"]:
                    finished[name] = frame
                    replays.pop(name, None)
                    continue
                if name in finished:
                    raise ValueError("Evaluator reactivated a finished episode")
                if name not in replays:
                    print(f"Initialize {binding['deployment']}: {name}", flush=True)
                    replays[name] = Replay(binding, data, ticks)
                targets, diagnostic = replays[name].control(frame, hello["joint_names"])
                commands[name] = {"tick": frame["tick"], "joint_targets": targets}
                log.write(
                    json.dumps({"episode_id": name, **diagnostic}, allow_nan=False)
                    + "\n"
                )
                log.flush()
            if message["done"]:
                break
            client.send(stream, {"type": "step", "commands": commands})
    result = {
        "task_id": task.task_id,
        "plan_id": binding["plan_id"],
        "control_dt_s": binding["control_dt_s"],
        "evaluation_dt_s": hello["dt_s"],
        "control_ticks": ticks,
        "elapsed_s": time.perf_counter() - started,
        "episodes": finished,
    }
    write_json(output / "controller_result.json", result)
    return result


def main():
    """Replay one sealed plan for all evaluator episodes; CPU only."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "task", "socket", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    task = load_task(args.task)
    plan = load_plan(args.plan, task)  # Reject an unusable plan before connecting.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(args.timeout_s)
            stream.connect(str(args.socket))
            run(stream, task, plan, args.output)
    except Exception as error:
        write_json(
            args.output / "error.json",
            {"type": type(error).__name__, "detail": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
