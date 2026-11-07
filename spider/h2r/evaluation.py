"""Deploy a sealed SPIDER plan, or explicit online MPC, to the evaluator."""

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from .bundle import digest, load_prepared, load_task, read_json, write_json
from .protocol import receive, send


def start_request(hello, task, binding, initial):
    """Validate all task identities and physical semantics before initialization."""
    if hello.get("type") != "hello" or hello.get("protocol") != "h2r-evaluation-v1":
        raise ValueError("Expected h2r-evaluation-v1 hello")
    for key in (
        "task_id",
        "bundle_id",
        "robot_manifest_sha256",
        "robot_name",
        "dt_s",
        "control_ticks",
        "end_tick",
        "object",
        "observation",
        "source_frame_range",
    ):
        if hello[key] != task[key]:
            raise ValueError(f"Evaluator differs from prepared task: {key}")
    if hello.get("environment", {}) != task.get("environment", {}):
        raise ValueError("Evaluator environment differs")
    names = hello["joint_names"]
    units = dict(
        zip(
            task["semantics"]["joint_names"],
            task["semantics"]["joint_units"],
            strict=True,
        )
    )
    if (
        len(names) != len(set(names))
        or set(names) != set(units)
        or [units[name] for name in names] != hello["joint_units"]
        or set(initial) != set(names)
        or not np.isfinite(list(initial.values())).all()
    ):
        raise ValueError("Robot joint names, units or initial state differ")
    ids = [ep["episode_id"] for ep in hello["episodes"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Invalid episode identities")
    return {
        "type": "start",
        "task_id": task["task_id"],
        "binding": {
            "bundle_id": task["bundle_id"],
            "robot_manifest_sha256": task["robot_manifest_sha256"],
            "method": "spider",
            "controller": binding,
        },
        "requirements": {
            "bodies": list(task["object"]["parts"].values()),
            "contacts": False,
            "control_dt_s": task["dt_s"] * task["control_ticks"],
        },
        "initial_robot": dict.fromkeys(ids, initial),
    }


def run(
    stream,
    prepared,
    task,
    binding,
    output,
    device,
    controller_factory,
    binding_file="h2r_binding.json",
):
    """Handle independent episode lifecycles; finished slots never issue controls."""
    hello = receive(stream)
    request = start_request(
        hello, task, binding, read_json(prepared / "initial_robot.json")
    )
    send(stream, request)
    seeds = {ep["episode_id"]: ep["controller_seed"] for ep in hello["episodes"]}
    controllers, finished = {}, {}
    started = time.perf_counter()
    with (output / "controller.jsonl").open("w") as log:
        while True:
            message = receive(stream)
            if message["type"] == "error":
                raise RuntimeError(f"Evaluator rejected controller: {message}")
            if message["type"] != "frames":
                raise ValueError("Expected frames")
            frames = message["episodes"]
            if (
                len(frames) != len(seeds)
                or {f["episode_id"] for f in frames} != set(seeds)
                or message["done"] != all(f["done"] for f in frames)
            ):
                raise ValueError("Invalid episode frame set")
            commands = {}
            for frame in frames:
                name = frame["episode_id"]
                if frame["done"]:
                    finished[name] = frame
                    controllers.pop(name, None)
                    continue
                if name in finished:
                    raise ValueError("Evaluator reactivated a finished episode")
                if name not in controllers:
                    print(
                        f"Initialize {binding.get('deployment', 'online-mpc')}: {name}",
                        flush=True,
                    )
                    controllers[name] = controller_factory(
                        prepared, task, device, seeds[name]
                    )
                targets, diagnostic = controllers[name].control(
                    frame, hello["joint_names"]
                )
                commands[name] = {"tick": frame["tick"], "joint_targets": targets}
                log.write(
                    json.dumps({"episode_id": name, **diagnostic}, allow_nan=False)
                    + "\n"
                )
                log.flush()
            if message["done"]:
                break
            send(stream, {"type": "step", "commands": commands})
    result = {
        "task_id": task["task_id"],
        "binding_sha256": digest(prepared / binding_file),
        "elapsed_s": time.perf_counter() - started,
        "episodes": finished,
    }
    write_json(output / "controller_result.json", result)
    return result


def main():
    """Run one persistent controller process for all evaluator episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--plan", type=Path, help="Sealed offline plan, reused for every eval seed"
    )
    source.add_argument("--prepared", type=Path, help="Legacy online MPC deployment")
    for key in ("task", "socket", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    task = load_task(args.task)
    if args.plan is not None:
        from .replay import Replay, load_plan

        binding, data = load_plan(args.plan, task)
        prepared, binding_file = args.plan, "plan.json"

        def factory(prepared, task, device, seed):
            return Replay(task, binding, data)
    else:
        from .controller import Controller

        prepared, binding_file = args.prepared, "h2r_binding.json"
        binding = load_prepared(prepared, task)
        factory = Controller
    args.output.mkdir(parents=True, exist_ok=False)

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(args.timeout_s)
            stream.connect(str(args.socket))
            run(
                stream,
                prepared,
                task,
                binding,
                args.output,
                args.device,
                factory,
                binding_file=binding_file,
            )
    except Exception as error:
        write_json(
            args.output / "error.json",
            {"type": type(error).__name__, "detail": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
