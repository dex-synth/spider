"""Persistent SPIDER MPC process for the explicit Leap/articulated task adaptation."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

from spider.h2r_controller import ExternalStateController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.socket.exists():
        raise FileExistsError(args.socket)
    controller = ExternalStateController(args.prepared, args.device, args.samples, args.iterations, args.seed)
    description = {"protocol": "h2r-state-control-v1", "physics_dt": 1/120,
        "recipe": controller.meta["recipe"], "replan_decimation": controller.cfg.ctrl_steps,
        "preparation": {key: controller.meta.get(key) for key in ("ik_collision_avoidance", "max_ik_step_norm")},
        "preparation_sha256": hashlib.sha256((args.prepared / "input.json").read_bytes()).hexdigest(),
        "planning_scene_sha256": hashlib.sha256(Path(controller.cfg.model_path).read_bytes()).hexdigest(),
        "reference_duration_s": controller.meta["duration_s"],
        "task_reference_sha256": controller.meta["common_reference_sha256"],
        "reference_sha256": hashlib.sha256((args.prepared / "reference.npz").read_bytes()).hexdigest(),
        "method_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "method_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "joint_names": {"robot": controller.joint_names, "object": ["scissors_joint"]},
        "samples": args.samples, "iterations": args.iterations, "horizon_s": controller.cfg.horizon,
        "internal_physics_dt": controller.cfg.sim_dt / controller.cfg.physics_substeps,
        "future_padding": "hold final pose/control, zero velocity inside planner only"}
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    bound = False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(args.socket)); bound = True
            args.socket.chmod(0o600); server.listen(1)
            print(f"Controller ready: {args.socket}", flush=True)
            connection, _ = server.accept()
            with connection, connection.makefile("rwb") as stream:
                while line := stream.readline(1024*1024+1):
                    if len(line) > 1024*1024:
                        raise ValueError("Request exceeds 1 MiB")
                    request = json.loads(line)
                    start = time.perf_counter()
                    if request["op"] == "describe":
                        response = description.copy()
                    elif request["op"] == "reset":
                        controller.reset(request["state"], request.get("seed", args.seed))
                        response = {"reset": "method_internal_state"}
                    elif request["op"] == "control":
                        response = controller.control(request["state"], request["tick"], request["time_s"])
                    elif request["op"] == "close":
                        stream.write(b'{"closed":true}\n'); stream.flush(); break
                    else:
                        raise ValueError("Unsupported operation")
                    response["compute_s"] = time.perf_counter()-start
                    stream.write((json.dumps(response, allow_nan=False)+"\n").encode()); stream.flush()
    finally:
        if bound:
            args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
