"""CPU-only deployment of one sealed SPIDER plan, independent of eval seeds."""

import time
from pathlib import Path

import numpy as np

from .bundle import arrays, identity, read_json, verify_files


def load_plan(path, task):
    """Require a complete, finite control sequence on the fixed task clock."""
    path = Path(path)
    binding = read_json(path / "plan.json")
    if binding.get("format") != "spider-h2r-plan-v1" or binding.get(
        "plan_id"
    ) != identity({k: v for k, v in binding.items() if k != "plan_id"}):
        raise ValueError("Invalid SPIDER plan identity")
    for key in (
        "task_id",
        "bundle_id",
        "scene_id",
        "robot_manifest_sha256",
        "dt_s",
        "control_ticks",
        "end_tick",
    ):
        if binding[key] != task[key]:
            raise ValueError(f"Plan differs from task: {key}")
    names = task["semantics"]["joint_names"]
    if (
        binding["joint_names"] != names
        or binding["joint_units"] != task["semantics"]["joint_units"]
    ):
        raise ValueError("Plan robot joint semantics differ")
    required = {"trajectory_mjwp.npz", "initial_robot.json", "resolved_config.yaml"}
    if not required <= binding["files"].keys():
        raise ValueError("Incomplete SPIDER plan files")
    verify_files(path, binding["files"])
    data = arrays(path / "trajectory_mjwp.npz")
    ticks = np.arange(0, task["end_tick"], task["control_ticks"], dtype=np.int64)
    if (
        data["ticks"].dtype.kind not in "iu"
        or not np.array_equal(data["ticks"], ticks)
        or binding["controls"] != len(ticks)
        or not np.array_equal(data["time_s"], ticks * task["dt_s"])
        or data["joint_targets"].shape != (len(ticks), len(names))
        or not np.isfinite(data["joint_targets"]).all()
    ):
        raise ValueError(
            "Plan must cover the complete task control grid with finite targets"
        )
    initial = read_json(path / "initial_robot.json")
    if set(initial) != set(names) or not np.isfinite(list(initial.values())).all():
        raise ValueError("Invalid plan initial robot proposal")
    return binding, data


class Replay:
    """Advance a per-episode cursor; physics observations never alter the plan."""

    def __init__(self, task, binding, data):
        self.task, self.binding, self.data = task, binding, data
        self.index = 0

    def control(self, frame, joint_names):
        """Return the saved target at the current common control tick."""
        started = time.perf_counter()
        names = self.binding["joint_names"]
        tick = frame["tick"]
        if len(joint_names) != len(names) or set(joint_names) != set(names):
            raise ValueError("Evaluation robot joints differ from the plan")
        if (
            type(tick) is not int
            or self.index >= len(self.data["ticks"])
            or tick != self.data["ticks"][self.index]
            or not np.isclose(
                frame["time_s"], tick * self.task["dt_s"], atol=1e-9, rtol=0
            )
        ):
            raise ValueError("Stale or inconsistent replay clock")
        targets = dict(
            zip(names, self.data["joint_targets"][self.index].tolist(), strict=True)
        )
        self.index += 1
        return targets, {
            "tick": tick,
            "time_s": frame["time_s"],
            "plan_id": self.binding["plan_id"],
            "compute_s": time.perf_counter() - started,
        }
