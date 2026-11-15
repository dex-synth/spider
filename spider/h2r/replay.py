"""CPU-only deployment of one sealed SPIDER plan, independent of eval seeds."""

import math
import time
from pathlib import Path

import numpy as np

from .artifacts import (
    INITIAL_POSE_ONLY,
    PLAN_FORMAT,
    arrays,
    validate_binding,
    validate_prepared,
)
from .h2r_io.files import content_hash, read_json, verify_files


def control_count(duration_s, control_dt_s):
    """Controls that cover the task; the last interval may be shorter."""
    if type(control_dt_s) not in (int, float) or not 0 < control_dt_s < math.inf:
        raise ValueError(f"Invalid control period: {control_dt_s!r}")
    ratio = duration_s / control_dt_s
    # 0.07 / 0.01 is 7.000000000000001 in floating point: not one more control.
    whole = round(ratio)
    if math.isclose(ratio, whole, rel_tol=0, abs_tol=1e-6):
        return whole
    return math.ceil(ratio)


def load_plan(path, task):
    """Require a complete, finite control sequence on the plan's own control clock.

    An initial-pose-only plan holds no controls. Returns the plan manifest, its arrays
    and the named initial robot proposal.
    """
    path = Path(path)
    binding = read_json(path / "plan.json")
    validate_binding(binding, task, PLAN_FORMAT, "plan")
    if binding.get("plan_id") != content_hash(
        {k: v for k, v in binding.items() if k != "plan_id"}
    ):
        raise ValueError("Invalid SPIDER plan identity")
    validate_prepared(binding["preparation"], task)
    semantics = task.robot.semantics
    names = semantics["joint_names"]
    if (
        binding["joint_names"] != names
        or binding["joint_units"] != semantics["joint_units"]
    ):
        raise ValueError("Plan robot joint semantics differ")
    required = {
        "trajectory_mjwp.npz",
        "initial_robot.json",
        "initialization.json",
        "resolved_config.yaml",
    }
    if not required <= binding["files"].keys():
        raise ValueError("Incomplete SPIDER plan files")
    verify_files(path, binding["files"])
    data = arrays(path / "trajectory_mjwp.npz")
    period = binding["control_dt_s"]
    count = control_count(task.duration_s, period)
    if binding["deployment"] == INITIAL_POSE_ONLY:
        count = 0
    if (
        binding["controls"] != count
        or not np.array_equal(data["time_s"], np.arange(count) * period)
        or data["joint_targets"].shape != (count, len(names))
        or not np.isfinite(data["joint_targets"]).all()
    ):
        raise ValueError("Plan must cover the complete task with finite targets")
    initial = read_json(path / "initial_robot.json")
    if set(initial) != set(names) or not np.isfinite(list(initial.values())).all():
        raise ValueError("Invalid plan initial robot proposal")
    return binding, data, initial


class Replay:
    """Advance a per-episode cursor; physics observations never alter the plan."""

    def __init__(self, binding, data, ticks):
        self.binding, self.targets = binding, data["joint_targets"]
        self.ticks = ticks  # Evaluation physics steps per control period.
        self.index = 0

    def control(self, frame, joint_names):
        """Return the saved target of the current control period."""
        started = time.perf_counter()
        names = self.binding["joint_names"]
        tick = frame["tick"]
        if len(joint_names) != len(names) or set(joint_names) != set(names):
            raise ValueError("Evaluation robot joints differ from the plan")
        if self.binding["deployment"] == INITIAL_POSE_ONLY:
            raise ValueError(
                "SPIDER did not plan this task: its IK initial pose fails SPIDER's "
                "pre-check (see the plan's initialization.json), yet the evaluation "
                "accepted it under a looser admission threshold"
            )
        if (
            type(tick) is not int
            or self.index >= len(self.targets)
            or tick != self.index * self.ticks
        ):
            raise ValueError("Stale or inconsistent replay clock")
        targets = dict(zip(names, self.targets[self.index].tolist(), strict=True))
        self.index += 1
        return targets, {
            "tick": tick,
            "control": self.index - 1,
            "plan_id": self.binding["plan_id"],
            "compute_s": time.perf_counter() - started,
        }
