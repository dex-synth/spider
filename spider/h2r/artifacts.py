"""SPIDER's own prepared and plan manifests; the task is read by the vendored h2r_io."""

from pathlib import Path

import numpy as np

from .h2r_io.files import read_json, verify_files
from .settings import PIPELINE

PREPARED_FORMAT = "spider-h2r-v5"
PLAN_FORMAT = "spider-h2r-plan-v5"
PREPARED_FILES = {
    "trajectory_ikrollout.npz",
    "trajectory_keypoints.npz",
    "trajectory_kinematic.npz",
    "initial_robot.json",
    "initialization.json",
    "resolved_config.yaml",
    "calibration.json",
}


def arrays(path):
    """Load plain NPZ arrays, never pickled objects."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def task_identity(task):
    """Identities that bind a SPIDER artifact to one benchmark task."""
    return {
        "task_id": task.task_id,
        "demo_id": task.demo.demo_id,
        "robot_id": task.robot.robot_id,
    }


def validate_binding(binding, task, expected, name):
    """Reject artifacts of another format generation or another task."""
    if binding.get("format") != expected:
        raise ValueError(
            f"Unsupported SPIDER {name} format {binding.get('format')!r}: "
            f"regenerate it from --task (requires {expected})"
        )
    for key, value in task_identity(task).items():
        if binding.get(key) != value:
            raise ValueError(f"SPIDER {name} was made for another task: {key}")


def validate_prepared(binding, task):
    """Accept method references only; the physical model always comes from the task."""
    validate_binding(binding, task, PREPARED_FORMAT, "preparation")
    if set(binding.get("files", ())) != PREPARED_FILES:
        raise ValueError(
            "Prepared files must contain only method references and configuration"
        )
    processing = (binding.get("pipeline") or {}).get("rotation_processing")
    if processing != PIPELINE["rotation_processing"]:
        raise ValueError(
            "Regenerate SPIDER preparation from --task "
            f"(requires rotation_processing={PIPELINE['rotation_processing']})"
        )


def load_prepared(path, task):
    """Verify a prepared directory against its manifest and the task."""
    path = Path(path)
    binding = read_json(path / "h2r_binding.json")
    validate_prepared(binding, task)
    verify_files(path, binding["files"])
    return binding
