"""SPIDER's own prepared and plan manifests; the task is read by the vendored h2r_io."""

from pathlib import Path

import numpy as np

from .h2r_io.files import read_json, verify_files
from .settings import PIPELINE

PREPARED_FORMAT = "spider-h2r-v6"
PLAN_FORMAT = "spider-h2r-plan-v6"
# Where a prepared robot reference comes from. A recipe accepts one kind only: the
# fast MJWarp recipe SPIDER's own Mink IK, the CPU recipe the V2D Sharpa retargeting
# that upstream's CPU workflow is built on (imported from a FlashCHORD preparation).
REFERENCE_MINK = "spider-mink-fast"
REFERENCE_V2D = "v2d-sharpa-ik"
# A plan without controls: the IK initial pose failed SPIDER's pre-check, so it is only
# proposed to the evaluation, which records the method's failure at initialization.
INITIAL_POSE_ONLY = "initial-pose-only"
PREPARED_FILES = {
    REFERENCE_MINK: {
        "trajectory_ikrollout.npz",
        "trajectory_keypoints.npz",
        "trajectory_kinematic.npz",
        "initial_robot.json",
        "initialization.json",
        "resolved_config.yaml",
        "calibration.json",
    },
    REFERENCE_V2D: {
        "trajectory_kinematic.npz",
        "initial_robot.json",
        "initialization.json",
        "reference_source.json",
    },
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


def validate_prepared(binding, task, reference=None):
    """Accept method references only; the physical model always comes from the task.

    `reference` is the kind of robot reference the calling recipe requires; a
    preparation of the other kind is rejected instead of being mixed in.
    """
    validate_binding(binding, task, PREPARED_FORMAT, "preparation")
    kind = binding.get("reference")
    if kind not in PREPARED_FILES:
        raise ValueError(f"Unknown SPIDER reference kind: {kind!r}")
    if reference is not None and kind != reference:
        raise ValueError(
            f"This recipe needs a {reference} preparation, not {kind}: "
            "prepare the task for this recipe"
        )
    if set(binding.get("files", ())) != PREPARED_FILES[kind]:
        raise ValueError(
            "Prepared files must contain only method references and configuration"
        )
    if kind != REFERENCE_MINK:
        return
    processing = (binding.get("pipeline") or {}).get("rotation_processing")
    if processing != PIPELINE["rotation_processing"]:
        raise ValueError(
            "Regenerate SPIDER preparation from --task "
            f"(requires rotation_processing={PIPELINE['rotation_processing']})"
        )


def load_prepared(path, task, reference=None):
    """Verify a prepared directory against its manifest and the task."""
    path = Path(path)
    binding = read_json(path / "h2r_binding.json")
    validate_prepared(binding, task, reference)
    verify_files(path, binding["files"])
    return binding
