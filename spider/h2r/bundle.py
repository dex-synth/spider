"""Small file boundary; no benchmark or dataset runtime dependency."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def read_json(path):
    """Read a JSON object from a local artifact."""
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Write finite, human-readable artifact metadata."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    """Hash a file without loading it into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    """Match the benchmark's canonical JSON content identity."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def local_file(root, name):
    """Resolve only package-local dependencies."""
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Dependency is outside the bundle: {name}")
    return path


def verify_files(root, files):
    """Verify every declared dependency before consuming it."""
    if not files:
        raise ValueError("Empty artifact file manifest")
    for name, expected in files.items():
        if digest(local_file(root, name)) != expected:
            raise ValueError(f"Changed artifact dependency: {name}")


def verify_xml(root, entry, files):
    """Validate the flattened MJCF exported by bench and its explicit resources."""
    if entry not in files:
        raise ValueError(f"Unsealed MJCF dependency: {entry}")
    for node in ET.parse(local_file(root, entry)).iter():
        # Bench assembly expands includes. Reject alternative resolution rules
        # rather than verify one path and let MuJoCo load another.
        if node.tag in {"include", "plugin", "model"}:
            raise ValueError(f"Unsupported task MJCF dependency: {node.tag}")
        if node.tag == "compiler" and (
            any(node.get(key) for key in ("assetdir", "meshdir", "texturedir"))
            or node.get("strippath", "false") != "false"
        ):
            raise ValueError("Task MJCF must use explicit relative asset paths")
        for key, value in node.attrib.items():
            if not key.startswith("file"):
                continue
            child = local_file(root, str(Path(entry).parent / value))
            relative = str(child.relative_to(Path(root).resolve()))
            if relative not in files:
                raise ValueError(f"Unsealed MJCF dependency: {relative}")


def arrays(path):
    """Load plain NPZ arrays, never pickled objects."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def load_task(path):
    """Verify a v1/v2/v3 task, retaining its original semantic mappings."""
    path = Path(path).resolve()
    task = read_json(path / "task.json")
    if task["format"] not in {f"h2r-evaluation-v{i}" for i in (1, 2, 3)}:
        raise ValueError("Unsupported evaluation task format")
    if identity({k: v for k, v in task.items() if k != "task_id"}) != task["task_id"]:
        raise ValueError("Changed task identity")
    if not {"reference.json", "reference.npz", task["scene"]} <= task["files"].keys():
        raise ValueError("Incomplete task files")
    verify_files(path, task["files"])
    verify_xml(path, task["scene"], task["files"])
    if (
        task["observation"]["quaternion"] != "wxyz"
        or task["observation"]["body_velocity"] != "world_at_body_origin"
    ):
        raise ValueError("Unsupported observation coordinates")
    if (
        not np.isfinite(task["dt_s"])
        or task["dt_s"] <= 0
        or type(task["control_ticks"]) is not int
        or task["control_ticks"] < 1
        or type(task["end_tick"]) is not int
        or task["end_tick"] < 1
    ):
        raise ValueError("Invalid task clock")
    return task


def load_inputs(task_path):
    """Read decoded demonstration and robot semantics from the sealed task only."""
    task_path = Path(task_path).resolve()
    task = load_task(task_path)
    reference = read_json(task_path / "reference.json")
    if reference["reference_sha256"] != task["files"]["reference.npz"]:
        raise ValueError("Reference checksum differs")
    coordinates = reference["coordinates"]
    if isinstance(coordinates, dict):
        valid_coordinates = (
            coordinates.get("units") == "metres"
            and coordinates.get("quaternion") == "wxyz"
        )
    else:
        valid_coordinates = (
            coordinates == "ARCTIC world, unchanged; metres, radians; quaternion wxyz"
        )
    if not valid_coordinates or reference.get("speed") != 1.0:
        raise ValueError("Unsupported reference coordinates or speed")
    data = arrays(task_path / "reference.npz")
    times = data["time_s"]
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or times[0] != 0
        or not (np.diff(times) > 0).all()
        or not np.isclose(
            times[-1], task["end_tick"] * task["dt_s"], atol=1e-10, rtol=0
        )
    ):
        raise ValueError("Reference and evaluation clocks differ")
    sides = set(task["semantics"]["hands"])
    if not sides or not sides <= {"left", "right"}:
        raise ValueError("Missing or unsupported robot hands")
    if "task" in reference and sides != set(reference["task"]["hand_sides"]):
        raise ValueError("Robot hands differ from the task")
    return task, reference, data


PREPARED_FILES = {
    "trajectory_ikrollout.npz",
    "trajectory_keypoints.npz",
    "trajectory_kinematic.npz",
    "initial_robot.json",
    "initialization.json",
    "resolved_config.yaml",
    "calibration.json",
}


def validate_prepared(binding, task):
    """Accept method artifacts without any method-supplied physical model."""
    if binding.get("format") != "spider-h2r-v4":
        raise ValueError(
            "Regenerate SPIDER preparation from --task (requires spider-h2r-v4)"
        )
    for key in ("task_id", "bundle_id", "scene_id", "robot_manifest_sha256"):
        if binding[key] != task[key]:
            raise ValueError(f"Prepared input differs from task: {key}")
    if binding.get("model_source") != "task" or "planning_scene" in binding:
        raise ValueError("Planning model must come from the task")
    if set(binding["files"]) != PREPARED_FILES:
        raise ValueError(
            "Prepared files must contain only method references and configuration"
        )


def load_prepared(path, task):
    """Validate method artifacts against the selected evaluation task."""
    path = Path(path).resolve()
    binding = read_json(path / "h2r_binding.json")
    validate_prepared(binding, task)
    verify_files(path, binding["files"])
    return binding
