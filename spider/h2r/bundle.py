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
    """Require executable MJCF dependencies to belong to the sealed closure."""
    pending, visited = [entry], set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if name not in files:
            raise ValueError(f"Unsealed MJCF dependency: {name}")
        for node in ET.parse(local_file(root, name)).iter():
            if "file" not in node.attrib:
                continue
            child = local_file(root, str(Path(name).parent / node.attrib["file"]))
            relative = str(child.relative_to(Path(root).resolve()))
            if relative not in files:
                raise ValueError(f"Unsealed MJCF dependency: {relative}")
            if node.tag == "include":
                pending.append(relative)


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


def load_inputs(human, robot, task_path):
    """Bind decoded human input and robot package to the fixed task."""
    human, robot = Path(human).resolve(), Path(robot).resolve()
    manifest = read_json(human / "bundle.json")
    if (
        manifest["format"] not in {f"h2r-preprocess-v{i}" for i in (1, 2, 3)}
        or manifest.get("kind", "physical") != "physical"
    ):
        raise ValueError("Expected a physical preprocessing bundle v1/v2/v3")
    if identity(manifest["files"]) != manifest["bundle_id"]:
        raise ValueError("Changed human bundle identity")
    verify_files(human, manifest["files"])
    verify_xml(human, "scene.xml", manifest["files"])
    robot_meta = read_json(robot / "robot.json")
    verify_files(robot, robot_meta["files_sha256"])
    task = load_task(task_path)
    for key in ("bundle_id", "scene_id"):
        if task[key] != manifest[key]:
            raise ValueError(f"Task differs from human bundle: {key}")
    if task["robot_manifest_sha256"] != digest(robot / "robot.json"):
        raise ValueError("Task differs from robot package")
    if task["semantics"] != read_json(robot / "semantics.json"):
        raise ValueError("Task robot semantics changed")
    for name in ("reference.json", "reference.npz"):
        if task["files"][name] != manifest["files"][name]:
            raise ValueError(f"Task reference differs: {name}")
    reference = read_json(human / "reference.json")
    if reference["reference_sha256"] != manifest["files"]["reference.npz"]:
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
    data = arrays(human / "reference.npz")
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


def load_prepared(path, task):
    """Validate a portable deployment against the selected evaluation task."""
    path = Path(path).resolve()
    binding = read_json(path / "h2r_binding.json")
    if binding["format"] != "spider-h2r-v1":
        raise ValueError("Unsupported SPIDER deployment")
    for key in ("task_id", "bundle_id", "scene_id", "robot_manifest_sha256"):
        if binding[key] != task[key]:
            raise ValueError(f"Prepared input differs from task: {key}")
    verify_files(path, binding["files"])
    verify_xml(path, binding["planning_scene"], binding["files"])
    return binding
