"""A task: one demo, one robot and their assembled physical scene."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from .demo import Demo, load_demo
from .files import content_hash, manifest_fields, read_json, read_manifest, sha256, text_hash, write_json
from .robot import Robot, load_robot

FORMAT = "h2r-task-v1"
MANIFEST = "task.json"
SCENE = "scene.xml"
COMPILED = "compiled.json"


@dataclass(frozen=True)
class Task:
    path: Path
    task_id: str
    demo: Demo
    robot: Robot
    compiled: dict            # compiled.json: compiled parameters of the assembled scene

    dataset = property(lambda self: self.demo.dataset)
    profile = property(lambda self: self.demo.profile)
    hand_sides = property(lambda self: self.demo.hand_sides)
    duration_s = property(lambda self: self.demo.duration_s)
    timestep_s = property(lambda self: self.demo.timestep_s)  # nominal; evaluation may override it
    scene_xml = property(lambda self: self.path / SCENE)

    @property
    def object_binding(self):
        """MJCF names of the tracked objects, as also announced by the evaluation server."""
        objects = self.demo.objects.values()
        joints = [joint for obj in objects for joint in obj.joints]
        return {"roots": {obj.name: {"body": obj.root_body, "joint": obj.root_joint}
                          for obj in objects if obj.root_joint},
                "parts": {part: body for obj in objects for part, body in obj.parts.items()},
                "scoring_parts": list(self.demo.scoring_parts),
                "surface_geoms": {part: [f"{part}_visual"] for part in self.demo.part_order},
                "joints": {joint.name: joint.mjcf_name for joint in joints},
                "reference_joint_keys": {joint.name: joint.spec["reference_key"] for joint in joints}}

    @property
    def initial_object_state(self):
        """Object state at the first reference frame: root poses [x, y, z, qw, qx, qy, qz] and joint angles."""
        roots, joints = {}, {}
        for obj in self.demo.objects.values():
            if obj.root_joint:
                position, rotation = self.demo.part_pose(obj.root_part)
                roots[obj.name] = [*position[0].tolist(), *rotation[0].tolist()]
            joints.update({joint.name: float(joint.position_rad[0]) for joint in obj.joints})
        return {"object_roots": roots, "object_joints": joints}


def _portable(text, demo, robot):  # demo, robot: their paths as written in task.json
    """Scene text with resource paths as in a standalone task, so its hash ignores the layout."""
    return text.replace(f'file="{demo}/', 'file="demo/').replace(f'file="{robot}/', 'file="robot/')


def _identity(manifest):
    return {"format": manifest["format"], "demo_id": manifest["demo"]["demo_id"],
            "robot_id": manifest["robot"]["robot_id"],
            "scene_sha256": manifest["scene_sha256"], "compiled_sha256": manifest["compiled_sha256"]}


def _reference(path, relative, root):
    target = (path / relative).resolve()
    if Path(relative).is_absolute() or root not in target.parents:
        raise ValueError(f"Task reference leaves its package: {relative}")
    return target


def load_task(path, *, root=None):
    """Verify the task, its demo and its robot; all three must be the sealed ones.

    A standalone task holds its demo and robot; a task inside a suite references the suite's
    shared packages. `root` is the directory those references must stay in (default: the
    enclosing suite, else the task itself).
    """
    path = Path(path).resolve()
    if root is None:
        root = next((parent for parent in path.parents if (parent / "suite.json").is_file()), path)
    root = Path(root).resolve()
    manifest = read_manifest(path, MANIFEST, FORMAT, _identity)
    # The package paths are outside the task identity: a standalone export changes them.
    with manifest_fields(f"{path.name}/{MANIFEST}"):
        references = {key: manifest[key]["path"] for key in ("demo", "robot")}
        if not all(isinstance(value, str) for value in references.values()):
            raise TypeError("demo and robot paths must be strings")
    demo = load_demo(_reference(path, references["demo"], root))
    robot = load_robot(_reference(path, references["robot"], root))
    if demo.demo_id != manifest["demo"]["demo_id"] or robot.robot_id != manifest["robot"]["robot_id"]:
        raise ValueError("Task references a different demo or robot")
    if set(demo.hand_sides) != set(robot.hands):
        raise ValueError(f"Robot hands {sorted(robot.hands)} differ from task hands {sorted(demo.hand_sides)}")
    text = (path / SCENE).read_text()
    if (text_hash(_portable(text, **references)) != manifest["scene_sha256"]
            or sha256(path / COMPILED) != manifest["compiled_sha256"]):
        raise ValueError("Task scene changed")
    # Every scene resource must be a sealed file of the referenced demo or robot.
    sealed = {f"{references['demo']}/{name}" for name in read_json(demo.path / "demo.json")["files"]}
    sealed |= {f"{references['robot']}/{name}" for name in read_json(robot.path / "robot.json")["files"]}
    for node in ET.fromstring(text).iter():
        if "file" in node.attrib and node.attrib["file"] not in sealed:
            raise ValueError(f"Unsealed scene resource: {node.attrib['file']}")
    return Task(path, manifest["task_id"], demo, robot, read_json(path / COMPILED))


def seal_task(path, demo_path, robot_path):
    """Write task.json next to scene.xml and compiled.json; paths are relative to the task directory."""
    path = Path(path)
    demo_path, robot_path = Path(demo_path).as_posix(), Path(robot_path).as_posix()
    demo, robot = load_demo(path / demo_path), load_robot(path / robot_path)
    manifest = {"format": FORMAT,
                "demo": {"path": demo_path, "demo_id": demo.demo_id},
                "robot": {"path": robot_path, "robot_id": robot.robot_id},
                "scene_sha256": text_hash(_portable((path / SCENE).read_text(), demo_path, robot_path)),
                "compiled_sha256": sha256(path / COMPILED)}
    manifest["task_id"] = content_hash(_identity(manifest))
    write_json(path / MANIFEST, manifest)
    return manifest


def export_task(task, destination):
    """Copy a task with its demo and robot into one standalone directory; the task_id is unchanged."""
    task = task if isinstance(task, Task) else load_task(task)
    destination = Path(destination)
    destination.mkdir(parents=True)
    manifest = read_json(task.path / MANIFEST)
    shutil.copytree(task.demo.path, destination / "demo")
    shutil.copytree(task.robot.path, destination / "robot")
    (destination / SCENE).write_text(_portable(task.scene_xml.read_text(), manifest["demo"]["path"],
                                               manifest["robot"]["path"]))
    shutil.copyfile(task.path / COMPILED, destination / COMPILED)
    exported = seal_task(destination, "demo", "robot")
    if exported["task_id"] != task.task_id:
        raise ValueError("Exported task identity changed")
    return destination
