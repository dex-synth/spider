"""A robot package restricted to the hands of one task."""

from dataclasses import dataclass
from pathlib import Path

from .files import file_hashes, read_json, read_manifest, seal, verify_files

FORMAT = "h2r-robot-v1"
MANIFEST = "robot.json"


@dataclass(frozen=True)
class Robot:
    path: Path
    robot_id: str
    name: str                 # hand model, e.g. "sharpa_wave"
    hands: tuple              # sides present in this package
    semantics: dict           # semantics.json: joint names, units, groups and per-hand frames
    compiled: dict            # compiled.json: compiled physical parameters by name
    mjcf: Path
    urdf: Path                # the whole package as one URDF
    hand_urdfs: dict          # side -> single-hand URDF


def _identity(manifest):
    return {key: value for key, value in manifest.items() if key != "robot_id"}


def load_robot(path):
    path = Path(path).resolve()
    manifest = read_manifest(path, MANIFEST, FORMAT, _identity)
    verify_files(path, manifest["files"])
    semantics = read_json(path / "semantics.json")
    if set(semantics["hands"]) != set(manifest["hands"]):
        raise ValueError("Robot hands differ from its semantics")
    return Robot(path, manifest["robot_id"], manifest["name"], tuple(manifest["hands"]), semantics,
                 read_json(path / "compiled.json"), path / manifest["mjcf"], path / manifest["urdf"],
                 {side: path / name for side, name in manifest["hand_urdfs"].items()})


def seal_robot(path, name, hands, *, mjcf, urdf, hand_urdfs):
    path = Path(path)
    return seal(path, MANIFEST, {"format": FORMAT, "name": name, "hands": list(hands), "mjcf": mjcf,
                                 "urdf": urdf, "hand_urdfs": hand_urdfs,
                                 "files": file_hashes(path, exclude=(MANIFEST,))}, _identity)
