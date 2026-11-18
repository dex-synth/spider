"""A preprocessed human demonstration: hands, object reference and the nominal physical scene."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .files import file_hashes, read_json, read_manifest, seal, verify_files

FORMAT = "h2r-demo-v1"
MANIFEST = "demo.json"
SIDES = ("left", "right")
MANO_CONVENTION = "h2r-mano-v1"  # docs/formats.md: world-frame absolute rotations, wrist translation
QUATERNION_CONVENTION = "wxyz-unit-continuous-v1"
# Per-hand arrays and their shapes after the leading time axis (None: not time-varying).
HAND_ARRAYS = {"joints_m": (21, 3), "vertices_m": (778, 3), "mano_quat": (16, 4), "mano_trans": (3,),
               "mano_shape": None, "faces": None}


@dataclass(frozen=True)
class Hand:
    joints_m: np.ndarray      # (T, 21, 3): 16 MANO joints, then thumb..little fingertips
    vertices_m: np.ndarray    # (T, 778, 3)
    faces: np.ndarray         # (F, 3)
    mano_quat: np.ndarray     # (T, 16, 4) wxyz: world wrist rotation, then 15 local joint rotations
    mano_trans: np.ndarray    # (T, 3): wrist position, equal to joints_m[:, 0]
    mano_shape: np.ndarray    # (10,)


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str               # part names
    child: str
    mjcf_name: str
    position_rad: np.ndarray  # (T,)
    spec: dict                # axis, origin_m, limits_rad, damping, ... as sealed


@dataclass(frozen=True)
class InteractionObject:
    name: str
    root_part: str
    root_body: str
    root_joint: str           # MJCF free joint; empty for a fixed root
    parts: dict               # part name -> MJCF body name
    joints: tuple


@dataclass(frozen=True)
class Demo:
    path: Path
    demo_id: str
    reference: dict           # reference.json
    scene: dict               # scene.json
    arrays: dict              # reference.npz

    dataset = property(lambda self: self.reference["dataset"])
    profile = property(lambda self: self.reference["profile"])
    hand_sides = property(lambda self: tuple(self.reference["task"]["hand_sides"]))
    reference_hz = property(lambda self: float(self.reference["reference_hz"]))
    duration_s = property(lambda self: float(self.reference["duration_s"]))
    time_s = property(lambda self: self.arrays["time_s"])
    part_order = property(lambda self: tuple(self.reference["object_geometry"]["part_order"]))
    scoring_parts = property(lambda self: tuple(self.scene["scoring_parts"]))
    environment = property(lambda self: self.scene["environment"])
    settings = property(lambda self: self.scene["settings"])
    timestep_s = property(lambda self: float(self.scene["settings"]["timestep_s"]))
    scene_xml = property(lambda self: self.path / "scene.xml")

    @property
    def hands(self):
        """Participating hands only."""
        return {side: Hand(**{name: self.arrays[f"{side}_{name}"] for name in HAND_ARRAYS})
                for side in self.hand_sides}

    @property
    def objects(self):
        return {name: InteractionObject(
            name, obj["root"]["part"], obj["root"]["body"], obj["root"]["joint"] or "", dict(obj["parts"]),
            tuple(Joint(key, joint["parent"], joint["child"], joint["mjcf_name"],
                        self.arrays[joint["reference_key"]], joint)
                  for key, joint in obj["joints"].items()))
            for name, obj in self.scene["interaction_objects"].items()}

    def part_pose(self, part):
        """World position (T, 3) and wxyz rotation (T, 4) of one part."""
        index = self.part_order.index(part)
        return self.arrays["part_pos_m"][:, index], self.arrays["part_quat_wxyz"][:, index]


def _identity(manifest):
    return {key: value for key, value in manifest.items() if key != "demo_id"}


def _check(condition, message):
    if not condition:
        raise ValueError(f"Invalid demo: {message}")


def validate(reference, scene, arrays):
    """The format contract: the same keys, arrays and conventions for every dataset."""
    sides = reference["task"]["hand_sides"]
    _check(sides and len(set(sides)) == len(sides) and set(sides) <= set(SIDES), "hand_sides")
    _check(reference["mano"]["convention"] == MANO_CONVENTION, "MANO convention")
    _check(reference["quaternion_convention"] == QUATERNION_CONVENTION, "quaternion convention")
    time = arrays.get("time_s")
    _check(time is not None and time.ndim == 1 and len(time) >= 2, "time_s")
    count = len(time)
    _check(np.allclose(time, np.arange(count) / reference["reference_hz"], rtol=0, atol=1e-9), "time grid")
    _check(np.isclose(reference["duration_s"], time[-1], rtol=0, atol=1e-9), "duration_s")
    parts = reference["object_geometry"]["part_order"]
    shapes = {"part_pos_m": (count, len(parts), 3), "part_quat_wxyz": (count, len(parts), 4)}
    for side in SIDES:
        if side in sides:
            shapes.update({f"{side}_{name}": (count, *shape) for name, shape in HAND_ARRAYS.items() if shape})
            shapes[f"{side}_mano_shape"] = (10,)
        else:
            _check(not any(name.startswith(f"{side}_") for name in arrays),
                   f"arrays of the non-participating {side} hand")
    owned = {}
    for name, obj in scene["interaction_objects"].items():
        _check(obj["root"]["part"] in obj["parts"] and not set(obj["parts"]) & set(owned), f"parts of {name}")
        owned.update(obj["parts"])
        shapes.update({joint["reference_key"]: (count,) for joint in obj["joints"].values()})
    _check(set(owned) == set(parts) and len(parts) == len(owned), "interaction parts differ from part_order")
    _check(scene["scoring_parts"] and set(scene["scoring_parts"]) <= set(parts), "scoring_parts")
    _check("table" in scene["environment"], "environment.table")
    for name, shape in shapes.items():
        _check(name in arrays and arrays[name].shape == shape, f"{name} must have shape {shape}")
    for name, values in arrays.items():
        _check(values.dtype.kind in "iu" or np.isfinite(values).all(), f"non-finite {name}")
    for name in ("part_quat_wxyz", *(f"{side}_mano_quat" for side in sides)):
        _check(np.allclose(np.linalg.norm(arrays[name], axis=-1), 1, rtol=0, atol=1e-6), f"{name} is not unit")
        _check(np.all(np.sum(arrays[name][1:] * arrays[name][:-1], axis=-1) > -1e-7), f"{name} changes sign")
    for side in sides:
        faces = arrays.get(f"{side}_faces")
        _check(faces is not None and faces.ndim == 2 and faces.shape[1] == 3, f"{side}_faces")
        _check(np.allclose(arrays[f"{side}_mano_trans"], arrays[f"{side}_joints_m"][:, 0], rtol=0, atol=1e-5),
               f"{side}_mano_trans is not the wrist position")


def load_demo(path):
    """Verify every file, then the format contract."""
    path = Path(path).resolve()
    manifest = read_manifest(path, MANIFEST, FORMAT, _identity)
    verify_files(path, manifest["files"])
    with np.load(path / "reference.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    reference, scene = read_json(path / "reference.json"), read_json(path / "scene.json")
    try:
        validate(reference, scene, arrays)
    except KeyError as missing:
        raise ValueError(f"Invalid demo: missing {missing}") from None
    return Demo(path, manifest["demo_id"], reference, scene, arrays)


def seal_demo(path, input_key):
    """Write demo.json over the finished directory; `input_key` identifies the preprocessing recipe."""
    path = Path(path)
    return seal(path, MANIFEST, {"format": FORMAT, "input_key": input_key,
                                 "files": file_hashes(path, exclude=(MANIFEST,))}, _identity)
