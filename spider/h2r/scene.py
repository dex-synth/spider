"""Build private IK/MPC models from the verified benchmark task in memory."""

from pathlib import Path

import mujoco

from .h2r_io.files import read_json

# Hand tracking frames by robot name; sides are selected by the task.
CALIBRATIONS = {"sharpa_wave": "config/sharpa_wave.json"}


def calibration_for(task):
    """Select method-specific hand tracking frames, never physical parameters."""
    if task.robot.name not in CALIBRATIONS:
        raise ValueError(f"No SPIDER hand calibration for robot {task.robot.name!r}")
    calibration = read_json(Path(__file__).parent / CALIBRATIONS[task.robot.name])
    missing = set(task.hand_sides) - calibration.keys()
    if missing:
        raise ValueError(f"No SPIDER hand calibration for hands {sorted(missing)}")
    return calibration


def planning_spec(task, calibration):
    """Add massless tracking sites to the task scene; do not write a scene copy."""
    spec = mujoco.MjSpec.from_file(str(task.scene_xml))
    wrist_sites, finger_sites = [], []
    for side in task.hand_sides:
        frames = calibration[side]
        bodies = [frames["wrist_body"], *frames["fingertip_bodies"]]
        names = [f"h2r_{side}_wrist", *[f"h2r_{side}_tip_{i}" for i in range(5)]]
        for body, name in zip(bodies, names, strict=True):
            node = spec.body(body)
            if node is None or spec.site(name) is not None:
                raise ValueError(
                    f"Missing tracking body or duplicate site: {body}/{name}"
                )
            node.add_site(name=name, pos=[0, 0, 0], size=[0.001, 0, 0], group=4)
        wrist_sites.append(names[0])
        finger_sites.extend(names[1:])
    return spec, wrist_sites, finger_sites


def planning_model(task, calibration):
    """Compile the unchanged physical task plus tracking sites for MPC."""
    spec, wrists, fingers = planning_spec(task, calibration)
    return spec.compile(), wrists, fingers
