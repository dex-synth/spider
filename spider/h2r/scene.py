"""Build private IK/MPC models from the verified benchmark task in memory."""

from pathlib import Path

import mujoco

from .bundle import load_task, local_file, read_json


def calibration_for(task):
    """Select method-specific hand tracking frames, never physical parameters."""
    calibration = read_json(Path(__file__).parent / "config/sharpa_wave.json")
    if task["robot_name"] not in calibration["robots"]:
        raise ValueError(f"No SPIDER hand calibration for {task['robot_name']}")
    return calibration


def planning_spec(task_path, task, calibration):
    """Compile the task with massless tracking sites; do not write a scene copy."""
    if load_task(task_path) != task:
        raise ValueError("Planning task identity changed")
    spec = mujoco.MjSpec.from_file(str(local_file(task_path, task["scene"])))
    wrist_sites, finger_sites = [], []
    for side in task["semantics"]["hands"]:
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


def planning_model(task_path, task, calibration):
    """Compile the unchanged physical task plus tracking sites for MPC."""
    spec, wrists, fingers = planning_spec(task_path, task, calibration)
    return spec.compile(), wrists, fingers
