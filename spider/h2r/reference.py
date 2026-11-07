"""Human frame conversion and time-preserving generalized-position interpolation."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .model import finite


def hand_targets(reference, arrays, sides, calibration):
    """Use decoded wrist/tips, converting only raw MANO root rotations to world."""
    if (
        reference["mano"]["joint_order"]
        != "MANO 16 + thumb/index/middle/ring/little tips"
    ):
        raise ValueError("Unsupported MANO joint order")
    count = len(arrays["time_s"])
    coordinates = reference["coordinates"]
    world = Rotation.identity()
    if reference["mano"].get("parameter_coordinates") == "source_world":
        transform = finite(
            coordinates["source_to_benchmark"], (4, 4), "source transform"
        )
        if (
            not np.allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7
            )
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-7)
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7)
        ):
            raise ValueError("Non-rigid source transform")
        world = Rotation.from_matrix(transform[:3, :3])
    wrists, tips = {}, {}
    for side in sides:
        joints = finite(arrays[f"{side}_joints_m"], (count, 21, 3), "MANO joints")
        if f"{side}_mano_quat" in arrays:
            quat = finite(
                arrays[f"{side}_mano_quat"], (count, 16, 4), "MANO quaternion"
            )[:, 0]
            if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4, rtol=0):
                raise ValueError("Invalid MANO root quaternion")
            rotation = Rotation.from_quat(quat, scalar_first=True)
        else:
            rotation = Rotation.from_rotvec(
                finite(arrays[f"{side}_mano_rot"], (count, 3), "MANO rotation")
            )
        rotation = (
            world
            * rotation
            * Rotation.from_matrix(calibration[side]["mano_from_wrist_rotation"])
        )
        wrists[side] = np.column_stack(
            (joints[:, 0], rotation.as_quat(scalar_first=True))
        )
        tips[side] = joints[:, 16:21].copy()
    return wrists, tips


def velocities(model, times, qpos):
    """Finite differences on MuJoCo's configuration manifold, with a zero initial velocity."""
    qvel = np.zeros((len(times), model.nv))
    for i in range(1, len(times)):
        mujoco.mj_differentiatePos(
            model, qvel[i], times[i] - times[i - 1], qpos[i - 1], qpos[i]
        )
    return qvel


def sample_reference(model, layout, arrays, query):
    """Sample at exact physical times; pad only the private planning horizon."""
    times = arrays["time_s"]
    clipped = np.clip(query, times[0], times[-1])
    qpos = np.column_stack(
        [np.interp(clipped, times, values) for values in arrays["qpos"].T]
    )
    for q, _ in layout.root_indices.values():
        qpos[:, q + 3 : q + 7] = Slerp(
            times,
            Rotation.from_quat(arrays["qpos"][:, q + 3 : q + 7], scalar_first=True),
        )(clipped).as_quat(scalar_first=True)
    qvel = velocities(model, np.asarray(query), qpos)
    qvel[0] = [np.interp(clipped[0], times, values) for values in arrays["qvel"].T]
    qvel[np.asarray(query) > times[-1]] = 0
    return qpos, qvel, layout.controls(qpos)
