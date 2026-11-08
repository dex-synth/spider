"""Human frame conversion and time-preserving generalized-position interpolation."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

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


def component_sample(times, values, query):
    """Independent scalar interpolation, including quaternion and velocity columns."""
    return np.column_stack([np.interp(query, times, column) for column in values.T])


def native_reference(layout, arrays, config):
    """Native io.interp values plus explicit source-time mapping to the task clock.

    Upstream labels H*n align_corners samples as sim_dt and re-zeros the crop.
    Retain those values, but map their source times explicitly so that a dataset
    rate or filter cannot change the shared task speed or remove its endpoints.
    """
    import torch

    from spider.interp import interp

    times = arrays["filtered_time_s"]
    values = [arrays["filtered_qpos"], arrays["filtered_qvel"]]
    values.append(layout.controls(values[0]))
    tensors = [torch.as_tensor(v, dtype=torch.float32) for v in values]
    if config.ref_dt > config.sim_dt:
        # process_config computes ref_steps before reading task_info.ref_dt.
        # The pinned base YAML has .02; do not silently repair that native bug.
        factor = config.ref_steps
        if factor < 1:
            raise ValueError("Native interpolation requires a positive ref_steps")
        tensors = [interp(v[None], factor)[0] for v in tensors]
        source_times = np.linspace(times[0], times[-1], len(tensors[0]))
    else:
        factor = int(config.sim_dt / config.ref_dt)
        tensors = [v[::factor] for v in tensors]
        source_times = times[::factor]
    return {
        "time_s": source_times,
        **{
            k: v.numpy() for k, v in zip(("qpos", "qvel", "ctrl"), tensors, strict=True)
        },
    }


def sample_reference(model, layout, arrays, query):
    """Sample all native reference channels independently; hold endpoint velocities."""
    qpos, qvel = [
        component_sample(arrays["time_s"], arrays[key], query)
        for key in ("qpos", "qvel")
    ]
    ctrl = component_sample(
        arrays["time_s"], arrays.get("ctrl", layout.controls(arrays["qpos"])), query
    )
    return qpos, qvel, ctrl


def hold_indices(times, query):
    """Select the latest available action; never look ahead or stretch time."""
    # Tolerance only handles arithmetic disagreement at identical timestamps.
    return np.clip(
        np.searchsorted(times, np.asarray(query) + 1e-12, side="right") - 1,
        0,
        len(times) - 1,
    )
