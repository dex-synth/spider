"""Human frame conversion and time-preserving generalized-position interpolation."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from spider.quaternion import normalize_qpos


def hand_targets(task, calibration):
    """Wrist pose and fingertip targets of every participating hand.

    The MANO wrist rotation is already in the benchmark world frame; only the
    method's wrist-frame calibration is applied.
    """
    wrists, tips = {}, {}
    for side in task.hand_sides:
        hand = task.demo.hands[side]
        rotation = Rotation.from_quat(
            hand.mano_quat[:, 0], scalar_first=True
        ) * Rotation.from_matrix(calibration[side]["mano_from_wrist_rotation"])
        wrists[side] = np.column_stack(
            (hand.joints_m[:, 0], rotation.as_quat(scalar_first=True))
        )
        tips[side] = hand.joints_m[:, 16:21].copy()
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
    """Independent scalar interpolation with endpoint holds."""
    return np.column_stack([np.interp(query, times, column) for column in values.T])


def qpos_sample(layout, times, values, query):
    """Keep scalar interpolation; use sign-aligned NLERP for free-root rotations."""
    source = normalize_qpos(values, layout.root_indices)
    sampled = component_sample(times, source, query)
    return normalize_qpos(sampled, layout.root_indices, require_unit=False)


def native_reference(layout, arrays, config):
    """Native scalar interpolation and source times, with normalized rotations.

    Upstream labels H*n align_corners samples as sim_dt and re-zeros the crop.
    Retain scalar values, but map source times explicitly so that a dataset
    rate or filter cannot change the shared task speed or remove its endpoints.
    """
    import torch

    from spider.interp import interp

    times = arrays["filtered_time_s"]
    values = [
        normalize_qpos(arrays["filtered_qpos"], layout.root_indices),
        arrays["filtered_qvel"],
    ]
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
    result = {
        "time_s": source_times,
        **{
            k: v.numpy() for k, v in zip(("qpos", "qvel", "ctrl"), tensors, strict=True)
        },
    }
    result["qpos"] = normalize_qpos(
        result["qpos"], layout.root_indices, require_unit=False
    )
    return result


def sample_reference(model, layout, arrays, query):
    """Sample NLERP poses and independent velocities, retaining endpoint holds."""
    qpos = qpos_sample(layout, arrays["time_s"], arrays["qpos"], query)
    qvel = component_sample(arrays["time_s"], arrays["qvel"], query)
    ctrl = component_sample(
        arrays["time_s"], arrays.get("ctrl", layout.controls(arrays["qpos"])), query
    )
    return qpos, qvel, ctrl
