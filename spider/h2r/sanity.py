"""Native fast initial-state stability check on a private CPU model."""

import copy

import mujoco
import numpy as np


def check_stability(model, layout, initial_qpos, ctrl, config):
    """Hold the first control at .002 s; check maximum root drift over the run."""
    seconds = config.sanity_check_seconds
    if seconds <= 0:
        return {"enabled": False}
    model = copy.copy(model)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    pos_error = rot_error = 0.0
    for _ in range(int(seconds / 0.002)):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        for q, _ in layout.root_indices.values():
            pos_error = max(
                pos_error,
                float(np.linalg.norm(data.qpos[q : q + 3] - initial[q : q + 3])),
            )
            a, b = initial[q + 3 : q + 7], data.qpos[q + 3 : q + 7]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            angle = (
                0.0
                if min(na, nb) < 1e-8
                else 2 * np.arccos(np.clip(abs(np.dot(a / na, b / nb)), 0, 1))
            )
            rot_error = max(rot_error, float(angle))
    result = {
        "enabled": True,
        "seconds": seconds,
        "max_pos_error": pos_error,
        "max_rot_error": rot_error,
    }
    if (
        pos_error > config.sanity_check_pos_thresh
        or rot_error > config.sanity_check_rot_thresh
    ):
        raise RuntimeError(f"Upstream initial-state sanity check failed: {result}")
    return result
