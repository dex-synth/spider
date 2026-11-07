"""Native fast-IK objectives and updates for a caller-provided task model."""

import mink
import numpy as np


def make_tasks(
    configuration,
    wrist_sites,
    finger_sites,
    *,
    wrist_pos_cost=0.3,
    wrist_ori_cost=3.0,
    finger_pos_cost=10.0,
    posture_cost=1e-2,
):
    """Build the native wrist, fingertip and posture objective on task tracking sites."""
    wrists = [
        mink.FrameTask(
            frame_name=site,
            frame_type="site",
            position_cost=wrist_pos_cost,
            orientation_cost=wrist_ori_cost,
            lm_damping=1.0,
        )
        for site in wrist_sites
    ]
    fingers = [
        mink.FrameTask(
            frame_name=site,
            frame_type="site",
            position_cost=finger_pos_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for site in finger_sites
    ]
    posture = mink.PostureTask(configuration.model, cost=posture_cost)
    posture.set_target(configuration.q.copy())
    return wrists, fingers, posture


def solve_velocity(configuration, tasks, dt, *, max_step_norm=None, **options):
    """Solve one native Mink update, optionally bounding its displacement."""
    velocity = mink.solve_ik(configuration, tasks, dt, "daqp", damping=1e-5, **options)
    if max_step_norm is not None:
        if max_step_norm <= 0:
            raise ValueError("max_step_norm must be positive")
        velocity *= min(1.0, max_step_norm / max(np.linalg.norm(velocity) * dt, 1e-30))
    return velocity
