"""Map the fixed task's named targets into upstream fast Mink preprocessing."""

import copy

import mujoco
import numpy as np

from spider.preprocess.ik_fast import postprocess, solve

from .reference import component_sample, qpos_sample


def solve_kinematics(spec, layout, reference_q, times, wrists, tips, sites, options):
    """Keep native filter timing, correcting quaternion signs and unit norms."""
    dt = options["ref_dt"]
    if not np.allclose(np.diff(times), dt, rtol=1e-6, atol=1e-9):
        raise ValueError("Fast IK requires the declared uniform dataset reference rate")
    # Native start_idx=0/end_idx=-1 excludes the last source frame.
    object_indices = sorted(set(range(layout.model.nq)) - set(layout.robot_q))
    wrist_targets = np.stack(list(wrists.values()), axis=1)[:-1]
    finger_targets = np.concatenate(list(tips.values()), axis=1)[:-1]
    raw = solve(
        layout.model,
        wrist_targets,
        finger_targets,
        sites,
        object_indices,
        reference_q[:-1, object_indices],
        options,
    )
    filtered, filtered_v = postprocess(
        layout.model, raw, dt, options["average_frame_size"]
    )
    valid_times = (
        np.arange(len(filtered)) + (options["average_frame_size"] - 1) / 2 + 1
    ) * dt
    qpos = qpos_sample(layout, valid_times, filtered, times)
    qvel = component_sample(valid_times, filtered_v, times)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("Non-finite fast IK reference")
    return (
        qpos,
        qvel,
        {
            "native_time_s": times[:-1],
            "native_qpos": raw,
            "filtered_time_s": valid_times,
            "filtered_qpos": filtered,
            "filtered_qvel": filtered_v,
            "frequency": 1 / dt,
        },
    )


def diagnostic_rollout(model, layout, qpos, qvel, options):
    """Native noisy forward diagnostic; never used as the planning reference."""
    model = copy.copy(model)
    steps = max(1, int(round(options["ref_dt"] / options["sim_dt"])))
    model.opt.timestep = options["ref_dt"] / steps
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos[0], qvel[0]
    data.ctrl[:] = layout.controls(qpos[0])
    mujoco.mj_step(model, data, nstep=steps)
    result = np.empty_like(qpos)
    result[0] = qpos[0]
    groups = layout.task["semantics"]["joint_groups"]
    wrist_names = groups["wrist_translation"] + groups["wrist_rotation"]
    wrist_actuators = [layout.actuators[layout.names.index(n)] for n in wrist_names]
    for i in range(1, len(qpos)):
        noise = np.random.randn(model.nu) * 0.2
        noise[wrist_actuators] = 0
        data.ctrl[:] = layout.controls(qpos[i]) + noise
        mujoco.mj_step(model, data, nstep=steps)
        result[i] = data.qpos
    return result
