"""Native fast-IK objectives and updates for a caller-provided task model."""

import mink
import numpy as np

from spider.quaternion import normalize_qpos


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


def solve_velocity(configuration, tasks, dt):
    """One unchanged upstream DAQP update, without extra constraints."""
    return mink.solve_ik(configuration, tasks, dt, "daqp", damping=1e-5)


def solve(
    model, wrist_targets, finger_targets, sites, object_indices, objects, options
):
    """Upstream fast stages with named sites and object qpos addresses."""
    configuration = mink.Configuration(model)
    wrists, fingers, posture = make_tasks(
        configuration,
        *sites,
        **{
            k: options[k]
            for k in (
                "wrist_pos_cost",
                "wrist_ori_cost",
                "finger_pos_cost",
                "posture_cost",
            )
        },
    )
    tasks_wrist = [posture, *wrists]
    tasks_all = [posture, *wrists, *fingers]
    dt = options["sim_dt"]

    def set_objects(frame):
        configuration.data.qpos[object_indices] = objects[frame]

    def set_wrists(frame):
        for task, pose in zip(wrists, wrist_targets[frame], strict=True):
            task.set_target(mink.SE3(wxyz_xyz=np.r_[pose[3:], pose[:3]]))

    def set_fingers(frame):
        for task, pos in zip(fingers, finger_targets[frame], strict=True):
            task.set_target(mink.SE3(wxyz_xyz=np.r_[1.0, 0.0, 0.0, 0.0, pos]))

    def integrate(tasks, frame, steps):
        for _ in range(steps):
            velocity = solve_velocity(configuration, tasks, dt)
            configuration.integrate_inplace(velocity, dt)
            set_objects(frame)

    set_objects(0)
    configuration.update()
    set_wrists(0)
    integrate(tasks_wrist, 0, options["wrist_init_steps"])
    configuration.update()
    set_fingers(0)
    integrate(tasks_all, 0, options["finger_init_steps"])
    steps = max(1, int(round(options["ref_dt"] / dt)))
    result = []
    for frame in range(len(objects)):
        set_wrists(frame)
        set_fingers(frame)
        set_objects(frame)
        configuration.update()
        integrate(tasks_all, frame, steps)
        result.append(configuration.q.copy())
    return np.asarray(result)


def postprocess(model, qpos, dt, window):
    """Native mean/crop with quaternion sign alignment and normalized output."""
    import mujoco

    if len(qpos) <= window:
        raise ValueError("Fast IK needs more input frames than its filter window")
    roots = {
        model.joint(i).name: (int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i]))
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    }
    qpos = normalize_qpos(qpos, roots)
    filtered = np.column_stack(
        [
            np.convolve(column, np.ones(window) / window, mode="valid")
            for column in qpos.T
        ]
    )
    filtered = normalize_qpos(filtered, roots, require_unit=False)
    qvel = np.zeros((len(filtered) - 1, model.nv))
    for i in range(1, len(filtered)):
        mujoco.mj_differentiatePos(model, qvel[i - 1], dt, filtered[i - 1], filtered[i])
    return filtered[1:], qvel
