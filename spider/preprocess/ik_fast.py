"""Run IK for the given hand type and mode.

This is a simplified version of IK based on mink.
No act scene, open-hand, or contact support.

Input: mujoco scene xml file + hand keypoints + object trajectories.
Output: npz file which contains qpos for key frames in xml.

Strategy:
1. Frame 0: set wrist targets first, integrate many steps, then add finger
   tip targets and integrate more steps.
2. Subsequent frames: solve IK with all targets, integrating sim_dt steps
   for each ref_dt interval.
3. Object qpos is set directly (no retargeting).

Author: Chaoyi Pan
Date: 2026-03-07
"""

import json
import os
from pathlib import Path

import loguru
import mink
import mujoco
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.mujoco_utils import get_viewer

# MANO 21-joint skeleton bones (parent -> child) for reference visualization
_MANO_SKELETON_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
_MANO_FINGERTIP_INDICES = [4, 8, 12, 16, 20]


class _DisplacementCollisionLimit(mink.CollisionAvoidanceLimit):
    """Geometric displacement bound matching Mink's delta-q QP variable.

    The installed CollisionAvoidanceLimit divides its distance bound by dt,
    while solve_ik optimizes delta_q and only then divides by dt. Evaluate the
    geometric bound at unit dt so it constrains displacement, not velocity.
    """
    def compute_qp_inequalities(self, configuration, dt):
        return super().compute_qp_inequalities(configuration, 1.)


def _build_ref_skeleton(server, mano_assets_root, qpos_ref, ref_idx, embodiment_type):
    """Render the reference MANO skeleton in viser using fingertip + wrist targets.

    Returns a callable update_fn(t) -> None that updates joint/bone positions
    for frame t. We don't actually need MANO FK here because qpos_ref already
    has wrist + 5 fingertips per frame; we draw a simple wrist+fingertip
    skeleton plus straight bones (wrist->each fingertip).
    """
    if "right" in embodiment_type or embodiment_type == "bimanual":
        wrist_idx = ref_idx["right_palm"]
        tip_idxs = [
            ref_idx[f"right_{n}"]
            for n in ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        ]
    elif "left" in embodiment_type:
        wrist_idx = ref_idx["left_palm"]
        tip_idxs = [
            ref_idx[f"left_{n}"]
            for n in ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        ]
    else:
        return None

    wrist_h = server.scene.add_icosphere(
        "ref_hand/wrist",
        radius=0.012,
        color=(80, 130, 220),
        position=tuple(qpos_ref[0, wrist_idx, :3]),
    )
    tip_handles = [
        server.scene.add_icosphere(
            f"ref_hand/tip_{i}",
            radius=0.008,
            color=(220, 80, 80),
            position=tuple(qpos_ref[0, tidx, :3]),
        )
        for i, tidx in enumerate(tip_idxs)
    ]
    bone_handles = []
    for tidx in tip_idxs:
        h = server.scene.add_spline_catmull_rom(
            f"ref_hand/bone_w_to_{tidx}",
            positions=np.stack([qpos_ref[0, wrist_idx, :3], qpos_ref[0, tidx, :3]]),
            color=(255, 200, 50),
            line_width=2.5,
        )
        bone_handles.append((h, tidx))

    def update(t):
        wrist_pos = qpos_ref[t, wrist_idx, :3]
        wrist_h.position = tuple(wrist_pos)
        for k, tidx in enumerate(tip_idxs):
            tip_handles[k].position = tuple(qpos_ref[t, tidx, :3])
        for h, tidx in bone_handles:
            h.positions = np.stack([wrist_pos, qpos_ref[t, tidx, :3]])

    return update


def make_tasks(
    configuration, wrist_sites, finger_sites, *, wrist_pos_cost=0.3,
    wrist_ori_cost=3.0, finger_pos_cost=10.0, posture_cost=1e-2,
):
    """Build the native wrist, fingertip and posture objective for either entrypoint."""
    wrists = [
        mink.FrameTask(
            frame_name=site, frame_type="site", position_cost=wrist_pos_cost,
            orientation_cost=wrist_ori_cost, lm_damping=1.0,
        )
        for site in wrist_sites
    ]
    fingers = [
        mink.FrameTask(
            frame_name=site, frame_type="site", position_cost=finger_pos_cost,
            orientation_cost=0.0, lm_damping=1.0,
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


def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "xhand",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    show_viewer: bool = True,
    show_viser_viewer: bool = False,
    save_video: bool = True,
    save_viser: bool = False,
    mano_assets_root: str | None = None,
    data_id: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    sim_dt: float = 0.005,
    ref_dt: float | None = None,
    wrist_pos_cost: float = 0.3,
    wrist_ori_cost: float = 3.0,
    finger_pos_cost: float = 10.0,
    posture_cost: float = 1e-2,
    wrist_init_steps: int = 200,
    finger_init_steps: int = 300,
    average_frame_size: int = 3,
    z_offset: float = 0.0,
    articulated_object: bool = False,
    floating_hand: bool = False,
    preserve_timing: bool = False,
    collision_avoidance: bool = False,
    max_step_norm: float | None = None,
):
    if preserve_timing and average_frame_size != 1:
        raise ValueError("Time-preserving task adaptation requires average_frame_size=1")
    if articulated_object and embodiment_type != "right":
        raise ValueError("Articulated task adaptation currently supports the right hand")
    # Resolve directories
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    processed_dir_mano = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir_robot, exist_ok=True)
    model_path = f"{processed_dir_robot}/../scene.xml"

    # Resolve ref_dt: prefer the per-task value the dataset processor wrote
    # to task_info.json (mocap rate varies per dataset — Arctic 30 Hz,
    # OakInk2 50 Hz, etc.) so IK plays back at the recorded rate. Fall back
    # to 50 Hz / 0.02 s only if no task_info is present.
    task_info_path = Path(processed_dir_mano).parent / "task_info.json"
    if ref_dt is None:
        if task_info_path.exists():
            with open(task_info_path) as f:
                ref_dt = float(json.load(f).get("ref_dt", 0.02))
            loguru.logger.info(
                f"ref_dt={ref_dt:.4f}s ({1.0/ref_dt:.1f}Hz) loaded from {task_info_path}"
            )
        else:
            ref_dt = 0.02
            loguru.logger.warning(
                f"No task_info.json at {task_info_path}; defaulting ref_dt=0.02s"
            )

    # Load reference keypoints
    file_path = f"{processed_dir_mano}/trajectory_keypoints.npz"
    loaded_data = np.load(file_path)
    qpos_finger_right = loaded_data["qpos_finger_right"][start_idx:end_idx]
    qpos_finger_left = loaded_data["qpos_finger_left"][start_idx:end_idx]
    qpos_wrist_right = loaded_data["qpos_wrist_right"][start_idx:end_idx]
    qpos_wrist_left = loaded_data["qpos_wrist_left"][start_idx:end_idx]
    qpos_obj_right = loaded_data["qpos_obj_right"][start_idx:end_idx]
    qpos_obj_left = loaded_data["qpos_obj_left"][start_idx:end_idx]

    # Build reference array: (H, num_sites, 7) where 7 = [x, y, z, qw, qx, qy, qz]
    qpos_ref = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )
    qpos_ref[:, :, 2] += z_offset

    num_frames = qpos_finger_right.shape[0]

    # Reference index mapping:
    # 0: right_palm, 1-5: right fingers, 6: left_palm, 7-11: left fingers,
    # 12: right_object, 13: left_object
    ref_idx = {}
    ref_idx["right_palm"] = 0
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"right_{name}"] = i + 1
    ref_idx["left_palm"] = 6
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"left_{name}"] = i + 7
    ref_idx["right_object"] = 12
    ref_idx["left_object"] = 13

    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)

    # -- Viser setup (optional) --
    viser_state = {"enabled": False}
    if show_viser_viewer or save_viser:
        try:
            from spider.viewers.viser_viewer import (
                _STATE as _VISER_STATE,
                build_and_log_scene,
                init_viser,
                log_frame,
            )

            init_viser(app_name=f"spider-ik-{task}")
            spec, viser_model, viewer_body_entity_and_ids = build_and_log_scene(
                Path(model_path)
            )
            del spec, viser_model
            viser_state.update(
                {
                    "enabled": True,
                    "log_frame": log_frame,
                    "body_ids": viewer_body_entity_and_ids,
                    "state": _VISER_STATE,
                }
            )
            loguru.logger.info(
                "Viser scene initialized (will record IK rollout for inspection)."
            )
        except Exception as e:
            loguru.logger.warning(f"Failed to init viser: {e}")

    # Optional reference MANO skeleton (only if MANO is loaded correctly)
    ref_skeleton = None
    if viser_state["enabled"] and mano_assets_root is not None:
        ref_skeleton = _build_ref_skeleton(
            viser_state["state"].server,
            mano_assets_root,
            qpos_ref,
            ref_idx,
            embodiment_type,
        )

    # Object DOF count
    if embodiment_type == "bimanual":
        nq_obj = 14
    elif embodiment_type in ["right", "left"]:
        nq_obj = 7
    else:
        nq_obj = 0
    if articulated_object:
        nq_obj = 8

    # Create mink configuration
    configuration = mink.Configuration(model)
    data = configuration.data
    if floating_hand:
        data.qpos[:7] = qpos_ref[0, ref_idx["right_palm"]]
        configuration.update()

    # Build site name lists
    finger_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    if robot_type in ["allegro", "metahand", "leap"]:
        finger_names = finger_names[:4]

    wrist_sites = []
    finger_sites = []
    if embodiment_type in ["right", "bimanual"]:
        wrist_sites.append("right_palm")
        finger_sites.extend([f"right_{f}" for f in finger_names])
    if embodiment_type in ["left", "bimanual"]:
        wrist_sites.append("left_palm")
        finger_sites.extend([f"left_{f}" for f in finger_names])

    # Create IK tasks
    # Cost priority: finger_pos > wrist_pos > wrist_ori
    wrist_tasks, finger_tasks, posture_task = make_tasks(
        configuration, wrist_sites, finger_sites, wrist_pos_cost=wrist_pos_cost,
        wrist_ori_cost=wrist_ori_cost, finger_pos_cost=finger_pos_cost,
        posture_cost=posture_cost,
    )

    tasks_wrist = [posture_task, *wrist_tasks]
    tasks_all = [posture_task, *wrist_tasks, *finger_tasks]

    n_substeps = max(1, int(round(ref_dt / sim_dt)))
    solve_options = {}
    if collision_avoidance:
        if not articulated_object or not floating_hand:
            raise ValueError("Collision-limited task adaptation expects floating hand + articulated object")
        hand_geoms = [i for i in range(model.ngeom) if model.geom(i).name.startswith("robot_")
                      and model.geom_contype[i]]
        object_geoms = [i for i in range(model.ngeom) if model.geom(i).name.startswith("object_")
                        and model.geom_contype[i]]
        solve_options["limits"] = [mink.ConfigurationLimit(model), _DisplacementCollisionLimit(
            model, [(hand_geoms, object_geoms), (hand_geoms, hand_geoms)], gain=.5,
            minimum_distance_from_collisions=.001, collision_detection_distance=.02)]
        solve_options["constraints"] = [mink.DofFreezingTask(model, list(range(model.nv-7, model.nv)))]

    def solve(tasks):
        return solve_velocity(
            configuration, tasks, sim_dt, max_step_norm=max_step_norm, **solve_options
        )

    # -- Helpers --
    def set_object_qpos(t):
        if articulated_object:
            data.qpos[-8:-1] = qpos_ref[t, ref_idx["right_object"]]
            data.qpos[-1] = loaded_data["object_hinge_rad"][start_idx:end_idx][t]
        elif embodiment_type == "bimanual":
            data.qpos[-14:-7] = qpos_ref[t, ref_idx["right_object"]]
            data.qpos[-7:] = qpos_ref[t, ref_idx["left_object"]]
        elif embodiment_type == "right":
            data.qpos[-7:] = qpos_ref[t, ref_idx["right_object"]]
        elif embodiment_type == "left":
            data.qpos[-7:] = qpos_ref[t, ref_idx["left_object"]]

    def set_wrist_targets(t):
        for wrist_task, site_name in zip(wrist_tasks, wrist_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            quat_wxyz = qpos_ref[t, ref_idx[site_name], 3:]
            wrist_task.set_target(mink.SE3(wxyz_xyz=np.concatenate([quat_wxyz, pos])))

    def set_finger_targets(t):
        for finger_task, site_name in zip(finger_tasks, finger_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            finger_task.set_target(
                mink.SE3(wxyz_xyz=np.array([1.0, 0.0, 0.0, 0.0, *pos]))
            )

    # -- Phase 1: Initialize wrist (frame 0) --
    loguru.logger.info("Phase 1: Initializing wrist position...")
    set_object_qpos(0)
    configuration.update()
    set_wrist_targets(0)
    for _ in range(wrist_init_steps):
        vel = solve(tasks_wrist)
        configuration.integrate_inplace(vel, sim_dt)
        set_object_qpos(0)

    # -- Phase 2: Add finger tips (frame 0) --
    loguru.logger.info("Phase 2: Initializing finger positions...")
    configuration.update()
    set_finger_targets(0)
    for _ in range(finger_init_steps):
        vel = solve(tasks_all)
        configuration.integrate_inplace(vel, sim_dt)
        set_object_qpos(0)

    # -- Main IK loop --
    loguru.logger.info(f"Running IK for {num_frames} frames...")
    qpos_list = []
    images = []

    if save_video:
        import imageio

        model.vis.global_.offwidth = 720
        model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(model, height=480, width=720)

    run_viewer = get_viewer(show_viewer, model, data)
    rate_limiter = RateLimiter(1 / ref_dt)

    with run_viewer() as gui:
        for t in range(num_frames):
            set_wrist_targets(t)
            set_finger_targets(t)
            set_object_qpos(t)
            configuration.update()

            for _ in range(n_substeps):
                vel = solve(tasks_all)
                configuration.integrate_inplace(vel, sim_dt)
                set_object_qpos(t)

            qpos_list.append(configuration.q.copy())

            if save_video:
                mujoco.mj_forward(model, data)
                renderer.update_scene(data=data, camera="front")
                images.append(renderer.render())

            if viser_state["enabled"]:
                mujoco.mj_forward(model, data)
                viser_state["log_frame"](
                    data,
                    sim_time=t * ref_dt,
                    viewer_body_entity_and_ids=viser_state["body_ids"],
                    show_ui=True,
                    playback_fps=1.0 / ref_dt,
                )
                if ref_skeleton is not None:
                    ref_skeleton(t)

            if show_viewer:
                mujoco.mj_forward(model, data)
                gui.sync()
                rate_limiter.sleep()

            if t % 100 == 0:
                loguru.logger.info(f"  Frame {t}/{num_frames}")

    qpos_list = np.array(qpos_list)

    # -- Post-processing: moving average filter --
    def moving_average_filter(signal_data, window_size=5):
        return np.convolve(
            signal_data, np.ones(window_size) / window_size, mode="valid"
        )

    filtered = np.zeros(
        (qpos_list.shape[0] - average_frame_size + 1, qpos_list.shape[1])
    )
    for i in range(qpos_list.shape[1]):
        filtered[:, i] = moving_average_filter(qpos_list[:, i], average_frame_size)
    qpos_list = filtered

    # -- Compute qvel via finite differences --
    n_filtered = qpos_list.shape[0]
    qvel_list = np.zeros((n_filtered - 1, model.nv))
    for i in range(1, n_filtered):
        mujoco.mj_differentiatePos(
            model, qvel_list[i - 1], ref_dt, qpos_list[i - 1], qpos_list[i]
        )
    if preserve_timing:
        qvel_list = np.vstack((np.zeros((1, model.nv)), qvel_list))
    else:
        qpos_list = qpos_list[1:]
    assert qpos_list.shape[0] == qvel_list.shape[0]

    ctrl_list = qpos_list[:, : model.nq - nq_obj]
    if floating_hand:
        ctrl_data = mujoco.MjData(model)
        ctrl_list = []
        for q in qpos_list:
            ctrl_data.qpos[:] = q
            mujoco.mj_forward(model, ctrl_data)
            ctrl_list.append(ctrl_data.actuator_length.copy())
        ctrl_list = np.asarray(ctrl_list)

    # -- Forward rollout for validation --
    mj_data_rollout = mujoco.MjData(model)
    n_rollout_substeps = max(1, int(round(ref_dt / sim_dt)))
    model.opt.timestep = ref_dt / n_rollout_substeps
    mj_data_rollout.qpos[:] = qpos_list[0]
    mj_data_rollout.qvel[:] = qvel_list[0]
    mj_data_rollout.ctrl[:] = ctrl_list[0]
    for _ in range(n_rollout_substeps):
        mujoco.mj_step(model, mj_data_rollout)
    n_final = qpos_list.shape[0]
    qpos_rollout = np.zeros((n_final, model.nq))
    qpos_rollout[0] = qpos_list[0]
    for i in range(1, n_final):
        mj_data_rollout.ctrl[:] = ctrl_list[i]
        noise = np.random.randn(model.nu) * 0.2
        noise[:6] *= 0.0
        noise[22:28] *= 0.0
        mj_data_rollout.ctrl[:] += noise
        for _ in range(n_rollout_substeps):
            mujoco.mj_step(model, mj_data_rollout)
        qpos_rollout[i] = mj_data_rollout.qpos.copy()

    # -- Save outputs --
    file_dir = processed_dir_robot
    if save_video:
        import imageio

        video_path = f"{file_dir}/visualization_ik.mp4"
        imageio.mimsave(video_path, images, fps=int(1 / ref_dt))
        loguru.logger.info(f"Saved video to {video_path}")

    out_npz = f"{file_dir}/trajectory_kinematic.npz"
    np.savez(out_npz, qpos=qpos_list, qvel=qvel_list, ctrl=ctrl_list, frequency=1 / ref_dt)
    loguru.logger.info(f"Saved {out_npz}")

    out_npz = f"{file_dir}/trajectory_ikrollout.npz"
    np.savez(out_npz, qpos=qpos_rollout)
    loguru.logger.info(f"Saved {out_npz}")

    # -- Save viser export for offline inspection --
    if save_viser and viser_state["enabled"]:
        viser_path = f"{file_dir}/visualization_ik.viser"
        try:
            server = viser_state["state"].server
            data_bytes = server.get_scene_serializer().serialize()
            Path(viser_path).write_bytes(data_bytes)
            loguru.logger.info(f"Saved viser scene to {viser_path}")
        except Exception as e:
            loguru.logger.warning(f"Failed to save viser scene: {e}")

    if show_viser_viewer and viser_state["enabled"]:
        loguru.logger.info(
            "Viser viewer running. Use timeline slider in browser. Ctrl-C to exit."
        )
        import time

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    tyro.cli(main)
