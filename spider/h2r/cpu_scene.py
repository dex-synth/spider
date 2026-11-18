"""Private model and tables of the CPU recipe, built from one benchmark task.

Upstream's CPU workflow reads a scene exported from FlashCHORD
(`spider/simulators/chord_export.py`): a MuJoCo model with FlashCHORD's gains and
the tables of its control law. Here the scene is the benchmark task. The model is
compiled from `task.scene_xml` in memory and keeps every physical parameter; the
tables have upstream's layout, filled by joint name for a robot whose wrists are
scalar joints with position actuators.
"""

import mujoco
import numpy as np

from spider.simulators.cpu import (
    GROUP_ARTICULATION,
    GROUP_FINGER,
    GROUP_OBJECT_POS,
    GROUP_OBJECT_ROT,
    GROUP_WRIST_POS,
)

from .model import Layout
from .reference import qpos_sample
from .replay import control_count

# Compiled parameters must match the task's record within the benchmark's own
# tolerance for an exported scene.
RELATIVE, ABSOLUTE = 2e-5, 1e-9


def _same(value, recorded):
    return np.isclose(value, recorded, rtol=RELATIVE, atol=ABSOLUTE)


def private_model(task, timestep):
    """Compile the task scene for planning: its own step, the evaluation's robot.

    A position actuator's damping given as `dampratio` is converted at compile
    time, and the result depends on the MuJoCo version: the planning version
    (>= 3.17, for `mujoco.batch`) gives up to 1 % other values than the one the
    task was compiled with. The task records the compiled gains, so they are set
    from that record. Everything else is checked against it and left alone.

    Returns:
        The model, its named layout, and the largest relative damping change.
    """
    model = mujoco.MjModel.from_xml_path(str(task.scene_xml))
    layout = Layout(model, task)
    try:
        robot = task.compiled["robot"]["joints"]
        objects = task.compiled["base_scene"]["joints"]
    except (KeyError, TypeError):
        raise ValueError("The task's compiled.json lacks joint parameters") from None
    changed = 0.0
    for name, actuator in zip(layout.names, layout.actuators, strict=True):
        record, dof = robot[name], int(model.joint(name).dofadr[0])
        if not _same(model.actuator_gainprm[actuator, 0], record["kp"]):
            raise ValueError(f"Actuator stiffness differs from the task: {name}")
        compiled = -model.actuator_biasprm[actuator, 2]
        changed = max(changed, abs(compiled - record["kv"]) / max(record["kv"], 1e-12))
        model.actuator_biasprm[actuator, 2] = -record["kv"]
        for field, value in (
            ("friction", model.dof_frictionloss[dof]),
            ("damping", model.dof_damping[dof]),
            ("armature", model.dof_armature[dof]),
        ):
            if not _same(value, record[field]):
                raise ValueError(f"Joint {field} differs from the task: {name}")
    for key, name in layout.joints.items():
        record, dof = objects[key], int(model.joint(name).dofadr[0])
        for field, value in (
            ("friction", model.dof_frictionloss[dof]),
            ("damping", model.dof_damping[dof]),
            ("armature", model.dof_armature[dof]),
        ):
            if not _same(value, record[field]):
                raise ValueError(f"Object joint {field} differs from the task: {key}")
    model.opt.timestep = timestep
    return model, layout, changed


def _hands(model, layout, task):
    """Per participating hand: wrist translation, wrist rotation and finger joints."""
    semantics = task.robot.semantics
    hands = []
    for side in task.hand_sides:
        hand = semantics["hands"][side]
        groups = (
            list(hand["wrist_translation_joints"]),
            list(hand["wrist_rotation_joints"]),
            list(hand["finger_joints"]),
        )
        if len(groups[0]) != 3 or len(groups[1]) != 3 or not groups[2]:
            raise ValueError(f"Expected 3 + 3 wrist joints and fingers: {side} hand")
        hands.append((side, model.body(hand["wrist_body"]).id, groups))
    owned = [name for _, _, groups in hands for group in groups for name in group]
    if len(set(owned)) != len(owned) or set(owned) != set(layout.names):
        raise ValueError("Hand joint groups do not partition the robot joints")
    if len({len(groups[2]) for _, _, groups in hands}) != 1:
        raise ValueError("Hands with different finger counts are not supported")
    return hands


def tables(model, layout, task, reference, options):
    """The reference and the control tables of the task, in upstream's layout.

    Row `i` is the reference at the end of control frame `i`, the pose that frame
    should reach: the evaluation scores on an absolute clock, on which upstream's
    own row convention (row `i` is the pose at the start of frame `i`) would lag
    by one frame. The initial state is the reference at time zero, returned apart.

    Args:
        model: The private model of :func:`private_model`.
        layout: Its named layout.
        task: The benchmark task.
        reference: Arrays of the preparation's `trajectory_kinematic.npz`.
        options: The adaptation settings (`config/h2r_cpu.yaml`).

    Returns:
        The tables and the initial `qpos`.
    """
    frame_dt, substeps = float(options["frame_dt"]), int(options["substeps"])
    source_times, source = reference["time_s"], reference["qpos"]
    if source_times[0] != 0 or source.shape != (len(source_times), model.nq):
        raise ValueError("The prepared reference must start at time zero")
    count = control_count(task.duration_s, frame_dt)
    times = np.minimum((np.arange(count) + 1) * frame_dt, task.duration_s)
    qpos_ref = qpos_sample(layout, source_times, source, times)

    def addresses(names):
        joints = [model.joint(name) for name in names]
        q = np.array([joint.qposadr[0] for joint in joints], dtype=np.int64)
        v = np.array([joint.dofadr[0] for joint in joints], dtype=np.int64)
        return q, v

    hands = _hands(model, layout, task)
    clip = options["action_clip"]
    scalar_q, scalar_group, vel_q, vel_dof, action_clip, residual_dof = (
        [] for _ in range(6)
    )
    wrist_pos_dof, wrist_ori_dof, finger_dof = [], [], []
    for _, _, (translation, rotation, fingers) in hands:
        pos_q, pos_v = addresses(translation)
        _, rot_v = addresses(rotation)
        finger_q, finger_v = addresses(fingers)
        wrist_pos_dof.append(pos_v)
        wrist_ori_dof.append(rot_v)
        finger_dof.append(finger_v)
        residual_dof += [*pos_v, *rot_v, *finger_v]
        action_clip += (
            [clip["wrist_position_m"]] * 3
            + [clip["wrist_rotation_rad"]] * 3
            + [clip["finger_rad"]] * len(fingers)
        )
        # The wrist rotation is not compared joint by joint: the cost reads the
        # wrist body's orientation (cpu_world.get_reward).
        scalar_q += [*pos_q, *finger_q]
        scalar_group += [GROUP_WRIST_POS] * 3 + [GROUP_FINGER] * len(fingers)
        vel_q += [*pos_q, *finger_q]
        vel_dof += [*pos_v, *finger_v]
    quat_q = []
    for q, _ in layout.root_indices.values():
        scalar_q += [q, q + 1, q + 2]
        scalar_group += [GROUP_OBJECT_POS] * 3
        quat_q.append(q + 3)
    scalar_q += layout.object_q.tolist()
    scalar_group += [GROUP_ARTICULATION] * len(layout.object_q)
    vel_q += layout.object_q.tolist()
    vel_dof += layout.object_v.tolist()

    parts = layout.binding["parts"]
    bodies = np.array([model.body(body).id for body in parts.values()], dtype=np.int64)
    if not len(bodies):
        raise ValueError("The task has no interaction object to track")
    data = mujoco.MjData(model)
    poses = np.zeros((count, len(bodies), 7))
    for frame, qpos in enumerate(qpos_ref):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        poses[frame, :, :3] = data.xpos[bodies]
        poses[frame, :, 3:] = data.xquat[bodies][:, [1, 2, 3, 0]]
    targets = np.zeros((count, model.nv))
    targets[:, layout.robot_v] = qpos_ref[:, layout.robot_q]
    integers = lambda values: np.asarray(values, dtype=np.int64)  # noqa: E731
    scene = {
        "frame_dt": np.asarray(frame_dt),
        "substeps": np.asarray(substeps),
        "sides": np.array([side for side, _, _ in hands]),
        "body_names": np.array(list(parts)),
        "qpos_ref": qpos_ref,
        "targets0": targets,
        "position_actuator": integers(layout.actuators),
        "position_actuator_dof": integers(layout.robot_v),
        "wrist_pos_dof": integers(wrist_pos_dof),
        "wrist_ori_dof": integers(wrist_ori_dof),
        "finger_dof": integers(finger_dof),
        # No ball wrists: nothing for upstream's wrist torque law to act on.
        "wrist_qpos": integers([]),
        "wrist_kp": np.zeros(0),
        "wrist_tau_max": np.zeros(0),
        "action_clip": np.asarray(action_clip, dtype=float),
        "action_ema": np.asarray(float(options["action_ema"])),
        "scalar_q": integers(scalar_q),
        "scalar_group": integers(scalar_group),
        "quat_q": integers(quat_q),
        "quat_group": integers([GROUP_OBJECT_ROT] * len(quat_q)),
        "vel_q": integers(vel_q),
        "vel_dof": integers(vel_dof),
        "articulation_q": integers(layout.object_q),
        "object_body": bodies,
        "wrist_body": integers([body for _, body, _ in hands]),
        "object_pose_ref": poses,
        # Upstream measures a constant offset between the reference's body frames
        # and the model's at reset. Here they are the same frames: no offset.
        "object_pose_reset": poses[0].copy(),
        # H2R only: the dof each residual dimension moves, in action order.
        "residual_dof": integers(residual_dof),
    }
    return scene, source[0].copy()
