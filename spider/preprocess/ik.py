# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the LICENSE file in the root directory of this source tree.

"""Paper IK numerical core, extracted from upstream 44717007's ik.py.

Dataset paths, viewer and contact export are omitted. Named sites and actuator
indices replace the original hand/object slices; the solve procedure is retained.
"""

import copy

import mujoco
import numpy as np


def make_ik_model(spec, targets, options):
    """Add the paper's mocap equalities to a private copy of the task model."""
    spec = spec.copy()
    for joint in spec.joints:
        joint.actfrclimited = 0
    for geom in spec.geoms:
        geom.contype = geom.conaffinity = 0
    for site, kind in targets:
        name = f"target_mocap_body_{site}"
        body = spec.worldbody.add_body(name=name, mocap=True)
        body.add_site(name=name, size=[0.001, 0, 0], group=4)
        if kind == "wrist":
            eq_type = mujoco.mjtEq.mjEQ_WELD
            solimp = [0, 0.95, options["wrist_solimp_width"], 0.5, 2]
            solref, torque = [0.02, 1], options["wrist_torque_scale"]
        elif kind == "object":
            eq_type = mujoco.mjtEq.mjEQ_WELD
            solimp = [0.9, 0.95, options["object_solimp_width"], 0.5, 2]
            solref, torque = [0.002, 1], 10.0
        else:
            eq_type = mujoco.mjtEq.mjEQ_CONNECT
            width = options["finger_solimp_width"] * (3 if kind >= 3 else 1)
            solimp, solref, torque = [0, 0.95, width, 0.5, 2], [0.01, 1], 1.0
        data = np.zeros(11)
        data[10] = torque
        spec.add_equality(
            name=f"{site}_{name}_equality_constraint",
            type=eq_type,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=site,
            name2=name,
            data=data,
            solimp=solimp,
            solref=solref,
        )
        spec.add_sensor(
            name=f"pos_{site}",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=site,
        )
    model = spec.compile()
    model.opt.timestep = options["sim_dt"]
    model.opt.iterations = 20
    model.opt.ls_iterations = 50
    model.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
    )
    mocap_ids = [
        int(model.body(f"target_mocap_body_{s}").mocapid[0]) for s, _ in targets
    ]
    return model, mocap_ids


def solve(model, physical_model, poses, mocap_ids, finger_ids, controls, options):
    """Retain random guesses, stability selection and per-frame physics solves."""
    physical_model = copy.copy(physical_model)
    physical_model.opt.timestep = options["sim_dt"]
    data, physical = mujoco.MjData(model), mujoco.MjData(physical_model)
    # RandomState uses the same stream as the native np.random calls, isolated
    # from other preparations. The original CLI did not pin an IK seed.
    rng = np.random.RandomState(options["seed"])

    def set_targets(frame, noisy=False):
        data.mocap_pos[mocap_ids] = poses[frame, :, :3]
        data.mocap_quat[mocap_ids] = poses[frame, :, 3:]
        if noisy:
            for index in finger_ids:
                data.mocap_pos[mocap_ids[index]] += rng.randn(3) * 0.002

    best_qpos, best_cost = np.zeros(model.nq), np.inf
    for _ in range(options["max_num_initial_guess"]):
        data.qpos[:] = rng.rand(model.nq)
        data.qvel[:] = 0
        data.ctrl[:] = rng.rand(model.nu)
        set_targets(0, noisy=True)
        for _ in range(30):
            data.ctrl[:] = controls(data.qpos)
            mujoco.mj_step(model, data)
        physical.qpos[:] = data.qpos
        physical.qvel[:] = 0
        physical.ctrl[:] = controls(data.qpos)
        mujoco.mj_forward(physical_model, physical)
        cost = 0.0
        # Native mocap_id_list/qpos_id_list are empty: no extra target cost.
        for _ in range(30):
            mujoco.mj_step(physical_model, physical)
            cost += np.linalg.norm(physical.qpos - data.qpos)
        if cost < best_cost:
            best_qpos, best_cost = data.qpos.copy(), cost
    data.qpos[:] = best_qpos
    data.qvel[:] = 0
    data.ctrl[:] = controls(data.qpos)
    mujoco.mj_step(model, data)
    result = []
    for frame in range(len(poses)):
        set_targets(frame)
        for _ in range(max(1, int(options["ref_dt"] / options["sim_dt"]))):
            mujoco.mj_step(model, data)
        result.append(data.qpos.copy())
    return np.asarray(result)


def postprocess(model, qpos, ref_dt, window):
    """Original valid moving average, adjacent differentiation and first-row drop."""
    if len(qpos) <= window:
        raise ValueError("Paper IK needs more frames than the averaging window")
    filtered = np.column_stack(
        [
            np.convolve(column, np.ones(window) / window, mode="valid")
            for column in qpos.T
        ]
    )
    qvel = np.empty((len(filtered) - 1, model.nv))
    for i in range(1, len(filtered)):
        mujoco.mj_differentiatePos(
            model, qvel[i - 1], ref_dt, filtered[i - 1], filtered[i]
        )
    return filtered[1:], qvel
