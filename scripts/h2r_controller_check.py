"""Check deployed MPC against native actuator forces and evolving real states."""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

from spider.h2r_controller import ExternalStateController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trace.read_text().splitlines()[:8]]
    controller = ExternalStateController(args.prepared, seed=args.seed)
    model, data = controller.env.model_cpu, controller.data
    body = model.body("robot_base").id
    native_optimize = controller.optimize
    synchronization_errors = []
    plan_inputs, plans = [], []

    def checked_optimize(*parameters):
        plan_inputs.append({"qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
                            "ctrl": parameters[2].clone(), "rng": torch.cuda.get_rng_state().clone()})
        for key in ("qpos", "qvel"):
            actual = wp.to_torch(getattr(controller.env.data_wp, key)).cpu().numpy()
            synchronization_errors.append(float(np.max(np.abs(actual-getattr(data, key)))))
        result = native_optimize(*parameters)
        plans.append(result[0].clone())
        return result

    controller.optimize = checked_optimize
    controller.reset(records[0]["state"], args.seed)
    force_errors, joint_errors, root_state_errors, updates, commands = [], [], [], [], []
    for record in records:
        command = controller.control(record["state"], record["tick"], record["time_s"])
        commands.append(command)
        if command["policy_updated"]:
            updates.append(command["tick"])
        reconstructed = np.zeros(model.nv)
        mujoco.mj_applyFT(model, data, np.asarray(command["root_force_world_n"]),
                         np.asarray(command["root_torque_world_nm"]), data.xipos[body], body, reconstructed)
        force_errors.append(float(np.max(np.abs(reconstructed[:6]-data.qfrc_actuator[:6]))))
        # The benchmark's declared finger servo is the same kp/kd/effort map
        # as the native planning actuator at this instant.
        qindex = [model.joint(name).qposadr[0] for name in controller.joint_names]
        vindex = [model.joint(name).dofadr[0] for name in controller.joint_names]
        targets = np.clip(command["position_targets_rad"], model.actuator_ctrlrange[6:, 0],
                          model.actuator_ctrlrange[6:, 1])
        effort = np.clip(3*(targets-data.qpos[qindex])-.1*data.qvel[vindex], -.5, .5)
        joint_errors.append(float(np.max(np.abs(effort-data.qfrc_actuator[vindex]))))
        for name, body_name in (("robot", "robot_base"), ("object", "object_bottom")):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body(body_name).id, velocity, 0)
            expected = record["state"][name]["root_link_pose_com_velocity"][7:]
            root_state_errors.append(float(np.max(np.abs(np.r_[velocity[3:], velocity[:3]]-expected))))
    # Check reset inputs and report native GPU optimization repeat variation.
    # For decoding parity, hold the native plan fixed and perturb a non-replan
    # tick to verify the high-frequency servo uses its latest state.
    controller.reset(records[0]["state"], args.seed)
    repeat = controller.control(records[0]["state"], 0, 0.)
    fields = ("position_targets_rad", "root_force_world_n", "root_torque_world_nm")
    reset_error = max(float(np.max(np.abs(np.array(repeat[k])-commands[0][k]))) for k in fields)
    reset_input_error = max(float(np.max(np.abs(plan_inputs[0][key]-plan_inputs[-1][key])))
                            for key in ("qpos", "qvel"))
    reset_seed_equal = torch.equal(plan_inputs[0]["rng"], plan_inputs[-1]["rng"])
    reset_plan_equal = torch.equal(plan_inputs[0]["ctrl"], plan_inputs[-1]["ctrl"])
    controller.ctrls = plans[0].clone()
    shifted = copy.deepcopy(records[1]["state"])
    shifted["robot"]["root_link_pose_com_velocity"][0] += .001
    changed = controller.control(shifted, 1, 1/120)
    feedback_delta = float(np.linalg.norm(np.array(changed["root_force_world_n"])-commands[1]["root_force_world_n"]))
    rejected_clock = False
    try:
        controller.control(shifted, 3, 3/120)
    except ValueError:
        rejected_clock = True
    result = {"steps": len(records), "replan_ticks": updates,
              "max_generalized_force_error": max(force_errors), "max_finger_effort_error": max(joint_errors),
              "max_com_velocity_roundtrip_error": max(root_state_errors),
              "max_planner_initial_state_error": max(synchronization_errors),
              "native_repeated_optimization_control_delta": reset_error,
              "reset_state_error": reset_input_error, "reset_rng_equal": reset_seed_equal,
              "reset_initial_plan_equal": reset_plan_equal,
              "non_replan_feedback_force_delta_n": feedback_delta,
              "wrong_clock_rejected": rejected_clock,
              "scope": "native optimizer/state synchronization and actuator export, not paper performance"}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    assert max(force_errors+joint_errors+root_state_errors) < 1e-7
    assert max(synchronization_errors) < 1e-5
    assert reset_input_error == 0 and reset_seed_equal and reset_plan_equal
    assert feedback_delta > .01 and updates == [0, 4] and rejected_clock


if __name__ == "__main__":
    main()
