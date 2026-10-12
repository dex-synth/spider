"""Common-task deployment of native SPIDER sampling MPC.

The only added task dimensions are a floating Leap root and one object hinge.
The native optimizer/reward pipeline remains in this codebase. Evaluation state
is synchronized into private planning worlds; only robot controls leave them.
"""
import json
import math
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from spider.config import Config, compute_steps, compute_noise_schedule
from spider.optimizers.sampling import make_rollout_fn, make_optimize_once_fn, make_optimize_fn
from spider.simulators import mjwp


class ExternalStateController:
    def __init__(self, prepared: Path, device="cuda:0", samples=128, iterations=1, seed=0):
        self.meta = json.loads((prepared / "input.json").read_text())
        self.joint_names = self.meta["joint_names"]
        self.cfg = Config(robot_type="leap", embodiment_type="floating_hand_articulated", device=device,
                          model_path=self.meta["planning_scene"], num_samples=samples,
                          max_num_iterations=iterations, use_torch_compile=False, num_dr=1,
                          sim_dt=1/120, physics_substeps=4, ctrl_dt=4/120, ref_dt=1/120, horizon=24/120, knot_dt=4/120,
                          nq_obj=8, sanity_check_seconds=0, contact_guidance=False, contact_rew_scale=0,
                          nconmax_per_env=256, njmax_per_env=1024, improvement_threshold=0.)
        if samples < 100 or iterations < 1:
            raise ValueError("Use >=100 samples and >=1 iterations; keeps native exploit subset nonempty")
        model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.cfg.nq, self.cfg.nv, self.cfg.nu, self.cfg.npair = model.nq, model.nv, model.nu, model.npair
        compute_steps(self.cfg)
        compute_noise_schedule(self.cfg)
        self.cfg.env_params_list = [[{}] for _ in range(iterations)]
        arrays = np.load(prepared / "reference.npz", allow_pickle=False)
        self.frame_count = len(arrays["time_s"])
        if not np.allclose(arrays["time_s"], np.arange(self.frame_count)/120, atol=1e-12, rtol=0):
            raise ValueError("Method reference clock changed")
        self.ref = []
        for key in ("qpos", "qvel", "ctrl"):
            values = arrays[key]
            tail = np.repeat(values[-1:], self.cfg.horizon_steps+1, axis=0)
            if key == "qvel":
                tail[:] = 0.
            self.ref.append(torch.tensor(np.concatenate((values, tail)), dtype=torch.float32, device=device))
        self.ref.extend([torch.zeros((len(self.ref[0]), 0), device=device),
                         torch.zeros((len(self.ref[0]), 0, 3), device=device)])
        torch.manual_seed(seed)
        self.env = mjwp.setup_env(self.cfg, tuple(self.ref))
        self.data = mujoco.MjData(self.env.model_cpu)
        rollout = make_rollout_fn(mjwp.step_env, mjwp.save_state, mjwp.load_state, mjwp.get_reward,
                                 mjwp.get_terminal_reward, mjwp.get_terminate, mjwp.get_trace,
                                 mjwp.save_env_params, mjwp.load_env_params, mjwp.copy_sample_state)
        self.optimize = make_optimize_fn(make_optimize_once_fn(rollout))
        self.tick = None
        self.replans = 0

    def _write_state(self, state):
        model = self.env.model_cpu
        for name, root_name, body_name, names in (
            ("robot", "robot_free", "robot_base", self.joint_names),
            ("object", "object_free", "object_bottom", ["scissors_joint"]),
        ):
            packet = state[name]
            if packet["joint_names"] != names:
                raise ValueError(f"Unexpected {name} joint order")
            root = np.asarray(packet["root_link_pose_com_velocity"], dtype=float)
            q = np.asarray(packet["joint_pos"], dtype=float)
            v = np.asarray(packet["joint_vel"], dtype=float)
            if root.shape != (13,) or q.shape != (len(names),) or v.shape != (len(names),):
                raise ValueError("Wrong physical state dimensions")
            if not all(np.isfinite(x).all() for x in (root, q, v)) or abs(np.linalg.norm(root[3:7])-1)>1e-4:
                raise ValueError("Invalid physical state")
            free = model.joint(root_name)
            self.data.qpos[free.qposadr[0]:free.qposadr[0]+7] = root[:7]
            rotation = Rotation.from_quat(root[3:7], scalar_first=True)
            offset = rotation.apply(model.body(body_name).ipos)
            self.data.qvel[free.dofadr[0]:free.dofadr[0]+6] = np.r_[
                root[7:10]-np.cross(root[10:], offset), rotation.inv().apply(root[10:])]
            for joint_name, pos, vel in zip(names, q, v):
                joint = model.joint(joint_name)
                self.data.qpos[joint.qposadr[0]] = pos
                self.data.qvel[joint.dofadr[0]] = vel
        self.data.qacc_warmstart[:] = 0.
        mujoco.mj_forward(model, self.data)

    def reset(self, state, seed=0):
        torch.manual_seed(seed)
        mjwarp.reset_data(self.env.model_wp, self.env.data_wp)
        mjwarp.reset_data(self.env.model_wp, self.env.data_wp_prev)
        mujoco.mj_resetData(self.env.model_cpu, self.data)
        self._write_state(state)
        self.ctrls = self.ref[2][:self.cfg.horizon_steps].clone()
        self.tick = 0
        self.replans = 0

    def control(self, state, tick, time_s):
        if type(tick) is not int or tick != self.tick or not math.isfinite(time_s) or abs(time_s-tick/120)>1e-9:
            raise ValueError("Wrong external clock; reset first and advance contiguous physical ticks")
        if tick >= self.frame_count-1:
            raise ValueError("Method reference ended")
        self._write_state(state)
        self.data.time = time_s
        update = tick % self.cfg.ctrl_steps == 0
        if update:
            if tick:
                h, c = self.cfg.horizon_steps, self.cfg.ctrl_steps
                self.ctrls = torch.cat((self.ctrls[c:], self.ref[2][tick+h-c:tick+h]))
            for field in ("qpos", "qvel"):
                value = torch.tensor(getattr(self.data, field), dtype=torch.float32, device=self.cfg.device)
                wp.copy(getattr(self.env.data_wp, field), wp.from_torch(value.repeat(self.cfg.num_samples, 1)))
            wp.to_torch(self.env.data_wp.qacc_warmstart).zero_()
            wp.to_torch(self.env.data_wp.xfrc_applied).zero_()
            wp.to_torch(self.env.data_wp.time).fill_(time_s)
            mjwarp.forward(self.env.model_wp, self.env.data_wp)
            ref_slice = tuple(r[tick+1:tick+self.cfg.horizon_steps+1] for r in self.ref)
            self.ctrls, self.last_plan_info = self.optimize(self.cfg, self.env, self.ctrls, ref_slice)
            self.replans += 1
        self.data.ctrl[:] = self.ctrls[tick % self.cfg.ctrl_steps].detach().cpu().numpy()
        # Decode this method's site servo at the actual physical state every
        # servo tick, without advancing any private physics or the task clock.
        mujoco.mj_forward(self.env.model_cpu, self.data)
        generalized = self.data.qfrc_actuator[:6]
        # Freejoint rotational generalized force is about the link origin in
        # local axes. The wire protocol applies the wrench at the body COM.
        body = self.env.model_cpu.body("robot_base")
        rotation = Rotation.from_quat(self.data.qpos[3:7], scalar_first=True)
        torque = rotation.apply(generalized[3:]) - np.cross(rotation.apply(body.ipos), generalized[:3])
        result = {"joint_names": self.joint_names, "position_targets_rad": self.data.ctrl[6:].tolist(),
                  "root_force_world_n": generalized[:3].tolist(), "root_torque_world_nm": torque.tolist(),
                  "wrench_application_point": "root_body_com", "tick": tick, "time_s": time_s,
                  "policy_updated": update}
        if not all(np.isfinite(result[field]).all() for field in
                   ("position_targets_rad", "root_force_world_n", "root_torque_world_nm")):
            raise ValueError("Non-finite MPC control")
        self.tick += 1
        return result
