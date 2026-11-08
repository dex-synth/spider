"""Private upstream fast MPC, with best-sample replay and named task mappings."""

import time
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from spider.config import Config, compute_noise_schedule, compute_steps
from spider.optimizers.sampling_fast import (
    _restore_best_state_to_env,
    make_optimize_fn_fast,
    make_optimize_once_fn_fast,
    make_rollout_fn_fast,
)
from spider.simulators import mjwp

from .bundle import arrays, load_inputs, load_prepared, read_json
from .model import Layout
from .reference import native_reference, sample_reference
from .sanity import check_stability
from .scene import planning_model
from .settings import MPCSettings, settings, validate_clock
from .tracking import Tracking, load_parameters, save_parameters


class Controller:
    """One private fast planning pass; checkpoints never touch evaluation state."""

    def __init__(self, prepared, task_path, device, seed):
        started = time.perf_counter()
        if torch.device(device).type != "cuda":
            raise ValueError("SPIDER H2R MPC requires a CUDA device for MJWarp")
        prepared = Path(prepared)
        task, reference, data = load_inputs(task_path)
        self.binding = load_prepared(prepared, task)
        self.task = task
        source = arrays(prepared / "trajectory_kinematic.npz")
        options = settings(prepared / "resolved_config.yaml")["mpc"]
        validate_clock(options, task)
        self.config = cfg = Config(embodiment_type="h2r", device=device, **options)
        compute_steps(cfg)
        # Upstream derives this before the task_info reference-rate override.
        cfg.ref_steps = int(np.round(MPCSettings().ref_dt / cfg.sim_dt))
        model, _, _ = planning_model(
            task_path, task, read_json(prepared / "calibration.json")
        )
        cfg.nq, cfg.nv, cfg.nu, cfg.npair = model.nq, model.nv, model.nu, model.npair
        self.layout = Layout(model, task)
        self.tracking = Tracking(self.layout, cfg)
        self.source = native_reference(self.layout, source, cfg)
        compute_noise_schedule(cfg)
        # Explicit user-authorized exception: num_dr=1 is truly nominal.
        cfg.env_params_list = [[{}] for _ in range(cfg.max_num_iterations)]
        initial = self.layout.reference_positions(reference, data)[0]
        initial[self.layout.robot_q] = source["qpos"][0, self.layout.robot_q]
        initial_ctrl = self.layout.controls(initial)
        self.sanity = check_stability(model, self.layout, initial, initial_ctrl, cfg)
        self.env = mjwp.setup_env(cfg, self.reference(0), model)
        self.data = mujoco.MjData(self.env.model_cpu)
        self.data.qpos[:] = initial
        self.data.qvel[:] = 0
        self.data.ctrl[:] = initial_ctrl
        mujoco.mj_forward(self.env.model_cpu, self.data)
        self.synchronize()
        rollout = make_rollout_fn_fast(
            mjwp.step_env,
            mjwp.save_state,
            mjwp.load_state,
            self.tracking.reward,
            self.tracking.terminal_reward,
            self.tracking.terminate,
            mjwp.get_trace,
            save_parameters,
            load_parameters,
            mjwp.copy_sample_state,
            mjwp.get_qpos,
            mjwp.get_qvel,
        )
        self.optimize = make_optimize_fn_fast(
            make_optimize_once_fn_fast(rollout), self.tracking.check_error
        )
        self.device_index = torch.device(device).index
        if self.device_index is None:
            self.device_index = torch.cuda.current_device()
        self.cuda_rng = torch.Generator(device=device).manual_seed(seed).get_state()
        self.cpu_rng = torch.Generator().manual_seed(seed).get_state()
        self.controls = self.reference(0)[2][:-1].clone()
        self.last_snapshot = self.last_index = None
        self.segments = []
        self.setup_s = time.perf_counter() - started

    @property
    def step(self):
        """Native integer clock, as in run_mjwp_fast."""
        return int(np.round(self.data.time / self.config.sim_dt))

    def reference(self, step):
        """Current frame followed by the +1-lookahead native horizon."""
        cfg = self.config
        query = (step + np.arange(cfg.horizon_steps + 1)) * cfg.sim_dt
        values = sample_reference(self.layout.model, self.layout, self.source, query)
        return tuple(
            torch.as_tensor(value, dtype=torch.float32, device=cfg.device)
            for value in (
                *values,
                np.zeros((len(query), 0)),
                np.zeros((len(query), 0, 3)),
            )
        )

    def synchronize(self):
        """Initialize private worlds from the common initial proposal, without warmup."""
        cfg = self.config
        with wp.ScopedDevice(cfg.device):
            mjwarp.reset_data(self.env.model_wp, self.env.data_wp)
            for field in ("qpos", "qvel", "ctrl"):
                value = torch.as_tensor(
                    getattr(self.data, field), dtype=torch.float32, device=cfg.device
                )
                wp.copy(
                    getattr(self.env.data_wp, field),
                    wp.from_torch(value.repeat(cfg.num_samples, 1)),
                )
            wp.to_torch(self.env.data_wp.time).fill_(self.data.time)
            mjwarp.forward(self.env.model_wp, self.env.data_wp)

    def save_checkpoint(self):
        """Save the native rollback boundary, including output length."""
        return {
            **{k: getattr(self.data, k).copy() for k in ("qpos", "qvel", "ctrl")},
            "time": self.data.time,
            "ctrls": self.controls.clone(),
            "env_snap": self.last_snapshot,
            "env_idx": self.last_index,
            "segments_len": len(self.segments),
        }

    def truncate(self, checkpoint):
        """Preserve upstream's truncation to the latest pre-execution commit."""
        del self.segments[checkpoint["segments_len"] if checkpoint else 0 :]

    def load_checkpoint(self, checkpoint):
        """Restore a private snapshot and truncate provisional outputs."""
        for key in ("qpos", "qvel", "ctrl"):
            getattr(self.data, key)[:] = checkpoint[key]
        self.data.time = checkpoint["time"]
        mujoco.mj_forward(self.env.model_cpu, self.data)
        self.controls = checkpoint["ctrls"].clone()
        self.last_snapshot, self.last_index = (
            checkpoint["env_snap"],
            checkpoint["env_idx"],
        )
        if self.last_snapshot is None:
            self.synchronize()
        else:
            _restore_best_state_to_env(self.env, self.last_snapshot, self.last_index)
        self.truncate(checkpoint)

    def optimize_chunk(self):
        """Run native fixed iteration attempts, isolating this plan's random stream."""
        step = self.step
        with torch.random.fork_rng(devices=[self.device_index]):
            torch.set_rng_state(self.cpu_rng)
            torch.cuda.set_rng_state(self.cuda_rng, self.device_index)
            self.controls, info = self.optimize(
                self.config,
                self.env,
                self.controls,
                tuple(value[1:] for value in self.reference(step)),
            )
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.device_index)
        return step, info

    def replay_chunk(self, step, info):
        """Reuse selected post-step states and full snapshot; never re-simulate controls."""
        cfg = self.config
        controls = self.controls[: cfg.ctrl_steps].detach().cpu().numpy().copy()
        for i in range(cfg.ctrl_steps):
            self.data.qpos[:] = info["best_qpos"][i]
            self.data.qvel[:] = info["best_qvel"][i]
            self.data.ctrl[:] = controls[i]
            self.data.time += cfg.sim_dt
            mujoco.mj_forward(self.env.model_cpu, self.data)
        self.segments.append(
            {
                "step": step,
                "qpos": info["best_qpos"],
                "qvel": info["best_qvel"],
                "ctrl": controls,
            }
        )
        self.last_snapshot, self.last_index = info["state_snapshot"], info["best_idx"]
        _restore_best_state_to_env(self.env, self.last_snapshot, self.last_index)
        remaining = cfg.horizon_steps - cfg.ctrl_steps
        self.controls = torch.cat(
            (
                self.controls[cfg.ctrl_steps :],
                self.reference(self.step)[2][remaining:-1],
            )
        )
        return self.step

    def control(self, frame, joint_names):
        """Native rollback requires offline planning; reject online equivalence claims."""
        raise ValueError(
            "Upstream fast rollback requires --plan; online --prepared is unsupported"
        )
