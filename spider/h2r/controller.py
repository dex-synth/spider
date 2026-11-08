"""Online native sampling MPC, synchronized from the independent evaluator."""

import time
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from spider.config import Config, compute_noise_schedule, compute_steps
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.simulators import mjwp

from .bundle import arrays, load_prepared, load_task, read_json
from .model import Layout
from .reference import sample_reference
from .scene import planning_model
from .settings import settings, validate_clock
from .tracking import Tracking, load_parameters, save_parameters


class Controller:
    """One episode's planning worlds, rolling controls and isolated random stream."""

    def __init__(self, prepared, task_path, device, seed):
        started = time.perf_counter()
        if torch.device(device).type != "cuda":
            raise ValueError("SPIDER H2R MPC requires a CUDA device for MJWarp")
        prepared = Path(prepared)
        task = load_task(task_path)
        self.binding = load_prepared(prepared, task)
        self.task = task
        self.source = arrays(prepared / "trajectory_kinematic.npz")
        options = settings(prepared / "resolved_config.yaml")["mpc"]
        validate_clock(options, task)
        self.config = Config(
            embodiment_type="h2r",
            device=device,
            contact_guidance=False,
            contact_rew_scale=0,
            num_dr=1,
            num_dyn=1,
            **options,
        )
        cfg = self.config
        compute_steps(cfg)
        if (
            cfg.horizon_steps < 1
            or cfg.knot_steps < 1
            or cfg.horizon_steps % cfg.knot_steps
        ):
            raise ValueError(
                "Horizon must be a positive integer number of control knots"
            )
        model, _, _ = planning_model(
            task_path, task, read_json(prepared / "calibration.json")
        )
        cfg.nq, cfg.nv, cfg.nu, cfg.npair = model.nq, model.nv, model.nu, model.npair
        self.layout = Layout(model, task)
        self.tracking = Tracking(self.layout, cfg)
        compute_noise_schedule(cfg)
        cfg.env_params_list = [[{}] for _ in range(cfg.max_num_iterations)]
        initial = self.reference(0)
        self.env = mjwp.setup_env(cfg, initial, model)
        self.data = mujoco.MjData(self.env.model_cpu)
        # Discard private setup warm-up at the first real initial/observed state.
        self.layout = Layout(self.env.model_cpu, task)
        rollout = make_rollout_fn(
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
        )
        self.optimize = make_optimize_fn(make_optimize_once_fn(rollout))
        self.device_index = torch.device(device).index
        if self.device_index is None:
            self.device_index = torch.cuda.current_device()
        self.cuda_rng = torch.Generator(device=device).manual_seed(seed).get_state()
        self.cpu_rng = torch.Generator().manual_seed(seed).get_state()
        self.controls = None
        self.chunk_start = None
        self.current_ctrl = None
        self.tick = 0
        self.setup_s = time.perf_counter() - started

    def reference(self, time_s):
        """Include the current target and the native +1-lookahead horizon."""
        cfg = self.config
        query = time_s + np.arange(cfg.horizon_steps + 1) * cfg.sim_dt
        values = sample_reference(self.layout.model, self.layout, self.source, query)
        result = [
            torch.as_tensor(value, dtype=torch.float32, device=cfg.device)
            for value in values
        ]
        result += [
            torch.zeros((len(query), 0), device=cfg.device),
            torch.zeros((len(query), 0, 3), device=cfg.device),
        ]
        return tuple(result)

    def control(self, frame, joint_names):
        """Use real observations at replan boundaries; publish a held native action."""
        tick, now = frame["tick"], frame["time_s"]
        ratio = self.config.ctrl_dt / (self.task["dt_s"] * self.task["control_ticks"])
        if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise ValueError("Online replans must fall on the evaluation control grid")
        if (
            type(tick) is not int
            or tick != self.tick
            or tick >= self.task["end_tick"]
            or not np.isfinite(now)
            or not np.isclose(now, tick * self.task["dt_s"], rtol=0, atol=1e-9)
        ):
            raise ValueError("Stale or inconsistent evaluation clock")
        self.layout.write_observation(self.data, frame, joint_names)
        step = int(np.floor(now / self.config.sim_dt + 1e-9))
        targets, diagnostic = self.action(step, external=True)
        self.tick += self.task["control_ticks"]
        return targets, {
            **diagnostic,
            "tick": tick,
            "time_s": now,
            "planning_qpos": self.data.qpos.tolist(),
            "planning_qvel": self.data.qvel.tolist(),
        }

    def synchronize(self):
        """Copy the current physical state into every private candidate world."""
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

    def advance(self):
        """Execute one native action, preserving the world's solver state."""
        cfg = self.config
        mjwp.step_env(cfg, self.env, self.current_ctrl)
        self.data.qpos[:] = mjwp.get_qpos(cfg, self.env)[0].detach().cpu().numpy()
        self.data.qvel[:] = mjwp.get_qvel(cfg, self.env)[0].detach().cpu().numpy()
        self.data.ctrl[:] = self.current_ctrl.detach().cpu().numpy()
        expected = self.data.time + cfg.sim_dt
        clock = float(wp.to_torch(self.env.data_wp.time)[0].item())
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            raise ValueError("Non-finite private planning state")
        # Native float32 accumulation is diagnostic only; action indices use the
        # integer step counter. No reset/forward between actions in a chunk.
        if not np.isclose(clock, expected, rtol=2e-4, atol=1e-5):
            raise ValueError("Private planning physics clock changed")
        self.data.time = expected
        mujoco.mj_forward(self.env.model_cpu, self.data)

    def action(self, step, *, external=False):
        """Ordinary run_mjwp loop: optimize, execute ctrl_steps, shift that chunk."""
        cfg = self.config
        started = time.perf_counter()
        replan = self.chunk_start is None or step - self.chunk_start >= cfg.ctrl_steps
        info = {"opt_steps": []}
        if replan:
            if (
                self.chunk_start is not None
                and step - self.chunk_start != cfg.ctrl_steps
            ):
                raise ValueError("Skipped native replanning boundary")
            if external or self.chunk_start is None:
                self.synchronize()
            else:
                mjwp.sync_env(cfg, self.env, self.data)
            reference = self.reference(step * cfg.sim_dt)
            if self.controls is None:
                self.controls = reference[2][:-1].clone()
            else:
                # Native runner appends [new_time + remaining : + ctrl_steps].
                remaining = cfg.horizon_steps - cfg.ctrl_steps
                self.controls = torch.cat(
                    (self.controls[cfg.ctrl_steps :], reference[2][remaining:-1])
                )
            with torch.random.fork_rng(devices=[self.device_index]):
                torch.set_rng_state(self.cpu_rng)
                torch.cuda.set_rng_state(self.cuda_rng, self.device_index)
                self.controls, info = self.optimize(
                    cfg,
                    self.env,
                    self.controls,
                    tuple(value[1:] for value in reference),
                )
                self.cpu_rng = torch.get_rng_state()
                self.cuda_rng = torch.cuda.get_rng_state(self.device_index)
            self.chunk_start = step
        self.current_ctrl = self.controls[step - self.chunk_start]
        targets = self.layout.targets(self.current_ctrl.detach().cpu().numpy())
        opt_steps = info["opt_steps"]
        return targets, {
            "native_step": step,
            "time_s": step * cfg.sim_dt,
            "replanned": replan,
            "compute_s": time.perf_counter() - started,
            "setup_s": self.setup_s if step == 0 else 0.0,
            "opt_steps": opt_steps.tolist()
            if hasattr(opt_steps, "tolist")
            else opt_steps,
        }
