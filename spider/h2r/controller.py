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

from .bundle import arrays, load_prepared
from .model import Layout
from .reference import sample_reference
from .settings import settings, validate_clock
from .tracking import Tracking, load_parameters, save_parameters


class Controller:
    """One episode's planning worlds, rolling controls and isolated random stream."""

    def __init__(self, prepared, task, device, seed):
        started = time.perf_counter()
        if torch.device(device).type != "cuda":
            raise ValueError("SPIDER H2R MPC requires a CUDA device for MJWarp")
        prepared = Path(prepared)
        self.binding = load_prepared(prepared, task)
        self.task = task
        self.source = arrays(prepared / "trajectory_kinematic.npz")
        options = settings(prepared / "resolved_config.yaml")["mpc"]
        validate_clock(options, task)
        dt = task["dt_s"] * task["control_ticks"]
        self.config = Config(
            embodiment_type="h2r",
            device=device,
            model_path=str(prepared / self.binding["planning_scene"]),
            sim_dt=dt,
            ref_dt=dt,
            ctrl_dt=dt,
            use_torch_compile=False,
            sanity_check_seconds=0,
            contact_guidance=False,
            contact_rew_scale=0,
            num_dr=1,
            num_dyn=1,
            improvement_threshold=0.0,
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
        model = mujoco.MjModel.from_xml_path(cfg.model_path)
        cfg.nq, cfg.nv, cfg.nu, cfg.npair = model.nq, model.nv, model.nu, model.npair
        self.layout = Layout(model, task)
        self.tracking = Tracking(self.layout, cfg)
        compute_noise_schedule(cfg)
        cfg.env_params_list = [[{}] for _ in range(cfg.max_num_iterations)]
        initial = self.reference(0)
        self.env = mjwp.setup_env(cfg, initial)
        self.data = mujoco.MjData(self.env.model_cpu)
        # The setup warm-up remains private; every external frame overwrites it.
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
        """Replan at each external control tick, returning named robot targets only."""
        cfg = self.config
        tick, now = frame["tick"], frame["time_s"]
        if (
            type(tick) is not int
            or tick != self.tick
            or tick >= self.task["end_tick"]
            or not np.isfinite(now)
            or not np.isclose(now, tick * self.task["dt_s"], rtol=0, atol=1e-9)
        ):
            raise ValueError("Stale or inconsistent evaluation clock")
        start = time.perf_counter()
        self.layout.write_observation(self.data, frame, joint_names)
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
            wp.to_torch(self.env.data_wp.time).fill_(now)
            mjwarp.forward(self.env.model_wp, self.env.data_wp)
        reference = self.reference(now)
        if self.controls is None:
            self.controls = reference[2][:-1].clone()
        else:
            self.controls = torch.cat((self.controls[1:], reference[2][-2:-1]))
        with torch.random.fork_rng(devices=[self.device_index]):
            torch.set_rng_state(self.cpu_rng)
            torch.cuda.set_rng_state(self.cuda_rng, self.device_index)
            self.controls, info = self.optimize(
                cfg, self.env, self.controls, tuple(value[1:] for value in reference)
            )
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.device_index)
        targets = self.layout.targets(self.controls[0].detach().cpu().numpy())
        self.tick += self.task["control_ticks"]
        diagnostic = {
            "tick": tick,
            "time_s": now,
            "compute_s": time.perf_counter() - start,
            "setup_s": self.setup_s if tick == 0 else 0.0,
            "opt_steps": info["opt_steps"].tolist(),
            "planning_qpos": self.data.qpos.tolist(),
            "planning_qvel": self.data.qvel.tolist(),
        }
        return targets, diagnostic
