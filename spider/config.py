# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Define the configuration for the optimizer.

Author: Chaoyi Pan
Date: 2025-08-10
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch



@dataclass
class Config:
    # Layout labels are used by native-equation regression checks. Runtime uses
    # named H2R semantics; there are no robot, dataset or model-path defaults.
    embodiment_type: str = "h2r"

    # === SIMULATOR CONFIGURATION ===
    simulator: str = "mjwp"  # "isaac" | "mujoco" | "mjwp" | "mjwp_cons" | "mjwp_eq" | "mjwp_cons_eq" | "kinematic"
    device: str = "cuda:0"
    # Simulation timing
    sim_dt: float = 0.01  # simulation timestep
    physics_substeps: int = 1  # Internal integrator steps per state/control sample
    ctrl_dt: float = 0.4  # control timestep
    ref_dt: float = 0.02  # reference data timestep
    render_dt: float = 0.02  # rendering timestep
    horizon: float = 1.6  # planning horizon
    knot_dt: float = 0.4  # knot point spacing
    max_sim_steps: int = -1  # maximum simulation steps (-1 for unlimited)
    # Simulation constraints
    nconmax_per_env: int = 100  # max contacts per environment
    njmax_per_env: int = 350  # max joints per environment
    # Simulation annealing
    num_dyn: int = (
        1  # number of environments for annealing, used for virtual contact constraint
    )
    # Domain randomization
    num_dr: int = (
        1  # number of domain randomization groups, used for domain randomization
    )
    pair_margin_range: tuple[float, float] = (-0.005, 0.005)
    xy_offset_range: tuple[float, float] = (-0.005, 0.005)
    perturb_force: float = 0.0
    perturb_torque: float = 0.0
    contact_guidance: bool = False
    object_pos_actuator_names: list[str] = field(
        default_factory=lambda: [
            "right_object_pos_x",
            "right_object_pos_y",
            "right_object_pos_z",
            "left_object_pos_x",
            "left_object_pos_y",
            "left_object_pos_z",
        ]
    )
    object_rot_actuator_names: list[str] = field(
        default_factory=lambda: [
            "right_object_rot_x",
            "right_object_rot_y",
            "right_object_rot_z",
            "left_object_rot_x",
            "left_object_rot_y",
            "left_object_rot_z",
        ]
    )
    object_action_dims: int = 0
    object_actuator_ids: list[int] = field(default_factory=list)
    object_actuator_names: list[str] = field(default_factory=list)
    init_pos_actuator_gain: float = 10.0
    init_pos_actuator_bias: float = 10.0
    init_rot_actuator_gain: float = 0.1
    init_rot_actuator_bias: float = 0.1
    guidance_decay_ratio: float = 0.5
    gibbs_sampling: bool = False

    # === OPTIMIZER CONFIGURATION ===
    # Sampling parameters
    num_samples: int = 2048
    temperature: float = 0.3
    max_num_iterations: int = 16
    improvement_threshold: float = 0.01
    improvement_check_steps: int = 1
    # Termination parameters
    terminate_resample: bool = False
    object_pos_threshold: float = 0.1
    object_rot_threshold: float = 0.3
    max_local_retries: int = 3
    max_revert_depth: int = 3
    base_pos_threshold: float = 0.5
    base_rot_threshold: float = 0.4
    # Compilation
    use_torch_compile: bool = True  # use torch.compile for acceleration
    # Noise scheduling
    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    final_noise_scale: float = 0.1
    exploit_ratio: float = 0.01
    exploit_noise_scale: float = 0.01
    # Noise scaling by component
    joint_noise_scale: float = 0.15
    # Optional explicit native-actuator ordering (external named-model tasks).
    actuator_noise_scales: list[float] = field(default_factory=list)
    pos_noise_scale: float = 0.03
    rot_noise_scale: float = 0.03
    # Reward mode
    use_rl_reward: bool = False  # use dexmachina RL training reward formulation
    # Reward scaling
    base_pos_rew_scale: float = 1.0
    base_rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.003
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    vel_rew_scale: float = 0.0001
    terminal_rew_scale: float = 1.0
    contact_rew_scale: float = 0.0

    # === TRACE RECORDING ===
    trace_dt: float = 1 / 50.0
    num_trace_uniform_samples: int = 4
    num_trace_topk_samples: int = 2
    trace_site_ids: list = field(default_factory=list)

    # === CONTACT GUIDANCE (DERIVED) ===
    contact_order: list = field(default_factory=list)
    hand_contact_site_ids: list = field(default_factory=list)
    right_contact_indices: list = field(default_factory=list)
    left_contact_indices: list = field(default_factory=list)
    right_pos_ctrl_ids: list = field(default_factory=list)
    left_pos_ctrl_ids: list = field(default_factory=list)
    contact_len: int = 0

    # === AUTOMATICALLY SET PROPERTIES ===
    # Computed timesteps
    horizon_steps: int = -1
    knot_steps: int = -1
    ref_steps: int = -1
    ctrl_steps: int = -1
    # Model dimensions
    nq_obj: int = -1  # object DOF
    nq: int = -1  # total position DOF
    nv: int = -1  # total velocity DOF
    nu: int = -1  # total control DOF
    npair: int = -1  # total pair DOF
    # Computed tensors
    noise_scale: torch.Tensor = field(default_factory=lambda: torch.ones(1))
    beta_traj: float = -1.0
    # Runtime state
    env_params_list: list = field(default_factory=list)
    viewer_body_entity_and_ids: list = field(default_factory=list)
    output_dir: str = ""


def get_noise_scale(config: Config) -> torch.Tensor:
    """Get the noise scale for sampling.

    Args:
        config: Config

    Returns:
        Noise scale, shape (num_samples, knot_steps, nu)
    """
    noise_scale = torch.logspace(
        start=torch.log10(torch.tensor(config.first_ctrl_noise_scale)),
        end=torch.log10(torch.tensor(config.last_ctrl_noise_scale)),
        steps=int(round(config.horizon / config.knot_dt)),
        device=config.device,
        base=10,
    )[None, :, None]  # Shape: (1, num_knot_steps, 1)
    noise_scale = noise_scale.repeat(1, 1, config.nu)
    if config.actuator_noise_scales:
        scales = torch.as_tensor(config.actuator_noise_scales, device=config.device)
        if (
            scales.shape != (config.nu,)
            or not torch.isfinite(scales).all()
            or (scales < 0).any()
        ):
            raise ValueError("Expected one finite nonnegative noise scale per actuator")
        noise_scale *= scales
    elif config.embodiment_type in ["bimanual", "right", "left", "floating_hand_articulated"]:
        object_action_dims = max(int(config.object_action_dims), 0)
        robot_nu = max(int(config.nu - object_action_dims), 0)
        noise_scale[:, :, :3] *= config.pos_noise_scale
        noise_scale[:, :, 3:6] *= config.rot_noise_scale
        if config.embodiment_type == "bimanual":
            half_dof = robot_nu // 2
            noise_scale[:, :, 6:half_dof] *= config.joint_noise_scale
            noise_scale[:, :, half_dof : half_dof + 3] *= config.pos_noise_scale
            noise_scale[:, :, half_dof + 3 : half_dof + 6] *= config.rot_noise_scale
            noise_scale[:, :, half_dof + 6 : robot_nu] *= config.joint_noise_scale
        elif config.embodiment_type in ["right", "left", "floating_hand_articulated"]:
            noise_scale[:, :, 6:robot_nu] *= config.joint_noise_scale
    else:
        noise_scale *= config.joint_noise_scale
    if config.contact_guidance and config.object_actuator_ids:
        object_ids = torch.as_tensor(
            config.object_actuator_ids, device=config.device, dtype=torch.long
        )
        noise_scale[:, :, object_ids] *= 0.0
    # repeat to match num_samples; same samples used across DR groups
    noise_scale = noise_scale.repeat(config.num_samples, 1, 1)
    # set first sample to 0
    noise_scale[0] *= 0.0
    # set last few samples to exploit_noise_scale
    num_exploit_samples = int(config.num_samples * config.exploit_ratio)
    noise_scale[-num_exploit_samples:] *= config.exploit_noise_scale
    return noise_scale


def compute_steps(config: Config):
    # make sure every dt can be divided by sim_dt
    config.horizon_steps = int(np.round(config.horizon / config.sim_dt))
    config.knot_steps = int(np.round(config.knot_dt / config.sim_dt))
    config.ref_steps = int(np.round(config.ref_dt / config.sim_dt))
    config.ctrl_steps = int(np.round(config.ctrl_dt / config.sim_dt))
    assert np.isclose(
        config.horizon - config.horizon_steps * config.sim_dt, 0, atol=1e-5
    ), "horizon must be divisible by sim_dt"
    assert np.isclose(
        config.ctrl_dt - config.ctrl_steps * config.sim_dt, 0, atol=1e-5
    ), "ctrl_dt must be divisible by sim_dt"
    assert np.isclose(
        config.knot_dt - config.knot_steps * config.sim_dt, 0, atol=1e-5
    ), "knot_dt must be divisible by sim_dt"
    return config


def compute_noise_schedule(config: Config) -> Config:
    config.noise_scale = get_noise_scale(config)
    if config.max_num_iterations > 0:
        config.beta_traj = config.final_noise_scale ** (1 / config.max_num_iterations)
    else:
        config.beta_traj = 1.0
    return config
