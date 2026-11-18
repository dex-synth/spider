"""H2R twin of upstream's CPU simulator, for the benchmark's scalar-wrist robots.

It has the interface of :mod:`spider.simulators.cpu`, so `examples/run_cpu.py`'s
`track` drives it unchanged (as upstream's own GPU twin does). Two things differ
from upstream's simulator, both because the task's robot has three serial wrist
hinges with position actuators instead of a ball wrist with a torque law:

- the wrist rotation residual is added to the three hinge targets, and no torque
  is applied;
- the wrist rotation term of the tracking cost reads the wrist body's orientation
  (the hinge angles themselves are no rotation distance).

Everything else is upstream's function, re-exported.
"""

import mujoco
import numpy as np
import torch

from spider.simulators import cpu
from spider.simulators.cpu import (  # noqa: F401  (the simulator interface)
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    run_arrays,
    save_env_params,
    save_state,
    set_qpos,
    sync_env,
)


def build_env(config, model, scene, initial_qpos, num_threads=0):
    """Worlds of the private model at the task's initial state, at rest.

    Upstream starts every world on row 0 of the reference. Here row 0 is the end
    of the first control frame (`cpu_scene.tables`), so the worlds are put on the
    initial state itself: the pose the evaluation starts from.
    """
    env = cpu.setup_env(config, None, num_threads=num_threads, model=model, scene=scene)
    env.batch.reset(stage=int(mujoco.mjtStage.mjSTAGE_NONE))
    env.qpos[:] = initial_qpos
    env.qvel[:] = 0.0
    env.batch.forward(stage=int(mujoco.mjtStage.mjSTAGE_POS))
    env.filtered[:] = 0.0
    env.timestep[:] = 0
    env.frame[:] = 0
    env.failed[:] = False
    return env


def reference_data(config, env):
    """The reference as `track` takes it, as upstream's `load_data` gives it."""
    frames = env.num_frames
    qvel = np.zeros((frames, env.nv))
    qvel[:, env.scene["vel_dof"]] = env.qvel_ref
    return (
        torch.from_numpy(env.mj_qpos_ref.astype(np.float64)).to(config.device),
        torch.from_numpy(qvel).to(config.device),
        torch.zeros(frames, env.nu, device=config.device),
        torch.zeros(frames, 0, device=config.device),
        torch.zeros(frames, 0, 3, device=config.device),
    )


def _command(env, ctrl):
    """Residual action -> filtered residual -> position targets of every joint."""
    scene = env.scene
    ema, clip = float(scene["action_ema"]), scene["action_clip"]
    env.filtered[:] = np.clip(ema * env.filtered + (1.0 - ema) * ctrl, -clip, clip)
    env.frame[:] = np.clip(env.timestep, 0, env.num_frames - 1)
    target = scene["targets0"][env.frame]  # (N, nv), a copy
    target[:, scene["residual_dof"]] += env.filtered
    env.ctrl[:, scene["position_actuator"]] = target[:, scene["position_actuator_dof"]]


def step_env(config, env, ctrl):
    """Advance one control frame with residual controls of shape (N, nu) or (nu,)."""
    del config
    from mujoco import batch

    command = ctrl.detach().cpu().numpy().astype(np.float64)
    if command.ndim == 1:
        command = np.broadcast_to(command, (env.num_worlds, env.nu))
    _command(env, command)
    for substep in range(env.substeps):
        # Body poses are wanted at the new state after the last substep only.
        last = substep == env.substeps - 1
        stage = mujoco.mjtStage.mjSTAGE_POS if last else mujoco.mjtStage.mjSTAGE_NONE
        try:
            env.batch.step(nstep=1, stage=int(stage))
        except batch.SimulationError as error:
            env.failed[np.asarray(error.ids, dtype=np.int64)] = True
    env.timestep[:] = np.minimum(env.timestep + 1, env.num_frames - 1)


def wrist_rotation_errors(env):
    """Angle [rad] between each wrist body and its reference, (N, hands)."""
    wrists = env.scene["wrist_body"]
    _, angle = cpu._pose_errors(
        env.xpos[:, wrists],
        cpu._wxyz_to_xyzw(env.xquat[:, wrists]),
        env.wrist_pos_ref[env.frame],
        env.wrist_quat_ref[env.frame],
    )
    return angle


def get_reward(config, env, ref=None):
    """Per-world SPIDER tracking reward of the last transition.

    Upstream's cost is the norm of one weighted vector that holds, among others,
    the wrist rotation vector times `base_rot_rew_scale`. Adding that term's
    square under the root is the same norm.
    """
    del ref
    qpos_dist, qvel_dist = cpu.spider_cost(config, env)
    wrist = config.base_rot_rew_scale * wrist_rotation_errors(env)
    qpos_dist = np.sqrt(qpos_dist**2 + (wrist**2).sum(axis=1))
    reward = -qpos_dist - config.vel_rew_scale * qvel_dist
    reward[env.failed] = -np.inf
    pos, rot = cpu._object_errors(env)
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()  # noqa: E731
    info = {
        "qpos_dist": tensor(qpos_dist),
        "object_pos_err": tensor(pos.max(axis=1)),
        "object_rot_err": tensor(rot.max(axis=1)),
    }
    return tensor(reward), info


def get_terminal_reward(config, env, ref=None):
    """Reward of the last rollout step."""
    reward, info = get_reward(config, env, ref)
    return config.terminal_rew_scale * reward, info


def read_step(config, env, world=0):
    """Record of the last transition in one world, with the targets it was sent.

    `joint_target` holds the final position target of every robot joint during
    that control frame, in the task's joint order: what a replay submits.
    """
    record = cpu.read_step(config, env, world)
    reward, _ = get_reward(config, env)
    record["reward"] = reward[world].numpy()
    record["joint_target"] = env.ctrl[world, env.scene["position_actuator"]].copy()
    return record
