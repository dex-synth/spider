"""Native SPIDER tracking equations with named joint layouts."""

import torch
import warp as wp

from spider.math import quat_sub


class Tracking:
    """Keep the native reward, replacing only fixed state/actuator slices."""

    def __init__(self, layout, config):
        self.layout = layout
        self.weights = torch.ones(layout.model.nv, device=config.device)
        self.scalar_q = [*layout.robot_q.tolist(), *layout.object_q.tolist()]
        self.scalar_v = [*layout.robot_v.tolist(), *layout.object_v.tolist()]
        groups = layout.task["semantics"]["joint_groups"]
        scales = [0.0] * layout.model.nu
        for group, noise, weight in (
            ("wrist_translation", config.pos_noise_scale, config.base_pos_rew_scale),
            ("wrist_rotation", config.rot_noise_scale, config.base_rot_rew_scale),
            ("fingers", config.joint_noise_scale, config.joint_rew_scale),
        ):
            _, dofs = layout.scalar_indices(groups[group])
            self.weights[dofs.tolist()] = weight
            for name in groups[group]:
                scales[layout.actuators[layout.names.index(name)]] = noise
        config.actuator_noise_scales = scales
        for _, v in layout.root_indices.values():
            self.weights[v : v + 3] = config.pos_rew_scale
            self.weights[v + 3 : v + 6] = config.rot_rew_scale
        self.weights[layout.object_v.tolist()] = config.rot_rew_scale

    def difference(self, actual, reference):
        """Return qpos error in nv tangent coordinates, including all free roots."""
        result = torch.zeros(
            (*actual.shape[:-1], self.layout.model.nv),
            device=actual.device,
            dtype=actual.dtype,
        )
        result[..., self.scalar_v] = (
            actual[..., self.scalar_q] - reference[..., self.scalar_q]
        )
        for q, v in self.layout.root_indices.values():
            result[..., v : v + 3] = actual[..., q : q + 3] - reference[..., q : q + 3]
            result[..., v + 3 : v + 6] = quat_sub(
                actual[..., q + 3 : q + 7], reference[..., q + 3 : q + 7]
            )
        return result

    def reward(self, config, env, ref):
        """Match mjwp.get_reward's weighted norm and velocity terms."""
        qpos = wp.to_torch(env.data_wp.qpos)
        qvel = wp.to_torch(env.data_wp.qvel)
        difference = self.difference(qpos, ref[0].expand_as(qpos))
        pos_dist = torch.linalg.vector_norm(difference * self.weights, dim=1)
        vel_dist = torch.linalg.vector_norm(qvel - ref[1], dim=1)
        qpos_reward = -pos_dist
        qvel_reward = -config.vel_rew_scale * vel_dist
        return qpos_reward + qvel_reward, {
            "qpos_dist": pos_dist,
            "qvel_dist": vel_dist,
            "qpos_rew": qpos_reward,
            "qvel_rew": qvel_reward,
        }

    def terminal_reward(self, config, env, ref):
        """Apply the unchanged native terminal multiplier."""
        reward, info = self.reward(config, env, ref)
        return config.terminal_rew_scale * reward, info

    def terminate(self, config, env, ref):
        """Flag failed private candidates without terminating an evaluation episode."""
        qpos = wp.to_torch(env.data_wp.qpos)
        delta = self.difference(qpos, ref[0].expand_as(qpos))
        failed = ~torch.isfinite(qpos).all(dim=1)
        for _, v in self.layout.root_indices.values():
            failed |= (
                torch.linalg.vector_norm(delta[:, v : v + 3], dim=1)
                > config.object_pos_threshold
            )
            failed |= (
                torch.linalg.vector_norm(delta[:, v + 3 : v + 6], dim=1)
                > config.object_rot_threshold
            )
        if len(self.layout.object_v):
            failed |= (
                delta[:, self.layout.object_v.tolist()].abs()
                > config.object_rot_threshold
            ).any(dim=1)
        return failed


def save_parameters(config, env):
    """Nominal H2R planning has no parameter annealing state."""
    return {}


def load_parameters(config, env, parameters):
    """Reject undeclared private physics overrides instead of applying legacy slices."""
    if parameters:
        raise ValueError("H2R nominal planning does not accept parameter overrides")
    return env
