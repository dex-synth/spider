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
        groups = layout.task.robot.semantics["joint_groups"]
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

    def check_error(self, config, recorded_qpos, ref_qpos):
        """Native max-over-window root checks using task names instead of slices."""
        positions, rotations = [], []
        for q, _ in self.layout.root_indices.values():
            pos = torch.norm(
                recorded_qpos[:, q : q + 3] - ref_qpos[:, q : q + 3], dim=1
            )
            rot = torch.norm(
                quat_sub(recorded_qpos[:, q + 3 : q + 7], ref_qpos[:, q + 3 : q + 7]),
                dim=1,
            )
            # Preserve upstream's absent-object sentinel for bimanual inputs.
            if len(self.layout.task.hand_sides) == 2 and torch.all(
                ref_qpos[:, q : q + 3].abs() < 1e-4
            ):
                pos, rot = pos * 0, rot * 0
            positions.append(pos.max().item())
            rotations.append(rot.max().item())
        max_pos, max_rot = max(positions, default=0.0), max(rotations, default=0.0)
        exceeded = (
            max_pos > config.object_pos_threshold
            or max_rot > config.object_rot_threshold
        )
        return exceeded, {"max_pos_error": max_pos, "max_rot_error": max_rot}

    def terminate(self, config, env, ref):
        """Native root thresholds; this flag is inert with terminate_resample=False."""
        qpos = wp.to_torch(env.data_wp.qpos)
        failed = torch.zeros(qpos.shape[0], device=qpos.device, dtype=torch.bool)
        for q, _ in self.layout.root_indices.values():
            pos = torch.norm(qpos[:, q : q + 3] - ref[0][q : q + 3], dim=1)
            rot = torch.norm(
                quat_sub(
                    qpos[:, q + 3 : q + 7],
                    ref[0][q + 3 : q + 7].expand(qpos.shape[0], -1),
                ),
                dim=1,
            )
            if len(self.layout.task.hand_sides) == 2 and torch.all(
                ref[0][q : q + 3].abs() < 1e-4
            ):
                pos, rot = pos * 0, rot * 0
            failed |= (pos > config.object_pos_threshold) | (
                rot > config.object_rot_threshold
            )
        return failed


def save_parameters(config, env):
    """Nominal H2R planning has no parameter annealing state."""
    return {}


def load_parameters(config, env, parameters):
    """Reject undeclared private physics overrides instead of applying legacy slices."""
    if parameters:
        raise ValueError("H2R nominal planning does not accept parameter overrides")
    return env
