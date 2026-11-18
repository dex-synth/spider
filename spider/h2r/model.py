"""Named model layout shared by IK, reference construction and external control."""

import mujoco
import numpy as np


def finite(value, shape, name):
    """Validate a physical array before assigning it to a simulator."""
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: expected finite shape {shape}")
    return result


class Layout:
    """Resolve all dynamic joints and independent actuator ordering once."""

    def __init__(self, model, task):
        self.model, self.task = model, task
        semantics, self.binding = task.robot.semantics, task.object_binding
        self.names = semantics["joint_names"]
        self.robot_q, self.robot_v = self.scalar_indices(self.names)
        self.roots = self.binding["roots"]
        self.joints = self.binding["joints"]
        self.object_q, self.object_v = self.scalar_indices(list(self.joints.values()))
        self.root_indices = {}
        for name, root in self.roots.items():
            joint = model.joint(root["joint"])
            body = model.body(root["body"])
            if (
                joint.type[0] != mujoco.mjtJoint.mjJNT_FREE
                or joint.bodyid[0] != body.id
                or body.parentid[0] != 0
                or body.jntnum[0] != 1
            ):
                raise ValueError(f"Unsupported free-root frame: {name}")
            self.root_indices[name] = (int(joint.qposadr[0]), int(joint.dofadr[0]))
        expected = (
            set(self.names)
            | set(self.joints.values())
            | {r["joint"] for r in self.roots.values()}
        )
        if expected != {model.joint(i).name for i in range(model.njnt)}:
            raise ValueError("Unmapped or duplicate model joints")
        if len(expected) != len(self.names) + len(self.joints) + len(self.roots):
            raise ValueError("Joint ownership overlaps")
        self.actuators = []
        for name, unit in zip(self.names, semantics["joint_units"], strict=True):
            joint = model.joint(name)
            expected_unit = (
                "m" if joint.type[0] == mujoco.mjtJoint.mjJNT_SLIDE else "rad"
            )
            ids = np.flatnonzero(
                (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (model.actuator_trnid[:, 0] == joint.id)
            )
            if len(ids) != 1 or unit != expected_unit:
                raise ValueError(
                    f"Expected one position actuator and correct units: {name}"
                )
            aid = int(ids[0])
            if (
                not np.array_equal(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0])
                or model.actuator_dyntype[aid] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[aid] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_AFFINE
                or model.actuator_gainprm[aid, 0] <= 0
                or not np.isclose(
                    model.actuator_biasprm[aid, 1], -model.actuator_gainprm[aid, 0]
                )
            ):
                raise ValueError(f"Unsupported position actuator: {name}")
            self.actuators.append(aid)
        if set(self.actuators) != set(range(model.nu)):
            raise ValueError("Additional or missing actuators")
        self.actuators = np.array(self.actuators, dtype=int)
        self.passive_dofs = sorted(set(range(model.nv)) - set(self.robot_v))

    def scalar_indices(self, names):
        """Resolve scalar joint state addresses independently of ordering."""
        q, v = [], []
        for name in names:
            joint = self.model.joint(name)
            # Plain integers: `numpy scalar in (enum, ...)` is False from MuJoCo 3.17.
            if int(joint.type[0]) not in (
                int(mujoco.mjtJoint.mjJNT_SLIDE),
                int(mujoco.mjtJoint.mjJNT_HINGE),
            ):
                raise ValueError(f"Expected scalar joint: {name}")
            q.append(int(joint.qposadr[0]))
            v.append(int(joint.dofadr[0]))
        return np.array(q, dtype=int), np.array(v, dtype=int)

    def controls(self, qpos):
        """Convert generalized positions into native actuator targets."""
        ctrl = np.empty((*qpos.shape[:-1], self.model.nu))
        ctrl[..., self.actuators] = qpos[..., self.robot_q]
        return ctrl

    def targets(self, ctrl):
        """Return only the complete set of robot joint position targets."""
        ctrl = finite(ctrl, (self.model.nu,), "control")
        return dict(zip(self.names, ctrl[self.actuators].tolist(), strict=True))

    def reference_positions(self):
        """Fill every object root and joint from the task reference; keep the scene."""
        demo = self.task.demo
        qpos = np.tile(self.model.qpos0, (len(demo.time_s), 1))
        parts = {body: part for part, body in self.binding["parts"].items()}
        for name, root in self.roots.items():
            q, _ = self.root_indices[name]
            qpos[:, q : q + 3], qpos[:, q + 3 : q + 7] = demo.part_pose(
                parts[root["body"]]
            )
        keys = self.binding["reference_joint_keys"]
        for q, name in zip(self.object_q, self.joints, strict=True):
            qpos[:, q] = demo.arrays[keys[name]]
        return finite(qpos, (len(qpos), self.model.nq), "object reference")
