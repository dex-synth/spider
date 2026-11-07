"""Named model layout shared by IK, reference construction and external control."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def finite(value, shape, name):
    """Validate a physical array before assigning it to a simulator."""
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: expected finite shape {shape}")
    return result


def quaternion(value):
    """Validate a unit wxyz quaternion without silently normalizing bad input."""
    value = finite(value, (4,), "quaternion")
    if not np.isclose(np.linalg.norm(value), 1, atol=1e-4, rtol=0):
        raise ValueError("Non-unit quaternion")
    return Rotation.from_quat(value, scalar_first=True)


class Layout:
    """Resolve all dynamic joints and independent actuator ordering once."""

    def __init__(self, model, task):
        self.model, self.task = model, task
        self.names = task["semantics"]["joint_names"]
        self.robot_q, self.robot_v = self.scalar_indices(self.names)
        obj = task["object"]
        self.roots = obj.get("roots", {})
        if "root_joint" in obj:
            self.roots = {
                "object": {"joint": obj["root_joint"], "body": obj["root_body"]}
            }
        self.joints = obj["joints"]
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
        for name, unit in zip(
            self.names, task["semantics"]["joint_units"], strict=True
        ):
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
            if joint.type[0] not in (
                mujoco.mjtJoint.mjJNT_SLIDE,
                mujoco.mjtJoint.mjJNT_HINGE,
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

    def reference_positions(self, reference, arrays):
        """Fill all objects from their named root/hinge reference, preserving the scene."""
        qpos = np.tile(self.model.qpos0, (len(arrays["time_s"]), 1))
        obj = self.task["object"]
        order = reference["object_geometry"]["part_order"]
        for name, root in self.roots.items():
            parts = [p for p, b in obj["parts"].items() if b == root["body"]]
            if len(parts) != 1:
                raise ValueError(f"Missing root reference part: {name}")
            index = order.index(parts[0])
            q, _ = self.root_indices[name]
            qpos[:, q : q + 3] = arrays["part_pos_m"][:, index]
            qpos[:, q + 3 : q + 7] = arrays["part_quat_wxyz"][:, index]
            for quat in qpos[:, q + 3 : q + 7]:
                quaternion(quat)
        keys = obj.get("reference_joint_keys", {"top_hinge": "object_hinge_rad"})
        for i, name in enumerate(self.joints):
            qpos[:, self.object_q[i]] = arrays[keys[name]]
        return finite(qpos, (len(qpos), self.model.nq), "object reference")

    def write_observation(self, data, frame, joint_names):
        """Synchronize private planning state using allowed body-origin observations."""
        if len(joint_names) != len(set(joint_names)) or set(joint_names) != set(
            self.names
        ):
            raise ValueError("Observed robot joint names differ")
        order = [joint_names.index(name) for name in self.names]
        mujoco.mj_resetData(self.model, data)
        data.qpos[self.robot_q] = finite(frame["robot_q"], (len(order),), "robot_q")[
            order
        ]
        data.qvel[self.robot_v] = finite(frame["robot_dq"], (len(order),), "robot_dq")[
            order
        ]
        data.qpos[self.object_q] = finite(
            frame["object_q"], (len(self.joints),), "object_q"
        )
        data.qvel[self.object_v] = finite(
            frame["object_dq"], (len(self.joints),), "object_dq"
        )
        for name, root in self.roots.items():
            body = finite(frame["bodies"][root["body"]], (13,), root["body"])
            rotation = quaternion(body[3:7])
            q, v = self.root_indices[name]
            data.qpos[q : q + 7] = body[:7]
            data.qvel[v : v + 3] = body[7:10]
            data.qvel[v + 3 : v + 6] = rotation.inv().apply(body[10:13])
        data.ctrl[self.actuators] = finite(
            frame["effective_target"], (len(order),), "effective_target"
        )[order]
        data.time = float(frame["time_s"])
        if not np.isfinite(data.time):
            raise ValueError("Invalid observation time")
        mujoco.mj_forward(self.model, data)
