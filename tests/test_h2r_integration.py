"""Boundary tests independent of licensed data and GPU physics."""

import copy
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from spider.config import Config, compute_noise_schedule, compute_steps
from spider.h2r import h2r_io
from spider.h2r.artifacts import PLAN_FORMAT, PREPARED_FORMAT, load_prepared
from spider.h2r.evaluation import run, start_request
from spider.h2r.h2r_io import client
from spider.h2r.h2r_io.files import write_json
from spider.h2r.model import Layout
from spider.h2r.reference import hand_targets, sample_reference, velocities
from spider.h2r.scene import calibration_for
from spider.h2r.settings import settings, validate_clock
from spider.h2r.tracking import Tracking
from spider.optimizers.sampling import sample_ctrls
from spider.simulators import mjwp

# Interleaved object/robot joints and reversed actuator ordering.
XML = """<mujoco><worldbody>
  <body name="root_a" pos="1 0 1"><freejoint name="free_a"/><geom size=".02"/>
    <body name="lid" pos="0 0 .1"><joint name="hinge"/><geom size=".01"/></body></body>
  <body name="hand"><joint name="x" type="slide" axis="1 0 0"/>
    <joint name="finger"/><geom size=".01"/></body>
  <body name="root_b" pos="-1 0 1"><freejoint name="free_b"/><geom size=".02"/></body>
  </worldbody><actuator><position joint="finger" kp="3"/><position joint="x" kp="20"/></actuator></mujoco>"""
SEMANTICS = {
    "joint_names": ["x", "finger"],
    "joint_units": ["m", "rad"],
    "joint_groups": {
        "wrist_translation": ["x"],
        "wrist_rotation": [],
        "fingers": ["finger"],
    },
}
OBJECTS = {
    "a": {
        "root": {"part": "a", "body": "root_a", "joint": "free_a", "motion": "free"},
        "parts": {"a": "root_a", "lid": "lid"},
        "joints": {
            "angle": {
                "type": "hinge",
                "parent": "a",
                "child": "lid",
                "mjcf_name": "hinge",
                "reference_key": "joint_angle_rad",
            }
        },
    },
    "b": {
        "root": {"part": "b", "body": "root_b", "joint": "free_b", "motion": "free"},
        "parts": {"b": "root_b"},
        "joints": {},
    },
}


def demo_content(objects, sides, frames, dataset):
    """Reference, scene and arrays of one demo, as the common format defines them."""
    parts = [part for obj in objects.values() for part in obj["parts"]]
    unit = np.array([1.0, 0, 0, 0])
    arrays = {
        "time_s": np.arange(frames) / 30,
        "part_pos_m": np.zeros((frames, len(parts), 3)),
        "part_quat_wxyz": np.tile(unit, (frames, len(parts), 1)),
    }
    for obj in objects.values():
        for joint in obj["joints"].values():
            arrays[joint["reference_key"]] = np.zeros(frames)
    for side in sides:
        joints = np.arange(frames * 63).reshape(frames, 21, 3) / 100
        arrays.update(
            {
                f"{side}_joints_m": joints,
                f"{side}_vertices_m": np.zeros((frames, 778, 3)),
                f"{side}_faces": np.zeros((2, 3), dtype=np.int32),
                f"{side}_mano_quat": np.tile(unit, (frames, 16, 1)),
                f"{side}_mano_trans": joints[:, 0],
                f"{side}_mano_shape": np.zeros(10),
            }
        )
    reference = {
        "dataset": dataset,
        "profile": "clip",
        "task": {"hand_sides": list(sides)},
        "reference_hz": 30.0,
        "duration_s": (frames - 1) / 30,
        "mano": {"convention": "h2r-mano-v1"},
        "quaternion_convention": "wxyz-unit-continuous-v1",
        "object_geometry": {"part_order": parts},
    }
    scene = {
        "interaction_objects": objects,
        "scoring_parts": parts,
        "environment": {"table": {}},
        "settings": {"timestep_s": 1 / 300},
    }
    return reference, scene, arrays


def memory_task(
    semantics=SEMANTICS,
    objects=OBJECTS,
    *,
    sides=("right",),
    frames=3,
    dataset="arctic",
    robot="synthetic",
):
    """An h2r_io.Task held in memory; identities are plain names."""
    reference, scene, arrays = demo_content(objects, sides, frames, dataset)
    root = Path("task")
    demo = h2r_io.Demo(root / "demo", "demo", reference, scene, arrays)
    hands = {"hands": {side: {} for side in sides}}
    machine = h2r_io.Robot(
        root / "robot",
        "robot",
        robot,
        tuple(sides),
        {**semantics, **hands},
        {},
        root / "robot/robot.xml",
        root / "robot/robot.urdf",
        {},
    )
    return h2r_io.Task(root, "task", demo, machine, {})


def fixture():
    """A two-object, right-hand task on the toy model."""
    model = mujoco.MjModel.from_xml_string(XML)
    task = memory_task()
    return model, task, Layout(model, task)


class StateTests(unittest.TestCase):
    """Check manifold, velocity-frame, control-order and reference boundaries."""

    def test_named_control_order_and_object_reference(self):
        """Map reordered actuators by name; fill objects from the task reference."""
        model, task, layout = fixture()
        qpos = model.qpos0.copy()
        qpos[layout.robot_q] = [0.1, 0.2]
        ctrl = layout.controls(qpos)
        np.testing.assert_allclose(ctrl, [0.2, 0.1])
        self.assertEqual(layout.targets(ctrl), {"x": 0.1, "finger": 0.2})
        arrays = task.demo.arrays
        arrays["part_pos_m"][:, 2] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        arrays["joint_angle_rad"][:] = [0.1, 0.2, 0.3]
        reference = layout.reference_positions()
        q, _ = layout.root_indices["b"]
        np.testing.assert_allclose(reference[:, q : q + 3], arrays["part_pos_m"][:, 2])
        np.testing.assert_allclose(reference[:, q + 3 : q + 7], [[1, 0, 0, 0]] * 3)
        np.testing.assert_allclose(reference[:, layout.object_q[0]], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            reference[:, layout.robot_q], np.tile(model.qpos0[layout.robot_q], (3, 1))
        )

    def test_tangent_difference_and_noise_layout(self):
        """Compare named manifold errors to MuJoCo's independent implementation."""
        model, _, layout = fixture()
        cfg = Config(
            device="cpu",
            embodiment_type="h2r",
            nv=model.nv,
            nu=model.nu,
            num_samples=100,
            horizon=0.4,
            knot_dt=0.1,
            sim_dt=1 / 30,
            ctrl_dt=1 / 30,
            pos_noise_scale=0.01,
            joint_noise_scale=0.2,
        )
        tracking = Tracking(layout, cfg)
        compute_steps(cfg)
        compute_noise_schedule(cfg)
        np.testing.assert_allclose(cfg.actuator_noise_scales, [0.2, 0.01])
        np.testing.assert_allclose(
            cfg.noise_scale[1, 0].numpy(), [0.1, 0.005], atol=1e-7
        )
        before = model.qpos0.copy()
        after = before.copy()
        mujoco.mj_integratePos(model, after, np.linspace(-0.2, 0.3, model.nv), 0.2)
        expected = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, expected, 1.0, before, after)
        actual = tracking.difference(
            torch.tensor(after[None]), torch.tensor(before[None])
        ).numpy()[0]
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_reference_keeps_endpoint_and_quaternion(self):
        """Normalize rotations while keeping scalars and endpoint velocities."""
        model, _, layout = fixture()
        times = np.array([0.0, 0.1, 0.3])
        qpos = np.tile(model.qpos0, (3, 1))
        q, _ = layout.root_indices["a"]
        qpos[:, q + 3 : q + 7] = Rotation.from_rotvec(
            [[0, 0, 0], [0, 0, 0.6], [0, 0, 1.2]]
        ).as_quat(scalar_first=True)
        qpos[:, layout.robot_q[0]] = [0, 0.1, 0.3]
        reference = {
            "time_s": times,
            "qpos": qpos,
            "qvel": velocities(model, times, qpos),
        }
        p, v, c = sample_reference(
            model, layout, reference, np.array([0, 0.05, 0.3, 0.4])
        )
        np.testing.assert_allclose(p[2:], np.tile(qpos[-1], (2, 1)))
        expected = (qpos[0] + qpos[1]) / 2
        expected[q + 3 : q + 7] /= np.linalg.norm(expected[q + 3 : q + 7])
        np.testing.assert_allclose(p[1], expected)
        np.testing.assert_allclose(np.linalg.norm(p[:, q + 3 : q + 7], axis=1), 1)
        np.testing.assert_allclose(
            Rotation.from_quat(p[1, q + 3 : q + 7], scalar_first=True).as_rotvec(),
            [0, 0, 0.3],
        )
        np.testing.assert_allclose(c[:, 1], [0, 0.05, 0.3, 0.3])
        np.testing.assert_allclose(v[-1], reference["qvel"][-1])
        np.testing.assert_allclose(
            v[1], (reference["qvel"][0] + reference["qvel"][1]) / 2
        )

    def test_native_reward_and_fixed_noise_equivalence(self):
        """Match native single-hand equations and samples on a compatible layout."""
        axes = ["1 0 0", "0 1 0", "0 0 1"]
        joints = "".join(
            f'<joint name="j{i}" type="{"slide" if i < 3 else "hinge"}" '
            f'axis="{axes[i % 3]}"/>'
            for i in range(6)
        )
        names = [f"j{i}" for i in range(7)]
        actuators = "".join(f'<position joint="{name}" kp="5"/>' for name in names)
        model = mujoco.MjModel.from_xml_string(
            f'<mujoco><worldbody><body>{joints}<geom size=".01"/>'
            '<body pos="0 0 .1"><joint name="j6"/><geom size=".01"/></body></body>'
            '<body name="object" pos="0 0 1"><freejoint name="free"/>'
            '<geom size=".02"/></body></worldbody>'
            f"<actuator>{actuators}</actuator></mujoco>"
        )
        task = memory_task(
            {
                "joint_names": names,
                "joint_units": ["m"] * 3 + ["rad"] * 4,
                "joint_groups": {
                    "wrist_translation": names[:3],
                    "wrist_rotation": names[3:6],
                    "fingers": names[6:],
                },
            },
            {
                "object": {
                    "root": {"part": "object", "body": "object", "joint": "free"},
                    "parts": {"object": "object"},
                    "joints": {},
                }
            },
        )
        native = Config(
            device="cpu",
            embodiment_type="right",
            nq=model.nq,
            nv=model.nv,
            nu=model.nu,
            nq_obj=7,
            num_samples=100,
            horizon=0.4,
            knot_dt=0.1,
            sim_dt=1 / 30,
            ctrl_dt=1 / 30,
            use_torch_compile=False,
        )
        adapted = copy.deepcopy(native)
        adapted.embodiment_type = "h2r"
        tracking = Tracking(Layout(model, task), adapted)
        for cfg in (native, adapted):
            compute_steps(cfg)
            compute_noise_schedule(cfg)
        torch.testing.assert_close(native.noise_scale, adapted.noise_scale)
        controls = torch.zeros((native.horizon_steps, model.nu))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37)
            a = sample_ctrls(native, controls, {})
            torch.manual_seed(37)
            b = sample_ctrls(adapted, controls, {})
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        qpos = np.tile(model.qpos0, (4, 1))
        qvel = np.linspace(-0.4, 0.5, 4 * model.nv).reshape(4, model.nv)
        for row, velocity in zip(qpos, qvel, strict=True):
            mujoco.mj_integratePos(model, row, velocity, 0.2)
        env = SimpleNamespace(
            data_wp=SimpleNamespace(
                qpos=wp.array(qpos.astype(np.float32), device="cpu"),
                qvel=wp.array(qvel.astype(np.float32), device="cpu"),
            )
        )
        ref = (
            torch.tensor(model.qpos0, dtype=torch.float32),
            torch.zeros(model.nv),
            None,
            None,
            None,
        )
        for original, mapped in (
            (mjwp.get_reward, tracking.reward),
            (mjwp.get_terminal_reward, tracking.terminal_reward),
        ):
            expected, _ = original(native, env, ref)
            actual, _ = mapped(adapted, env, ref)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_control_grid(self):
        """Reject rounding or fractional knots before expensive preparation."""
        options = settings()["mpc"]
        validate_clock(options)
        for key, value in (("horizon", 0.41), ("horizon", 0.5), ("knot_dt", 0.001)):
            changed = {**options, key: value}
            if value == 0.5:
                changed["knot_dt"] = 0.2
            with self.assertRaises(ValueError):
                validate_clock(changed)

    def test_hand_targets_for_each_hand_set(self):
        """World MANO wrist rotation times the calibration; decoded wrist and tips."""
        wrist = Rotation.from_euler("z", 0.5)
        frame = Rotation.from_euler("x", 0.3)
        calibration = {
            side: {"mano_from_wrist_rotation": frame.as_matrix()}
            for side in ("left", "right")
        }
        for sides in (("right",), ("left",), ("left", "right")):
            task = memory_task(sides=sides)
            for side in sides:
                task.demo.arrays[f"{side}_mano_quat"][:, 0] = wrist.as_quat(
                    scalar_first=True
                )
            wrists, tips = hand_targets(task, calibration)
            self.assertEqual(tuple(wrists), sides)
            for side in sides:
                joints = task.demo.arrays[f"{side}_joints_m"]
                np.testing.assert_allclose(wrists[side][:, :3], joints[:, 0])
                np.testing.assert_allclose(tips[side], joints[:, 16:21])
                np.testing.assert_allclose(
                    wrists[side][0, 3:], (wrist * frame).as_quat(scalar_first=True)
                )

    def test_calibration_by_robot_name_and_task_hands(self):
        """Select frames by robot name; every participating hand needs one."""
        for sides in (("right",), ("left",), ("left", "right")):
            calibration = calibration_for(memory_task(sides=sides, robot="sharpa_wave"))
            for side in sides:
                self.assertEqual(len(calibration[side]["fingertip_bodies"]), 5)
        with self.assertRaisesRegex(ValueError, "calibration for robot"):
            calibration_for(memory_task())


def plan_artifact(task, targets, control_dt_s=0.01):
    """A plan manifest, arrays and initial proposal as `load_plan` returns them."""
    names = task.robot.semantics["joint_names"]
    binding = {
        "format": PLAN_FORMAT,
        "plan_id": "plan",
        "deployment": "offline-control-replay",
        "joint_names": names,
        "control_dt_s": control_dt_s,
    }
    return binding, {"joint_targets": np.asarray(targets)}, dict.fromkeys(names, 0.0)


def hello_for(task, *, dt_s=1 / 300, end_tick=20, episodes=("one", "two")):
    """The evaluator's announcement of a task, without any control period."""
    semantics = task.robot.semantics
    return {
        "type": "hello",
        "protocol": client.PROTOCOL,
        "task_id": task.task_id,
        "demo_id": task.demo.demo_id,
        "robot_id": task.robot.robot_id,
        "dt_s": dt_s,
        "end_tick": end_tick,
        "joint_names": semantics["joint_names"],
        "joint_units": semantics["joint_units"],
        "episodes": [
            {"episode_id": name, "controller_seed": i}
            for i, name in enumerate(episodes)
        ],
    }


class ArtifactAndProtocolTests(unittest.TestCase):
    """Protect artifact identity and episode lifecycle."""

    def test_binding_mismatch_and_old_format(self):
        """Refuse another task's or an older generation's preparation."""
        task = memory_task()
        identities = {"task_id": "task", "demo_id": "demo", "robot_id": "robot"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for changed in identities:
                write_json(
                    root / "h2r_binding.json",
                    {"format": PREPARED_FORMAT, **identities, changed: "other"},
                )
                with self.assertRaisesRegex(ValueError, f"another task: {changed}"):
                    load_prepared(root, task)
            write_json(
                root / "h2r_binding.json",
                {"format": "spider-h2r-v4", "task_id": "task", "bundle_id": "b"},
            )
            with self.assertRaisesRegex(ValueError, "requires spider-h2r-v5"):
                load_prepared(root, task)

    def test_start_declares_native_period_and_checks_task(self):
        """Bind to the announced identities; reject another task or robot."""
        task = memory_task()
        binding, _, initial = plan_artifact(task, np.zeros((7, 2)))
        hello = hello_for(task)
        request = start_request(hello, task, binding, initial)
        self.assertEqual(
            request["requirements"],
            {"bodies": [], "contacts": False, "control_dt_s": 0.01},
        )
        self.assertEqual(
            request["binding"],
            {
                "demo_id": "demo",
                "robot_id": "robot",
                "method": "spider",
                "controller": binding,
            },
        )
        self.assertEqual(set(request["initial_robot"]), {"one", "two"})
        for change in (
            {"protocol": "h2r-evaluation-v1"},
            {"task_id": "other"},
            {"joint_names": ["x", "thumb"]},
            {"joint_units": ["rad", "rad"]},
            {"dt_s": 1 / 240},  # .01 s is not a whole number of 240 Hz steps.
        ):
            with self.assertRaises(ValueError):
                start_request({**hello, **change}, task, binding, initial)

    def test_multi_episode_lifecycle(self):
        """Keep an active episode running after another fails initialization."""
        task = memory_task()
        targets = np.arange(14.0).reshape(7, 2)
        plan = plan_artifact(task, targets)
        hello = hello_for(task)
        errors, received = [], []

        def evaluator(stream):
            try:
                client.send(stream, hello)
                start = client.receive(stream)
                self.assertEqual(set(start["initial_robot"]), {"one", "two"})
                rejected = {
                    "episode_id": "two",
                    "tick": 0,
                    "done": True,
                    "reason": "initial_collision",
                }
                for tick in (0, 3):  # .01 s control on 300 Hz physics.
                    client.send(
                        stream,
                        {
                            "type": "frames",
                            "done": False,
                            "episodes": [
                                {"episode_id": "one", "tick": tick, "done": False},
                                rejected,
                            ],
                        },
                    )
                    command = client.receive(stream)
                    self.assertEqual(set(command["commands"]), {"one"})
                    received.append(command["commands"]["one"])
                client.send(
                    stream,
                    {
                        "type": "frames",
                        "done": True,
                        "episodes": [
                            {"episode_id": "one", "tick": 20, "done": True},
                            rejected,
                        ],
                    },
                )
            except BaseException as error:
                errors.append(error)

        with tempfile.TemporaryDirectory() as tmp:
            a, b = socket.socketpair()
            with a, b:
                a.settimeout(5)
                b.settimeout(5)
                worker = threading.Thread(target=evaluator, args=(a,))
                worker.start()
                result = run(b, task, plan, Path(tmp))
                worker.join(5)
            self.assertFalse(worker.is_alive())
            if errors:
                raise errors[0]
            self.assertEqual(result["control_ticks"], 3)
            self.assertEqual(result["episodes"]["two"]["reason"], "initial_collision")
            self.assertEqual(
                received,
                [
                    {"tick": 0, "joint_targets": {"x": 0.0, "finger": 1.0}},
                    {"tick": 3, "joint_targets": {"x": 2.0, "finger": 3.0}},
                ],
            )


if __name__ == "__main__":
    unittest.main()
