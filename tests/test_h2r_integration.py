"""Boundary tests independent of licensed data and GPU physics."""

import copy
import socket
import struct
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
from spider.h2r.bundle import (
    digest,
    load_prepared,
    verify_files,
    verify_xml,
    write_json,
)
from spider.h2r.evaluation import run, start_request
from spider.h2r.model import Layout
from spider.h2r.protocol import receive, send
from spider.h2r.reference import hand_targets, sample_reference, velocities
from spider.h2r.settings import settings, validate_clock
from spider.h2r.tracking import Tracking
from spider.optimizers.sampling import sample_ctrls
from spider.simulators import mjwp


def fixture():
    """Use interleaved object/robot joints and reversed actuator ordering."""
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="root_a" pos="1 0 1"><freejoint name="free_a"/><geom size=".02"/>
        <body name="lid" pos="0 0 .1"><joint name="hinge"/><geom size=".01"/></body></body>
      <body name="hand"><joint name="x" type="slide" axis="1 0 0"/>
        <joint name="finger"/><geom size=".01"/></body>
      <body name="root_b" pos="-1 0 1"><freejoint name="free_b"/><geom size=".02"/></body>
      </worldbody><actuator><position joint="finger" kp="3"/><position joint="x" kp="20"/></actuator></mujoco>""")
    semantics = {
        "joint_names": ["x", "finger"],
        "joint_units": ["m", "rad"],
        "joint_groups": {
            "wrist_translation": ["x"],
            "wrist_rotation": [],
            "fingers": ["finger"],
        },
        "hands": {"right": {}},
    }
    task = {
        "task_id": "task",
        "bundle_id": "bundle",
        "scene_id": "scene",
        "robot_manifest_sha256": "robot",
        "robot_name": "synthetic",
        "dt_s": 1 / 240,
        "control_ticks": 8,
        "end_tick": 16,
        "source_frame_range": [0, 3],
        "semantics": semantics,
        "observation": {
            "bodies": ["hand", "root_a", "lid", "root_b"],
            "contacts": False,
            "quaternion": "wxyz",
            "body_velocity": "world_at_body_origin",
        },
        "object": {
            "roots": {
                "a": {"body": "root_a", "joint": "free_a"},
                "b": {"body": "root_b", "joint": "free_b"},
            },
            "parts": {"a": "root_a", "lid": "lid", "b": "root_b"},
            "joints": {"angle": "hinge"},
            "reference_joint_keys": {"angle": "joint_angle_rad"},
        },
    }
    return model, task, Layout(model, task)


class StateTests(unittest.TestCase):
    """Check manifold, velocity-frame, control-order and reference boundaries."""

    def test_named_control_and_body_origin_velocity(self):
        """Round-trip world twists and reordered joint targets through MuJoCo."""
        model, task, layout = fixture()
        data = mujoco.MjData(model)
        quat = Rotation.from_rotvec([0.3, -0.2, 0.7]).as_quat(scalar_first=True)
        frame = {
            "robot_q": [0.2, 0.1],
            "robot_dq": [0.4, 0.3],
            "object_q": [0.5],
            "object_dq": [0.6],
            "effective_target": [0.25, 0.15],
            "time_s": 0.1,
            "bodies": {
                root["body"]: [1, 2, 3, *quat, 0.7, 0.8, 0.9, 1, 2, 3]
                for root in task["object"]["roots"].values()
            },
        }
        layout.write_observation(data, frame, ["finger", "x"])
        np.testing.assert_allclose(data.qpos[layout.robot_q], [0.1, 0.2])
        np.testing.assert_allclose(data.ctrl, [0.25, 0.15])
        ctrl = layout.controls(data.qpos)
        self.assertEqual(layout.targets(ctrl), {"x": 0.1, "finger": 0.2})
        for root in task["object"]["roots"].values():
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                model.body(root["body"]).id,
                velocity,
                0,
            )
            np.testing.assert_allclose(velocity, [1, 2, 3, 0.7, 0.8, 0.9], atol=1e-12)
        broken = copy.deepcopy(frame)
        del broken["bodies"]["root_b"]
        with self.assertRaises(KeyError):
            layout.write_observation(data, broken, ["finger", "x"])

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
        """Check the exact endpoint, SLERP and zero-velocity private padding."""
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
        np.testing.assert_allclose(np.linalg.norm(p[:, q + 3 : q + 7], axis=1), 1.0)
        np.testing.assert_allclose(
            Rotation.from_quat(p[1, q + 3 : q + 7], scalar_first=True).as_rotvec(),
            [0, 0, 0.3],
        )
        np.testing.assert_allclose(c[:, 1], [0, 0.05, 0.3, 0.3])
        np.testing.assert_allclose(v[-1], 0.0)
        expected_velocity = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, expected_velocity, 0.25, p[1], p[2])
        np.testing.assert_allclose(v[2], expected_velocity)

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
        task = {
            "semantics": {
                "joint_names": names,
                "joint_units": ["m"] * 3 + ["rad"] * 4,
                "joint_groups": {
                    "wrist_translation": names[:3],
                    "wrist_rotation": names[3:6],
                    "fingers": names[6:],
                },
            },
            "object": {"root_joint": "free", "root_body": "object", "joints": {}},
        }
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
        _, task, _ = fixture()
        options = settings()["mpc"]
        validate_clock(options, task)
        for key, value in (("horizon", 0.41), ("horizon", 0.5), ("knot_dt", 0.001)):
            changed = {**options, key: value}
            if value == 0.5:
                changed["knot_dt"] = 0.2
            with self.assertRaises(ValueError):
                validate_clock(changed, task)

    def test_decoded_geometry_is_not_transformed_twice(self):
        """Apply source-to-world rotation only to raw MANO orientation."""
        root = Rotation.from_euler("z", 0.5)
        transform = np.eye(4)
        transform[:3, :3] = root.as_matrix()
        ref = {
            "mano": {
                "joint_order": "MANO 16 + thumb/index/middle/ring/little tips",
                "parameter_coordinates": "source_world",
            },
            "coordinates": {"source_to_benchmark": transform.tolist()},
        }
        joints = np.arange(126).reshape(2, 21, 3) / 100
        data = {
            "time_s": np.array([0.0, 0.1]),
            "right_joints_m": joints,
            "right_mano_quat": np.tile([1.0, 0, 0, 0], (2, 16, 1)),
            "right_mano_trans": np.zeros((2, 3)),
        }
        wrists, tips = hand_targets(
            ref, data, ["right"], {"right": {"mano_from_wrist_rotation": np.eye(3)}}
        )
        np.testing.assert_allclose(wrists["right"][:, :3], joints[:, 0])
        np.testing.assert_allclose(tips["right"], joints[:, 16:21])
        np.testing.assert_allclose(
            wrists["right"][0, 3:], root.as_quat(scalar_first=True)
        )


class FileAndProtocolTests(unittest.TestCase):
    """Protect sealed input identity and episode lifecycle."""

    def test_manifest_tamper_and_unsealed_resource(self):
        """Reject altered files, unsealed includes and path escapes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scene.xml").write_text(
                '<mujoco><include file="missing.xml"/></mujoco>'
            )
            files = {"scene.xml": digest(root / "scene.xml")}
            verify_files(root, files)
            with self.assertRaises(ValueError):
                verify_xml(root, "scene.xml", files)
            (root / "scene.xml").write_text("changed")
            with self.assertRaises(ValueError):
                verify_files(root, files)
            with self.assertRaises(ValueError):
                verify_files(root, {"../escape": "anything"})

    def test_wire_rejects_ambiguous_json(self):
        """Reject duplicate keys and non-finite JSON, accepting ordinary framing."""
        for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b"[]"):
            a, b = socket.socketpair()
            with a, b:
                a.sendall(struct.pack("!I", len(payload)) + payload)
                with self.assertRaises(ValueError):
                    receive(b)
        a, b = socket.socketpair()
        with a, b:
            send(a, {"type": "step", "commands": {}})
            self.assertEqual(receive(b), {"type": "step", "commands": {}})

    def test_binding_mismatch(self):
        """Refuse a deployment bound to another task before model loading."""
        _, task, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "h2r_binding.json",
                {"format": "spider-h2r-v1", "task_id": "wrong"},
            )
            with self.assertRaisesRegex(ValueError, "task_id"):
                load_prepared(root, task)

    def test_multi_episode_lifecycle(self):
        """Keep an active episode running after another fails initialization."""
        _, task, _ = fixture()
        hello = {
            "type": "hello",
            "protocol": "h2r-evaluation-v1",
            **{k: v for k, v in task.items() if k not in {"semantics", "scene_id"}},
            "joint_names": ["x", "finger"],
            "joint_units": ["m", "rad"],
            "episodes": [
                {"episode_id": "one", "controller_seed": 11},
                {"episode_id": "two", "controller_seed": 22},
            ],
        }
        initial = {"x": 0.0, "finger": 0.0}
        changed = copy.deepcopy(hello)
        changed["object"]["parts"].pop("lid")
        with self.assertRaises(ValueError):
            start_request(changed, task, {}, initial)
        calls = []

        class FakeController:
            def __init__(self, prepared, task, device, seed):
                self.seed = seed

            def control(self, frame, names):
                calls.append((self.seed, frame["tick"]))
                return initial, {"tick": frame["tick"]}

        errors = []

        def evaluator(stream):
            try:
                send(stream, hello)
                self.assertEqual(set(receive(stream)["initial_robot"]), {"one", "two"})
                for tick in (0, 8):
                    send(
                        stream,
                        {
                            "type": "frames",
                            "done": False,
                            "episodes": [
                                {"episode_id": "one", "tick": tick, "done": False},
                                {
                                    "episode_id": "two",
                                    "tick": 0,
                                    "done": True,
                                    "reason": "initial_collision",
                                },
                            ],
                        },
                    )
                    command = receive(stream)
                    self.assertEqual(set(command["commands"]), {"one"})
                    self.assertEqual(command["commands"]["one"]["tick"], tick)
                send(
                    stream,
                    {
                        "type": "frames",
                        "done": True,
                        "episodes": [
                            {"episode_id": "one", "tick": 16, "done": True},
                            {"episode_id": "two", "tick": 0, "done": True},
                        ],
                    },
                )
            except BaseException as error:
                errors.append(error)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "initial_robot.json", initial)
            write_json(root / "h2r_binding.json", {})
            a, b = socket.socketpair()
            with a, b:
                a.settimeout(5)
                b.settimeout(5)
                worker = threading.Thread(target=evaluator, args=(a,))
                worker.start()
                run(b, root, task, {}, root, "cpu", FakeController)
                worker.join(5)
            self.assertFalse(worker.is_alive())
            if errors:
                raise errors[0]
            self.assertEqual(calls, [(11, 0), (11, 8)])


if __name__ == "__main__":
    unittest.main()
