"""The CPU recipe on a toy task: reference import, tables, private world, sealing.

Runs in the CPU recipe's environment (`scripts/h2r_cpu_environment.sh`); skipped
where MuJoCo has no `mujoco.batch`.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import yaml

from spider.h2r import h2r_io
from spider.h2r.artifacts import (
    PLAN_FORMAT,
    PREPARED_FILES,
    PREPARED_FORMAT,
    REFERENCE_MINK,
    REFERENCE_V2D,
    arrays,
    validate_prepared,
)
from spider.h2r.h2r_io.files import read_json, sha256, write_json
from spider.h2r.model import Layout

BATCH = importlib.util.find_spec("mujoco.batch") is not None

# One hand (three slides, three hinges, two fingers) with reversed actuator order,
# and a box with a lid. No gravity: nothing moves unless the hand moves it.
XML = """<mujoco><option gravity="0 0 0" timestep="0.002" integrator="implicitfast"/>
  <default><default class="hand"><joint armature=".02"/></default></default><worldbody>
  <body name="box" pos=".3 0 .1"><freejoint name="box_free"/>
    <geom type="box" size=".03 .03 .03" mass=".1"/>
    <body name="lid" pos="0 0 .04"><joint name="lid_hinge" axis="0 1 0" damping=".01"/>
      <geom type="box" size=".03 .03 .005" mass=".02"/></body></body>
  <body name="palm" childclass="hand">
    <joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/>
    <joint name="z" type="slide" axis="0 0 1"/><joint name="roll" axis="1 0 0"/>
    <joint name="pitch" axis="0 1 0"/><joint name="yaw" axis="0 0 1"/>
    <geom type="box" size=".02 .02 .01" mass=".2"/>
    <body name="finger_a" pos=".03 0 0"><joint name="fa" axis="0 1 0"/>
      <geom type="capsule" size=".005 .02" mass=".01"/></body>
    <body name="finger_b" pos="-.03 0 0"><joint name="fb" axis="0 1 0"/>
      <geom type="capsule" size=".005 .02" mass=".01"/></body></body>
  </worldbody><actuator>
  <position joint="fb" kp="5" kv=".5"/><position joint="fa" kp="5" kv=".5"/>
  <position joint="yaw" kp="20" kv="2"/><position joint="pitch" kp="20" kv="2"/>
  <position joint="roll" kp="20" kv="2"/><position joint="z" kp="300" kv="20"/>
  <position joint="y" kp="300" kv="20"/><position joint="x" kp="300" kv="20"/>
  </actuator></mujoco>"""
TRANSLATION, ROTATION, FINGERS = ["x", "y", "z"], ["roll", "pitch", "yaw"], ["fa", "fb"]
NAMES = TRANSLATION + ROTATION + FINGERS
SEMANTICS = {
    "joint_names": NAMES,
    "joint_units": ["m"] * 3 + ["rad"] * 5,
    "joint_groups": {
        "wrist_translation": TRANSLATION,
        "wrist_rotation": ROTATION,
        "fingers": FINGERS,
    },
    "hands": {
        "right": {
            "wrist_body": "palm",
            "wrist_translation_joints": TRANSLATION,
            "wrist_rotation_joints": ROTATION,
            "finger_joints": FINGERS,
        }
    },
}
OBJECTS = {
    "box": {
        "root": {"part": "bottom", "body": "box", "joint": "box_free"},
        "parts": {"bottom": "box", "top": "lid"},
        "joints": {
            "hinge": {
                "parent": "bottom",
                "child": "top",
                "mjcf_name": "lid_hinge",
                "reference_key": "joint_hinge_rad",
            }
        },
    }
}
FRAMES = 16  # 30 Hz: half a second, ten control frames of 0.05 s


def compiled(kv_scale=1.0):
    """The task's record of its compiled joints, as compiled.json holds it."""
    model = mujoco.MjModel.from_xml_string(XML)
    robot = {}
    for name in NAMES:
        joint = model.joint(name)
        actuator = int(np.flatnonzero(model.actuator_trnid[:, 0] == joint.id)[0])
        dof = int(joint.dofadr[0])
        robot[name] = {
            "kp": float(model.actuator_gainprm[actuator, 0]),
            "kv": float(-model.actuator_biasprm[actuator, 2]) * kv_scale,
            "friction": float(model.dof_frictionloss[dof]),
            "damping": float(model.dof_damping[dof]),
            "armature": float(model.dof_armature[dof]),
        }
    hinge = {"friction": 0.0, "damping": 0.01, "armature": 0.0}
    return {"robot": {"joints": robot}, "base_scene": {"joints": {"hinge": hinge}}}


def toy_task(root, *, kv_scale=1.0, object_shift=0.0):
    """A task on disk as far as the CPU recipe reads it: the scene and its record."""
    root = Path(root) / "task"
    root.mkdir(exist_ok=True)
    (root / "scene.xml").write_text(XML)
    times = np.arange(FRAMES) / 30
    position = np.tile([0.3, 0.0, 0.1], (FRAMES, 1))
    position[:, 0] += object_shift * times / times[-1]
    lid = position + [0, 0, 0.04]
    unit = np.tile([1.0, 0, 0, 0], (FRAMES, 2, 1))
    arrays_ = {
        "time_s": times,
        "part_pos_m": np.stack([position, lid], axis=1),
        "part_quat_wxyz": unit,
        "joint_hinge_rad": np.zeros(FRAMES),
    }
    reference = {
        "dataset": "arctic",
        "profile": "clip",
        "task": {"hand_sides": ["right"]},
        "reference_hz": 30.0,
        "duration_s": float(times[-1]),
        "object_geometry": {"part_order": ["bottom", "top"]},
    }
    scene = {"interaction_objects": OBJECTS, "scoring_parts": ["bottom", "top"]}
    demo = h2r_io.Demo(root / "demo", "demo", reference, scene, arrays_)
    robot = h2r_io.Robot(
        root / "robot", "robot", "toy", ("right",), SEMANTICS, {}, None, None, {}
    )
    return h2r_io.Task(root, "task", demo, robot, compiled(kv_scale))


def joint_reference():
    """A smooth robot reference in the task's joint order, (FRAMES, 8)."""
    ramp = np.linspace(0, 1, FRAMES)[:, None]
    return ramp * np.array([0.02, -0.01, 0.015, 0.1, -0.05, 0.08, 0.2, -0.15])


def flashchord_preparation(root, task, joint_q, *, shuffle=True):
    """A FlashCHORD preparation as far as the import reads it."""
    root = Path(root)
    root.mkdir()
    order = NAMES[::-1] if shuffle else NAMES
    np.savez(
        root / "robot_reference.npz",
        joint_q=joint_q[:, [NAMES.index(name) for name in order]],
        joint_names=np.array(order),
        time_s=task.demo.time_s,
    )
    binding = {
        "format": "flashchord-h2r-v3",
        "task_id": "task",
        "demo_id": "demo",
        "robot_id": "robot",
        "method": {"mano_to_robot_scale": 1.0},
        "files": {"robot_reference.npz": sha256(root / "robot_reference.npz")},
    }
    write_json(root / "h2r_binding.json", binding)
    return root


class ImportTests(unittest.TestCase):
    """The V2D reference is taken as is, from a preparation of the same task."""

    def run_import(self, root, task, source):
        """Import into `root/prepared` with the in-memory task."""
        from spider.h2r.prepare import import_reference

        with patch("spider.h2r.prepare.load_task", return_value=task):
            return import_reference(task.path, source, Path(root) / "prepared")

    def test_reference_is_imported_by_name_with_the_task_objects(self):
        """Joints map by name; objects, pre-check and provenance are sealed."""
        with tempfile.TemporaryDirectory() as root:
            task, joint_q = toy_task(root), joint_reference()
            source = flashchord_preparation(Path(root) / "fc", task, joint_q)
            prepared = self.run_import(root, task, source)
            binding = read_json(prepared / "h2r_binding.json")
            self.assertEqual(binding["format"], PREPARED_FORMAT)
            self.assertEqual(binding["reference"], REFERENCE_V2D)
            self.assertEqual(set(binding["files"]), PREPARED_FILES[REFERENCE_V2D])
            validate_prepared(binding, task, REFERENCE_V2D)
            with self.assertRaisesRegex(ValueError, "needs a spider-mink-fast"):
                validate_prepared(binding, task, REFERENCE_MINK)
            model = mujoco.MjModel.from_xml_string(XML)
            layout = Layout(model, task)
            data = arrays(prepared / "trajectory_kinematic.npz")
            np.testing.assert_array_equal(data["qpos"][:, layout.robot_q], joint_q)
            box = model.joint("box_free").qposadr[0]
            np.testing.assert_allclose(
                data["qpos"][:, box : box + 3], np.tile([0.3, 0, 0.1], (FRAMES, 1))
            )
            self.assertEqual(
                read_json(prepared / "initial_robot.json"),
                dict(zip(NAMES, joint_q[0].tolist(), strict=True)),
            )
            self.assertTrue(read_json(prepared / "initialization.json")["accepted"])
            source_record = read_json(prepared / "reference_source.json")
            self.assertEqual(
                source_record["robot_reference_sha256"],
                sha256(source / "robot_reference.npz"),
            )

    def test_other_task_changed_file_or_wrist_jump_rejected(self):
        """Nothing is imported from another task, a changed file or a wrapped angle."""
        with tempfile.TemporaryDirectory() as root:
            task, joint_q = toy_task(root), joint_reference()
            source = flashchord_preparation(Path(root) / "fc", task, joint_q)
            binding = read_json(source / "h2r_binding.json")
            write_json(source / "h2r_binding.json", {**binding, "task_id": "other"})
            with self.assertRaisesRegex(ValueError, "another task: task_id"):
                self.run_import(root, task, source)
            write_json(
                source / "h2r_binding.json",
                {**binding, "files": {"robot_reference.npz": "0" * 64}},
            )
            with self.assertRaisesRegex(ValueError, "differs from its FlashCHORD"):
                self.run_import(root, task, source)
            write_json(source / "h2r_binding.json", {**binding, "format": "v1"})
            with self.assertRaisesRegex(ValueError, "Unsupported FlashCHORD"):
                self.run_import(root, task, source)
            jump = joint_q.copy()
            jump[8:, NAMES.index("yaw")] += 2 * np.pi
            wrapped = flashchord_preparation(Path(root) / "jump", task, jump)
            with self.assertRaisesRegex(ValueError, "wrist angles jump"):
                self.run_import(root, task, wrapped)
            self.assertFalse((Path(root) / "prepared").exists())


def write_prepared(root, task, joint_q, accepted=True):
    """A sealed V2D preparation of the toy task and its binding."""
    root = Path(root)
    root.mkdir()
    model = mujoco.MjModel.from_xml_string(XML)
    layout = Layout(model, task)
    qpos = layout.reference_positions()
    qpos[:, layout.robot_q] = joint_q
    np.savez(root / "trajectory_kinematic.npz", time_s=task.demo.time_s, qpos=qpos)
    write_json(
        root / "initial_robot.json", dict(zip(NAMES, joint_q[0].tolist(), strict=True))
    )
    write_json(root / "initialization.json", {"accepted": accepted})
    write_json(root / "reference_source.json", {"kind": REFERENCE_V2D})
    binding = {
        "format": PREPARED_FORMAT,
        "task_id": "task",
        "demo_id": "demo",
        "robot_id": "robot",
        "reference": REFERENCE_V2D,
        "files": {p.name: sha256(p) for p in sorted(root.iterdir())},
    }
    write_json(root / "h2r_binding.json", binding)
    return root


@unittest.skipUnless(BATCH, "needs mujoco.batch (MuJoCo >= 3.17)")
class SceneAndWorldTests(unittest.TestCase):
    """Tables by joint name, the evaluation's damping, and the scalar-wrist law."""

    def setUp(self):
        """The recipe's default settings and a directory for the toy task."""
        from spider.h2r import cpu_plan

        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.options = cpu_plan.settings()
        self.cpu_plan = cpu_plan

    def scene(self, task):
        """Private model and tables of the toy task with the smooth reference."""
        from spider.h2r.cpu_scene import private_model, tables

        model, layout, changed = private_model(task, 0.01)
        qpos = layout.reference_positions()
        qpos[:, layout.robot_q] = joint_reference()
        reference = {"time_s": task.demo.time_s, "qpos": qpos}
        scene, initial = tables(model, layout, task, reference, self.options["h2r"])
        return model, layout, scene, initial, changed

    def test_private_model_takes_damping_from_the_task_record(self):
        """Damping follows the task's record; any other difference is an error."""
        from spider.h2r.cpu_scene import private_model

        task = toy_task(self.root.name, kv_scale=1.01)
        model, layout, changed = private_model(task, 0.01)
        self.assertEqual(model.opt.timestep, 0.01)
        self.assertAlmostEqual(changed, 0.01 / 1.01)
        x = layout.actuators[NAMES.index("x")]
        self.assertAlmostEqual(-model.actuator_biasprm[x, 2], 20 * 1.01)
        self.assertEqual(model.actuator_gainprm[x, 0], 300)
        task.compiled["robot"]["joints"]["x"]["kp"] = 301.0
        with self.assertRaisesRegex(ValueError, "stiffness differs"):
            private_model(task, 0.01)
        task = toy_task(self.root.name)
        task.compiled["base_scene"]["joints"]["hinge"]["damping"] = 0.02
        with self.assertRaisesRegex(ValueError, "Object joint damping"):
            private_model(task, 0.01)

    def test_tables_follow_names_and_end_of_frame_rows(self):
        """Rows are frame ends; joints, actuators and bodies resolve by name."""
        task = toy_task(self.root.name)
        model, layout, scene, initial, _ = self.scene(task)
        rows = len(scene["qpos_ref"])
        self.assertEqual(rows, 10)  # ceil(0.5 / 0.05)
        times = np.minimum((np.arange(rows) + 1) * 0.05, task.duration_s)
        expected = np.column_stack(
            [
                np.interp(times, task.demo.time_s, column)
                for column in joint_reference().T
            ]
        )
        # Row i is the reference at the end of control frame i; the start is apart.
        np.testing.assert_allclose(scene["qpos_ref"][:, layout.robot_q], expected)
        np.testing.assert_array_equal(initial[layout.robot_q], joint_reference()[0])
        np.testing.assert_allclose(scene["targets0"][:, layout.robot_v], expected)
        # Actuators are listed in reverse; targets reach them by joint name.
        for name, actuator, dof in zip(
            NAMES,
            scene["position_actuator"],
            scene["position_actuator_dof"],
            strict=True,
        ):
            self.assertEqual(model.actuator_trnid[actuator, 0], model.joint(name).id)
            self.assertEqual(dof, model.joint(name).dofadr[0])
        dofs = [model.joint(name).dofadr[0] for name in NAMES]
        np.testing.assert_array_equal(scene["residual_dof"], dofs)
        np.testing.assert_array_equal(scene["action_clip"], [0.2] * 3 + [1.0] * 5)
        self.assertEqual(scene["wrist_qpos"].size, 0)
        hinge = model.joint("lid_hinge").qposadr[0]
        np.testing.assert_array_equal(scene["articulation_q"], [hinge])
        self.assertEqual(list(scene["body_names"]), ["bottom", "top"])
        box = np.tile([0.3, 0, 0.1], (rows, 1))
        np.testing.assert_allclose(scene["object_pose_ref"][:, 0, :3], box)
        np.testing.assert_allclose(
            scene["object_pose_ref"][:, 1, :3], box + [0, 0, 0.04], atol=1e-12
        )
        np.testing.assert_array_equal(
            scene["object_pose_ref"][0, :, 3:], [[0, 0, 0, 1]] * 2
        )
        # The wrist hinges are scored through the wrist body, not joint by joint.
        rotation = {model.joint(name).qposadr[0] for name in ROTATION}
        self.assertFalse(rotation & set(scene["scalar_q"].tolist()))

    def world(self, task, samples=4):
        """A few worlds of the toy task at its initial state."""
        from spider.h2r import cpu_world

        model, layout, scene, initial, _ = self.scene(task)
        config, _ = self.cpu_plan.planner_config(self.options, self.root.name, 0)
        config.num_samples = samples
        env = cpu_world.build_env(config, model, scene, initial, num_threads=2)
        return cpu_world, config, env, layout, scene, initial

    def test_scalar_wrist_command_filter_and_reward(self):
        """Residuals reach hinge targets through the filter; the cost adds the wrist."""
        import torch

        world, config, env, layout, scene, initial = self.world(
            toy_task(self.root.name)
        )
        np.testing.assert_array_equal(env.qpos[0], initial)
        self.assertTrue((env.qvel == 0).all())
        world.step_env(config, env, torch.zeros(env.nu))
        record = world.read_step(config, env)
        # A zero residual sends the reference itself, in the task's joint order.
        np.testing.assert_array_equal(
            record["joint_target"], scene["qpos_ref"][0, layout.robot_q]
        )
        # A wrist rotation residual moves the hinge targets, filtered by the EMA.
        residual = torch.zeros(env.nu)
        residual[3:6] = torch.tensor([0.2, -0.1, 0.3])
        residual[0] = 1.0  # beyond the 0.2 m clip only after the filter: 0.5
        world.step_env(config, env, residual)
        record = world.read_step(config, env)
        expected = scene["qpos_ref"][1, layout.robot_q].copy()
        expected[:6] += [0.2, 0, 0, 0.1, -0.05, 0.15]
        np.testing.assert_allclose(record["joint_target"], expected)
        self.assertEqual(env.qfrc.max(), 0.0)  # no wrist torque law
        # The reward adds the wrist body's rotation error to upstream's cost.
        from spider.simulators import cpu

        qpos_dist, qvel_dist = cpu.spider_cost(config, env)
        angle = world.wrist_rotation_errors(env)
        self.assertGreater(angle[0, 0], 1e-3)
        reward, _ = world.get_reward(config, env)
        manual = (
            -np.sqrt(qpos_dist**2 + (config.base_rot_rew_scale * angle[:, 0]) ** 2)
            - config.vel_rew_scale * qvel_dist
        )
        np.testing.assert_allclose(reward.numpy(), manual, rtol=1e-6)
        terminal, _ = world.get_terminal_reward(config, env)
        np.testing.assert_allclose(terminal.numpy(), reward.numpy())

    def test_states_restore_and_rerun_bitwise(self):
        """A restored state replays to the same bits; other plans end elsewhere."""
        import torch

        world, config, env, *_ = self.world(toy_task(self.root.name))
        torch.manual_seed(0)
        plans = 0.05 * torch.randn(3, env.num_worlds, env.nu)
        start = world.save_state(env)
        runs = []
        for _ in range(2):
            world.load_state(env, start)
            for control in plans:
                world.step_env(config, env, control)
            runs.append(env.qpos.copy())
        np.testing.assert_array_equal(runs[0], runs[1])
        self.assertFalse(np.array_equal(runs[0][0], runs[0][1]))


@unittest.skipUnless(BATCH, "needs mujoco.batch (MuJoCo >= 3.17)")
class PlanTests(unittest.TestCase):
    """End to end on a tiny budget: seal, replay check, and the two refusals."""

    def plan(self, root, task, prepared, overrides):
        """Plan the toy task with a debugging budget."""
        from spider.h2r import cpu_plan

        root = Path(root)
        (root / "debug.yaml").write_text(yaml.safe_dump(overrides))
        binding = read_json(prepared / "h2r_binding.json")
        with (
            patch("spider.h2r.cpu_plan.load_task", return_value=task),
            patch("spider.h2r.cpu_plan.load_prepared", return_value=binding),
        ):
            return cpu_plan.plan(
                prepared, task.path, root / "plan", 0, 2, root / "debug.yaml"
            )

    BUDGET = {"mpc": {"num_samples": 16, "max_num_iterations": 1}}

    def test_plan_is_sealed_and_replays_to_the_planned_states(self):
        """A complete run is sealed with its verdict, replays, and repeats."""
        from spider.h2r.replay import load_plan

        with tempfile.TemporaryDirectory() as root:
            task = toy_task(Path(root))
            prepared = write_prepared(Path(root) / "prepared", task, joint_reference())
            output = self.plan(root, task, prepared, self.BUDGET)
            manifest = read_json(output / "plan.json")
            self.assertEqual(manifest["format"], PLAN_FORMAT)
            self.assertEqual(manifest["recipe"], "cpu")
            self.assertEqual(manifest["planning_backend"], "mujoco-cpu")
            self.assertEqual(manifest["control_dt_s"], 0.05)
            self.assertEqual(manifest["controls"], 10)
            self.assertEqual(manifest["controls_file"], "controls.npz")
            self.assertEqual(manifest["joint_names"], NAMES)
            self.assertLessEqual(manifest["replay_check_max_abs_qpos"], 1e-9)
            self.assertEqual(manifest["planner"]["mujoco_version"], mujoco.__version__)
            self.assertEqual(manifest["native"]["success"], 1)
            for name in ("trajectory_cpu.npz", "metrics.json", "scene_tables.npz"):
                self.assertIn(name, manifest["files"])
            resolved = yaml.safe_load((output / "resolved_config.yaml").read_text())
            self.assertEqual(resolved["mpc"]["num_samples"], 16)
            self.assertEqual(resolved["h2r"]["substeps"], 5)
            binding, data, initial = load_plan(output, task)
            self.assertEqual(binding["sha256"], sha256(output / "plan.json"))
            self.assertEqual(data["joint_targets"].shape, (10, len(NAMES)))
            np.testing.assert_allclose(data["time_s"], np.arange(10) * 0.05)
            self.assertEqual(initial, read_json(prepared / "initial_robot.json"))
            native = arrays(output / "trajectory_cpu.npz")
            np.testing.assert_array_equal(native["joint_target"], data["joint_targets"])
            # The same seed and settings give the same plan, on any thread count.
            again = Path(root) / "again"
            again.mkdir()
            second = self.plan(again, task, prepared, self.BUDGET)
            np.testing.assert_array_equal(
                arrays(second / "controls.npz")["joint_targets"], data["joint_targets"]
            )

    def test_early_stop_leaves_no_plan(self):
        """A run upstream stops early is a failure with diagnostics, not a plan."""
        with tempfile.TemporaryDirectory() as root:
            # The reference carries the box a metre away; nothing moves the box.
            task = toy_task(Path(root), object_shift=1.0)
            prepared = write_prepared(Path(root) / "prepared", task, joint_reference())
            overrides = {
                **self.BUDGET,
                "recovery": {"stop_frames": 2, "max_backtracks": 1, "max_retries": 0},
            }
            with self.assertRaisesRegex(RuntimeError, "stopped early"):
                self.plan(root, task, prepared, overrides)
            output = Path(root) / "plan"
            self.assertFalse((output / "plan.json").exists())
            failure = read_json(output / "failure.json")
            self.assertEqual(failure["reason"], "upstream-cpu-stopped-early")
            self.assertLess(failure["executed_controls"], failure["required_controls"])
            self.assertTrue((output / "trajectory_cpu.npz").is_file())

    def test_rejected_initial_pose_is_proposed_without_planning(self):
        """A start that fails the pre-check is sealed without controls."""
        from spider.h2r.replay import load_plan

        with tempfile.TemporaryDirectory() as root:
            task = toy_task(Path(root))
            prepared = write_prepared(
                Path(root) / "prepared", task, joint_reference(), accepted=False
            )
            output = self.plan(root, task, prepared, self.BUDGET)
            manifest = read_json(output / "plan.json")
            self.assertEqual(manifest["deployment"], "initial-pose-only")
            self.assertEqual(manifest["controls"], 0)
            self.assertFalse((output / "trajectory_cpu.npz").exists())
            _, data, _ = load_plan(output, task)
            self.assertEqual(data["joint_targets"].shape, (0, len(NAMES)))

    def test_settings_cannot_select_assets_or_unknown_options(self):
        """Overrides reach algorithm settings only."""
        from spider.h2r import cpu_plan

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bad.yaml"
            for bad in (
                {"mpc": {"model_path": "scene.xml"}},
                {"mpc": {"max_sim_steps": 5}},
                {"recovery": {"unknown": 1}},
                {"h2r": {"substeps": 0}},
                {"mpc": {"num_samples": "many"}},
                {"scene": {}},
            ):
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError, msg=json.dumps(bad)):
                    cpu_plan.settings(path)
            options = cpu_plan.settings()
            self.assertEqual(options["mpc"]["num_samples"], 128)
            self.assertEqual(options["h2r"]["frame_dt"], 0.05)


if __name__ == "__main__":
    unittest.main()
