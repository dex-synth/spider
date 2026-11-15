"""Offline controls stay fixed across observations, seeds and evaluation modes."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from test_h2r_integration import memory_task

from spider.h2r.artifacts import PLAN_FORMAT, PREPARED_FILES, PREPARED_FORMAT
from spider.h2r.h2r_io import client
from spider.h2r.h2r_io.files import content_hash, read_json, sha256, write_json
from spider.h2r.replay import Replay, control_count, load_plan
from spider.h2r.settings import PIPELINE

NAMES = ["x", "finger"]
IDENTITIES = {"task_id": "task", "demo_id": "demo", "robot_id": "robot"}
PREPARATION = {
    **IDENTITIES,
    "format": PREPARED_FORMAT,
    "pipeline": PIPELINE,
    "files": dict.fromkeys(PREPARED_FILES, "test-digest"),
}


def write_plan(root, task, targets, control_dt_s=0.01):
    """Write a sealed synthetic plan without GPU dependencies."""
    np.savez(
        root / "trajectory_mjwp.npz",
        time_s=np.arange(len(targets)) * control_dt_s,
        joint_targets=targets,
    )
    write_json(root / "initial_robot.json", {"x": 0.01, "finger": 0.02})
    write_json(root / "initialization.json", {"accepted": True})
    (root / "resolved_config.yaml").write_text("{}\n")
    semantics = task.robot.semantics
    binding = {
        **IDENTITIES,
        "format": PLAN_FORMAT,
        "deployment": "offline-control-replay",
        "preparation": PREPARATION,
        "joint_names": semantics["joint_names"],
        "joint_units": semantics["joint_units"],
        "control_dt_s": control_dt_s,
        "controls": len(targets),
        "files": {p.name: sha256(p) for p in root.iterdir() if p.name != "plan.json"},
    }
    binding["plan_id"] = content_hash(binding)
    write_json(root / "plan.json", binding)
    return binding


class ReplayTests(unittest.TestCase):
    """Protect a shared offline plan and each episode's independent cursor."""

    def setUp(self):
        """A 2/30 s task: seven .01 s controls, the last one shorter."""
        self.task = memory_task()
        self.targets = np.arange(14.0).reshape(7, 2) / 10

    def test_control_count_covers_task_despite_float_rounding(self):
        """A whole number of periods is not rounded up to one more control."""
        self.assertGreater(0.07 / 0.01, 7)  # 7.000000000000001
        self.assertEqual(control_count(0.07, 0.01), 7)
        self.assertEqual(control_count(29.6, 0.01), 2960)
        self.assertEqual(control_count(2 / 30, 0.01), 7)
        self.assertEqual(control_count(0.06, 0.01), 6)
        self.assertEqual(control_count(1.0, 1 / 30), 30)
        for invalid in (0, -0.01, float("nan"), float("inf"), "0.01", None):
            with self.assertRaises(ValueError):
                control_count(1.0, invalid)

    def test_native_plan_export_on_its_own_clock(self):
        """Run the real exporter and plan reader; a failed plan is never sealed."""
        from spider.h2r.plan import plan

        class FakeController:
            def __init__(self, *args):
                self.config = SimpleNamespace(
                    sim_dt=0.01, ctrl_dt=0.4, physics_substeps=1
                )
                self.layout = SimpleNamespace(
                    names=NAMES, targets=lambda v: dict(zip(NAMES, v, strict=True))
                )
                self.segments = [
                    {
                        "step": 0,
                        "ctrl": np.column_stack((np.arange(40), -np.arange(40))),
                        "qpos": np.zeros((40, 2)),
                        "qvel": np.zeros((40, 2)),
                    }
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, output = root / "prepared", root / "plan"
            prepared.mkdir()
            write_json(prepared / "h2r_binding.json", PREPARATION)
            write_json(prepared / "initial_robot.json", {"x": 0.0, "finger": 0.0})
            write_json(prepared / "initialization.json", {"accepted": True})
            (prepared / "resolved_config.yaml").write_text("{}\n")
            with (
                patch("spider.h2r.plan.load_task", return_value=self.task),
                patch("spider.h2r.plan.load_prepared", return_value=PREPARATION),
                patch("spider.h2r.controller.Controller", FakeController),
                patch("spider.h2r.plan.run_fast", side_effect=[True, False]),
            ):
                plan(prepared, root / "task", output, "cpu", 0)
                failed = root / "failed"
                with self.assertRaisesRegex(RuntimeError, "denominator"):
                    plan(prepared, root / "task", failed, "cpu", 0)
                self.assertFalse((failed / "plan.json").exists())
                self.assertTrue((failed / "trajectory_failed.npz").exists())
                self.assertEqual(
                    read_json(failed / "failure.json")["required_native_steps"], 7
                )
            manifest, data, initial = load_plan(output, self.task)
            self.assertEqual(manifest["format"], "spider-h2r-plan-v5")
            self.assertEqual(manifest["preparation"]["pipeline"], PIPELINE)
            self.assertEqual(
                (manifest["control_dt_s"], manifest["controls"]), (0.01, 7)
            )
            for removed in ("ticks", "control_ticks", "dt_s", "end_tick"):
                self.assertNotIn(removed, manifest)
            self.assertNotIn("ticks", data)
            np.testing.assert_array_equal(data["time_s"], np.arange(7) * 0.01)
            np.testing.assert_array_equal(
                data["joint_targets"], np.column_stack((np.arange(7), -np.arange(7)))
            )
            self.assertEqual(initial, {"x": 0.0, "finger": 0.0})

    def test_rejected_initial_pose_is_proposed_without_planning(self):
        """A penetrating IK start is the method's failure: no MPC, and no controls."""
        from spider.h2r.plan import plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, output = root / "prepared", root / "plan"
            prepared.mkdir()
            write_json(prepared / "h2r_binding.json", PREPARATION)
            write_json(prepared / "initial_robot.json", {"x": 0.0, "finger": 0.0})
            write_json(prepared / "initialization.json", {"accepted": False})
            (prepared / "resolved_config.yaml").write_text("{}\n")
            with (
                patch("spider.h2r.plan.load_task", return_value=self.task),
                patch("spider.h2r.plan.load_prepared", return_value=PREPARATION),
                patch("spider.h2r.controller.Controller", side_effect=AssertionError),
            ):
                plan(prepared, root / "task", output, "cpu", 0)
            manifest, data, initial = load_plan(output, self.task)
            self.assertEqual(manifest["deployment"], "initial-pose-only")
            self.assertEqual((manifest["controls"], len(data["joint_targets"])), (0, 0))
            self.assertEqual(initial, {"x": 0.0, "finger": 0.0})
            self.assertFalse(read_json(output / "initialization.json")["accepted"])
            replay = Replay(manifest, data, 3)
            with self.assertRaisesRegex(ValueError, "did not plan this task"):
                replay.control({"tick": 0}, NAMES)

    def test_evaluations_share_identical_controls_at_any_physics_step(self):
        """Ignore observations; derive ticks from the evaluation's own step."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_plan(root, self.task, self.targets)
            binding, data, _ = load_plan(root, self.task)
            for hz, ticks in ((300, 3), (600, 6), (100, 1)):
                hello = {"dt_s": 1 / hz}
                self.assertEqual(client.control_ticks(hello, 0.01), ticks)
                for seed in range(3):
                    replay = Replay(binding, data, ticks)
                    for i in range(7):
                        targets, diagnostic = replay.control(
                            {"tick": i * ticks, "robot_q": [seed * 100, -seed]},
                            ["finger", "x"],
                        )
                        self.assertEqual(
                            targets, dict(zip(NAMES, self.targets[i], strict=True))
                        )
                        self.assertEqual(diagnostic["plan_id"], binding["plan_id"])
                    for tick in (6 * ticks, 7 * ticks):  # Repeated; beyond the plan.
                        with self.assertRaisesRegex(ValueError, "clock"):
                            replay.control({"tick": tick}, NAMES)
            with self.assertRaisesRegex(ValueError, "whole number"):
                client.control_ticks({"dt_s": 1 / 240}, 0.01)

    def test_older_plan_formats_rejected_with_clear_message(self):
        """A resealed older plan or preparation cannot be replayed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = write_plan(root, self.task, self.targets)
            for legacy in ("spider-h2r-plan-v4", "spider-h2r-plan-v1", None):
                changed = copy.deepcopy(original)
                if legacy:
                    changed["format"] = legacy
                else:
                    changed["preparation"]["format"] = "spider-h2r-v4"
                changed["plan_id"] = content_hash(
                    {k: v for k, v in changed.items() if k != "plan_id"}
                )
                write_json(root / "plan.json", changed)
                with self.assertRaisesRegex(ValueError, "regenerate it from --task"):
                    load_plan(root, self.task)

    def test_incomplete_nonfinite_or_changed_plan_rejected(self):
        """Reject truncation, non-finite controls, mismatched tasks and tampering."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_plan(root, self.task, self.targets[:6])
            with self.assertRaisesRegex(ValueError, "complete task"):
                load_plan(root, self.task)
            invalid = self.targets.copy()
            invalid[1, 0] = np.nan
            write_plan(root, self.task, invalid)
            with self.assertRaisesRegex(ValueError, "finite targets"):
                load_plan(root, self.task)
            original = write_plan(root, self.task, self.targets)
            longer = memory_task(frames=4)
            with self.assertRaisesRegex(ValueError, "complete task"):
                load_plan(root, longer)
            for key in IDENTITIES:
                other = copy.deepcopy(original)
                other[key] = "different"
                other["plan_id"] = content_hash(
                    {k: v for k, v in other.items() if k != "plan_id"}
                )
                write_json(root / "plan.json", other)
                with self.assertRaisesRegex(ValueError, f"another task: {key}"):
                    load_plan(root, self.task)
            write_plan(root, self.task, self.targets)
            (root / "initial_robot.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed file"):
                load_plan(root, self.task)


if __name__ == "__main__":
    unittest.main()
