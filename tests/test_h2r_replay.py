"""Offline controls stay fixed across observations, seeds and evaluation modes."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spider.h2r.bundle import PREPARED_FILES, digest, identity, read_json, write_json
from spider.h2r.replay import Replay, load_plan
from spider.h2r.settings import PIPELINE


class ReplayTests(unittest.TestCase):
    """Protect a shared offline plan and each episode's independent cursor."""

    def setUp(self):
        """Include a task whose final interval is shorter than a control period."""
        self.task = {
            "task_id": "task",
            "bundle_id": "bundle",
            "scene_id": "scene",
            "robot_manifest_sha256": "robot",
            "dt_s": 1 / 240,
            "control_ticks": 8,
            "end_tick": 19,
            "semantics": {
                "joint_names": ["wrist", "finger"],
                "joint_units": ["m", "rad"],
            },
        }
        self.targets = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])

    def artifact(self, root, *, ticks=None, targets=None):
        """Write a sealed synthetic control artifact without GPU dependencies."""
        ticks = np.array([0, 8, 16]) if ticks is None else ticks
        np.savez(
            root / "trajectory_mjwp.npz",
            ticks=ticks,
            time_s=ticks * self.task["dt_s"],
            joint_targets=self.targets if targets is None else targets,
        )
        write_json(root / "initial_robot.json", {"wrist": 0, "finger": 0})
        (root / "resolved_config.yaml").write_text("{}\n")
        binding = {
            **{k: v for k, v in self.task.items() if k != "semantics"},
            "format": "spider-h2r-plan-v4",
            "preparation": {
                **{
                    k: self.task[k]
                    for k in (
                        "task_id",
                        "bundle_id",
                        "scene_id",
                        "robot_manifest_sha256",
                    )
                },
                "format": "spider-h2r-v4",
                "model_source": "task",
                "files": dict.fromkeys(PREPARED_FILES, "test-digest"),
            },
            "controls": len(ticks),
            **self.task["semantics"],
            "files": {
                p.name: digest(p) for p in root.iterdir() if p.name != "plan.json"
            },
        }
        binding["plan_id"] = identity(binding)
        write_json(root / "plan.json", binding)

    def test_native_plan_export_covers_common_grid(self):
        """Run the real exporter and replay reader across nonmatching grids."""
        from spider.h2r.plan import plan

        class FakeController:
            def __init__(self, *args):
                self.config = SimpleNamespace(
                    sim_dt=0.01, ctrl_dt=0.4, physics_substeps=1
                )
                self.layout = SimpleNamespace(
                    names=["wrist", "finger"],
                    targets=lambda v: dict(zip(["wrist", "finger"], v, strict=True)),
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
            prepared, task_path, output = (
                root / "prepared",
                root / "task",
                root / "plan",
            )
            prepared.mkdir()
            self.artifact(prepared)
            binding = read_json(prepared / "plan.json")["preparation"]
            binding["pipeline"] = PIPELINE
            write_json(prepared / "h2r_binding.json", binding)
            with (
                patch("spider.h2r.plan.load_task", return_value=self.task),
                patch("spider.h2r.plan.load_prepared", return_value=binding),
                patch("spider.h2r.controller.Controller", FakeController),
                patch("spider.h2r.plan.run_fast", side_effect=[True, False]),
            ):
                plan(prepared, task_path, output, "cpu", 0)
                failed = root / "failed"
                with self.assertRaisesRegex(RuntimeError, "denominator"):
                    plan(prepared, task_path, failed, "cpu", 0)
                self.assertFalse((failed / "plan.json").exists())
                self.assertTrue((failed / "trajectory_failed.npz").exists())
                self.assertEqual(
                    read_json(failed / "failure.json")["task_end_tick"], 19
                )
            manifest, data = load_plan(output, self.task)
            self.assertEqual(manifest["preparation"]["pipeline"], PIPELINE)
            self.assertEqual(manifest["native_controls"], 8)
            self.assertEqual(manifest["controls"], 3)
            np.testing.assert_array_equal(
                data["joint_targets"], [[0, 0], [3, -3], [6, -6]]
            )
            np.testing.assert_allclose(data["source_time_s"], [0, 0.03, 0.06])

    def test_twenty_evaluations_share_identical_controls(self):
        """Ignore perturbed observations in both modes and all ten seeds."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.artifact(root)
            binding, data = load_plan(root, self.task)
            self.assertNotIn("pipeline", binding["preparation"])
            for mode in ("sim", "sim-dr"):
                for seed in range(10):
                    replay = Replay(self.task, binding, data)
                    for i, tick in enumerate((0, 8, 16)):
                        targets, diagnostic = replay.control(
                            {
                                "tick": tick,
                                "time_s": tick / 240,
                                "robot_q": [seed * 100, -seed],
                                "bodies": {"mode": mode},
                            },
                            ["finger", "wrist"],
                        )
                        self.assertEqual(
                            targets,
                            dict(
                                zip(["wrist", "finger"], self.targets[i], strict=True)
                            ),
                        )
                        self.assertEqual(diagnostic["plan_id"], binding["plan_id"])
                    with self.assertRaisesRegex(ValueError, "clock"):
                        replay.control(
                            {"tick": 16, "time_s": 16 / 240}, ["wrist", "finger"]
                        )

    def test_resealed_plan_cannot_restore_legacy_model_source(self):
        """Require task-only provenance even after an attacker updates plan_id."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.artifact(root)
            original = read_json(root / "plan.json")
            for legacy in ("spider-h2r-plan-v1", "spider-h2r-plan-v2", None):
                changed = copy.deepcopy(original)
                if legacy:
                    changed["format"] = legacy
                else:
                    changed["preparation"]["planning_scene"] = "custom.xml"
                changed["plan_id"] = identity(
                    {k: v for k, v in changed.items() if k != "plan_id"}
                )
                write_json(root / "plan.json", changed)
                with self.assertRaises(ValueError):
                    load_plan(root, self.task)

    def test_incomplete_nonfinite_or_changed_plan_rejected(self):
        """Reject truncation, non-finite controls, mismatched tasks and tampering."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.artifact(root, ticks=np.array([0, 8]), targets=self.targets[:2])
            with self.assertRaisesRegex(ValueError, "complete task"):
                load_plan(root, self.task)
            invalid = self.targets.copy()
            invalid[1, 0] = np.nan
            self.artifact(root, targets=invalid)
            with self.assertRaisesRegex(ValueError, "finite targets"):
                load_plan(root, self.task)
            self.artifact(root)
            changed = copy.deepcopy(self.task)
            changed["task_id"] = "different"
            with self.assertRaisesRegex(ValueError, "task_id"):
                load_plan(root, changed)
            (root / "initial_robot.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Changed artifact"):
                load_plan(root, self.task)


if __name__ == "__main__":
    unittest.main()
