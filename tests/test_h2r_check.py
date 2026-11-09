"""Audit actual submission records without confusing DR or failure with mismatch."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider.h2r.bundle import PREPARED_FILES, digest, identity, read_json, write_json
from spider.h2r.check import check, json_lines


def write_lines(path, rows):
    """Write the evaluator's JSONL representation with real line boundaries."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class CheckTests(unittest.TestCase):
    """Exercise sealed inputs, exact control boundaries and independent episodes."""

    def setUp(self):
        """Create one completed seed, one partial failure and one rejected initial state."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.task_path, self.plan, self.run = [
            root / name for name in ("task", "plan", "run")
        ]
        for path in (self.task_path, self.plan, self.run):
            path.mkdir()
        (self.task_path / "scene.xml").write_text("<mujoco/>")
        (self.task_path / "reference.json").write_text("{}")
        np.savez(self.task_path / "reference.npz", time_s=[0, 7 / 300])
        self.task = {
            "format": "h2r-evaluation-v3",
            "scene": "scene.xml",
            "bundle_id": "bundle",
            "scene_id": "scene",
            "robot_manifest_sha256": "robot",
            "dt_s": 1 / 300,
            "control_ticks": 3,
            "end_tick": 7,
            "observation": {
                "quaternion": "wxyz",
                "body_velocity": "world_at_body_origin",
            },
            "semantics": {
                "joint_names": ["wrist", "finger"],
                "joint_units": ["m", "rad"],
            },
            "files": {p.name: digest(p) for p in self.task_path.iterdir()},
        }
        self.task["task_id"] = identity(self.task)
        write_json(self.task_path / "task.json", self.task)
        self.targets = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        np.savez(
            self.plan / "trajectory_mjwp.npz",
            ticks=[0, 3, 6],
            time_s=np.array([0, 3, 6]) / 300,
            joint_targets=self.targets,
        )
        self.initial = {"finger": 0.02, "wrist": 0.01}
        write_json(self.plan / "initial_robot.json", self.initial)
        (self.plan / "resolved_config.yaml").write_text("{}")
        identities = {
            key: self.task[key]
            for key in ("task_id", "bundle_id", "scene_id", "robot_manifest_sha256")
        }
        binding = {
            **identities,
            **{key: self.task[key] for key in ("dt_s", "control_ticks", "end_tick")},
            **self.task["semantics"],
            "format": "spider-h2r-plan-v4",
            "controls": 3,
            "preparation": {
                **identities,
                "format": "spider-h2r-v4",
                "model_source": "task",
                "files": dict.fromkeys(PREPARED_FILES, "unused-preparation-digest"),
            },
            "files": {p.name: digest(p) for p in self.plan.iterdir()},
        }
        binding["plan_id"] = identity(binding)
        write_json(self.plan / "plan.json", binding)
        recorded_binding = {
            "method": "spider",
            "controller": binding,
            "bundle_id": "bundle",
            "robot_manifest_sha256": "robot",
        }
        results = {}
        for seed, last in ((0, 7), (1, 4), (2, None)):
            name = f"seed-{seed}"
            directory = self.run / name
            directory.mkdir()
            rows = []
            for tick in range(last + 1 if last is not None else 0):
                # Tick 3 still records the target issued at tick 0; tick 4 has the next one.
                target = (
                    [0.01, 0.02]
                    if tick == 0
                    else self.targets[(tick - 1) // 3].tolist()
                )
                rows.append(
                    {
                        "tick": tick,
                        "time_s": tick / 300,
                        "submitted_target": target,
                        "effective_target": [9, 9],
                        "robot_q": [0.01, 0.02],
                        "robot_dq": [0, 0],
                    }
                )
            write_lines(directory / "physical.jsonl", rows)
            if rows:
                np.savez(
                    directory / "physical.npz",
                    **{key: [row[key] for row in rows] for key in rows[0]},
                )
            results[name] = {
                "episode_id": name,
                "last_valid_tick": last,
                "ended_tick": last or 0,
                "end_tick": 7,
                "complete": seed == 0,
                "reason": ["target_end", "physics_error", "initial_collision"][seed],
            }
            write_json(directory / "result.json", results[name])
        metadata = {
            "format": "h2r-rollout-v1",
            "mode": "sim-dr",
            "status": "finished",
            **{
                key: self.task[key]
                for key in ("task_id", "dt_s", "control_ticks", "end_tick")
            },
            "binding": recorded_binding,
            "seeds": [0, 1, 2],
            "episodes": results,
        }
        write_json(self.run / "run.json", metadata)
        self.events = [
            {
                "type": "start",
                "task_id": self.task["task_id"],
                "binding": recorded_binding,
                "initial_robot": dict.fromkeys(results, self.initial),
            }
        ]
        for index, tick in enumerate((0, 3, 6)):
            command = {
                "tick": tick,
                "joint_targets": {
                    "finger": float(self.targets[index, 1]),
                    "wrist": float(self.targets[index, 0]),
                },
            }
            self.events.append(
                {
                    "type": "step",
                    "commands": {
                        name: command
                        for name in results
                        if name == "seed-0" or name == "seed-1" and tick < 4
                    },
                }
            )
        write_lines(self.run / "events.jsonl", self.events)

    def audit(self, episode=None):
        """Call the public checker through real artifact loaders."""
        return check(self.plan, self.task_path, self.run, episode)

    def test_complete_and_failed_seeds_in_both_modes(self):
        """DR effective targets may differ; failure prefixes stay explicitly incomplete."""
        for mode in ("sim", "sim-dr"):
            metadata = read_json(self.run / "run.json")
            metadata["mode"] = mode
            write_json(self.run / "run.json", metadata)
            report = self.audit()
            self.assertEqual(report["control_hz"], 100)
            self.assertEqual(report["expected_controls"], 3)
            self.assertTrue(report["episodes"]["seed-0"]["full_time_coverage"])
            self.assertEqual(report["episodes"]["seed-0"]["physical_frames_checked"], 8)
            self.assertFalse(report["episodes"]["seed-1"]["complete"])
            self.assertEqual(report["episodes"]["seed-1"]["coverage"], 4 / 7)
            self.assertEqual(report["episodes"]["seed-2"]["controls_checked"], 0)
            self.assertEqual(
                report["episodes"]["seed-2"]["reason"], "initial_collision"
            )
        self.assertEqual(set(self.audit("seed-1")["episodes"]), {"seed-1"})
        with self.assertRaisesRegex(ValueError, "absent"):
            self.audit("seed-99")

    def test_changed_command_or_control_clock_rejected(self):
        """A named target mismatch, missing event or duplicate tick must fail."""
        original = json.dumps(self.events)
        for change in ("target", "missing", "duplicate"):
            events = json.loads(original)
            if change == "target":
                events[1]["commands"]["seed-0"]["joint_targets"]["wrist"] += 0.1
            elif change == "missing":
                del events[2]["commands"]["seed-0"]
            else:
                events[2]["commands"]["seed-0"]["tick"] = 0
            write_lines(self.run / "events.jsonl", events)
            with self.assertRaises(ValueError):
                self.audit()

    def test_binding_and_initialization_rejected(self):
        """Do not validate controls belonging to another plan or initial proposal."""
        metadata = read_json(self.run / "run.json")
        metadata["binding"]["controller"]["plan_id"] = "wrong-plan"
        write_json(self.run / "run.json", metadata)
        with self.assertRaisesRegex(ValueError, "different SPIDER plan"):
            self.audit()
        metadata["binding"] = self.events[0]["binding"]
        write_json(self.run / "run.json", metadata)
        self.events[0]["initial_robot"]["seed-0"]["wrist"] = 1
        write_lines(self.run / "events.jsonl", self.events)
        with self.assertRaisesRegex(ValueError, "Initial proposal"):
            self.audit()

    def test_physical_submission_and_npz_disagreement_rejected(self):
        """Check physical evidence as well as the socket event log."""
        path = self.run / "seed-0/physical.jsonl"
        rows = json_lines(path)
        rows[3]["submitted_target"] = self.targets[1].tolist()
        write_lines(path, rows)
        with self.assertRaisesRegex(ValueError, "Recorded submitted target"):
            self.audit()
        rows[3]["submitted_target"] = self.targets[0].tolist()
        write_lines(path, rows)
        np.savez(
            path.with_suffix(".npz"),
            tick=[0],
            time_s=[0],
            submitted_target=[[0.01, 0.02]],
        )
        with self.assertRaisesRegex(ValueError, "NPZ and JSONL"):
            self.audit()

    def test_false_completion_rejected(self):
        """A valid prefix must not become a complete task by changing result flags."""
        metadata = read_json(self.run / "run.json")
        result = metadata["episodes"]["seed-1"]
        result.update(complete=True, reason="target_end")
        write_json(self.run / "run.json", metadata)
        write_json(self.run / "seed-1/result.json", result)
        with self.assertRaisesRegex(ValueError, "without full task coverage"):
            self.audit()

    def test_physical_clock_and_termination_rejected(self):
        """Neither a timestamp gap nor a conflicting end tick is valid coverage."""
        path = self.run / "seed-0/physical.jsonl"
        rows = json_lines(path)
        rows[4]["time_s"] += 1 / 300
        write_lines(path, rows)
        with self.assertRaisesRegex(ValueError, "contiguous task-time prefix"):
            self.audit()
        rows[4]["time_s"] = 4 / 300
        write_lines(path, rows)
        metadata = read_json(self.run / "run.json")
        metadata["episodes"]["seed-0"]["ended_tick"] = 6
        write_json(self.run / "run.json", metadata)
        write_json(self.run / "seed-0/result.json", metadata["episodes"]["seed-0"])
        with self.assertRaisesRegex(ValueError, "Termination precedes"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
