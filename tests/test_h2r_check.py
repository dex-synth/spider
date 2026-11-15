"""Audit actual submission records without confusing DR or failure with mismatch."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_h2r_integration import memory_task
from test_h2r_replay import write_plan

from spider.h2r.check import check, json_lines
from spider.h2r.h2r_io.files import read_json, write_json


def write_lines(path, rows):
    """Write the evaluator's JSONL representation with real line boundaries."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class CheckTests(unittest.TestCase):
    """Exercise sealed plans, the run's own clock and independent episodes."""

    def setUp(self):
        """A 2/30 s task at 300 Hz: 20 ticks, seven .01 s controls, records every 10.

        One completed seed, one partial failure and one rejected initial state.
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.plan, self.run = root / "plan", root / "run"
        self.plan.mkdir()
        self.run.mkdir()
        self.task = memory_task()
        self.targets = np.arange(14.0).reshape(7, 2) / 10
        binding = write_plan(self.plan, self.task, self.targets)
        self.initial = read_json(self.plan / "initial_robot.json")
        recorded_binding = {
            "demo_id": "demo",
            "robot_id": "robot",
            "method": "spider",
            "controller": binding,
        }
        results = {}
        for seed, last_valid, ended in ((0, 20, 20), (1, 13, 14), (2, None, 0)):
            name = f"seed-{seed}"
            directory = self.run / name
            directory.mkdir()
            rows = [
                {"tick": tick, "physics_time_s": tick / 300, "robot_q": [0.01, 0.02]}
                for tick in range(
                    0, (last_valid if last_valid is not None else -1) + 1, 10
                )
            ]
            write_lines(directory / "physical.jsonl", rows)
            if rows:
                np.savez(
                    directory / "physical.npz",
                    **{key: [row[key] for row in rows] for key in rows[0]},
                )
            results[name] = {
                "episode_id": name,
                "last_valid_tick": last_valid,
                "last_recorded_tick": rows[-1]["tick"] if rows else None,
                "ended_tick": ended,
                "end_tick": 20,
                "complete": seed == 0,
                "reason": ["target_end", "physics_error", "initial_collision"][seed],
            }
            write_json(directory / "result.json", results[name])
        write_json(
            self.run / "run.json",
            {
                "format": "h2r-rollout-v2",
                "mode": "sim-dr",
                "status": "finished",
                "task_id": "task",
                "dt_s": 1 / 300,
                "control_ticks": 3,
                "end_tick": 20,
                "record_hz": 30,
                "record_stride_ticks": 10,
                "binding": recorded_binding,
                "seeds": [0, 1, 2],
                "episodes": results,
            },
        )
        self.events = [
            {
                "type": "start",
                "task_id": "task",
                "binding": recorded_binding,
                "initial_robot": dict.fromkeys(results, self.initial),
            }
        ]
        for index in range(7):
            tick = 3 * index
            command = {
                "tick": tick,
                "joint_targets": {
                    "finger": float(self.targets[index, 1]),
                    "x": float(self.targets[index, 0]),
                },
            }
            self.events.append(
                {
                    "type": "step",
                    "commands": {
                        name: command
                        for name in results
                        if name == "seed-0" or name == "seed-1" and tick < 14
                    },
                }
            )
        write_lines(self.run / "events.jsonl", self.events)

    def audit(self, episode=None):
        """Call the public checker through the real plan loader."""
        with patch("spider.h2r.check.load_task", return_value=self.task):
            return check(self.plan, "task", self.run, episode)

    def change_run(self, **changes):
        """Rewrite fields of the run manifest."""
        write_json(
            self.run / "run.json", {**read_json(self.run / "run.json"), **changes}
        )

    def test_complete_and_failed_seeds_in_both_modes(self):
        """Failure prefixes stay explicitly incomplete in either mode."""
        for mode in ("sim", "sim-dr"):
            self.change_run(mode=mode)
            report = self.audit()
            self.assertAlmostEqual(report["control_hz"], 100)
            self.assertEqual(report["control_ticks"], 3)
            self.assertEqual(report["expected_controls"], 7)
            self.assertTrue(report["episodes"]["seed-0"]["full_time_coverage"])
            self.assertEqual(report["episodes"]["seed-0"]["controls_checked"], 7)
            self.assertEqual(report["episodes"]["seed-0"]["physical_frames_checked"], 3)
            self.assertFalse(report["episodes"]["seed-1"]["complete"])
            self.assertEqual(report["episodes"]["seed-1"]["controls_checked"], 5)
            self.assertEqual(report["episodes"]["seed-1"]["coverage"], 13 / 20)
            self.assertEqual(report["episodes"]["seed-2"]["controls_checked"], 0)
            self.assertEqual(
                report["episodes"]["seed-2"]["reason"], "initial_collision"
            )
        self.assertEqual(set(self.audit("seed-1")["episodes"]), {"seed-1"})
        with self.assertRaisesRegex(ValueError, "absent"):
            self.audit("seed-99")

    def test_clock_comes_from_the_run_not_the_task(self):
        """Accept an overridden physics step; reject a foreign period or endpoint."""
        events = json.loads(json.dumps(self.events))
        for event in events[1:]:
            for command in event["commands"].values():
                command["tick"] *= 2
        write_lines(self.run / "events.jsonl", events)
        metadata = read_json(self.run / "run.json")
        for name, result in metadata["episodes"].items():
            for key in (
                "last_valid_tick",
                "last_recorded_tick",
                "ended_tick",
                "end_tick",
            ):
                result[key] = None if result[key] is None else 2 * result[key]
            write_json(self.run / name / "result.json", result)
            rows = json_lines(self.run / name / "physical.jsonl")
            for row in rows:
                row["tick"] *= 2
            write_lines(self.run / name / "physical.jsonl", rows)
            (self.run / name / "physical.npz").unlink(missing_ok=True)
        metadata.update(
            dt_s=1 / 600, control_ticks=6, end_tick=40, record_stride_ticks=20
        )
        write_json(self.run / "run.json", metadata)
        report = self.audit()
        self.assertEqual(report["control_ticks"], 6)
        self.assertTrue(report["episodes"]["seed-0"]["full_time_coverage"])
        self.change_run(control_ticks=3)
        with self.assertRaisesRegex(ValueError, "control period"):
            self.audit()
        self.change_run(control_ticks=6, end_tick=41)
        with self.assertRaisesRegex(ValueError, "task duration"):
            self.audit()
        self.change_run(end_tick=40, format="h2r-rollout-v1")
        with self.assertRaisesRegex(ValueError, "h2r-rollout-v2"):
            self.audit()

    def test_changed_command_or_control_clock_rejected(self):
        """A named target mismatch, missing event or duplicate tick must fail."""
        original = json.dumps(self.events)
        for change in ("target", "missing", "duplicate"):
            events = json.loads(original)
            if change == "target":
                events[1]["commands"]["seed-0"]["joint_targets"]["x"] += 0.1
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
        self.change_run(binding=self.events[0]["binding"])
        self.events[0]["initial_robot"]["seed-0"]["x"] = 1
        write_lines(self.run / "events.jsonl", self.events)
        with self.assertRaisesRegex(ValueError, "Initial proposal"):
            self.audit()

    def test_physical_initial_state_and_npz_disagreement_rejected(self):
        """Check physical evidence as well as the socket event log."""
        path = self.run / "seed-0/physical.jsonl"
        rows = json_lines(path)
        rows[0]["robot_q"] = [0.5, 0.02]
        write_lines(path, rows)
        with self.assertRaisesRegex(ValueError, "initial robot state"):
            self.audit()
        rows[0]["robot_q"] = [0.01, 0.02]
        write_lines(path, rows)
        np.savez(path.with_suffix(".npz"), tick=[0], robot_q=[[0.01, 0.02]])
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

    def test_recording_grid_and_termination_rejected(self):
        """Neither a grid gap nor a conflicting end tick is valid coverage."""
        path = self.run / "seed-0/physical.jsonl"
        rows = json_lines(path)
        rows[1]["tick"] = 11
        write_lines(path, rows)
        with self.assertRaisesRegex(ValueError, "recording grid"):
            self.audit()
        rows[1]["tick"] = 10
        write_lines(path, rows)
        metadata = read_json(self.run / "run.json")
        metadata["episodes"]["seed-0"]["ended_tick"] = 19
        write_json(self.run / "run.json", metadata)
        write_json(self.run / "seed-0/result.json", metadata["episodes"]["seed-0"])
        with self.assertRaisesRegex(ValueError, "Termination precedes"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
