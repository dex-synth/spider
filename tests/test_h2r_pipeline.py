"""Behavioral checks for upstream fast preprocessing, selection and recovery."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from test_h2r_integration import fixture

from spider.config import Config
from spider.h2r.controller import Controller
from spider.h2r.fast_loop import run_fast
from spider.h2r.reference import native_reference, sample_reference
from spider.h2r.settings import DATASET_RECIPES, settings, validate_clock
from spider.interp import interp
from spider.optimizers.sampling_fast import (
    make_optimize_fn_fast,
    make_optimize_once_fn_fast,
)
from spider.preprocess.ik_fast import postprocess


class FakeDriver:
    """Deterministic private-world driver for failure/recovery transitions."""

    def __init__(self, outcomes, retries=1, reverts=2):
        self.config = Config(
            device="cpu",
            max_num_iterations=2,
            max_local_retries=retries,
            max_revert_depth=reverts,
        )
        self.config.env_params_list = [[{}], [{}]]
        self.config.noise_scale = torch.ones(1)
        self.step = 0
        self.segments = []
        self.outcomes = iter(outcomes)
        self.calls, self.restored = [], []

    def save_checkpoint(self):
        """Capture a rollback position and provisional output length."""
        return {"step": self.step, "segments_len": len(self.segments)}

    def truncate(self, checkpoint):
        """Discard provisional segments after the native commit boundary."""
        del self.segments[checkpoint["segments_len"] if checkpoint else 0 :]

    def load_checkpoint(self, checkpoint):
        """Restore position without rewinding the random outcome stream."""
        self.restored.append(checkpoint["step"])
        self.step = checkpoint["step"]
        self.truncate(checkpoint)

    def optimize_chunk(self):
        """Expose the step, iteration budget and noise for comparison."""
        self.calls.append(
            (
                self.step,
                self.config.max_num_iterations,
                float(self.config.noise_scale[0]),
            )
        )
        return self.step, {
            "succeeded": np.array([next(self.outcomes)]),
            "opt_steps": np.array([self.config.max_num_iterations]),
            "num_attempts": np.array([1]),
        }

    def replay_chunk(self, step, info):
        """Commit one synthetic execution interval."""
        self.segments.append(step)
        self.step += 1
        return self.step


class PipelineTests(unittest.TestCase):
    """Check selected fast semantics independently of GPU throughput."""

    def test_dataset_defaults(self):
        """Keep each benchmark dataset's own recipe and its reference rate."""
        for dataset in ("hot3d", "taco"):  # No upstream fast override: base default.
            base = settings(dataset=dataset, ref_dt=1 / 30)
            self.assertEqual(base["mpc"]["max_num_iterations"], 32)
            self.assertEqual(base["mpc"]["ctrl_dt"], 0.4)
            self.assertEqual(base["mpc"]["nconmax_per_env"], 120)
            self.assertEqual(base["mpc"]["njmax_per_env"], 350)
            self.assertEqual(base["mpc"]["sanity_check_seconds"], 0)  # h2r.yaml
            self.assertEqual(base["ik"]["ref_dt"], 1 / 30)
            self.assertEqual(base, settings(ref_dt=1 / 30))
            validate_clock(base["mpc"])
        for dataset, rotation in [("arctic", 2.5), ("oakink2", 1.0)]:
            cfg = settings(dataset=dataset)["mpc"]
            self.assertEqual(cfg["max_num_iterations"], 4)
            self.assertEqual(cfg["object_rot_threshold"], rotation)
            self.assertEqual(cfg["sanity_check_seconds"], 0)
            self.assertEqual(cfg["ctrl_dt"], 1.5)
            self.assertEqual(cfg["temperature"], 1.0)
            self.assertEqual(cfg["improvement_threshold"], -1.0)
        self.assertEqual(set(DATASET_RECIPES), {"arctic", "oakink2", "hot3d", "taco"})
        for unknown in ("arcticv2", "oakink", "gigahand", "synthetic"):
            with self.assertRaisesRegex(ValueError, "No SPIDER recipe"):
                settings(dataset=unknown)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.yaml"
            path.write_text("mpc:\n  num_dr: 2\n")
            with self.assertRaisesRegex(ValueError, "nominal"):
                settings(path)
            # The upstream sanity check is off by default; --config restores it.
            path.write_text("mpc:\n  sanity_check_seconds: 3.0\n")
            self.assertEqual(
                settings(path, dataset="oakink2")["mpc"]["sanity_check_seconds"], 3
            )

    def test_filter_and_reference_values(self):
        """Preserve scalar filtering, interpolation and endpoint velocity."""
        model, _, layout = fixture()
        qpos = np.tile(model.qpos0, (8, 1))
        root, _ = layout.root_indices["a"]
        qpos[:, root] = [0, 0, 0, 3, 0, 0, 0, 0]
        qpos[:, layout.robot_q[0]] = np.arange(8) ** 2
        filtered, speed = postprocess(model, qpos, 0.02, 3)
        np.testing.assert_allclose(filtered[:, root], [1, 1, 1, 0, 0])
        expected_v = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, expected_v, 0.02, filtered[-2], filtered[-1])
        np.testing.assert_allclose(speed[-1], expected_v)
        source = {
            "filtered_time_s": np.arange(2, 7) * 0.02,
            "filtered_qpos": filtered,
            "filtered_qvel": speed,
        }
        cfg = Config(device="cpu", ref_steps=2)
        native = native_reference(layout, source, cfg)
        np.testing.assert_array_equal(
            native["qpos"],
            interp(torch.tensor(filtered[None], dtype=torch.float32), 2)[0].numpy(),
        )
        result, velocity, _ = sample_reference(model, layout, native, [-1, 0.12, 1])
        np.testing.assert_allclose(result[0], filtered[0], atol=1e-6)
        np.testing.assert_allclose(result[-1], filtered[-1], atol=1e-6)
        np.testing.assert_allclose(velocity[-1], speed[-1], atol=1e-5)
        self.assertGreater(np.linalg.norm(velocity[-1]), 0)
        np.testing.assert_allclose(native["time_s"][[0, -1]], [0.04, 0.12])

    def test_optimizer_fixed_iterations_and_attempts(self):
        """Run all rounds and only select the final sample on each attempt."""
        for succeeded, total in [(True, 4), (False, 16)]:
            cfg = Config(
                device="cpu",
                max_num_iterations=4,
                ctrl_steps=2,
                improvement_threshold=100,
                use_torch_compile=False,
            )
            cfg.env_params_list = [[{}]] * 4
            cfg.beta_traj = 0.9
            calls = []

            def once(cfg, env, controls, ref, env_params, sample_params, select_best):
                calls.append(select_best)  # noqa: B023
                return (
                    controls,
                    None,
                    {
                        "improvement": 0.0,
                        "recorded_qpos": torch.zeros(2, 1),
                        "recorded_qvel": torch.zeros(2, 1),
                        "state_snapshot": {"sentinel": 42},
                        "best_idx": 1,
                    },
                )

            optimize = make_optimize_fn_fast(
                once,
                lambda *_: (
                    not succeeded,  # noqa: B023
                    {"max_pos_error": 1.0, "max_rot_error": 0.0},
                ),
            )
            _, info = optimize(cfg, None, torch.zeros(4, 1), (torch.zeros(4, 1),))
            self.assertEqual(calls, [False, False, False, True] * (total // 4))
            self.assertEqual(info["opt_steps"].item(), total)
            self.assertEqual(info["succeeded"].item(), succeeded)
            self.assertEqual(info["state_snapshot"], {"sentinel": 42})

    def test_final_control_is_actual_best_sample(self):
        """Distinguish the final sampled control from a soft weighted mean."""
        cfg = Config(device="cpu", num_samples=100, use_torch_compile=False)
        samples = torch.arange(100.0)[:, None, None]

        def rollout(*args, record_states):
            return (
                samples,
                -torch.abs(torch.arange(100.0) - 1),
                torch.zeros(100),
                {
                    "trace": torch.empty(100, 1, 0, 3),
                    "recorded_qpos": samples + 100,
                    "recorded_qvel": samples + 200,
                    "state_snapshot": {"state": samples + 300},
                },
            )

        optimize = make_optimize_once_fn_fast(rollout)
        with patch(
            "spider.optimizers.sampling_fast.sample_ctrls", return_value=samples
        ):
            mean, _, _ = optimize(cfg, None, samples[0], (), select_best=False)
            best, _, info = optimize(cfg, None, samples[0], (), select_best=True)
        torch.testing.assert_close(best, samples[1])
        self.assertFalse(torch.equal(mean, best))
        self.assertEqual(info["best_idx"], 1)
        torch.testing.assert_close(info["recorded_qpos"], samples[1] + 100)

    def test_local_retry_boost_persists_until_recovery(self):
        """Do not reset a local boost on an ordinary successful segment."""
        driver = FakeDriver([False, True, True])
        self.assertTrue(run_fast(driver, 2, lambda _: None))
        self.assertEqual([c[:2] for c in driver.calls], [(0, 2), (0, 4), (1, 4)])
        self.assertEqual(driver.restored, [0])
        np.testing.assert_allclose(driver.calls[1][2], np.sqrt(2))
        self.assertEqual(driver.segments, [0, 1])

    def test_recovery_reuses_anchor_then_resets_boost(self):
        """Return to the same checkpoint and reset only beyond the anchor."""
        driver = FakeDriver([True, False, False, False, False, True, True, True])
        self.assertTrue(run_fast(driver, 3, lambda _: None))
        self.assertEqual(
            [c[:2] for c in driver.calls],
            [(0, 2), (1, 2), (1, 4), (0, 2), (0, 4), (0, 2), (1, 2), (2, 2)],
        )
        self.assertEqual(driver.restored, [1, 0, 0, 0])
        self.assertEqual(driver.segments, [0, 1, 2])

    def test_failure_truncates_then_replays_last_candidate(self):
        """Keep the original failure output, including its possible time gap."""
        driver = FakeDriver([True, True] + [False] * 6)
        self.assertFalse(run_fast(driver, 4, lambda _: None))
        self.assertEqual(driver.segments, [1])  # Native truncation can leave a gap.
        self.assertEqual(driver.config.max_num_iterations, 2)
        self.assertEqual(driver.restored, [2, 1, 1, 1, 1])

    def test_no_history_failure(self):
        """Replay the final candidate once before declaring failure."""
        driver = FakeDriver([False, False])
        self.assertFalse(run_fast(driver, 2, lambda _: None))
        self.assertEqual(driver.segments, [0])

    def test_snapshot_replay_and_reference_shift(self):
        """Advance with recorded states and shift exactly one control interval."""
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody><actuator><position joint="0"/></actuator></mujoco>'.replace(
                "<joint/>", '<joint name="0"/>'
            )
        )
        driver = Controller.__new__(Controller)
        driver.config = Config(device="cpu", horizon_steps=4, ctrl_steps=2)
        driver.env = SimpleNamespace(model_cpu=model)
        driver.data = mujoco.MjData(model)
        driver.controls = torch.arange(4.0)[:, None]
        driver.reference = lambda step: (torch.arange(step, step + 5.0)[:, None],) * 5
        driver.segments = []
        info = {
            "best_qpos": np.array([[0.1], [0.2]]),
            "best_qvel": np.array([[1.0], [2.0]]),
            "state_snapshot": object(),
            "best_idx": 7,
        }
        with patch("spider.h2r.controller._restore_best_state_to_env") as restore:
            step = driver.replay_chunk(0, info)
        self.assertEqual(step, 2)
        np.testing.assert_allclose(driver.data.qpos, [0.2])
        np.testing.assert_allclose(driver.data.qvel, [2.0])
        torch.testing.assert_close(
            driver.controls[:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0])
        )
        restore.assert_called_once_with(driver.env, info["state_snapshot"], 7)
        np.testing.assert_array_equal(driver.segments[0]["ctrl"].ravel(), [0, 1])


if __name__ == "__main__":
    unittest.main()
