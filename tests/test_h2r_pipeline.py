"""Behavioral checks for paper preprocessing and ordinary chunked MPC."""

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from test_h2r_task import CALIBRATION, task_fixture

from spider.config import Config, compute_steps
from spider.h2r.controller import Controller
from spider.h2r.model import Layout
from spider.h2r.reference import hold_indices, sample_reference, velocities
from spider.h2r.scene import planning_spec
from spider.h2r.settings import settings
from spider.preprocess.ik import make_ik_model, postprocess


class PipelineTests(unittest.TestCase):
    """Protect numerical preprocessing and execution timing."""

    def test_private_ik_constraints_and_physics_isolation(self):
        """Keep the original constraint recipe private to IK."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = task_fixture(root)
            spec, wrists, tips = planning_spec(root, task, CALIBRATION)
            physical = spec.compile()
            ik, ids = make_ik_model(
                spec,
                [(wrists[0], "wrist"), *zip(tips, range(5), strict=True)],
                settings()["ik"],
            )
            self.assertEqual(len(set(ids)), 6)
            np.testing.assert_array_equal(ik.eq_type, [1, 0, 0, 0, 0, 0])
            np.testing.assert_allclose(
                ik.eq_solimp[:, 2], [10, 0.01, 0.01, 0.01, 0.03, 0.03]
            )
            self.assertTrue(ik.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            self.assertTrue(ik.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
            self.assertEqual(spec.compile().neq, 0)
            for field in (
                "geom_contype",
                "geom_conaffinity",
                "actuator_gainprm",
                "body_mass",
            ):
                np.testing.assert_array_equal(
                    getattr(spec.compile(), field), getattr(physical, field)
                )

    def test_filter_interior_and_timestamp_coverage(self):
        """Retain native interior values without discarding common endpoints."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = task_fixture(root)
            spec, _, _ = planning_spec(root, task, CALIBRATION)
            model = spec.compile()
            layout = Layout(model, task)
            qpos = np.tile(model.qpos0, (8, 1))
            q, v = layout.robot_q[0], layout.robot_v[0]
            qpos[:, q] = [0, 0, 0, 3, 0, 0, 0, 0]
            filtered, speed = postprocess(model, qpos, 0.02, 3)
            np.testing.assert_allclose(filtered[:, q], [1, 1, 1, 0, 0])
            np.testing.assert_allclose(speed[:, v], [50, 0, 0, -50, 0])
            times = np.arange(8) * 0.02
            source = {
                "time_s": times,
                "qpos": qpos,
                "qvel": velocities(model, times, qpos),
                "filtered_time_s": np.arange(2, 7) * 0.02,
                "filtered_qpos": filtered,
                "filtered_qvel": speed,
            }
            query = np.array([0, 0.02, 0.04, 0.05, 0.08, 0.1, 0.14, 0.2])
            result, vel, _ = sample_reference(model, layout, source, query)
            np.testing.assert_allclose(result[:, q], [1, 1, 1, 1, 1, 0, 0, 0])
            np.testing.assert_allclose(vel[:, v], [0, 0, 50, 25, 0, -50, 0, 0])
            for index, _ in layout.root_indices.values():
                np.testing.assert_allclose(
                    result[:, index : index + 7],
                    np.tile(model.qpos0[index : index + 7], (len(query), 1)),
                )

    def test_native_chunk_execution_lookahead_and_shift(self):
        """Execute a full chunk and append the correct future reference."""
        # Use the production loop with a deterministic optimizer. Assert the
        # full multi-action trace and future references, not merely call count.
        ctrl = Controller.__new__(Controller)
        ctrl.config = Config(device="cpu", **settings()["mpc"])
        compute_steps(ctrl.config)
        ctrl.controls = ctrl.chunk_start = None
        ctrl.setup_s = 0
        ctrl.device_index = 0
        ctrl.cpu_rng = torch.get_rng_state()
        ctrl.cuda_rng = torch.get_rng_state()
        ctrl.env, ctrl.data = object(), object()
        ctrl.layout = SimpleNamespace(targets=lambda x: {"joint": float(x[0])})
        ctrl.synchronize = lambda: None
        ctrl.reference = lambda t: (torch.arange(161)[:, None] + round(t / 0.01),) * 5
        calls, warm = [], []

        def optimize(cfg, env, controls, reference):
            calls.append(reference[2].clone())
            warm.append(controls.clone())
            return controls + 1000, {"opt_steps": torch.tensor([2])}

        ctrl.optimize = optimize
        with (
            patch("torch.random.fork_rng", return_value=contextlib.nullcontext()),
            patch("torch.cuda.set_rng_state"),
            patch("torch.cuda.get_rng_state", return_value=ctrl.cuda_rng),
            patch("spider.h2r.controller.mjwp.sync_env") as sync,
        ):
            outputs = [ctrl.action(i) for i in range(85)]
        self.assertEqual(
            [i for i, (_, d) in enumerate(outputs) if d["replanned"]], [0, 40, 80]
        )
        self.assertEqual(sync.call_count, 2)
        self.assertEqual([v[0]["joint"] for v in outputs[:40]], list(range(1000, 1040)))
        self.assertEqual(outputs[40][0]["joint"], 2040)
        self.assertEqual(outputs[80][0]["joint"], 3080)
        torch.testing.assert_close(calls[1][:, 0], torch.arange(41, 201))
        torch.testing.assert_close(warm[1][-40:, 0], torch.arange(160, 200))
        self.assertEqual(sum(bool(d["opt_steps"]) for _, d in outputs), 3)

    def test_online_observation_used_only_when_replanning(self):
        """Cached actions retain their indices; each replan sees fresh state."""
        ctrl = Controller.__new__(Controller)
        ctrl.config = Config(device="cpu", **settings()["mpc"])
        compute_steps(ctrl.config)
        ctrl.task = {"dt_s": 1 / 240, "control_ticks": 8, "end_tick": 105}
        ctrl.tick, ctrl.setup_s, ctrl.device_index = 0, 0, 0
        ctrl.controls = ctrl.chunk_start = None
        ctrl.cpu_rng = ctrl.cuda_rng = torch.get_rng_state()
        ctrl.env = object()
        ctrl.data = SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1))
        ctrl.synchronize = lambda: None
        ctrl.layout = SimpleNamespace(
            targets=lambda x: {"joint": float(x[0])},
            write_observation=lambda data, frame, names: data.qpos.fill(frame["tick"]),
        )
        ctrl.reference = lambda t: (torch.arange(161)[:, None].float(),) * 5
        seen = []

        def optimize(cfg, env, controls, reference):
            seen.append(float(ctrl.data.qpos[0]))
            return controls + ctrl.data.qpos[0], {"opt_steps": [1]}

        ctrl.optimize = optimize
        with (
            patch("torch.random.fork_rng", return_value=contextlib.nullcontext()),
            patch("torch.cuda.set_rng_state"),
            patch("torch.cuda.get_rng_state", return_value=ctrl.cuda_rng),
        ):
            output = [
                ctrl.control({"tick": t, "time_s": t / 240}, ["joint"])
                for t in range(0, 105, 8)
            ]
        self.assertEqual(seen, [0, 96])
        self.assertEqual([x[0]["joint"] for x in output[:4]], [0, 3, 6, 10])
        self.assertEqual(output[12][0]["joint"], 136)
        self.assertEqual(output[13][1]["planning_qpos"], [104])
        with self.assertRaisesRegex(ValueError, "clock"):
            ctrl.control({"tick": 104, "time_s": 104 / 240}, ["joint"])

    def test_noncommensurate_publication_and_short_final_interval(self):
        """Publish past native actions at every common tick, including the tail."""
        native = np.arange(8) * 0.01
        ticks = np.arange(0, 19, 8)
        indices = hold_indices(native, ticks / 240)
        np.testing.assert_array_equal(indices, [0, 3, 6])
        np.testing.assert_array_equal(
            hold_indices(native, [0.03, 0.06 + 1e-16]), [3, 6]
        )
        self.assertTrue(np.all(native[indices] <= ticks / 240 + 1e-12))


if __name__ == "__main__":
    unittest.main()
