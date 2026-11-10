"""Geometric regressions for quaternion means and all reference sampling paths."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from test_h2r_integration import fixture

from spider.h2r.kinematics import solve_kinematics
from spider.h2r.reference import native_reference, qpos_sample, sample_reference
from spider.interp import interp
from spider.preprocess.ik_fast import postprocess


class QuaternionTests(unittest.TestCase):
    """Verify rotation geometry without invoking IK or GPU planning."""

    def setUp(self):
        """Use multiple interleaved roots and scalar robot/object joints."""
        self.model, _, self.layout = fixture()
        self.slices = [
            slice(q + 3, q + 7) for q, _ in self.layout.root_indices.values()
        ]
        self.scalar = np.ones(self.model.nq, dtype=bool)
        for sl in self.slices:
            self.scalar[sl] = False

    def assert_rotations(self, values):
        """All output roots must be unit and sign-continuous."""
        for sl in self.slices:
            q = values[:, sl]
            np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-7)
            self.assertTrue(np.all(np.sum(q[1:] * q[:-1], axis=1) >= -1e-7))

    def test_mean_sign_invariance_and_scalar_values(self):
        """Uniform-axis motion retains the window center across pi and sign flips."""
        qpos = np.tile(self.model.qpos0, (8, 1))
        angles = np.deg2rad(np.arange(150, 230, 10))
        for axis, sl in zip(("z", "x"), self.slices, strict=True):
            qpos[:, sl] = Rotation.from_euler(axis, angles[:, None]).as_quat(
                scalar_first=True
            )
        qpos[:, self.layout.robot_q[0]] = np.arange(8) ** 2
        signed = qpos.copy()
        signed[1::2, self.slices[0]] *= -1
        signed[::2, self.slices[1]] *= -1
        before = signed.copy()
        filtered, speed = postprocess(self.model, signed, 0.02, 3)
        clean, clean_speed = postprocess(self.model, qpos, 0.02, 3)
        np.testing.assert_array_equal(signed, before)
        np.testing.assert_allclose(filtered, clean, atol=1e-14)
        np.testing.assert_allclose(speed, clean_speed, atol=1e-12)
        self.assert_rotations(filtered)
        for axis, sl in zip(("z", "x"), self.slices, strict=True):
            expected = Rotation.from_euler(axis, angles[2:-1, None])
            actual = Rotation.from_quat(filtered[:, sl], scalar_first=True)
            np.testing.assert_allclose(
                (actual.inv() * expected).magnitude(), 0, atol=1e-14
            )
        scalar_mean = np.column_stack(
            [
                np.convolve(c, np.ones(3) / 3, "valid")[1:]
                for c in qpos[:, self.scalar].T
            ]
        )
        np.testing.assert_array_equal(filtered[:, self.scalar], scalar_mean)

    def test_nlerp_geometry_and_native_scalar_equivalence(self):
        """NLERP follows the short arc without imposing SLERP's angular speed."""
        qpos = np.tile(self.model.qpos0, (2, 1))
        qpos[:, self.layout.robot_q[0]] = [0, 1]
        qpos[:, self.slices[0]] = Rotation.from_euler(
            "z", [[0], [120]], degrees=True
        ).as_quat(scalar_first=True)
        qpos[1, self.slices[0]] *= -1
        qpos[0, self.slices[1]] *= -1
        query = [-1, 0, 0.25, 1, 2]
        sampled = qpos_sample(self.layout, [0, 1], qpos, query)
        self.assert_rotations(sampled)
        actual = Rotation.from_quat(sampled[2, self.slices[0]], scalar_first=True)
        np.testing.assert_allclose(actual.as_rotvec(), [0, 0, 0.48512774819097054])
        np.testing.assert_allclose(
            sampled[:, self.layout.robot_q[0]], [0, 0, 0.25, 1, 1]
        )
        speed = np.arange(2 * self.model.nv).reshape(2, -1).astype(float)
        values, velocity, controls = sample_reference(
            self.model,
            self.layout,
            {"time_s": [0, 1], "qpos": qpos, "qvel": speed},
            query,
        )
        np.testing.assert_allclose(values, sampled)
        np.testing.assert_allclose(velocity[2], 0.75 * speed[0] + 0.25 * speed[1])
        np.testing.assert_array_equal(velocity[-1], speed[-1])
        np.testing.assert_array_equal(controls, self.layout.controls(values))
        source = {
            "filtered_time_s": np.array([0.0, 1.0]),
            "filtered_qpos": qpos,
            "filtered_qvel": speed,
        }
        native = native_reference(
            self.layout, source, SimpleNamespace(ref_dt=1.0, sim_dt=0.5, ref_steps=2)
        )
        self.assert_rotations(native["qpos"])
        expected = interp(torch.tensor(qpos[None], dtype=torch.float32), 2)[0].numpy()
        np.testing.assert_array_equal(
            native["qpos"][:, self.scalar], expected[:, self.scalar]
        )
        np.testing.assert_array_equal(native["ctrl"], self.layout.controls(expected))
        np.testing.assert_allclose(native["time_s"], np.linspace(0, 1, 4))
        angle = Rotation.from_quat(
            native["qpos"][1, self.slices[0]], scalar_first=True
        ).magnitude()
        self.assertAlmostEqual(angle, 2 * np.arctan(np.sqrt(3) / 5), places=6)
        self.assert_rotations(
            sample_reference(self.model, self.layout, native, query)[0]
        )
        downsampled = native_reference(
            self.layout, source, SimpleNamespace(ref_dt=0.5, sim_dt=1.0, ref_steps=2)
        )
        np.testing.assert_allclose(downsampled["qpos"], sampled[:1])
        np.testing.assert_array_equal(downsampled["time_s"], [0])
        single = qpos_sample(self.layout, [0], qpos[:1], [-1, 0, 1])
        self.assert_rotations(single)
        np.testing.assert_array_equal(single[0], single[-1])

    def test_ik_resampling_uses_corrected_filter_and_original_clock(self):
        """Exercise the actual IK output path without rerunning the solver."""
        qpos = np.tile(self.model.qpos0, (8, 1))
        qpos[1::2, self.slices[0]] *= -1
        times = np.arange(8) * 0.02
        with patch("spider.h2r.kinematics.solve", return_value=qpos[:-1]):
            values, velocity, source = solve_kinematics(
                None,
                self.layout,
                qpos,
                times,
                {"right": np.zeros((8, 7))},
                {"right": np.zeros((8, 5, 3))},
                (None, None),
                {"ref_dt": 0.02, "average_frame_size": 3},
            )
        self.assert_rotations(values)
        np.testing.assert_allclose(velocity, 0)
        np.testing.assert_array_equal(source["native_qpos"], qpos[:-1])
        np.testing.assert_allclose(source["filtered_time_s"], [0.04, 0.06, 0.08, 0.1])

    def test_invalid_input_and_degenerate_mean_are_rejected(self):
        """Never normalize malformed sources or an ambiguous zero mean into a pose."""
        for bad in ([0, 0, 0, 0], [2, 0, 0, 0], [np.nan, 0, 0, 0]):
            qpos = np.tile(self.model.qpos0, (8, 1))
            qpos[2, self.slices[0]] = bad
            with self.assertRaisesRegex(ValueError, "root free_a at frame 2"):
                postprocess(self.model, qpos, 0.02, 3)
            with self.assertRaisesRegex(ValueError, "root a at frame 2"):
                qpos_sample(self.layout, np.arange(8), qpos, [0.5])
        qpos = np.tile(self.model.qpos0, (8, 1))
        qpos[:, self.slices[0]] = Rotation.from_euler(
            "z", np.arange(8)[:, None] * 120, degrees=True
        ).as_quat(scalar_first=True)
        with self.assertRaisesRegex(ValueError, "root free_a at frame 0"):
            postprocess(self.model, qpos, 0.02, 6)


if __name__ == "__main__":
    unittest.main()
