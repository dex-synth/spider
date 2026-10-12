"""Focused task adaptation checks, no licensed dataset or GUI required."""
import unittest

import mink
import mujoco
import numpy as np
import torch

from spider.config import Config
from spider.preprocess.ik_fast import _DisplacementCollisionLimit
from spider.simulators.mjwp import _diff_qpos


class BoundaryChecks(unittest.TestCase):
    def test_collision_limit_bounds_displacement_independently_of_dt(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body pos="0 0 0"><joint type="slide" axis="1 0 0"/><geom name="a" type="sphere" size=".01"/></body>
          <body pos=".025 0 0"><joint type="slide" axis="1 0 0"/><geom name="b" type="sphere" size=".01"/></body>
          </worldbody></mujoco>''')
        config = mink.Configuration(model)
        limit = _DisplacementCollisionLimit(model, [(["a"], ["b"])], gain=.5,
                                            minimum_distance_from_collisions=.001,
                                            collision_detection_distance=.02)
        one = limit.compute_qp_inequalities(config, 1/30)
        fast = limit.compute_qp_inequalities(config, 1/480)
        np.testing.assert_allclose(one.h, [.002], atol=1e-12)
        np.testing.assert_allclose(one.G, fast.G)
        np.testing.assert_allclose(one.h, fast.h)
        self.assertTrue(np.all(one.G @ np.array([.001, 0.]) <= one.h))
        self.assertFalse(np.all(one.G @ np.array([.003, 0.]) <= one.h))

    def test_floating_hand_and_hinge_difference_matches_mujoco(self):
        fingers = ''.join('<body><joint/><geom size=".01"/></body>' for _ in range(16))
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <body><freejoint/><geom size=".01"/>{fingers}</body>
          <body><freejoint/><geom size=".01"/><body><joint/><geom size=".01"/></body></body>
          </worldbody></mujoco>''')
        before = model.qpos0.copy()
        after = before.copy()
        velocity = np.linspace(-.1, .2, model.nv)
        mujoco.mj_integratePos(model, after, velocity, .2)
        expected = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, expected, 1., before, after)
        config = Config(embodiment_type="floating_hand_articulated", nv=model.nv, device="cpu")
        actual = _diff_qpos(config, torch.tensor(after[None], dtype=torch.float32),
                            torch.tensor(before[None], dtype=torch.float32)).numpy()[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
