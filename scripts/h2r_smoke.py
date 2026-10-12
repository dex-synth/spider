"""Check real CUDA kernels and MuJoCo Warp stepping in the dedicated env."""
import json

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


@wp.kernel
def add_one(values: wp.array(dtype=float)):
    i = wp.tid()
    values[i] += 1.0


def main():
    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda:0")
    y = x @ x.T
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
    wp.init()
    assert wp.is_cuda_available(), "Warp CUDA is unavailable; CPU fallback is not a pass"
    values = wp.zeros(16, dtype=float, device="cuda:0")
    wp.launch(add_one, dim=16, inputs=[values], device="cuda:0")
    wp.synchronize()
    np.testing.assert_allclose(values.numpy(), 1)
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <worldbody><body pos="0 0 1"><freejoint/>
      <geom type="sphere" size="0.03" mass="0.1"/></body></worldbody>
      </mujoco>''')
    data = mujoco.MjData(model)
    with wp.ScopedDevice("cuda:0"):
        model_wp = mjw.put_model(model)
        data_wp = mjw.put_data(model, data, nworld=1)
        for _ in range(100):
            mjw.step(model_wp, data_wp)
    wp.synchronize()
    qpos = data_wp.qpos.numpy()
    assert np.isfinite(qpos).all() and 0.7 < qpos[0, 2] < 0.9, qpos
    print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                      "warp": wp.__version__, "mujoco": mujoco.__version__,
                      "drop_z_m": float(qpos[0, 2]), "status": "passed"}))


if __name__ == "__main__":
    main()
