"""MANO decoding in the common convention of docs/formats.md.

Optional module: it needs smplx and torch, and the licensed MANO_LEFT.pkl / MANO_RIGHT.pkl files.
Preprocessing uses it to produce `joints_m` and `vertices_m`; a method that re-decodes a hand
gets the same geometry from the same parameters.
"""

from contextlib import contextmanager
from functools import lru_cache
import inspect

import numpy as np

CONVENTION = "h2r-mano-v1"
TIP_VERTEX_IDS = (744, 320, 443, 554, 671)  # thumb, index, middle, ring, little; the same for both hands


@contextmanager
def legacy_mano_pickle():
    """Let the licensed Chumpy-era MANO pickles load under current NumPy and Python.

    The aliases exist only while the model is constructed. Use this on trusted local assets only.
    """
    aliases = dict(bool=bool, int=int, float=float, complex=complex, object=object, unicode=str, str=str)
    added = {key: value for key, value in aliases.items() if key not in np.__dict__}
    old = getattr(inspect, "getargspec", None)
    try:
        for key, value in added.items():
            setattr(np, key, value)
        if old is None:
            inspect.getargspec = inspect.getfullargspec
        yield
    finally:
        for key in added:
            delattr(np, key)
        if old is None:
            delattr(inspect, "getargspec")


@lru_cache(maxsize=4)
def model(model_dir, side, left_shapedirs_x_mirrored=False):
    """The flat-mean MANO model; rotations passed to it are absolute."""
    import smplx

    with legacy_mano_pickle():
        hand = smplx.MANO(str(model_dir), is_rhand=side == "right", use_pca=False,
                          flat_hand_mean=True, create_transl=False).eval()
    if side == "left" and left_shapedirs_x_mirrored:
        right = model(str(model_dir), "right")
        # The legacy left asset repeats the right hand's x shape directions; mirror them once.
        if (hand.shapedirs[:, 0] - right.shapedirs[:, 0]).abs().sum() < 1:
            hand.shapedirs[:, 0] *= -1
    return hand


def rotation_vectors(quat):
    """Unit wxyz quaternions (..., 4) to axis-angle vectors (..., 3), taking the shorter rotation."""
    quat = np.where(quat[..., :1] < 0, -quat, quat)
    norm = np.linalg.norm(quat[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(norm, quat[..., :1])
    return quat[..., 1:] * np.divide(angle, norm, out=np.full_like(norm, 2.0), where=norm > 1e-12)


def decode(model_dir, side, quat, trans, shape, *, left_shapedirs_x_mirrored=False):
    """Geometry for `mano_quat` (T, 16, 4), `mano_trans` (T, 3) and `mano_shape` (10,).

    Returns `vertices_m` (T, 778, 3), `joints_m` (T, 21, 3: 16 model joints, then the fingertips)
    and `faces`; the wrist joint lies at `trans`.
    """
    import torch

    hand = model(str(model_dir), side, left_shapedirs_x_mirrored)
    count = len(quat)
    angles = torch.as_tensor(rotation_vectors(np.asarray(quat, dtype=float)).reshape(count, 48), dtype=torch.float32)
    betas = torch.as_tensor(np.broadcast_to(shape, (count, 10)).copy(), dtype=torch.float32)
    wrist = torch.as_tensor(np.asarray(trans), dtype=torch.float32)
    vertices, joints = [], []
    with torch.inference_mode():
        for rows in (slice(start, start + 256) for start in range(0, count, 256)):
            out = hand(global_orient=angles[rows, :3], hand_pose=angles[rows, 3:], betas=betas[rows])
            offset = wrist[rows, None] - out.joints[:, :1]
            vertices.append((out.vertices + offset).numpy())
            joints.append((torch.cat([out.joints[:, :16], out.vertices[:, list(TIP_VERTEX_IDS)]], dim=1) + offset).numpy())
    return {"vertices_m": np.concatenate(vertices), "joints_m": np.concatenate(joints),
            "faces": hand.faces.astype(np.int32)}


def rest_wrist(model_dir, side, shape, *, left_shapedirs_x_mirrored=False):
    """Wrist joint of the unposed hand, (3,): the offset between smplx `transl` and the wrist position."""
    import torch

    hand = model(str(model_dir), side, left_shapedirs_x_mirrored)
    with torch.inference_mode():
        out = hand(betas=torch.as_tensor(np.asarray(shape, dtype=np.float32)[None]))
    return out.joints[0, 0].numpy().astype(float)
