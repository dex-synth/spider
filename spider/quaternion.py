"""Quaternion representation fixes shared by fast IK and H2R references."""

import numpy as np


def normalize_qpos(qpos, roots, *, require_unit=True):
    """Copy qpos, normalizing and aligning each named free-root quaternion.

    Args:
        qpos: Time-ordered generalized positions, shape (T, nq).
        roots: Mapping from root name to (qpos address, velocity address).
        require_unit: Validate source norms; false for mean/interpolation outputs.

    Returns:
        Positions with unit, sign-continuous wxyz quaternions. Scalars are intact.
    """
    result = np.array(qpos, copy=True)
    for name, (address, _) in roots.items():
        q = result[:, address + 3 : address + 7]
        norms = np.linalg.norm(q, axis=-1, keepdims=True)
        valid = (
            np.isfinite(q).all(axis=-1)
            & np.isfinite(norms[:, 0])
            & (norms[:, 0] > 1e-12)
        )
        if require_unit:
            valid &= np.isclose(norms[:, 0], 1, atol=1e-4, rtol=0)
        if not valid.all():
            frame = int(np.flatnonzero(~valid)[0])
            raise ValueError(f"Invalid quaternion for root {name} at frame {frame}")
        q /= norms
        largest = q[np.arange(len(q)), np.abs(q).argmax(axis=-1)]
        q *= np.where(largest < 0, -1, 1)[:, None]
        for i in range(1, len(q)):
            if q[i - 1] @ q[i] < 0:
                q[i] *= -1
    return result
