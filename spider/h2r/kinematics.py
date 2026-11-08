"""Map shared targets and physical timestamps to the paper IK numerical core."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from spider.preprocess.ik import make_ik_model, postprocess, solve

from .reference import sample_reference, velocities


def solve_kinematics(spec, layout, reference_q, times, wrists, tips, sites, options):
    """Solve at native 50 Hz, then preserve the complete common reference clock."""
    dt = options["ref_dt"]
    native_times = np.arange(int(np.ceil(times[-1] / dt)) + 1) * dt
    query = np.clip(native_times, times[0], times[-1])
    targets, poses = [], []
    wrist_sites, finger_sites = sites
    for side in (s for s in ("right", "left") if s in wrists):
        hand = list(wrists).index(side)
        wrist = wrists[side]
        pose = np.column_stack(
            [
                *[np.interp(query, times, column) for column in wrist[:, :3].T],
                Slerp(times, Rotation.from_quat(wrist[:, 3:], scalar_first=True))(
                    query
                ).as_quat(scalar_first=True),
            ]
        )
        targets.append((wrist_sites[hand], "wrist"))
        poses.append(pose)
        for finger in range(5):
            pose = np.zeros((len(query), 7))
            pose[:, 3] = 1
            for axis in range(3):
                pose[:, axis] = np.interp(query, times, tips[side][:, finger, axis])
            targets.append((finger_sites[5 * hand + finger], finger))
            poses.append(pose)
    reference = {
        "time_s": times,
        "qpos": reference_q,
        "qvel": velocities(layout.model, times, reference_q),
    }
    object_q, _, _ = sample_reference(layout.model, layout, reference, native_times)
    spec = spec.copy()
    for name, root in layout.roots.items():
        site = f"h2r_object_{name}"
        spec.body(root["body"]).add_site(name=site, size=[0.001, 0, 0], group=4)
        q, _ = layout.root_indices[name]
        targets.append((site, "object"))
        poses.append(object_q[:, q : q + 7])
    # Root targets are native welds. Passive hinges remain passive during IK;
    # all object references are restored at export, as owned by the benchmark.
    model, mocap_ids = make_ik_model(spec, targets, options)
    raw = solve(
        model,
        layout.model,
        np.stack(poses, axis=1),
        mocap_ids,
        [i for i, (_, kind) in enumerate(targets) if isinstance(kind, int)],
        layout.controls,
        options,
    )
    filtered, filtered_v = postprocess(model, raw, dt, options["average_frame_size"])
    valid_times = (
        np.arange(len(filtered)) + (options["average_frame_size"] - 1) / 2 + 1
    ) * dt
    # Keep the original scalar filter interior. Endpoint holds fill the common
    # task coverage instead of re-zeroing the cropped native trajectory clock.
    qpos = reference_q.copy()
    qvel = velocities(layout.model, times, reference_q)
    for q, v in zip(layout.robot_q, layout.robot_v, strict=True):
        qpos[:, q] = np.interp(times, valid_times, filtered[:, q])
        qvel[:, v] = np.interp(times, valid_times, filtered_v[:, v])
        qvel[(times < valid_times[0]) | (times > valid_times[-1]), v] = 0
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("Non-finite paper IK reference")
    return (
        qpos,
        qvel,
        {
            "native_time_s": native_times,
            "native_qpos": raw,
            "filtered_time_s": valid_times,
            "filtered_qpos": filtered,
            "filtered_qvel": filtered_v,
        },
    )
