"""Prepare a portable SPIDER deployment from a fixed benchmark task.

Two robot references exist, one per recipe: SPIDER's own fast Mink IK (the fast
MJWarp recipe, the default here) and the V2D Sharpa retargeting that upstream's CPU
workflow is built on, imported with `--reference-from` from a FlashCHORD
preparation of the same task.
"""

import argparse
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .artifacts import (
    PREPARED_FORMAT,
    REFERENCE_MINK,
    REFERENCE_V2D,
    arrays,
    task_identity,
)
from .h2r_io import load_task
from .h2r_io.files import read_json, sha256, write_json
from .model import Layout
from .settings import PIPELINE, recipe, settings, validate_clock

# FlashCHORD preparations whose `robot_reference.npz` holds the V2D IK in the
# task's joint names on the demo's time grid.
FLASHCHORD_FORMATS = {"flashchord-h2r-v3"}

# SPIDER's own check of its IK initial pose, equal to the default evaluation admission.
# A pose that fails it is still proposed: the native IK has no collision avoidance, and
# its penetration is a failure of the method that the evaluation records as such. The
# check only lets planning skip the MPC for a start the evaluation would reject.
PRECHECK_MAX_PENETRATION_M = 0.02


def check_initialization(model, layout, qpos):
    """Check the proposed zero-velocity initial frame without advancing physics."""
    joint_limits = []
    for name, q in zip(layout.names, layout.robot_q, strict=True):
        joint = model.joint(name)
        if (
            joint.limited[0]
            and not joint.range[0] - 1e-6 <= qpos[q] <= joint.range[1] + 1e-6
        ):
            joint_limits.append(
                {"joint": name, "qpos": float(qpos[q]), "range": joint.range.tolist()}
            )
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = layout.controls(qpos)
    mujoco.mj_forward(model, data)
    threshold = PRECHECK_MAX_PENETRATION_M
    collisions = [
        {
            "geom1": model.geom(c.geom1).name,
            "geom2": model.geom(c.geom2).name,
            "depth_m": float(-c.dist),
        }
        for c in data.contact
        if c.dist < -threshold
    ]
    return {
        "precheck_max_penetration_m": threshold,
        "accepted": not collisions and not joint_limits,
        "joint_limit_violations": joint_limits,
        "collisions": collisions,
    }


def revision():
    """Commit and dirtiness of this checkout, recorded with every artifact."""
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    )
    return commit, dirty


def load_v2d_reference(source, task, model, layout):
    """Read the V2D joint reference of a FlashCHORD preparation of this very task.

    Returns the joint positions in the task's joint order, (T, joints), and the
    record of where they came from. Anything that would need a guess is rejected.
    """
    source = Path(source).resolve()
    theirs = read_json(source / "h2r_binding.json")
    if theirs.get("format") not in FLASHCHORD_FORMATS:
        raise ValueError(
            f"Unsupported FlashCHORD preparation {theirs.get('format')!r}: "
            f"expected one of {sorted(FLASHCHORD_FORMATS)}"
        )
    for key, value in task_identity(task).items():
        if theirs.get(key) != value:
            raise ValueError(f"FlashCHORD preparation was made for another task: {key}")
    path = source / "robot_reference.npz"
    digest = sha256(path)
    if theirs.get("files", {}).get("robot_reference.npz") != digest:
        raise ValueError("robot_reference.npz differs from its FlashCHORD binding")
    data = arrays(path)
    names = [str(name) for name in data["joint_names"]]
    joint_q, times = data["joint_q"], task.demo.time_s
    if len(set(names)) != len(names) or set(names) != set(layout.names):
        raise ValueError("V2D reference joints differ from the task robot")
    if data["time_s"].shape != times.shape or not np.allclose(
        data["time_s"], times, rtol=0, atol=1e-9
    ):
        raise ValueError("V2D reference is not on the demo's time grid")
    if joint_q.shape != (len(times), len(names)) or not np.isfinite(joint_q).all():
        raise ValueError("Invalid V2D joint reference")
    joint_q = joint_q[:, [names.index(name) for name in layout.names]]
    for column, name in enumerate(layout.names):
        joint = model.joint(name)
        low, high = joint.range
        if (
            joint.limited[0]
            and not (
                (joint_q[:, column] >= low - 1e-5) & (joint_q[:, column] <= high + 1e-5)
            ).all()
        ):
            raise ValueError(f"V2D reference leaves the joint range: {name}")
    # The V2D wrist decomposition wraps by 2 pi past its limits; joint coordinates
    # cannot be interpolated across such a jump.
    rotation = task.robot.semantics["joint_groups"]["wrist_rotation"]
    columns = [layout.names.index(name) for name in rotation]
    if len(times) > 1 and columns:
        jump = np.abs(np.diff(joint_q[:, columns], axis=0)).max()
        if jump >= np.pi:
            raise ValueError(f"V2D wrist angles jump by {jump:.3f} rad between frames")
    record = {
        "kind": REFERENCE_V2D,
        "flashchord_format": theirs["format"],
        "flashchord_binding_sha256": sha256(source / "h2r_binding.json"),
        "robot_reference_sha256": digest,
        "ik_settings": theirs.get("method"),
    }
    return joint_q, record


def import_reference(task_path, source, output):
    """Seal a preparation around the V2D reference; no IK runs here.

    The robot reference is taken as is, without SPIDER's own filtering; objects
    come from the task reference. The initial pose gets the same pre-check as the
    Mink path.
    """
    started = time.perf_counter()
    output = Path(output).resolve()
    task = load_task(task_path)
    if output.is_relative_to(task.path):
        raise ValueError("Output must be outside the task")
    model = mujoco.MjModel.from_xml_path(str(task.scene_xml))
    layout = Layout(model, task)
    joint_q, record = load_v2d_reference(source, task, model, layout)
    output.mkdir(parents=True, exist_ok=False)
    times = task.demo.time_s
    qpos = layout.reference_positions()
    qpos[:, layout.robot_q] = joint_q
    np.savez_compressed(
        output / "trajectory_kinematic.npz",
        time_s=times,
        qpos=qpos,
        ctrl=layout.controls(qpos),
    )
    write_json(output / "reference_source.json", record)
    check = check_initialization(model, layout, qpos[0])
    write_json(output / "initialization.json", check)
    if not check["accepted"]:
        print(
            "V2D initial pose fails SPIDER's pre-check; it is proposed as is and "
            f"planning will be skipped (see {output / 'initialization.json'})",
            flush=True,
        )
    write_json(output / "initial_robot.json", layout.targets(layout.controls(qpos[0])))
    commit, dirty = revision()
    binding = {
        "format": PREPARED_FORMAT,
        **task_identity(task),
        "dataset": task.dataset,
        "profile": task.profile,
        "hand_sides": list(task.hand_sides),
        "reference": REFERENCE_V2D,
        "reference_mapping": "v2d-joint-coordinates-at-dataset-rate; "
        "objects-from-task-reference; no-method-filtering",
        "method_revision": commit,
        "method_worktree_dirty": dirty,
        "object_reward": "all dynamic roots and passive hinges",
        "time_s": [float(times[0]), float(times[-1])],
        "frames": len(times),
        "prepare_elapsed_s": time.perf_counter() - started,
        "files": {
            str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(output / "h2r_binding.json", binding)
    return output


def prepare(task_path, output, config=None):
    """Build references and seal the deployment only after successful validation."""
    # Mink and the fast IK are needed on this path only; the CPU recipe's
    # environment has neither.
    from .kinematics import diagnostic_rollout, solve_kinematics
    from .reference import hand_targets
    from .scene import calibration_for, planning_spec

    started = time.perf_counter()
    output = Path(output).resolve()
    task = load_task(task_path)
    if output.is_relative_to(task.path):
        raise ValueError("Output must be outside the task")
    options = settings(config, dataset=task.dataset, ref_dt=1 / task.demo.reference_hz)
    validate_clock(options["mpc"])
    calibration = calibration_for(task)
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(options, sort_keys=False)
    )
    write_json(output / "calibration.json", calibration)
    spec, wrist_sites, finger_sites = planning_spec(task, calibration)
    model = spec.compile()
    layout = Layout(model, task)
    wrists, tips = hand_targets(task, calibration)
    reference_q = layout.reference_positions()
    times = task.demo.time_s
    qpos, qvel, native = solve_kinematics(
        spec,
        layout,
        reference_q,
        times,
        wrists,
        tips,
        (wrist_sites, finger_sites),
        options["ik"],
    )
    np.savez_compressed(
        output / "trajectory_kinematic.npz",
        time_s=times,
        qpos=qpos,
        qvel=qvel,
        ctrl=layout.controls(qpos),
        **native,
    )
    np.savez_compressed(
        output / "trajectory_keypoints.npz",
        time_s=times,
        **{f"{side}_wrist": pose for side, pose in wrists.items()},
        **{f"{side}_tips": points for side, points in tips.items()},
    )
    np.savez_compressed(
        output / "trajectory_ikrollout.npz",
        qpos=diagnostic_rollout(
            model,
            layout,
            native["filtered_qpos"],
            native["filtered_qvel"],
            options["ik"],
        ),
        time_s=native["filtered_time_s"],
    )
    # Method-smoothed objects must never redefine the shared initial object pose.
    initial_qpos = reference_q[0].copy()
    initial_qpos[layout.robot_q] = qpos[0, layout.robot_q]
    check = check_initialization(model, layout, initial_qpos)
    write_json(output / "initialization.json", check)
    if not check["accepted"]:
        print(
            "IK initial pose fails SPIDER's pre-check; it is proposed as is and "
            f"planning will be skipped (see {output / 'initialization.json'})",
            flush=True,
        )
    write_json(output / "initial_robot.json", layout.targets(layout.controls(qpos[0])))
    commit, dirty = revision()
    binding = {
        "format": PREPARED_FORMAT,
        **task_identity(task),
        "dataset": task.dataset,
        "profile": task.profile,
        "hand_sides": list(task.hand_sides),
        "reference": REFERENCE_MINK,
        "pipeline": {**PIPELINE, "dataset_override": recipe(task.dataset)},
        "reference_mapping": "dataset-rate; sign-aligned-normalized-quaternion-mean; source-time-mapped-linear-and-nlerp; endpoint-value-hold",
        "method_revision": commit,
        "method_worktree_dirty": dirty,
        "planning_backend": "mjwarp",
        "contact_guidance": False,
        "contact_rew_scale": 0,
        "object_reward": "all dynamic roots and passive hinges",
        "time_s": [float(times[0]), float(times[-1])],
        "frames": len(times),
        "prepare_elapsed_s": time.perf_counter() - started,
        "files": {
            str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(output / "h2r_binding.json", binding)
    return output


def main():
    """Prepare from one benchmark task directory, without loading a raw dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("task", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="fast MJWarp recipe overrides")
    parser.add_argument(
        "--reference-from",
        type=Path,
        help="FlashCHORD preparation of the same task: import its V2D reference "
        "for the CPU recipe instead of running SPIDER's Mink IK",
    )
    args = parser.parse_args()
    if args.reference_from is not None:
        if args.config is not None:
            parser.error("--config belongs to the Mink path, not to --reference-from")
        print(import_reference(args.task, args.reference_from, args.output))
    else:
        print(prepare(args.task, args.output, args.config))


if __name__ == "__main__":
    main()
