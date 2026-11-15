"""Prepare a portable SPIDER deployment from a fixed benchmark task."""

import argparse
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .artifacts import PREPARED_FORMAT, task_identity
from .h2r_io import load_task
from .h2r_io.files import sha256, write_json
from .kinematics import diagnostic_rollout, solve_kinematics
from .model import Layout
from .reference import hand_targets
from .scene import calibration_for, planning_spec
from .settings import PIPELINE, recipe, settings, validate_clock

# SPIDER's own pre-check of its initial pose proposal. The evaluation applies its
# own admission threshold, which is an evaluation setting and not known here.
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


def prepare(task_path, output, config=None):
    """Build references and seal the deployment only after successful validation."""
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
        raise ValueError(
            f"IK initialization rejected; see {output / 'initialization.json'}"
        )
    write_json(output / "initial_robot.json", layout.targets(layout.controls(qpos[0])))
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    )
    binding = {
        "format": PREPARED_FORMAT,
        **task_identity(task),
        "dataset": task.dataset,
        "profile": task.profile,
        "hand_sides": list(task.hand_sides),
        "pipeline": {**PIPELINE, "dataset_override": recipe(task.dataset)},
        "reference_mapping": "dataset-rate; sign-aligned-normalized-quaternion-mean; source-time-mapped-linear-and-nlerp; endpoint-value-hold",
        "method_revision": revision,
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
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(prepare(args.task, args.output, args.config))


if __name__ == "__main__":
    main()
