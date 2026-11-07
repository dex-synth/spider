"""Prepare a portable SPIDER deployment from a fixed human/robot/task bundle."""

import argparse
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mink
import mujoco
import numpy as np
import yaml

from spider.preprocess.ik_fast import make_tasks, solve_velocity

from .bundle import digest, load_inputs, read_json, write_json
from .model import Layout
from .reference import hand_targets, velocities
from .settings import settings, validate_clock


def planning_model(task_path, task, output, calibration):
    """Copy the sealed scene; add only massless, non-colliding tracking sites."""
    source = Path(task_path) / task["scene"]
    shutil.copytree(source.parent, output)
    path = output / source.name
    tree = ET.parse(path)
    wrist_sites, finger_sites = [], []
    for side in task["semantics"]["hands"]:
        spec = calibration[side]
        bodies = [spec["wrist_body"], *spec["fingertip_bodies"]]
        names = [f"h2r_{side}_wrist", *[f"h2r_{side}_tip_{i}" for i in range(5)]]
        for body, name in zip(bodies, names, strict=True):
            node = tree.find(f".//body[@name='{body}']")
            if node is None or tree.find(f".//site[@name='{name}']") is not None:
                raise ValueError(
                    f"Missing tracking body or duplicate site: {body}/{name}"
                )
            ET.SubElement(node, "site", name=name, pos="0 0 0", size="0.001", group="4")
        wrist_sites.append(names[0])
        finger_sites.extend(names[1:])
    tree.write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    return path, model, wrist_sites, finger_sites


def solve_kinematics(layout, reference_q, times, wrists, tips, sites, options):
    """Run native fast IK while freezing every non-robot degree of freedom."""
    model = layout.model
    configuration = mink.Configuration(model)
    initial = reference_q[0].copy()
    for name, q in zip(layout.names, layout.robot_q, strict=True):
        joint = model.joint(name)
        if joint.limited[0]:
            initial[q] = np.clip(initial[q], *joint.range)
    configuration.update(initial)
    costs = {k: v for k, v in options.items() if k.endswith("_cost")}
    wrist_tasks, finger_tasks, posture = make_tasks(configuration, *sites, **costs)
    constraints = (
        [mink.DofFreezingTask(model, layout.passive_dofs)]
        if layout.passive_dofs
        else []
    )
    fixed_q = sorted(set(range(model.nq)) - set(layout.robot_q))
    sides = list(wrists)

    def targets(frame):
        for i, side in enumerate(sides):
            pose = wrists[side][frame]
            wrist_tasks[i].set_target(mink.SE3(wxyz_xyz=np.r_[pose[3:], pose[:3]]))
            for j in range(5):
                finger_tasks[5 * i + j].set_target(
                    mink.SE3(wxyz_xyz=np.r_[1.0, 0, 0, 0, tips[side][frame, j]])
                )
        value = configuration.q.copy()
        value[fixed_q] = reference_q[frame, fixed_q]
        configuration.update(value)

    def advance(tasks, count, dt):
        for _ in range(count):
            velocity = solve_velocity(configuration, tasks, dt, constraints=constraints)
            configuration.integrate_inplace(velocity, dt)

    targets(0)
    advance([posture, *wrist_tasks], options["wrist_init_steps"], options["dt"])
    advance(
        [posture, *wrist_tasks, *finger_tasks],
        options["finger_init_steps"],
        options["dt"],
    )
    result = []
    for frame in range(len(times)):
        targets(frame)
        interval = times[frame] - times[frame - 1] if frame else times[1] - times[0]
        count = max(1, int(np.ceil(interval / options["dt"])))
        advance([posture, *wrist_tasks, *finger_tasks], count, interval / count)
        value = configuration.q.copy()
        value[fixed_q] = reference_q[frame, fixed_q]
        result.append(value)
        if frame % 100 == 0:
            print(f"IK frame {frame}/{len(times)}", flush=True)
    result = np.asarray(result)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite IK result")
    return result


def check_initialization(model, layout, qpos, task):
    """Check the proposed zero-velocity initial frame without advancing physics."""
    for name, q in zip(layout.names, layout.robot_q, strict=True):
        joint = model.joint(name)
        if (
            joint.limited[0]
            and not joint.range[0] - 1e-6 <= qpos[q] <= joint.range[1] + 1e-6
        ):
            raise ValueError(f"Initial joint outside limits: {name}")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = layout.controls(qpos)
    mujoco.mj_forward(model, data)
    threshold = task["initialization"]["max_penetration_m"]
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
        "max_penetration_m": threshold,
        "accepted": not collisions,
        "collisions": collisions,
    }


def prepare(human, robot, task_path, output, config=None):
    """Build references and seal the deployment only after successful validation."""
    started = time.perf_counter()
    task_path, output = Path(task_path).resolve(), Path(output).resolve()
    for source in (human, robot, task_path):
        if output.is_relative_to(Path(source).resolve()):
            raise ValueError("Output must be outside input bundles")
    task, reference, data = load_inputs(human, robot, task_path)
    options = settings(config)
    validate_clock(options["mpc"], task)
    calibration = read_json(Path(__file__).parent / "config/sharpa_wave.json")
    if task["robot_name"] not in calibration["robots"]:
        raise ValueError(f"No SPIDER hand calibration for {task['robot_name']}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(options, sort_keys=False)
    )
    write_json(output / "calibration.json", calibration)
    path, model, wrist_sites, finger_sites = planning_model(
        task_path, task, output / "planning", calibration
    )
    layout = Layout(model, task)
    wrists, tips = hand_targets(
        reference, data, task["semantics"]["hands"], calibration
    )
    reference_q = layout.reference_positions(reference, data)
    qpos = solve_kinematics(
        layout,
        reference_q,
        data["time_s"],
        wrists,
        tips,
        (wrist_sites, finger_sites),
        options["ik"],
    )
    times = data["time_s"]
    qvel = velocities(model, times, qpos)
    np.savez_compressed(
        output / "trajectory_kinematic.npz",
        time_s=times,
        qpos=qpos,
        qvel=qvel,
        ctrl=layout.controls(qpos),
    )
    np.savez_compressed(
        output / "trajectory_keypoints.npz",
        time_s=times,
        **{f"{side}_wrist": pose for side, pose in wrists.items()},
        **{f"{side}_tips": points for side, points in tips.items()},
    )
    check = check_initialization(model, layout, qpos[0], task)
    write_json(output / "initialization.json", check)
    if not check["accepted"]:
        raise ValueError(f"IK initial collision; see {output / 'initialization.json'}")
    write_json(output / "initial_robot.json", layout.targets(layout.controls(qpos[0])))
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    )
    binding = {
        "format": "spider-h2r-v1",
        "recipe": "fast-ik-native-sampling-h2r",
        "method_revision": revision,
        "method_worktree_dirty": dirty,
        "planning_backend": "mjwarp",
        "planning_scene": str(path.relative_to(output)),
        "contact_guidance": False,
        "contact_rew_scale": 0,
        "object_reward": "all dynamic roots and passive hinges",
        "time_s": [float(times[0]), float(times[-1])],
        "frames": len(times),
        "prepare_elapsed_s": time.perf_counter() - started,
        **{
            k: task[k]
            for k in ("task_id", "bundle_id", "scene_id", "robot_manifest_sha256")
        },
        "files": {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(output / "h2r_binding.json", binding)
    return output


def main():
    """Prepare from explicit artifact paths, without loading a raw dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("human-bundle", "robot-bundle", "task", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.human_bundle, args.robot_bundle, args.task, args.output, args.config
        )
    )


if __name__ == "__main__":
    main()
