"""Hand a CPU plan's own trajectory to the benchmark as a method rollout.

Next to its independent evaluation, the benchmark reports how a method does in its
own simulator: the same deployed artifact, the simulator it was planned in, the
evaluation initial state, no assistance and no disturbance. For the CPU recipe that
rollout is the plan itself. This entry replays the sealed targets in one private
world, checks that they reproduce the states the planner recorded, and writes the
state of every physics step as an `h2r-method-rollout-v1` package. The benchmark
imports and scores it as `method-sim`. Nothing is planned again.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .artifacts import INITIAL_POSE_ONLY, arrays
from .cpu_plan import REPLAY_TOLERANCE
from .cpu_scene import private_model
from .evaluation import METHOD
from .h2r_io import load_task
from .h2r_io.method_rollout import write_method_rollout
from .replay import load_plan

# How this rollout is run. The benchmark accepts a method rollout only as such.
CONDITIONS = {
    "initial_state": "evaluation",  # the proposed robot pose, objects at frame 0
    "initial_velocity": "zero",
    "assistance": "none",  # the private model has the task's actuators only
    "disturbance": "none",
    "action": "deterministic",  # the sealed targets
}
CRITERIA = (
    "Upstream evaluate_cpu `success` on the planner's states in the private world: "
    "every object body's mean error under 3 cm and 30 deg, mean hinge error under "
    "10 deg, every opening of the reference followed, and no early stop."
)
NATIVE_METRICS = (
    "success_object",
    "frac_ok",
    "obj_pos_cm",
    "obj_rot_deg",
    "backtracks",
)


def initial_qpos(model, layout, task, initial):
    """The evaluation initial state: the plan's robot pose, objects at frame 0."""
    qpos = np.zeros(model.nq)
    qpos[layout.robot_q] = [initial[name] for name in layout.names]
    start = task.initial_object_state
    for name, (address, _) in layout.root_indices.items():
        qpos[address : address + 7] = start["object_roots"][name]
    qpos[layout.object_q] = [start["object_joints"][key] for key in layout.joints]
    return qpos


def replay(model, layout, qpos, targets, substeps):
    """`qpos` after every physics step, holding each target for one control frame."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    states = [data.qpos.copy()]
    for target in targets:
        data.ctrl[layout.actuators] = target
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            states.append(data.qpos.copy())
    return np.array(states)


def export(plan, task_path, output):
    """Write the method rollout of one sealed CPU plan.

    A plan without controls (its initial pose failed SPIDER's pre-check) gives a
    rollout that ends after one physics step in that pose: the benchmark applies
    its own admission check to the first state.

    Raises:
        ValueError: The plan is not a CPU plan of this task.
        RuntimeError: The replay does not reproduce the planned states.
    """
    plan, task = Path(plan), load_task(task_path)
    binding, controls, initial = load_plan(plan, task)
    if binding.get("recipe") != "cpu":
        raise ValueError("Method rollouts are exported for CPU plans only")
    control_dt = binding["control_dt_s"]
    substeps = int(
        yaml.safe_load((plan / "resolved_config.yaml").read_text())["h2r"]["substeps"]
    )
    physics_dt = control_dt / substeps
    model, layout, _ = private_model(task, physics_dt)
    start = initial_qpos(model, layout, task, initial)
    if binding["deployment"] == INITIAL_POSE_ONLY:
        qpos = np.array([start, start])
        verdict = {
            "success": False,
            "reason": "initial pose fails SPIDER's pre-check; not planned",
            "ended_at_s": physics_dt,
            "metrics": {},
        }
    else:
        qpos = replay(model, layout, start, controls["joint_targets"], substeps)
        planned = arrays(plan / "trajectory_cpu.npz")["qpos"]
        deviation = float(np.abs(qpos[substeps::substeps] - planned).max())
        if not deviation <= REPLAY_TOLERANCE:
            raise RuntimeError(
                f"The replay differs from the planned states by {deviation:.3e}; "
                f"the plan was made with MuJoCo {binding['planner']['mujoco_version']}"
            )
        native = binding["native"]
        verdict = {
            "success": bool(native["success"]),
            "reason": None,
            "ended_at_s": None,
            "metrics": {name: native[name] for name in NATIVE_METRICS},
        }
    roots = [qpos[:, a : a + 7] for a, _ in layout.root_indices.values()]
    states = {
        "time_s": np.arange(len(qpos)) * physics_dt,
        "robot_q": qpos[:, layout.robot_q],
        "object_root_poses": np.stack(roots, axis=1),
        "object_q": qpos[:, layout.object_q],
    }
    return write_method_rollout(
        output,
        task,
        method=METHOD,
        controller=binding,  # as reported to the evaluation (spider.h2r.evaluation)
        training_seed=None,
        simulator={
            "name": "mujoco",
            "version": mujoco.__version__,
            "physics_dt_s": physics_dt,
            "control_dt_s": control_dt,
            "device": "cpu",
            "deterministic": True,
        },
        conditions=CONDITIONS,
        criteria=CRITERIA,
        episodes=[{"states": states, "verdict": verdict}],
    )


def main():
    """Export the method rollout of one plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "task", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.plan, args.task, args.output))


if __name__ == "__main__":
    main()
