"""A method's rollout of one task in its own simulator, handed to the benchmark as files.

The method runs the artifact it deploys (the same checkpoint or plan) in the simulator it trained or
planned in, from the evaluation initial state, and records the generalized coordinates of every physics
step. The benchmark derives body poses with its own model, resamples the states to its grid and scores
them with the common metrics (`h2r-bench import-method-rollout`). The format is specified in
docs/formats.md.
"""

from pathlib import Path

import numpy as np

from .files import manifest_fields, member, read_json, sha256, write_json

FORMAT = "h2r-method-rollout-v1"
MANIFEST = "method_rollout.json"
# How the rollout was run. Every package states all of these; the benchmark accepts only these values.
CONDITIONS = {"initial_state": "evaluation", "initial_velocity": "zero", "assistance": "none",
              "disturbance": "none", "action": "deterministic"}
SIMULATOR_FIELDS = ("name", "version", "physics_dt_s", "control_dt_s", "device", "deterministic")
VERDICT_FIELDS = ("success", "reason", "ended_at_s", "metrics")


def state_names(task):
    """Column names of the state arrays, in the order they are stored."""
    binding = task.object_binding
    return {"joint_names": list(task.robot.semantics["joint_names"]),
            "object_roots": list(binding["roots"]), "object_joints": list(binding["joints"])}


def _check_states(states, names):
    """time_s (N,), robot_q (N, joints), object_root_poses (N, roots, 7: xyz + wxyz), object_q (N, object joints)."""
    time = np.asarray(states["time_s"])
    shapes = {"time_s": (len(time),), "robot_q": (len(time), len(names["joint_names"])),
              "object_root_poses": (len(time), len(names["object_roots"]), 7),
              "object_q": (len(time), len(names["object_joints"]))}
    if set(states) != set(shapes):
        raise ValueError(f"States hold exactly {sorted(shapes)}")
    for key, shape in shapes.items():
        value = np.asarray(states[key])
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"State {key} must be finite floating point of shape {shape}")
    if len(time) < 2 or time[0] != 0 or not (np.diff(time) > 0).all():
        raise ValueError("time_s must start at 0 and increase strictly")
    norms = np.linalg.norm(np.asarray(states["object_root_poses"])[..., 3:], axis=-1)
    if not np.allclose(norms, 1, rtol=0, atol=1e-4):
        raise ValueError("Object root quaternions must have unit length")


def _check_manifest(manifest):
    if any(not isinstance(manifest["binding"].get(key), str) for key in ("demo_id", "robot_id", "method")) \
            or not isinstance(manifest["binding"].get("controller"), dict):
        raise ValueError("binding needs demo_id, robot_id, method and the controller reported to the evaluation")
    simulator, conditions = manifest["simulator"], manifest["conditions"]
    if set(simulator) != set(SIMULATOR_FIELDS) or type(simulator["deterministic"]) is not bool \
            or any(not isinstance(simulator[key], str) or not simulator[key] for key in ("name", "version", "device")) \
            or any(type(simulator[key]) not in (int, float) or not 0 < simulator[key] < np.inf
                   for key in ("physics_dt_s", "control_dt_s")):
        raise ValueError(f"simulator needs {SIMULATOR_FIELDS}")
    if set(conditions) != set(CONDITIONS) or not all(isinstance(value, str) for value in conditions.values()):
        raise ValueError(f"conditions state each of {sorted(CONDITIONS)}")
    if not isinstance(manifest["criteria"], str) or not manifest["criteria"].strip():
        raise ValueError("criteria describes the method's own success rule")
    for episode in manifest["episodes"]:
        verdict = episode["verdict"]
        if set(verdict) != set(VERDICT_FIELDS) or verdict["success"] not in (True, False, None) \
                or not isinstance(verdict["metrics"], dict):
            raise ValueError(f"Each episode verdict needs {VERDICT_FIELDS}")


def write_method_rollout(output, task, *, method, controller, training_seed, simulator, conditions, criteria,
                         episodes):
    """Write one package for `task` (an h2r_io Task).

    `controller` is the same record the method reports to the evaluation in its start binding.
    `simulator` has SIMULATOR_FIELDS, `conditions` every key of CONDITIONS as the rollout was actually run,
    and `criteria` describes the method's own success rule. Each episode is
    `{"states": {...}, "verdict": {...}}`: state columns follow `state_names(task)`, `time_s` is the step
    count times the physics step and the first state is the initial one; the verdict has VERDICT_FIELDS,
    with `ended_at_s` set when the method ended the episode before the end of the task.
    """
    output, names = Path(output), state_names(task)
    manifest = {"format": FORMAT, "task_id": task.task_id,
                "binding": {"demo_id": task.demo.demo_id, "robot_id": task.robot.robot_id, "method": method,
                            "controller": controller},
                "training_seed": training_seed, "simulator": dict(simulator), "conditions": dict(conditions),
                "criteria": criteria, **names,
                "episodes": [{"states": f"episode-{i}/states.npz", "verdict": dict(episode["verdict"])}
                             for i, episode in enumerate(episodes)]}
    with manifest_fields(MANIFEST):
        _check_manifest(manifest)
        for episode in episodes:
            _check_states(episode["states"], names)
    output.mkdir(parents=True)
    for entry, episode in zip(manifest["episodes"], episodes):
        path = output / entry["states"]
        path.parent.mkdir()
        np.savez_compressed(path, **{key: np.asarray(value, dtype=float) for key, value in episode["states"].items()})
        entry["sha256"] = sha256(path)
    write_json(output / MANIFEST, manifest)
    return output


def load_method_rollout(path, task):
    """Return the verified manifest and the states of each episode; the package must belong to `task`."""
    path, names = Path(path).resolve(), state_names(task)
    manifest = read_json(path / MANIFEST)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError(f"Expected {FORMAT} in {path.name}/{MANIFEST}")
    with manifest_fields(f"{path.name}/{MANIFEST}"):
        _check_manifest(manifest)
        binding = manifest["binding"]
        if (manifest["task_id"], binding["demo_id"], binding["robot_id"]) != (
                task.task_id, task.demo.demo_id, task.robot.robot_id):
            raise ValueError("Method rollout belongs to another task")
        if any(manifest[key] != value for key, value in names.items()):
            raise ValueError("Method rollout names differ from the task")
        if not manifest["episodes"]:
            raise ValueError("Method rollout has no episodes")
        episodes = []
        for entry in manifest["episodes"]:
            file = member(path, entry["states"])
            if not file.is_file() or sha256(file) != entry["sha256"]:
                raise ValueError(f"Missing or changed file: {entry['states']}")
            with np.load(file, allow_pickle=False) as data:
                states = {key: data[key] for key in data.files}
            _check_states(states, names)
            episodes.append(states)
    return manifest, episodes
