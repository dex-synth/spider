"""Inspect saved common-task SPIDER IK with Viser; no optimization or rollout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import mujoco
import numpy as np

from spider.viewers import viser_viewer as viewer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--status", type=Path)
    args = parser.parse_args()
    meta = json.loads((args.prepared / "input.json").read_text())
    source_meta = json.loads((args.bundle / "reference.json").read_text())
    if digest(args.bundle / "reference.npz") != meta["common_reference_sha256"]:
        raise ValueError("Human demonstration does not match the prepared IK")
    scene = Path(meta["planning_scene"])
    trajectory = scene.parent / "0/trajectory_kinematic.npz"
    with np.load(trajectory, allow_pickle=False) as arrays:
        qpos = arrays["qpos"].copy()
        frequency = float(arrays["frequency"])
    with np.load(args.bundle / "reference.npz", allow_pickle=False) as arrays:
        source = {k: arrays[k].copy() for k in
                  ("time_s", "source_frame", "right_vertices_m", "right_faces")}
    if len(qpos) != len(source["time_s"]) or frequency != source_meta["reference_hz"]:
        raise ValueError("The saved kinematic result does not match the source clock")
    if not np.isfinite(qpos).all():
        raise ValueError("Non-finite IK trajectory")
    # Fail on an occupied requested port, rather than silently displaying a new URL.
    with socket.socket() as probe:
        probe.bind((args.host, args.port))

    spec = mujoco.MjSpec.from_file(str(scene))
    # Display-only colors. The saved scene and all trajectory arrays stay intact.
    for body in spec.bodies:
        for geom in body.geoms:
            if body.name == "object_bottom":
                geom.rgba = [0.15, 0.55, 0.95, 1.]
            elif body.name == "object_top":
                geom.rgba = [1., 0.48, 0.12, 1.]
    model = spec.compile()
    if qpos.shape[1] != model.nq:
        raise ValueError("IK trajectory and planning model dimensions differ")
    data = mujoco.MjData(model)
    body_ids = [model.body(name).id for name in ("robot_base", "object_bottom", "object_top")]
    hinge_address = model.joint("scissors_joint").qposadr[0]
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.xpos).all():
            raise ValueError("Non-finite forward kinematics")
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    center = np.mean(data.xpos[body_ids], axis=0)

    server = viewer.init_viser("SPIDER · Kinematic retarget", host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    handles = viewer.build_and_log_scene_from_spec(spec, model, scene, build_ref=False)
    viewer._STATE.scene_checkboxes["collision_meshes"].value = False
    for handle in viewer._STATE.collision_geom_handles:
        handle.visible = False
    viewer._STATE.scene_checkboxes["ref"].visible = False
    human = server.scene.add_mesh_simple(
        "/human/right", vertices=source["right_vertices_m"][0], faces=source["right_faces"],
        color=(120, 220, 165), opacity=0.3, side="double", visible=False)
    server.gui.add_markdown(
        f"**SPIDER kinematic retarget**\n\n"
        f"ARCTIC `{source_meta['subject']}/{source_meta['sequence']}` · "
        f"frames **{source['source_frame'][0]}–{source['source_frame'][-1]}**\n\n"
        "Blue: scissors bottom · Orange: scissors top.\n\n"
        "Saved IK poses; this view is not a physical rollout.")
    with server.gui.add_folder("Playback"):
        frame = server.gui.add_slider("Frame", min=0, max=len(qpos)-1, step=1, initial_value=0)
        playing = server.gui.add_checkbox("Play", initial_value=False)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        speed = server.gui.add_slider("Speed", min=0.25, max=2., step=0.25, initial_value=1.)
        restart = server.gui.add_button("First frame")
        information = server.gui.add_markdown("")
    with server.gui.add_folder("View"):
        show_human = server.gui.add_checkbox("Human demonstration", initial_value=False)
        reset_camera = server.gui.add_button("Reset camera")

    def set_camera(client):
        client.camera.up_direction = (0., 0., 1.)
        client.camera.look_at = tuple(center)
        client.camera.position = tuple(center + np.array([0.32, -0.48, 0.27]))

    @server.on_client_connect
    def on_connect(client):
        set_camera(client)

    @reset_camera.on_click
    def on_camera(event):
        if event.client is not None:
            set_camera(event.client)

    @restart.on_click
    def on_restart(_):
        playing.value = False
        frame.value = 0

    @show_human.on_update
    def on_human(_):
        human.visible = show_human.value

    def render(index):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        with server.atomic():
            viewer.update_frame(data, handles)
            human.vertices = source["right_vertices_m"][index]
            information.content = (
                f"Source frame: **{source['source_frame'][index]}**  \n"
                f"Time: **{source['time_s'][index]:.3f} / {source['time_s'][-1]:.3f} s**  \n"
                f"Scissors hinge: **{qpos[index, hinge_address]:.3f} rad**")
        server.flush()

    render(0)
    report = {"kind": "kinematic_visualization", "physical_rollout": False, "status": "ready",
              "pid": os.getpid(), "host": args.host, "port": server.get_port(),
              "trajectory": str(trajectory), "trajectory_sha256": digest(trajectory),
              "scene": str(scene), "scene_sha256": digest(scene), "frames": len(qpos),
              "frequency_hz": frequency, "duration_s": float(source["time_s"][-1]),
              "all_frames_forward_kinematics_finite": True,
              "object_bodies": ["object_bottom", "object_top"],
              "hinge_range_rad": [float(qpos[:, hinge_address].min()), float(qpos[:, hinge_address].max())]}
    if args.status:
        args.status.parent.mkdir(parents=True, exist_ok=True)
        args.status.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    previous_frame = 0
    next_tick = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if playing.value and now >= next_tick:
                if frame.value < len(qpos)-1:
                    frame.value += 1
                elif loop.value:
                    frame.value = 0
                else:
                    playing.value = False
                next_tick = now + 1 / (frequency * speed.value)
            elif not playing.value:
                next_tick = now
            index = int(frame.value)
            if index != previous_frame:
                render(index)
                previous_frame = index
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
