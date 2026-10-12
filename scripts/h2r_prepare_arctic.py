"""SPIDER fast-IK task adaptation for the common floating Leap/scissors asset."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from spider.preprocess.ik_fast import main as native_ik


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-avoidance", action="store_true", help="Declared fast-IK collision-limit variant")
    parser.add_argument("--max-ik-step-norm", type=float, help="Declared bound on each IK tangent update")
    args = parser.parse_args()
    source_meta = json.loads((args.bundle / "reference.json").read_text())
    asset_meta = json.loads((args.asset / "asset.json").read_text())
    if source_meta["reference_sha256"] != digest(args.bundle / "reference.npz"):
        raise ValueError("Changed common reference")
    if asset_meta["scene_sha256"] != digest(args.asset / "scene.xml"):
        raise ValueError("Changed common scene")
    args.output.mkdir(parents=True, exist_ok=False)
    source = np.load(args.bundle / "reference.npz", allow_pickle=False)
    count = len(source["time_s"])
    robot_dir = args.output / "processed/h2r/leap/right/scissors"
    mano_dir = args.output / "processed/h2r/mano/right/scissors"
    (robot_dir / "0").mkdir(parents=True)
    (mano_dir / "0").mkdir(parents=True)
    # Only the private planning actuator model changes. Evaluation assets are
    # never submitted back to the benchmark.
    tree = ET.parse(args.asset / "scene.xml")
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        mesh.set("file", str((args.asset / mesh.get("file")).resolve()))
    world = root.find("worldbody")
    ET.SubElement(world, "site", name="root_reference", pos="0 0 0", size="0.001")
    robot = root.find(".//body[@name='robot_base']")
    ET.SubElement(robot, "site", name="right_palm", pos="0 0 0", size="0.001")
    for finger, body in (("thumb", "thumb_tip_head"), ("index", "index_tip_head"),
                         ("middle", "middle_tip_head"), ("ring", "ring_tip_head")):
        ET.SubElement(root.find(f".//body[@name='robot_{body}']"), "site",
                      name=f"right_{finger}_tip", pos="0 0 0", size="0.001")
    root.remove(root.find("actuator"))
    actuators = ET.SubElement(root, "actuator")
    for axis in range(6):
        gear = np.eye(6)[axis]
        ET.SubElement(actuators, "position", name=f"root_{axis}", site="right_palm", refsite="root_reference",
                      gear=" ".join(map(str, gear)), kp="300" if axis < 3 else "3",
                      kv="30" if axis < 3 else "0.3", forcelimited="true",
                      forcerange="-57.7350269 57.7350269" if axis < 3 else "-5.77350269 5.77350269")
    for name in asset_meta["joint_names"]:
        limits = root.find(f".//joint[@name='{name}']").get("range")
        ET.SubElement(actuators, "position", name=name, joint=name, kp="3", kv="0.1",
                      ctrllimited="true", ctrlrange=limits, forcelimited="true", forcerange="-0.5 0.5")
    ET.indent(root)
    tree.write(robot_dir / "scene.xml", encoding="unicode")
    quats = Rotation.from_rotvec(source["right_mano_rot"]).as_quat(scalar_first=True)
    wrist = np.column_stack((source["right_mano_trans"], quats))
    fingers = np.zeros((count, 5, 7))
    fingers[:, :, :3] = source["right_joints_m"][:, [16, 17, 18, 19, 20]]
    fingers[:, :, 3] = 1.
    obj = np.column_stack((source["object_pos_m"], source["object_quat_wxyz"]))
    np.savez(mano_dir / "0/trajectory_keypoints.npz", qpos_wrist_right=wrist, qpos_wrist_left=wrist,
             qpos_finger_right=fingers, qpos_finger_left=fingers, qpos_obj_right=obj, qpos_obj_left=obj,
             object_hinge_rad=source["object_hinge_rad"])
    native_ik(dataset_dir=str(args.output), dataset_name="h2r", robot_type="leap", embodiment_type="right",
              task="scissors", show_viewer=False, save_video=False, start_idx=0, end_idx=count,
              sim_dt=1/480, ref_dt=1/30, average_frame_size=1,
              articulated_object=True, floating_hand=True, preserve_timing=True,
              collision_avoidance=args.collision_avoidance, max_step_norm=args.max_ik_step_norm)
    ik = np.load(robot_dir / "0/trajectory_kinematic.npz", allow_pickle=False)
    assert len(ik["qpos"]) == count
    model = mujoco.MjModel.from_xml_path(str(robot_dir / "scene.xml"))
    times = np.arange((count-1)*4+1)/120
    qpos = np.column_stack([np.interp(times, source["time_s"], ik["qpos"][:, i]) for i in range(model.nq)])
    for offset in (3, model.nq-5):
        qpos[:, offset:offset+4] = Slerp(source["time_s"], Rotation.from_quat(
            ik["qpos"][:, offset:offset+4], scalar_first=True))(times).as_quat(scalar_first=True)
    qvel = np.zeros((len(times), model.nv))
    for i in range(1, len(times)):
        mujoco.mj_differentiatePos(model, qvel[i], 1/120, qpos[i-1], qpos[i])
    data = mujoco.MjData(model)
    ctrl = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        ctrl.append(data.actuator_length.copy())
    np.savez(args.output / "reference.npz", time_s=times, qpos=qpos, qvel=qvel, ctrl=np.asarray(ctrl))
    joints = [qpos[0, model.joint(n).qposadr[0]] for n in asset_meta["joint_names"]]
    (args.output / "initial.json").write_text(json.dumps({"joint_names": asset_meta["joint_names"],
        "root_pose_wxyz": qpos[0, :7].tolist(), "joint_pos_rad": list(map(float, joints))}, indent=2)+"\n")
    info = {"recipe": "SPIDER-fast-Leap-articulated-site-servo", "common_reference_sha256": source_meta["reference_sha256"],
            "ik_collision_avoidance": args.collision_avoidance,
            "max_ik_step_norm": args.max_ik_step_norm,
            "asset_manifest_sha256": digest(args.asset / "asset.json"), "reference_hz": 120,
            "duration_s": float(times[-1]), "source_frames": count, "method_frames": len(times),
            "planning_scene": str((robot_dir / "scene.xml").resolve()), "joint_names": asset_meta["joint_names"],
            "adaptation": "Native fast IK, no time crop/filter; full object hinge; floating-root site position servo"}
    (args.output / "input.json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
