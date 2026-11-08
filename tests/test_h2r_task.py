"""Only a sealed benchmark task may supply the planning world's physics."""

import copy
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from test_h2r_integration import fixture

from spider.config import Config
from spider.h2r.bundle import (
    PREPARED_FILES,
    digest,
    identity,
    load_inputs,
    load_prepared,
    load_task,
    verify_xml,
    write_json,
)
from spider.h2r.scene import planning_model
from spider.h2r.settings import settings
from spider.simulators.mjwp import setup_mj_model


def task_fixture(root):
    """Seal a small multi-object scene and decoded reference without source assets."""
    model, task, _ = fixture()
    (root / "scene").mkdir()
    scene = root / "scene/scene.xml"
    mujoco.mj_saveLastXML(str(scene), model)
    mesh = root / "scene/probe.obj"
    mesh.write_text(
        "v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    tree = ET.parse(scene)
    assets = ET.SubElement(tree.getroot(), "asset")
    ET.SubElement(assets, "mesh", name="probe", file="probe.obj")
    ET.SubElement(
        tree.find("worldbody"), "geom", type="mesh", mesh="probe", pos="3 0 0"
    )
    tree.write(scene)
    np.savez(root / "reference.npz", time_s=np.arange(3) / 30)
    write_json(
        root / "reference.json",
        {
            "reference_sha256": digest(root / "reference.npz"),
            "coordinates": {"units": "metres", "quaternion": "wxyz"},
            "speed": 1.0,
            "task": {"hand_sides": ["right"]},
        },
    )
    task.update(format="h2r-evaluation-v3", scene="scene/scene.xml")
    task["files"] = {
        str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()
    }
    task["task_id"] = identity({k: v for k, v in task.items() if k != "task_id"})
    write_json(root / "task.json", task)
    return task


CALIBRATION = {"right": {"wrist_body": "hand", "fingertip_bodies": ["hand"] * 5}}


class TaskTests(unittest.TestCase):
    """Protect task physics, file closure and the artifact boundary."""

    def test_planning_model_preserves_physics_and_task_files(self):
        """Sites and a private timestep leave the task's physical model intact."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = task_fixture(root)
            loaded, _, arrays = load_inputs(root)
            self.assertEqual(loaded, task)
            np.testing.assert_array_equal(arrays["time_s"], np.arange(3) / 30)
            original = mujoco.MjModel.from_xml_path(str(root / task["scene"]))
            planning, wrists, tips = planning_model(root, task, CALIBRATION)
            self.assertEqual((len(wrists), len(tips)), (1, 5))
            self.assertEqual(planning.nsite, original.nsite + 6)
            physics_fields = (
                "body_mass",
                "body_inertia",
                "body_pos",
                "body_quat",
                "body_gravcomp",
                "jnt_type",
                "jnt_range",
                "jnt_stiffness",
                "dof_damping",
                "dof_armature",
                "dof_frictionloss",
                "geom_type",
                "geom_size",
                "geom_pos",
                "geom_quat",
                "geom_friction",
                "geom_solref",
                "geom_solimp",
                "geom_contype",
                "geom_conaffinity",
                "mesh_vert",
                "mesh_face",
                "actuator_gear",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_dynprm",
                "actuator_ctrlrange",
                "actuator_forcerange",
                "qpos0",
            )
            private = setup_mj_model(
                Config(sim_dt=1 / 30, physics_substeps=4), planning
            )
            for field in physics_fields:
                for actual in (planning, private):
                    np.testing.assert_array_equal(
                        getattr(actual, field), getattr(original, field), err_msg=field
                    )
            for field in (
                "gravity",
                "integrator",
                "solver",
                "iterations",
                "ls_iterations",
                "o_solref",
                "o_solimp",
            ):
                np.testing.assert_array_equal(
                    getattr(private.opt, field), getattr(original.opt, field)
                )
            self.assertEqual(planning.opt.timestep, original.opt.timestep)
            self.assertEqual(private.opt.timestep, 1 / 120)
            self.assertEqual(load_task(root), task)

    def test_changed_physics_or_mesh_rejected_before_compilation(self):
        """Updating a method's hashes cannot authorize changes to task resources."""
        for target in ("mass", "actuator", "mesh"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                task = task_fixture(root)
                if target == "mesh":
                    path = root / "scene/probe.obj"
                    path.write_text(path.read_text().replace("v .1", "v .2"))
                else:
                    path = root / task["scene"]
                    tree = ET.parse(path)
                    if target == "mass":
                        tree.find(".//body/geom").set("mass", "99")
                    else:
                        tree.find("actuator")[0].set("gear", "2")
                    tree.write(path)
                with self.assertRaisesRegex(ValueError, "Changed artifact"):
                    planning_model(root, task, CALIBRATION)
                # Even resealing the task creates a different task identity.
                changed = copy.deepcopy(task)
                changed["files"][str(path.relative_to(root))] = digest(path)
                changed["task_id"] = identity(
                    {k: v for k, v in changed.items() if k != "task_id"}
                )
                write_json(root / "task.json", changed)
                with self.assertRaisesRegex(ValueError, "task identity"):
                    planning_model(root, task, CALIBRATION)

    def test_prepared_cannot_supply_a_resealed_scene(self):
        """Reject legacy scenes and added model dependencies even with valid hashes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, task, _ = fixture()
            for name in PREPARED_FILES:
                (root / name).write_text("test method artifact")
            binding = {
                **{
                    k: task[k]
                    for k in (
                        "task_id",
                        "bundle_id",
                        "scene_id",
                        "robot_manifest_sha256",
                    )
                },
                "format": "spider-h2r-v4",
                "model_source": "task",
                "files": {name: digest(root / name) for name in PREPARED_FILES},
            }
            write_json(root / "h2r_binding.json", binding)
            self.assertEqual(load_prepared(root, task), binding)
            (root / "custom.xml").write_text(
                '<mujoco><option gravity="0 0 0"/></mujoco>'
            )
            for mutation in (
                {"planning_scene": "custom.xml"},
                {"model_source": "custom.xml"},
                {
                    "files": {
                        **binding["files"],
                        "custom.xml": digest(root / "custom.xml"),
                    }
                },
                {"format": "spider-h2r-v1"},
                {"format": "spider-h2r-v2"},
                {"format": "spider-h2r-v3"},
            ):
                write_json(root / "h2r_binding.json", {**binding, **mutation})
                with self.assertRaises(ValueError):
                    load_prepared(root, task)

    def test_xml_search_paths_and_symlinks_cannot_escape_task(self):
        """Prevent MuJoCo from loading a different file from the one verified."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for xml in (
                '<mujoco><compiler meshdir="/tmp"/></mujoco>',
                '<mujoco><asset><texture fileup="../outside.png"/></asset></mujoco>',
                '<mujoco><asset><model file="external.xml"/></asset></mujoco>',
            ):
                (root / "scene.xml").write_text(xml)
                with self.assertRaises(ValueError):
                    verify_xml(
                        root, "scene.xml", {"scene.xml": digest(root / "scene.xml")}
                    )
            (root / "escape.obj").symlink_to(root.parent / "outside.xml")
            (root / "scene.xml").write_text(
                '<mujoco><asset><mesh file="escape.obj"/></asset></mujoco>'
            )
            with self.assertRaisesRegex(ValueError, "outside the bundle"):
                verify_xml(root, "scene.xml", {"scene.xml": digest(root / "scene.xml")})

    def test_method_settings_cannot_select_physics_assets(self):
        """The recipe accepts algorithm budgets, never a model path or robot choice."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.yaml"
            for key in ("model_path", "robot_type", "gravity"):
                path.write_text(f"mpc:\n  {key}: custom\n")
                with self.assertRaises((TypeError, ValueError)):
                    settings(path)


if __name__ == "__main__":
    unittest.main()
