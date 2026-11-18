"""Only a sealed benchmark task may supply the planning world's physics."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from test_h2r_integration import OBJECTS, SEMANTICS, XML, demo_content

from spider.config import Config
from spider.h2r import h2r_io
from spider.h2r.artifacts import (
    PREPARED_FILES,
    PREPARED_FORMAT,
    REFERENCE_MINK,
    load_prepared,
)
from spider.h2r.h2r_io.demo import seal_demo
from spider.h2r.h2r_io.files import sha256, write_json
from spider.h2r.h2r_io.robot import seal_robot
from spider.h2r.h2r_io.task import seal_task
from spider.h2r.model import Layout
from spider.h2r.scene import planning_model
from spider.h2r.settings import PIPELINE, settings
from spider.simulators.mjwp import setup_mj_model

CALIBRATION = {"right": {"wrist_body": "hand", "fingertip_bodies": ["hand"] * 5}}
MESH = "v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


def write_task(root):
    """Seal the toy scene as a standalone task with its demo and robot packages."""
    demo, robot = root / "demo", root / "robot"
    (demo / "meshes").mkdir(parents=True)
    (demo / "meshes/probe.obj").write_text(MESH)
    (demo / "scene.xml").write_text("<mujoco/>")
    reference, scene, arrays = demo_content(OBJECTS, ("right",), 3, "arctic")
    np.savez(demo / "reference.npz", **arrays)
    write_json(demo / "reference.json", reference)
    write_json(demo / "scene.json", scene)
    seal_demo(demo, input_key="test")
    (robot / "assets").mkdir(parents=True)
    for name in ("robot.xml", "assets/robot.urdf", "assets/right.urdf"):
        (robot / name).write_text("<robot/>")
    write_json(robot / "semantics.json", {**SEMANTICS, "hands": {"right": {}}})
    write_json(robot / "compiled.json", {})
    seal_robot(
        robot,
        "synthetic",
        ["right"],
        mjcf="robot.xml",
        urdf="assets/robot.urdf",
        hand_urdfs={"right": "assets/right.urdf"},
    )
    tree = ET.ElementTree(ET.fromstring(XML))
    assets = ET.SubElement(tree.getroot(), "asset")
    ET.SubElement(assets, "mesh", name="probe", file="demo/meshes/probe.obj")
    ET.SubElement(
        tree.find("worldbody"), "geom", type="mesh", mesh="probe", pos="3 0 0"
    )
    tree.write(root / "scene.xml")
    write_json(root / "compiled.json", {})
    seal_task(root, "demo", "robot")
    return root


class TaskTests(unittest.TestCase):
    """Protect task physics, the loader boundary and the artifact boundary."""

    def test_planning_model_preserves_physics_of_loaded_task(self):
        """Sites and a private timestep leave the task's physical model intact."""
        with tempfile.TemporaryDirectory() as tmp:
            task = h2r_io.load_task(write_task(Path(tmp)))
            self.assertEqual((task.dataset, task.hand_sides), ("arctic", ("right",)))
            original = mujoco.MjModel.from_xml_path(str(task.scene_xml))
            planning, wrists, tips = planning_model(task, CALIBRATION)
            self.assertEqual((len(wrists), len(tips)), (1, 5))
            self.assertEqual(planning.nsite, original.nsite + 6)
            layout = Layout(planning, task)
            self.assertEqual(set(layout.root_indices), {"a", "b"})
            self.assertEqual(layout.reference_positions().shape, (3, planning.nq))
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

    def test_changed_task_or_old_input_rejected_by_loader(self):
        """A changed scene, mesh or manifest never reaches model compilation."""
        for target in ("mass", "mesh", "format"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = write_task(Path(tmp))
                if target == "mesh":
                    path = root / "demo/meshes/probe.obj"
                    path.write_text(path.read_text().replace("v .1", "v .2"))
                elif target == "mass":
                    tree = ET.parse(root / "scene.xml")
                    tree.find(".//body/geom").set("mass", "99")
                    tree.write(root / "scene.xml")
                else:  # The former task directory format.
                    manifest = json.loads((root / "task.json").read_text())
                    write_json(
                        root / "task.json", {**manifest, "format": "h2r-evaluation-v3"}
                    )
                with self.assertRaises(ValueError):
                    h2r_io.load_task(root)

    def test_prepared_holds_only_current_method_references(self):
        """Reject scenes, unknown processing and older formats even with valid hashes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = h2r_io.load_task(write_task(root / "task"))
            prepared = root / "prepared"
            prepared.mkdir()
            files = PREPARED_FILES[REFERENCE_MINK]
            for name in files:
                (prepared / name).write_text("test method artifact")
            binding = {
                "format": PREPARED_FORMAT,
                "task_id": task.task_id,
                "demo_id": task.demo.demo_id,
                "robot_id": task.robot.robot_id,
                "reference": REFERENCE_MINK,
                "pipeline": PIPELINE,
                "files": {name: sha256(prepared / name) for name in files},
            }
            write_json(prepared / "h2r_binding.json", binding)
            self.assertEqual(load_prepared(prepared, task), binding)
            (prepared / "custom.xml").write_text(
                '<mujoco><option gravity="0 0 0"/></mujoco>'
            )
            for mutation in (
                {"pipeline": {**PIPELINE, "rotation_processing": "unknown"}},
                {"pipeline": None},
                {"reference": "another-ik"},
                {"reference": None},
                {"files": {}},
                {
                    "files": {
                        **binding["files"],
                        "custom.xml": sha256(prepared / "custom.xml"),
                    }
                },
                *({"format": f"spider-h2r-v{i}"} for i in range(1, 6)),
            ):
                write_json(prepared / "h2r_binding.json", {**binding, **mutation})
                with self.assertRaises(ValueError):
                    load_prepared(prepared, task)
            write_json(prepared / "h2r_binding.json", binding)
            (prepared / "calibration.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed file"):
                load_prepared(prepared, task)

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
