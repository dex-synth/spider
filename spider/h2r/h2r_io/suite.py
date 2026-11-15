"""A suite: the ordered task list that every method receives."""

from dataclasses import dataclass
from pathlib import Path

from .files import read_manifest, seal
from .task import load_task

FORMAT = "h2r-suite-v1"
MANIFEST = "suite.json"


@dataclass(frozen=True)
class Case:
    case_id: str              # "<dataset>/<profile>/<robot>"
    dataset: str
    profile: str
    robot: str
    hand_sides: tuple
    task_id: str
    task: Path                # task directory

    def load(self):
        task = load_task(self.task)
        if task.task_id != self.task_id:
            raise ValueError(f"Suite case {self.case_id} points at a different task")
        return task


@dataclass(frozen=True)
class Suite:
    path: Path
    suite_id: str
    name: str
    cases: tuple              # in distribution order
    provenance: dict          # how the suite was generated; not part of its identity


def _identity(manifest):
    return {"format": manifest["format"], "name": manifest["name"],
            "cases": [[case["case_id"], case["task_id"]] for case in manifest["cases"]]}


def load_suite(path):
    """Read the task list; each task is verified when its case is loaded."""
    path = Path(path).resolve()
    manifest = read_manifest(path, MANIFEST, FORMAT, _identity)
    cases = tuple(Case(case["case_id"], case["dataset"], case["profile"], case["robot"],
                       tuple(case["hand_sides"]), case["task_id"], path / case["task"])
                  for case in manifest["cases"])
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate suite case")
    return Suite(path, manifest["suite_id"], manifest["name"], cases, manifest.get("provenance", {}))


def seal_suite(path, name, tasks, provenance=None):
    """Write suite.json for task directories (relative to the suite, in order) that are already sealed."""
    path = Path(path)
    cases = []
    for relative in tasks:
        task = load_task(path / relative, root=path)
        cases.append({"case_id": f"{task.dataset}/{task.profile}/{task.robot.name}",
                      "dataset": task.dataset, "profile": task.profile, "robot": task.robot.name,
                      "hand_sides": list(task.hand_sides), "task_id": task.task_id,
                      "task": Path(relative).as_posix()})
    return seal(path, MANIFEST, {"format": FORMAT, "name": name, "cases": cases,
                                 "provenance": provenance or {}}, _identity)
