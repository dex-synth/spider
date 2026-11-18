"""Read-only access to h2r-bench demos, robots, tasks and suites, and the evaluation wire protocol.

Depends only on NumPy and the standard library (Python >= 3.10), so a method can vendor this
directory instead of installing h2r-bench. The formats are specified in docs/formats.md.
"""

from . import client
from .demo import Demo, Hand, InteractionObject, Joint, load_demo
from .robot import Robot, load_robot
from .suite import Case, Suite, load_suite
from .task import Task, export_task, load_task

__version__ = "1"
__all__ = ["Case", "Demo", "Hand", "InteractionObject", "Joint", "Robot", "Suite", "Task",
           "client", "export_task", "load_demo", "load_robot", "load_suite", "load_task"]
