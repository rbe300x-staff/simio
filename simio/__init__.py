"""simio: the course's simulator interface (RBE 300x, api-spec §5).

* ``Sim``: the real CoppeliaSim, through the ZMQ remote API.
* ``FakeSim``: the same interface with no simulator (scripted or kinematic).
* ``SimInterface``: the Protocol both satisfy; ``sim_bridge`` depends on it.
* ``LineWorld``: the 2D track that FakeSim's kinematic mode drives on.

Course-provided: teams use simio but don't edit it.
"""

from .errors import SimConnectionError, SimError, SimStoppedError, SimThreadError
from .fake import FakeSim
from .interface import SimInterface
from .robot import Geometry, Handles, RobotSpec, load_robot_yaml
from .sim import Sim
from .world import LineWorld, Obstacle

__all__ = [
    "Sim",
    "FakeSim",
    "SimInterface",
    "LineWorld",
    "Obstacle",
    "Handles",
    "RobotSpec",
    "Geometry",
    "load_robot_yaml",
    "SimError",
    "SimConnectionError",
    "SimThreadError",
    "SimStoppedError",
]

__version__ = "0.1.1"
