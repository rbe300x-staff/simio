"""Exceptions raised by simio.

Every simio failure raises one of these. simio never returns ``None`` or an
error dict to signal a problem (the 2026 ``coppeliaBridge`` did both).
"""


class SimError(Exception):
    """A simulator call failed. The message names the object and the cause."""


class SimConnectionError(SimError):
    """Could not connect to CoppeliaSim, or its version is too old."""


class SimThreadError(SimError):
    """The simulator was used from a thread other than the one that created it.

    A CoppeliaSim ZMQ client must be created and used on one thread. In miniROS,
    only ``sim_bridge`` touches the simulator, and it creates the client from its
    own callback (api-spec §5, §20).
    """


class SimStoppedError(SimError):
    """The simulation is not running (for example, it was stopped from the GUI)."""
