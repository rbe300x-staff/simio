"""Sim: the real CoppeliaSim, through the ZMQ remote API (api-spec §5.1).

This replaces the 2026 ``coppeliaBridge``. Differences that matter:

* No internal thread or request queue. One thread creates the Sim and only that
  thread may use it (calls from other threads raise SimThreadError). In miniROS,
  ``sim_bridge`` is that owner.
* Failures raise SimError subclasses with the object path and the cause. Nothing
  returns ``None`` or an error dict.
* ``connect`` fails fast if CoppeliaSim isn't running, instead of hanging.
* ``start()`` returns once the sim is running and its first step is done, so no
  ``sleep()`` is needed after it.
* Stepping mode (``start(stepping=True)`` + ``step()``) gives exact sim time.
* ``stop()``, ``close()``, and the ``with`` form zero the wheels.

Verified with tools/check_sim.py on CoppeliaSim 4.10.0 with lineFollower.ttt
(2026-11-10): connect, handles, stepping start, the vision-sensor packet layout
(mean intensity at index 10), readings, pose, driving, the thread rule, and stop.
Not yet verified live: the proximity sensor and a GUI stop (run check_sim.py with
lineTracer_prox.yaml and --interactive on the obstacle scene). Version-dependent
calls (``require``/``getObject``, ``sim.setStepping``/``client.setStepping``,
``sim.step``/``client.step``) try the CoppeliaSim 4.6+ form first and fall back
to the older one.

Readings are on the scene's scale, not 0-to-1: in lineFollower.ttt bare floor
reads about 0.83 and the darkest line pixels about 0.38. Controllers should take
their threshold from configuration, not assume 0.5.
"""

from __future__ import annotations

import math
import socket
import time as _time
from typing import Any, Callable

from ._owner import OwnerThread
from .errors import SimConnectionError, SimError, SimStoppedError
from .robot import Handles, load_robot_yaml

#: Oldest CoppeliaSim version simio accepts (major, minor). The course installs and
#: tests 4.10.0; 4.6 is the oldest version with the API calls simio uses.
MIN_VERSION = (4, 6)
TESTED_VERSION = (4, 10, 0)

#: Index of the mean intensity in readVisionSensor's first auxiliary packet
#: (min[0:5], max[5:10], mean[10:15]; each block is intensity, r, g, b, depth).
#: Confirmed on CoppeliaSim 4.10.0 with tools/check_sim.py (2026-11-10).
MEAN_INTENSITY_INDEX = 10

_START_TIMEOUT_SEC = 5.0
_POLL_SEC = 0.01


class Sim:
    """A connection to a running CoppeliaSim. Create it with :meth:`connect`."""

    # ------------------------------------------------------------------ setup

    def __init__(self, client: Any, sim: Any, *, host: str = "?", port: int = 0) -> None:
        """Use :meth:`connect`. (Tests may pass a stand-in client and sim object.)"""
        self._owner = OwnerThread()
        self._client = client
        self._sim = sim
        self._where = f"{host}:{port}"
        self._stepping = False
        self._started = False
        self._closed = False
        self._wheels: tuple[int, int] | None = None
        self._last_yaml = "the robot YAML"

    @classmethod
    def connect(cls, host: str = "localhost", port: int = 23000, timeout_sec: float = 5.0) -> "Sim":
        """Connect to CoppeliaSim. Raises SimConnectionError if it isn't reachable or is too old."""
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient  # type: ignore
        except ImportError:
            raise SimConnectionError(
                "the CoppeliaSim client isn't installed. Run: pip install coppeliasim-zmqremoteapi-client"
            ) from None

        # The ZMQ client blocks forever if nothing is listening, so probe the port first.
        try:
            with socket.create_connection((host, port), timeout=timeout_sec):
                pass
        except OSError as e:
            raise SimConnectionError(
                f"nothing is listening at {host}:{port} ({e}). Is CoppeliaSim open? "
                "Its ZMQ remote API server starts with the program (default port 23000)."
            ) from None

        try:
            client = RemoteAPIClient(host, port)
            if hasattr(client, "require"):
                sim = client.require("sim")
            else:
                sim = client.getObject("sim")
        except Exception as e:  # noqa: BLE001 - anything from the client is a connection failure here
            raise SimConnectionError(f"connected to {host}:{port} but could not load the 'sim' API: {e}") from None

        self = cls(client, sim, host=host, port=port)
        self._check_version()
        return self

    def _check_version(self) -> None:
        try:
            v = int(self._sim.getInt32Param(self._sim.intparam_program_version))
        except Exception as e:  # noqa: BLE001
            raise SimConnectionError(f"could not read the CoppeliaSim version: {e}") from None
        major, minor = v // 10000, (v // 100) % 100
        self.version = (major, minor, v % 100)
        if (major, minor) < MIN_VERSION:
            raise SimConnectionError(
                f"CoppeliaSim {major}.{minor}.{v % 100} is too old; the course needs "
                f"{MIN_VERSION[0]}.{MIN_VERSION[1]} or later (see the install guide)."
            )

    # ---------------------------------------------------------------- helpers

    def _guard(self, what: str) -> None:
        self._owner.check(what)
        if self._closed:
            raise SimError(f"{what}: this Sim was closed")

    def _call(self, what: str, fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return fn(*args)
        except SimError:
            raise
        except Exception as e:  # noqa: BLE001 - remote errors arrive as plain exceptions
            raise SimError(f"{what} failed: {e}") from None

    def _ensure_running(self, what: str) -> None:
        state = self._call("getSimulationState", self._sim.getSimulationState)
        if state == self._sim.simulation_stopped:
            self._started = False
            raise SimStoppedError(
                f"{what}: the simulation is not running (it was stopped, or never started). "
                "Call start() to run it again."
            )

    def _set_stepping(self, on: bool) -> None:
        if hasattr(self._sim, "setStepping"):
            self._call("setStepping", self._sim.setStepping, on)
        else:
            self._call("setStepping", self._client.setStepping, on)

    def _do_step(self) -> None:
        if hasattr(self._sim, "step"):
            self._call("step", self._sim.step)
        else:
            self._call("step", self._client.step)

    def _wait(self, what: str, condition: Callable[[], bool]) -> None:
        """Poll a simulator condition until it holds; time out as a failure."""
        deadline = _time.monotonic() + _START_TIMEOUT_SEC
        while not condition():
            if _time.monotonic() > deadline:
                raise SimError(f"{what}: timed out after {_START_TIMEOUT_SEC:.0f} s")
            _time.sleep(_POLL_SEC)

    def _is_stopped(self) -> bool:
        return self._call("getSimulationState", self._sim.getSimulationState) == self._sim.simulation_stopped

    # --------------------------------------------------------------- handles

    def handles_from_yaml(self, path: str) -> Handles:
        """Look up the robot's scene objects named in a robot YAML file."""
        self._guard("handles_from_yaml")
        spec = load_robot_yaml(path)

        def get(obj_path: str) -> int:
            try:
                return int(self._sim.getObject(obj_path))
            except Exception as e:  # noqa: BLE001
                raise SimError(
                    f"scene object '{obj_path}' (from {spec.path}) was not found: {e}. "
                    "Check that the scene is open and the YAML names match the scene hierarchy."
                ) from None

        handles = Handles(
            base=get(spec.base_path),
            left=get(spec.object_path(spec.left_motor)),
            right=get(spec.object_path(spec.right_motor)),
            line_sensors=tuple(get(spec.object_path(s)) for s in spec.line_sensors),
            proximity=get(spec.object_path(spec.proximity)) if spec.proximity else None,
            yaml_path=spec.path,
        )
        self._last_yaml = spec.path
        return handles

    # ------------------------------------------------------------- lifecycle

    def start(self, stepping: bool = False) -> None:
        """Start the simulation and return once its first step is done.

        If the simulation is already running (for example, someone pressed Play),
        it is stopped and restarted so the stepping mode is what you asked for.
        """
        self._guard("start")
        if not self._is_stopped():
            self._call("stopSimulation", self._sim.stopSimulation)
            self._wait("waiting for the running simulation to stop", self._is_stopped)
        self._set_stepping(stepping)
        self._stepping = stepping
        self._call("startSimulation", self._sim.startSimulation)
        if stepping:
            self._do_step()
        else:
            self._wait(
                "waiting for the first simulation step",
                lambda: self._call("getSimulationTime", self._sim.getSimulationTime) > 0.0,
            )
        self._started = True

    def is_running(self) -> bool:
        self._guard("is_running")
        return not self._is_stopped()

    def step(self) -> None:
        """Advance one simulation step (stepping mode only)."""
        self._guard("step")
        if not self._stepping:
            raise SimError("step() needs stepping mode: call start(stepping=True)")
        self._ensure_running("step")
        self._do_step()

    def stop(self) -> None:
        """Zero the wheels, then stop the simulation and wait until it has stopped."""
        self._guard("stop")
        self._zero_wheels_quietly()
        if not self._is_stopped():
            self._call("stopSimulation", self._sim.stopSimulation)
            self._wait("waiting for the simulation to stop", self._is_stopped)
        self._started = False

    def close(self) -> None:
        """Zero the wheels if the sim is running, then disconnect. Does not stop the sim."""
        self._owner.check("close")
        if self._closed:
            return
        self._zero_wheels_quietly()
        self._closed = True
        self._client = None

    def _zero_wheels_quietly(self) -> None:
        if self._wheels is None:
            return
        try:
            if not self._is_stopped():
                left, right = self._wheels
                self._sim.setJointTargetVelocity(left, 0.0)
                self._sim.setJointTargetVelocity(right, 0.0)
        except Exception:  # noqa: BLE001 - best effort during shutdown
            pass

    def __enter__(self) -> "Sim":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.stop()
        except SimError:
            pass
        finally:
            self.close()
        # Returns None: never swallows the original exception (including KeyboardInterrupt).

    # ------------------------------------------------------------ sense, act

    def read_line_sensors(self, handles: tuple[int, ...]) -> tuple[float, ...]:
        """Mean intensity in [0, 1] for each vision sensor, in the order given. Low = dark (line)."""
        self._guard("read_line_sensors")
        self._ensure_running("read_line_sensors")
        values = []
        for h in handles:
            result = self._call(f"readVisionSensor(handle {h})", self._sim.readVisionSensor, h)
            if isinstance(result, (list, tuple)):
                code, packet = result[0], (result[1] if len(result) > 1 else None)
            else:
                code, packet = result, None
            if code is None or code < 0 or not packet or len(packet) <= MEAN_INTENSITY_INDEX:
                raise SimError(
                    f"vision sensor (handle {h}) returned no image data (result {code!r}). "
                    "Is the sensor enabled, and has the simulation taken a step?"
                )
            values.append(min(1.0, max(0.0, float(packet[MEAN_INTENSITY_INDEX]))))
        return tuple(values)

    def read_proximity(self, handle: int | None) -> tuple[bool, float]:
        """``(detected, distance_m)``; distance is ``math.inf`` when nothing is detected."""
        self._guard("read_proximity")
        if handle is None:
            raise SimError(f"this robot has no proximity sensor in {self._last_yaml}")
        self._ensure_running("read_proximity")
        result = self._call(f"readProximitySensor(handle {handle})", self._sim.readProximitySensor, handle)
        code = result[0] if isinstance(result, (list, tuple)) else result
        if code and code > 0:
            return True, float(result[1])
        return False, math.inf

    def set_wheel_speeds(self, left: int, right: int, left_rad_s: float, right_rad_s: float) -> None:
        """Set the target velocity (rad/s) of the left and right wheel joints."""
        self._guard("set_wheel_speeds")
        self._ensure_running("set_wheel_speeds")
        self._call(f"setJointTargetVelocity(handle {left})", self._sim.setJointTargetVelocity, left, float(left_rad_s))
        self._call(f"setJointTargetVelocity(handle {right})", self._sim.setJointTargetVelocity, right, float(right_rad_s))
        self._wheels = (left, right)

    def get_pose2d(self, handle: int) -> tuple[float, float, float]:
        """Ground-truth ``(x, y, theta)`` of an object in the world frame.

        ``theta`` is the heading of the object's x-axis. For the robot base, add
        ``geometry.base_yaw_offset`` if the robot's forward direction isn't the
        base's x-axis (tools/measure_robot.py reports it).
        """
        self._guard("get_pose2d")
        self._ensure_running("get_pose2d")
        m = self._call(f"getObjectMatrix(handle {handle})", self._sim.getObjectMatrix, handle, self._sim.handle_world)
        # m is a 3x4 row-major matrix: [Xx Yx Zx Px, Xy Yy Zy Py, Xz Yz Zz Pz]
        return float(m[3]), float(m[7]), math.atan2(float(m[4]), float(m[0]))

    def time(self) -> float:
        """Simulation time in seconds."""
        self._guard("time")
        return float(self._call("getSimulationTime", self._sim.getSimulationTime))

    @property
    def api(self) -> Any:
        """The raw CoppeliaSim 'sim' API object, for staff tools (tools/*.py).

        Lab code shouldn't need this; if it does, the missing call belongs in simio.
        The owner-thread rule still applies to anything done through it.
        """
        self._guard("api")
        return self._sim

    @property
    def dt(self) -> float:
        """The scene's simulation time step, in seconds."""
        self._guard("dt")
        return float(self._call("getSimulationTimeStep", self._sim.getSimulationTimeStep))
