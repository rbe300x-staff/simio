"""FakeSim: the same interface as Sim, with no CoppeliaSim (api-spec §5.3).

Two modes:

* **Scripted** (``world=None``): tests set the sensor readings with
  ``set_line_sensors`` / ``set_proximity`` and check what the code under test
  does with them (``last_wheel_speeds``, ``wheel_history``).
* **Kinematic** (``world=LineWorld(...)``): the robot moves. Each ``step()``
  integrates differential-drive motion, and the sensors read the world. Tests
  check whether a closed-loop behavior reaches its goal (``lap_progress``,
  ``line_error``, the pose).

In both modes time advances only on ``step()``, by exactly ``dt``, so tests are
deterministic. FakeSim follows Sim's rules: the creating thread owns it, failures
raise the same errors, ``step()`` needs stepping mode, and calls after the sim is
stopped raise SimStoppedError.

Fidelity limit (kinematic mode): no slip, inertia, or motor lag. It checks logic
(does the robot follow the line, stop, wait for the sensed condition), not gain
tuning. Tune gains in CoppeliaSim.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Sequence

import numpy as np

from ._owner import OwnerThread
from .errors import SimError, SimStoppedError
from .robot import Geometry, Handles, RobotSpec, load_robot_yaml
from .world import LineWorld

_BASE, _LEFT, _RIGHT, _SENSOR0, _PROX = 1, 2, 3, 100, 200


class FakeSim:
    """A stand-in for :class:`simio.Sim` that needs no simulator."""

    def __init__(
        self,
        robot_yaml: str,
        *,
        world: LineWorld | None = None,
        start_pose: Sequence[float] | None = None,
        dt: float = 0.05,
        noise_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._owner = OwnerThread()
        self._spec: RobotSpec = load_robot_yaml(robot_yaml)
        if dt <= 0:
            raise SimError("dt must be positive")
        self._dt = float(dt)
        self._world = world
        self._geom: Geometry | None = self._spec.geometry
        if world is not None:
            self._geom = self._spec.require_geometry("FakeSim's kinematic mode")
            if self._spec.proximity is not None and self._geom.proximity is None:
                raise SimError(f"{self._spec.path}: geometry.proximity is required for the proximity sensor")
        if start_pose is None:
            start_pose = world.start_pose if world is not None else (0.0, 0.0, 0.0)
        if len(start_pose) != 3:
            raise SimError("start_pose must be (x, y, theta)")
        self._start_pose: tuple[float, float, float] = (float(start_pose[0]), float(start_pose[1]), float(start_pose[2]))
        self._noise_std = float(noise_std)
        self._seed = seed
        self._rng = random.Random(seed)

        n = len(self._spec.line_sensors)
        self._handles = Handles(
            base=_BASE,
            left=_LEFT,
            right=_RIGHT,
            line_sensors=tuple(_SENSOR0 + i for i in range(n)),
            proximity=_PROX if self._spec.proximity else None,
            yaml_path=self._spec.path,
        )
        self._scripted_line: tuple[float, ...] = (1.0,) * n
        self._scripted_prox: tuple[bool, float] = (False, math.inf)

        self._closed = False
        self._running = False
        self._stepping = False
        self._reset_state()

    # ------------------------------------------------------------------ state

    def _reset_state(self) -> None:
        self._time = 0.0
        self._pose = self._start_pose
        self._wheels = (0.0, 0.0)  # (left, right) rad/s
        self._history: list[tuple[float, float, float]] = []
        self._progress = 0.0
        self._last_s: float | None = None
        self._line_error = 0.0
        self._max_line_error = 0.0
        self._rng = random.Random(self._seed)
        self._cache_step: int | None = None
        self._cache_line: tuple[float, ...] = ()
        self._steps = 0

    def _guard(self, what: str) -> None:
        self._owner.check(what)
        if self._closed:
            raise SimError(f"{what}: this FakeSim was closed")

    def _ensure_running(self, what: str) -> None:
        if not self._running:
            raise SimStoppedError(
                f"{what}: the simulation is not running (it was stopped, or never started). "
                "Call start() to run it again."
            )

    def _check_wheel(self, handle: int) -> None:
        if handle not in (_LEFT, _RIGHT):
            raise SimError(f"handle {handle} is not a wheel joint of this robot; use the handles from handles_from_yaml")

    # ---------------------------------------------------------------- handles

    def handles_from_yaml(self, path: str) -> Handles:
        """Handles for the robot this FakeSim was built from.

        ``path`` must describe the same robot (same sensor count and proximity
        sensor), as it would have to match the scene in CoppeliaSim.
        """
        self._guard("handles_from_yaml")
        spec = load_robot_yaml(path)
        if len(spec.line_sensors) != len(self._spec.line_sensors) or (spec.proximity is None) != (
            self._spec.proximity is None
        ):
            raise SimError(
                f"{spec.path} describes a robot with {len(spec.line_sensors)} line sensors"
                f"{' and a proximity sensor' if spec.proximity else ''}, but this FakeSim was built from "
                f"{self._spec.path} ({len(self._spec.line_sensors)} line sensors"
                f"{' and a proximity sensor' if self._spec.proximity else ''})"
            )
        return dataclasses.replace(self._handles, yaml_path=spec.path)

    # ------------------------------------------------------------- lifecycle

    def start(self, stepping: bool = False) -> None:
        """Start (or restart) the simulation; returns after its first step, like Sim."""
        self._guard("start")
        self._reset_state()
        self._stepping = stepping
        self._running = True
        self._advance()  # the first step, so time == dt after start(), as with Sim

    def is_running(self) -> bool:
        self._guard("is_running")
        return self._running

    def step(self) -> None:
        """Advance one step of ``dt`` (stepping mode only)."""
        self._guard("step")
        if not self._stepping:
            raise SimError("step() needs stepping mode: call start(stepping=True)")
        self._ensure_running("step")
        self._advance()

    def stop(self) -> None:
        """Zero the wheels, then stop."""
        self._guard("stop")
        if self._running:
            self._record_wheels(0.0, 0.0)
        self._running = False

    def close(self) -> None:
        """Zero the wheels if running, then disconnect. Later calls raise SimError."""
        self._owner.check("close")
        if self._closed:
            return
        if self._running:
            self._record_wheels(0.0, 0.0)
        self._closed = True

    def stop_from_gui(self) -> None:
        """Simulate someone pressing Stop in CoppeliaSim: the next call raises SimStoppedError.

        Unlike :meth:`stop`, this doesn't zero the wheels first (the code under
        test never got the chance). Test-only; any thread may call it.
        """
        self._running = False

    def __enter__(self) -> "FakeSim":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self.stop()
        except SimError:
            pass
        finally:
            self.close()

    # -------------------------------------------------------- scripted inputs

    def set_line_sensors(self, values: Sequence[float]) -> None:
        """Scripted mode: the readings ``read_line_sensors`` returns until changed."""
        if self._world is not None:
            raise SimError("set_line_sensors is for scripted mode; in kinematic mode readings come from the world")
        vals = tuple(float(v) for v in values)
        if len(vals) != len(self._handles.line_sensors):
            raise SimError(f"expected {len(self._handles.line_sensors)} readings (one per line sensor), got {len(vals)}")
        if any(not 0.0 <= v <= 1.0 for v in vals):
            raise SimError(f"line sensor readings must be in [0, 1], got {vals}")
        self._scripted_line = vals

    def set_proximity(self, detected: bool, distance: float = math.inf) -> None:
        """Scripted mode: the reading ``read_proximity`` returns until changed."""
        if self._world is not None:
            raise SimError("set_proximity is for scripted mode; in kinematic mode use world.add_obstacle")
        if self._handles.proximity is None:
            raise SimError(f"this robot has no proximity sensor in {self._spec.path}")
        self._scripted_prox = (True, float(distance)) if detected else (False, math.inf)

    # ------------------------------------------------------------ sense, act

    def read_line_sensors(self, handles: tuple[int, ...]) -> tuple[float, ...]:
        self._guard("read_line_sensors")
        self._ensure_running("read_line_sensors")
        index = []
        for h in handles:
            i = h - _SENSOR0
            if not 0 <= i < len(self._handles.line_sensors):
                raise SimError(f"handle {h} is not a line sensor of this robot; use the handles from handles_from_yaml")
            index.append(i)
        values = self._current_line_values()
        return tuple(values[i] for i in index)

    def read_proximity(self, handle: int | None) -> tuple[bool, float]:
        self._guard("read_proximity")
        if handle is None or self._handles.proximity is None:
            raise SimError(f"this robot has no proximity sensor in {self._spec.path}")
        if handle != _PROX:
            raise SimError(f"handle {handle} is not the proximity sensor; use the handles from handles_from_yaml")
        self._ensure_running("read_proximity")
        if self._world is None:
            return self._scripted_prox
        g = self._geom
        assert g is not None and g.proximity is not None
        x, y, th = self._pose
        p = g.proximity
        sx = x + p.x * math.cos(th) - p.y * math.sin(th)
        sy = y + p.x * math.sin(th) + p.y * math.cos(th)
        d = self._world.ray_cast(sx, sy, th + p.heading, p.max_range)
        return (True, d) if math.isfinite(d) else (False, math.inf)

    def set_wheel_speeds(self, left: int, right: int, left_rad_s: float, right_rad_s: float) -> None:
        self._guard("set_wheel_speeds")
        self._check_wheel(left)
        self._check_wheel(right)
        self._ensure_running("set_wheel_speeds")
        # Map by handle, as the real sim does: swapping the handles swaps the wheels.
        speeds = {left: float(left_rad_s), right: float(right_rad_s)}
        self._record_wheels(speeds.get(_LEFT, self._wheels[0]), speeds.get(_RIGHT, self._wheels[1]))

    def get_pose2d(self, handle: int) -> tuple[float, float, float]:
        """Ground-truth pose of the base, as Sim reports it (the base's x-axis heading)."""
        self._guard("get_pose2d")
        self._ensure_running("get_pose2d")
        if handle != _BASE:
            raise SimError(f"FakeSim tracks only the robot base (handle {_BASE}), not handle {handle}")
        x, y, th = self._pose
        offset = self._geom.base_yaw_offset if self._geom else 0.0
        return x, y, _wrap(th - offset)

    def time(self) -> float:
        self._guard("time")
        return self._time

    # --------------------------------------------------------- test helpers

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def handles(self) -> Handles:
        return self._handles

    @property
    def pose(self) -> tuple[float, float, float]:
        """Robot pose ``(x, y, heading)``, heading = the robot's forward direction."""
        return self._pose

    @property
    def last_wheel_speeds(self) -> tuple[float, float]:
        """``(left, right)`` rad/s most recently set."""
        return self._wheels

    @property
    def wheel_history(self) -> tuple[tuple[float, float, float], ...]:
        """``(sim_time, left, right)`` for every wheel command, including stop/close zeroing."""
        return tuple(self._history)

    @property
    def lap_progress(self) -> float:
        """Kinematic mode: distance traveled along the main line, as a fraction of its length.

        1.0 is one full lap of a closed track, in either direction. Only progress
        made while the sensors are near the line counts, so driving off and around doesn't.
        """
        return abs(self._progress)

    @property
    def line_error(self) -> float:
        """Kinematic mode: current distance (m) from the sensor array's center to the line."""
        return self._line_error

    @property
    def max_line_error(self) -> float:
        """Kinematic mode: largest ``line_error`` since start()."""
        return self._max_line_error

    @property
    def world(self) -> LineWorld | None:
        return self._world

    # -------------------------------------------------------------- internal

    def _record_wheels(self, left: float, right: float) -> None:
        self._wheels = (left, right)
        self._history.append((self._time, left, right))

    def _sensor_centers(self) -> np.ndarray:
        g = self._geom
        assert g is not None
        x, y, th = self._pose
        c, s = math.cos(th), math.sin(th)
        offs = np.array(g.line_sensors)
        return np.column_stack([x + offs[:, 0] * c - offs[:, 1] * s, y + offs[:, 0] * s + offs[:, 1] * c])

    def _current_line_values(self) -> tuple[float, ...]:
        if self._world is None:
            values = self._scripted_line
        elif self._cache_step == self._steps:
            values = self._cache_line
        else:
            g = self._geom
            assert g is not None
            raw = self._world.sensor_intensities(self._sensor_centers(), g.line_sensor_radius)
            values = tuple(float(v) for v in raw)
            self._cache_step, self._cache_line = self._steps, values
        if self._noise_std > 0:
            values = tuple(min(1.0, max(0.0, v + self._rng.gauss(0.0, self._noise_std))) for v in values)
        return values

    def _advance(self) -> None:
        if self._world is not None:
            g = self._geom
            assert g is not None
            wl, wr = self._wheels
            v = g.wheel_radius * (wl + wr) / 2
            w = g.wheel_radius * (wr - wl) / g.track_width
            x, y, th = self._pose
            dt = self._dt
            if abs(w) < 1e-9:
                x += v * dt * math.cos(th)
                y += v * dt * math.sin(th)
            else:
                x += v / w * (math.sin(th + w * dt) - math.sin(th))
                y -= v / w * (math.cos(th + w * dt) - math.cos(th))
            th = _wrap(th + w * dt)
            self._pose = (x, y, th)
            self._update_progress()
        self._time += self._dt
        self._steps += 1

    def _update_progress(self) -> None:
        world = self._world
        assert world is not None
        cx, cy = self._sensor_centers().mean(axis=0)
        s, dist = world.project_on_main(float(cx), float(cy))
        self._line_error = dist
        self._max_line_error = max(self._max_line_error, dist)
        near = dist <= max(0.05, 2 * world.line_width)
        if self._last_s is not None and near:
            ds = s - self._last_s
            if world.main_closed:
                half = world.main_length / 2
                if ds > half:
                    ds -= world.main_length
                elif ds < -half:
                    ds += world.main_length
            # Ignore jumps bigger than any plausible single step (projection switching branches).
            if abs(ds) < 0.25:
                self._progress += ds / world.main_length
        self._last_s = s if near else None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
