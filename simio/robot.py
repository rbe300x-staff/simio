"""Robot description: the YAML file format shared by simio and robot_parser.py.

The format is the 2026 one, plus an optional ``geometry:`` section::

    name: LineTracer
    motors:
      left: DynamicLeftJoint
      right: DynamicRightJoint
    sensors:            # line sensors; sorted by key, so order is left → right
      0: LeftSensor
      1: MiddleSensor
      2: RightSensor
    proximity_sensors:  # optional
      0: proximitySensor
    geometry:           # optional; required by FakeSim's kinematic mode and sim_bridge
      wheel_radius: 0.027          # m
      track_width: 0.119           # m, distance between the wheel contact points
      base_yaw_offset: 0.0         # rad, robot forward direction relative to the base object's x-axis
      line_sensors:                # (x, y) in the robot frame: x forward, y left; same order as sensors
        - [0.06, 0.02]
        - [0.06, 0.0]
        - [0.06, -0.02]
      line_sensor_radius: 0.005    # m, radius of each sensor's footprint on the floor
      proximity:                   # only if proximity_sensors is present
        x: 0.07
        y: 0.0
        heading: 0.0               # rad, relative to robot forward
        max_range: 0.3             # m

Handles are scene object handles (ints). In FakeSim they are made-up ints.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import SimError


@dataclass(frozen=True)
class Handles:
    """Scene handles for one robot, as returned by ``handles_from_yaml``."""

    base: int
    left: int
    right: int
    line_sensors: tuple[int, ...]
    proximity: int | None
    yaml_path: str = ""


@dataclass(frozen=True)
class ProximityGeometry:
    x: float
    y: float
    heading: float
    max_range: float


@dataclass(frozen=True)
class Geometry:
    wheel_radius: float
    track_width: float
    line_sensors: tuple[tuple[float, float], ...]
    line_sensor_radius: float
    proximity: ProximityGeometry | None
    base_yaw_offset: float = 0.0


@dataclass(frozen=True)
class RobotSpec:
    """Everything simio reads from a robot YAML file."""

    path: str
    name: str
    left_motor: str
    right_motor: str
    line_sensors: tuple[str, ...]
    proximity: str | None
    geometry: Geometry | None

    # Scene paths, in the 2026 convention: "/<robot name>/<object name>".
    @property
    def base_path(self) -> str:
        return f"/{self.name}"

    def object_path(self, object_name: str) -> str:
        return f"/{self.name}/{object_name}"

    def require_geometry(self, purpose: str) -> Geometry:
        if self.geometry is None:
            raise SimError(
                f"{self.path} has no 'geometry:' section, which {purpose} needs. "
                "Run tools/measure_robot.py with the scene open to generate it."
            )
        return self.geometry


def _require(d: dict[str, Any], key: str, where: str, path: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise SimError(f"{path}: missing '{key}' in {where}")
    return d[key]


def _float(value: Any, field: str, path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SimError(f"{path}: geometry field '{field}' must be a number, got {value!r}") from None


def _parse_geometry(raw: Any, n_sensors: int, has_proximity: bool, path: str) -> Geometry | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SimError(f"{path}: 'geometry' must be a mapping")
    sensors_raw = _require(raw, "line_sensors", "geometry", path)
    if not isinstance(sensors_raw, list) or len(sensors_raw) != n_sensors:
        raise SimError(
            f"{path}: geometry.line_sensors must list {n_sensors} [x, y] offsets "
            f"(one per entry in 'sensors'), got {sensors_raw!r}"
        )
    offsets = []
    for i, xy in enumerate(sensors_raw):
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            raise SimError(f"{path}: geometry.line_sensors[{i}] must be [x, y], got {xy!r}")
        offsets.append((_float(xy[0], f"line_sensors[{i}]", path), _float(xy[1], f"line_sensors[{i}]", path)))
    prox = None
    if has_proximity:
        p = _require(raw, "proximity", "geometry (the robot has a proximity sensor)", path)
        prox = ProximityGeometry(
            x=_float(_require(p, "x", "geometry.proximity", path), "proximity.x", path),
            y=_float(_require(p, "y", "geometry.proximity", path), "proximity.y", path),
            heading=_float(p.get("heading", 0.0), "proximity.heading", path),
            max_range=_float(_require(p, "max_range", "geometry.proximity", path), "proximity.max_range", path),
        )
    geom = Geometry(
        wheel_radius=_float(_require(raw, "wheel_radius", "geometry", path), "wheel_radius", path),
        track_width=_float(_require(raw, "track_width", "geometry", path), "track_width", path),
        line_sensors=tuple(offsets),
        line_sensor_radius=_float(_require(raw, "line_sensor_radius", "geometry", path), "line_sensor_radius", path),
        proximity=prox,
        base_yaw_offset=_float(raw.get("base_yaw_offset", 0.0), "base_yaw_offset", path),
    )
    if geom.wheel_radius <= 0 or geom.track_width <= 0 or geom.line_sensor_radius <= 0:
        raise SimError(f"{path}: wheel_radius, track_width, and line_sensor_radius must be positive")
    return geom


def _did_you_mean(path: str) -> str:
    """' (did you mean config/x.yaml?)' if a similarly named YAML file sits in the same folder."""
    p = Path(path)
    folder = p.parent if str(p.parent) else Path(".")
    try:
        candidates = [f.name for f in folder.iterdir() if f.suffix in (".yaml", ".yml")]
    except OSError:
        return ""
    match = difflib.get_close_matches(p.name, candidates, n=1, cutoff=0.6)
    return f" (did you mean {folder / match[0]}?)" if match else ""


def load_robot_yaml(path: str | Path) -> RobotSpec:
    """Read a robot YAML file. Raises SimError naming the file and the problem."""
    p = str(path)
    try:
        with open(p) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise SimError(f"robot YAML file not found: {p}{_did_you_mean(p)}") from None
    except yaml.YAMLError as e:
        raise SimError(f"{p} is not valid YAML: {e}") from None
    if not isinstance(data, dict):
        raise SimError(f"{p}: expected a mapping at the top level")

    name = str(_require(data, "name", "the top level", p))
    motors = _require(data, "motors", "the top level", p)
    left = str(_require(motors, "left", "motors", p))
    right = str(_require(motors, "right", "motors", p))
    sensors = _require(data, "sensors", "the top level", p)
    if not isinstance(sensors, dict) or not sensors:
        raise SimError(f"{p}: 'sensors' must be a non-empty mapping like {{0: LeftSensor, 1: RightSensor}}")
    line_sensors = tuple(str(sensors[k]) for k in sorted(sensors))
    prox_map = data.get("proximity_sensors")
    proximity = None
    if prox_map:
        if not isinstance(prox_map, dict):
            raise SimError(f"{p}: 'proximity_sensors' must be a mapping like {{0: proximitySensor}}")
        proximity = str(prox_map[sorted(prox_map)[0]])
    geometry = _parse_geometry(data.get("geometry"), len(line_sensors), proximity is not None, p)
    return RobotSpec(
        path=p,
        name=name,
        left_motor=left,
        right_motor=right,
        line_sensors=line_sensors,
        proximity=proximity,
        geometry=geometry,
    )
