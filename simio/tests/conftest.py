"""Shared fixtures: robot YAML files with made-up (test-only) geometry."""

import math
import textwrap

import pytest

from simio import LineWorld

GEOMETRY_5 = """
geometry:
  wheel_radius: 0.02
  track_width: 0.10
  base_yaw_offset: 0.0
  line_sensors: [[0.06, 0.03], [0.06, 0.015], [0.06, 0.0], [0.06, -0.015], [0.06, -0.03]]
  line_sensor_radius: 0.006
  proximity: {x: 0.07, y: 0.0, heading: 0.0, max_range: 0.3}
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return str(p)


@pytest.fixture
def yaml3(tmp_path):
    """The 2026 lineTracer.yaml: 3 sensors, no proximity, no geometry."""
    return _write(tmp_path, "lineTracer.yaml", """
        motors:
          left: DynamicLeftJoint
          right: DynamicRightJoint
        name: LineTracer
        sensors:
          0: LeftSensor
          1: MiddleSensor
          2: RightSensor
        """)


@pytest.fixture
def yaml5(tmp_path):
    """lineTracer_prox.yaml with test geometry: 5 sensors and a proximity sensor."""
    return _write(tmp_path, "lineTracer_prox.yaml", textwrap.dedent("""
        motors:
          left: DynamicLeftJoint
          right: DynamicRightJoint
        name: LineTracer
        sensors:
          0: LeftMostSensor
          1: LeftSensor
          2: MiddleSensor
          3: RightSensor
          4: RightMostSensor
        proximity_sensors:
          0: proximitySensor
        """) + GEOMETRY_5)


@pytest.fixture
def yaml3_geom(yaml3):
    """lineTracer.yaml plus test geometry shaped like the lap robot: wheel radius ~0.027 m
    (from the live drive check) and sensor footprints wider than the line, so no sensor ever
    reads the full line darkness (the live scene's darkest mean was ~0.58)."""
    with open(yaml3, "a") as f:
        f.write(textwrap.dedent("""
            geometry:
              wheel_radius: 0.027
              track_width: 0.11
              base_yaw_offset: 0.0
              line_sensors: [[0.07, 0.02], [0.07, 0.0], [0.07, -0.02]]
              line_sensor_radius: 0.015
            """))
    return yaml3


@pytest.fixture
def stadium():
    return LineWorld.stadium(straight=1.0, radius=0.4, line_width=0.02)


def follow_line(sim, h, *, steps, base_speed=0.15, gain=25.0, stop_distance=None, track_width=0.10, wheel_radius=0.02,
                floor=1.0, line=0.0):
    """A minimal closed-loop line follower used by the tests (and a model for students).

    Steers toward the darkness-weighted centroid of the sensors. If ``stop_distance``
    is given, it stops while the proximity sensor sees something closer than that,
    and resumes when it clears. Returns the list of (time, stopped) per step.

    ``floor`` and ``line`` are what the sensors read over bare floor and over the line.
    Real scenes don't read 1.0 / 0.0 (lineFollower.ttt reads about 0.83 / 0.38), so
    darkness is normalized between them; in the lab these come from the robot's YAML.
    """
    ys = [0.03, 0.015, 0.0, -0.015, -0.03] if len(h.line_sensors) == 5 else [0.02, 0.0, -0.02]
    last_side = 0.0
    log = []
    for _ in range(steps):
        values = sim.read_line_sensors(h.line_sensors)
        dark = [min(1.0, max(0.0, (floor - v) / (floor - line))) for v in values]
        total = sum(dark)
        if max(dark) > 0.1:
            y_c = sum(d * y for d, y in zip(dark, ys)) / total
            last_side = math.copysign(1.0, y_c) if abs(y_c) > 1e-6 else last_side
            omega = gain * y_c
            v = base_speed
        else:  # line lost: turn toward the side it was last seen on
            omega, v = 1.5 * (last_side or 1.0), 0.0
        stopped = False
        if stop_distance is not None:
            detected, dist = sim.read_proximity(h.proximity)
            if detected and dist < stop_distance:
                v, omega, stopped = 0.0, 0.0, True
        wl = (v - omega * track_width / 2) / wheel_radius
        wr = (v + omega * track_width / 2) / wheel_radius
        sim.set_wheel_speeds(h.left, h.right, wl, wr)
        sim.step()
        log.append((sim.time(), stopped))
    return log
