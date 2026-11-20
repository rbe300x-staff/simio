"""Run the staff tools against a stand-in CoppeliaSim built on FakeSim's kinematic world.

This checks the tools' own math (robot frame, geometry, tracing). It can't check
CoppeliaSim's real API; tools/check_sim.py does that on a live simulator.
"""

import math
import os
import sys

import pytest
import yaml

from simio import FakeSim, LineWorld, Sim, load_robot_yaml

TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", "tools")
sys.path.insert(0, os.path.abspath(TOOLS))

SENSOR_HEIGHT = 0.02


class KinematicRemote:
    """Answers CoppeliaSim API calls from a FakeSim in kinematic mode."""

    simulation_stopped = 0
    simulation_advancing_running = 17
    handle_world = -1
    intparam_program_version = 1
    visionintparam_perspective_operation = 2
    visionfloatparam_ortho_size = 3

    def __init__(self, robot_yaml, world, yaw_offset=0.0):
        self.fake = FakeSim(robot_yaml, world=world)
        self.spec = load_robot_yaml(robot_yaml)
        self.g = self.spec.geometry
        self.h = self.fake.handles
        self.yaw_offset = yaw_offset
        names = {
            self.spec.base_path: self.h.base,
            self.spec.object_path(self.spec.left_motor): self.h.left,
            self.spec.object_path(self.spec.right_motor): self.h.right,
        }
        for n, s in zip(self.spec.line_sensors, self.h.line_sensors):
            names[self.spec.object_path(n)] = s
        if self.spec.proximity:
            names[self.spec.object_path(self.spec.proximity)] = self.h.proximity
        self.names = names
        self.state = 0
        self.stepping = False

    # lifecycle
    def getInt32Param(self, p):
        return 40600

    def getObject(self, path):
        return self.names[path]

    def getSimulationState(self):
        return self.state

    def setStepping(self, on):
        self.stepping = on

    def startSimulation(self):
        self.state = 17
        self.fake.start(stepping=True)

    def stopSimulation(self):
        if self.state:
            self.fake.stop()
        self.state = 0

    def step(self):
        self.fake.step()

    def getSimulationTime(self):
        return self.fake.time() if self.state else 0.0

    def getSimulationTimeStep(self):
        return self.fake.dt

    # sense / act
    def readVisionSensor(self, handle):
        if not self.state:
            return (-1, [], [])
        (v,) = self.fake.read_line_sensors((handle,))
        packet = [0.0] * 15
        packet[10] = v
        return (1, packet, [])

    def readProximitySensor(self, handle):
        detected, d = self.fake.read_proximity(handle)
        return (1, d, [0, 0, d], 5, [0, 0, 1]) if detected else (0, 0.0, [0, 0, 0], -1, [0, 0, 0])

    def setJointTargetVelocity(self, handle, v):
        left, right = self.fake.last_wheel_speeds
        if handle == self.h.left:
            left = v
        else:
            right = v
        self.fake.set_wheel_speeds(self.h.left, self.h.right, left, right)

    # geometry
    def _robot_point(self, rx, ry):
        x, y, th = self.fake.pose if self.state else self.fake._start_pose
        return x + rx * math.cos(th) - ry * math.sin(th), y + rx * math.sin(th) + ry * math.cos(th)

    def getObjectPosition(self, handle, rel):
        g = self.g
        if handle == self.h.left:
            return [*self._robot_point(0.0, g.track_width / 2), g.wheel_radius]
        if handle == self.h.right:
            return [*self._robot_point(0.0, -g.track_width / 2), g.wheel_radius]
        if handle in self.h.line_sensors:
            rx, ry = g.line_sensors[handle - self.h.line_sensors[0]]
            return [*self._robot_point(rx, ry), SENSOR_HEIGHT]
        if handle == self.h.proximity:
            return [*self._robot_point(g.proximity.x, g.proximity.y), 0.03]
        return [*self._robot_point(0.0, 0.0), 0.03]

    def getObjectMatrix(self, handle, rel):
        x, y, z = self.getObjectPosition(handle, rel)
        th = (self.fake.pose if self.state else self.fake._start_pose)[2]
        if handle == self.h.proximity:  # detects along its z-axis, pointing forward
            a = th + self.g.proximity.heading
            return [0, 0, math.cos(a), x, 0, 0, math.sin(a), y, -1, 0, 0, z]
        a = th - self.yaw_offset  # the base object's x-axis
        return [math.cos(a), -math.sin(a), 0, x, math.sin(a), math.cos(a), 0, y, 0, 0, 1, z]

    def getObjectInt32Param(self, handle, p):
        return 0  # orthographic

    def getObjectFloatParam(self, handle, p):
        return 2 * self.g.line_sensor_radius

    def getObjectChild(self, handle, i):
        return -1


@pytest.fixture
def remote_factory(monkeypatch):
    def make(robot_yaml, world, yaw_offset=0.0):
        remote = KinematicRemote(robot_yaml, world, yaw_offset)
        monkeypatch.setattr(Sim, "connect", classmethod(lambda cls, *a, **k: Sim(object(), remote)))
        return remote

    return make


def test_measure_robot_recovers_geometry(remote_factory, yaml5, stadium, tmp_path, monkeypatch, capsys):
    import measure_robot

    remote_factory(yaml5, stadium, yaw_offset=math.pi / 2)
    out = tmp_path / "measured.yaml"
    monkeypatch.setattr(sys, "argv", ["measure_robot.py", "--yaml", yaml5, "--out", str(out)])
    measure_robot.main()
    g = yaml.safe_load(out.read_text())["geometry"]
    truth = load_robot_yaml(yaml5).geometry
    assert g["wheel_radius"] == pytest.approx(truth.wheel_radius, abs=1e-4)
    assert g["track_width"] == pytest.approx(truth.track_width, abs=1e-4)
    assert g["base_yaw_offset"] == pytest.approx(math.pi / 2, abs=1e-4)
    for got, want in zip(g["line_sensors"], truth.line_sensors):
        assert got == pytest.approx(list(want), abs=1e-4)
    assert g["line_sensor_radius"] == pytest.approx(truth.line_sensor_radius, abs=1e-4)
    assert g["proximity"]["x"] == pytest.approx(truth.proximity.x, abs=1e-4)
    assert g["proximity"]["heading"] == pytest.approx(0.0, abs=1e-4)
    assert g["proximity"]["max_range"] == "MEASURE_BY_HAND"


def test_trace_track_recovers_the_line(remote_factory, yaml5, tmp_path, monkeypatch):
    import trace_track

    truth = LineWorld.stadium(straight=1.0, radius=0.4, line_width=0.02, floor_intensity=0.9, line_intensity=0.1)
    remote_factory(yaml5, truth)
    out = tmp_path / "traced.yaml"
    monkeypatch.setattr(sys, "argv", ["trace_track.py", "--yaml", yaml5, "--out", str(out), "--speed", "0.1"])
    trace_track.main()
    traced = LineWorld.from_yaml(out)
    assert traced.main_closed
    assert traced.main_length == pytest.approx(truth.main_length, rel=0.05)
    assert traced.floor_intensity == pytest.approx(0.9, abs=0.01)
    assert traced.line_intensity < 0.4
    # Every traced point lies close to the true line.
    import numpy as np

    pts = np.array(yaml.safe_load(out.read_text())["lines"][0]["points"])
    assert truth.distance_to_line(pts).max() < 0.01


@pytest.mark.parametrize("robot", ["yaml3_geom", "yaml5"])
@pytest.mark.parametrize("radius", [0.4, 0.2])
def test_trace_track_on_scene_scale(robot, radius, remote_factory, request, tmp_path, monkeypatch):
    """Readings on lineFollower.ttt's scale (floor ~0.83, line ~0.38), not 1.0 / 0.0.

    Regression: the follower used darkness = 1 - reading, so the bare floor counted as
    0.17 dark on every sensor. It steered too weakly for curves and never noticed it had
    lost the line.
    """
    import numpy as np

    import trace_track

    robot_yaml = request.getfixturevalue(robot)
    truth = LineWorld.stadium(straight=1.0, radius=radius, line_width=0.02, floor_intensity=0.83, line_intensity=0.38)
    remote = remote_factory(robot_yaml, truth)
    out = tmp_path / "traced.yaml"
    monkeypatch.setattr(sys, "argv", ["trace_track.py", "--yaml", robot_yaml, "--out", str(out)])
    trace_track.main()
    traced = LineWorld.from_yaml(out)
    assert traced.main_closed
    assert traced.main_length == pytest.approx(truth.main_length, rel=0.05)
    assert remote.fake.max_line_error < 0.03
    pts = np.array(yaml.safe_load(out.read_text())["lines"][0]["points"])
    assert truth.distance_to_line(pts).max() < 0.015


def test_trace_track_stops_when_line_is_lost(remote_factory, yaml5, tmp_path, monkeypatch, capsys):
    """An open line: the tracer must stop near its end, not drive off recording floor."""
    import trace_track

    truth = LineWorld.from_points([(0, 0), (1.0, 0)], closed=False, floor_intensity=0.83, line_intensity=0.38)
    remote = remote_factory(yaml5, truth)
    out = tmp_path / "traced.yaml"
    monkeypatch.setattr(sys, "argv", ["trace_track.py", "--yaml", yaml5, "--out", str(out)])
    trace_track.main()
    assert "line lost" in capsys.readouterr().out
    assert not LineWorld.from_yaml(out).main_closed
    assert remote.fake.pose[0] < 1.2  # stopped near the end of the line


def test_trace_track_refuses_to_start_off_the_line(remote_factory, yaml5, tmp_path, monkeypatch):
    import trace_track

    truth = LineWorld.from_points([(0, 1.0), (1.0, 1.0)], closed=False, floor_intensity=0.83, line_intensity=0.38)
    remote_factory(yaml5, truth)
    monkeypatch.setattr(sys, "argv", ["trace_track.py", "--yaml", yaml5, "--out", str(tmp_path / "t.yaml")])
    with pytest.raises(SystemExit, match="no line under the array"):
        trace_track.main()
