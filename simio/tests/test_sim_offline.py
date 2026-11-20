"""Sim against a stand-in for the CoppeliaSim remote API (no simulator needed).

These check simio's own logic: path building, error wrapping, start/stop, the
owner-thread rule, and shutdown. Whether the real API behaves as the stand-in
assumes is checked by tools/check_sim.py against a live CoppeliaSim.
"""

import math
import sys
import threading
import types

import pytest

from simio import Sim, SimConnectionError, SimError, SimStoppedError, SimThreadError


class FakeRemoteSim:
    """Just enough of CoppeliaSim's 'sim' API object for Sim to run."""

    simulation_stopped = 0
    simulation_advancing_running = 17
    handle_world = -1
    intparam_program_version = 1

    def __init__(self, objects, version=40600):
        self.objects = dict(objects)  # path -> handle
        self.version = version
        self.state = self.simulation_stopped
        self.t = 0.0
        self.dt = 0.05
        self.stepping = None
        self.joint_velocity = {}
        self.vision = {}  # handle -> (result, packet)
        self.proximity = {}  # handle -> tuple
        self.calls = []

    def getInt32Param(self, p):
        return self.version

    def getObject(self, path):
        if path not in self.objects:
            raise Exception(f"object does not exist. ({path})")
        return self.objects[path]

    def getSimulationState(self):
        return self.state

    def setStepping(self, on):
        self.stepping = on

    def startSimulation(self):
        self.state = self.simulation_advancing_running
        self.t = 0.0 if self.stepping else self.dt  # free-running: the first step happens on its own

    def stopSimulation(self):
        self.state = self.simulation_stopped

    def step(self):
        self.t += self.dt

    def getSimulationTime(self):
        return self.t

    def getSimulationTimeStep(self):
        return self.dt

    def readVisionSensor(self, h):
        return self.vision.get(h, (0, [0.0] * 15, []))

    def readProximitySensor(self, h):
        return self.proximity.get(h, (0, 0.0, [0, 0, 0], -1, [0, 0, 0]))

    def setJointTargetVelocity(self, h, v):
        self.calls.append(("setJointTargetVelocity", h, v))
        self.joint_velocity[h] = v

    def getObjectMatrix(self, h, rel):
        c, s = math.cos(0.5), math.sin(0.5)
        return [c, -s, 0, 1.25, s, c, 0, -0.5, 0, 0, 1, 0.02]


OBJECTS = {
    "/LineTracer": 10,
    "/LineTracer/DynamicLeftJoint": 11,
    "/LineTracer/DynamicRightJoint": 12,
    "/LineTracer/LeftMostSensor": 20,
    "/LineTracer/LeftSensor": 21,
    "/LineTracer/MiddleSensor": 22,
    "/LineTracer/RightSensor": 23,
    "/LineTracer/RightMostSensor": 24,
    "/LineTracer/proximitySensor": 30,
}


@pytest.fixture
def remote():
    return FakeRemoteSim(OBJECTS)


@pytest.fixture
def sim(remote):
    return Sim(client=object(), sim=remote, host="test", port=0)


def packet(mean):
    p = [0.0] * 15
    p[10] = mean
    return p


def test_handles_from_yaml_builds_2026_paths(sim, yaml5):
    h = sim.handles_from_yaml(yaml5)
    assert (h.base, h.left, h.right) == (10, 11, 12)
    assert h.line_sensors == (20, 21, 22, 23, 24)
    assert h.proximity == 30


def test_missing_object_names_path_and_yaml(remote, sim, yaml5):
    del remote.objects["/LineTracer/MiddleSensor"]
    with pytest.raises(SimError) as e:
        sim.handles_from_yaml(yaml5)
    assert "/LineTracer/MiddleSensor" in str(e.value) and "lineTracer_prox.yaml" in str(e.value)


def test_start_stepping_takes_first_step(remote, sim):
    sim.start(stepping=True)
    assert remote.stepping is True
    assert sim.time() == pytest.approx(0.05)


def test_start_free_running_waits_for_first_step(remote, sim):
    sim.start(stepping=False)
    assert remote.stepping is False
    assert sim.time() > 0


def test_start_restarts_a_running_sim(remote, sim):
    remote.state = remote.simulation_advancing_running  # someone pressed Play
    sim.start(stepping=True)
    assert remote.stepping is True and sim.is_running()


def test_step_requires_stepping(sim):
    sim.start(stepping=False)
    with pytest.raises(SimError, match="stepping"):
        sim.step()


def test_line_sensors_use_mean_intensity_and_clamp(remote, sim, yaml5):
    h = sim.handles_from_yaml(yaml5)
    sim.start(stepping=True)
    for handle, mean in zip(h.line_sensors, (0.9, 0.1, -0.2, 1.3, 0.5)):
        remote.vision[handle] = (1, packet(mean), [])
    assert sim.read_line_sensors(h.line_sensors) == (0.9, 0.1, 0.0, 1.0, 0.5)


def test_line_sensor_without_image_raises(remote, sim, yaml5):
    h = sim.handles_from_yaml(yaml5)
    sim.start(stepping=True)
    remote.vision[h.line_sensors[0]] = (-1, [], [])
    with pytest.raises(SimError, match="no image data"):
        sim.read_line_sensors(h.line_sensors)


def test_proximity(remote, sim, yaml5):
    h = sim.handles_from_yaml(yaml5)
    sim.start(stepping=True)
    assert sim.read_proximity(h.proximity) == (False, math.inf)
    remote.proximity[h.proximity] = (1, 0.12, [0, 0, 0.12], 99, [0, 0, 1])
    assert sim.read_proximity(h.proximity) == (True, 0.12)


def test_no_proximity_sensor_names_yaml(sim, yaml3, remote):
    h = sim.handles_from_yaml(yaml3)
    sim.start(stepping=True)
    with pytest.raises(SimError, match="lineTracer.yaml"):
        sim.read_proximity(h.proximity)


def test_pose_from_matrix(sim, yaml5):
    h = sim.handles_from_yaml(yaml5)
    sim.start(stepping=True)
    assert sim.get_pose2d(h.base) == pytest.approx((1.25, -0.5, 0.5))


def test_stopped_from_gui_raises(remote, sim, yaml5):
    h = sim.handles_from_yaml(yaml5)
    sim.start(stepping=True)
    remote.stopSimulation()  # pressed Stop in the GUI
    with pytest.raises(SimStoppedError):
        sim.read_line_sensors(h.line_sensors)
    with pytest.raises(SimStoppedError):
        sim.step()


def test_remote_errors_are_wrapped(remote, sim, yaml5):
    h = sim.handles_from_yaml(yaml5)
    sim.start(stepping=True)

    def boom(handle, v):
        raise Exception("joint is not dynamically enabled")

    remote.setJointTargetVelocity = boom
    with pytest.raises(SimError, match="joint is not dynamically enabled"):
        sim.set_wheel_speeds(h.left, h.right, 1.0, 1.0)


def test_with_block_zeroes_wheels_and_stops_on_ctrl_c(remote, yaml5):
    with pytest.raises(KeyboardInterrupt):
        with Sim(client=object(), sim=remote) as sim:
            h = sim.handles_from_yaml(yaml5)
            sim.start(stepping=True)
            sim.set_wheel_speeds(h.left, h.right, 3.0, 3.0)
            raise KeyboardInterrupt
    assert remote.joint_velocity == {11: 0.0, 12: 0.0}
    assert remote.state == remote.simulation_stopped


def test_close_zeroes_wheels_but_leaves_sim_running(remote, sim, yaml5):
    h = sim.handles_from_yaml(yaml5)
    sim.start(stepping=True)
    sim.set_wheel_speeds(h.left, h.right, 3.0, 3.0)
    sim.close()
    assert remote.joint_velocity == {11: 0.0, 12: 0.0}
    assert remote.state != remote.simulation_stopped
    with pytest.raises(SimError, match="closed"):
        sim.time()


def test_other_thread_rejected(sim):
    errors = []
    t = threading.Thread(target=lambda: errors.append(_try(sim.time)), name="worker")
    t.start()
    t.join()
    assert isinstance(errors[0], SimThreadError)


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return e
    return None


# ----------------------------------------------------------------- connect


def test_connect_without_client_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "coppeliasim_zmqremoteapi_client", None)
    with pytest.raises(SimConnectionError, match="pip install"):
        Sim.connect()


def _install_fake_client(monkeypatch, remote):
    mod = types.ModuleType("coppeliasim_zmqremoteapi_client")

    class RemoteAPIClient:
        def __init__(self, host, port):
            pass

        def require(self, name):
            return remote

    mod.RemoteAPIClient = RemoteAPIClient
    monkeypatch.setitem(sys.modules, "coppeliasim_zmqremoteapi_client", mod)


def test_connect_fails_fast_when_nothing_listens(monkeypatch, remote):
    _install_fake_client(monkeypatch, remote)
    with pytest.raises(SimConnectionError, match="Is CoppeliaSim open"):
        Sim.connect(port=1, timeout_sec=0.5)  # port 1: nothing listens there


def test_connect_rejects_old_version(monkeypatch):
    import socket

    remote = FakeRemoteSim(OBJECTS, version=40300)
    _install_fake_client(monkeypatch, remote)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    try:
        with pytest.raises(SimConnectionError, match="4.3.0 is too old"):
            Sim.connect(host="127.0.0.1", port=server.getsockname()[1])
    finally:
        server.close()


def test_connect_ok(monkeypatch, remote):
    import socket

    _install_fake_client(monkeypatch, remote)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    try:
        sim = Sim.connect(host="127.0.0.1", port=server.getsockname()[1])
        assert sim.version == (4, 6, 0)
    finally:
        server.close()
