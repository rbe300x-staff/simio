"""FakeSim in kinematic mode: the robot moves, and closed-loop behavior can be checked sim-free."""

import math

import pytest

from simio import FakeSim, LineWorld, SimError, load_robot_yaml

from .conftest import follow_line


def kinematic(yaml_path, world, **kw):
    sim = FakeSim(yaml_path, world=world, **kw)
    sim.start(stepping=True)
    return sim, sim.handles_from_yaml(yaml_path)


def test_needs_geometry(yaml3, stadium):
    with pytest.raises(SimError, match="geometry"):
        FakeSim(yaml3, world=stadium)


def test_scripted_setters_rejected(yaml5, stadium):
    sim, _ = kinematic(yaml5, stadium)
    with pytest.raises(SimError, match="scripted mode"):
        sim.set_line_sensors((1.0,) * 5)
    with pytest.raises(SimError, match="add_obstacle"):
        sim.set_proximity(True, 0.1)


def test_straight_drive_distance(yaml5, stadium):
    sim, h = kinematic(yaml5, stadium)
    sim.set_wheel_speeds(h.left, h.right, 5.0, 5.0)  # 5 rad/s × 0.02 m = 0.1 m/s
    for _ in range(20):  # 1.0 s
        sim.step()
    x, y, th = sim.get_pose2d(h.base)
    assert (x, y, th) == pytest.approx((0.1, 0.0, 0.0), abs=1e-9)


def test_spin_in_place(yaml5, stadium):
    sim, h = kinematic(yaml5, stadium)
    # ω = r (ωr − ωl) / b = 0.02 × 10 / 0.1 = 2 rad/s; 10 steps of 0.05 s → 1 rad
    sim.set_wheel_speeds(h.left, h.right, -5.0, 5.0)
    for _ in range(10):
        sim.step()
    x, y, th = sim.pose
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert th == pytest.approx(1.0)


def test_sensors_see_the_line_at_start(yaml5, stadium):
    sim, h = kinematic(yaml5, stadium)
    values = sim.read_line_sensors(h.line_sensors)
    assert values[2] == pytest.approx(0.0)  # the middle sensor sits on the line
    assert values[0] == pytest.approx(1.0) and values[4] == pytest.approx(1.0)


@pytest.mark.parametrize("dt", [0.05, 0.02])
def test_closed_loop_follower_completes_a_lap(yaml5, stadium, dt):
    """R5 in miniature: a sensed-condition follower gives the same outcome at different step sizes."""
    sim, h = kinematic(yaml5, stadium, dt=dt)
    follow_line(sim, h, steps=int(40 / dt))  # 40 s of sim time
    assert sim.lap_progress >= 1.0
    assert sim.max_line_error < 0.02


def test_follower_with_sensor_noise(yaml5, stadium):
    sim, h = kinematic(yaml5, stadium, noise_std=0.05, seed=3)
    follow_line(sim, h, steps=800)
    assert sim.lap_progress >= 1.0


def test_open_loop_timed_drive_fails(yaml5, stadium):
    """A behavior that ends on elapsed time leaves the line: the straight ends before the timer does."""
    sim, h = kinematic(yaml5, stadium)
    wheel = 0.15 / 0.02
    for _ in range(int(6.0 / sim.dt)):  # "drive straight for 6 s": 0.9 m, but the straight ends at 0.5 m
        sim.set_wheel_speeds(h.left, h.right, wheel, wheel)
        sim.step()
    assert sim.line_error > 0.05


def test_stops_for_obstacle_until_removed(yaml5, stadium):
    sim, h = kinematic(yaml5, stadium)
    oid = stadium.add_obstacle(0.4, 0.0, 0.03)
    log = follow_line(sim, h, steps=100, stop_distance=0.08)
    x_stopped = sim.pose[0]
    assert log[-1][1] is True  # still stopped after 5 s: it waits for the condition, not a timer
    assert x_stopped < 0.4 - 0.03 - 0.07 + 0.02  # stopped before reaching the obstacle
    stadium.remove_obstacle(oid)
    follow_line(sim, h, steps=40, stop_distance=0.08)
    assert sim.pose[0] > x_stopped + 0.1  # resumed once the obstacle was gone


def test_deterministic_with_seed(yaml5, stadium):
    runs = []
    for _ in range(2):
        sim, h = kinematic(yaml5, stadium, noise_std=0.05, seed=7)
        follow_line(sim, h, steps=200)
        runs.append(sim.pose)
    assert runs[0] == runs[1]


def test_base_yaw_offset_is_applied(tmp_path, yaml5, stadium):
    text = open(yaml5).read().replace("base_yaw_offset: 0.0", "base_yaw_offset: 1.5707963267948966")
    p = tmp_path / "offset.yaml"
    p.write_text(text)
    sim, h = kinematic(str(p), stadium)
    assert sim.pose[2] == pytest.approx(0.0)  # robot heading (forward)
    assert sim.get_pose2d(h.base)[2] == pytest.approx(-math.pi / 2)  # the base object's x-axis


def test_reference_follower_on_scene_scale(yaml5):
    """The reference follower laps a track whose readings match lineFollower.ttt (0.83 / 0.38)."""
    world = LineWorld.stadium(straight=1.0, radius=0.4, line_width=0.02, floor_intensity=0.83, line_intensity=0.38)
    sim = FakeSim(yaml5, world=world)
    h = sim.handles_from_yaml(yaml5)
    sim.start(stepping=True)
    follow_line(sim, h, steps=int(40 / sim.dt), floor=0.83, line=0.38)
    assert sim.lap_progress >= 1.0  # laps
    assert sim.max_line_error < 0.03


@pytest.mark.parametrize("robot,track", [
    ("lineTracer.measured.yaml", "lineFollower.yaml"),
    ("lineTracer_prox.measured.yaml", "lineFollower_prox.yaml"),
])
def test_reference_follower_laps_the_course_track(robot, track):
    """Traced course tracks with the measured robots: the template for Lab private tests.

    Files from the instructor session (2026-11-10): tools/measure_robot.py and tools/trace_track.py
    on CoppeliaSim 4.10.0.
    """
    robot, world = _course(robot, track)
    g = load_robot_yaml(robot).geometry
    for dt in (0.05, 0.02):
        sim, h = kinematic(robot, world, dt=dt)
        follow_line(sim, h, steps=int(60 / dt), base_speed=0.15, gain=25.0,
                    track_width=g.track_width, wheel_radius=g.wheel_radius,
                    floor=world.floor_intensity, line=world.line_intensity)
        assert sim.lap_progress >= 1.0  # laps
        assert sim.max_line_error < 0.05


def test_reference_follower_waits_for_obstacle_on_the_course_track():
    """Obstacle 1 m along the traced obstacle-scene track: stop until it's removed, then go on."""
    robot, world = _course("lineTracer_prox.measured.yaml", "lineFollower_prox.yaml")
    g = load_robot_yaml(robot).geometry
    run = dict(base_speed=0.15, gain=25.0, track_width=g.track_width, wheel_radius=g.wheel_radius,
               floor=world.floor_intensity, line=world.line_intensity, stop_distance=0.08)
    sim, h = kinematic(robot, world)
    follow_line(sim, h, steps=1, **run)
    # Put the obstacle on the line about 1 m ahead of the robot.
    target = 1.0 / world.main_length
    probe, ph = kinematic(robot, LineWorld.from_yaml(_course_path("tracks", "lineFollower_prox.yaml")))
    while probe.lap_progress < target:
        follow_line(probe, ph, steps=1, **run)
    x, y, th = probe.pose
    oid = world.add_obstacle(x + 0.13 * math.cos(th), y + 0.13 * math.sin(th), 0.03)
    log = follow_line(sim, h, steps=400, **run)
    assert log[-1][1] is True  # still waiting: the stop ends on a sensed condition, not a timer
    stopped_at = sim.lap_progress
    world.remove_obstacle(oid)
    follow_line(sim, h, steps=200, **run)
    assert (sim.lap_progress - stopped_at) * world.main_length > 1.0


def _course_path(*parts):
    import os

    return os.path.join(os.path.dirname(__file__), "..", "..", *parts)


def _course(robot, track):
    return _course_path("config", robot), LineWorld.from_yaml(_course_path("tracks", track))
