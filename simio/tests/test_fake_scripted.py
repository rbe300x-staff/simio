"""FakeSim in scripted mode: the behaviors sim_bridge's public tests rely on."""

import math
import threading

import pytest

from simio import FakeSim, SimError, SimInterface, SimStoppedError, SimThreadError


def started(yaml_path, stepping=True):
    sim = FakeSim(yaml_path)
    sim.start(stepping=stepping)
    return sim, sim.handles_from_yaml(yaml_path)


def test_satisfies_sim_interface(yaml3):
    assert isinstance(FakeSim(yaml3), SimInterface)


def test_start_completes_first_step(yaml3):
    sim, _ = started(yaml3)
    assert sim.time() == pytest.approx(0.05)
    assert sim.is_running()


def test_time_advances_only_on_step(yaml3):
    sim, _ = started(yaml3)
    for _ in range(9):
        sim.step()
    assert sim.time() == pytest.approx(0.5)


def test_step_needs_stepping_mode(yaml3):
    sim, _ = started(yaml3, stepping=False)
    with pytest.raises(SimError, match="stepping"):
        sim.step()


def test_scripted_readings_in_handle_order(yaml5):
    sim, h = started(yaml5)
    sim.set_line_sensors((0.9, 0.2, 0.9, 0.8, 0.7))
    assert sim.read_line_sensors(h.line_sensors) == (0.9, 0.2, 0.9, 0.8, 0.7)
    assert sim.read_line_sensors(tuple(reversed(h.line_sensors))) == (0.7, 0.8, 0.9, 0.2, 0.9)


def test_default_readings_are_bare_floor(yaml5):
    sim, h = started(yaml5)
    assert sim.read_line_sensors(h.line_sensors) == (1.0,) * 5
    assert sim.read_proximity(h.proximity) == (False, math.inf)


def test_scripted_proximity(yaml5):
    sim, h = started(yaml5)
    sim.set_proximity(True, 0.12)
    assert sim.read_proximity(h.proximity) == (True, 0.12)
    sim.set_proximity(False)
    assert sim.read_proximity(h.proximity) == (False, math.inf)


def test_wrong_reading_count_rejected(yaml3):
    sim, _ = started(yaml3)
    with pytest.raises(SimError, match="expected 3"):
        sim.set_line_sensors((0.1, 0.2))


def test_robot_without_proximity(yaml3):
    sim, h = started(yaml3)
    assert h.proximity is None
    with pytest.raises(SimError, match="no proximity sensor"):
        sim.read_proximity(h.proximity)


def test_wheel_speeds_recorded_with_time(yaml3):
    sim, h = started(yaml3)
    sim.set_wheel_speeds(h.left, h.right, 1.0, 2.0)
    sim.step()
    sim.set_wheel_speeds(h.left, h.right, 0.5, 0.5)
    assert sim.last_wheel_speeds == (0.5, 0.5)
    assert sim.wheel_history == ((pytest.approx(0.05), 1.0, 2.0), (pytest.approx(0.10), 0.5, 0.5))


def test_swapped_wheel_handles_swap_the_wheels(yaml3):
    """Like the real sim: the handle decides which wheel moves, not the argument position."""
    sim, h = started(yaml3)
    sim.set_wheel_speeds(h.right, h.left, 1.0, 2.0)
    assert sim.last_wheel_speeds == (2.0, 1.0)


def test_unknown_handles_rejected(yaml3):
    sim, h = started(yaml3)
    with pytest.raises(SimError, match="not a wheel"):
        sim.set_wheel_speeds(h.base, h.right, 1.0, 1.0)
    with pytest.raises(SimError, match="not a line sensor"):
        sim.read_line_sensors((h.left,))


def test_handles_must_match_robot(yaml3, yaml5):
    sim = FakeSim(yaml3)
    with pytest.raises(SimError, match="5 line sensors"):
        sim.handles_from_yaml(yaml5)


def test_calls_before_start_raise_stopped(yaml3):
    sim = FakeSim(yaml3)
    h = sim.handles_from_yaml(yaml3)
    with pytest.raises(SimStoppedError):
        sim.read_line_sensors(h.line_sensors)


def test_stop_from_gui_raises_on_next_call(yaml3):
    sim, h = started(yaml3)
    sim.set_wheel_speeds(h.left, h.right, 1.0, 1.0)
    sim.stop_from_gui()
    with pytest.raises(SimStoppedError, match="not running"):
        sim.read_line_sensors(h.line_sensors)
    with pytest.raises(SimStoppedError):
        sim.step()


def test_stop_zeroes_wheels(yaml3):
    sim, h = started(yaml3)
    sim.set_wheel_speeds(h.left, h.right, 1.0, 1.0)
    sim.stop()
    assert sim.last_wheel_speeds == (0.0, 0.0)
    assert not sim.is_running()


def test_close_zeroes_wheels_and_blocks_further_use(yaml3):
    sim, h = started(yaml3)
    sim.set_wheel_speeds(h.left, h.right, 1.0, 1.0)
    sim.close()
    assert sim.last_wheel_speeds == (0.0, 0.0)
    with pytest.raises(SimError, match="closed"):
        sim.time()


def test_with_block_zeroes_wheels_on_exception(yaml3):
    with pytest.raises(KeyboardInterrupt):
        with FakeSim(yaml3) as sim:
            sim.start(stepping=True)
            h = sim.handles_from_yaml(yaml3)
            sim.set_wheel_speeds(h.left, h.right, 1.0, 1.0)
            raise KeyboardInterrupt  # Ctrl-C
    assert sim.last_wheel_speeds == (0.0, 0.0)


def test_other_thread_is_rejected(yaml3):
    sim, h = started(yaml3)
    errors = []

    def worker():
        try:
            sim.read_line_sensors(h.line_sensors)
        except SimThreadError as e:
            errors.append(str(e))

    t = threading.Thread(target=worker, name="executor-worker-1")
    t.start()
    t.join()
    assert errors and "executor-worker-1" in errors[0] and "MainThread" in errors[0]


def test_created_on_worker_thread_is_owned_by_it(yaml3):
    """The sim_bridge pattern: the factory runs on the bridge's thread, which then owns the sim."""
    result = {}

    def bridge_thread():
        sim = FakeSim(yaml3)
        sim.start(stepping=True)
        result["sim"] = sim
        result["time"] = sim.time()

    t = threading.Thread(target=bridge_thread)
    t.start()
    t.join()
    assert result["time"] == pytest.approx(0.05)
    with pytest.raises(SimThreadError):
        result["sim"].time()  # the main thread doesn't own it


def test_restart_resets_time_and_history(yaml3):
    sim, h = started(yaml3)
    sim.set_wheel_speeds(h.left, h.right, 1.0, 1.0)
    sim.step()
    sim.start(stepping=True)
    assert sim.time() == pytest.approx(0.05)
    assert sim.wheel_history == ()
