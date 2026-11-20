# simio

The course's simulator interface for RBE 300x (Spring 2027). It replaces the 2026 `coppeliaBridge`. The contract is `miniros/api-spec.md` §5. Course-provided: teams use `simio` but don't edit it.

| Part | What it is | Who uses it |
|---|---|---|
| `Sim` | The real CoppeliaSim, through the ZMQ remote API | Lab 1 monolith; `sim_bridge` from Lab 2 |
| `FakeSim` | The same interface with no simulator: **scripted** (set the readings) or **kinematic** (the robot moves on a 2D track) | Public tests, private tests, team tests |
| `SimInterface` | The `Protocol` both satisfy | `sim_bridge` depends on it, never on `Sim` |
| `LineWorld` | The 2D track (lines, line width, obstacles) for kinematic mode | Tests; track files in `tracks/` |

Requirements: Python ≥ 3.10, `pyyaml`, `numpy`; for `Sim` also `pip install coppeliasim-zmqremoteapi-client` (CoppeliaSim 4.6 or later).

## Lab 1: driving the robot directly

```python
from simio import Sim

with Sim.connect() as sim:                       # Ctrl-C zeroes the wheels and stops the sim
    h = sim.handles_from_yaml("config/lineTracer.yaml")
    sim.start(stepping=False)                     # returns once the sim is running; no sleep() needed
    while True:
        left, middle, right = sim.read_line_sensors(h.line_sensors)   # low = line (floor ~0.83, line ~0.38 in lineFollower.ttt)
        ...
        sim.set_wheel_speeds(h.left, h.right, left_rad_s, right_rad_s)
```

Errors name what went wrong: a misspelled object in the YAML raises `SimError` with the scene path and the YAML file; stopping the sim from the GUI raises `SimStoppedError` on the next call; using the sim from another thread raises `SimThreadError`.

## From Lab 2: `sim_bridge` gets a factory

A CoppeliaSim client belongs to the thread that creates it. So `sim_bridge` takes a factory and calls it from its own callback, not from `__init__` (api-spec §20). That way the same node runs on a dedicated thread in Lab 3 with no edits.

```python
SimBridge(lambda: Sim.connect(port=23000), robot_yaml="config/lineTracer_prox.yaml")      # lab
SimBridge(lambda: FakeSim("config/lineTracer_prox.yaml"), robot_yaml="config/lineTracer_prox.yaml")  # tests
```

## Tests: `FakeSim`

Time advances only on `step()`, by exactly `dt`, so tests are deterministic.

**Scripted mode**: check what code does with given readings.

```python
sim = FakeSim("config/lineTracer_prox.yaml")
sim.start(stepping=True)
h = sim.handles_from_yaml("config/lineTracer_prox.yaml")
sim.set_line_sensors((1.0, 0.2, 1.0, 1.0, 1.0))
sim.set_proximity(True, 0.05)
# ... run the code under test ...
assert sim.last_wheel_speeds == (0.0, 0.0)
```

**Kinematic mode**: check that a closed-loop behavior reaches its goal. Needs the robot YAML's `geometry:` section.

```python
world = LineWorld.from_yaml("tracks/lineFollower.yaml")
sim = FakeSim("config/lineTracer_prox.yaml", world=world)
sim.start(stepping=True)
# ... run the follower for N steps ...
assert sim.lap_progress >= 1.0 and sim.max_line_error < 0.02
oid = world.add_obstacle(1.2, 0.0, 0.05)   # later: world.remove_obstacle(oid)
```

Kinematic mode checks logic, not tuning: there's no slip, inertia, or motor lag. Tune gains in CoppeliaSim. One small difference from `Sim`: `FakeSim.get_pose2d(h.base)` reports the position of the robot-frame origin (the midpoint between the wheels), while CoppeliaSim reports the base object's origin, which may sit a few millimeters away.

See `simio/tests/` for working examples, including a reference line follower (`conftest.follow_line`) that completes a lap at two step sizes, stops for an obstacle until it's removed, and an open-loop "drive for N seconds" that leaves the line.

## Staff tools (run with CoppeliaSim open)

From the repo root, with a scene open and the simulation stopped:

| Script | What it does | Output |
|---|---|---|
| `python tools/check_sim.py --yaml config/lineTracer.yaml [--interactive]` | Exercises every `Sim` call against the live scene: connect and version, handles, the vision-sensor packet layout, readings, pose, a short drive, the thread rule, stop (and, with `--interactive`, pressing Stop in the GUI) | PASS/FAIL summary plus the floor and line intensities |
| `python tools/measure_robot.py --yaml config/lineTracer_prox.yaml --out config/lineTracer_prox.measured.yaml` | Reads wheel radius, track width, sensor positions and footprint, and the proximity sensor's pose from the scene | The `geometry:` section; anything it can't read is marked `MEASURE_BY_HAND` |
| `python tools/trace_track.py --yaml config/lineTracer.measured.yaml --out tracks/lineFollower.yaml` | Place the robot on the line first. It reads the floor level from the brightest sensor (or `--floor`), drives one lap with a built-in follower in stepping mode, and records the line's path. It stops if it loses the line for 1 s | A track file for kinematic `FakeSim` |

The measurement and tracing math is tested against a simulated scene (`simio/tests/test_tools.py`); what still needs a live run is CoppeliaSim's real API.

## Status

- `FakeSim`, `LineWorld`, robot YAML parsing: complete and tested (86 tests). A missing YAML file suggests the closest name in the folder (`did you mean config/lineTracer_prox.yaml?`).
- `Sim`: **verified on CoppeliaSim 4.10.0** with `lineFollower.ttt` (`tools/check_sim.py`, 2026-11-10: all 9 checks pass; mean intensity is at index 10 of the vision-sensor packet). Still to run live: the obstacle scene with `lineTracer_prox.yaml --interactive` (proximity sensor, GUI stop). Install CoppeliaSim 4.10.0 for the course; `simio` accepts 4.6 or later.
- Sensor scale in `lineFollower.ttt`: bare floor reads about 0.83, the darkest line pixels about 0.38. Readings aren't 0-to-1 in practice, so controllers should take thresholds from configuration, and track files should set `floor_intensity` / `line_intensity` from the scene (`tools/trace_track.py` records both). Fixed in 0.1.1: `trace_track.py` and the test reference follower assumed the floor reads 1.0, so on the real scene they steered too weakly for curves and didn't notice losing the line.
- Not yet implemented (waiting on scene decisions, api-spec §5.4): the in-sim command watchdog signal and batched sensor reads.
- `config/` holds the 2026 YAML files unchanged, plus `lineTracer.measured.yaml` and `lineTracer_prox.measured.yaml` (measured 2026-11-10: wheel radius 0.027 m, track width 0.1163 m, line sensors 0.1284 m ahead of the axle and 0.0225 m apart; proximity sensor 0.044 m ahead, `max_range` 0.2 m set by hand). `tracks/lineFollower.yaml` (closed, 4.39 m) and `tracks/lineFollower_prox.yaml` (closed, 6.64 m; obstacle not yet recorded) are traced from the course scenes. The reference follower laps both at two step sizes and waits for an obstacle on the second (`test_fake_kinematic.py`). `tracks/example_stadium.yaml` is a made-up track for examples, not a course scene.
