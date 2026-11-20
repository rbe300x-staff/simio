"""Check simio.Sim against a live CoppeliaSim and the course scenes.

Staff tool. Open a scene in CoppeliaSim (simulation stopped), then from the repo root:

    python tools/check_sim.py --yaml config/lineTracer.yaml
    python tools/check_sim.py --yaml config/lineTracer_prox.yaml --interactive

Put the robot so its middle sensor is over the line before running. The script
drives the robot briefly, then stops the simulation. Paste the whole output back
to whoever is maintaining simio.
"""

import argparse
import math
import threading
import traceback

import _common  # noqa: F401  (puts the repo root on sys.path)
from _common import Sim  # noqa: E402

from simio import SimError, SimStoppedError, SimThreadError  # noqa: E402
from simio.sim import MEAN_INTENSITY_INDEX  # noqa: E402

results: list[tuple[str, str]] = []


def check(name):
    def wrap(fn):
        def run(*args, **kwargs):
            print(f"\n== {name}")
            try:
                out = fn(*args, **kwargs)
                results.append((name, "PASS"))
                return out
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                results.append((name, f"FAIL: {type(e).__name__}: {e}"))
                return None

        return run

    return wrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True, help="robot YAML for the open scene")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--wheel", type=float, default=2.0, help="wheel speed for the drive test, rad/s")
    ap.add_argument("--interactive", action="store_true", help="also test pressing Stop in the GUI")
    args = ap.parse_args()

    sim = check("connect")(lambda: Sim.connect(args.host, args.port))()
    if sim is None:
        return summary()
    print("CoppeliaSim version:", ".".join(map(str, sim.version)))

    h = check("handles_from_yaml")(lambda: sim.handles_from_yaml(args.yaml))()
    if h is None:
        return summary()
    print(h)

    @check("start(stepping=True) returns after the first step")
    def start():
        sim.start(stepping=True)
        t = sim.time()
        print(f"sim time after start: {t:.4f} s, dt = {sim.dt:.4f} s")
        assert t > 0, "time should be > 0 after start()"

    start()

    @check(f"readVisionSensor packet layout (mean intensity at index {MEAN_INTENSITY_INDEX})")
    def packets():
        api = sim.api
        for name, s in zip(("sensor %d" % i for i in range(len(h.line_sensors))), h.line_sensors):
            res = api.readVisionSensor(s)
            print(f"{name}: result={res[0]!r}")
            pkt = list(res[1]) if len(res) > 1 else []
            print(f"  aux packet ({len(pkt)} values):", [round(float(v), 3) for v in pkt[:15]])
        print("Expect 15 values: min[0:5], max[5:10], mean[10:15]; mean intensity at [10].")

    packets()

    @check("read_line_sensors")
    def lines():
        v = sim.read_line_sensors(h.line_sensors)
        print("readings:", [round(x, 3) for x in v], "(low = dark = line)")
        return v

    lines()

    if h.proximity is not None:

        @check("read_proximity")
        def prox():
            print("reading:", sim.read_proximity(h.proximity))

        prox()

    @check("get_pose2d(base)")
    def pose():
        print("pose:", tuple(round(v, 4) for v in sim.get_pose2d(h.base)))

    pose()

    @check("drive: positive wheel speeds move the robot forward")
    def drive():
        x0, y0, _ = sim.get_pose2d(h.base)
        lo, hi = 1.0, 0.0
        sim.set_wheel_speeds(h.left, h.right, args.wheel, args.wheel)
        for _ in range(20):
            sim.step()
            v = sim.read_line_sensors(h.line_sensors)
            lo, hi = min(lo, *v), max(hi, *v)
        sim.set_wheel_speeds(h.left, h.right, 0.0, 0.0)
        x1, y1, _ = sim.get_pose2d(h.base)
        d = math.hypot(x1 - x0, y1 - y0)
        print(f"moved {d:.3f} m in {20 * sim.dt:.2f} s of sim time")
        print(f"sensor range seen while driving: min {lo:.3f} (line?), max {hi:.3f} (floor?)")
        print("Use these as line_intensity / floor_intensity in the track files.")
        assert d > 0.005, "the robot didn't move; check motors.left/right in the YAML and the joint modes"

    drive()

    @check("another thread is rejected with SimThreadError")
    def thread():
        caught = []

        def worker():
            try:
                sim.time()
            except SimThreadError as e:
                caught.append(e)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert caught, "expected SimThreadError"

    thread()

    if args.interactive:

        @check("pressing Stop in the GUI raises SimStoppedError")
        def gui_stop():
            input("\nPress Stop in CoppeliaSim, then press Enter here... ")
            try:
                sim.read_line_sensors(h.line_sensors)
            except SimStoppedError as e:
                print("raised as expected:", e)
                return
            raise AssertionError("no SimStoppedError after Stop")

        gui_stop()
        check("start again after a GUI stop")(lambda: sim.start(stepping=True))()

    @check("stop() zeroes wheels and stops the sim")
    def stop():
        sim.set_wheel_speeds(h.left, h.right, args.wheel, args.wheel)
        sim.stop()
        assert not sim.is_running()

    stop()
    try:
        sim.close()
    except SimError:
        pass
    summary()


def summary():
    print("\n== Summary")
    for name, status in results:
        print(f"{status[:4]}  {name}" + ("" if status == "PASS" else f"\n      {status}"))


if __name__ == "__main__":
    main()
