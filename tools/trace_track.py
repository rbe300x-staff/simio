"""Trace a scene's line into a FakeSim track file by driving the robot around it.

Staff tool. Needs the robot YAML with its geometry section (tools/measure_robot.py).
Open the scene, place the robot on the line where FakeSim should start it, then:

    python tools/trace_track.py --yaml config/lineTracer.measured.yaml --out tracks/lineFollower.yaml

A built-in closed-loop follower drives one lap in stepping mode while the script
records the center of the sensor array. It stops when it is back at the start
(closed track), when the line is lost, or after --max-steps. Check the result by
comparing the track's shape and length with the scene before using it in tests.
"""

import argparse
import math

import yaml

import _common  # noqa: F401
from _common import Sim, robot_frame, world_pos  # noqa: E402

from simio import load_robot_yaml  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True, help="robot YAML that includes geometry")
    ap.add_argument("--out", required=True, help="track file to write")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--speed", type=float, default=0.08, help="forward speed, m/s (slow traces more accurately)")
    ap.add_argument("--gain", type=float, default=20.0, help="steering gain, rad/s per m of line offset")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--spacing", type=float, default=0.01, help="keep a point every this many m")
    ap.add_argument("--floor", type=float, default=None,
                    help="sensor reading over bare floor (default: brightest sensor at the start)")
    ap.add_argument("--line-width", type=float, default=0.02, help="line width in the scene, m (measure it)")
    args = ap.parse_args()

    spec = load_robot_yaml(args.yaml)
    g = spec.require_geometry("trace_track.py")
    ys = [y for _, y in g.line_sensors]

    sim = Sim.connect(args.host, args.port)
    h = sim.handles_from_yaml(args.yaml)
    ox, oy, heading, _ = robot_frame(sim, h)
    start_pose = [round(ox, 4), round(oy, 4), round(heading, 4)]

    sim.start(stepping=True)
    points, lo, hi = [], 1.0, 0.0
    traveled, last = 0.0, None
    closed = False
    lost_steps = 0
    try:
        # Calibrate: scenes don't read 1.0 over the floor (lineFollower.ttt reads ~0.83), so
        # darkness is measured from the floor level. At the start the robot sits on the line,
        # so the brightest sensor sees floor and the darkest sees (part of) the line.
        first = sim.read_line_sensors(h.line_sensors)
        floor = args.floor if args.floor is not None else max(first)
        contrast = floor - min(first)
        if contrast < 0.05:
            raise SystemExit(
                f"sensors read {[round(v, 3) for v in first]}: no line under the array. Place the robot "
                "with the line between its sensors (or pass --floor if every sensor starts over the line)."
            )
        margin = 0.1 * contrast  # ignore floor noise
        print(f"floor {floor:.3f}, starting contrast {contrast:.3f}")
        for step in range(args.max_steps):
            values = sim.read_line_sensors(h.line_sensors)
            lo, hi = min(lo, *values), max(hi, *values)
            dark = [max(0.0, floor - v - margin) for v in values]
            total = sum(dark)
            if max(dark) < 0.25 * contrast:
                lost_steps += 1
                if lost_steps * sim.dt > 1.0:
                    print(f"line lost for 1 s at step {step}; stopping (open track, or the follower needs tuning)")
                    break
                v, omega = 0.0, 0.0
            else:
                lost_steps = 0
                y_c = sum(d * y for d, y in zip(dark, ys)) / total
                v, omega = args.speed, args.gain * y_c
            wl = (v - omega * g.track_width / 2) / g.wheel_radius
            wr = (v + omega * g.track_width / 2) / g.wheel_radius
            sim.set_wheel_speeds(h.left, h.right, wl, wr)
            sim.step()

            sensors = [world_pos(sim, s) for s in h.line_sensors]
            # The darkness-weighted point under the array approximates the line center.
            if lost_steps == 0:
                px = sum(d * p[0] for d, p in zip(dark, sensors)) / total
                py = sum(d * p[1] for d, p in zip(dark, sensors)) / total
                if last is None or math.hypot(px - last[0], py - last[1]) >= args.spacing:
                    if last is not None:
                        traveled += math.hypot(px - last[0], py - last[1])
                    points.append([round(px, 4), round(py, 4)])
                    last = (px, py)
                    if traveled > 1.0 and math.hypot(px - points[0][0], py - points[0][1]) < 3 * args.spacing:
                        closed = True
                        print(f"back at the start after {traveled:.2f} m; closing the track")
                        break
    finally:
        sim.stop()
        sim.close()

    if closed:
        points = points[:-1]
    track = {
        "line_width": args.line_width,
        "floor_intensity": round(hi, 3),
        "line_intensity": round(lo, 3),
        "start_pose": start_pose,
        "lines": [{"closed": closed, "points": points}],
        "obstacles": [],
    }
    with open(args.out, "w") as f:
        f.write(f"# Traced from the scene by tools/trace_track.py ({len(points)} points, {traveled:.2f} m)\n")
        yaml.safe_dump(track, f, sort_keys=False, default_flow_style=None)
    print(f"wrote {args.out}: {len(points)} points, {'closed' if closed else 'open'}, "
          f"floor {hi:.3f}, line {lo:.3f}")


if __name__ == "__main__":
    main()
