"""Measure the robot's geometry in an open scene and print the YAML 'geometry:' section.

Staff tool. Open the scene in CoppeliaSim (simulation can be stopped), then:

    python tools/measure_robot.py --yaml config/lineTracer_prox.yaml
    python tools/measure_robot.py --yaml config/lineTracer_prox.yaml --out config/lineTracer_prox.measured.yaml

The robot frame used by FakeSim and sim_bridge: origin at the midpoint between the
wheel joints, x toward the line sensors (forward), y to the left. Anything the
script can't read is printed as MEASURE_BY_HAND with a hint; fill those in from
the CoppeliaSim object properties.
"""

import argparse
import math

import yaml

import _common  # noqa: F401
from _common import Sim, robot_frame, to_robot, world_pos  # noqa: E402

MANUAL = "MEASURE_BY_HAND"


def footprint_radius(sim, sensor):
    """Radius of the floor patch a downward vision sensor sees."""
    api = sim.api
    z = world_pos(sim, sensor)[2]
    try:
        perspective = api.getObjectInt32Param(sensor, api.visionintparam_perspective_operation)
        if perspective:
            angle = api.getObjectFloatParam(sensor, api.visionfloatparam_perspective_angle)
            return z * math.tan(angle / 2), f"perspective, angle {math.degrees(angle):.1f} deg, height {z:.4f} m"
        size = api.getObjectFloatParam(sensor, api.visionfloatparam_ortho_size)
        return size / 2, f"orthographic, size {size:.4f} m"
    except Exception as e:  # noqa: BLE001
        return None, f"could not read the vision sensor parameters ({e})"


def wheel_radius(sim, joint):
    """Axle height above the floor (assumes the floor is at z = 0), cross-checked with the wheel's bounding box."""
    api = sim.api
    axle_z = world_pos(sim, joint)[2]
    note = f"axle height {axle_z:.4f} m (floor assumed at z = 0)"
    try:
        child = api.getObjectChild(joint, 0)
        if child != -1:
            size = api.getShapeBB(child)
            note += f"; wheel shape bounding box {[round(float(s), 4) for s in size]} → radius {max(size) / 2:.4f} m"
    except Exception:  # noqa: BLE001
        pass
    return axle_z, note


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", help="write the YAML with the geometry section to this path (never overwrites --yaml)")
    args = ap.parse_args()

    sim = Sim.connect(args.host, args.port)
    h = sim.handles_from_yaml(args.yaml)
    frame = robot_frame(sim, h)
    notes = []

    lx, ly, _ = world_pos(sim, h.left)
    rx, ry, _ = world_pos(sim, h.right)
    track = math.hypot(lx - rx, ly - ry)
    notes.append(f"track_width: distance between the wheel joints = {track:.4f} m")

    r_left, note_l = wheel_radius(sim, h.left)
    r_right, _ = wheel_radius(sim, h.right)
    radius = (r_left + r_right) / 2
    notes.append(f"wheel_radius: {note_l}")

    sensors = []
    for s in h.line_sensors:
        x, y, _ = world_pos(sim, s)
        sensors.append([round(v, 4) for v in to_robot(x, y, frame)])
    order_ok = all(sensors[i][1] >= sensors[i + 1][1] for i in range(len(sensors) - 1))
    if not order_ok:
        notes.append("WARNING: sensors in YAML order are not left → right; FakeSim doesn't care, but controllers might.")

    fp, fp_note = footprint_radius(sim, h.line_sensors[len(h.line_sensors) // 2])
    notes.append(f"line_sensor_radius: {fp_note}")

    geometry = {
        "wheel_radius": round(radius, 4),
        "track_width": round(track, 4),
        "base_yaw_offset": round(frame[3], 4),
        "line_sensors": sensors,
        "line_sensor_radius": round(fp, 4) if fp else MANUAL,
    }
    if h.proximity is not None:
        px, py, _ = world_pos(sim, h.proximity)
        m = _common.world_matrix(sim, h.proximity)
        # A proximity sensor detects along its own z-axis (third column of the rotation).
        heading = math.atan2(m[6], m[2]) - frame[2]
        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        rx_, ry_ = to_robot(px, py, frame)
        geometry["proximity"] = {
            "x": round(rx_, 4),
            "y": round(ry_, 4),
            "heading": round(heading, 4),
            "max_range": MANUAL,
        }
        notes.append(
            "proximity.max_range: read it in CoppeliaSim (proximity sensor → Volume parameters → Range) and fill it in"
        )

    block = yaml.safe_dump({"geometry": geometry}, sort_keys=False, default_flow_style=None)
    print("# --- measured from the scene ---")
    for n in notes:
        print("#", n)
    print(block)
    if args.out:
        with open(args.yaml) as f:
            data = yaml.safe_load(f)
        data["geometry"] = geometry
        with open(args.out, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
        print(f"wrote {args.out}")
    sim.close()


if __name__ == "__main__":
    main()
