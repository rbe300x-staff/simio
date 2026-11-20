"""Shared helpers for the staff tools. Run the tools from the repo root: python tools/<name>.py"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simio import Sim  # noqa: E402


def world_pos(sim: Sim, handle: int) -> tuple[float, float, float]:
    api = sim.api
    p = api.getObjectPosition(handle, api.handle_world)
    return float(p[0]), float(p[1]), float(p[2])


def world_matrix(sim: Sim, handle: int) -> list[float]:
    api = sim.api
    return [float(v) for v in api.getObjectMatrix(handle, api.handle_world)]


def robot_frame(sim: Sim, h) -> tuple[float, float, float, float]:
    """Robot frame from the scene: origin = midpoint of the wheel joints, x = toward the line sensors.

    Returns (origin_x, origin_y, heading, base_yaw_offset), where base_yaw_offset is the
    robot heading minus the heading of the base object's x-axis.
    """
    lx, ly, _ = world_pos(sim, h.left)
    rx, ry, _ = world_pos(sim, h.right)
    ox, oy = (lx + rx) / 2, (ly + ry) / 2
    sensors = [world_pos(sim, s) for s in h.line_sensors]
    cx = sum(p[0] for p in sensors) / len(sensors)
    cy = sum(p[1] for p in sensors) / len(sensors)
    heading = math.atan2(cy - oy, cx - ox)
    # Check that "left" is really to the left of forward; warn if the YAML has them swapped.
    left_side = -math.sin(heading) * (lx - ox) + math.cos(heading) * (ly - oy)
    if left_side < 0:
        print("WARNING: the joint named as 'left' in the YAML is on the robot's right side. Swap motors.left/right.")
    m = world_matrix(sim, h.base)
    base_heading = math.atan2(m[4], m[0])
    offset = (heading - base_heading + math.pi) % (2 * math.pi) - math.pi
    return ox, oy, heading, offset


def to_robot(x: float, y: float, frame: tuple[float, float, float, float]) -> tuple[float, float]:
    ox, oy, heading, _ = frame
    dx, dy = x - ox, y - oy
    c, s = math.cos(heading), math.sin(heading)
    return dx * c + dy * s, -dx * s + dy * c
