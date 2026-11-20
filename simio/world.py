"""LineWorld: a 2D floor with a dark line and round obstacles, for FakeSim's kinematic mode.

Track files (``tracks/*.yaml``) look like this::

    line_width: 0.02          # m
    floor_intensity: 1.0      # what a sensor reads over bare floor
    line_intensity: 0.0       # what a sensor reads fully over the line
    start_pose: [0.0, 0.0, 0.0]   # x, y (m), theta (rad): where the robot starts
    lines:                    # the first line is the "main" line used for lap progress
      - closed: true
        points: [[0, 0], [1, 0], [1, 1], [0, 1]]
    obstacles:                # optional
      - {x: 1.2, y: 0.0, radius: 0.05}

Set ``floor_intensity`` and ``line_intensity`` to what the real sensors read over
the floor and over the line (tools/check_sim.py prints both), so fake readings
are on the same scale as CoppeliaSim's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from .errors import SimError


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


def _footprint_pattern() -> np.ndarray:
    """37 points spread over the unit disc: the center plus rings of 6, 12, and 18."""
    pts = [(0.0, 0.0)]
    for ring, count in ((1 / 3, 6), (2 / 3, 12), (1.0, 18)):
        for k in range(count):
            a = 2 * math.pi * k / count
            pts.append((ring * math.cos(a), ring * math.sin(a)))
    return np.array(pts)


_FOOTPRINT = _footprint_pattern()


class LineWorld:
    """A floor with one or more dark polylines and circular obstacles."""

    def __init__(
        self,
        lines: Sequence[tuple[Sequence[Sequence[float]], bool]],
        *,
        line_width: float = 0.02,
        floor_intensity: float = 1.0,
        line_intensity: float = 0.0,
        start_pose: Sequence[float] = (0.0, 0.0, 0.0),
        obstacles: Iterable[Obstacle] = (),
        source: str = "<LineWorld>",
    ) -> None:
        if not lines:
            raise SimError(f"{source}: a LineWorld needs at least one line")
        if line_width <= 0:
            raise SimError(f"{source}: line_width must be positive")
        self.source = source
        self.line_width = float(line_width)
        self.floor_intensity = float(floor_intensity)
        self.line_intensity = float(line_intensity)
        if len(start_pose) != 3:
            raise SimError(f"{source}: start_pose must be [x, y, theta]")
        self.start_pose = tuple(float(v) for v in start_pose)

        seg_a, seg_b = [], []
        self._main: np.ndarray | None = None
        self._main_closed = False
        for i, (points, closed) in enumerate(lines):
            pts = np.asarray(points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
                raise SimError(f"{source}: lines[{i}].points must be a list of at least two [x, y] points")
            if closed:
                pts = np.vstack([pts, pts[:1]])
            if i == 0:
                self._main, self._main_closed = pts, bool(closed)
            seg_a.append(pts[:-1])
            seg_b.append(pts[1:])
        self._a = np.vstack(seg_a)
        self._b = np.vstack(seg_b)
        self._ab = self._b - self._a
        self._ab_len2 = np.maximum((self._ab**2).sum(axis=1), 1e-18)

        main = self._main
        assert main is not None
        seg_len = np.linalg.norm(np.diff(main, axis=0), axis=1)
        self._main_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.main_length = float(self._main_cum[-1])

        self._obstacles: dict[int, Obstacle] = {}
        self._next_id = 1
        for ob in obstacles:
            self.add_obstacle(ob.x, ob.y, ob.radius)

    # ------------------------------------------------------------ construction

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]], *, closed: bool = True, **kwargs: Any) -> "LineWorld":
        """A world with a single line through ``points``."""
        return cls([(points, closed)], **kwargs)

    @classmethod
    def stadium(
        cls,
        straight: float = 1.0,
        radius: float = 0.4,
        *,
        n_arc: int = 24,
        **kwargs: Any,
    ) -> "LineWorld":
        """A closed stadium (two straights joined by half circles), counter-clockwise.

        The robot starts at the middle of the bottom straight, heading +x. Handy
        for tests until the course tracks are traced from the scenes.
        """
        pts: list[tuple[float, float]] = []
        half = straight / 2
        for k in range(n_arc + 1):  # right arc, bottom → top
            a = -math.pi / 2 + math.pi * k / n_arc
            pts.append((half + radius * math.cos(a), radius + radius * math.sin(a)))
        for k in range(n_arc + 1):  # left arc, top → bottom
            a = math.pi / 2 + math.pi * k / n_arc
            pts.append((-half + radius * math.cos(a), radius + radius * math.sin(a)))
        kwargs.setdefault("start_pose", (0.0, 0.0, 0.0))
        return cls([(pts, True)], **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any], source: str = "<dict>") -> "LineWorld":
        if not isinstance(d, dict) or "lines" not in d:
            raise SimError(f"{source}: a track needs a 'lines' list")
        lines = []
        for i, line in enumerate(d["lines"]):
            if not isinstance(line, dict) or "points" not in line:
                raise SimError(f"{source}: lines[{i}] needs 'points'")
            lines.append((line["points"], bool(line.get("closed", False))))
        obstacles = []
        for i, ob in enumerate(d.get("obstacles") or []):
            try:
                obstacles.append(Obstacle(float(ob["x"]), float(ob["y"]), float(ob["radius"])))
            except (KeyError, TypeError, ValueError):
                raise SimError(f"{source}: obstacles[{i}] needs numeric x, y, radius") from None
        return cls(
            lines,
            line_width=float(d.get("line_width", 0.02)),
            floor_intensity=float(d.get("floor_intensity", 1.0)),
            line_intensity=float(d.get("line_intensity", 0.0)),
            start_pose=d.get("start_pose", (0.0, 0.0, 0.0)),
            obstacles=obstacles,
            source=source,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LineWorld":
        p = str(path)
        try:
            with open(p) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise SimError(f"track file not found: {p}") from None
        except yaml.YAMLError as e:
            raise SimError(f"{p} is not valid YAML: {e}") from None
        return cls.from_dict(data, source=p)

    # --------------------------------------------------------------- obstacles

    def add_obstacle(self, x: float, y: float, radius: float) -> int:
        """Add a round obstacle; returns its id (for remove_obstacle)."""
        if radius <= 0:
            raise SimError("obstacle radius must be positive")
        oid = self._next_id
        self._next_id += 1
        self._obstacles[oid] = Obstacle(float(x), float(y), float(radius))
        return oid

    def remove_obstacle(self, obstacle_id: int) -> None:
        """Remove an obstacle ("the LA removes the obstacle")."""
        if obstacle_id not in self._obstacles:
            raise SimError(f"no obstacle with id {obstacle_id}")
        del self._obstacles[obstacle_id]

    @property
    def obstacles(self) -> tuple[Obstacle, ...]:
        return tuple(self._obstacles.values())

    # ---------------------------------------------------------------- geometry

    def distance_to_line(self, points: np.ndarray) -> np.ndarray:
        """Distance from each point (shape (M, 2)) to the nearest line segment."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        ap = p[:, None, :] - self._a[None, :, :]
        t = np.clip((ap * self._ab[None, :, :]).sum(axis=2) / self._ab_len2[None, :], 0.0, 1.0)
        closest = self._a[None, :, :] + t[:, :, None] * self._ab[None, :, :]
        d = np.linalg.norm(p[:, None, :] - closest, axis=2)
        return d.min(axis=1)

    def sensor_intensities(self, centers: np.ndarray, radius: float) -> np.ndarray:
        """Intensity seen by circular sensor footprints centered at ``centers`` (shape (N, 2))."""
        c = np.atleast_2d(np.asarray(centers, dtype=float))
        samples = (c[:, None, :] + radius * _FOOTPRINT[None, :, :]).reshape(-1, 2)
        on_line = self.distance_to_line(samples) <= self.line_width / 2
        frac = on_line.reshape(len(c), len(_FOOTPRINT)).mean(axis=1)
        return self.floor_intensity - frac * (self.floor_intensity - self.line_intensity)

    def ray_cast(self, x: float, y: float, theta: float, max_range: float) -> float:
        """Distance along a ray to the nearest obstacle, or ``math.inf`` beyond ``max_range``."""
        dx, dy = math.cos(theta), math.sin(theta)
        best = math.inf
        for ob in self._obstacles.values():
            fx, fy = x - ob.x, y - ob.y
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - ob.radius * ob.radius
            if c <= 0:  # the sensor is inside the obstacle
                return 0.0
            disc = b * b - c
            if disc < 0:
                continue
            t = -b - math.sqrt(disc)
            if 0 <= t < best:
                best = t
        return best if best <= max_range else math.inf

    def project_on_main(self, x: float, y: float) -> tuple[float, float]:
        """``(arclength, distance)`` of the closest point on the main line."""
        main = self._main
        assert main is not None
        a, b = main[:-1], main[1:]
        ab = b - a
        p = np.array([x, y])
        t = np.clip(((p - a) * ab).sum(axis=1) / np.maximum((ab**2).sum(axis=1), 1e-18), 0.0, 1.0)
        closest = a + t[:, None] * ab
        d = np.linalg.norm(closest - p, axis=1)
        i = int(np.argmin(d))
        seg_len = float(self._main_cum[i + 1] - self._main_cum[i])
        return float(self._main_cum[i] + t[i] * seg_len), float(d[i])

    @property
    def main_closed(self) -> bool:
        return self._main_closed
