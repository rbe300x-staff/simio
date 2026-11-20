import math

import numpy as np
import pytest

from simio import LineWorld, SimError


def test_intensity_on_and_off_the_line(stadium):
    on, off = stadium.sensor_intensities(np.array([[0.0, 0.0], [0.0, 0.2]]), radius=0.006)
    assert on == pytest.approx(0.0)
    assert off == pytest.approx(1.0)


def test_partial_cover_is_between(stadium):
    # Footprint centered on the line's edge: about half covered.
    (v,) = stadium.sensor_intensities(np.array([[0.0, 0.01]]), radius=0.006)
    assert 0.3 < v < 0.7


def test_custom_intensity_scale():
    w = LineWorld.from_points([[0, 0], [1, 0]], closed=False, floor_intensity=0.85, line_intensity=0.1)
    on, off = w.sensor_intensities(np.array([[0.5, 0.0], [0.5, 0.5]]), radius=0.005)
    assert on == pytest.approx(0.1) and off == pytest.approx(0.85)


def test_distance_to_line(stadium):
    d = stadium.distance_to_line(np.array([[0.0, 0.1], [0.0, 0.4]]))
    assert d[0] == pytest.approx(0.1)
    assert d[1] == pytest.approx(0.4)


def test_stadium_length():
    w = LineWorld.stadium(straight=1.0, radius=0.4, n_arc=200)
    assert w.main_length == pytest.approx(2 * 1.0 + 2 * math.pi * 0.4, rel=1e-3)


def test_ray_cast_obstacles(stadium):
    oid = stadium.add_obstacle(1.0, 0.0, 0.05)
    assert stadium.ray_cast(0.0, 0.0, 0.0, max_range=2.0) == pytest.approx(0.95)
    assert stadium.ray_cast(0.0, 0.0, 0.0, max_range=0.5) == math.inf  # beyond range
    assert stadium.ray_cast(0.0, 0.0, math.pi, max_range=2.0) == math.inf  # behind
    stadium.remove_obstacle(oid)
    assert stadium.ray_cast(0.0, 0.0, 0.0, max_range=2.0) == math.inf
    with pytest.raises(SimError):
        stadium.remove_obstacle(oid)


def test_from_yaml(tmp_path):
    p = tmp_path / "track.yaml"
    p.write_text(
        "line_width: 0.03\nfloor_intensity: 0.9\nline_intensity: 0.05\nstart_pose: [0, 0, 1.5]\n"
        "lines:\n  - closed: true\n    points: [[0, 0], [1, 0], [1, 1], [0, 1]]\n"
        "obstacles:\n  - {x: 2, y: 2, radius: 0.1}\n"
    )
    w = LineWorld.from_yaml(p)
    assert w.line_width == 0.03 and w.start_pose == (0.0, 0.0, 1.5)
    assert w.main_length == pytest.approx(4.0)
    assert len(w.obstacles) == 1


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("line_width: 0.02\n", "'lines'"),
        ("lines:\n  - closed: true\n", "needs 'points'"),
        ("lines:\n  - points: [[0, 0]]\n", "at least two"),
        ("lines:\n  - points: [[0, 0], [1, 0]]\nobstacles:\n  - {x: 1}\n", "numeric x, y, radius"),
    ],
)
def test_track_errors_name_the_file(tmp_path, text, fragment):
    p = tmp_path / "bad_track.yaml"
    p.write_text(text)
    with pytest.raises(SimError) as e:
        LineWorld.from_yaml(p)
    assert fragment in str(e.value) and "bad_track.yaml" in str(e.value)
