import pytest

from simio import SimError, load_robot_yaml


def test_2026_format_without_geometry(yaml3):
    spec = load_robot_yaml(yaml3)
    assert spec.name == "LineTracer"
    assert spec.line_sensors == ("LeftSensor", "MiddleSensor", "RightSensor")
    assert spec.proximity is None
    assert spec.geometry is None
    assert spec.object_path(spec.left_motor) == "/LineTracer/DynamicLeftJoint"


def test_geometry_parsed(yaml5):
    spec = load_robot_yaml(yaml5)
    g = spec.geometry
    assert g.wheel_radius == 0.02 and g.track_width == 0.10
    assert len(g.line_sensors) == 5 and g.line_sensors[0] == (0.06, 0.03)
    assert g.proximity.max_range == 0.3


def test_sensors_sorted_by_key(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("name: R\nmotors: {left: L, right: R}\nsensors: {2: C, 0: A, 1: B}\n")
    assert load_robot_yaml(p).line_sensors == ("A", "B", "C")


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("motors: {left: L, right: R}\nsensors: {0: A}\n", "missing 'name'"),
        ("name: R\nsensors: {0: A}\n", "missing 'motors'"),
        ("name: R\nmotors: {left: L}\nsensors: {0: A}\n", "missing 'right'"),
        ("name: R\nmotors: {left: L, right: R}\nsensors: {}\n", "non-empty"),
        (
            "name: R\nmotors: {left: L, right: R}\nsensors: {0: A, 1: B}\n"
            "geometry: {wheel_radius: 0.02, track_width: 0.1, line_sensor_radius: 0.005, line_sensors: [[0.06, 0]]}\n",
            "must list 2",
        ),
    ],
)
def test_errors_name_the_file_and_problem(tmp_path, text, fragment):
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(SimError) as e:
        load_robot_yaml(p)
    assert fragment in str(e.value) and "bad.yaml" in str(e.value)


def test_missing_file():
    with pytest.raises(SimError, match="not found"):
        load_robot_yaml("no/such/file.yaml")


def test_missing_file_suggests_close_name(yaml5, tmp_path):
    typo = str(tmp_path / "lineTracerprox.yaml")
    with pytest.raises(SimError, match=r"did you mean .*lineTracer_prox\.yaml"):
        load_robot_yaml(typo)
