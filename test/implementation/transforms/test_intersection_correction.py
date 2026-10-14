import numpy as np
import pytest
import shapely
from shapely.affinity import rotate

from synthscript.transforms.geometric_helpers import (
    intersection_correction as correction,
)
from synthscript.transforms.line import Blur
from synthscript.transforms.ocr_transform_pack import OCRTransformPack
from synthscript.transforms.paragraph import ParagraphTilt


def _center(polygons):
    bounds = shapely.bounds(polygons)
    return (bounds[:, :2].min(axis=0) + bounds[:, 2:].max(axis=0)) / 2


def test_intersection_risk_tracks_live_probability():
    pack = OCRTransformPack(rng=123)
    assert not pack.may_cause_intersections
    pack.add_transform(Blur(1))
    risky = ParagraphTilt(0.2, probability=0)
    pack.add_transform(risky)
    assert not pack.may_cause_intersections
    risky.probability = 0.01
    assert pack.may_cause_intersections
    risky.probability = 0
    assert not pack.may_cause_intersections


def test_disabled_risky_transform_skips_correction_and_preserves_rng(monkeypatch):
    pack = OCRTransformPack(rng=123)
    pack.add_transform(Blur(1))
    pack.add_transform(ParagraphTilt(0.2, probability=0))

    def unexpected_correction(*args, **kwargs):
        pytest.fail("Disabled risky transforms must not trigger layout correction.")

    monkeypatch.setattr(
        "synthscript.transforms.ocr_transform_pack.avoid_line_intersections",
        unexpected_correction,
    )
    polygons = [shapely.box(0, 0, 80, 10), shapely.box(0, 8, 80, 18)]
    images = [np.zeros((10, 80, 4), dtype=np.uint8) for _ in polygons]
    _, result = pack([(images, list(polygons))])
    assert all(a is b for a, b in zip(polygons, result, strict=True))
    expected_rng = np.random.default_rng(123)
    expected_rng.random()  # The disabled paragraph transform still draws its trigger.
    assert pack.rng.bit_generator.state == expected_rng.bit_generator.state


@pytest.mark.parametrize("count", [0, 1, 2, 30])
def test_clear_lines_keep_geometry_identity_without_group_analysis(monkeypatch, count):
    def unexpected_analysis(*args, **kwargs):
        pytest.fail("Already separated lines must bypass geometry analysis.")

    monkeypatch.setattr(correction.LineGroupInfo, "from_polygons", unexpected_analysis)
    # Adjacent gaps equal delta, including the exact early-return boundary.
    polygons = [shapely.box(0, i * 10.5, 80, i * 10.5 + 10) for i in range(count)]
    original = list(polygons)
    result = correction.avoid_line_intersections(polygons, delta=0.5)
    assert result is polygons
    assert all(a is b for a, b in zip(original, result, strict=True))


@pytest.mark.parametrize("angle", [-30, 0, 25])
@pytest.mark.parametrize("concave", [False, True])
@pytest.mark.parametrize("binary_iterations", [0, 32, 100])
def test_correction_separates_lines_preserves_area_and_center(
    angle, concave, binary_iterations
):
    polygons = []
    for i in range(12):
        y = i * 8
        polygon = (
            shapely.Polygon([(0, y), (80, y), (80, y + 10), (40, y + 7), (0, y + 10)])
            if concave
            else shapely.box(0, y, 80, y + 10)
        )
        polygons.append(rotate(polygon, angle, origin=(0, 0)))
    original_center = _center(polygons)
    original_areas = shapely.area(polygons)
    result = correction.avoid_line_intersections(
        polygons, binary_iterations=binary_iterations
    )
    np.testing.assert_allclose(_center(result), original_center, atol=1e-9)
    np.testing.assert_allclose(shapely.area(result), original_areas, atol=1e-9)
    assert all(polygon.is_valid for polygon in result)
    assert all(
        a.intersection(b).area <= 1e-8
        for i, a in enumerate(result)
        for b in result[i + 1 :]
    )


def test_clear_paragraphs_keep_polygon_identity():
    groups = [[shapely.box(0, 0, 80, 10)], [shapely.box(0, 40, 80, 50)]]
    original = [group[0] for group in groups]
    result = correction.avoid_paragraph_intersections(groups)
    assert result is groups
    assert all(
        group[0] is polygon for group, polygon in zip(result, original, strict=True)
    )


@pytest.mark.parametrize("angle", [-20, 20])
def test_nonadjacent_rotated_lines_are_separated(angle):
    widths = [300, 30, 600] if angle > 0 else [600, 30, 300]
    polygons = [
        rotate(shapely.box(0, i * 25, width, i * 25 + 12), angle)
        for i, width in enumerate(widths)
    ]
    assert polygons[0].distance(polygons[1]) > 0.5
    assert polygons[1].distance(polygons[2]) > 0.5
    assert polygons[0].intersection(polygons[2]).area > 0
    original_center = _center(polygons)
    original_areas = shapely.area(polygons)
    result = correction.avoid_line_intersections(polygons)
    assert all(
        a.distance(b) >= 0.5 - 1e-8
        for i, a in enumerate(result)
        for b in result[i + 1 :]
    )
    np.testing.assert_allclose(_center(result), original_center, atol=1e-9)
    np.testing.assert_allclose(shapely.area(result), original_areas, atol=1e-9)


def test_varying_widths_remain_separated_after_contraction():
    # The short lines around a long formula reproduce the LaTeX page's layout.
    bounds = [
        (291, 267, 512, 289),
        (259, 292, 830, 314),
        (259, 316, 625, 339),
        (259, 342, 441, 359),
        (259, 361, 604, 389),
        (259, 391, 465, 413),
        (259, 415, 483, 439),
        (259, 441, 550, 463),
        (259, 465, 423, 491),
        (259, 490, 945, 514),
        (259, 515, 759, 539),
    ]
    polygons = [rotate(shapely.box(*bound), 20) for bound in bounds]
    result = correction.avoid_line_intersections(polygons)
    assert all(
        a.distance(b) >= 0.5 - 1e-8
        for i, a in enumerate(result)
        for b in result[i + 1 :]
    )


def test_overlapping_paragraphs_still_move():
    groups = [[shapely.box(0, 0, 80, 10)], [shapely.box(0, 8, 80, 18)]]
    original = [group[0] for group in groups]
    result = correction.avoid_paragraph_intersections(groups)
    assert all(
        group[0] is not polygon for group, polygon in zip(result, original, strict=True)
    )
    assert result[0][0].distance(result[1][0]) >= 5 - 0.01
