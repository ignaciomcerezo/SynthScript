import numpy as np
import pytest
from shapely.geometry import Polygon

from synthscript.transforms import LineTransform, StrokeMaskConvexHull
from synthscript.transforms.ocr_transform_pack import OCRTransformPack


def test_hull_uses_visible_ink_and_keeps_crop_aligned_in_page_coordinates():
    image = np.zeros((8, 9, 4), dtype=np.uint8)
    image[..., 3] = 255
    image[2, 1, 0] = 100
    image[5, 6, 0] = 150
    image[1, 7, 0] = 200
    image[6, 2, 0] = 220
    image[0, 0, 0] = 255
    image[0, 0, 3] = 0
    polygon = Polygon([(30, 40), (39, 40), (39, 48), (30, 48)])

    crop, hull = StrokeMaskConvexHull()(image, polygon)

    assert isinstance(StrokeMaskConvexHull(), LineTransform)
    assert hull.bounds == (31, 41, 38, 47)
    assert hull.area < polygon.area
    assert crop.shape == (6, 7, 4)
    np.testing.assert_array_equal(crop, image[1:7, 1:8])
    for y, x in [(2, 1), (5, 6), (1, 7), (6, 2)]:
        assert hull.covers(
            Polygon(
                [(30 + x, 40 + y), (31 + x, 40 + y), (31 + x, 41 + y), (30 + x, 41 + y)]
            )
        )


def test_single_pixel_and_empty_crop():
    polygon = Polygon([(10, 20), (14, 20), (14, 24), (10, 24)])
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    transform = StrokeMaskConvexHull()

    empty_crop, empty_polygon = transform(image, polygon)
    assert empty_crop is image
    assert empty_polygon is polygon

    image[2, 3, 0] = 50
    image[2, 3, 3] = 255
    crop, hull = transform(image, polygon)
    assert crop.shape == (1, 1, 4)
    assert hull.bounds == (13, 22, 14, 23)
    assert hull.area == 1


def test_pack_preserves_line_order_and_probability():
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[1, 2, 0] = 100
    image[1, 2, 3] = 255
    first = Polygon([(10, 20), (14, 20), (14, 24), (10, 24)])
    second = Polygon([(30, 40), (34, 40), (34, 44), (30, 44)])
    pack = OCRTransformPack()
    pack.add_transform(StrokeMaskConvexHull())

    crops, polygons = pack([([image.copy(), image.copy()], [first, second])])

    assert len(crops) == len(polygons) == 2
    assert polygons[0].bounds == (12, 21, 13, 22)
    assert polygons[1].bounds == (32, 41, 33, 42)
    assert [crop.shape for crop in crops] == [(1, 1, 4), (1, 1, 4)]

    disabled = OCRTransformPack()
    disabled.add_transform(StrokeMaskConvexHull(probability=0))
    _, unchanged = disabled([([image.copy()], [first])])
    assert unchanged == [first]


def test_requires_bgra_crop():
    polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    with pytest.raises(ValueError, match="BGRA"):
        StrokeMaskConvexHull()(np.zeros((2, 2), dtype=np.uint8), polygon)
