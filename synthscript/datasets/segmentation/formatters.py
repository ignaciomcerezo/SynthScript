from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np
from shapely import MultiPolygon, Polygon

_formatter_type = Callable[[np.ndarray, Sequence[Polygon], int], Any]


def _polygon_to_mask(poly: Polygon | MultiPolygon, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly.is_empty:
        return mask

    if poly.area == 0:
        poly = poly.buffer(1.5)

    if isinstance(poly, Polygon):
        polys = [poly]
    elif isinstance(poly, MultiPolygon):
        polys = poly.geoms
    else:
        polys = []

    for p in polys:
        coords = np.array(p.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [coords], 1)

    return mask
