from collections.abc import Sequence

import numpy as np
import shapely
from shapely.affinity import translate
from shapely.geometry import Polygon

_epsilon = 1e-12


def paragraph_hulls(polygon_groups: Sequence[Sequence[Polygon]]) -> list[Polygon]:
    """Return one convex hull per paragraph equivalent."""
    hulls: list[Polygon] = []
    for polygons in polygon_groups:
        if not polygons:
            raise ValueError(
                "Interparagraph transforms cannot process an empty paragraph."
            )
        hull = shapely.GeometryCollection(polygons).convex_hull
        if hull.is_empty or hull.area <= 0:
            raise ValueError("Paragraph geometry must have positive area.")
        hulls.append(hull)
    return hulls


def polygon_center(polygon: Polygon) -> np.ndarray:
    x0, y0, x1, y1 = polygon.bounds
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=float)


def paragraph_centers(hulls: Sequence[Polygon]) -> np.ndarray:
    return np.vstack([polygon_center(hull) for hull in hulls])


def ordered_layout_axes(
    hulls: Sequence[Polygon],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return paragraph centers, the INTER-PARAGRAPH reading direction, and its orthogonal
    axis.

    The elsewhere used reading direction is an intra-paragraph property.
    """
    centers = paragraph_centers(hulls)
    if len(centers) < 2:
        reading_direction = np.array([0.0, 1.0], dtype=float)
    else:
        trend = centers[-1] - centers[0]
        norm = float(np.linalg.norm(trend))
        if norm <= _epsilon:
            successive = np.diff(centers, axis=0)
            trend = successive.sum(axis=0)
            norm = float(np.linalg.norm(trend))
        reading_direction = (
            np.array([0.0, 1.0], dtype=float) if norm <= _epsilon else trend / norm
        )

    orthogonal_direction = np.array(
        [-reading_direction[1], reading_direction[0]], dtype=float
    )
    return centers, reading_direction, orthogonal_direction


def projected_extention(polygon: Polygon, direction: np.ndarray) -> float:
    coords = np.asarray(polygon.exterior.coords, dtype=float)[:, :2]
    projections = coords @ direction
    return float(projections.max() - projections.min())


def median_extention(hulls: Sequence[Polygon], direction: np.ndarray) -> float:
    extents = np.array(
        [projected_extention(hull, direction) for hull in hulls], dtype=float
    )
    positive = extents[extents > _epsilon]
    if len(positive) == 0:
        raise ValueError("Could not determine a positive paragraph extent.")
    return float(np.median(positive))


def translate_polygon_group(
    polygons: Sequence[Polygon], displacement: np.ndarray
) -> list[Polygon]:
    return [
        translate(polygon, xoff=float(displacement[0]), yoff=float(displacement[1]))
        for polygon in polygons
    ]


def center_data(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return values - float(np.mean(values))
