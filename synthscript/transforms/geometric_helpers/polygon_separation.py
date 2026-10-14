from dataclasses import dataclass

import numpy as np
from shapely import Polygon
from shapely.affinity import translate

from synthscript.shared.parameters import Vector2D

_EPSILON = 1e-12


def _coords_of(polygon: Polygon) -> np.ndarray:
    coords = np.asarray(polygon.exterior.coords, dtype=float)[:, :2]
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def _polygon_axes(coords: np.ndarray) -> np.ndarray:
    """Return the outward edge normals of a convex polygon."""
    edges = np.roll(coords, -1, axis=0) - coords
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > _EPSILON
    return normals[valid] / lengths[valid, None]


@dataclass(slots=True)
class _SatPair:
    """SAT cache (translation-invariant) for pairs of convex polygons."""

    axes: np.ndarray
    min_a: np.ndarray
    max_a: np.ndarray
    min_b: np.ndarray
    max_b: np.ndarray

    @classmethod
    def from_coords(cls, coords_a: np.ndarray, coords_b: np.ndarray) -> "_SatPair":
        axes = np.concatenate((_polygon_axes(coords_a), _polygon_axes(coords_b)))
        if len(axes) == 0:
            empty = np.empty(0, dtype=float)
            return cls(axes, empty, empty, empty, empty)

        projections_a = coords_a @ axes.T
        projections_b = coords_b @ axes.T
        return cls(
            axes,
            projections_a.min(axis=0),
            projections_a.max(axis=0),
            projections_b.min(axis=0),
            projections_b.max(axis=0),
        )

    def minimum_translation_vector(
        self, shift_a: Vector2D, shift_b: Vector2D
    ) -> Vector2D | None:
        if len(self.axes) == 0:
            return None

        offset_a = self.axes @ shift_a
        offset_b = self.axes @ shift_b
        min_a = self.min_a + offset_a
        max_a = self.max_a + offset_a
        min_b = self.min_b + offset_b
        max_b = self.max_b + offset_b
        overlaps = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)

        if np.any(overlaps <= 0):
            return None

        axis_index = int(np.argmin(overlaps))
        center_a = 0.5 * (min_a[axis_index] + max_a[axis_index])
        center_b = 0.5 * (min_b[axis_index] + max_b[axis_index])
        sign = 1.0 if center_b >= center_a else -1.0
        return self.axes[axis_index] * overlaps[axis_index] * sign


def separate_polygons(
    polygons: list[Polygon],
    *,
    delta: float = 5,
    max_iterations: int = 100,
    damping: float = 0.5,
    tol: float = 1e-3,
) -> tuple[list[Polygon], list[Vector2D]]:
    """Linearly translate convex polygons until they have ``delta`` clearance."""
    n = len(polygons)
    shifts = np.zeros((n, 2), dtype=float)
    if n <= 1:
        return list(polygons), [shift for shift in shifts]

    current = list(polygons)
    coords = [_coords_of(polygon) for polygon in polygons]
    centroids = np.array(
        [(polygon.centroid.x, polygon.centroid.y) for polygon in polygons],
        dtype=float,
    )
    bounds = np.array([polygon.bounds for polygon in polygons], dtype=float)

    pairs = [
        (i, j, _SatPair.from_coords(coords[i], coords[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]

    for _ in range(max_iterations):
        moves = np.zeros((n, 2), dtype=float)
        any_adjustment = False

        for i, j, sat_pair in pairs:
            bounds_i = bounds[i]
            bounds_j = bounds[j]
            if (
                bounds_i[0] - delta > bounds_j[2]
                or bounds_i[2] + delta < bounds_j[0]
                or bounds_i[1] - delta > bounds_j[3]
                or bounds_i[3] + delta < bounds_j[1]
            ):
                continue

            mtv = sat_pair.minimum_translation_vector(shifts[i], shifts[j])
            if mtv is None:
                gap = current[i].distance(current[j])
                if gap >= delta:
                    continue

                direction = (centroids[j] + shifts[j]) - (centroids[i] + shifts[i])
                norm = np.linalg.norm(direction)
                direction = (
                    direction / norm if norm > _EPSILON else np.array([1.0, 0.0])
                )
                push = (delta - gap) * direction
            else:
                mtv_length = np.linalg.norm(mtv)
                push = mtv + delta * mtv / max(mtv_length, _EPSILON)

            half_push = 0.5 * damping * push
            moves[j] += half_push
            moves[i] -= half_push
            any_adjustment = True

        if not any_adjustment:
            break

        move_lengths_sq = np.einsum("ij,ij->i", moves, moves)
        moved = move_lengths_sq > _EPSILON**2
        for i in np.flatnonzero(moved):
            move = moves[i]
            current[i] = translate(current[i], xoff=move[0], yoff=move[1])
            shifts[i] += move
            bounds[i, (0, 2)] += move[0]
            bounds[i, (1, 3)] += move[1]

        if not np.any(moved) or float(move_lengths_sq.max()) < tol * tol:
            break

    return current, [shift for shift in shifts]
