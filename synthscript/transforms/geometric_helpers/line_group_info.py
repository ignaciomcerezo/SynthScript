from collections.abc import Collection, Sequence

import numpy as np
import shapely
from shapely import Polygon
from shapely.geometry import MultiPolygon

from synthscript.ocr_units import OCRLine, OCRParagraph
from synthscript.shared.geometry_processing import calculate_reading_angle
from synthscript.shared.parameters import Vector2D


class LineGroupInfo:
    """
    Some helpful data to calculate page layouts and OCRTransforms.
    """

    def __init__(
        self,
        line_group: OCRParagraph | Sequence[OCRLine],
    ):
        if not line_group:
            raise ValueError("Empty line group.")

        self.line_bounds: list[tuple[float, float, float, float]] = [
            line.polygon.bounds for line in line_group
        ]
        self.centers: list[Vector2D] = [
            np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) for b in self.line_bounds
        ]
        self.rotations = [line.rotation for line in line_group]

        total_area = sum(box.polygon.area for box in line_group)

        self.avg_rotation = (
            0.0
            if not total_area
            else sum(box.rotation * box.polygon.area for box in line_group) / total_area
        )

        self.union_bounds = (
            min(b[0] for b in self.line_bounds),
            min(b[1] for b in self.line_bounds),
            max(b[2] for b in self.line_bounds),
            max(b[3] for b in self.line_bounds),
        )

        self.center = (
            sum(self.union_bounds[::2]) / 2,
            sum(self.union_bounds[1::2]) / 2,
        )

        self._reading_direction: Vector2D | None = None
        self._orthogonal_direction: Vector2D | None = None

    @property
    def reading_direction(self) -> Vector2D:
        if self._reading_direction is None:
            self._reading_direction = self.compute_reading_direction(
                self.centers, self.avg_rotation
            )

        return self._reading_direction

    @property
    def orthogonal_direction(self) -> Vector2D:
        if self._orthogonal_direction is None:
            self._orthogonal_direction = self.compute_orthogonal_direction(
                self.centers, self.reading_direction
            )
        return self._orthogonal_direction

    @property
    def x0(self) -> float:
        return self.union_bounds[0]

    @property
    def xf(self) -> float:
        return self.union_bounds[2]

    @property
    def y0(self) -> float:
        return self.union_bounds[1]

    @property
    def yf(self) -> float:
        return self.union_bounds[3]

    @property
    def x0s(self) -> list[float]:
        return [bound[0] for bound in self.line_bounds]

    @property
    def xfs(self) -> list[float]:
        return [bound[2] for bound in self.line_bounds]

    @property
    def y0s(self) -> list[float]:
        return [bound[1] for bound in self.line_bounds]

    @property
    def yfs(self) -> list[float]:
        return [bound[3] for bound in self.line_bounds]

    @property
    def w(self) -> float:
        return abs(self.xf - self.x0)

    @property
    def h(self) -> float:
        return abs(self.yf - self.y0)

    @property
    def ws(self) -> list[float]:
        return [abs(x0 - xf) for (x0, xf) in zip(self.x0s, self.xfs, strict=True)]

    @property
    def hs(self) -> list[float]:
        return [abs(y0 - yf) for (y0, yf) in zip(self.y0s, self.yfs, strict=True)]

    @classmethod
    def from_polygons(
        cls, polygons: Sequence[Polygon | MultiPolygon]
    ) -> "LineGroupInfo":
        if not polygons:
            raise ValueError(
                "Cannot get geometric information from an empty sequence of polygons."
            )
        instance = object.__new__(cls)
        instance.line_bounds = [polygon.bounds for polygon in polygons]

        instance.centers = [
            np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            for b in instance.line_bounds
        ]

        instance.rotations = [calculate_reading_angle(poly) for poly in polygons]
        total_area = sum(polygon.area for polygon in polygons)
        instance.avg_rotation = (
            0.0
            if total_area == 0
            else sum(
                rotation * polygon.area
                for rotation, polygon in zip(instance.rotations, polygons, strict=True)
            )
            / total_area
        )

        if not instance.line_bounds:
            instance.union_bounds = (0.0, 0.0, 0.0, 0.0)
            instance.center = (0.0, 0.0)
        else:
            min_x = min(b[0] for b in instance.line_bounds)
            min_y = min(b[1] for b in instance.line_bounds)
            max_x = max(b[2] for b in instance.line_bounds)
            max_y = max(b[3] for b in instance.line_bounds)
            instance.union_bounds = (min_x, min_y, max_x, max_y)
            instance.center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

        instance.center = (
            (instance.x0 + instance.xf) / 2,
            (instance.y0 + instance.yf) / 2,
        )

        instance._reading_direction = None
        instance._orthogonal_direction = None

        return instance

    @staticmethod
    def polygon_union(
        polygons: Collection[Polygon | MultiPolygon],
    ) -> Polygon | MultiPolygon:
        snapped_geoms = [
            shapely.set_precision(shapely.make_valid(pol), grid_size=1e-6)
            for pol in polygons
        ]
        return shapely.unary_union(snapped_geoms)

    @staticmethod
    def compute_reading_direction(
        centers_sequence: Sequence[Vector2D],
        avg_rotation: float,
    ) -> Vector2D:
        default_direction = np.array([1.0, 0.0], dtype=float)

        if not np.isfinite(avg_rotation):
            return default_direction

        angle = np.radians(avg_rotation)
        reading_dir = np.array(
            [np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)],
            dtype=float,
        )

        if not np.all(np.isfinite(reading_dir)):
            return default_direction

        centers = np.asarray(centers_sequence, dtype=float)

        if centers.ndim != 2 or centers.shape[0] < 2 or centers.shape[1] != 2:
            return default_direction

        if not np.all(np.isfinite(centers)):
            return default_direction

        unique_centers = np.unique(centers, axis=0)
        if len(unique_centers) < 2:
            return default_direction

        projections_by_index = centers @ reading_dir

        if not np.all(np.isfinite(projections_by_index)):
            return default_direction

        x = np.arange(len(projections_by_index), dtype=float)

        try:
            slope = np.polyfit(x, projections_by_index, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            return default_direction

        if not np.isfinite(slope):
            return default_direction

        if slope < 0:
            reading_dir = -reading_dir

        return reading_dir

    @staticmethod
    def compute_orthogonal_direction(
        centers_sequence: Sequence[Vector2D], reading_dir: np.ndarray
    ) -> np.ndarray:
        perp = np.array([-reading_dir[1], reading_dir[0]], dtype=float)

        if len(centers_sequence) > 1:
            centers = np.array(centers_sequence, dtype=float)
            projections = centers @ perp
            slope = np.polyfit(np.arange(len(centers_sequence)), projections, 1)[0]

            if slope < 0:
                perp = -perp

        return perp

    @staticmethod
    def center_to_center_vector(
        poly_a: Polygon | MultiPolygon,
        poly_b: Polygon | MultiPolygon,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        c_a = LineGroupInfo.poly_center(poly_a)
        c_b = LineGroupInfo.poly_center(poly_b)

        v = c_b - c_a
        if direction is None:
            return v
        else:
            direction_norm = direction / np.linalg.norm(direction)
            return np.dot(v, direction_norm) * direction_norm

    @staticmethod
    def centroid_to_centroid_normalized_vector(
        poly_a: Polygon | MultiPolygon,
        poly_b: Polygon | MultiPolygon,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        v = LineGroupInfo.center_to_center_distance(poly_a, poly_b, direction)
        return v / np.linalg.norm(v)

    @staticmethod
    def center_to_center_distance(
        poly_a: Polygon | MultiPolygon,
        poly_b: Polygon | MultiPolygon,
        direction: np.ndarray | None = None,
        direction_is_normalized: bool = False,
    ) -> float:
        c_a = LineGroupInfo.poly_center(poly_a)
        c_b = LineGroupInfo.poly_center(poly_b)

        v = c_b - c_a
        if direction is None:
            return float(np.linalg.norm(v))
        else:
            if not direction_is_normalized:
                direction_norm = direction / np.linalg.norm(direction)
            else:
                direction_norm = direction
            return abs(np.dot(v, direction_norm))

    @staticmethod
    def poly_center(polygon: Polygon | MultiPolygon) -> Vector2D:
        return np.array((sum(polygon.bounds[::2]) / 2, sum(polygon.bounds[1::2]) / 2))
