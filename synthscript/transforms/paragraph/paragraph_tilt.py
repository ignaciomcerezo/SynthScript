from typing import Literal

import cv2
import numpy as np
from shapely.geometry import Polygon

from synthscript.shared.geometry_processing import calculate_reading_angle
from synthscript.shared.parameters import Parameter
from synthscript.transforms.helpers.line_group_info import LineGroupInfo, Vector2D
from synthscript.transforms.transforms import (
    ParagraphTransform,
    line_group_equivalent_type,
)


class ParagraphTilt(ParagraphTransform):
    def __init__(
        self,
        strength: Parameter | float = 0.2,
        tilt_axis: Literal["vertical", "horizontal"] = "horizontal",
        *,
        probability: float = 1,
    ):
        self.relative = Parameter(strength)
        self.probability = probability
        assert self.relative.is_bounded(-1, 1), (
            "The strength of the tilt must lie be between (-1, 1)."
        )
        self._tilt_horizontal = tilt_axis == "horizontal"
        self.may_cause_intersections = True

    def __call__(
        self,
        line_equivalent_group: line_group_equivalent_type,
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        images, polygons = self._extract_polygons_and_images(line_equivalent_group)
        images = list(images)
        polygons = list(polygons)

        if not polygons:
            return images, polygons

        all_coords = np.vstack([p.exterior.coords[:-1] for p in polygons]).astype(
            np.float32
        )
        rect = cv2.minAreaRect(all_coords)
        (cx, cy), _, _ = rect
        center = np.array([cx, cy], dtype=np.float64)
        points = cv2.boxPoints(rect).astype(np.float64)

        centers = [
            np.array(
                [(p.bounds[0] + p.bounds[2]) / 2.0, (p.bounds[1] + p.bounds[3]) / 2.0]
            )
            for p in polygons
        ]
        total_area = sum(p.area for p in polygons)
        avg_rotation = (
            0.0
            if total_area == 0
            else sum(calculate_reading_angle(p) * p.area for p in polygons) / total_area
        )

        reading_direction = np.asarray(
            LineGroupInfo.compute_reading_direction(centers, avg_rotation),
            dtype=np.float64,
        )
        orthogonal_direction = np.asarray(
            LineGroupInfo.compute_orthogonal_direction(centers, reading_direction),
            dtype=np.float64,
        )

        reading_norm = np.linalg.norm(reading_direction)
        orthogonal_norm = np.linalg.norm(orthogonal_direction)

        if reading_norm == 0 or orthogonal_norm == 0:
            raise ValueError("Geometry failed: invalid reading/orthogonal direction.")

        reading_direction /= reading_norm
        orthogonal_direction /= orthogonal_norm

        projected_points = [
            (
                np.dot(reading_direction, point - center),
                np.dot(orthogonal_direction, point - center),
                point,
            )
            for point in points
        ]

        projected_points.sort(key=lambda item: item[0])
        low_reading = projected_points[:2]
        high_reading = projected_points[2:]

        low_reading.sort(key=lambda item: item[1])
        high_reading.sort(key=lambda item: item[1])

        a = low_reading[0][2]
        b = low_reading[1][2]
        d = high_reading[0][2]
        c = high_reading[1][2]

        if not self._tilt_horizontal:
            a, b, c, d = a, d, b, c

        t = 0.5 * self.relative()
        a_moved, b_moved = self._symm_lerp(a, b, t)
        c_moved, d_moved = self._symm_lerp(c, d, -t)

        src_points = np.array([a, b, c, d], dtype=np.float32)
        dst_points = np.array([a_moved, b_moved, c_moved, d_moved], dtype=np.float32)

        H_global = cv2.getPerspectiveTransform(src_points, dst_points).astype(
            np.float64
        )

        for i, (image, polygon) in enumerate(zip(images, polygons, strict=True)):
            orig_bounds = polygon.bounds

            transformed_polygon = self._transform_polygon(polygon, H_global)
            polygons[i] = transformed_polygon

            trans_bounds = transformed_polygon.bounds
            orig_minx, orig_miny, _, _ = orig_bounds
            trans_minx, trans_miny, trans_maxx, trans_maxy = trans_bounds

            new_width = max(1, int(np.ceil(trans_maxx - trans_minx)))
            new_height = max(1, int(np.ceil(trans_maxy - trans_miny)))

            T_dst_inv = np.array(
                [
                    [1.0, 0.0, -trans_minx],
                    [0.0, 1.0, -trans_miny],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            T_src = np.array(
                [
                    [1.0, 0.0, orig_minx],
                    [0.0, 1.0, orig_miny],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            H_local = T_dst_inv @ H_global @ T_src

            warped_image = cv2.warpPerspective(
                image,
                H_local,
                (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            images[i] = warped_image

        return images, polygons

    @staticmethod
    def _symm_lerp(A: np.ndarray, B: np.ndarray, t: float) -> tuple[Vector2D, Vector2D]:
        return ((1.0 - t) * A + t * B, t * A + (1.0 - t) * B)

    @staticmethod
    def _transform_polygon(polygon: Polygon, H: np.ndarray) -> Polygon:
        ext_coords = np.asarray(polygon.exterior.coords, dtype=np.float64)
        ext_warped = cv2.perspectiveTransform(ext_coords.reshape(-1, 1, 2), H).reshape(
            -1, 2
        )

        if not polygon.interiors:
            return Polygon(ext_warped)

        int_warped = [
            cv2.perspectiveTransform(
                np.asarray(ring.coords, dtype=np.float64).reshape(-1, 1, 2), H
            ).reshape(-1, 2)
            for ring in polygon.interiors
        ]
        return Polygon(ext_warped, int_warped)
