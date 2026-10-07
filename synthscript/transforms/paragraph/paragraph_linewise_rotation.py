from typing import Literal

import cv2
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import Polygon

from synthscript.shared.parameters import Parameter
from synthscript.transforms.transforms import (
    ParagraphTransform,
    line_group_equivalent_type,
)


class ParagraphLinewiseRotation(ParagraphTransform):
    """
    Rotates the lines of a paragraph individually.
    """

    def __init__(
        self,
        absolute: float | Parameter,
        *,
        metric: Literal[
            "degrees",
            "pi radians",
            "radians",
        ] = "degrees",
        probability: float = 1,
    ):
        self._absolute = Parameter(absolute)
        self._metric = metric
        self.probability = probability
        self.may_cause_intersections = True

    def __call__(
        self,
        line_equivalent_group: line_group_equivalent_type,
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        images, polygons = self._extract_polygons_and_images(line_equivalent_group)

        match self._metric:
            case "degrees":
                rotation = float(self._absolute())

            case "radians":
                rotation = float(self._absolute()) / np.pi * 180.0

            case "pi radians":
                rotation = float(self._absolute()) * 180.0

            case _:
                raise ValueError(f"Unknown metric: {self._metric}")

        for i, (image, polygon) in enumerate(zip(images, polygons, strict=True)):
            orig_bounds = polygon.bounds

            x0, y0, x1, y1 = orig_bounds
            center = (
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
            )

            polygons[i] = self._rotate_poly(
                polygon,
                rotation,
                center,
            )

            images[i] = self._rotate_img(
                image,
                rotation,
                center,
                orig_bounds,
                polygons[i].bounds,
            )

        return images, polygons

    @staticmethod
    def _rotate_poly(
        poly: Polygon,
        angle: float,
        center: tuple[float, float],
    ) -> Polygon:
        return rotate(
            poly,
            angle,
            origin=center,
            use_radians=False,
        )

    @staticmethod
    def _rotate_img(
        image: np.ndarray,
        angle: float,
        center: tuple[float, float],
        orig_bounds: tuple[float, float, float, float],
        new_bounds: tuple[float, float, float, float],
    ) -> np.ndarray:

        orig_x0, orig_y0, _, _ = orig_bounds
        new_x0, new_y0, new_x1, new_y1 = new_bounds

        new_width = max(1, int(np.ceil(new_x1 - new_x0)))
        new_height = max(1, int(np.ceil(new_y1 - new_y0)))

        cx, cy = center
        theta = np.radians(angle)
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        tx = c * (orig_x0 - cx) - s * (orig_y0 - cy) + cx - new_x0
        ty = s * (orig_x0 - cx) + c * (orig_y0 - cy) + cy - new_y0

        affine_matrix = np.array(
            [
                [c, -s, tx],
                [s, c, ty],
            ],
            dtype=np.float32,
        )

        return cv2.warpAffine(
            image,
            affine_matrix,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
