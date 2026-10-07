import cv2
import numpy as np
from shapely import Polygon
from shapely.affinity import rotate

from synthscript.shared.parameters import Parameter
from synthscript.transforms.transforms import (
    ParagraphTransform,
    line_group_equivalent_type,
)


class ParagraphwiseRotation(ParagraphTransform):
    """
    Rotates a whole paragraph around its centroid.
    """

    def __init__(
        self,
        absolute: Parameter | float,
        *,
        metric: str = "degrees",
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

        if not polygons:
            return images, polygons

        min_x = min(p.bounds[0] for p in polygons)
        min_y = min(p.bounds[1] for p in polygons)
        max_x = max(p.bounds[2] for p in polygons)
        max_y = max(p.bounds[3] for p in polygons)
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

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
        img_array: np.ndarray,
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
            img_array,
            affine_matrix,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
