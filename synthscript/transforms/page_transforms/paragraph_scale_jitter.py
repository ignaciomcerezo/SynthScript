from collections.abc import Sequence

import cv2
import numpy as np
from shapely.affinity import scale
from shapely.geometry import Polygon

from synthscript.shared.parameters import Parameter, TrimmedNormalDistribution
from synthscript.transforms.transforms import (
    PageTransform,
    line_group_equivalent_type,
)

from ._layout_helpers import paragraph_hulls, polygon_center


class ParagraphScaleJitter(PageTransform):
    """
    Rescales each paragraph independently from each other.
    """

    def __init__(
        self,
        scale_factor: Parameter | float | None = None,
        *,
        probability: float = 1,
    ):
        if scale_factor is None:
            scale_factor = TrimmedNormalDistribution(
                clip_low=0.85,
                clip_high=1.15,
                mean=1.0,
                sigma=0.05,
            )
        self._scale_factor = Parameter(scale_factor)
        self.probability = probability
        low, _ = self._scale_factor.bounds
        if low <= 0:
            raise ValueError("scale_factor must be bounded strictly above zero.")
        self.may_cause_intersections = True

    def __call__(
        self,
        line_equivalent_groups: Sequence[line_group_equivalent_type],
    ) -> tuple[list[list[np.ndarray]], list[list[Polygon]]]:
        image_groups, polygon_groups = self._extract_polygon_and_image_groups(
            line_equivalent_groups
        )
        if not polygon_groups:
            return image_groups, polygon_groups

        hulls = paragraph_hulls(polygon_groups)
        new_image_groups: list[list[np.ndarray]] = []
        new_polygon_groups: list[list[Polygon]] = []

        for images, polygons, hull in zip(
            image_groups, polygon_groups, hulls, strict=True
        ):
            factor = self._scale_factor()
            center = tuple(polygon_center(hull))

            scaled_polygons = [
                scale(
                    polygon,
                    xfact=factor,
                    yfact=factor,
                    origin=center,
                )
                for polygon in polygons
            ]

            interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
            scaled_images: list[np.ndarray] = []
            for image in images:
                height, width = image.shape[:2]
                new_width = max(1, round(width * factor))
                new_height = max(1, round(height * factor))
                scaled_images.append(
                    cv2.resize(
                        image,
                        (new_width, new_height),
                        interpolation=interpolation,
                    )
                )

            new_image_groups.append(scaled_images)
            new_polygon_groups.append(scaled_polygons)

        return new_image_groups, new_polygon_groups
