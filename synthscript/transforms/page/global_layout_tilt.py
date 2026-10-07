from collections.abc import Sequence

import numpy as np
from shapely.geometry import Polygon

from synthscript.shared.parameters import Parameter, TrimmedNormalDistribution
from synthscript.transforms.transforms import (
    PageTransform,
    line_group_equivalent_type,
)

from ._layout_helpers import (
    ordered_layout_axes,
    paragraph_hulls,
    translate_polygon_group,
)


class GlobalLayoutTilt(PageTransform):
    """
    Translates paragraphs to place them in a sloped layout. The paragraph orientation
    remains invariant.
    """

    def __init__(
        self,
        angle_degrees: Parameter | float | None = None,
        *,
        probability: float = 1,
    ):
        if angle_degrees is None:
            angle_degrees = TrimmedNormalDistribution(
                clip_low=-7.5,
                clip_high=7.5,
                mean=0.0,
                sigma=2.5,
            )
        self._angle_degrees = Parameter(angle_degrees)
        self.probability = probability
        low, high = self._angle_degrees.bounds
        if low <= -89.0 or high >= 89.0:
            raise ValueError("angle_degrees must be less, in absolute value, than 89.")

        self.may_cause_intersections = False

    def __call__(
        self,
        line_equivalent_groups: Sequence[line_group_equivalent_type],
    ) -> tuple[list[list[np.ndarray]], list[list[Polygon]]]:
        image_groups, polygon_groups = self._extract_polygon_and_image_groups(
            line_equivalent_groups
        )
        if len(polygon_groups) < 2:
            return image_groups, polygon_groups

        hulls = paragraph_hulls(polygon_groups)
        centers, reading_direction, orthogonal_direction = ordered_layout_axes(hulls)

        coordinates = centers @ reading_direction
        centered_coordinates = coordinates - float(np.mean(coordinates))
        slope = float(np.tan(np.deg2rad(self._angle_degrees())))
        offsets = slope * centered_coordinates

        new_polygon_groups = [
            translate_polygon_group(polygons, orthogonal_direction * offset)
            for polygons, offset in zip(polygon_groups, offsets, strict=True)
        ]
        return image_groups, new_polygon_groups
