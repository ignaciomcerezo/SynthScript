from collections.abc import Sequence

import numpy as np
from shapely.geometry import Polygon

from synthscript.shared.parameters import Parameter, TrimmedNormalDistribution
from synthscript.transforms.transforms import (
    PageTransform,
    line_group_equivalent_type,
)

from ._layout_helpers import (
    center_data,
    median_extention,
    ordered_layout_axes,
    paragraph_hulls,
    translate_polygon_group,
)


class ParagraphIndentationJitter(PageTransform):
    """
    Adds smooth horizontal movement to the paragraphs using random walks.
    """

    def __init__(
        self,
        relative_step: Parameter | float | None = None,
        *,
        probability: float = 1,
    ):
        if relative_step is None:
            relative_step = TrimmedNormalDistribution(
                clip_low=-0.12,
                clip_high=0.12,
                mean=0.0,
                sigma=0.04,
            )
        self._relative_step = Parameter(relative_step)
        self.probability = probability
        self.may_cause_intersections = True

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
        _, _, orthogonal_direction = ordered_layout_axes(hulls)
        reference_width = median_extention(hulls, orthogonal_direction)

        offsets = np.zeros(len(hulls), dtype=float)
        for i in range(1, len(hulls)):
            offsets[i] = offsets[i - 1] + self._relative_step() * reference_width

        offsets = center_data(offsets)
        new_polygon_groups = [
            translate_polygon_group(polygons, orthogonal_direction * offset)
            for polygons, offset in zip(polygon_groups, offsets, strict=True)
        ]
        return image_groups, new_polygon_groups
