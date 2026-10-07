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
    ordered_layout_axes,
    paragraph_hulls,
    translate_polygon_group,
)


class ParagraphSpacingJitter(PageTransform):
    """
    Adds vertical traslations between consecutive paragraphs.
    """

    def __init__(
        self,
        relative_gap_noise: Parameter | float | None = None,
        *,
        probability: float = 1,
    ):
        if relative_gap_noise is None:
            relative_gap_noise = TrimmedNormalDistribution(
                clip_low=-0.35,
                clip_high=0.35,
                mean=0.0,
                sigma=0.12,
            )
        self._relative_gap_noise = Parameter(relative_gap_noise)
        self.probability = probability
        low, _ = self._relative_gap_noise.bounds
        if low <= -1.0:
            raise ValueError(
                "relative_gap_noise must greater than -1 to preserve paragraph "
                "ordering."
            )
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
        centers, reading_direction, _ = ordered_layout_axes(hulls)

        cumulative_shift = np.zeros(len(hulls), dtype=float)
        for i in range(1, len(hulls)):
            original_gap = float(np.dot(centers[i] - centers[i - 1], reading_direction))
            if original_gap <= 0:
                original_gap = float(np.linalg.norm(centers[i] - centers[i - 1]))
            if original_gap <= 0:
                continue

            relative_change = self._relative_gap_noise()
            new_gap = original_gap * (1.0 + relative_change)
            cumulative_shift[i] = cumulative_shift[i - 1] + (new_gap - original_gap)

        cumulative_shift = center_data(cumulative_shift)
        new_polygon_groups = [
            translate_polygon_group(polygons, reading_direction * shift)
            for polygons, shift in zip(polygon_groups, cumulative_shift, strict=True)
        ]
        return image_groups, new_polygon_groups
