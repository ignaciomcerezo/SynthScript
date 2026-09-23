import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon

from synthscript.shared.parameters import Parameter
from synthscript.transforms.helpers.line_group_info import LineGroupInfo
from synthscript.transforms.transforms import (
    ParagraphTransform,
    line_group_equivalent_type,
)


class ReadingDirectionClearance(ParagraphTransform):
    """
    Spreads lines of a paragraph apart in the reading direction.
    """

    def __init__(
        self,
        relative_size_increment: Parameter | float,
        add_probabilistic_noise: bool = False,
    ):

        self._relative = Parameter(relative_size_increment)
        self.noise = add_probabilistic_noise
        self.may_cause_intersections = True

    def __call__(
        self,
        line_equivalent_group: line_group_equivalent_type,
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        images, polygons = self._extract_polygons_and_images(line_equivalent_group)
        if len(polygons) < 2:
            raise ValueError("Paragraph too small")

        info = LineGroupInfo.from_polygons(polygons)

        reading_direction_norm = info.reading_direction / np.linalg.norm(
            info.reading_direction
        )

        size_in_reading_direction = LineGroupInfo.center_to_center_distance(
            polygons[0],
            polygons[-1],
            direction=reading_direction_norm,
            direction_is_normalized=True,
        )

        Delta = size_in_reading_direction * self._relative()

        delta_i = Delta / (len(polygons) - 1)

        new_images = []
        new_polygons = []

        for k, (image, polygon) in enumerate(zip(images, polygons, strict=True)):
            # -Delta moves upwards, as topmost vertex has the most negative y coordinate

            if k:
                perturbation = 1 if not self.noise else self.rng.random()
                displacement_norm = -Delta / 2 + delta_i * (k + perturbation - 2)
            else:
                displacement_norm = -Delta / 2

            displacement_vector = reading_direction_norm * displacement_norm
            new_polygons.append(
                translate(
                    polygon, xoff=displacement_vector[0], yoff=displacement_vector[1]
                )
            )
            new_images.append(image)

        return new_images, new_polygons
