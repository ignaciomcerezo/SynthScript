from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon

from synthscript.shared.parameters import (
    Parameter,
)
from synthscript.transforms.helpers.line_group_info import LineGroupInfo
from synthscript.transforms.transforms import (
    ParagraphFromLineTransform,
    line_group_equivalent_type,
)

_NOISES = Literal["linear", "wave", "from_amplitude_parameter", "zigzag"]
_PARAMETERS = Literal["period", "amplitude", "slope", "intercept"]
_NOISE2PARAMETERS: dict[_NOISES, list[_PARAMETERS]] = {
    "linear": ["slope", "intercept"],
    "wave": ["amplitude", "period"],
    "from_amplitude_parameter": ["amplitude"],
    "zigzag": ["amplitude"],
}

scalar = Parameter | float


def _centroid(poly: Polygon) -> tuple[float, float]:
    return poly.centroid.x, poly.centroid.y


@dataclass
class HorizontalMovement(ParagraphFromLineTransform):
    """
    Moves (adds noise) to the position of the lines. The noise is only added in the orthogonal
    direction to the reading axis.
    """

    def __init__(
        self,
        noise_type: _NOISES,
        period: scalar | None = None,
        amplitude: scalar | None = None,
        slope: scalar | None = None,
        intercept: scalar | None = 0,
        *,
        probability: float = 1,
    ):
        self.noise_type = noise_type
        self.period = Parameter(period) if period is not None else None
        self.amplitude = Parameter(amplitude) if amplitude is not None else None
        self.slope = Parameter(slope) if slope is not None else None
        self.intercept = Parameter(intercept) if intercept is not None else None
        self.type2map: dict[
            _NOISES, Callable[[list[Polygon], np.ndarray, np.ndarray], list[Polygon]]
        ] = {
            "linear": self._call_linear_polygons,
            "wave": self._call_wave_polygons,
            "from_amplitude_parameter": self._call_random_polygons,
            "zigzag": self._call_zigzag_polygons,
        }
        self.probability = probability
        self._validate_parameters()
        self.may_cause_intersections = True

    def __repr__(self) -> str:
        return super().__repr__()

    def _validate_parameters(self):
        needed_parameters = _NOISE2PARAMETERS[self.noise_type]

        for parameter_name in needed_parameters:
            att_value = self.__getattribute__(parameter_name)
            if att_value is None:
                raise ValueError(
                    f"Noise type {self.noise_type} requires parameter {parameter_name} to be given a value."
                )

    def _call_linear_polygons(
        self,
        polygons: list[Polygon],
        vertical_direction: np.ndarray,
        horizontal_direction: np.ndarray,
    ) -> list[Polygon]:

        first = polygons[0]
        new_polygons = []
        intercept = self.intercept()  # ty: ignore[call-non-callable]
        slope = self.slope()  # ty: ignore[call-non-callable]
        for polygon in polygons:
            distance_in_reading_dir = LineGroupInfo.center_to_center_distance(
                first,
                polygon,
                direction=vertical_direction,
                direction_is_normalized=True,
            )
            delta = intercept + slope * distance_in_reading_dir
            v = delta * horizontal_direction
            new_polygons.append(translate(polygon, v[0], v[1]))

        return new_polygons

    def _call_wave_polygons(
        self,
        polygons: list[Polygon],
        vertical_direction: np.ndarray,
        horizontal_direction: np.ndarray,
    ) -> list[Polygon]:

        first = polygons[0]
        new_polygons = []
        for polygon in polygons:
            distance_in_reading_dir = LineGroupInfo.center_to_center_distance(
                first,
                polygon,
                direction=vertical_direction,
                direction_is_normalized=True,
            )

            delta = self.amplitude() * np.cos(  # ty: ignore[call-non-callable]
                2 * np.pi * distance_in_reading_dir / self.period()  # ty: ignore[call-non-callable]
            )
            v = delta * horizontal_direction

            new_polygons.append(translate(polygon, v[0], v[1]))

        return new_polygons

    def _call_zigzag_polygons(
        self,
        polygons: list[Polygon],
        vertical_direction: np.ndarray,
        horizontal_direction: np.ndarray,
    ) -> list[Polygon]:
        _, y0 = _centroid(polygons[0])

        new_polygons = []

        for i, polygon in enumerate(polygons):
            amplitude = self.amplitude()  # ty: ignore[call-non-callable]

            delta = amplitude if i % 2 == 0 else -amplitude

            v = horizontal_direction * delta

            new_polygons.append(translate(polygon, xoff=v[0], yoff=v[1]))

        return new_polygons

    def _call_random_polygons(
        self,
        polygons: list[Polygon],
        vertical_direction: np.ndarray,
        horizontal_direction: np.ndarray,
    ) -> list[Polygon]:

        new_polygons = []
        for i, polygon in enumerate(polygons):
            v = horizontal_direction * self.amplitude()  # ty: ignore[call-non-callable]
            new_polygons.append(translate(polygon, xoff=v[0], yoff=v[1]))

        return new_polygons

    def __call__(
        self,
        line_equivalent_group: line_group_equivalent_type,
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        images, polygons = self._extract_polygons_and_images(line_equivalent_group)
        v = LineGroupInfo.from_polygons(polygons).reading_direction
        horizontal_direciton = np.array([[0, -1], [1, 0]]) @ v
        polygons = self.type2map[self.noise_type](polygons, v, horizontal_direciton)

        return images, polygons
