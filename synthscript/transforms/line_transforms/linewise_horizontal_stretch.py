import cv2
import numpy as np
from shapely.affinity import scale
from shapely.geometry import Polygon

from synthscript.shared.parameters import Parameter
from synthscript.transforms.transforms import LineTransform


class LinewiseHorizontalStretch(LineTransform):
    def __init__(
        self, scale_factor: Parameter | float = 1.2, *, probability: float = 1
    ):

        self.scale_factor = Parameter(scale_factor)
        self.probability = probability
        self.may_cause_intersections = True

    def __call__(
        self, image: np.ndarray, polygon: Polygon
    ) -> tuple[np.ndarray, Polygon]:
        instanciated_scale_factor = self.scale_factor()
        height, width = image.shape[:2]
        new_width = max(1, round(width * abs(instanciated_scale_factor)))
        image = image
        stretched_image = cv2.resize(
            image,
            (new_width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        min_x, min_y, max_x, max_y = polygon.bounds
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        stretched_polygon = scale(
            polygon,
            xfact=instanciated_scale_factor,
            yfact=1.0,
            origin=center,
        )

        return stretched_image, stretched_polygon
