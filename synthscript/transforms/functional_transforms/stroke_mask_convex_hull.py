import cv2
import numpy as np
from shapely.geometry import Polygon

from synthscript.transforms.transforms import LineTransform


class StrokeMaskConvexHull(LineTransform):
    """
    Replaces a polygon with the convex hull of the image mask.
    It always holds that, under expected image processing, this convex hull
    lies entirely inside the original polygon.
    """

    def __init__(self, *, probability: float = 1):
        self.probability = probability
        self.may_cause_intersections = False
        # it only may maintains them if they exists already

    def __call__(
        self, image: np.ndarray, polygon: Polygon
    ) -> tuple[np.ndarray, Polygon]:
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError("StrokeMaskConvexHull requires a BGRA stroke crop.")

        ink = (image[..., 0] > 0) & (image[..., 3] > 0)
        points = cv2.findNonZero(ink.astype(np.uint8))
        if points is None:
            return image, polygon
        points = points.reshape(-1, 2)

        # we hull twice: the first one uses all pixels centers in the original image
        pixel_hull = cv2.convexHull(points).reshape(-1, 2)
        # the second one offsets only the hull-incumbent pixels by 1 in each direction
        # and make a second hull. this saves us a hull with all quadrupled points.
        corners = pixel_hull[:, None, :] + np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32
        )
        hull = cv2.convexHull(corners.reshape(-1, 2)).reshape(-1, 2)

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        origin_x, origin_y = map(int, polygon.bounds[:2])
        global_hull = hull + np.array([origin_x, origin_y])

        return (
            image[y_min : y_max + 1, x_min : x_max + 1].copy(),
            Polygon(global_hull),
        )
