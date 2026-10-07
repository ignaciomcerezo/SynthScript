import cv2
import numpy as np

from synthscript.shared.parameters import Parameter
from synthscript.transforms.transforms import StrokeTransform


class DirectionalBlur(StrokeTransform):
    """
    Anisotropic gaussian blur, mimicking movement of the page.
    """

    def __init__(
        self,
        sigma_x: Parameter | float = 1,
        sigma_y: Parameter | float = 0,
        angle: Parameter | float = 0,
        *,
        probability: float = 1,
    ):
        self.sigma_x = Parameter(sigma_x)
        self.sigma_y = Parameter(sigma_y)
        self.angle = Parameter(angle)
        self.probability = probability

    def __call__(self, image: np.ndarray) -> np.ndarray:
        sigma_x = max(0.0, float(self.sigma_x()))
        sigma_y = max(0.0, float(self.sigma_y()))
        if sigma_x == 0.0 and sigma_y == 0.0:
            return image

        angle = np.deg2rad(float(self.angle()))
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)

        r_x = int(np.ceil(3.0 * np.hypot(sigma_x * cos_angle, sigma_y * sin_angle)))
        r_y = int(np.ceil(3.0 * np.hypot(sigma_x * sin_angle, sigma_y * cos_angle)))
        x = np.arange(-r_x, r_x + 1, dtype=np.float64)
        y = np.arange(-r_y, r_y + 1, dtype=np.float64)[:, None]
        rotated_x = x * cos_angle + y * sin_angle
        rotated_y = -x * sin_angle + y * cos_angle

        exponent = np.zeros((y.size, x.size), dtype=np.float64)
        if sigma_x > 0.0:
            exponent += (rotated_x / sigma_x) ** 2
        else:
            exponent[~np.isclose(rotated_x, 0.0, atol=1e-12)] = np.inf
        if sigma_y > 0.0:
            exponent += (rotated_y / sigma_y) ** 2
        else:
            exponent[~np.isclose(rotated_y, 0.0, atol=1e-12)] = np.inf

        kernel = np.exp(-0.5 * exponent)
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
