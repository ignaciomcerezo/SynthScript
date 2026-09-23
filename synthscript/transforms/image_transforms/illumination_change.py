import cv2
import numpy as np

from synthscript.shared.parameters import Parameter
from synthscript.transforms.transforms import GlobalImageTransform


class IlluminationChange(GlobalImageTransform):
    """
    Adition of lighting noise to the page, mimicking scanner shadows.
    """

    def __init__(
        self, intensity: Parameter | float = 0.05, relative_scale: Parameter | float = 3
    ):
        self.intensity = Parameter(intensity)
        self.relative_scale = Parameter(relative_scale)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        intensity = max(0.0, float(self.intensity()))
        relative_scale = max(0, float(self.relative_scale()))

        grid_height = max(2, int(np.ceil(relative_scale)) + 1)
        grid_width = max(2, int(np.ceil(relative_scale)) + 1)
        low_frequency_noise = self.rng.uniform(
            -1.0, 1.0, (grid_height, grid_width)
        ).astype(np.float32)
        illumination = cv2.resize(
            low_frequency_noise, (width, height), interpolation=cv2.INTER_CUBIC
        )
        illumination = 1.0 + intensity * illumination

        if image.ndim > 2:
            illumination = illumination[..., None]
        transformed = image.astype(np.float32) * illumination

        if np.issubdtype(image.dtype, np.integer):
            limits = np.iinfo(image.dtype)
            np.clip(transformed, limits.min, limits.max, out=transformed)

        return transformed.astype(image.dtype)
