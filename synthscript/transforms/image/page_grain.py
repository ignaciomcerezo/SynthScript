import numpy as np

from synthscript.shared.parameters import Parameter
from synthscript.transforms.transforms import BackgroundTransform


class PageGrain(BackgroundTransform):
    """Adds fine Gaussian grain to the paper before strokes are composited.

    ``intensity`` is the noise standard deviation as a fraction of the image's
    value range (e.g. 0.02 gives a standard deviation of 5.1 for uint8 images).
    Floating-point images are expected in [0, 1]. Grain is shared across colour
    channels; an alpha channel, when present, is preserved. The input is not
    modified, and the result retains its shape and dtype.
    """

    def __init__(self, intensity: Parameter | float = 0.02, *, probability: float = 1):
        self.intensity = Parameter(intensity)
        self.probability = probability

    def __call__(self, image: np.ndarray) -> np.ndarray:
        intensity = max(0.0, float(self.intensity()))
        if intensity == 0 or image.size == 0:
            return image.copy()

        if np.issubdtype(image.dtype, np.integer):
            limits = np.iinfo(image.dtype)
            low, high = limits.min, limits.max
        elif np.issubdtype(image.dtype, np.floating):
            low, high = 0.0, 1.0
        else:
            raise ValueError("PageGrain requires an integer or floating-point image.")

        grain = self.rng.normal(0.0, intensity * (high - low), image.shape[:2])
        if image.ndim > 2:
            grain = grain[..., None]

        transformed = image.astype(np.float64) + grain
        np.clip(transformed, low, high, out=transformed)
        result = transformed.astype(image.dtype)
        if image.ndim == 3 and image.shape[2] == 4:
            result[..., 3] = image[..., 3]
        return result
