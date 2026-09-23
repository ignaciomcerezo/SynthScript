from typing import Literal

import numpy as np

from synthscript.shared.parameters import DiscreteDistribution, Parameter
from synthscript.transforms.transforms import StrokeTransform

Direction = Literal["left", "right"]


class InkFade(StrokeTransform):
    """
    Ink fading in a specific direction.
    """

    def __init__(
        self,
        strength: Parameter | float = 0.2,
        direction: DiscreteDistribution[Direction] | Direction = "right",
    ):
        self.strength = Parameter(strength)
        self.direction = (
            direction
            if isinstance(direction, DiscreteDistribution)
            else DiscreteDistribution([direction], [1])
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        strength = np.clip(float(self.strength()), 0.0, 1.0)
        direction = self.direction()

        fade = np.linspace(1.0, 1.0 - strength, image.shape[1], dtype=np.float32)
        if direction == "left":
            fade = fade[::-1]
        elif direction != "right":
            raise ValueError(f"Unsupported fade direction: {direction!r}")

        shape = (1, image.shape[1]) + (1,) * (image.ndim - 2)
        faded = image.astype(np.float32) * fade.reshape(shape)
        return faded.astype(image.dtype)
