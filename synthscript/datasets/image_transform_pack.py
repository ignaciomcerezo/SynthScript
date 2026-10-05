import numpy as np

from synthscript.transforms import (
    BackgroundTransform,
    GlobalImageTransform,
    StrokeTransform,
)
from synthscript.transforms.transforms import ImageTransform

ImageTransformType = StrokeTransform | BackgroundTransform | GlobalImageTransform


class ImageTransformPack:
    """Groups transforms by the image-composition stage they operate on."""

    def __init__(self, rng: np.random.Generator | int | None = None):
        self._stroke: list[StrokeTransform] = []
        self._background: list[BackgroundTransform] = []
        self._global_image: list[GlobalImageTransform] = []
        self.rng = rng

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator | int | None) -> None:
        self._rng = (
            value
            if isinstance(value, np.random.Generator)
            else np.random.default_rng(value)
        )
        for transform in self._all_transforms():
            transform.rng = self._rng

    def _all_transforms(self) -> list[ImageTransformType]:
        return [*self._stroke, *self._background, *self._global_image]

    @property
    def is_identity(self) -> bool:
        return all(transform.probability == 0 for transform in self._all_transforms())

    def add_transform(self, transform: ImageTransform) -> None:
        transform.rng = self._rng
        if isinstance(transform, StrokeTransform):
            self._stroke.append(transform)
        elif isinstance(transform, BackgroundTransform):
            self._background.append(transform)
        elif isinstance(transform, GlobalImageTransform):
            self._global_image.append(transform)
        else:
            raise ValueError(f"Unsupported image transform type {type(transform)}.")

    def transform_strokes(self, strokes: list[np.ndarray]) -> list[np.ndarray]:
        for index, stroke in enumerate(strokes):
            for transform in self._stroke:
                if transform.should_apply():
                    stroke = transform(stroke)
            strokes[index] = stroke
        return strokes

    def transform_background(self, background: np.ndarray) -> np.ndarray:
        for transform in self._background:
            if transform.should_apply():
                background = transform(background)
        return background

    def transform_global_image(self, image: np.ndarray) -> np.ndarray:
        for transform in self._global_image:
            if transform.should_apply():
                image = transform(image)
        return image
