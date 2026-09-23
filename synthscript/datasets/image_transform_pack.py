import numpy as np

from synthscript.transforms import (
    BackgroundTransform,
    GlobalImageTransform,
    StrokeTransform,
)
from synthscript.transforms.transforms import ImageTransform


class ImageTransformPack:
    """Groups transforms by the image-composition stage they operate on."""

    def __init__(self, rng: np.random.Generator | int | None = None):
        self._stroke: list[StrokeTransform] = []
        self._stroke_prob: list[float] = []
        self._background: list[BackgroundTransform] = []
        self._background_prob: list[float] = []
        self._global_image: list[GlobalImageTransform] = []
        self._global_image_prob: list[float] = []
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
        for transform in self._stroke + self._background + self._global_image:
            transform.rng = self._rng

    @property
    def is_identity(self) -> bool:
        return (
            sum(self._stroke_prob)
            + sum(self._background_prob)
            + sum(self._global_image_prob)
        ) == 0

    def _should_call(self, probability: float) -> bool:
        return probability == 1 or self._rng.random() < probability

    def add_transform(self, transform: ImageTransform, probability: float = 1) -> None:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be between 0 and 1")

        transform.rng = self._rng
        if isinstance(transform, StrokeTransform):
            self._stroke.append(transform)
            self._stroke_prob.append(probability)
        elif isinstance(transform, BackgroundTransform):
            self._background.append(transform)
            self._background_prob.append(probability)
        elif isinstance(transform, GlobalImageTransform):
            self._global_image.append(transform)
            self._global_image_prob.append(probability)
        else:
            raise ValueError(f"Unsupported image transform type {type(transform)}.")

    def transform_strokes(self, strokes: list[np.ndarray]) -> list[np.ndarray]:
        for index, stroke in enumerate(strokes):
            for transform, probability in zip(
                self._stroke, self._stroke_prob, strict=True
            ):
                if self._should_call(probability):
                    transform.rng = self._rng
                    stroke = transform(stroke)
            strokes[index] = stroke
        return strokes

    def transform_background(self, background: np.ndarray) -> np.ndarray:
        for transform, probability in zip(
            self._background, self._background_prob, strict=True
        ):
            if self._should_call(probability):
                transform.rng = self._rng
                background = transform(background)
        return background

    def transform_global_image(self, image: np.ndarray) -> np.ndarray:
        for transform, probability in zip(
            self._global_image, self._global_image_prob, strict=True
        ):
            if self._should_call(probability):
                transform.rng = self._rng
                image = transform(image)
        return image
