from collections.abc import Iterator
from typing import Generic, TypeVar

import numpy as np

from synthscript.transforms import (
    BackgroundTransform,
    GlobalImageTransform,
    StrokeTransform,
)
from synthscript.transforms.transforms import ImageTransform

ImageTransformType = StrokeTransform | BackgroundTransform | GlobalImageTransform
T = TypeVar("T", bound=ImageTransformType)


class _TransformProbabilityPair(Generic[T]):
    __slots__ = ("probability", "transform")

    def __init__(self, transform: T, probability: float):
        if (probability < 0) or (probability > 1):
            raise ValueError("The given probability must lie between 0 and 1.")
        self.transform = transform
        self.probability = probability


class ImageTransformPack:
    """Groups transforms by the image-composition stage they operate on."""

    def __init__(self, rng: np.random.Generator | int | None = None):
        self._stroke: list[_TransformProbabilityPair[StrokeTransform]] = []
        self._background: list[_TransformProbabilityPair[BackgroundTransform]] = []
        self._global_image: list[
            _TransformProbabilityPair[GlobalImageTransform]
        ] = []
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

    def _all_tp_pairs(
        self,
    ) -> Iterator[
        _TransformProbabilityPair[StrokeTransform]
        | _TransformProbabilityPair[BackgroundTransform]
        | _TransformProbabilityPair[GlobalImageTransform]
    ]:
        yield from self._stroke
        yield from self._background
        yield from self._global_image

    def _all_transforms(self) -> list[ImageTransformType]:
        return [pair.transform for pair in self._all_tp_pairs()]

    @property
    def is_identity(self) -> bool:
        return sum(pair.probability for pair in self._all_tp_pairs()) == 0

    def _should_call(self, probability: float) -> bool:
        return probability == 1 or self._rng.random() < probability

    def add_transform(self, transform: ImageTransform, probability: float = 1) -> None:
        transform.rng = self._rng
        if isinstance(transform, StrokeTransform):
            self._stroke.append(_TransformProbabilityPair(transform, probability))
        elif isinstance(transform, BackgroundTransform):
            self._background.append(_TransformProbabilityPair(transform, probability))
        elif isinstance(transform, GlobalImageTransform):
            self._global_image.append(
                _TransformProbabilityPair(transform, probability)
            )
        else:
            raise ValueError(f"Unsupported image transform type {type(transform)}.")

    def transform_strokes(self, strokes: list[np.ndarray]) -> list[np.ndarray]:
        for index, stroke in enumerate(strokes):
            for pair in self._stroke:
                if self._should_call(pair.probability):
                    pair.transform.rng = self._rng
                    stroke = pair.transform(stroke)
            strokes[index] = stroke
        return strokes

    def transform_background(self, background: np.ndarray) -> np.ndarray:
        for pair in self._background:
            if self._should_call(pair.probability):
                pair.transform.rng = self._rng
                background = pair.transform(background)
        return background

    def transform_global_image(self, image: np.ndarray) -> np.ndarray:
        for pair in self._global_image:
            if self._should_call(pair.probability):
                pair.transform.rng = self._rng
                image = pair.transform(image)
        return image
