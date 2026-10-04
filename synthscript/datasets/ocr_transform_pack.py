import functools
import operator
from collections.abc import Iterator
from typing import Generic, TypeVar

import numpy as np
from shapely.geometry import Polygon

from synthscript.datasets.helpers.intersection_correction import (
    avoid_line_intersections,
    avoid_paragraph_intersections,
)
from synthscript.transforms import (
    LineTransform,
    PageTransform,
    ParagraphTransform,
)

OCRTransformType = LineTransform | ParagraphTransform | PageTransform
T = TypeVar("T", bound=OCRTransformType)


class _TransformProbabilityPair(Generic[T]):
    __slots__ = ("probability", "transform")

    def __init__(self, transform: T, probability: float):
        if (probability < 0) or (probability > 1):
            raise ValueError("The given probability must lie between 0 and 1.")
        self.transform = transform
        self.probability = probability


class OCRTransformPack:
    def __init__(
        self,
        avoid_intersections: bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        self._line: list[_TransformProbabilityPair[LineTransform]] = []
        self._paragraph: list[_TransformProbabilityPair[ParagraphTransform]] = []
        self._page: list[_TransformProbabilityPair[PageTransform]] = []

        self.avoid_intersections = avoid_intersections
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
        _TransformProbabilityPair[LineTransform]
        | _TransformProbabilityPair[ParagraphTransform]
        | _TransformProbabilityPair[PageTransform]
    ]:
        yield from self._line
        yield from self._paragraph
        yield from self._page

    def _all_transforms(
        self,
    ) -> list[OCRTransformType]:
        return [x.transform for x in self._all_tp_pairs()]

    @property
    def is_identity(self) -> bool:
        return sum(x.probability for x in self._all_tp_pairs()) == 0

    @property
    def may_cause_intersections(self) -> bool:
        return any(
            transform.may_cause_intersections for transform in self._all_transforms()
        )

    def add_transform(
        self,
        transform: OCRTransformType,
        probability: float = 1,
    ) -> None:
        """Append a supported OCR layout transform."""
        transform.rng = self._rng
        if isinstance(transform, LineTransform):
            self._line.append(_TransformProbabilityPair(transform, probability))
        elif isinstance(transform, ParagraphTransform):
            self._paragraph.append(_TransformProbabilityPair(transform, probability))
        elif isinstance(transform, PageTransform):
            self._page.append(_TransformProbabilityPair(transform, probability))
        else:
            raise ValueError(f"Unsupported OCR transform type {type(transform)}.")

    def should_call(self, probability: float) -> bool:
        return probability == 1 or self._rng.random() < probability

    def __call__(
        self,
        paragraph_eq_list: list[tuple[list[np.ndarray], list[Polygon]]],
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        """
        Takes a list of image-list/polygon-list pairs representing each paragraph's
        crops and polygons.
        """

        for i in range(len(paragraph_eq_list)):
            images, polygons = paragraph_eq_list[i]

            # Process each line
            for j in range(len(images)):
                cur_image = images[j]
                cur_polygon = polygons[j]
                for pair in self._line:
                    if self.should_call(pair.probability):
                        pair.transform.rng = self._rng
                        cur_image, cur_polygon = pair.transform(
                            cur_image, cur_polygon
                        )
                images[j] = cur_image
                polygons[j] = cur_polygon

            current_paragraph = (images, polygons)

            # Process paragraph-level transforms
            for pair in self._paragraph:
                if self.should_call(pair.probability):
                    pair.transform.rng = self._rng
                    current_paragraph = pair.transform(current_paragraph)

            paragraph_eq_list[i] = current_paragraph

        # Process interparagraph transforms
        for pair in self._page:
            if self.should_call(pair.probability):
                pair.transform.rng = self._rng
                paragraph_eq_list = list(
                    zip(*pair.transform(paragraph_eq_list), strict=True)
                )

        polys_by_par = [
            tuple_images_polygons[1] for tuple_images_polygons in paragraph_eq_list
        ]

        if (
            self.avoid_intersections
            and self.may_cause_intersections
            and not self.is_identity
        ):

            for (
                i,
                par_polys,
            ) in enumerate(polys_by_par):
                polys_by_par[i] = avoid_line_intersections(par_polys)

            if len(polys_by_par) > 1:
                polys_by_par = avoid_paragraph_intersections(polys_by_par)

        polygons = functools.reduce(operator.iadd, polys_by_par, [])

        crops: list[np.ndarray] = functools.reduce(
            operator.iadd, (paragraph_eq[0] for paragraph_eq in paragraph_eq_list), []
        )

        return crops, polygons
