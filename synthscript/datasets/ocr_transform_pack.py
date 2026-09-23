import functools
import operator

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

Transform = LineTransform | ParagraphTransform | PageTransform


class OCRTransformPack:
    def __init__(
        self,
        avoid_intersections: bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        self._linewise: list[LineTransform] = []
        self._linewise_prob: list[float] = []
        self._intra: list[ParagraphTransform] = []
        self._intra_prob: list[float] = []
        self._inter: list[PageTransform] = []
        self._inter_prob: list[float] = []
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

    def _all_transforms(
        self,
    ) -> list[Transform]:
        return self._linewise + self._intra + self._inter

    @property
    def is_identity(self) -> bool:
        return (
            sum(self._intra_prob) + sum(self._inter_prob) + sum(self._linewise_prob)
        ) == 0

    @property
    def may_cause_intersections(self) -> bool:
        return any(
            transform.may_cause_intersections for transform in self._all_transforms()
        )

    def add_transform(
        self,
        transform: Transform,
        probability: float = 1,
    ):
        """Append a supported OCR layout transform."""
        if (probability > 1) or (probability < 0):
            raise ValueError("probability must be between 0 and 1")
        if isinstance(transform, LineTransform):
            self._linewise.append(transform)
            self._linewise_prob.append(probability)
        elif isinstance(transform, ParagraphTransform):
            self._intra.append(transform)
            self._intra_prob.append(probability)
        elif isinstance(transform, PageTransform):
            self._inter.append(transform)
            self._inter_prob.append(probability)
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
                for linewise_transform, p in zip(
                    self._linewise, self._linewise_prob, strict=True
                ):
                    if self.should_call(p):
                        cur_image, cur_polygon = linewise_transform(
                            cur_image, cur_polygon
                        )
                images[j] = cur_image
                polygons[j] = cur_polygon

            current_paragraph = (images, polygons)

            # Process paragraph-level transforms
            for intraparagraph_transform, p in zip(
                self._intra, self._intra_prob, strict=True
            ):
                if self.should_call(p):
                    current_paragraph = intraparagraph_transform(current_paragraph)

            paragraph_eq_list[i] = current_paragraph

        # Process interparagraph transforms
        for interparagraph, p in zip(self._inter, self._inter_prob, strict=True):
            if self.should_call(p):
                paragraph_eq_list = list(
                    zip(*interparagraph(paragraph_eq_list), strict=True)
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
