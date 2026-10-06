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
from synthscript.transforms.transforms import OCRTransform


class OCRTransformPack:
    def __init__(
        self,
        avoid_intersections: bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        self._line: list[LineTransform] = []
        self._paragraph: list[ParagraphTransform] = []
        self._page: list[PageTransform] = []

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

    def _all_transforms(self) -> list[OCRTransform]:
        return [*self._line, *self._paragraph, *self._page]

    @property
    def is_identity(self) -> bool:
        return all(transform.probability == 0 for transform in self._all_transforms())

    @property
    def may_cause_intersections(self) -> bool:
        return any(
            transform.may_cause_intersections for transform in self._all_transforms()
        )

    def add_transform(
        self,
        transform: OCRTransform,
    ) -> None:
        """Append a supported OCR layout transform."""
        transform.rng = self._rng
        if isinstance(transform, LineTransform):
            self._line.append(transform)
        elif isinstance(transform, ParagraphTransform):
            self._paragraph.append(transform)
        elif isinstance(transform, PageTransform):
            self._page.append(transform)
        else:
            raise ValueError(f"Unsupported OCR transform type {type(transform)}.")

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
                for transform in self._line:
                    if transform.should_apply():
                        cur_image, cur_polygon = transform(cur_image, cur_polygon)
                images[j] = cur_image
                polygons[j] = cur_polygon

            current_paragraph = (images, polygons)

            # Process paragraph-level transforms
            for transform in self._paragraph:
                if transform.should_apply():
                    current_paragraph = transform(current_paragraph)

            paragraph_eq_list[i] = current_paragraph

        # Process interparagraph transforms
        for transform in self._page:
            if transform.should_apply():
                paragraph_eq_list = list(
                    zip(*transform(paragraph_eq_list), strict=True)
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
