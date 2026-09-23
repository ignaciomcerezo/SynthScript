from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy

import numpy as np
import shapely
from shapely.geometry import Polygon

from synthscript.ocr_units import OCRLine, OCRParagraph
from synthscript.shared.parameters import RandomizedParameter, RNGInput

line_group_equivalent_type = (
    OCRParagraph
    | OCRParagraph
    | Sequence[OCRLine]
    | tuple[Sequence[np.ndarray], Sequence[Polygon]]
)


class _RandomizedTransform:
    @property
    def rng(self) -> np.random.Generator:
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()
        return self._rng

    @rng.setter
    def rng(self, value: RNGInput) -> None:
        self._rng = (
            value
            if isinstance(value, np.random.Generator)
            else np.random.default_rng(value)
        )
        for attribute in vars(self).values():
            if isinstance(attribute, (RandomizedParameter, _RandomizedTransform)):
                attribute.rng = self._rng


class OCRTransform(_RandomizedTransform, ABC):
    may_cause_intersections: bool

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LineTransform(OCRTransform):
    """
    Base transform to be applied over individual lines, as stretching
    a single line or distorting it.

    It should be called over a crop and its corresponding polygon.
    """

    @abstractmethod
    def __call__(
        self, image: np.ndarray, polygon: Polygon
    ) -> tuple[np.ndarray, shapely.Polygon]:
        raise NotImplementedError

    def bulk_transform(
        self,
        line_equivalent_group: line_group_equivalent_type,
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        new_imgs, new_polygons = [], []

        for img, poly in zip(
            *self._extract_polygons_and_images(line_equivalent_group), strict=True
        ):
            new_img, new_polygon = self(img, poly)
            new_imgs.append(new_img)
            new_polygons.append(new_polygon)

        return new_imgs, new_polygons

    def in_place(self, line: OCRLine) -> None:
        """
        Transforms the polygon and image of a Line instance in-place.
        """
        img, poly = self(line.crop, line.polygon)
        line.crop = img
        line.polygon = poly

    @staticmethod
    def _extract_polygons_and_images(
        line_equivalent_group: line_group_equivalent_type,
    ) -> tuple[list[np.ndarray], list[shapely.Polygon]]:
        if isinstance(line_equivalent_group, tuple) and isinstance(
            line_equivalent_group[0], list
        ):
            return (
                copy(line_equivalent_group[0]),
                copy(line_equivalent_group[1]),
            )  # ty: ignore[invalid-return-type]
        return (
            [
                line.crop  # ty: ignore[unresolved-attribute]
                for line in line_equivalent_group
            ],
            [
                line.polygon  # ty: ignore[unresolved-attribute]
                for line in line_equivalent_group
            ],
        )

    @staticmethod
    def _extract_polygon_and_image(line: OCRLine) -> tuple[np.ndarray, shapely.Polygon]:
        return line.crop, line.polygon


class ParagraphTransform(OCRTransform):
    """
    Base transform to be applied to a paragraph or group of lines, for example shears,
    paragraph rotations (both linewise and paragraphwise) or relative line movements.

    It is called over groups of stroke images with their respective polygons.
    Base class used to modify layouts for individual paragraphs.
    """

    @abstractmethod
    def __call__(
        self,
        line_equivalent_group: line_group_equivalent_type,
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        raise NotImplementedError

    @staticmethod
    def from_linewise(transform: LineTransform):
        return ParagraphFromLineTransform(transform)

    def in_place(
        self,
        line_group: OCRParagraph | Sequence[OCRLine],
    ) -> None:
        imgs, polys = self(line_group)
        for line, img, poly in zip(line_group, imgs, polys, strict=True):
            line.crop = img
            line.polygon = poly

    @staticmethod
    def _extract_polygons_and_images(
        line_equivalent_group: (
            OCRParagraph
            | Sequence[OCRLine]
            | tuple[Sequence[np.ndarray], Sequence[shapely.Polygon]]
        ),
    ) -> tuple[list[np.ndarray], list[shapely.Polygon]]:
        if isinstance(line_equivalent_group, tuple) and isinstance(
            line_equivalent_group[0], list
        ):
            return (
                copy(line_equivalent_group[0]),
                copy(line_equivalent_group[1]),
            )  # ty: ignore[invalid-return-type]
        return (
            [
                line.crop  # ty: ignore[unresolved-attribute]
                for line in line_equivalent_group
            ],
            [
                line.polygon  # ty: ignore[unresolved-attribute]
                for line in line_equivalent_group
            ],
        )

    @staticmethod
    def _extract_polygon_and_image(line: OCRLine) -> tuple[np.ndarray, shapely.Polygon]:
        return line.crop, line.polygon


class ParagraphFromLineTransform(ParagraphTransform):
    """
    LineTransform turned ParagraphTransform by applying it to all lines in a paragraph.
    """

    def __init__(self, transform: LineTransform):
        self._transform = transform

    def __call__(
        self,
        line_equivalent_group: line_group_equivalent_type,
    ) -> tuple[list[np.ndarray], list[shapely.Polygon]]:
        return self._transform.bulk_transform(line_equivalent_group)


class PageTransform(OCRTransform):
    """
    Base transform to be applied to a complete page divided in paragraphs.

    For example paragraphs separation, global page rotation, etc.
    """

    @abstractmethod
    def __call__(
        self,
        line_equivalent_groups: Sequence[
            OCRParagraph
            | Sequence[OCRLine]
            | tuple[Sequence[np.ndarray], Sequence[Polygon]]
        ],
    ) -> tuple[list[list[np.ndarray]], list[list[Polygon]]]:
        raise NotImplementedError

    def in_place(self, line_groups: Sequence[OCRParagraph | Sequence[OCRLine]]) -> None:
        img_groups, poly_groups = self(line_groups)

        for line_group, img_group, poly_group in zip(
            line_groups, img_groups, poly_groups, strict=True
        ):
            for line, img, poly in zip(line_group, img_group, poly_group, strict=True):
                line.crop = img
                line.polygon = poly

    @staticmethod
    def _extract_polygon_and_image_groups(
        line_equivalent_groups: Sequence[line_group_equivalent_type,],
    ) -> tuple[list[list[np.ndarray]], list[list[Polygon]]]:

        groups = [
            ParagraphTransform._extract_polygons_and_images(element)
            for element in line_equivalent_groups
        ]
        image_groups = []
        polygon_groups = []
        for group in groups:
            image_groups.append(group[0])
            polygon_groups.append(group[1])

        return image_groups, polygon_groups


class ImageTransform(_RandomizedTransform, ABC):
    may_cause_intersections = False

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class StrokeTransform(ImageTransform, ABC):
    """
    Stroke-only base image transform.
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BackgroundTransform(ImageTransform, ABC):
    """
    Background-only base image transform.
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GlobalImageTransform(ImageTransform, ABC):
    """
    Base transforms for modifying the complete image.
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
