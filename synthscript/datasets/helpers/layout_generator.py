from copy import deepcopy

import cv2
import numpy as np
from shapely.affinity import translate

from synthscript.datasets.ocr_transform_pack import OCRTransformPack
from synthscript.ocr_units import OCRPage
from synthscript.shared.geometry_processing import get_union_rect
from synthscript.transforms import (
    LineTransform,
    PageTransform,
    ParagraphTransform,
)

Transform = LineTransform | ParagraphTransform | PageTransform


class LayoutGenerator:
    """Generate transformed OCRPage objects with an OCRTransformPack.

    Unlike OCRDataset, which applies its pack to each requested sample, a layout
    generator applies the pack to every line and builds a new page annotation
    from the returned crops and polygons.
    """

    def __init__(
        self,
        avoid_line_intersections: bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        # Retain the old argument name for compatibility. OCRTransformPack also
        # handles intersections between paragraphs.
        self._avoid_intersections = avoid_line_intersections
        self._transforms = OCRTransformPack(
            avoid_intersections=avoid_line_intersections,
            rng=rng,
        )

    @property
    def rng(self) -> np.random.Generator:
        return self._transforms.rng

    @rng.setter
    def rng(self, value: np.random.Generator | int | None) -> None:
        self._transforms.rng = value

    @property
    def transforms(self) -> OCRTransformPack | None:
        if self._transforms.is_identity:
            return None
        return self._transforms

    def add_transform(
        self,
        transform: Transform | None,
        probability: float = 1,
    ) -> None:
        """Append a transform, using the same interface as OCRDataset."""
        if transform is not None:
            self._transforms.add_transform(transform, probability)

    def set_transform(
        self,
        *transform_probability_pairs: tuple[Transform | None, float],
    ) -> None:
        """Replace all transforms, using the same interface as OCRDataset."""
        for transform, _ in transform_probability_pairs:
            if transform is not None and not isinstance(
                transform,
                (LineTransform, ParagraphTransform, PageTransform),
            ):
                raise ValueError(
                    "Only accepts LinewiseTransform, IntraparagraphTransform or "
                    f"InterparagraphTransform, got {type(transform)}"
                )

        self._transforms = OCRTransformPack(
            avoid_intersections=self._avoid_intersections,
            rng=self.rng,
        )
        for transform, probability in transform_probability_pairs:
            if probability != 0:
                self.add_transform(transform, probability)

    def apply(self, annotation: OCRPage) -> OCRPage:
        """Return a new page made from the transform pack's crops and polygons."""
        new_annotation = deepcopy(annotation)
        if not new_annotation.lines:
            return new_annotation

        lines_by_paragraph = [
            list(paragraph.lines) for paragraph in new_annotation.paragraphs
        ]
        paragraph_equivalents = [
            ([line.crop for line in lines], [line.polygon for line in lines])
            for lines in lines_by_paragraph
        ]

        crops, polygons = self._transforms(paragraph_equivalents)
        lines = [line for paragraph in lines_by_paragraph for line in paragraph]
        if len(crops) != len(lines) or len(polygons) != len(lines):
            raise ValueError("Layout transforms must preserve the number of OCR lines.")

        min_x, min_y, max_x, max_y = get_union_rect(polygons)
        polygons = [
            translate(polygon, xoff=-min_x, yoff=-min_y) for polygon in polygons
        ]

        for line, crop, polygon in zip(lines, crops, polygons, strict=True):
            line.crop = crop
            line.polygon = polygon

        width = max(1, int(np.ceil(max_x - min_x)) + 1)
        height = max(1, int(np.ceil(max_y - min_y)) + 1)
        new_annotation.background = cv2.resize(
            new_annotation.background,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        self._refresh_geometric_info(new_annotation)
        return new_annotation

    @staticmethod
    def _refresh_geometric_info(annotation: OCRPage) -> None:
        for paragraph in annotation.paragraphs:
            if not paragraph.lines:
                continue

            total_area = sum(line.polygon.area for line in paragraph.lines)
            if total_area:
                paragraph.avg_rotation = (
                    sum(line.rotation * line.polygon.area for line in paragraph.lines)
                    / total_area
                )

            shape = paragraph.lines[0].polygon
            for line in paragraph.lines[1:]:
                shape = shape.union(line.polygon)
            paragraph.centroid = (  # ty: ignore[invalid-assignment]
                shape.centroid.x,
                shape.centroid.y,
            )

            theta_rad = np.radians(paragraph.avg_rotation)
            cos_theta = float(np.cos(theta_rad))
            sin_theta = float(np.sin(theta_rad))
            cx_para, cy_para = paragraph.centroid
            for line in paragraph.lines:
                cx, cy = line.centroid()
                dx, dy = cx - cx_para, cy - cy_para
                line.corrected_centroid = (
                    dx * cos_theta - dy * sin_theta + cx_para,
                    dx * sin_theta + dy * cos_theta + cy_para,
                )

    # Compatibility alias for callers of the old public helper.
    refresh_annotations_geometric_info = _refresh_geometric_info
