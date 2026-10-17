from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from shapely import Polygon

from synthscript.shared.image_processing import crop_image_with_polygon

if TYPE_CHECKING:
    from synthscript.ocr_units.ocr_line import OCRLine
    from synthscript.ocr_units.ocr_paragraph import OCRParagraph


@dataclass
class _PlacedLine:
    line: OCRLine
    mask: np.ndarray
    labels: np.ndarray
    areas: np.ndarray
    x0: int
    y0: int


@dataclass(frozen=True, slots=True)
class LineImageSplitter:
    """Crop line polygons from a stroke image and conservatively resolve shared ink"""

    remove_shared_ink: bool = True
    ink_threshold: int = 10
    min_overlap_area: int = 5
    min_owner_extension_ratio: float = 1.5
    max_other_extension_ratio: float = 0.1

    def __post_init__(self) -> None:
        if not 0 <= self.ink_threshold < 255:
            raise ValueError("ink_threshold must be between 0 and 254.")
        if self.min_overlap_area < 1:
            raise ValueError("min_overlap_area must be positive.")
        if self.min_owner_extension_ratio < 1:
            raise ValueError("min_owner_extension_ratio must be at least 1.")
        if not 0 <= self.max_other_extension_ratio < 1:
            raise ValueError("max_other_extension_ratio must be in [0, 1).")

    def split(self, stroke: np.ndarray, polygons: list[Polygon]) -> list[np.ndarray]:
        return [crop_image_with_polygon(stroke, polygon) for polygon in polygons]

    def resolve(
        self, paragraphs: list[OCRParagraph], page_shape: tuple[int, int]
    ) -> int:
        """
        Remove owned overlap connected components from the other line's alpha channel
        """
        if not self.remove_shared_ink:
            return 0

        placed: dict[str, _PlacedLine] = {}
        coverage = np.zeros(page_shape, dtype=np.uint16)
        for paragraph in paragraphs:
            for line in paragraph.lines:
                if line.id in placed or line.crop.size == 0:
                    continue
                mask = (line.crop[..., 0] > self.ink_threshold) & (
                    line.crop[..., 3] > 0
                )
                _, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8
                )
                x0 = max(0, int(line.polygon.bounds[0]))
                y0 = max(0, int(line.polygon.bounds[1]))
                placed[line.id] = _PlacedLine(
                    line, mask, labels, stats[:, cv2.CC_STAT_AREA], x0, y0
                )
                height, width = mask.shape
                coverage[y0 : y0 + height, x0 : x0 + width] += mask

        pending = {
            line_id: np.zeros_like(data.mask) for line_id, data in placed.items()
        }
        kernel = np.ones((3, 3), dtype=np.uint8)
        removed_islands = 0

        for paragraph in paragraphs:
            for first_line, second_line in zip(
                paragraph.lines, paragraph.lines[1:], strict=False
            ):
                if first_line.id not in placed or second_line.id not in placed:
                    continue
                first = placed[first_line.id]
                second = placed[second_line.id]
                x0 = max(first.x0, second.x0)
                y0 = max(first.y0, second.y0)
                x1 = min(
                    first.x0 + first.mask.shape[1], second.x0 + second.mask.shape[1]
                )
                y1 = min(
                    first.y0 + first.mask.shape[0], second.y0 + second.mask.shape[0]
                )
                if x0 >= x1 or y0 >= y1:
                    continue

                first_slice = np.s_[
                    y0 - first.y0 : y1 - first.y0, x0 - first.x0 : x1 - first.x0
                ]
                second_slice = np.s_[
                    y0 - second.y0 : y1 - second.y0,
                    x0 - second.x0 : x1 - second.x0,
                ]
                overlap = first.mask[first_slice] & second.mask[second_slice]
                if not overlap.any():
                    continue

                count, overlap_labels, overlap_stats, _ = (
                    cv2.connectedComponentsWithStats(
                        overlap.astype(np.uint8), connectivity=8
                    )
                )
                first_exclusive = first.mask.copy()
                second_exclusive = second.mask.copy()
                first_exclusive[first_slice] &= ~second.mask[second_slice]
                second_exclusive[second_slice] &= ~first.mask[first_slice]
                _, first_exclusive_labels, first_stats, _ = (
                    cv2.connectedComponentsWithStats(
                        first_exclusive.astype(np.uint8), connectivity=8
                    )
                )
                _, second_exclusive_labels, second_stats, _ = (
                    cv2.connectedComponentsWithStats(
                        second_exclusive.astype(np.uint8), connectivity=8
                    )
                )
                pair_coverage = coverage[y0:y1, x0:x1]

                for component_number in range(1, count):
                    island = overlap_labels == component_number
                    if np.any(pair_coverage[island] != 2):
                        continue
                    first_ids = np.unique(first.labels[first_slice][island])
                    second_ids = np.unique(second.labels[second_slice][island])
                    if len(first_ids) != 1 or len(second_ids) != 1:
                        continue

                    first_seed = np.zeros_like(first.mask, dtype=np.uint8)
                    second_seed = np.zeros_like(second.mask, dtype=np.uint8)
                    first_seed[first_slice] = island
                    second_seed[second_slice] = island
                    first_neighbors = cv2.dilate(first_seed, kernel) > 0
                    second_neighbors = cv2.dilate(second_seed, kernel) > 0
                    first_touching = np.unique(first_exclusive_labels[first_neighbors])
                    second_touching = np.unique(
                        second_exclusive_labels[second_neighbors]
                    )
                    first_areas = [
                        int(first_stats[label, cv2.CC_STAT_AREA])
                        for label in first_touching
                        if label > 0
                    ]
                    second_areas = [
                        int(second_stats[label, cv2.CC_STAT_AREA])
                        for label in second_touching
                        if label > 0
                    ]
                    overlap_area = int(
                        overlap_stats[component_number, cv2.CC_STAT_AREA]
                    )
                    if overlap_area < self.min_overlap_area:
                        continue
                    first_wins = (
                        len(first_areas) == 1
                        and first_areas[0]
                        >= self.min_owner_extension_ratio * overlap_area
                        and second.areas[second_ids[0]] - overlap_area
                        <= self.max_other_extension_ratio * overlap_area
                    )
                    second_wins = (
                        len(second_areas) == 1
                        and second_areas[0]
                        >= self.min_owner_extension_ratio * overlap_area
                        and first.areas[first_ids[0]] - overlap_area
                        <= self.max_other_extension_ratio * overlap_area
                    )
                    if first_wins == second_wins:
                        continue

                    if first_wins:
                        pending[second.line.id][second_slice] |= island
                    else:
                        pending[first.line.id][first_slice] |= island
                    removed_islands += 1

        for line_id, removal in pending.items():
            placed[line_id].line.crop[..., 3][removal] = 0
        return removed_islands
