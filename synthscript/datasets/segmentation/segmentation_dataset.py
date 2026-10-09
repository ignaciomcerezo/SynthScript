from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from synthscript.datasets.base_annotation_dataset import (
    BaseAnnotationDataset,
    ClusterParams,
    RNGInput,
)
from synthscript.datasets.image_transform_pack import ImageTransformPack
from synthscript.datasets.ocr_transform_pack import OCRTransformPack
from synthscript.datasets.segmentation.formatters import _formatter_type
from synthscript.ocr_units import OCRPage


class SegmentationDataset(BaseAnnotationDataset):
    """
    Dataset variant intended to be used for line segmentation training. Takes as input a
    sequence of annotations (of type AnnotatedPage). Sampling orders must be assigned
    through .orders before requesting an item.

    When an item is requested, the dataset deterministically chooses an item (taken from
    all possible contiguous clusters of lines of length one of the orders provided),
    transforms it using the transform given (via .set_transform()) and returns the crop
    and line polygons.
    """

    def __init__(
        self,
        annotations: Sequence[OCRPage],
        *,
        rng: RNGInput = None,
        return_bounding_boxes: bool = True,
        cluster_transform_params: ClusterParams | None = None,
    ):
        self._annotated_pages = annotations
        self._orders: list[int] | None = None
        self._use_paragraphs = False
        self._use_full_pages = False
        self.rng = rng
        self._transforms: OCRTransformPack = OCRTransformPack(rng=self.rng)
        self._image_transforms = ImageTransformPack(rng=self.rng)
        self._recalculate_size_and_sampling_params()
        self.return_bounding_boxes = return_bounding_boxes
        self._formatter: _formatter_type | None = None

        self._cluster_params = (
            replace(ClusterParams(), **asdict(cluster_transform_params))
            if cluster_transform_params is not None
            else ClusterParams()
        )
        self._transforms.avoid_intersections = self._cluster_params.avoid_intersections
        self._sync_renderer()

    def __repr__(self):
        return (
            f"<OCRDataset ({len(self)} samples: {len(self._annotated_pages)}"
            f" pages using orders {self.orders})>"
        )

    def __getitem__(self, index: int) -> Any:
        image, polygons = self.getitem_no_formatter(index)
        if self._formatter is None:
            return image, polygons
        return self._formatter(image, polygons, index)

    def getitem_no_formatter(self, index: int) -> tuple[np.ndarray, list[Polygon]]:
        """
        Chooses a line cluster/paragraph/full page according to the available orders.
        Each sample is chosen uniformly, and applies the layout transforms defined.
        The formatter is not applied.
        """
        page, selected_line_ids, _, _ = self._gets_ann_ids_order_and_identifier(index)

        image, polygons = self.renderer.compose(selected_line_ids, page)

        if not self.return_bounding_boxes:
            return image, polygons
        else:
            return image, [polygon.minimum_rotated_rectangle for polygon in polygons]

    @property
    def formatter(self) -> _formatter_type | None:
        """Optional callable receiving (image, polygons, index) after rendering."""
        return self._formatter

    @formatter.setter
    def formatter(self, value: _formatter_type | None) -> None:
        self._formatter = value

    def collate_fn(self, batch):
        """Use the formatter's collator if provided, otherwise PyTorch's default"""
        collator = getattr(self._formatter, "collate_fn", None)
        if collator is None:
            from torch.utils.data import default_collate

            collator = default_collate
        return collator(batch)
