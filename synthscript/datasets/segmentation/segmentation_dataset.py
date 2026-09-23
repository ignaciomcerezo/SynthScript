from collections.abc import Sequence
from dataclasses import asdict, replace

import numpy as np
from shapely.geometry import Polygon

from synthscript.datasets.base_annotation_dataset import (
    BaseAnnotationDataset,
    ClusterParams,
    RNGInput,
    orders_type,
)
from synthscript.datasets.image_transform_pack import ImageTransformPack
from synthscript.datasets.ocr_transform_pack import OCRTransformPack
from synthscript.ocr_units import OCRPage


class SegmentationDataset(BaseAnnotationDataset):
    """
    Dataset variant intended to be used for line segmentation training. Takes as input a
    sequence of annotations (of type AnnotatedPage) and a collecion of orders that will
    be used to sample the pages.

    When an item is requested, the dataset deterministically chooses an item (taken from
    all possible contiguous clusters of lines of length one of the orders provided),
    transforms it using the transform given (via .set_transform()) and returns the crop
    and line polygons.
    """

    def __init__(
        self,
        annotations: Sequence[OCRPage],
        *,
        orders: orders_type,
        rng: RNGInput = None,
        return_bounding_boxes: bool = True,
        cluster_transform_params: ClusterParams | None = None,
    ):
        self._annotated_pages = annotations
        self._orders: list[int] = []
        self._use_paragraphs = False
        self._use_full_pages = False
        self.rng = rng
        self._transforms: OCRTransformPack = OCRTransformPack(rng=self.rng)
        self._image_transforms = ImageTransformPack(rng=self.rng)
        self._update_orders(orders)  # the three previous attributes are updated here
        self.return_bounding_boxes = return_bounding_boxes

        self._cluster_params = (
            replace(ClusterParams(), **asdict(cluster_transform_params))
            if cluster_transform_params is not None
            else ClusterParams()
        )
        self._transforms.avoid_intersections = self._cluster_params.avoid_intersections

    def __repr__(self):
        return (
            f"<OCRDataset ({len(self)} samples: {len(self._annotated_pages)}"
            f" pages using orders {self.orders})>"
        )

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[Polygon]]:
        """
        Chooses a line cluster/paragraph/full page according to the available orders.
        Each sample is chosen uniformly, and applies the layout transforms defined.
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {self._size}"
            )

        ann, selected_line_ids, _, _ = self._gets_ann_ids_order_and_identifier(index)

        image, polygons = ann.synthetic_manuscript(
            line_ids=list(selected_line_ids),
            tight_layout=self.cluster_params.tight_layout,
            margin_size_px=self.cluster_params.margin_size_px,
            img_poly_transform=self._transforms,
            stroke_transform=self._image_transforms.transform_strokes,
            background_transform=self._image_transforms.transform_background,
            global_image_transform=self._image_transforms.transform_global_image,
            overlay_polygons=self.cluster_params.overlay_polygons,
            overlay_mbr=self.cluster_params.overlay_mbr,
        )
        if not self.return_bounding_boxes:
            return image, polygons
        else:
            return image, [polygon.minimum_rotated_rectangle for polygon in polygons]
