from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from typing import (
    Any,
    Literal,
)

from synthscript.datasets.base_annotation_dataset import (
    BaseAnnotationDataset,
    ClusterParams,
    RNGInput,
    orders_type,
)
from synthscript.datasets.image_transform_pack import ImageTransformPack
from synthscript.datasets.ocr_transform_pack import OCRTransformPack
from synthscript.ocr_units import OCRPage

_OCRDataset_default_getitem_fields = Literal[
    "image",
    "text",
    "sindex",
    "context",
    "order",
    "id",
    "page_id",
]

_OCRDataset_formatter_signature = Callable[
    [dict[_OCRDataset_default_getitem_fields, Any]], Any
]


class OCRDataset(BaseAnnotationDataset):
    """
    Dataset variant intended to be used for OCR tasks. Takes as input a sequence of
    annotations (of type AnnotatedPage) and a collecion of orders that will be used to
    sample the pages.

    When an item is requested, the dataset deterministically chooses an item (taken from
    all possible contiguous clusters of lines of length one of the orders provided),
    transforms it using the transform given (via .set_transform()) and returns the crop,
    its transcription, and other data.

    The output may be formatted using .set_formatter().
    """

    def __init__(
        self,
        annotations: Sequence[OCRPage],
        *,
        orders: orders_type,
        rng: RNGInput = None,
        cluster_transform_params: ClusterParams | None = None,
    ):
        self._annotated_pages = annotations
        self._orders: list[int] = []
        self._use_paragraphs = False
        self._use_full_pages = False
        self._update_orders(orders)  # the three previous attributes are updated here
        self._formatter: _OCRDataset_formatter_signature | None = None

        self._cluster_params = (
            replace(ClusterParams(), **asdict(cluster_transform_params))
            if cluster_transform_params is not None
            else ClusterParams()
        )
        self.rng = rng
        self._transforms: OCRTransformPack = OCRTransformPack(
            avoid_intersections=self._cluster_params.avoid_intersections,
            rng=self.rng,
        )
        self._image_transforms = ImageTransformPack(rng=self.rng)

    def __repr__(self):
        return (
            f"<OCRDataset ({len(self)} samples: {len(self._annotated_pages)}"
            f" pages using orders {self.orders}>"
        )

    def __getitem__(self, index: int):
        """
        Chooses a line cluster / paragraph / full page using only the available orders.
        Each sample is chosen uniformly, and applies the layout transforms defined.
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {self._size}"
            )

        ann, selected_line_ids, order, identifier = (
            self._gets_ann_ids_order_and_identifier(index)
        )

        synthetic_img, synthetic_transcription, sindex = ann.synthetic_sample(
            list(selected_line_ids),
            tight_layout=self.cluster_params.tight_layout,
            margin_size_px=self.cluster_params.margin_size_px,
            img_poly_transform=self._transforms,
            stroke_transform=self._image_transforms.transform_strokes,
            background_transform=self._image_transforms.transform_background,
            global_image_transform=self._image_transforms.transform_global_image,
            overlay_polygons=self.cluster_params.overlay_polygons,
            overlay_mbr=self.cluster_params.overlay_mbr,
        )

        # TODO: improve context generation - implement the use_previous_page_in_context
        # cluster parameter here
        context = ann.full_transcription[:sindex] if sindex > 0 else ""

        sample: dict[_OCRDataset_default_getitem_fields, Any] = {
            "image": synthetic_img,
            "text": synthetic_transcription,
            "sindex": sindex,
            "context": context,
            "order": order,
            "id": identifier,
            "page_id": ann.task_id,
        }

        if self._formatter is None:
            return sample
        else:
            return self._formatter(sample)

    @property
    def formatter(self) -> _OCRDataset_formatter_signature | None:
        return self._formatter

    @formatter.setter
    def formatter(self, value: _OCRDataset_formatter_signature | None) -> None:
        self._formatter = value
