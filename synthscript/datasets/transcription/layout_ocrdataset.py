from collections.abc import Sequence
from copy import deepcopy
from typing import Literal

from synthscript.datasets.base_annotation_dataset import (
    BaseAnnotationDataset,
    ClusterParams,
    CurriculumHistoryEntry,
    RNGInput,
    orders_type,
)
from synthscript.datasets.helpers.layout_generator import LayoutGenerator
from synthscript.datasets.transcription.ocrdataset import OCRDataset
from synthscript.ocr_units import OCRPage


class LayoutOCRDataset(BaseAnnotationDataset):
    """
    Dataset variant intended to be used for OCR model training. It is built atop
    synthscript.datasets.OCRDataset, but implements more agressive layout modification:
    When .refresh_layouts() is called, the base OCRDataset is copied and each page
    modified, changing the layout (via InterparagraphTransform).
    """

    def __init__(
        self,
        annotations: Sequence[OCRPage],
        layout_generator: LayoutGenerator,
        *,
        orders: orders_type,
        rng: RNGInput = None,
        cluster_transform_params: ClusterParams | None = None,
    ):
        self._layout_generator = deepcopy(layout_generator)
        self._base_annotations = annotations
        self.rng = rng
        self._set_underlying(orders=orders, params=cluster_transform_params)

    @BaseAnnotationDataset.rng.setter
    def rng(self, value: RNGInput) -> None:
        BaseAnnotationDataset.rng.fset(self, value)
        if hasattr(self, "_layout_generator"):
            self._layout_generator.rng = self._rng
        if hasattr(self, "_underlying_dataset"):
            self._underlying_dataset.rng = self._rng

    def _set_underlying(
        self,
        orders: orders_type,
        params: ClusterParams | None = None,
    ):
        new_anns = []
        for ann in self._base_annotations:
            new_anns.append(self._layout_generator.apply(ann))
        self._underlying_dataset = OCRDataset(
            ocrpages=new_anns,
            orders=orders,
            rng=self.rng,
            cluster_transform_params=params,
        )

    def refresh_layouts(self):
        previous_history = self._underlying_dataset._previous_history
        samples_solicited = self._underlying_dataset._samples_solicited
        new_anns = []
        for ann in self._base_annotations:
            new_anns.append(self._layout_generator.apply(ann))
        self._underlying_dataset = OCRDataset(
            ocrpages=new_anns,
            orders=self.orders,
            rng=self.rng,
            cluster_transform_params=self.cluster_params,
        )
        self._underlying_dataset._previous_history = previous_history
        self._underlying_dataset._samples_solicited = samples_solicited

    @property
    def orders(self):
        return self._underlying_dataset.orders

    @orders.setter
    def orders(self, value: Sequence[int | Literal["paragraph", "page"]]):

        self._underlying_dataset.orders = value

    @property
    def history(self) -> list[CurriculumHistoryEntry]:
        return self._underlying_dataset.history

    @property
    def cluster_params(self):
        return self._underlying_dataset._cluster_params

    @cluster_params.setter
    def cluster_params(self, value: ClusterParams):
        self._underlying_dataset.cluster_params = value

    @property
    def layout_generator(self) -> LayoutGenerator:
        return self._layout_generator

    @layout_generator.setter
    def layout_generator(self, value: LayoutGenerator):
        self._layout_generator = deepcopy(value)
        self._layout_generator.rng = self.rng
        self.refresh_layouts()

    def __len__(self):
        return len(self._underlying_dataset)

    def __getitem__(self, index):
        return self._underlying_dataset[index]
