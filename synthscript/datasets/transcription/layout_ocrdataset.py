from collections.abc import Sequence
from copy import deepcopy

from synthscript.datasets.base_annotation_dataset import (
    ClusterParams,
    RNGInput,
    orders_type,
)
from synthscript.datasets.helpers.layout_generator import LayoutGenerator
from synthscript.datasets.transcription.ocrdataset import OCRDataset
from synthscript.ocr_units import OCRPage
from synthscript.transforms.ocr_transform_pack import OCRTransformPack
from synthscript.transforms.transforms import DatasetTransform


class LayoutOCRDataset(OCRDataset):
    """
    Dataset variant intended to be used for OCR model training. It is built atop
    synthscript.datasets.OCRDataset, but implements more agressive layout modification:
    Sample transforms are applied when items are requested, as in OCRDataset.
    Layout transforms configure whole-page generation; call refresh_layouts()
    after changing them to regenerate the pages.
    """

    def __init__(
        self,
        annotations: Sequence[OCRPage],
        layout_generator: LayoutGenerator | None = None,
        *,
        rng: RNGInput = None,
        cluster_transform_params: ClusterParams | None = None,
    ):
        self._layout_generator = (
            LayoutGenerator()
            if layout_generator is None
            else deepcopy(layout_generator)
        )
        self._base_annotations = annotations
        initial_rng = self._validate_rng(rng)
        self._layout_generator.rng = initial_rng

        super().__init__(
            ocrpages=self._generate_layouts(),
            rng=initial_rng,
            cluster_transform_params=cluster_transform_params,
        )

    @OCRDataset.rng.setter
    def rng(self, value: RNGInput) -> None:
        OCRDataset.rng.fset(self, value)
        self._layout_generator.rng = self._rng

    def _generate_layouts(self) -> list[OCRPage]:
        return [self._layout_generator.apply(ann) for ann in self._base_annotations]

    def refresh_layouts(self) -> None:
        """Regenerate annotations without resetting sampling or transform state."""
        self._annotated_pages = self._generate_layouts()
        self._recalculate_size_and_sampling_params()

    @property
    def layout_transforms(self) -> OCRTransformPack | None:
        """Transforms used to generate complete pages on the next refresh."""
        return self._layout_generator.transforms

    def add_transform(self, transform: DatasetTransform | None) -> None:
        """Append a transform applied when a sample is requested."""
        super().add_transform(transform)

    def set_transform(self, *transforms: DatasetTransform | None) -> None:
        """Replace transforms applied when samples are requested."""
        super().set_transform(*transforms)

    def add_layout_transform(self, transform: DatasetTransform | None) -> None:
        """Append a whole-page transform for the next layout refresh."""
        self._layout_generator.add_transform(transform)

    def set_layout_transform(self, *transforms: DatasetTransform | None) -> None:
        """Replace whole-page transforms for the next layout refresh."""
        self._layout_generator.set_transform(*transforms)

    @property
    def layout_generator(self) -> LayoutGenerator:
        return self._layout_generator

    @layout_generator.setter
    def layout_generator(self, value: LayoutGenerator) -> None:
        self._layout_generator = deepcopy(value)
        self._layout_generator.rng = self.rng
        self.refresh_layouts()

    @classmethod
    def from_split(
        cls,
        *groups_of_annotations: list[OCRPage],
        p: float,
        layout_generator: LayoutGenerator | None = None,
        orders_to_split_with: orders_type,
        rng_a: RNGInput = None,
        rng_b: RNGInput = None,
        n_trials: int | None = None,
    ) -> tuple["LayoutOCRDataset", "LayoutOCRDataset"]:
        """Build a split while retaining the required layout generator."""
        train: list[OCRPage] = []
        test: list[OCRPage] = []
        split_rng = cls._validate_rng(rng_a)
        generator = LayoutGenerator() if layout_generator is None else layout_generator

        for annotations in groups_of_annotations:
            train_i, test_i = cls.montecarlo_ann_split(
                annotations,
                p,
                orders=orders_to_split_with,
                rng=split_rng,
                n_trials=1000 if n_trials is None else n_trials,
            )
            train.extend(train_i)
            test.extend(test_i)

        return (
            cls(train, generator, rng=rng_b),
            cls(test, generator, rng=rng_b),
        )
