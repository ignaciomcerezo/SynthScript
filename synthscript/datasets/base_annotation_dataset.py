from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from synthscript.datasets.image_transform_pack import ImageTransform, ImageTransformPack
from synthscript.datasets.ocr_transform_pack import OCRTransformPack
from synthscript.ocr_units import OCRPage
from synthscript.transforms.transforms import (
    BackgroundTransform,
    GlobalImageTransform,
    LineTransform,
    PageTransform,
    ParagraphTransform,
    StrokeTransform,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset
else:
    try:
        from torch.utils.data import Dataset
    except ImportError:

        class Dataset:
            """Fallback class when torch is not installed (no train)."""

            def __len__(self) -> int:
                raise NotImplementedError

            def __getitem__(self, index: int):
                raise NotImplementedError


DatasetTransform = LineTransform | ParagraphTransform | PageTransform | ImageTransform

orders_type = Collection[int | Literal["paragraph", "page"]]
RNGInput = np.random.Generator | int | None

T = TypeVar("T", bound="BaseAnnotationDataset")


@dataclass
class ClusterParams:
    tight_layout: bool = True
    margin_size_px: int | dict[Literal["left", "right", "top", "bottom"], int] = field(
        default_factory=lambda: {"left": 0, "right": 0, "top": 0, "bottom": 0}
    )
    use_previous_page_in_context: bool = False
    avoid_intersections: bool = True
    overlay_polygons: bool = False
    overlay_mbr: bool = False


class BaseAnnotationDataset(Dataset, ABC):
    _annotated_pages: Sequence[OCRPage]
    _orders: list[int]
    _use_paragraphs: bool
    _use_full_pages: bool
    _transforms: OCRTransformPack = field(default_factory=lambda: OCRTransformPack())
    _image_transforms: ImageTransformPack = field(
        default_factory=lambda: ImageTransformPack()
    )
    _cluster_params: ClusterParams = field(default_factory=lambda: ClusterParams())
    _rng: np.random.Generator

    @abstractmethod
    def __init__(
        self,
        pages: Sequence[OCRPage],
        *,
        orders: orders_type,
        rng: RNGInput = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _make_rng(value: RNGInput = None) -> np.random.Generator:
        if isinstance(value, np.random.Generator):
            return value
        return np.random.default_rng(value)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: RNGInput) -> None:
        self._rng = self._make_rng(value)

        # Packs may already exist when callers replace a dataset's RNG. Keep
        # every source of transform-selection randomness on the same stream.
        if isinstance(getattr(self, "_transforms", None), OCRTransformPack):
            self._transforms.rng = self._rng
        if isinstance(getattr(self, "_image_transforms", None), ImageTransformPack):
            self._image_transforms.rng = self._rng

    @property
    def pages(self) -> list[str | None]:
        return [ann.page for ann in self._annotated_pages]

    @property
    def ids(self) -> list[int]:
        return [ann.task_id for ann in self._annotated_pages]

    @property
    def orders(self):
        return (
            self._orders
            + ["paragraph"] * self._use_paragraphs
            + ["page"] * self._use_full_pages
        )

    @property
    def cluster_params(self) -> ClusterParams:
        return self._cluster_params

    @cluster_params.setter
    def cluster_params(self, value: ClusterParams) -> None:
        self._cluster_params = value
        self._transforms.avoid_intersections = value.avoid_intersections

    @orders.setter
    def orders(self, value: Sequence[int | Literal["paragraph", "page"]]):
        self._update_orders(value)

    def _update_orders(
        self,
        new_orders: Collection[int | Literal["paragraph", "page"]],
    ):
        for order in new_orders:
            if (
                isinstance(order, float)
                or (
                    isinstance(order, str)
                    and ((order != "page") and (order != "paragraph"))
                )
                or (isinstance(order, int) and (order < 1))
            ):
                raise ValueError(
                    f"Value '{order}' found inside value for orders. "
                    "Only ints > 0, 'paragraph' and 'page' are acceptable orders."
                )

        pseudo_old_orders: set[int | Literal["paragraph", "page"]] = set(self._orders)

        if self._use_paragraphs:
            pseudo_old_orders.add("paragraph")
        if self._use_full_pages:
            pseudo_old_orders.add("page")

        if set(new_orders) != pseudo_old_orders:
            self._orders = [x for x in new_orders if isinstance(x, int)]
            self._use_paragraphs = "paragraph" in new_orders
            self._use_full_pages = "page" in new_orders
            self._recalculate_size_and_sampling_params()

    def update_stage(
        self,
        new_orders: list[int],
    ):
        self._update_orders(new_orders)

    @staticmethod
    def _is_single_paragraph_page(ann: OCRPage) -> bool:
        """
        Whether the page consists of exactly one paragraph.

        Such a page represents the same set of lines whether it is sampled
        as a paragraph or as a complete page.
        """
        return len(ann.paragraphs) == 1

    def _page_is_additional_sample(self, ann: OCRPage) -> bool:
        """
        Whether the complete-page sample should be counted separately.

        If a page has exactly one paragraph and both paragraph and page
        sampling are enabled, the paragraph and page refer to the exact
        same sample and must therefore only be counted once.
        """
        return self._use_full_pages and not (
            self._use_paragraphs and self._is_single_paragraph_page(ann)
        )

    def _recalculate_size_and_sampling_params(self):
        page_sample_counts: list[int] = []
        par_prefix_sums: list[np.ndarray] = []
        page_prefix_sums: list[int] = []
        running_total: int = 0

        for ann in self._annotated_pages:
            par_counts: list[int] = []

            for par in ann.paragraphs:
                par_len = len(par)

                # Number of valid sliding windows of each integer order,
                # plus one sample if paragraph sampling is enabled.
                ell = sum(max(0, par_len - order + 1) for order in self._orders) + int(
                    self._use_paragraphs
                )

                par_counts.append(ell)

            page_par_samples = sum(par_counts)

            # A single-paragraph page is already represented by its paragraph
            # sample when both paragraph and page sampling are enabled.
            page_is_additional_sample = self._page_is_additional_sample(ann)

            page_total_samples = page_par_samples + int(page_is_additional_sample)

            par_prefix_sums.append(np.cumsum(par_counts, dtype=np.int64))
            page_sample_counts.append(page_total_samples)

            running_total += page_total_samples
            page_prefix_sums.append(running_total)

        self._size = running_total
        self._page_sample_counts = page_sample_counts
        self._page_prefix_sums = np.array(
            page_prefix_sums,
            dtype=np.int64,
        )
        self._par_prefix_sums = par_prefix_sums

    def __len__(self) -> int:
        return self._size

    def _gets_ann_ids_order_and_identifier(self, index: int) -> tuple[
        OCRPage,
        Sequence[str],
        int | Literal["paragraph", "page"],
        str,
    ]:
        """
        Chooses an annotation instance and specific lines according to the
        available orders.

        Each sample is chosen uniformly.

        When a page contains exactly one paragraph and both paragraph and
        page sampling are enabled, that page contributes only one sample.
        The sample is represented as the paragraph sample because its line
        IDs are identical to those of the complete page.
        """
        if index < 0 or index >= self._size:
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {self._size}"
            )

        page_idx = int(
            np.searchsorted(
                self._page_prefix_sums,
                index,
                side="right",
            )
        )

        page_offset = int(self._page_prefix_sums[page_idx - 1]) if page_idx > 0 else 0

        rel_idx = index - page_offset

        ann: OCRPage = self._annotated_pages[page_idx]

        # A page sample only occupies its own index when it was actually
        # counted as an additional sample.
        page_is_additional_sample = self._page_is_additional_sample(ann)

        if (
            page_is_additional_sample
            and rel_idx == self._page_sample_counts[page_idx] - 1
        ):
            selected_line_ids = list(ann.lines.keys())
            order = "page"
            identifier = f"pg{page_idx}"

        else:
            par_prefix = self._par_prefix_sums[page_idx]

            par_idx = int(
                np.searchsorted(
                    par_prefix,
                    rel_idx,
                    side="right",
                )
            )

            par_offset = int(par_prefix[par_idx - 1]) if par_idx > 0 else 0

            item_idx = rel_idx - par_offset

            paragraph = ann.paragraphs[par_idx]
            line_ids = paragraph.line_ids
            num_lines = len(line_ids)

            curr = item_idx
            selected_line_ids: list[str] | None = None

            for order in self._orders:
                n_windows: int = max(0, num_lines - order + 1)

                if curr < n_windows:
                    start_line = curr

                    selected_line_ids = line_ids[start_line : start_line + order]

                    identifier = (
                        f"pg{page_idx}par{par_idx}L{start_line}"
                        f"{start_line + order - 1}" * (order > 1)
                    )
                    break

                curr -= n_windows

            if selected_line_ids is None and self._use_paragraphs:
                selected_line_ids = line_ids
                order = "paragraph"
                identifier = f"pg{page_idx}par{par_idx}"

            if selected_line_ids is None:
                raise RuntimeError(f"Failed to resolve sample target for index {index}")

        return ann, selected_line_ids, order, identifier

    @property
    def transforms(self) -> OCRTransformPack | None:
        if self._transforms is None or self._transforms.is_identity:
            return None
        else:
            return self._transforms

    @property
    def image_transforms(self) -> ImageTransformPack | None:
        if self._image_transforms is None or self._image_transforms.is_identity:
            return None
        return self._image_transforms

    def add_transform(
        self,
        transform: DatasetTransform | None,
        probability: float = 1,
    ) -> None:
        if isinstance(transform, ImageTransform):
            self._image_transforms.add_transform(transform, probability)
        elif transform is not None:
            self._transforms.add_transform(transform, probability)

    def set_transform(
        self,
        *transform_probability_pairs: tuple[
            DatasetTransform | None,
            float,
        ],
    ) -> None:

        for transform in (transform for transform, _ in transform_probability_pairs):
            if transform is not None and not isinstance(
                transform,
                (
                    LineTransform,
                    ParagraphTransform,
                    PageTransform,
                    StrokeTransform,
                    BackgroundTransform,
                    GlobalImageTransform,
                ),
            ):
                raise ValueError(f"Unsupported transform type {type(transform)}")
        transform: DatasetTransform | None
        self._transforms = OCRTransformPack(
            avoid_intersections=self.cluster_params.avoid_intersections,
            rng=self.rng,
        )
        self._image_transforms = ImageTransformPack(rng=self.rng)
        for transform, probability in transform_probability_pairs:
            if probability != 0:
                self.add_transform(transform, probability)

    @staticmethod
    def samples_in_annotation(
        ann: OCRPage,
        orders: Collection[int | Literal["paragraph", "page"]],
    ):
        use_paragraphs = "paragraph" in orders
        use_full_pages = "page" in orders

        paragraph_samples = sum(
            max(0, len(paragraph) - order + 1)
            for paragraph in ann.paragraphs
            for order in orders
            if isinstance(order, int)
        )

        paragraph_samples += len(ann.paragraphs) if use_paragraphs else 0

        # If the page consists of exactly one paragraph and both sampling
        # modes are enabled, the page and paragraph are the same sample.
        page_samples = int(
            use_full_pages
            and not (
                use_paragraphs and BaseAnnotationDataset._is_single_paragraph_page(ann)
            )
        )

        return paragraph_samples + page_samples

    @staticmethod
    def montecarlo_ann_split(
        annotations: list[OCRPage],
        p=0.95,
        orders: Collection[int | Literal["paragraph", "page"]] = [1],
        n_trials: int = 1000,
    ) -> tuple[list[OCRPage], list[OCRPage]]:

        print(f"Performing Monte Carlo page split with {n_trials} trials")

        weights = [
            (
                i,
                BaseAnnotationDataset.samples_in_annotation(
                    ann,
                    orders,
                ),
            )
            for i, ann in enumerate(annotations)
        ]

        total_samples = sum(weight for _, weight in weights)

        if total_samples == 0:
            raise ValueError("Total number of samples is zero.")

        if not 0 <= p <= 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")

        target_a = total_samples * p

        best_error = float("inf")
        best_pages: set[int] = set()

        for _ in range(n_trials):
            candidate_pages = {i for i, _ in weights if np.random.rand() < p}

            candidate_samples = sum(
                weight for i, weight in weights if i in candidate_pages
            )

            error = abs(candidate_samples - target_a)

            if error < best_error:
                best_error = error
                best_pages = candidate_pages

                # Exact match, so there is no reason to keep searching.
                if error == 0:
                    break

        train_annotations = [
            annotations[i] for i in range(len(annotations)) if i in best_pages
        ]

        test_annotations = [
            annotations[i] for i in range(len(annotations)) if i not in best_pages
        ]

        return train_annotations, test_annotations

    @classmethod
    def from_split(
        cls: type[T],
        *groups_of_annotations: list[OCRPage],
        p: float,
        orders: orders_type,
        orders_to_split_with: orders_type | None = None,
    ) -> tuple[T, T]:
        """
        Generates two datasets (paradigmatically train and test) from
        various groups of AnnotatedPages.

        This could be used to generate splits that are balanced in
        difficulty, passing groups of annotations that differ in difficulty,
        or in any other characteristic, to get splits homogeneous in that
        characteristic.

        The split is done taking the number of samples in each annotation
        considering only samples that are of order orders_to_split_with
        (or 'orders', if the former is not given a value).

        Example:

            ann_group_1 = [...]
            ann_group_2 = [...]
            ann_group_3 = [...]

            train, test = OCRDataset.from_split(
                ann_group_1,
                ann_group_2,
                ann_group_3,
                p=0.95,
                orders_to_split_with=[1],
                orders=[1, 2, 3, 4, 5],
            )

        This produces a split that has approximately 0.95 of groups 1, 2,
        and 3 in train, with the remaining annotations in test.
        """

        train = []
        test = []

        for annotations in groups_of_annotations:
            train_i, test_i = cls.montecarlo_ann_split(
                annotations,
                p,
                orders=(
                    orders if orders_to_split_with is None else orders_to_split_with
                ),
            )

            train += train_i
            test += test_i

        return (
            cls(train, orders=orders, rng=np.random.default_rng()),
            cls(test, orders=orders, rng=np.random.default_rng()),
        )

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
