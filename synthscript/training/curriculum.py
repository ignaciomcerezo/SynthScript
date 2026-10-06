from collections.abc import Sequence
from typing import Any, Literal

from synthscript.datasets.base_annotation_dataset import (
    BaseAnnotationDataset,
    DatasetTransform,
)

Order = int | Literal["page", "paragraph"]
Phase = Sequence[Order]


class Curriculum:
    def __init__(
        self,
        phases: Sequence[Phase],
        epochs_per_phase: Sequence[int] | None = None,
        transforms_per_phase: Sequence[DatasetTransform | None] | None = None,
    ) -> None:
        if not self._validate_phases(phases):
            raise ValueError(
                "Only ints > 0, 'page' and 'paragraph' are valid orders in each phase."
            )
        self._order_phases = tuple(tuple(phase) for phase in phases)
        self._epochs = (
            tuple(epochs_per_phase)
            if epochs_per_phase is not None
            else (1,) * len(self._order_phases)
        )
        self._transforms_per_phase = (
            [None for _ in range(len(self._order_phases))]
            if transforms_per_phase is None
            else transforms_per_phase
        )
        if (len(self._epochs) != len(self._order_phases)) or (
            len(self._epochs) != len(self._transforms_per_phase)
        ):
            raise ValueError("Epochs and phases must coincide in length.")
        if any(
            isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
            for epochs in self._epochs
        ):
            raise ValueError("Epochs per phase must be positive integers.")

        self._current_phase_index = 0
        self._remaining_phase_epochs = self._epochs[0] if self._epochs else 0

    @staticmethod
    def _validate_phases(
        phases: Sequence[Phase],
    ) -> bool:
        for phase in phases:
            for order in phase:
                if not (
                    (
                        isinstance(order, int)
                        and not isinstance(order, bool)
                        and order > 0
                    )
                    or order in ("page", "paragraph")
                ):
                    return False
        return True

    def advance_epoch(self, dataset: BaseAnnotationDataset) -> None:
        """Mark the current epoch complete and configure the next phase, if any."""
        if self.is_complete():
            raise RuntimeError("Cannot advance a completed curriculum.")

        self._remaining_phase_epochs -= 1
        if self._remaining_phase_epochs == 0:
            self._current_phase_index += 1
            if self.is_complete():
                return
            self._remaining_phase_epochs = self._epochs[self._current_phase_index]

        dataset.orders = self._order_phases[self._current_phase_index]
        transform = self._transforms_per_phase[self._current_phase_index]
        dataset.set_transform(transform)

    def advance_phase(self, dataset: BaseAnnotationDataset) -> None:
        """Mark the current phase (not epoch) complete and configure the next."""
        if self.is_complete():
            raise RuntimeError("Cannot advance a completed curriculum.")

        self._current_phase_index += 1
        if self.is_complete():
            self._remaining_phase_epochs = 0
            return
        self._remaining_phase_epochs = self._epochs[self._current_phase_index]

        dataset.orders = self._order_phases[self._current_phase_index]
        transform = self._transforms_per_phase[self._current_phase_index]
        dataset.set_transform(transform)

    @property
    def current_phase(self) -> list[Order]:
        if self.is_complete():
            raise RuntimeError("A completed curriculum has no current phase.")
        return list(self._order_phases[self._current_phase_index])

    @property
    def current_phase_epochs(self) -> int:
        if self.is_complete():
            raise RuntimeError("A completed curriculum has no current phase.")
        return self._epochs[self._current_phase_index]

    @property
    def remaining_phase_epochs(self) -> int:
        return self._remaining_phase_epochs

    @property
    def remaining_epochs(self) -> int:
        if self.is_complete():
            return 0
        return self._remaining_phase_epochs + sum(
            self._epochs[i]
            for i in range(self._current_phase_index + 1, len(self._order_phases))
        )

    def summary(self) -> None:
        cfi = self._current_phase_index
        if cfi > 0:
            print("Consumed phases:")
            for i, (phase, epoch) in enumerate(
                zip(self._order_phases[:cfi], self._epochs[:cfi], strict=True)
            ):
                print(f"\tPhase {i}: phase={list(phase)} @ epochs={epoch}")

        if cfi < len(self._order_phases):
            print("Pending phases:")
            for i, (phase, epoch) in enumerate(
                zip(self._order_phases[cfi:], self._epochs[cfi:], strict=True),
                start=cfi,
            ):
                remaining = self._remaining_phase_epochs if i == cfi else epoch
                print(f"\tPhase {i}: phase={list(phase)} @ epochs={remaining}")

    def is_complete(self) -> bool:
        return self._current_phase_index >= len(self._order_phases)

    def asdict(self) -> dict[int, dict[str, Any]]:
        result = {}
        for i, (phase, epochs, transforms) in enumerate(
            zip(
                self._order_phases,
                self._epochs,
                self._transforms_per_phase,
                strict=True,
            )
        ):
            result[i] = {
                "orders": phase,
                "epochs": epochs,
                "transforms": repr(transforms),
            }

        return result
