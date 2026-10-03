from abc import ABC, abstractmethod
from typing import Literal

from synthscript.shared.path_bundle import PathBundle

Part = Literal[
    "raw_images",
    "background_images",
    "stroke_images",
    "annotations",
]
_PARTS = set[Part]


class ExternalInterface(ABC):
    @abstractmethod
    def setup(self, paths: PathBundle) -> None:
        """Create or update the data parts managed by this interface."""
        raise NotImplementedError

    @abstractmethod
    def parts_required(self) -> _PARTS:
        """Return the data parts that must already exist."""
        raise NotImplementedError

    @abstractmethod
    def parts_managed(self) -> _PARTS:
        """Return the data parts created and managed by this interface."""
        raise NotImplementedError
