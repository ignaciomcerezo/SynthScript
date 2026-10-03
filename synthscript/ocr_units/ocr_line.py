from dataclasses import dataclass

import numpy as np
from shapely import Polygon


@dataclass(slots=True, kw_only=True)
class OCRLine:
    """
    A text line and the image/geometry data used by SynthScript.
    """

    id: str
    crop: np.ndarray
    polygon: Polygon
    rotation: float
    page_id: str
    text: str
    corrected_centroid: tuple[float, float] | None = None
    sindex: int | None = None
    paragraph_index: int | None = None

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"<OCRLine {self.id!r} on page {self.page_id!r}>"

    @property
    def index(self) -> int | None:
        """Deprecated alias for the transcription start offset."""
        return self.sindex

    @index.setter
    def index(self, value: int | None) -> None:
        self.sindex = value

    def centroid(self) -> tuple[float, float]:
        """Centroid of the associated polygon."""
        pol_centroid = self.polygon.centroid
        return pol_centroid.x, pol_centroid.y

    @property
    def top(self):
        """Lowest y coordinate (documents usually are y-down)."""
        return self.polygon.bounds[1]

    @property
    def left(self):
        """Lowest x coordinate."""
        return self.polygon.bounds[0]

    @property
    def right(self):
        """Greatest x coordinate."""
        return self.polygon.bounds[2]

    @property
    def bot(self):
        """Greatest y coordinate (documents usually are y-down)."""
        return self.polygon.bounds[3]
