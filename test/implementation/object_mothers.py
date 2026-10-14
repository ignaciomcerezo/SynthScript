import numpy as np

from synthscript.ocr_units import OCRPage
from typing import Any


def make_page(
    *,
    page_id: str = "page",
    transcriptions: list[str] | None = None,
    polygon_coords: list[list[tuple[float, float]]] | None = None,
    line_ids: list[str] | None = None,
    rotations: list[float] | None = None,
    paragraph_line_ids: list[list[str]] | None = None,
    paragraph_ids: list[str | None] | None = None,
    metadata: dict[str, str] | None = None,
    shape: tuple[int, int] = (80, 120),
) -> OCRPage:
    if line_ids is None:
        line_ids = ["line"]
    if transcriptions is None:
        transcriptions = ["line"]
    if polygon_coords is None:
        polygon_coords = [[(2, 2), (50, 2), (50, 12), (2, 12)]]
    kwargs: dict[Any, Any] = dict(
        page_id=page_id,
        transcriptions=transcriptions,
        polygon_coords=polygon_coords,
        line_ids=line_ids,
        rotations=rotations or [0] * len(line_ids),
        stroke=np.zeros(shape, dtype=np.uint8),
        background=np.full(shape, 255, dtype=np.uint8),
        metadata=metadata,
    )
    if paragraph_line_ids is not None:
        return OCRPage.from_explicit_layout(
            **kwargs,
            paragraph_line_ids=paragraph_line_ids,
            paragraph_ids=paragraph_ids,
        )
    return OCRPage(**kwargs)


def make_two_line_page() -> OCRPage:
    return make_page(
        line_ids=["first", "second"],
        transcriptions=["first", "second"],
        polygon_coords=[
            [(2, 2), (50, 2), (50, 12), (2, 12)],
            [(2, 20), (50, 20), (50, 30), (2, 30)],
        ],
        paragraph_line_ids=[["first", "second"]],
    )
