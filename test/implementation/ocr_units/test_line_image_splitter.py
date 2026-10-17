from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely import Polygon

from synthscript.loading.page_loader import load_pages
from synthscript.loading.page_xml import save_page_xml
from synthscript.ocr_units import OCRPage
from synthscript.shared.image_processing import crop_image_with_polygon
from synthscript.shared.line_image_splitter import LineImageSplitter
from synthscript.shared.path_bundle import PathBundle


def make_overlapping_page(
    *, lower_extension: int = 1, splitter: LineImageSplitter | None = None
) -> OCRPage:
    stroke = np.zeros((40, 60), dtype=np.uint8)
    stroke[6:20, 20:25] = 100
    stroke[20, 20 : 20 + lower_extension] = 100
    return OCRPage.from_explicit_layout(
        transcriptions=["upper", "lower"],
        polygon_coords=[
            [(10, 5), (40, 5), (40, 20), (10, 20)],
            [(10, 16), (40, 16), (40, 35), (10, 35)],
        ],
        line_ids=["upper", "lower"],
        rotations=[0, 0],
        page_id="overlap",
        stroke=stroke,
        background=np.full_like(stroke, 255),
        paragraph_line_ids=[["upper", "lower"]],
        line_image_splitter=splitter,
    )


def ink_pixels(page: OCRPage, line_id: str) -> int:
    crop = page.lines[line_id].crop
    return int(((crop[..., 0] > 10) & (crop[..., 3] > 0)).sum())


def test_split_preserves_polygon_crop() -> None:
    stroke = np.arange(40 * 60, dtype=np.uint8).reshape(40, 60)
    polygon = Polygon([(10, 5), (40, 5), (40, 20), (10, 20)])
    crop = LineImageSplitter().split(stroke, [polygon])[0]
    np.testing.assert_array_equal(crop, crop_image_with_polygon(stroke, polygon))


@pytest.mark.parametrize(
    ("lower_extension", "expected_lower"), [(1, 1), (2, 2), (3, 23)]
)
def test_overlap_removed_only_below_ten_percent(
    lower_extension: int, expected_lower: int
) -> None:
    page = make_overlapping_page(
        lower_extension=lower_extension, splitter=LineImageSplitter()
    )
    assert ink_pixels(page, "upper") == 70
    assert ink_pixels(page, "lower") == expected_lower


def test_splitter_skips_overlap_that_bridges_owner_extensions() -> None:
    stroke = np.zeros((40, 60), dtype=np.uint8)
    stroke[10:20, 18:32] = 100
    page = OCRPage.from_explicit_layout(
        transcriptions=["upper", "lower"],
        polygon_coords=[
            [(5, 5), (45, 5), (45, 25), (5, 25)],
            [(24, 5), (26, 5), (26, 35), (24, 35)],
        ],
        line_ids=["upper", "lower"],
        rotations=[0, 0],
        page_id="bridge",
        stroke=stroke,
        background=np.full_like(stroke, 255),
        paragraph_line_ids=[["upper", "lower"]],
        line_image_splitter=LineImageSplitter(),
    )
    assert ink_pixels(page, "lower") == 20


def test_splitter_handles_multiple_overlap_islands() -> None:
    stroke = np.zeros((40, 60), dtype=np.uint8)
    for left in (10, 30):
        stroke[6:20, left : left + 5] = 100
        stroke[20, left] = 100
    page = OCRPage.from_explicit_layout(
        transcriptions=["upper", "lower"],
        polygon_coords=[
            [(5, 5), (45, 5), (45, 20), (5, 20)],
            [(5, 16), (45, 16), (45, 35), (5, 35)],
        ],
        line_ids=["upper", "lower"],
        rotations=[0, 0],
        page_id="multiple",
        stroke=stroke,
        background=np.full_like(stroke, 255),
        paragraph_line_ids=[["upper", "lower"]],
        line_image_splitter=LineImageSplitter(),
    )
    assert ink_pixels(page, "lower") == 2


def test_splitter_keeps_islands_connected_in_the_other_line() -> None:
    stroke = np.zeros((40, 60), dtype=np.uint8)
    for left in (20, 26):
        stroke[6:20, left : left + 5] = 100
    stroke[20, 25] = 100
    page = OCRPage.from_explicit_layout(
        transcriptions=["upper", "lower"],
        polygon_coords=[
            [(10, 5), (40, 5), (40, 20), (10, 20)],
            [(10, 16), (40, 16), (40, 35), (10, 35)],
        ],
        line_ids=["upper", "lower"],
        rotations=[0, 0],
        page_id="connected",
        stroke=stroke,
        background=np.full_like(stroke, 255),
        paragraph_line_ids=[["upper", "lower"]],
        line_image_splitter=LineImageSplitter(),
    )
    assert ink_pixels(page, "lower") == 41


def test_splitter_leaves_three_line_overlap_shared() -> None:
    stroke = np.zeros((40, 60), dtype=np.uint8)
    stroke[6:20, 20:25] = 100
    stroke[20, 20] = 100
    page = OCRPage.from_explicit_layout(
        transcriptions=["upper", "lower", "third"],
        polygon_coords=[
            [(10, 5), (40, 5), (40, 20), (10, 20)],
            [(10, 16), (40, 16), (40, 35), (10, 35)],
            [(19, 16), (26, 16), (26, 20), (19, 20)],
        ],
        line_ids=["upper", "lower", "third"],
        rotations=[0, 0, 0],
        page_id="triple",
        stroke=stroke,
        background=np.full_like(stroke, 255),
        paragraph_line_ids=[["upper", "lower"], ["third"]],
        line_image_splitter=LineImageSplitter(),
    )
    assert ink_pixels(page, "lower") == 21


def test_load_pages_uses_splitter_by_default(tmp_path: Path) -> None:
    paths = PathBundle(tmp_path)
    page = make_overlapping_page()
    stroke = np.zeros((40, 60), dtype=np.uint8)
    stroke[6:20, 20:25] = 100
    stroke[20, 20] = 100
    for path, image in (
        (paths.get_raw_image_path("overlap"), np.full_like(stroke, 255)),
        (paths.get_stroke_image_path("overlap"), stroke),
        (paths.get_background_image_path("overlap"), np.full_like(stroke, 255)),
    ):
        encoded, buffer = cv2.imencode(".png", image)
        assert encoded
        path.write_bytes(buffer.tobytes())
    save_page_xml(
        page,
        paths.get_page_xml_path("overlap"),
        raw_image_path=paths.get_raw_image_path("overlap"),
        stroke_image_path=paths.get_stroke_image_path("overlap"),
        background_image_path=paths.get_background_image_path("overlap"),
    )

    corrected = load_pages(root_path=paths, transcription_homogenizer=lambda text: text)
    uncorrected = load_pages(
        root_path=paths,
        transcription_homogenizer=lambda text: text,
        line_image_splitter=LineImageSplitter(remove_shared_ink=False),
    )
    assert ink_pixels(corrected[0], "lower") == 1
    assert ink_pixels(uncorrected[0], "lower") == 21
    assert [paragraph.line_ids for paragraph in corrected[0].paragraphs] == [
        ["upper", "lower"]
    ]
    assert corrected[0].full_transcription == uncorrected[0].full_transcription
    assert corrected[0].lines["lower"].sindex == uncorrected[0].lines["lower"].sindex
