import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from lxml import etree  # ty: ignore[unresolved-import]

import synthscript.loading.page_loader as page_loader_module
import synthscript.loading.page_xml.parser as page_xml_parser
from synthscript.loading.external_interfaces import LabelStudioInterface
from synthscript.loading.external_interfaces.label_studio.ls_typed_dicts import (
    RectangleResult,
    RelationResult,
    SimplifiedTextCorrectionResult,
)
from synthscript.loading.page_loader import load_pages
from synthscript.loading.page_xml import (
    PageXMLValidationError,
    load_page_xml,
    ocr_rotation_to_page_orientation,
    save_page_xml,
    validate_page_xml,
)
from synthscript.loading.page_xml.constants import PAGE_NS, page_tag
from synthscript.ocr_units import OCRPage
from synthscript.shared.path_bundle import PathBundle
from test.implementation.object_mothers import make_page


def _write_images(paths: PathBundle, page_id: str = "123") -> tuple[Path, Path, Path]:
    image = np.full((80, 120), 255, dtype=np.uint8)
    raw = paths.get_raw_image_path(page_id)
    stroke = paths.get_stroke_image_path(page_id)
    background = paths.get_background_image_path(page_id)
    for path, array in (
        (raw, image),
        (stroke, np.zeros_like(image)),
        (background, image),
    ):
        encoded, buffer = cv2.imencode(".png", array)
        assert encoded
        path.write_bytes(buffer.tobytes())
    return raw, stroke, background


@pytest.mark.parametrize("rotation", [0, 10, -10, 90, -90])
def test_orientation_sign(rotation: float) -> None:
    assert ocr_rotation_to_page_orientation(rotation) == -rotation


def test_page_round_trip_preserves_layout_and_ground_truth(tmp_path: Path) -> None:
    paths = PathBundle(tmp_path)
    raw, stroke_path, background_path = _write_images(paths)
    page = make_page(
        transcriptions=[r"$x < y$ & café", "β > λ"],
        polygon_coords=[
            [(1.4, 1.4), (50.6, 1.4), (50.6, 10.6), (1.4, 10.6)],
            [(2.4, 30.4), (60.6, 30.4), (60.6, 40.6), (2.4, 40.6)],
        ],
        line_ids=["1 bad&id", "line_valid"],
        rotations=[10, -90],
        page_id="123",
        metadata={
            "label_studio.task_id": "456",
            "label_studio.created_at": "2026-01-02T03:04:05Z",
            "label_studio.updated_at": "2026-01-02T04:04:05Z",
        },
        paragraph_line_ids=[["line_valid"], ["1 bad&id"]],
        paragraph_ids=["second region", "first region"],
    )

    xml_path = paths.get_page_xml_path("123")
    save_page_xml(
        page,
        xml_path,
        raw_image_path=raw,
        stroke_image_path=stroke_path,
        background_image_path=background_path,
    )
    validate_page_xml(xml_path)
    loaded = load_page_xml(xml_path, paths)

    assert loaded.page_id == "123"
    assert [paragraph.id for paragraph in loaded.paragraphs] == [
        "second region",
        "first region",
    ]
    assert [paragraph.line_ids for paragraph in loaded.paragraphs] == [
        ["line_valid"],
        ["1 bad&id"],
    ]
    assert loaded.full_transcription == "β > λ\n$x < y$ & café"
    assert loaded.lines["line_valid"].sindex == 0
    assert loaded.lines["1 bad&id"].sindex == len("β > λ\n")
    assert loaded.lines["1 bad&id"].rotation == pytest.approx(10)
    assert loaded.lines["line_valid"].rotation == pytest.approx(-90)
    assert loaded.lines["1 bad&id"].polygon.bounds == pytest.approx((1, 1, 51, 11))
    assert loaded.metadata["page_xml.created"] == "2026-01-02T03:04:05Z"

    tree = etree.parse(str(xml_path))
    page_element = tree.getroot().find(page_tag("Page"))
    assert page_element is not None
    assert not Path(page_element.get("imageFilename", "")).is_absolute()
    assert "\\" not in page_element.get("imageFilename", "")
    assert len(page_element.findall(page_tag("AlternativeImage"))) == 2


def test_external_page_without_synthscript_metadata(tmp_path: Path) -> None:
    paths = PathBundle(tmp_path)
    raw, _, _ = _write_images(paths, "external")
    relative_raw = Path(os.path.relpath(raw, paths.page_xml_path)).as_posix()
    root = etree.Element(
        page_tag("PcGts"),
        nsmap={None: PAGE_NS},
        pcGtsId="pc_external",
    )
    metadata = etree.SubElement(root, page_tag("Metadata"))
    etree.SubElement(metadata, page_tag("Creator")).text = "Third party"
    etree.SubElement(metadata, page_tag("Created")).text = "2026-01-01T00:00:00Z"
    etree.SubElement(metadata, page_tag("LastChange")).text = "2026-01-01T00:00:00Z"
    page_element = etree.SubElement(
        root,
        page_tag("Page"),
        imageFilename=relative_raw,
        imageWidth="120",
        imageHeight="80",
    )
    region = etree.SubElement(
        page_element,
        page_tag("TextRegion"),
        id="r1",
        type="paragraph",
    )
    etree.SubElement(region, page_tag("Coords"), points="1,1 60,1 60,20 1,20")
    line = etree.SubElement(region, page_tag("TextLine"), id="l1")
    etree.SubElement(line, page_tag("Coords"), points="1,1 60,1 60,10 1,10")
    high = etree.SubElement(line, page_tag("TextEquiv"), index="4")
    etree.SubElement(high, page_tag("Unicode")).text = "wrong"
    low = etree.SubElement(line, page_tag("TextEquiv"), index="1")
    etree.SubElement(low, page_tag("Unicode")).text = "chosen"
    xml_path = paths.page_xml_path / "external.xml"
    etree.ElementTree(root).write(str(xml_path), encoding="UTF-8", xml_declaration=True)

    loaded = load_page_xml(xml_path, paths)
    assert loaded.page_id == "external"
    assert loaded.full_transcription == "chosen"
    assert loaded.lines["l1"].rotation == pytest.approx(0)


def test_external_reading_order_and_line_indices_are_preserved(tmp_path: Path) -> None:
    paths = PathBundle(tmp_path)
    raw, _, _ = _write_images(paths, "ordered")
    relative_raw = Path(os.path.relpath(raw, paths.page_xml_path)).as_posix()
    root = etree.Element(
        page_tag("PcGts"),
        nsmap={None: PAGE_NS},
        pcGtsId="pc_ordered",
    )
    metadata = etree.SubElement(root, page_tag("Metadata"))
    etree.SubElement(metadata, page_tag("Creator")).text = "Third party"
    etree.SubElement(metadata, page_tag("Created")).text = "2026-01-01T00:00:00Z"
    etree.SubElement(metadata, page_tag("LastChange")).text = "2026-01-01T00:00:00Z"
    page_element = etree.SubElement(
        root,
        page_tag("Page"),
        imageFilename=relative_raw,
        imageWidth="120",
        imageHeight="80",
    )
    reading_order = etree.SubElement(page_element, page_tag("ReadingOrder"))
    group = etree.SubElement(reading_order, page_tag("OrderedGroup"), id="ro")
    etree.SubElement(
        group,
        page_tag("RegionRefIndexed"),
        regionRef="r2",
        index="0",
    )
    etree.SubElement(
        group,
        page_tag("RegionRefIndexed"),
        regionRef="r1",
        index="1",
    )

    region_one = etree.SubElement(
        page_element,
        page_tag("TextRegion"),
        id="r1",
        type="paragraph",
    )
    etree.SubElement(region_one, page_tag("Coords"), points="1,20 60,20 60,40 1,40")
    for line_id, index, text, points in (
        ("l1", "1", "second", "1,30 60,30 60,39 1,39"),
        ("l0", "0", "first", "1,20 60,20 60,29 1,29"),
    ):
        line = etree.SubElement(
            region_one,
            page_tag("TextLine"),
            id=line_id,
            index=index,
        )
        etree.SubElement(line, page_tag("Coords"), points=points)
        text_equiv = etree.SubElement(line, page_tag("TextEquiv"), index="0")
        etree.SubElement(text_equiv, page_tag("Unicode")).text = text

    region_two = etree.SubElement(
        page_element,
        page_tag("TextRegion"),
        id="r2",
        type="paragraph",
    )
    etree.SubElement(region_two, page_tag("Coords"), points="1,1 60,1 60,10 1,10")
    line = etree.SubElement(region_two, page_tag("TextLine"), id="l2", index="0")
    etree.SubElement(line, page_tag("Coords"), points="1,1 60,1 60,10 1,10")
    text_equiv = etree.SubElement(line, page_tag("TextEquiv"), index="0")
    etree.SubElement(text_equiv, page_tag("Unicode")).text = "top"

    xml_path = paths.page_xml_path / "ordered.xml"
    etree.ElementTree(root).write(str(xml_path), encoding="UTF-8", xml_declaration=True)
    loaded = load_page_xml(xml_path, paths)

    assert [paragraph.id for paragraph in loaded.paragraphs] == ["r2", "r1"]
    assert loaded.paragraphs[1].line_ids == ["l0", "l1"]
    assert loaded.full_transcription == "top\nfirst\nsecond"


def test_schema_validation_rejects_duplicate_ids(tmp_path: Path) -> None:
    paths = PathBundle(tmp_path)
    raw, stroke_path, background_path = _write_images(paths, "duplicates")
    page = make_page(
        transcriptions=["one", "two"],
        polygon_coords=[
            [(1, 1), (50, 1), (50, 10), (1, 10)],
            [(1, 10), (50, 10), (50, 20), (1, 20)],
        ],
        line_ids=["one", "two"],
        rotations=[0, 0],
        page_id="duplicates",
    )
    xml_path = paths.get_page_xml_path("duplicates")
    save_page_xml(
        page,
        xml_path,
        raw_image_path=raw,
        stroke_image_path=stroke_path,
        background_image_path=background_path,
    )
    tree = etree.parse(str(xml_path))
    lines = tree.findall(f".//{page_tag('TextLine')}")
    lines[1].set("id", lines[0].get("id"))
    tree.write(str(xml_path), encoding="UTF-8", xml_declaration=True)

    with pytest.raises(PageXMLValidationError):
        validate_page_xml(xml_path)


def test_load_pages_combines_annotations_by_page_id(tmp_path: Path) -> None:
    paths = PathBundle(tmp_path)
    raw, stroke_path, background_path = _write_images(paths, "combined")
    for subindex, (line_id, text, y) in enumerate(
        (("top", "first", 1), ("bottom", "second", 30))
    ):
        page = make_page(
            transcriptions=[text],
            polygon_coords=[[(1, y), (50, y), (50, y + 9), (1, y + 9)]],
            line_ids=[line_id],
            rotations=[0],
            page_id="combined",
            metadata={"label_studio.annotation_id": str(subindex)},
        )
        save_page_xml(
            page,
            paths.get_page_xml_path("combined", subindex),
            raw_image_path=raw,
            stroke_image_path=stroke_path,
            background_image_path=background_path,
        )

    combined = load_pages(
        root_path=paths,
        transcription_homogenizer=lambda text: text,
    )
    assert len(combined) == 1
    assert combined[0].page_id == "combined"
    assert combined[0].full_transcription == "first\nsecond"
    assert "synthscript.combined_annotations" in combined[0].metadata


def test_load_pages_filters_before_building_ocr_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = PathBundle(tmp_path)
    for page_id, task_id in (("selected", "17"), ("ignored", "23")):
        raw, stroke_path, background_path = _write_images(paths, page_id)
        page = make_page(
            transcriptions=[page_id],
            polygon_coords=[[(1, 1), (50, 1), (50, 10), (1, 10)]],
            line_ids=[f"line-{page_id}"],
            rotations=[0],
            page_id=page_id,
            metadata={"label_studio.task_id": task_id},
        )
        save_page_xml(
            page,
            paths.get_page_xml_path(page_id),
            raw_image_path=raw,
            stroke_image_path=stroke_path,
            background_image_path=background_path,
        )

    loaded_paths: list[Path] = []
    original_load = page_loader_module.load_page_xml_document

    def tracked_load(document, xml_path, **kwargs):
        loaded_paths.append(xml_path)
        return original_load(document, xml_path, **kwargs)

    monkeypatch.setattr(page_loader_module, "load_page_xml_document", tracked_load)

    loaded = load_pages(
        root_path=paths,
        page_ids={"selected"},
        transcription_homogenizer=lambda text: text,
    )
    assert [page.page_id for page in loaded] == ["selected"]
    assert [path.stem for path in loaded_paths] == ["s0_pgselected"]

    loaded_paths.clear()
    loaded = load_pages(
        root_path=paths,
        task_ids={23},
        transcription_homogenizer=lambda text: text,
    )
    assert [page.page_id for page in loaded] == ["ignored"]
    assert [path.stem for path in loaded_paths] == ["s0_pgignored"]


def test_load_pages_does_not_combine_a_single_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = PathBundle(tmp_path)
    raw, stroke_path, background_path = _write_images(paths, "single")
    page = make_page(
        transcriptions=["single"],
        polygon_coords=[[(1, 1), (50, 1), (50, 10), (1, 10)]],
        line_ids=["line-single"],
        rotations=[0],
        page_id="single",
    )
    save_page_xml(
        page,
        paths.get_page_xml_path("single"),
        raw_image_path=raw,
        stroke_image_path=stroke_path,
        background_image_path=background_path,
    )

    def unexpected_combine(*_annotations):
        pytest.fail("A single annotation must not be combined.")

    monkeypatch.setattr(OCRPage, "combine_annotations", unexpected_combine)

    loaded = load_pages(
        root_path=paths,
        transcription_homogenizer=lambda text: text,
    )
    assert [loaded_page.page_id for loaded_page in loaded] == ["single"]


def test_load_page_xml_parses_document_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = PathBundle(tmp_path)
    raw, stroke_path, background_path = _write_images(paths, "once")
    page = make_page(
        transcriptions=["once"],
        polygon_coords=[[(1, 1), (50, 1), (50, 10), (1, 10)]],
        line_ids=["line-once"],
        rotations=[0],
        page_id="once",
    )
    xml_path = paths.get_page_xml_path("once")
    save_page_xml(
        page,
        xml_path,
        raw_image_path=raw,
        stroke_image_path=stroke_path,
        background_image_path=background_path,
    )

    parse_count = 0
    original_parse = page_xml_parser.parse_xml_document

    def tracked_parse(path):
        nonlocal parse_count
        parse_count += 1
        return original_parse(path)

    monkeypatch.setattr(page_xml_parser, "parse_xml_document", tracked_parse)

    page_xml_parser.load_page_xml(xml_path, paths)
    assert parse_count == 1


def test_load_page_xml_reads_raw_png_dimensions_without_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = PathBundle(tmp_path)
    raw, stroke_path, background_path = _write_images(paths, "header")
    page = make_page(
        transcriptions=["header"],
        polygon_coords=[[(1, 1), (50, 1), (50, 10), (1, 10)]],
        line_ids=["line-header"],
        rotations=[0],
        page_id="header",
    )
    xml_path = paths.get_page_xml_path("header")
    save_page_xml(
        page,
        xml_path,
        raw_image_path=raw,
        stroke_image_path=stroke_path,
        background_image_path=background_path,
    )

    decoded_paths: list[Path] = []
    original_load = PathBundle.load_image_grayscale_np

    def tracked_load(path):
        decoded_paths.append(path)
        return original_load(path)

    monkeypatch.setattr(
        PathBundle,
        "load_image_grayscale_np",
        staticmethod(tracked_load),
    )

    load_page_xml(xml_path, paths)

    assert raw not in decoded_paths
    assert set(decoded_paths) == {stroke_path, background_path}


def test_label_studio_writes_page_with_inferred_reading_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = PathBundle(tmp_path)
    _write_images(paths)
    rectangle = RectangleResult.model_validate(
        {
            "id": "box id",
            "to_name": "image",
            "type": "rectanglelabels",
            "origin": "manual",
            "from_name": "img_regions",
            "original_width": 120,
            "original_height": 80,
            "image_rotation": 0,
            "value": {
                "x": 10,
                "y": 10,
                "width": 80,
                "height": 10,
                "rotation": 10,
                "rectanglelabels": ["line"],
            },
        }
    )
    transcription = SimplifiedTextCorrectionResult.model_validate(
        {
            "id": "text id",
            "to_name": "image",
            "type": "textarea",
            "origin": "manual",
            "from_name": "correction",
            "value": {"text": [r"$x & y$"]},
        }
    )
    relation = RelationResult.model_validate(
        {
            "from_id": "box id",
            "to_id": "text id",
            "type": "relation",
            "direction": "right",
        }
    )
    lower_rectangle = RectangleResult.model_validate(
        {
            "id": "lower box",
            "to_name": "image",
            "type": "rectanglelabels",
            "origin": "manual",
            "from_name": "img_regions",
            "original_width": 120,
            "original_height": 80,
            "image_rotation": 0,
            "value": {
                "x": 10,
                "y": 30,
                "width": 80,
                "height": 10,
                "rotation": 0,
                "rectanglelabels": ["line"],
            },
        }
    )
    lower_transcription = SimplifiedTextCorrectionResult.model_validate(
        {
            "id": "lower text",
            "to_name": "image",
            "type": "textarea",
            "origin": "manual",
            "from_name": "correction",
            "value": {"text": ["lower"]},
        }
    )
    lower_relation = RelationResult.model_validate(
        {
            "from_id": "lower box",
            "to_id": "lower text",
            "type": "relation",
            "direction": "right",
        }
    )
    annotation = SimpleNamespace(
        completed_by=1,
        updated_by=1,
        id=9,
        result=[
            lower_rectangle,
            lower_transcription,
            lower_relation,
            rectangle,
            transcription,
            relation,
        ],
        project=4,
        unique_id="annotation-9",
        ground_truth=True,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
        lead_time=3.5,
    )
    task = SimpleNamespace(
        id=77,
        data=SimpleNamespace(image_url="123.png"),
        annotations=[annotation],
    )
    interface = LabelStudioInterface("https://example.invalid", "token")
    monkeypatch.setattr(interface, "annotations_are_up_to_date", lambda *args: False)
    monkeypatch.setattr(interface, "fetch_simplified_tasks", lambda: [task])

    interface.setup(paths)

    assert interface.parts_managed() == {"annotations"}
    assert interface.parts_required() == {
        "raw_images",
        "background_images",
        "stroke_images",
    }
    xml_path = paths.get_page_xml_path("123")
    assert xml_path.is_file()
    assert not any(
        (tmp_path / "data_in" / name).exists()
        for name in (
            "metadata",
            "polygons",
            "rotations",
            "ids",
            "transcriptions",
        )
    )
    loaded = load_page_xml(xml_path, paths)
    assert [paragraph.line_ids for paragraph in loaded.paragraphs] == [
        ["box id-text id", "lower box-lower text"]
    ]
    assert loaded.full_transcription == "$x & y$\nlower"
    assert loaded.metadata["label_studio.task_id"] == "77"
    assert loaded.lines["box id-text id"].polygon.bounds == pytest.approx(
        (11, 8, 107, 33)
    )
    runtime_page = load_pages(
        root_path=paths,
        transcription_homogenizer=lambda text: text,
    )[0]
    assert [paragraph.line_ids for paragraph in runtime_page.paragraphs] == [
        ["box id-text id", "lower box-lower text"]
    ]
    assert runtime_page.full_transcription == loaded.full_transcription
    assert runtime_page.lines["box id-text id"].sindex == 0
    assert runtime_page.lines["lower box-lower text"].sindex == len("$x & y$\n")


def test_label_studio_checks_page_xml_creation_against_remote_update(
    tmp_path: Path,
) -> None:
    paths = PathBundle(tmp_path)
    xml_path = paths.get_page_xml_path("123")
    xml_path.write_text("<PcGts/>", encoding="utf-8")
    local_creation = datetime.fromtimestamp(xml_path.stat().st_ctime, timezone.utc)
    remote_update = local_creation - timedelta(seconds=1)
    requested: dict[str, object] = {}

    def get_paginated_tasks(**kwargs):
        requested.update(kwargs)
        return {"tasks": [{"updated_at": remote_update.isoformat()}]}

    project = SimpleNamespace(get_paginated_tasks=get_paginated_tasks)
    interface = LabelStudioInterface("https://example.invalid", "token")

    assert interface.annotations_are_up_to_date(paths, project)
    assert requested == {
        "ordering": ["-updated_at"],
        "page": 1,
        "page_size": 1,
    }

    remote_update = local_creation + timedelta(seconds=1)
    assert not interface.annotations_are_up_to_date(paths, project)


def test_label_studio_setup_skips_download_when_page_xml_is_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = PathBundle(tmp_path)
    interface = LabelStudioInterface("https://example.invalid", "token")
    monkeypatch.setattr(interface, "annotations_are_up_to_date", lambda *args: True)

    def fail_fetch():
        pytest.fail("The full Label Studio export should not be downloaded.")

    monkeypatch.setattr(interface, "fetch_simplified_tasks", fail_fetch)

    interface.setup(paths)
