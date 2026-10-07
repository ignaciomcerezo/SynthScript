import re
import warnings
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import cv2
import lxml.etree as etree  # ty: ignore[unresolved-import]
import numpy as np
from shapely.affinity import scale as scale_geometry

from synthscript.loading.page_xml.constants import (
    LABEL_STUDIO_TASK_ID,
    PAGE_NS,
    PAGE_XML_CREATED,
    PAGE_XML_CREATOR,
    PAGE_XML_LAST_CHANGE,
    SYNTHSCRIPT_BACKGROUND_IMAGE,
    SYNTHSCRIPT_ORIGINAL_ID,
    SYNTHSCRIPT_PAGE_ID,
    SYNTHSCRIPT_STROKE_IMAGE,
    page_tag,
)
from synthscript.loading.page_xml.helpers import (
    page_orientation_to_ocr_rotation,
    parse_page_points,
    polygon_from_page_points,
)
from synthscript.loading.page_xml.validation import (
    parse_xml_document,
    validate_page_xml_document,
)
from synthscript.metric.homogenizer.homogenizer import TextHomogenizer
from synthscript.ocr_units import OCRPage
from synthscript.shared.geometry_processing import calculate_reading_angle
from synthscript.shared.image_handling import read_image_shape
from synthscript.shared.path_bundle import PathBundle

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[/\\]")


def _resolve_image_path(xml_path: Path, value: str) -> Path:
    posix_path = PurePosixPath(value)
    if "\\" in value or posix_path.is_absolute() or _WINDOWS_ABSOLUTE_PATH.match(value):
        raise ValueError(
            f"PAGE image paths must be relative POSIX paths, got {value!r}."
        )
    resolved = xml_path.parent.joinpath(*posix_path.parts).resolve()
    if not resolved.is_file():
        raise ValueError(f"PAGE image does not exist: {resolved}")
    return resolved


def _metadata(root: etree._Element) -> dict[str, str]:
    metadata_element = root.find(page_tag("Metadata"))
    if metadata_element is None:
        raise ValueError("PAGE Metadata element is missing.")

    result: dict[str, str] = {}
    structural = {
        "Creator": PAGE_XML_CREATOR,
        "Created": PAGE_XML_CREATED,
        "LastChange": PAGE_XML_LAST_CHANGE,
    }
    for element_name, key in structural.items():
        element = metadata_element.find(page_tag(element_name))
        if element is not None and element.text is not None:
            result[key] = element.text
    for item in metadata_element.findall(page_tag("MetadataItem")):
        name = item.get("name")
        value = item.get("value")
        if name and value is not None:
            result[name] = value
    return result


def _original_id(element: etree._Element) -> str:
    for attribute in element.findall(
        f"{page_tag('UserDefined')}/{page_tag('UserAttribute')}"
    ):
        if attribute.get("name") == SYNTHSCRIPT_ORIGINAL_ID:
            value = attribute.get("value")
            if value:
                return value
    page_id = element.get("id")
    if not page_id:
        raise ValueError(f"PAGE {etree.QName(element).localname} is missing an ID.")
    return page_id


def _text_equiv(line_element: etree._Element) -> str:
    candidates = line_element.findall(page_tag("TextEquiv"))
    if not candidates:
        raise ValueError(f"TextLine {line_element.get('id')!r} has no TextEquiv.")

    indexed: list[tuple[int, etree._Element]] = []
    for candidate in candidates:
        index = candidate.get("index")
        if index is not None:
            indexed.append((int(index), candidate))
    selected = min(indexed, key=lambda item: item[0])[1] if indexed else candidates[0]
    unicode_element = selected.find(page_tag("Unicode"))
    if unicode_element is None:
        raise ValueError(f"TextLine {line_element.get('id')!r} has no Unicode text.")
    return unicode_element.text or ""


def _line_elements(region: etree._Element) -> list[etree._Element]:
    lines = region.findall(page_tag("TextLine"))
    seen_indices: set[int] = set()
    parsed_indices: list[int | None] = []
    for line in lines:
        raw_index = line.get("index")
        if raw_index is None:
            parsed_indices.append(None)
            continue
        index = int(raw_index)
        if index < 0:
            raise ValueError("TextLine@index cannot be negative.")
        if index in seen_indices:
            raise ValueError(
                f"Duplicate TextLine@index {index} in region {region.get('id')!r}."
            )
        seen_indices.add(index)
        parsed_indices.append(index)

    if any(index is not None for index in parsed_indices):
        return [
            line
            for _, line in sorted(
                enumerate(lines),
                key=lambda item: (
                    parsed_indices[item[0]] is None,
                    (
                        parsed_indices[item[0]]
                        if parsed_indices[item[0]] is not None
                        else item[0]
                    ),
                ),
            )
        ]
    return lines


def _reading_order_references(
    group: etree._Element,
    *,
    all_document_ids: set[str | None],
) -> list[str]:
    local_name = etree.QName(group).localname
    if local_name in {"RegionRef", "RegionRefIndexed"}:
        region_ref = group.get("regionRef")
        if region_ref not in all_document_ids:
            raise ValueError(f"ReadingOrder references unknown region {region_ref!r}.")
        assert region_ref is not None
        return [region_ref]

    members = [
        child
        for child in group
        if etree.QName(child).localname
        not in {
            "UserDefined",
            "Labels",
        }
    ]
    if local_name in {"OrderedGroup", "OrderedGroupIndexed"}:
        indexed_members: list[tuple[int, etree._Element]] = []
        seen_indices: set[int] = set()
        for member in members:
            raw_index = member.get("index")
            if raw_index is None:
                raise ValueError("An ordered ReadingOrder member has no index.")
            index = int(raw_index)
            if index < 0 or index in seen_indices:
                raise ValueError(f"Invalid or duplicate ReadingOrder index {index}.")
            seen_indices.add(index)
            indexed_members.append((index, member))
        members = [
            member for _, member in sorted(indexed_members, key=lambda item: item[0])
        ]

    references: list[str] = []
    for member in members:
        references.extend(
            _reading_order_references(
                member,
                all_document_ids=all_document_ids,
            )
        )
    return references


def _ordered_regions(
    page_element: etree._Element,
    regions: list[etree._Element],
) -> list[etree._Element]:
    by_id = {region.get("id"): region for region in regions}
    all_document_ids = {
        element.get("id")
        for element in page_element.iter()
        if element.get("id") is not None
    }
    reading_order = page_element.find(page_tag("ReadingOrder"))
    if reading_order is None:
        return regions

    groups = list(reading_order)
    if len(groups) != 1:
        raise ValueError("ReadingOrder must contain exactly one root group.")
    references = _reading_order_references(
        groups[0],
        all_document_ids=all_document_ids,
    )
    if len(references) != len(set(references)):
        raise ValueError("ReadingOrder references a region more than once.")

    result: list[etree._Element] = []
    for region_ref in references:
        if region_ref in by_id:
            result.append(by_id[region_ref])
        else:
            warnings.warn(
                "Ignoring ReadingOrder reference to unsupported region "
                f"{region_ref!r}.",
                stacklevel=2,
            )
    result.extend(region for region in regions if region not in result)
    return result


def _load_images(
    page_element: etree._Element,
    xml_path: Path,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    raw_filename = page_element.get("imageFilename")
    if not raw_filename:
        raise ValueError("PAGE imageFilename is missing.")
    raw_path = _resolve_image_path(xml_path, raw_filename)

    alternatives: dict[str, np.ndarray] = {}
    for alternative in page_element.findall(page_tag("AlternativeImage")):
        role = alternative.get("comments")
        if role not in {SYNTHSCRIPT_STROKE_IMAGE, SYNTHSCRIPT_BACKGROUND_IMAGE}:
            continue
        if role in alternatives:
            raise ValueError(f"Duplicate PAGE AlternativeImage role {role!r}.")
        filename = alternative.get("filename")
        if not filename:
            raise ValueError(f"AlternativeImage {role!r} has no filename.")
        alternatives[role] = PathBundle.load_image_grayscale_np(
            _resolve_image_path(xml_path, filename)
        )

    stroke = alternatives.get(SYNTHSCRIPT_STROKE_IMAGE)
    background = alternatives.get(SYNTHSCRIPT_BACKGROUND_IMAGE)
    raw: np.ndarray | None = None
    if stroke is not None and background is not None:
        raw_shape = read_image_shape(raw_path)
    else:
        raw = PathBundle.load_image_grayscale_np(raw_path)
        raw_shape = raw.shape[:2]
    expected = (
        int(page_element.get("imageHeight", "-1")),
        int(page_element.get("imageWidth", "-1")),
    )
    if raw_shape != expected:
        raise ValueError(
            f"Raw image dimensions {raw_shape} do not match PAGE dimensions "
            f"{expected}."
        )

    if stroke is None and background is None:
        assert raw is not None
        background = np.full_like(raw, 255)
        stroke = np.subtract(background, raw)
    else:
        reference = stroke if stroke is not None else background
        assert reference is not None
        target_shape = reference.shape[:2]
        if stroke is not None and stroke.shape[:2] != target_shape:
            raise ValueError("Stroke and background alternative image sizes differ.")
        if background is not None and background.shape[:2] != target_shape:
            raise ValueError("Stroke and background alternative image sizes differ.")
        target_height, target_width = target_shape
        if stroke is None or background is None:
            assert raw is not None
            resized_raw = (
                raw
                if raw.shape[:2] == target_shape
                else cv2.resize(
                    raw,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            )

    if stroke is None:
        assert background is not None
        stroke = np.clip(
            background.astype(np.int16) - resized_raw.astype(np.int16), 0, 255
        ).astype(np.uint8)
    elif background is None:
        background = np.clip(
            resized_raw.astype(np.int16) + stroke.astype(np.int16), 0, 255
        ).astype(np.uint8)

    if stroke.shape != background.shape:
        raise ValueError("Stroke and background image sizes differ.")
    target_height, target_width = stroke.shape[:2]
    return (
        stroke,
        background,
        target_width / expected[1],
        target_height / expected[0],
    )


def page_xml_identity(document: etree._ElementTree) -> tuple[str, int | None]:
    """Return the page and LabelStudio task identities from a (parsed) document"""
    root = document.getroot()
    if etree.QName(root).namespace != PAGE_NS:
        raise ValueError(f"Unsupported PAGE namespace {etree.QName(root).namespace!r}.")

    metadata = _metadata(root)
    page_element = root.find(page_tag("Page"))
    if page_element is None:
        raise ValueError("PAGE document has no Page element.")

    raw_filename = page_element.get("imageFilename", "page")
    page_id = metadata.get(SYNTHSCRIPT_PAGE_ID, PurePosixPath(raw_filename).stem)
    raw_task_id = metadata.get(LABEL_STUDIO_TASK_ID)
    try:
        task_id = int(raw_task_id) if raw_task_id is not None else None
    except ValueError:
        task_id = None
    return page_id, task_id


def load_page_xml_document(
    document: etree._ElementTree,
    xml_path: Path,
    *,
    transcription_homogenizer: TextHomogenizer | Callable[[str], str] | None = None,
) -> OCRPage:
    """Load a parsed PAGE document into SynthScript's in-memory model."""
    validate_page_xml_document(document, xml_path)
    root = document.getroot()
    if etree.QName(root).namespace != PAGE_NS:
        raise ValueError(f"Unsupported PAGE namespace {etree.QName(root).namespace!r}.")

    metadata = _metadata(root)
    page_element = root.find(page_tag("Page"))
    if page_element is None:
        raise ValueError("PAGE document has no Page element.")
    stroke, background, x_scale, y_scale = _load_images(page_element, xml_path)

    supported_page_children = {
        "AlternativeImage",
        "ReadingOrder",
        "TextRegion",
    }
    for child in page_element:
        local_name = etree.QName(child).localname
        if local_name not in supported_page_children:
            warnings.warn(
                f"Ignoring unsupported PAGE element {local_name}.",
                stacklevel=2,
            )

    regions = _ordered_regions(
        page_element,
        page_element.findall(page_tag("TextRegion")),
    )
    transcriptions: list[str] = []
    polygon_coords: list[list[tuple[float, float]]] = []
    line_ids: list[str] = []
    rotations: list[float] = []
    paragraph_line_ids: list[list[str]] = []
    paragraph_ids: list[str | None] = []
    seen_line_ids: set[str] = set()

    for region in regions:
        region_line_ids: list[str] = []
        for line in _line_elements(region):
            line_id = _original_id(line)
            if line_id in seen_line_ids:
                raise ValueError(f"Duplicate logical TextLine ID {line_id!r}.")
            seen_line_ids.add(line_id)

            coords = line.find(page_tag("Coords"))
            if coords is None or coords.get("points") is None:
                raise ValueError(f"TextLine {line_id!r} has no Coords.")
            polygon = polygon_from_page_points(
                parse_page_points(coords.get("points", ""))
            )
            polygon = scale_geometry(
                polygon,
                xfact=x_scale,
                yfact=y_scale,
                origin=(0, 0),
            )
            text = _text_equiv(line)
            if transcription_homogenizer is not None:
                text = transcription_homogenizer(text)

            raw_orientation = line.get("orientation")
            rotation = (
                calculate_reading_angle(polygon)
                if raw_orientation is None
                else page_orientation_to_ocr_rotation(float(raw_orientation))
            )
            transcriptions.append(text)
            polygon_coords.append(list(polygon.exterior.coords)[:-1])
            line_ids.append(line_id)
            rotations.append(rotation)
            region_line_ids.append(line_id)

        if not region_line_ids:
            warnings.warn(
                f"Ignoring empty TextRegion {region.get('id')!r}.",
                stacklevel=2,
            )
            continue
        paragraph_line_ids.append(region_line_ids)
        paragraph_ids.append(_original_id(region))

    raw_filename = page_element.get("imageFilename", "page")
    page_id = metadata.get(SYNTHSCRIPT_PAGE_ID, PurePosixPath(raw_filename).stem)
    return OCRPage.from_explicit_layout(
        transcriptions=transcriptions,
        polygon_coords=polygon_coords,
        line_ids=line_ids,
        rotations=rotations,
        page_id=page_id,
        stroke=stroke,
        background=background,
        metadata=metadata,
        paragraph_line_ids=paragraph_line_ids,
        paragraph_ids=paragraph_ids,
    )


def load_page_xml(
    xml_path: Path,
    paths: PathBundle,
    *,
    transcription_homogenizer: TextHomogenizer | Callable[[str], str] | None = None,
) -> OCRPage:
    """Load the supported PAGE subset into SynthScript's in-memory model."""
    del paths  # Image references are intentionally resolved relative to the XML file.
    xml_path = Path(xml_path)
    document = parse_xml_document(xml_path)
    return load_page_xml_document(
        document,
        xml_path,
        transcription_homogenizer=transcription_homogenizer,
    )
