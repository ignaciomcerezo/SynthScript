import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import lxml.etree as etree  # ty: ignore[unresolved-import]
from shapely.affinity import scale as scale_geometry
from shapely.geometry import Polygon
from shapely.ops import unary_union

from synthscript.loading.page_xml.constants import (
    LABEL_STUDIO_CREATED_AT,
    LABEL_STUDIO_UPDATED_AT,
    NSMAP,
    PAGE_NS,
    PAGE_SCHEMA_URL,
    PAGE_XML_CREATED,
    PAGE_XML_CREATOR,
    PAGE_XML_LAST_CHANGE,
    SYNTHSCRIPT_BACKGROUND_IMAGE,
    SYNTHSCRIPT_ORIGINAL_ID,
    SYNTHSCRIPT_PAGE_ID,
    SYNTHSCRIPT_STROKE_IMAGE,
    XSI_NS,
    page_tag,
)
from synthscript.loading.page_xml.helpers import (
    make_page_xml_id,
    ocr_rotation_to_page_orientation,
    polygon_to_page_points,
)
from synthscript.loading.page_xml.validation import validate_page_xml
from synthscript.ocr_units import OCRPage
from synthscript.shared.path_bundle import PathBundle


def _relative_path(image_path: Path, xml_path: Path) -> str:
    image_path = image_path.resolve()
    try:
        relative = os.path.relpath(image_path, start=xml_path.parent.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Image {image_path} cannot be referenced relative to {xml_path}."
        ) from exc
    return Path(relative).as_posix()


def _as_metadata_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _page_datetime(value: object | None, *, fallback: datetime) -> str:
    if value is None:
        parsed = fallback
    elif isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid PAGE timestamp {value!r}.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _add_original_id(parent: etree._Element, original_id: str) -> None:
    user_defined = etree.SubElement(parent, page_tag("UserDefined"))
    etree.SubElement(
        user_defined,
        page_tag("UserAttribute"),
        name=SYNTHSCRIPT_ORIGINAL_ID,
        type="xsd:string",
        value=original_id,
    )


def _region_polygon(line_polygons: list[Polygon]) -> Polygon:
    """Use the exact union when polygonal, otherwise its deterministic convex hull."""
    union = unary_union(line_polygons)
    if isinstance(union, Polygon):
        return union
    hull = union.convex_hull
    if not isinstance(hull, Polygon) or hull.is_empty:
        raise ValueError("Paragraph lines do not form a usable region polygon.")
    return hull


def _validate_image_shapes(
    page: OCRPage,
    raw_image_path: Path,
    stroke_image_path: Path | None,
    background_image_path: Path | None,
) -> tuple[int, int]:
    raw = PathBundle.load_image_grayscale_np(raw_image_path)
    alternatives = {}
    if stroke_image_path is not None:
        alternatives["stroke"] = PathBundle.load_image_grayscale_np(stroke_image_path)
    if background_image_path is not None:
        alternatives["background"] = PathBundle.load_image_grayscale_np(
            background_image_path
        )

    expected = page.image_dimensions
    for role, image in alternatives.items():
        if image.shape[:2] != expected:
            raise ValueError(
                f"{role.capitalize()} image dimensions {image.shape[:2]} do not "
                f"match OCRPage dimensions {expected}."
            )
    if not alternatives and raw.shape[:2] != expected:
        raise ValueError(
            f"Raw image dimensions {raw.shape[:2]} do not match OCRPage "
            f"dimensions {expected}, and no processed alternative image was given."
        )
    return raw.shape[:2]


def save_page_xml(
    page: OCRPage,
    xml_path: Path,
    *,
    raw_image_path: Path,
    stroke_image_path: Path | None = None,
    background_image_path: Path | None = None,
    metadata: Mapping[str, object] | None = None,
    validate: bool = True,
) -> None:
    """Serialize one OCRPage as canonical PAGE-XML 2024-07-15."""
    xml_path = Path(xml_path)
    raw_image_path = Path(raw_image_path)
    stroke_image_path = Path(stroke_image_path) if stroke_image_path else None
    background_image_path = (
        Path(background_image_path) if background_image_path else None
    )
    for image_path in (raw_image_path, stroke_image_path, background_image_path):
        if image_path is not None and not image_path.is_file():
            raise ValueError(f"Referenced image does not exist: {image_path}")

    height, width = _validate_image_shapes(
        page,
        raw_image_path,
        stroke_image_path,
        background_image_path,
    )
    page_height, page_width = page.image_dimensions
    x_scale = width / page_width
    y_scale = height / page_height

    def page_polygon(polygon: Polygon) -> Polygon:
        return scale_geometry(
            polygon,
            xfact=x_scale,
            yfact=y_scale,
            origin=(0, 0),
        )

    combined_metadata: dict[str, object] = dict(page.metadata)
    combined_metadata.update(metadata or {})
    combined_metadata[SYNTHSCRIPT_PAGE_ID] = page.page_id

    now = datetime.now(timezone.utc)
    creator = _as_metadata_value(combined_metadata.get(PAGE_XML_CREATOR, "SynthScript"))
    created = _page_datetime(
        combined_metadata.get(PAGE_XML_CREATED)
        or combined_metadata.get(LABEL_STUDIO_CREATED_AT),
        fallback=now,
    )
    last_change = _page_datetime(
        combined_metadata.get(PAGE_XML_LAST_CHANGE)
        or combined_metadata.get(LABEL_STUDIO_UPDATED_AT),
        fallback=now,
    )

    used_ids: set[str] = set()
    pcgts_id = make_page_xml_id(page.page_id, prefix="pcgts_", used_ids=used_ids)
    root = etree.Element(page_tag("PcGts"), nsmap=NSMAP, pcGtsId=pcgts_id)
    root.set(
        etree.QName(XSI_NS, "schemaLocation"),
        f"{PAGE_NS} {PAGE_SCHEMA_URL}",
    )

    metadata_element = etree.SubElement(root, page_tag("Metadata"))
    etree.SubElement(metadata_element, page_tag("Creator")).text = creator
    etree.SubElement(metadata_element, page_tag("Created")).text = created
    etree.SubElement(metadata_element, page_tag("LastChange")).text = last_change
    structural_keys = {PAGE_XML_CREATOR, PAGE_XML_CREATED, PAGE_XML_LAST_CHANGE}
    for name in sorted(combined_metadata):
        value = combined_metadata[name]
        if name in structural_keys or value is None:
            continue
        etree.SubElement(
            metadata_element,
            page_tag("MetadataItem"),
            type="other",
            name=name,
            value=_as_metadata_value(value),
        )

    page_element = etree.SubElement(
        root,
        page_tag("Page"),
        imageFilename=_relative_path(raw_image_path, xml_path),
        imageWidth=str(width),
        imageHeight=str(height),
    )
    if stroke_image_path is not None:
        etree.SubElement(
            page_element,
            page_tag("AlternativeImage"),
            filename=_relative_path(stroke_image_path, xml_path),
            comments=SYNTHSCRIPT_STROKE_IMAGE,
        )
    if background_image_path is not None:
        etree.SubElement(
            page_element,
            page_tag("AlternativeImage"),
            filename=_relative_path(background_image_path, xml_path),
            comments=SYNTHSCRIPT_BACKGROUND_IMAGE,
        )

    region_elements: list[tuple[etree._Element, str]] = []
    for region_index, paragraph in enumerate(page.paragraphs):
        original_region_id = paragraph.id or f"region_{region_index}"
        region_id = make_page_xml_id(
            original_region_id,
            prefix="region_",
            used_ids=used_ids,
        )
        region = etree.Element(
            page_tag("TextRegion"),
            id=region_id,
            type="paragraph",
        )
        etree.SubElement(
            region,
            page_tag("Coords"),
            points=polygon_to_page_points(
                page_polygon(
                    _region_polygon([line.polygon for line in paragraph.lines])
                )
            ),
        )
        if region_id != original_region_id:
            _add_original_id(region, original_region_id)

        for line_index, line in enumerate(paragraph.lines):
            line_id = make_page_xml_id(
                line.id,
                prefix="line_",
                used_ids=used_ids,
            )
            line_element = etree.SubElement(
                region,
                page_tag("TextLine"),
                id=line_id,
                index=str(line_index),
                orientation=f"{ocr_rotation_to_page_orientation(line.rotation):.12g}",
            )
            etree.SubElement(
                line_element,
                page_tag("Coords"),
                points=polygon_to_page_points(page_polygon(line.polygon)),
            )
            text_equiv = etree.SubElement(
                line_element,
                page_tag("TextEquiv"),
                index="0",
            )
            etree.SubElement(text_equiv, page_tag("Unicode")).text = line.text
            if line_id != line.id:
                _add_original_id(line_element, line.id)

        region_text_equiv = etree.SubElement(
            region,
            page_tag("TextEquiv"),
            index="0",
        )
        etree.SubElement(
            region_text_equiv, page_tag("Unicode")
        ).text = paragraph.transcription(page.line_separator)
        region_elements.append((region, region_id))

    if region_elements:
        reading_order = etree.SubElement(page_element, page_tag("ReadingOrder"))
        ordered_group = etree.SubElement(
            reading_order,
            page_tag("OrderedGroup"),
            id=make_page_xml_id(
                "reading_order",
                prefix="group_",
                used_ids=used_ids,
            ),
        )
        for index, (_, region_id) in enumerate(region_elements):
            etree.SubElement(
                ordered_group,
                page_tag("RegionRefIndexed"),
                regionRef=region_id,
                index=str(index),
            )

    for region, _ in region_elements:
        page_element.append(region)

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = xml_path.with_suffix(f"{xml_path.suffix}.tmp")
    etree.ElementTree(root).write(
        str(temporary_path),
        encoding="UTF-8",
        xml_declaration=True,
        pretty_print=True,
    )
    try:
        if validate:
            validate_page_xml(temporary_path)
        temporary_path.replace(xml_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
