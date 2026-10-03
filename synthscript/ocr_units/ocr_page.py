import json
import math
from collections.abc import Callable
from copy import deepcopy
from typing import Literal

import cv2
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon

from synthscript.ocr_units.ocr_line import OCRLine
from synthscript.ocr_units.ocr_paragraph import OCRParagraph
from synthscript.shared.geometry_processing import get_union_rect
from synthscript.shared.graph_utilities import (
    get_connected_components,
    subdictionary,
)
from synthscript.shared.image_processing import (
    crop_image_with_polygon,
    crop_or_resize,
    to_grayscale,
)

ocr_transform = Callable[
    [list[tuple[list[np.ndarray], list[Polygon]]]],
    tuple[list[np.ndarray], list[Polygon]],
]
StrokeTransformCallable = Callable[[list[np.ndarray]], list[np.ndarray]]
ImageTransformCallable = Callable[[np.ndarray], np.ndarray]


class OCRPage:
    """
    In-memory annotated page with lines, paragraphs, and synthetic sampling.
    """

    n_annotation_errors: int = 0

    __slots__ = (
        "_graph",
        "background",
        "full_transcription",
        "line_separator",
        "lines",
        "metadata",
        "page_id",
        "paragraphs",
    )

    def __init__(
        self,
        *,
        transcriptions: list[str],
        polygon_coords: list[list[tuple[float, float]]],
        line_ids: list[str],
        rotations: list[float],
        page_id: str,
        stroke: np.ndarray,
        background: np.ndarray,
        line_separtor: str = "\n",
        metadata: dict[str, str] | None = None,
        paragraph_line_ids: list[list[str]] | None = None,
        paragraph_ids: list[str | None] | None = None,
    ):
        stroke = to_grayscale(stroke)
        background = to_grayscale(background)
        if stroke.shape != background.shape:
            raise ValueError("Stroke and background must have the same dimensions.")
        self.background = background
        self.page_id = str(page_id)
        self.metadata = dict(metadata or {})
        self.line_separator = line_separtor

        list_lines = self._setup_lines(
            polygon_coords,
            transcriptions,
            line_ids,
            rotations,
            stroke,
        )

        self.lines = {line.id: line for line in list_lines}

        self._setup_intersection_graph()
        if paragraph_line_ids is None:
            self._infer_paragraphs_from_geometry()
        else:
            self._setup_paragraphs_from_explicit_layout(
                paragraph_line_ids,
                paragraph_ids=paragraph_ids,
            )

        # only pages that lay inside of a paragraph have an sindex
        self._set_indices_and_transcription()

    @classmethod
    def from_explicit_layout(
        cls,
        *,
        paragraph_line_ids: list[list[str]],
        paragraph_ids: list[str | None] | None = None,
        **kwargs,
    ) -> "OCRPage":
        """Build a page while preserving the supplied region and line order."""
        return cls(
            paragraph_line_ids=paragraph_line_ids,
            paragraph_ids=paragraph_ids,
            **kwargs,
        )

    @property
    def page(self) -> str:
        """Compatibility alias for the generic page identity."""
        return self.page_id

    @property
    def image_dimensions(self) -> tuple[int, int]:
        """Returns the dimensions of the background in the format (height, width)."""
        return self.background.shape[:2]

    @classmethod
    def combine_annotations(cls, *annotations: "OCRPage") -> "OCRPage":
        """
        Combines several AnnotatedPage instances into a single new one.

        Paragraph contents remain untouched. The combined paragraphs are ordered from
        top to bottom using each paragraph's topmost line.

        All annotations must describe the same physical page. Conflicting provenance
        is retained in ``synthscript.combined_annotations``.
        """
        if not annotations:
            raise ValueError("combine_annotations needs at least one AnnotatedPage.")

        all_paragraphs = [
            paragraph
            for annotation in annotations
            for paragraph in annotation.paragraphs
        ]

        if len({ann.page_id for ann in annotations}) != 1:
            raise ValueError("Can only combine annotations from the same page.")
        if len({ann.image_dimensions for ann in annotations}) != 1:
            raise ValueError("Cannot combine annotations with different image sizes.")
        background = annotations[0].background
        all_paragraphs.sort(key=lambda paragraph: paragraph.top)

        first = annotations[0]

        combined_ocr_page: OCRPage = object.__new__(OCRPage)

        combined_ocr_page._graph = {}

        for other in annotations:
            combined_ocr_page._graph.update(other.graph)

        combined_ocr_page.paragraphs = sorted(
            deepcopy(all_paragraphs), key=lambda paragraph: paragraph.lines[0].top
        )
        for index, paragraph in enumerate(combined_ocr_page.paragraphs):
            paragraph.index = index
        combined_ocr_page.line_separator = first.line_separator
        combined_ocr_page.background = background
        combined_ocr_page.page_id = first.page_id
        combined_ocr_page.metadata = cls._merge_metadata(annotations)

        combined_ocr_page.lines = {}
        for paragraph in combined_ocr_page.paragraphs:
            for line in paragraph.lines:
                if line.id in combined_ocr_page.lines:
                    raise ValueError(
                        f"Duplicate line ID {line.id!r} while combining paragraphs "
                        "into a single AnnotatedPage."
                    )
                combined_ocr_page.lines[line.id] = line

        combined_ocr_page._set_indices_and_transcription()

        return combined_ocr_page

    @staticmethod
    def _merge_metadata(annotations: tuple["OCRPage", ...]) -> dict[str, str]:
        keys = set().union(*(annotation.metadata for annotation in annotations))
        merged: dict[str, str] = {}
        for key in keys:
            values = {
                annotation.metadata[key]
                for annotation in annotations
                if key in annotation.metadata
            }
            if len(values) == 1:
                merged[key] = values.pop()
        if any(annotation.metadata != merged for annotation in annotations):
            merged["synthscript.combined_annotations"] = json.dumps(
                [annotation.metadata for annotation in annotations],
                ensure_ascii=False,
                sort_keys=True,
            )
        return merged

    @property
    def order(self) -> int:
        """Total number of lines"""
        return len(self.graph)

    @property
    def graph(self) -> dict[str, set[str]]:
        """Return the line-polygon intersection graph, keyed by line ID."""
        return self._graph

    def _setup_lines(
        self,
        polygon_coords: list[list[tuple[float, float]]],
        transcriptions: list[str],
        line_ids: list[str],
        rotations: list[float],
        stroke: np.ndarray,
    ) -> list[OCRLine]:

        if not (
            len(
                {
                    len(polygon_coords),
                    len(transcriptions),
                    len(line_ids),
                    len(rotations),
                }
            )
            == 1
        ):
            raise ValueError(
                "polygon_coords, transcriptions, line_ids, and rotations must "
                "have equal lengths."
            )

        if len(set(line_ids)) != len(line_ids):
            raise ValueError("Line IDs must be unique within an OCRPage.")

        polygons: list[Polygon] = []
        for polygon_coord in polygon_coords:
            if len(polygon_coord) < 3:
                raise ValueError("A line polygon needs at least three points.")
            if any(
                not math.isfinite(value) or value < 0
                for point in polygon_coord
                for value in point
            ):
                raise ValueError(
                    "Line polygon coordinates must be finite and non-negative."
                )
            polygon = Polygon(polygon_coord)
            if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                raise ValueError("A line polygon must be non-empty and valid.")
            polygons.append(polygon)

        if any(not math.isfinite(float(rotation)) for rotation in rotations):
            raise ValueError("Line rotations must be finite.")

        lines = []

        for polygon, transcription, line_id, rotation in zip(
            polygons, transcriptions, line_ids, rotations, strict=True
        ):
            stroke_crop = crop_image_with_polygon(stroke, polygon)
            lines.append(
                OCRLine(
                    id=line_id,
                    crop=stroke_crop,
                    polygon=polygon,
                    rotation=rotation,
                    page_id=self.page_id,
                    text=transcription,
                )
            )

        return lines

    def _setup_intersection_graph(self) -> None:
        """Build the line-polygon intersection graph used by transforms."""
        lines = list(self.lines.values())

        adj: dict[str, set[str]] = {line.id: set() for line in lines}

        for i, line_a in enumerate(lines):
            for line_b in lines[i + 1 :]:
                if line_a.polygon.intersects(line_b.polygon):
                    adj[line_a.id].add(line_b.id)
                    adj[line_b.id].add(line_a.id)

        self._graph = adj

    def _infer_paragraphs_from_geometry(self) -> None:
        """Infer paragraphs and reading order from connected line polygons."""
        adj = self.graph

        connected_components = get_connected_components(adj)

        line_ccs = [
            [self.lines[line_id] for line_id in component]
            for component in connected_components
        ]

        line_ccs.sort(key=lambda line_cc: min(line.top for line in line_cc))

        line_id_ccs = [
            subdictionary([line.id for line in line_cc], self.graph)
            for line_cc in line_ccs
        ]

        self.paragraphs = [
            OCRParagraph(
                lines=line_cc,
                page_id=self.page_id,
                subgraph=line_ids_cc,
                id=f"region_{idx}",
                index=idx,
            )
            for (idx, (line_cc, line_ids_cc)) in enumerate(
                zip(line_ccs, line_id_ccs, strict=True)
            )
        ]

    def _setup_paragraphs_from_explicit_layout(
        self,
        paragraph_line_ids: list[list[str]],
        *,
        paragraph_ids: list[str | None] | None,
    ) -> None:
        if paragraph_ids is None:
            paragraph_ids = [None] * len(paragraph_line_ids)
        if len(paragraph_ids) != len(paragraph_line_ids):
            raise ValueError("paragraph_ids must match paragraph_line_ids in length.")

        flattened = [line_id for group in paragraph_line_ids for line_id in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("A line cannot occur in more than one paragraph.")
        if set(flattened) != set(self.lines):
            raise ValueError(
                "Explicit layout must contain every OCRPage line exactly once."
            )

        self.paragraphs = []
        for index, (region_line_ids, paragraph_id) in enumerate(
            zip(paragraph_line_ids, paragraph_ids, strict=True)
        ):
            if not region_line_ids:
                raise ValueError("Explicit paragraphs cannot be empty.")
            self.paragraphs.append(
                OCRParagraph(
                    lines=[self.lines[line_id] for line_id in region_line_ids],
                    page_id=self.page_id,
                    subgraph=subdictionary(region_line_ids, self.graph),
                    id=paragraph_id or f"region_{index}",
                    index=index,
                    preserve_order=True,
                )
            )

    def _set_indices_and_transcription(self) -> None:
        """Set paragraph/line indices and assemble the page transcription."""
        sindex = 0
        for paragraph_index, paragraph in enumerate(self.paragraphs):
            paragraph.index = paragraph_index
            for line in paragraph.lines:
                line.sindex = sindex
                sindex += len(line.text) + len(self.line_separator)
        lines = sorted(list(self.lines.values()), key=lambda line: line.sindex)  # ty: ignore[no-matching-overload]
        self.full_transcription = self.line_separator.join(line.text for line in lines)

    def __repr__(self):
        return f"<OCRPage {self.page_id!r} with {self.order} lines>"

    def synthetic_starting_index(
        self, line_ids: set[str] | list[str] | Literal["all"]
    ) -> int:
        if None in set(self.lines[line_id].sindex for line_id in line_ids):
            raise ValueError(
                "Cannot compute transcription or sindex for unordered lines."
            )
        starting_index: int = min(
            self.lines[line_id].sindex
            for line_id in line_ids  # ty: ignore[invalid-argument-type]
        )

        return starting_index

    def synthetic_transcription(
        self,
        line_ids: set[str] | list[str] | Literal["all"],
    ) -> str:
        lines = (
            [self.lines[line_id] for line_id in line_ids]
            if line_ids != "all"
            else list(self.lines.values())
        )

        # sindex follows the preserved or inferred reading order.
        lines: list[OCRLine] = sorted(lines, key=lambda x: x.sindex)  # ty: ignore[no-matching-overload]

        return self.line_separator.join([line.text for line in lines])

    def synthetic_manuscript(
        self,
        line_ids: set[str] | list[str] | Literal["all"],
        *,
        tight_layout: bool = True,
        margin_size_px: int | dict[Literal["left", "right", "top", "bottom"], int] = 0,
        img_poly_transform: ocr_transform | None = None,
        stroke_transform: StrokeTransformCallable | None = None,
        background_transform: ImageTransformCallable | None = None,
        global_image_transform: ImageTransformCallable | None = None,
        refit_polygons: bool = True,
        overlay_polygons: bool = False,
        overlay_mbr: bool = False,
    ) -> tuple[np.ndarray, list[Polygon]]:

        if isinstance(margin_size_px, int):
            margin_size_px = {
                x: margin_size_px for x in ["left", "right", "top", "bottom"]
            }
        else:
            if not (set(margin_size_px.keys()) == {"left", "right", "top", "bottom"}):
                raise ValueError(
                    "margin_size_px must be an int or include left, right, top, "
                    f"and bottom; got {margin_size_px.keys()}."
                )
        if not all(val >= 0 for val in margin_size_px.values()):
            raise ValueError("The margin size cannot be negative.")

        if line_ids == "all":
            line_ids = set(self.lines.keys())

        if not isinstance(line_ids, (set, list)):
            raise ValueError(
                "line_ids must be a set[str], list[str], or 'all'; got "
                f"{type(line_ids)}"
            )
        if len(line_ids) != len(set(line_ids)):
            raise ValueError("Duplicate line_ids passed to synthetic_manuscript.")

        if img_poly_transform is not None:
            line_groups = self._group_sorted_by_paragraph(
                sorted(  # ty: ignore[no-matching-overload]
                    [self.lines[box_id] for box_id in line_ids],
                    key=lambda line: line.sindex,
                )
            )
            paragraph_equivalent_pairs = [
                (
                    [line.crop for line in line_group],
                    [line.polygon for line in line_group],
                )
                for line_group in line_groups
            ]
            crops, polygons = img_poly_transform(paragraph_equivalent_pairs)
        else:
            polygons = [self.lines[line_id].polygon for line_id in line_ids]
            crops = [self.lines[line_id].crop for line_id in line_ids]

        if stroke_transform is not None:
            crops = stroke_transform(crops)

        min_x, min_y, max_x, max_y = get_union_rect(polygons)
        bg_h, bg_w = self.image_dimensions

        if tight_layout:
            x0 = int(min_x) - margin_size_px["left"]
            xf = int(max_x) + 1 + margin_size_px["right"]
            y0 = int(min_y) - margin_size_px["top"]
            yf = int(max_y) + 1 + margin_size_px["bottom"]
            can_crop = True
        else:
            x0 = min(0, int(min_x) - margin_size_px["left"])
            xf = max(bg_w, int(max_x) + 1 + margin_size_px["right"])
            y0 = min(0, int(min_y) - margin_size_px["top"])
            yf = max(bg_h, int(max_y) + 1 + margin_size_px["bottom"])
            can_crop = False

        bg_np = np.asarray(self.background)
        canvas = crop_or_resize(
            bg_np, x0=x0, xf=xf, y0=y0, yf=yf, can_crop=can_crop
        ).copy()
        if background_transform is not None:
            canvas = background_transform(canvas)

        canvas_h, canvas_w = canvas.shape[:2]
        for stroke_img, polygon in zip(crops, polygons, strict=True):
            poly_x0, poly_y0, _, _ = polygon.bounds

            paste_x = int(poly_x0 - x0)
            paste_y = int(poly_y0 - y0)

            sh, sw = stroke_img.shape[:2]

            src_x0 = max(0, -paste_x)
            src_y0 = max(0, -paste_y)
            src_x1 = min(sw, canvas_w - paste_x)
            src_y1 = min(sh, canvas_h - paste_y)

            dst_x0 = max(0, paste_x)
            dst_y0 = max(0, paste_y)
            dst_x1 = min(canvas_w, paste_x + sw)
            dst_y1 = min(canvas_h, paste_y + sh)

            if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
                continue

            stroke_crop = stroke_img[src_y0:src_y1, src_x0:src_x1]
            if stroke_crop.ndim == 2:
                stroke_bgra = cv2.cvtColor(stroke_crop, cv2.COLOR_GRAY2BGRA)
            elif stroke_crop.ndim == 3 and stroke_crop.shape[2] == 4:
                stroke_bgra = stroke_crop
            else:
                raise ValueError(
                    "Transformed stroke crops must be grayscale or BGRA; "
                    f"got {stroke_crop.shape}."
                )

            stroke_value = stroke_bgra[..., 0].astype(np.float32)
            alpha = stroke_bgra[..., 3].astype(np.float32) / 255.0
            masked_stroke = stroke_value * alpha

            # Perform the blend strictly on the slice
            roi = canvas[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
            blended_roi = np.clip(roi - masked_stroke, 0, 255)

            canvas[dst_y0:dst_y1, dst_x0:dst_x1] = blended_roi.astype(np.uint8)

        if global_image_transform is not None:
            canvas = global_image_transform(canvas)

        if refit_polygons:
            # displace the polygons to the new dimensions of the image
            polygons = [translate(polygon, -x0, -y0) for polygon in polygons]

        if overlay_mbr or overlay_polygons:
            canvas = self._overlay_polygons_mbr(
                refitted_polygons=(
                    polygons
                    if refit_polygons
                    else [translate(polygon, -x0, -y0) for polygon in polygons]
                ),
                manuscript=canvas,
                overlay_polygons=overlay_polygons,
                overlay_mbr=overlay_mbr,
            )

        return canvas, polygons

    def _group_sorted_by_paragraph(self, lines: list[OCRLine]) -> list[list[OCRLine]]:
        """
        Groups lines by paragraph, assuming they are sorted by paragraph.
        """

        line_groups: list[list[OCRLine]] = []
        last_paragraph = None
        group = []

        for line in lines:
            if line.paragraph_index != last_paragraph:
                line_groups.append(group)
                group = []
            group.append(line)
            last_paragraph = line.paragraph_index

        line_groups.append(group)
        return [group for group in line_groups if group]

    @staticmethod
    def _overlay_polygons_mbr(
        *,
        refitted_polygons: list[Polygon],
        manuscript: np.ndarray,
        overlay_polygons: bool,
        overlay_mbr: bool,
    ):
        img = cv2.cvtColor(manuscript, cv2.COLOR_GRAY2BGR)

        for polygon in refitted_polygons:
            if overlay_polygons:
                poly_pts = np.round(polygon.exterior.coords).astype(np.int32)
                cv2.polylines(
                    img, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=3
                )

            if overlay_mbr:
                mbr_pts = np.round(
                    polygon.minimum_rotated_rectangle.exterior.coords
                ).astype(np.int32)
                cv2.polylines(
                    img, [mbr_pts], isClosed=True, color=(0, 255, 0), thickness=3
                )

        return img

    def synthetic_sample(
        self,
        line_ids: list["str"] | Literal["all"],
        *,
        tight_layout: bool = True,
        margin_size_px: int | dict[Literal["right", "left", "top", "bottom"], int] = 0,
        img_poly_transform: ocr_transform | None = None,
        stroke_transform: StrokeTransformCallable | None = None,
        background_transform: ImageTransformCallable | None = None,
        global_image_transform: ImageTransformCallable | None = None,
        overlay_polygons: bool = False,
        overlay_mbr: bool = True,
    ) -> tuple[np.ndarray, str, int]:
        """
        Given a list of ImageBox ids, returns:
        - the synthetic manuscript returned by .synthetic_manuscript,
        - the transcription corresponding to this image,
        - the starting index of this text in the page transcription.
        """

        if not line_ids:
            raise ValueError("Cannot create a synthetic sample with no lines.")
        elif line_ids == "all":
            line_ids = list(self.lines.keys())

        manuscript = self.synthetic_manuscript(
            line_ids,
            tight_layout=tight_layout,
            margin_size_px=margin_size_px,
            img_poly_transform=img_poly_transform,
            stroke_transform=stroke_transform,
            background_transform=background_transform,
            global_image_transform=global_image_transform,
            overlay_polygons=overlay_polygons,
            overlay_mbr=overlay_mbr,
        )[0]

        transcription = self.synthetic_transcription(line_ids)
        starting_index = self.synthetic_starting_index(line_ids)

        return manuscript, transcription, starting_index
