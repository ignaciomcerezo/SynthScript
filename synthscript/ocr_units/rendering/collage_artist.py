from collections.abc import Callable, Sequence
from typing import Literal

import cv2
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon

from synthscript.ocr_units import OCRPage
from synthscript.ocr_units.ocr_line import OCRLine
from synthscript.shared.geometry_processing import get_union_rect
from synthscript.shared.image_processing import crop_or_resize

OCRTransformCallable = Callable[
    [list[tuple[list[np.ndarray], list[Polygon]]]],
    tuple[list[np.ndarray], list[Polygon]],
]
StrokeTransformCallable = Callable[[list[np.ndarray]], list[np.ndarray]]
ImageTransformCallable = Callable[[np.ndarray], np.ndarray]


class Margins:
    def __init__(self, *, left: int, right: int, bottom: int, top: int):
        if min(left, right, bottom, top) < 0:
            raise ValueError(f"Margins must be all positive, but got: {self}")

        self._left = left
        self._right = right
        self._bottom = bottom
        self._top = top

    @property
    def left(self) -> int:
        return self._left

    @left.setter
    def left(self, value: int) -> None:
        if value < 0:
            raise ValueError("Margin sizes must all be positive.")

    @property
    def right(self) -> int:
        return self._right

    @right.setter
    def right(self, value: int) -> None:
        if value < 0:
            raise ValueError("Margin sizes must all be positive.")

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value: int) -> None:
        if value < 0:
            raise ValueError("Margin sizes must all be positive.")

    @property
    def bottom(self) -> int:
        return self._bottom

    @bottom.setter
    def bottom(self, value: int) -> None:
        if value < 0:
            raise ValueError("Margin sizes must all be positive.")


class CollageArtist:
    def __init__(
        self,
        tight_layout: bool = True,
        margin_size_px: (
            int | dict[Literal["left", "right", "top", "bottom"], int] | Margins
        ) = 0,
        img_poly_transform: OCRTransformCallable | None = None,
        stroke_transform: StrokeTransformCallable | None = None,
        background_transform: ImageTransformCallable | None = None,
        global_image_transform: ImageTransformCallable | None = None,
        refit_polygons: bool = True,
        overlay_polygons: bool = False,
        overlay_mbr: bool = False,
    ):
        self.configure(
            use_tight_layout=tight_layout,
            margin_size_px=margin_size_px,
            img_poly_transform=img_poly_transform,
            stroke_transform=stroke_transform,
            background_transform=background_transform,
            global_image_transform=global_image_transform,
            refit_polygons=refit_polygons,
            overlay_polygons=overlay_polygons,
            overlay_mbr=overlay_mbr,
        )

    def configure(
        self,
        *,
        use_tight_layout: bool | None = None,
        margin_size_px: (
            int | dict[Literal["left", "right", "top", "bottom"], int] | Margins
        ) | None = None,
        img_poly_transform: OCRTransformCallable | None = None,
        stroke_transform: StrokeTransformCallable | None = None,
        background_transform: ImageTransformCallable | None = None,
        global_image_transform: ImageTransformCallable | None = None,
        refit_polygons: bool | None = None,
        overlay_polygons: bool | None = None,
        overlay_mbr: bool | None = None,
    ) -> None:

        self.tight_layout = (
            use_tight_layout if use_tight_layout is not None else self.tight_layout
        )
        self._configure_margins(
            margin_size_px if margin_size_px is not None else self.margin_sizes
        )

        self.img_poly_transform = (
            img_poly_transform
            if img_poly_transform is not None
            else self.img_poly_transform
        )
        self.stroke_transform = (
            stroke_transform if stroke_transform is not None else self.stroke_transform
        )
        self.background_transform = (
            background_transform
            if background_transform is not None
            else self.background_transform
        )
        self.global_image_transform = (
            global_image_transform
            if global_image_transform is not None
            else self.global_image_transform
        )
        self.refit_polygons = (
            refit_polygons if refit_polygons is not None else self.refit_polygons
        )
        self.overlay_polygons = (
            overlay_polygons if overlay_polygons is not None else self.overlay_polygons
        )
        self.overlay_mbr = overlay_mbr if overlay_mbr is not None else self.overlay_mbr

    def _configure_margins(
        self,
        margin_size_px: (
            int | dict[Literal["left", "right", "top", "bottom"], int] | Margins
        ),
    ) -> None:
        if isinstance(margin_size_px, Margins):
            self._margin_sizes = margin_size_px
            return

        margin_sizes = {
            "left": (
                margin_size_px["left"]
                if isinstance(margin_size_px, dict)
                else margin_size_px
            ),
            "right": (
                margin_size_px["right"]
                if isinstance(margin_size_px, dict)
                else margin_size_px
            ),
            "top": (
                margin_size_px["top"]
                if isinstance(margin_size_px, dict)
                else margin_size_px
            ),
            "bottom": (
                margin_size_px["bottom"]
                if isinstance(margin_size_px, dict)
                else margin_size_px
            ),
        }
        if not all(val >= 0 for val in margin_sizes.values()):
            raise ValueError("The margin size cannot be negative.")
        self._margin_sizes = Margins(**margin_sizes)

    @property
    def margin_sizes(self) -> Margins:
        return self._margin_sizes

    @margin_sizes.setter
    def margin_sizes(
        self, value: int | dict[Literal["left", "right", "top", "bottom"], int]
    ) -> None:
        self._configure_margins(value)

    def compose(
        self,
        line_ids: Sequence[str],
        page: OCRPage,
    ) -> tuple[np.ndarray, list[Polygon]]:

        if line_ids == "all":
            line_ids = set(page.lines.keys())

        if not isinstance(line_ids, (set, list)):
            raise ValueError(
                "line_ids must be a set[str], list[str], or 'all'; got "
                f"{type(line_ids)}"
            )
        if len(line_ids) != len(set(line_ids)):
            raise ValueError("Duplicate line_ids passed to synthetic_manuscript.")

        lines = {page.lines[line_id] for line_id in line_ids}

        return self._raw_compose(
            lines,
            page.image_dimensions,
            page.background,
        )

    def _raw_compose(
        self,
        lines: set[OCRLine],
        image_dimensions: tuple[int, int],
        background: np.ndarray,
    ) -> tuple[np.ndarray, list[Polygon]]:

        if self.img_poly_transform is not None:
            line_groups = self._group_sorted_by_paragraph(
                sorted(
                    lines,
                    key=lambda line: int(
                        line.sindex  # ty: ignore[invalid-argument-type]
                    ),
                )
            )
            paragraph_equivalent_pairs = [
                (
                    [line.crop for line in line_group],
                    [line.polygon for line in line_group],
                )
                for line_group in line_groups
            ]
            crops, polygons = self.img_poly_transform(paragraph_equivalent_pairs)
        else:
            polygons = [line.polygon for line in lines]
            crops = [line.crop for line in lines]

        if self.stroke_transform is not None:
            crops = self.stroke_transform(crops)

        min_x, min_y, max_x, max_y = get_union_rect(polygons)
        bg_h, bg_w = image_dimensions

        if self.tight_layout:
            x0 = int(min_x) - self._margin_sizes.left
            xf = int(max_x) + 1 + self._margin_sizes.right
            y0 = int(min_y) - self._margin_sizes.top
            yf = int(max_y) + 1 + self._margin_sizes.bottom
            can_crop = True
        else:
            x0 = min(0, int(min_x) - self._margin_sizes.left)
            xf = max(bg_w, int(max_x) + 1 + self._margin_sizes.right)
            y0 = min(0, int(min_y) - self._margin_sizes.top)
            yf = max(bg_h, int(max_y) + 1 + self._margin_sizes.bottom)
            can_crop = False

        bg_np = np.asarray(background)
        canvas = crop_or_resize(
            bg_np, x0=x0, xf=xf, y0=y0, yf=yf, can_crop=can_crop
        ).copy()
        if self.background_transform is not None:
            canvas = self.background_transform(canvas)

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

        if self.global_image_transform is not None:
            canvas = self.global_image_transform(canvas)

        if self.refit_polygons:
            # displace the polygons to the new dimensions of the image
            polygons = [translate(polygon, -x0, -y0) for polygon in polygons]

        if self.overlay_mbr or self.overlay_polygons:
            canvas = self._line_overlay(
                refitted_polygons=(
                    polygons
                    if self.refit_polygons
                    else [translate(polygon, -x0, -y0) for polygon in polygons]
                ),
                manuscript=canvas,
                overlay_polygons=self.overlay_polygons,
                overlay_mbr=self.overlay_mbr,
            )

        return canvas, polygons

    @staticmethod
    def _line_overlay(
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

    @staticmethod
    def _group_sorted_by_paragraph(lines: list[OCRLine]) -> list[list[OCRLine]]:
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
