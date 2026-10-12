"""Compile marked LaTeX, read SVG geometry, and construct an OCRPage."""

import json
import math
import re
import shutil
import subprocess
import unicodedata
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from shapely.geometry import box

from synthscript.ocr_units import OCRPage

from .clean_and_tokenize import (
    SourceObject,
    UnsupportedLatexError,
    _remove_comments,
    clean_and_tokenize,
)


class LatexRenderError(RuntimeError):
    """Compilation, geometry extraction or rasterization failed."""


class LatexRenderer:
    """Render single-page article sources or body fragments at the requested DPI.

    Requires "latex" (or "xelatex"), "dvisvgm" and the "latex" Python extra.
    TeX runs with shell escape disabled in a temporary working directory. Sources
    must be trusted.
    """

    def __init__(
        self,
        *,
        dpi: float = 150,
        engine: str = "latex",
        dvisvgm: str = "dvisvgm",
        timeout: float = 60,
    ) -> None:
        if not math.isfinite(dpi) or dpi <= 0:
            raise ValueError("dpi must be positive and finite.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite.")
        if Path(engine).stem.lower() not in {"latex", "xelatex"}:
            raise ValueError(
                "engine must be latex or xelatex (optionally a full path)."
            )
        self.dpi = dpi
        self.engine = engine
        self.dvisvgm = dvisvgm
        self.timeout = timeout

    def _run(self, command: list[str], directory: Path) -> str:
        """Run a TeX tool in the temporary directory and report failures."""
        try:
            result = subprocess.run(
                command,
                cwd=directory,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LatexRenderError(f"Failed to run {command[0]}: {exc}") from exc
        output = result.stdout + result.stderr
        if result.returncode:
            raise LatexRenderError(f"{Path(command[0]).name} failed:\n{output[-6000:]}")
        return output

    def render(self, latex_source: str) -> OCRPage:
        """
        Compile annotated LaTeX, rasterize its SVG, and recover an OCR page.
        """
        document, objects = clean_and_tokenize(latex_source)
        engine = shutil.which(self.engine)
        converter = shutil.which(self.dvisvgm)
        if engine is None or converter is None:
            missing = self.engine if engine is None else self.dvisvgm
            raise LatexRenderError(
                f"Missing executable: {missing}. "
                "Install a TeX distribution with dvisvgm."
            )
        engine = str(Path(engine).absolute())
        converter = str(Path(converter).absolute())
        try:
            import resvg_py
        except ImportError as exc:
            raise LatexRenderError(
                "Install the renderer extra: pip install 'synthscript[latex]'."
            ) from exc
        with TemporaryDirectory(prefix="synthscript-latex-") as temporary:
            directory = Path(temporary)
            (directory / "page.tex").write_text(document, encoding="utf-8")
            # TeX produces DVI or XDV so dvisvgm can read the markers
            flags = ["-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error"]
            xdv = Path(engine).stem.lower() == "xelatex"
            if xdv:
                flags.append("-no-pdf")
            self._run([engine, *flags, "page.tex"], directory)
            self._run(
                [
                    converter,
                    "--page=1-",
                    "--bbox=papersize",
                    "--exact-bbox",
                    "--no-fonts",
                    "--output=page-%p.svg",
                    "page.xdv" if xdv else "page.dvi",
                ],
                directory,
            )
            svgs = sorted(directory.glob("page-*.svg"))
            if len(svgs) != 1:
                raise UnsupportedLatexError(
                    f"Expected one rendered page, got {len(svgs)}."
                )
            svg = svgs[0].read_text(encoding="utf-8")
            root = ET.fromstring(svg)
            text_root: ET.Element | None = None
            # word whose start and end markers have different baselines wrapped
            starts = {
                el.get("data-ss-start"): el
                for el in root.iter()
                if el.get("data-ss-start")
            }
            ends = {
                el.get("data-ss-object"): el
                for el in root.iter()
                if el.get("data-ss-object")
            }
            if any(
                obj.kind == "text"
                and obj.id in starts
                and obj.id in ends
                and abs(
                    float(starts[obj.id].get("data-y", "nan"))
                    - float(ends[obj.id].get("data-y", "nan"))
                )
                > 0.1
                for obj in objects
            ):
                # keep paths for rasterization: a companion SVG with text gives
                # unicode elements independently od font encodings.
                self._run(
                    [
                        converter,
                        "--page=1",
                        "--bbox=papersize",
                        "--exact-bbox",
                        "--output=text.svg",
                        "page.xdv" if xdv else "page.dvi",
                    ],
                    directory,
                )
                text_root = ET.parse(directory / "text.svg").getroot()
            try:
                vx, vy, vw, vh = map(float, root.attrib["viewBox"].split())
                if (
                    not all(math.isfinite(v) for v in (vx, vy, vw, vh))
                    or min(vw, vh) <= 0
                ):
                    raise ValueError("Invalid SVG viewport.")
                # 72 dvisvgm SVG units per inch (as per the PostScript standard)
                width = math.ceil(vw * self.dpi / 72)
                height = math.ceil(vh * self.dpi / 72)

                # equal scale on both axis
                scale = self.dpi / 72

                # preserve equal scale after rounding the raster dimensions to whole
                # pixels
                root.set("viewBox", f"{vx} {vy} {width / scale} {height / scale}")
                root.set("width", str(width))
                root.set("height", str(height))
                raster_svg = ET.tostring(root, encoding="unicode")
                png = resvg_py.svg_to_bytes(
                    svg_string=raster_svg,
                    width=width,
                    height=height,
                    background="white",
                )
                image = cv2.imdecode(
                    np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if image is None or image.shape != (height, width):
                    raise ValueError("Rasterizer returned an invalid image.")
            except Exception as exc:
                raise LatexRenderError(f"SVG rasterization failed: {exc}") from exc
            return self._to_ocrpage(
                root,
                objects,
                image,
                vx,
                vy,
                scale,
                latex_source,
                text_root,
            )

    def _split_wrapped_words(
        self,
        root: ET.Element,
        text_root: ET.Element,
        objects: list[SourceObject],
        width: int,
        height: int,
        vx: float,
        vy: float,
        scale: float,
        source: str,
    ) -> tuple[list[SourceObject], dict[str, tuple[SourceObject, bool]]]:
        """
        Wrapped words are split into rendered fragments with original source offsets.
        """
        import resvg_py

        starts = {
            el.get("data-ss-start"): el for el in root.iter() if el.get("data-ss-start")
        }
        ends = {
            el.get("data-ss-object"): el
            for el in root.iter()
            if el.get("data-ss-object")
        }
        text_groups = {
            el.get("data-ss-start"): el
            for el in text_root.iter()
            if el.get("data-ss-start")
        }
        expanded: list[SourceObject] = []
        sources: dict[str, tuple[SourceObject, bool]] = {}
        for obj in objects:
            if obj.id not in starts or obj.id not in ends:
                raise LatexRenderError(f"Missing position markers for {obj.id}.")
            group, end = starts[obj.id], ends[obj.id]
            if (
                obj.kind != "text"
                or abs(
                    float(group.get("data-y", "nan")) - float(end.get("data-y", "nan"))
                )
                <= 0.1
            ):
                expanded.append(obj)
                continue
            error = (
                f"Cannot map wrapped object {obj.id} ({obj.text!r}) to its source. "
                "Only literal text words and discretionary \\- are supported."
            )
            # retain a character-to-source map, including explicit discretionary
            # hyphens that TeX may or may not print at the selected break.
            positions: list[int] = []
            for match in re.finditer(r"\\-|.", obj.text):
                if match[0] != r"\-":
                    if match[0] in "\\{}~":
                        raise UnsupportedLatexError(error)
                    positions.append(match.start())
            literal = "".join(obj.text[p] for p in positions)

            # the clean object text and original source have different indices
            # when a comment was made inside the span
            _, original_positions = _remove_comments(source[obj.start : obj.end])
            runs: list[tuple[float, str]] = []
            baseline = float(group.get("data-y", "nan"))

            # text SVG elements expose the rendered characters and baselines,
            # the path SVG used for rasterization does not contain that text
            for element in text_groups[obj.id].iter():
                if element.tag.rsplit("}", 1)[-1] not in {"text", "tspan"}:
                    continue
                baseline = float(element.get("y", str(baseline)))
                if element.text:
                    text = unicodedata.normalize("NFKC", element.text)
                    if runs and abs(runs[-1][0] - baseline) <= 0.1:
                        runs[-1] = (baseline, runs[-1][1] + text)
                    else:
                        runs.append((baseline, text))
            if len(runs) < 2:
                raise UnsupportedLatexError(error)
            cursor_head = 0
            fragments: list[SourceObject] = []
            wrappers: list[ET.Element] = []
            for index, (baseline, text) in enumerate(runs):
                generated_hyphen = False
                source_text = text
                if not literal.startswith(text, cursor_head):
                    # a hyphen printed at a line break has no source character
                    # (as it was added by TeX itself)
                    if index == len(runs) - 1 or not text.endswith("-"):
                        raise UnsupportedLatexError(error)
                    source_text = text[:-1]
                    generated_hyphen = True
                if not source_text or not literal.startswith(source_text, cursor_head):
                    raise UnsupportedLatexError(error)
                fragment = replace(
                    obj,
                    id=f"{obj.id}f{index}",
                    text=text,
                    start=obj.start + original_positions[positions[cursor_head]],
                    end=obj.start
                    + original_positions[positions[cursor_head + len(source_text) - 1]]
                    + 1,
                    space_before=obj.space_before if index == 0 else False,
                )
                cursor_head += len(source_text)
                fragments.append(fragment)
                sources[fragment.id] = (obj, generated_hyphen)
                wrappers.append(
                    ET.Element(
                        group.tag,
                        {
                            "data-ss-start": fragment.id,
                            "data-y": str(baseline),
                        },
                    )
                )
            if cursor_head != len(literal):
                raise UnsupportedLatexError(error)
            # move each path glyph under the matching baseline fragment
            for glyph in list(group):
                if glyph.tag.rsplit("}", 1)[-1] != "use" or "y" not in glyph.attrib:
                    raise UnsupportedLatexError(error)
                y = float(glyph.get("y", "nan"))
                index = min(range(len(runs)), key=lambda i: abs(runs[i][0] - y))
                if abs(runs[index][0] - y) > 0.1:
                    raise UnsupportedLatexError(error)
                group.remove(glyph)
                wrappers[index].append(glyph)
            del group.attrib["data-ss-start"]
            group.set("data-ss-source", obj.id)
            group.extend(wrappers)
            # isolate each fragment to measure its own rectangle. The original
            # marker describes the entire word and cannot locate each piece.
            for fragment, wrapper in zip(fragments, wrappers, strict=True):
                isolated = deepcopy(root)
                for element in isolated.iter():
                    if (
                        element.get("data-ss-start")
                        and element.get("data-ss-start") != fragment.id
                    ):
                        element.set("display", "none")
                png = resvg_py.svg_to_bytes(
                    svg_string=ET.tostring(isolated, encoding="unicode"),
                    width=width,
                    height=height,
                    background="white",
                )
                image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise LatexRenderError(f"Rasterization failed for {fragment.id}.")
                ys, xs = np.nonzero(image < 255)
                if not len(xs):
                    raise UnsupportedLatexError(error)
                bbox = (
                    vx + int(xs.min()) / scale,
                    vy + int(ys.min()) / scale,
                    (int(xs.max()) - int(xs.min()) + 1) / scale,
                    (int(ys.max()) - int(ys.min()) + 1) / scale,
                )
                group.append(
                    ET.Element(
                        group.tag,
                        {
                            "data-ss-object": fragment.id,
                            "data-y": wrapper.get("data-y", "nan"),
                            "data-bbox": " ".join(map(str, bbox)),
                        },
                    )
                )
            expanded.extend(fragments)
        return expanded, sources

    def _to_ocrpage(
        self,
        root: ET.Element,
        objects: list[SourceObject],
        image: np.ndarray,
        vx: float,
        vy: float,
        scale: float,
        source: str,
        text_root: ET.Element | None = None,
    ) -> OCRPage:
        """Map SVG object geometry into OCR lines, crops, and source metadata"""
        height, width = image.shape
        sources: dict[str, tuple[SourceObject, bool]] = {}
        if text_root is not None:
            objects, sources = self._split_wrapped_words(
                root,
                text_root,
                objects,
                width,
                height,
                vx,
                vy,
                scale,
                source,
            )
        starts = {
            el.get("data-ss-start"): float(el.get("data-y", "nan"))
            for el in root.iter()
            if el.get("data-ss-start")
        }
        ends = {
            el.get("data-ss-object"): el
            for el in root.iter()
            if el.get("data-ss-object")
        }
        groups: list[list[tuple[SourceObject, tuple[int, int, int, int]]]] = []
        baseline: float | None = None
        object_metadata: list[dict[str, object]] = []
        for obj in objects:
            if obj.id not in starts or obj.id not in ends:
                raise LatexRenderError(f"Missing position markers for {obj.id}.")
            marker = ends[obj.id]
            try:
                x, y, w, h = map(float, marker.get("data-bbox", "").split())
                end_y = float(marker.get("data-y", "nan"))
            except (TypeError, ValueError) as exc:
                raise LatexRenderError(f"Invalid geometry for {obj.id}.") from exc
            if not all(math.isfinite(v) for v in (x, y, w, h, end_y)) or min(w, h) <= 0:
                raise UnsupportedLatexError(
                    f"Object {obj.id} has no visible ink: {obj.text!r}."
                )
            if (
                x < vx - 0.01
                or y < vy - 0.01
                or (x + w - vx) * scale > width + 0.01
                or (y + h - vy) * scale > height + 0.01
            ):
                raise UnsupportedLatexError(
                    f"Object {obj.id} extends outside the page."
                )
            atomic = obj.kind in {"display", "diagram"}
            if not atomic and abs(starts[obj.id] - end_y) > 0.1:
                raise UnsupportedLatexError(
                    f"Object {obj.id} wraps internally "
                    "(hyphenation or breakable math). "
                    f"Object text: {obj.text!r}."
                )
            # adding one pixel padding retains antialiasing at the boundary
            bounds = (
                math.floor((x - vx) * scale) - 1,
                math.floor((y - vy) * scale) - 1,
                math.ceil((x + w - vx) * scale) + 1,
                math.ceil((y + h - vy) * scale) + 1,
            )
            if (
                bounds[2] <= 0
                or bounds[3] <= 0
                or bounds[0] >= width
                or bounds[1] >= height
            ):
                raise UnsupportedLatexError(f"Object {obj.id} lies outside the page.")
            bounds = (
                max(0, bounds[0]),
                max(0, bounds[1]),
                min(width, bounds[2]),
                min(height, bounds[3]),
            )
            previous = groups[-1][-1][0] if groups else None
            # display-math and diagrams are whole lines, while other adjacent objects
            # share a line only when their paragraph and baseline agree.
            if (
                previous is None
                or atomic
                or previous.kind in {"display", "diagram"}
                or obj.paragraph != previous.paragraph
                or baseline is None
                or abs(starts[obj.id] - baseline) > 0.1
            ):
                groups.append([])
                baseline = starts[obj.id]
            groups[-1].append((obj, bounds))
            object_metadata.append(
                {
                    "id": obj.id,
                    "paragraph_id": f"paragraph_{obj.paragraph}",
                    "kind": obj.kind,
                    "source_start": obj.start,
                    "source_end": obj.end,
                    "text": obj.text,
                    "bbox": bounds,
                    "line_id": f"line_{len(groups) - 1}",
                }
            )
            if obj.id in sources:
                original, generated_hyphen = sources[obj.id]
                object_metadata[-1].update(
                    {
                        "source_object_id": original.id,
                        "source_text": original.text,
                        "generated_hyphen": generated_hyphen,
                    }
                )
        transcriptions: list[str] = []
        polygons: list[list[tuple[float, float]]] = []
        line_ids: list[str] = []
        paragraph_line_ids: dict[int, list[str]] = {}
        for index, group in enumerate(groups):
            identity = f"line_{index}"
            line_ids.append(identity)
            transcriptions.append(
                "".join(
                    (" " if j and obj.space_before else "") + obj.text
                    for j, (obj, _) in enumerate(group)
                )
            )
            bounds = (
                min(b[0] for _, b in group),
                min(b[1] for _, b in group),
                max(b[2] for _, b in group),
                max(b[3] for _, b in group),
            )
            polygons.append(list(box(*bounds).exterior.coords))
            paragraph_line_ids.setdefault(group[0][0].paragraph, []).append(identity)

        # some line boxes can overlap: we render each line's SVG objects so it owns
        # its rendered content
        import resvg_py

        crops: list[np.ndarray] = []
        metadata_by_id = {obj["id"]: obj for obj in object_metadata}
        for index, (identity, group) in enumerate(zip(line_ids, groups, strict=True)):
            x0, y0, x1, y1 = map(
                int,
                box(
                    min(x for x, _ in polygons[index]),
                    min(y for _, y in polygons[index]),
                    max(x for x, _ in polygons[index]),
                    max(y for _, y in polygons[index]),
                ).bounds,
            )
            isolated = deepcopy(root)
            selected = {obj.id for obj, _ in group}
            for element in isolated.iter():
                if (
                    element.get("data-ss-start")
                    and element.get("data-ss-start") not in selected
                ):
                    element.set("display", "none")
            png = resvg_py.svg_to_bytes(
                svg_string=ET.tostring(isolated, encoding="unicode"),
                width=width,
                height=height,
                background="white",
            )
            isolated_image = cv2.imdecode(
                np.frombuffer(png, np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            if isolated_image is None:
                raise LatexRenderError(f"Rasterization failed for {identity}.")
            if group[0][0].kind == "diagram":
                # PGF's virtual TeX bbox omits some stroke extents, so we refine
                # this atomic object's rectangle with its isolated raster.
                ys, xs = np.nonzero(isolated_image < 255)
                if not len(xs):
                    raise UnsupportedLatexError(f"Diagram {group[0][0].id} has no ink.")
                x0 = max(0, min(x0, int(xs.min()) - 1))
                y0 = max(0, min(y0, int(ys.min()) - 1))
                x1 = min(width, max(x1, int(xs.max()) + 2))
                y1 = min(height, max(y1, int(ys.max()) + 2))
                polygons[index] = list(box(x0, y0, x1, y1).exterior.coords)
                metadata_by_id[group[0][0].id]["bbox"] = (x0, y0, x1, y1)
            crop = isolated_image[y0:y1, x0:x1]
            crops.append(cv2.cvtColor(255 - crop, cv2.COLOR_GRAY2BGRA))
        page = OCRPage.from_explicit_layout(
            transcriptions=transcriptions,
            polygon_coords=polygons,
            line_ids=line_ids,
            rotations=[0.0] * len(line_ids),
            page_id="latex-"
            + sha256(
                f"{self.dpi}:{Path(self.engine).stem}:{source}".encode()
            ).hexdigest()[:16],
            stroke=255 - image,
            background=np.full_like(image, 255),
            paragraph_line_ids=list(paragraph_line_ids.values()),
            paragraph_ids=[f"paragraph_{p}" for p in paragraph_line_ids],
            metadata={
                "synthscript.latex.source": source,
                "synthscript.latex.dpi": str(self.dpi),
                "synthscript.latex.objects": json.dumps(
                    object_metadata, ensure_ascii=False
                ),
            },
        )
        for identity, crop in zip(line_ids, crops, strict=True):
            page.lines[identity].crop = crop
        return page
