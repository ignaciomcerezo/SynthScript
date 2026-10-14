import json
import os
import shutil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pytest
from lxml import etree  # ty: ignore[unresolved-import]

from synthscript.datasets import OCRDataset
from synthscript.rendering import CollageArtist
from synthscript.rendering.latex import (
    LatexRenderer,
    LatexRenderError,
    UnsupportedLatexError,
)
from synthscript.rendering.latex.clean_and_tokenize import (
    _remove_comments,
    clean_and_tokenize,
)
from synthscript.transforms.image.ink_fade import InkFade
from synthscript.transforms.line.linewise_horizontal_stretch import (
    LinewiseHorizontalStretch,
)
from synthscript.transforms.page.paragraph_spacing_jitter import ParagraphSpacingJitter
from synthscript.transforms.paragraph.horizontal_movement import HorizontalMovement
from synthscript.transforms.paragraph.paragraph_linewise_rotation import (
    ParagraphLinewiseRotation,
)


@pytest.fixture(scope="module")
def renderer():
    engine = os.environ.get("SYNTHSCRIPT_TEST_LATEX", "latex")
    converter = os.environ.get("SYNTHSCRIPT_TEST_DVISVGM", "dvisvgm")
    if not shutil.which(engine) or not shutil.which(converter):
        pytest.skip("Real rendering tests require latex and dvisvgm.")
    pytest.importorskip("resvg_py", exc_type=ImportError)
    return LatexRenderer(engine=engine, dvisvgm=converter, dpi=144)


def article(body, width="180pt"):
    return (
        r"\documentclass[a4paper]{article}"
        + "\n"
        + r"\setlength{\textwidth}{"
        + width
        + "}\n"
        + r"\begin{document}"
        + "\n"
        + body
        + "\n"
        + r"\end{document}"
    )


def compose(page):
    return CollageArtist(tight_layout=False).compose(list(page.lines), page)[0]


def unmarked_image(renderer, source, directory):
    """Independent clean_and_tokenized render: compare pixels and line breaking."""
    import resvg_py

    plain = r"\def\pgfsysdriver{pgfsys-dvisvgm.def}" + source.replace(
        r"\begin{document}",
        r"\usepackage{amsmath,amssymb,tikz}\begin{document}\pagestyle{empty}",
    )
    (directory / "plain.tex").write_text(plain, encoding="utf-8")
    xdv = Path(renderer.engine).stem.lower() == "xelatex"
    renderer._run(
        [
            renderer.engine,
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            *(["-no-pdf"] if xdv else []),
            "plain.tex",
        ],
        directory,
    )
    renderer._run(
        [
            renderer.dvisvgm,
            "--bbox=papersize",
            "--exact-bbox",
            "--no-fonts",
            "--output=plain.svg",
            "plain.xdv" if xdv else "plain.dvi",
        ],
        directory,
    )
    svg = etree.parse(str(directory / "plain.svg")).getroot()
    vx, vy, vw, vh = map(float, svg.get("viewBox").split())
    scale = renderer.dpi / 72
    width, height = int(np.ceil(vw * scale)), int(np.ceil(vh * scale))
    svg.set("viewBox", f"{vx} {vy} {width / scale} {height / scale}")
    svg.set("width", str(width))
    svg.set("height", str(height))
    png = resvg_py.svg_to_bytes(
        svg_string=etree.tostring(svg, encoding="unicode"),
        width=width,
        height=height,
        background="white",
    )
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)


def test_wrapping_and_paragraphs(renderer, tmp_path):
    first = " ".join(["The quick brown fox jumps over the lazy dog."] * 4)
    second = "A second short paragraph."
    source = article(first + "\n\n" + second)
    page = renderer.render(source)
    assert len(page.paragraphs) == 2
    assert len(page.paragraphs[0].lines) > 1
    assert " ".join(line.text for line in page.paragraphs[0].lines) == first
    assert page.paragraphs[1].lines[0].text == second
    assert np.all(page.background == 255)
    # Equality proves both full ink coverage and unchanged TeX line breaking.
    np.testing.assert_array_equal(
        compose(page), unmarked_image(renderer, source, tmp_path)
    )
    metadata = json.loads(page.metadata["synthscript.latex.objects"])
    assert len({obj["id"] for obj in metadata}) == len(metadata)
    for obj in metadata:
        assert source[obj["source_start"] : obj["source_end"]] == obj["text"]
        assert obj["line_id"] in page.lines
    for line in page.lines.values():
        assert line.polygon.equals(line.polygon.envelope)
        assert line.crop[..., 0].max() > 100
        assert np.all(line.crop[..., 3] == 255)


@pytest.mark.parametrize(
    "formula",
    [
        r"\[\frac{x^2}{2}+\sqrt{y}\]",
        r"\begin{equation}E=mc^2\end{equation}",
        r"\begin{align}a&=b+c\\d&=e-f\end{align}",
        r"\begin{tikzpicture}\draw (0,0) rectangle (2,1);"
        r"\draw[->] (0,0)--(2,1);\node at (1,.5) {$x$};\end{tikzpicture}",
    ],
)
def test_math_and_atomic_objects(renderer, tmp_path, formula):
    inline = r"Inline $x^2+\alpha$ and \(y+1\) work."
    source = article(inline + "\n\n" + formula + "\n\nAfter the object.", "300pt")
    page = renderer.render(source)
    assert next(iter(page.lines.values())).text == inline
    assert list(page.lines.values())[1].text == formula
    assert list(page.lines.values())[2].text == "After the object."
    assert len(page.paragraphs) == 3
    np.testing.assert_array_equal(
        compose(page), unmarked_image(renderer, source, tmp_path)
    )


def test_dataset_with_all_transform_levels(renderer):
    page = renderer.render(
        article(
            "One short line.\n\n"
            + r"\begin{tikzpicture}\draw (0,0)--(1,1);\end{tikzpicture}"
        )
    )
    dataset = OCRDataset([page], rng=42)
    orders: list[int | Literal["paragraph", "page"]] = [1, "paragraph", "page"]
    dataset.orders = orders
    original = [line.polygon.wkt for line in page.lines.values()]
    dataset.set_transform(
        LinewiseHorizontalStretch(1.05),
        HorizontalMovement("linear", intercept=2, slope=0),
        ParagraphSpacingJitter(0.1),
        InkFade(0.1),
    )
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["image"].size > 0
        assert sample["image"].min() < 240
        assert sample["text"]
    assert [line.polygon.wkt for line in page.lines.values()] == original


def test_multiple_pages_rejected(renderer):
    source = article("\n\n".join(["One short paragraph."] * 100))
    with pytest.raises(UnsupportedLatexError, match="one rendered page"):
        renderer.render(source)


def test_paragraphs_and_varying_line_widths_with_rotation(renderer):
    paragraphs = [
        [
            "A short introductory line.",
            "A longer line with ordinary words and an inline $x^2$ formula.",
            "Short.",
            "Another long line with enough words to cross preceding short lines.",
        ],
        ["A second paragraph.", "Its next line."],
    ]
    source = article("\n\n".join(r"\\{}".join(lines) for lines in paragraphs), "345pt")
    page = renderer.render(source)
    assert [[line.text for line in par.lines] for par in page.paragraphs] == paragraphs
    original = [line.polygon.wkt for line in page.lines.values()]
    dataset = OCRDataset([page], rng=42)
    orders: list[int | Literal["paragraph", "page"]] = ["page"]
    dataset.orders = orders
    dataset.set_transform(ParagraphLinewiseRotation(20))
    image, polygons = dataset.renderer.compose(list(page.lines), page)
    assert image.min() < 240
    assert all(
        a.intersection(b).area < 1e-8
        for i, a in enumerate(polygons)
        for b in polygons[i + 1 :]
    )
    assert dataset[0]["text"] == "\n".join(
        line for paragraph in paragraphs for line in paragraph
    )
    assert [line.polygon.wkt for line in page.lines.values()] == original


@pytest.mark.parametrize(
    "source",
    [
        "",
        r"\begin{itemize}\item A\end{itemize}",
        r"\input{other}",
        r"\textbf{two words}",
        r"\unknown{x}",
        r"$unfinished",
        r"\documentclass[twocolumn]{article}\begin{document}Hi\end{document}",
    ],
)
def test_unsupported_sources(source):
    with pytest.raises(UnsupportedLatexError):
        clean_and_tokenize(source)


def test_comments_and_escaped_percent():
    _, objects = clean_and_tokenize("hel% comment\nlo 50\\% world")
    assert [obj.text for obj in objects] == ["hello", r"50\%", "world"]
    _, objects = clean_and_tokenize("50\\% ok % comment\nnext")
    assert [obj.text for obj in objects] == [r"50\%", "ok", "next"]


def test_joined_lines_with_brackets_and_math(renderer, tmp_path):
    lines = [
        r"On peut definir aussi $\mathcal{E}_{[0]}$",
        r"[$\mathcal{E}_{[0]}$ est le topos discret defini par",
        r"$\pi_{0}(\mathcal{E})$, pour tout $i\in\mathbb{N}$.]",
        "An -- ordinary dash and nonbreaking~space.",
    ]
    source = article(r"\\{}".join(lines), "345pt")
    page = renderer.render(source)
    assert [line.text for line in page.lines.values()] == lines
    np.testing.assert_array_equal(
        compose(page), unmarked_image(renderer, source, tmp_path)
    )
    dataset = OCRDataset([page], rng=42)
    orders: list[int | Literal["paragraph", "page"]] = ["page"]
    dataset.orders = orders
    assert dataset[0]["text"] == "\n".join(lines)


def test_explicit_line_spacing_preserved(renderer, tmp_path):
    source = article(r"First line.\\[2pt]Second line.\\{}$x$ and {}text.")
    page = renderer.render(source)
    assert [line.text for line in page.lines.values()] == [
        "First line.",
        "Second line.",
        "$x$ and text.",
    ]
    np.testing.assert_array_equal(
        compose(page), unmarked_image(renderer, source, tmp_path)
    )


@pytest.mark.parametrize(
    "body",
    [
        r"Before $x$; \[x^2\] after the display.",
        r"\[x^2\]",
        r"\begin{align*}x&=y\\a&=b\end{align*}",
        r"\[x^2\]\[y^2\]",
        r"Before.\par After.",
        r"\textbf{Bold} text with 50\% ink.",
    ],
)
def test_additional_layouts(renderer, tmp_path, body):
    source = article(body)
    page = renderer.render(source)
    np.testing.assert_array_equal(
        compose(page), unmarked_image(renderer, source, tmp_path)
    )


@pytest.mark.parametrize("engine_name", ["latex", "xelatex"])
def test_french_proposition_hyphenation(renderer, tmp_path, engine_name):
    engine = str(
        Path(renderer.engine).with_name(engine_name + Path(renderer.engine).suffix)
    )
    if not shutil.which(engine):
        pytest.skip(f"{engine_name} is not installed.")
    renderer = LatexRenderer(engine=engine, dvisvgm=renderer.dvisvgm, dpi=144)
    body = (
        r"Proposition 1.3. a) Si $\mathcal{E}$ est multigaloisien, "
        r"tout topos induit $\mathcal{C}/S$ aussi.\\"
        r"b) Toute somme de topos multigaloisiens (i.e. tout produit "
        r"de catégories multigaloisiennes)\\"
        r"est un topos multigaloisien.\\N.B. on verra plus\\"
        r"loin que l'on peut choisir\\canoniquement"
    )
    source = article(body, "345pt")
    page = renderer.render(source)
    texts = [line.text for line in page.lines.values()]
    assert len(texts) == 7
    assert texts[1].endswith("multi-")
    assert texts[2] == "galoisiennes)"
    assert texts[-1] == "canoniquement"
    fragments = [
        obj
        for obj in json.loads(page.metadata["synthscript.latex.objects"])
        if obj.get("source_text") == "multigaloisiennes)"
    ]
    assert len(fragments) == 2
    assert len({obj["source_object_id"] for obj in fragments}) == 1
    assert [obj["generated_hyphen"] for obj in fragments] == [True, False]
    assert (
        "".join(source[obj["source_start"] : obj["source_end"]] for obj in fragments)
        == "multigaloisiennes)"
    )
    np.testing.assert_array_equal(
        compose(page), unmarked_image(renderer, source, tmp_path)
    )
    dataset = OCRDataset([page], rng=42)
    orders: list[int | Literal["paragraph", "page"]] = ["page"]
    dataset.orders = orders
    assert dataset[0]["text"] == "\n".join(texts)


@pytest.mark.parametrize("word", [r"hyphen\-ation", r"office\-work", "office-work"])
def test_discretionary_hyphens_and_ligatures(renderer, tmp_path, word):
    source = article(word, "35pt")
    page = renderer.render(source)
    assert len(page.lines) >= 2
    fragments = json.loads(page.metadata["synthscript.latex.objects"])
    assert all(obj["source_text"] == word for obj in fragments)
    np.testing.assert_array_equal(
        compose(page), unmarked_image(renderer, source, tmp_path)
    )


def test_internal_math_wrap_rejected(renderer):
    with pytest.raises(UnsupportedLatexError, match="wraps internally"):
        renderer.render(article(r"$a+b+c+d+e+f+g+h$", "30pt"))


def test_wrapped_word_source_offsets_with_comment(renderer, tmp_path):
    source = article("off% comment\nice\\-work", "35pt")
    page = renderer.render(source)
    fragments = json.loads(page.metadata["synthscript.latex.objects"])
    assert len(fragments) >= 2
    assert all(obj["source_text"] == r"office\-work" for obj in fragments)
    original_fragments = "".join(
        _remove_comments(source[obj["source_start"] : obj["source_end"]])[0]
        for obj in fragments
    )
    assert original_fragments == "officework"
    np.testing.assert_array_equal(
        compose(page), unmarked_image(renderer, source, tmp_path)
    )


def test_dpi_scaling(renderer):
    low = LatexRenderer(engine=renderer.engine, dvisvgm=renderer.dvisvgm, dpi=72)
    high_page = renderer.render("Hello world.")
    low_page = low.render("Hello world.")
    assert high_page.image_dimensions == tuple(
        2 * size for size in low_page.image_dimensions
    ) or all(
        abs(high_size - 2 * low_size) <= 1
        for high_size, low_size in zip(
            high_page.image_dimensions, low_page.image_dimensions, strict=True
        )
    )
    assert high_page.page_id != low_page.page_id
    high_bounds = next(iter(high_page.lines.values())).polygon.bounds
    low_bounds = next(iter(low_page.lines.values())).polygon.bounds
    np.testing.assert_allclose(high_bounds, np.array(low_bounds) * 2, atol=3)


def test_xdv_engine(renderer, tmp_path):
    engine = str(
        Path(renderer.engine).with_name("xelatex" + Path(renderer.engine).suffix)
    )
    if not shutil.which(engine):
        pytest.skip("XeLaTeX is not installed.")
    xdv_renderer = LatexRenderer(engine=engine, dvisvgm=renderer.dvisvgm, dpi=144)
    source = article(
        r"An inline $x^2$ formula."
        + "\n\n"
        + r"\begin{align}x&=y\\a&=b\end{align}"
        + "\n\n"
        + r"\begin{tikzpicture}\draw (0,0)--(1,1);\end{tikzpicture}"
    )
    page = xdv_renderer.render(source)
    assert len(page.lines) == 3
    np.testing.assert_array_equal(
        compose(page), unmarked_image(xdv_renderer, source, tmp_path)
    )


def test_missing_tool_error():
    renderer = LatexRenderer(engine="latex", dvisvgm="missing-synthscript-dvisvgm")
    with pytest.raises(LatexRenderError, match="Missing executable"):
        renderer.render("Hello.")


@pytest.mark.parametrize("dpi", [0, -1, float("nan"), float("inf")])
def test_invalid_dpi(dpi):
    with pytest.raises(ValueError, match="dpi"):
        LatexRenderer(dpi=dpi)
