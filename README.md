# SynthScript

SynthScript builds OCR training samples from annotated document pages. It loads
PAGE-XML annotations as `OCRPage` objects, composes samples from 
line images and polygons from a PyTorch-compatible datasets with configurable 
sampling and augmentation configuration. 
It also supports rendering pages from LaTeX code and metrics for comparing LaTeX
transcriptions.

## Main components

- **Loading:** `synthscript.loading` reads and writes PAGE-XML. Interfaces for
  Label Studio and an image bucket can populate the local dataset with the required
  images and transcriptions.
- **Page model:** `synthscript.ocr_units` represents pages (groups of paragraphs),
  paragraphs (groups of lines), lines (transcription, geometry and image).
- **Datasets:** `OCRDataset` returns image and transcription samples,
  `SegmentationDataset` returns images and line polygons. Both support line
  clusters, paragraphs, and full pages through their `orders` setting.
- **Rendering and augmentation:** `CollageArtist` composes sampled lines with
  stroke, background, geometry, and image transforms. `LatexRenderer` creates an
  `OCRPage` from a supported single-page LaTeX source.
- **Metrics and training adapters:** `synthscript.metric` provides LaTeX
  canonicalization and weighted TED from an AST. `synthscript.training`
  contains helpers for vision model and YOLO segmentation/OBB training.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m pip install .
```

Install the optional dependencies:

```bash
python -m pip install ".[latex]"  # resvg-py for LaTeX rendering
python -m pip install ".[train]"  # model-training dependencies
python -m pip install ".[dev]"    # pytest, Ruff, and development tools
```

LaTeX rendering also requires `latex` or `xelatex` and `dvisvgm` on `PATH`.
The YOLO trainer modules import `ultralytics`, which is not currently listed
in the `train` extra. Dependency versions and platform conditions are defined
in [pyproject.toml](pyproject.toml).

## Dataset workflow

The local data layout managed by `PathBundle` is:

```text
data_in/
|---images/
|   |-- raw/
|   |-- stroke/
|   \-- background/
|
|---page/
|   \-- s0_pg123.xml
|
\---exports/
```

SynthScript uses PAGE-XML 2024-07-15 for stored annotations. A PAGE file holds
line and paragraph geometry, reading order, orientation, original transcription
and other information for one annotation. It references the raw image and the
SynthScript stroke and background images by relative paths. Coordinates are
stored as integer pixels. 
`load_pages()` can combine annotations of the same physical page and applies
transcription homogenization while loading; it does not rewrite the PAGE ground 
truth.

Once the images and PAGE files are in `data_in/`, create an OCR dataset:

```python
from synthscript.datasets import OCRDataset
from synthscript.loading import load_pages

pages = load_pages()
dataset = OCRDataset(pages, rng=420)
dataset.orders = [1, 2, "paragraph", "page"]

sample = dataset[0]
image = sample["image"]
text = sample["text"]
```

Assign `orders` before requesting samples. Integer orders select contiguous
line sequences; `"paragraph"` and `"page"` select those larger units. Sample
transforms can be added with `add_transform()` or replaced with
`set_transform()`.

`LatexRenderer` is another way to construct an `OCRPage`:

```python
from synthscript.rendering.latex import LatexRenderer

page = LatexRenderer().render(r"A line with $x^2$.")
```

It supports a defined subset of prose, math, and TikZ in a single page and
raises `UnsupportedLatexError` for unsupported source constructs it detects.
The resulting page can be passed to `OCRDataset` without rerunning TeX for
each sample.

## Tests

Implementation tests use local or synthetic objects and do not need the
annotation server or image bucket:

```bash
python -m pytest test/implementation
```

Renderer cases requiring local TeX tools or `resvg_py` are skipped when those
dependencies are unavailable.
