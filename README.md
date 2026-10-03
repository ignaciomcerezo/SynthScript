# synthscript

`synthscript` is a Python package for generating OCR training datasets from annotated document images and performing fine-tunign and RL on models. It provides a PyTorch-compatible `OCRDataset` class that samples variable-length sequences of document lines with optional augmentation transforms.

## Overview

This repository provides dataset generation and augmentation for handwritten document OCR. It is designed to:

- ingest PAGE-XML annotations and document images (including exports from Label Studio),
- extract and normalize image crops with automatic layout analysis,
- produce an `OCRDataset` that samples contiguous line sequences at training time,
- apply augmentation layout-aware transforms (line, paragraph, page and image transforms),
- handle complex geometry with automatic intersection correction and stroke/background separation.

Also defines a structural syntactic metric for LaTeX snippets that can be used to compute error in LaTeX transcriptions.

## Main components

- **`synthscript.shared`**  
  Shared data structures and utilities used across the pipeline.

- **`synthscript.loading`**  
  Loading the data, both from external sources (downloading images from a bucket) and organizing them on disk, and loading them from there.

- **`synthscript.ocr_units`**
  Core classes, used to perform the geometric analysis of the layout and represent a more structured version of the page.

- **`synthscript.datasets`**  
  The main `OCRDataset` (and other dataset variants) class for training, with configurable line-sequence sampling and clustering.

- **`synthscript.transforms`**  
  Image and geometry augmentation transforms: individual linea, paragraph and whole page layout transforms. Also implementes some image-only transforms, more typical of image augmentation.

- **`synthscript.metric`**
  LaTeX AST based on pylatexenc and weighted tree edit distance to compute structural distances between base truth transcription and prediction.

## Package metadata

- **Package name:** `synthscript`
- **Python:** `>=3.10`
- **Core dependencies:** `numpy`, `levenshtein`, `label-studio-sdk`, `lxml`, `opencv-python`, `pydantic`, `pylatexenc`, `python-dotenv`, `requests`, `shapely`, `tqdm`, `torch`, `zss`
- **Training extras (`[train]`):** `torch`,  `torchvision`, `accelerate`, `trl`, `sentencepiece`, `protobuf`, `huggingface-hub`, `transformers`, `peft`, `pillow`; on Linux also `xformers`, `bitsandbytes`, `triton`, `cut-cross-entropy`, `unsloth-zoo`, and `unsloth`
- **Dev extras (`[dev]`)**: `ipywidgets`, `ty`, `ipykernel`, `ruff`, `pytest`

## Annotation format

SynthScript uses PAGE-XML 2024-07-15 as its canonical annotation and exchange
format. A dataset has this portable layout:

```text
data_in/
├── images/
│   ├── raw/
│   ├── stroke/
│   └── background/
├── page/
│   └── s0_pg123.xml
└── exports/
```

`raw` contains the original image. `stroke` and `background` contain the two
SynthScript-specific representations used to construct synthetic manuscripts;
the PAGE document references them as `AlternativeImage` elements. Image paths
in PAGE are relative and use POSIX separators, so the complete dataset can be
moved between machines.

Each PAGE file contains the line and paragraph geometry, original ground-truth
text, reading order, orientation, and provenance for one annotation. Persisted
coordinates are absolute integer pixels. Text homogenization is applied only
while loading a dataset and never overwrites the PAGE ground truth.

Existing five-JSON manifests can be converted once with
`migrate_legacy_manifest(root_path)`. The converter validates and reloads every
generated PAGE file before reporting success; legacy directories are retained
unless `delete_legacy=True` is explicitly requested.

## Intended use

The package is intended for OCR training workflows where:

- source documents are segmented in regions, with each region having its corresponding transcription,
- sampling variable-length sequences of document regions is desirable (single lines, paragraphs, full pages),
- on-the-fly augmentation is desired during training,
- geometric transforms (rotation, scaling, distortion) should be applied to line crops,
- the dataset integrates with `torch.utils.data.Dataset` for PyTorch training (otherwise it probably can be adapted using custom formatters or other integration).

## Installation

Install the core package:

```bash
pip install .
```

Install training-related extras:

```bash
pip install ".[train]"
```
