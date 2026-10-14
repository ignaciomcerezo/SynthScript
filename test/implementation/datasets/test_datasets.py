import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from synthscript.datasets.segmentation import SegmentationDataset
from synthscript.datasets.transcription.layout_ocrdataset import LayoutOCRDataset
from synthscript.datasets.transcription.ocrdataset import OCRDataset
from synthscript.transforms.line.linewise_horizontal_stretch import (
    LinewiseHorizontalStretch,
)
from test.implementation.object_mothers import make_page, make_two_line_page


def test_ocrdataset_sample_from_synthetic_page():
    page = make_two_line_page()
    dataset = OCRDataset([page])
    assert dataset.orders is None
    with pytest.raises(RuntimeError, match=r"Assign dataset\.orders"):
        dataset[0]

    dataset.orders = [2]
    sample = dataset[0]
    assert isinstance(sample["image"], np.ndarray)
    assert sample["text"] == "first\nsecond"
    assert isinstance(sample["sindex"], int)
    assert isinstance(sample["context"], str)
    assert sample["order"] == 2
    assert isinstance(sample["id"], str)
    assert sample["page_id"] == page.page_id


def test_segmentation_sample_from_synthetic_page():
    page = make_two_line_page()
    dataset = SegmentationDataset([page])
    assert dataset.orders is None
    with pytest.raises(RuntimeError, match=r"Assign dataset\.orders"):
        dataset[0]

    dataset.orders = [2]
    image, polygons = dataset[0]
    assert isinstance(image, np.ndarray)
    assert len(polygons) == 2
    assert all(isinstance(polygon, (Polygon, MultiPolygon)) for polygon in polygons)


def test_layout_dataset_exposes_both_transform_stages():
    dataset = LayoutOCRDataset([])
    renderer = dataset.renderer
    sample_transform = LinewiseHorizontalStretch(1.1)
    layout_transform = LinewiseHorizontalStretch(1.2)

    dataset.add_transform(sample_transform)
    assert dataset.mixed_transforms is not None
    assert dataset.layout_transforms is None

    dataset.add_layout_transform(layout_transform)
    assert dataset.layout_transforms is not None
    assert dataset.renderer is renderer

    dataset.set_transform()
    assert dataset.mixed_transforms is None
    assert dataset.layout_transforms is not None
    assert dataset.renderer is renderer

    dataset.set_layout_transform()
    assert dataset.layout_transforms is None
    assert dataset.renderer is renderer


def test_layout_transform_takes_effect_on_refresh():
    page = make_page(
        shape=(20, 30),
        polygon_coords=[[(2, 2), (12, 2), (12, 8), (2, 8)]],
        paragraph_line_ids=[["line"]],
    )
    dataset = LayoutOCRDataset([page])
    initial_width = dataset._annotated_pages[0].lines["line"].crop.shape[1]

    dataset.add_layout_transform(LinewiseHorizontalStretch(1.5))
    assert dataset._annotated_pages[0].lines["line"].crop.shape[1] == initial_width

    dataset.refresh_layouts()
    assert dataset._annotated_pages[0].lines["line"].crop.shape[1] > initial_width
