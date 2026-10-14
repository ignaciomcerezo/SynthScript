import numpy as np
import pytest

from synthscript.shared.image_processing import to_grayscale


@pytest.mark.parametrize("shape", [(5, 7), (5, 7, 1), (5, 7, 3), (5, 7, 4)])
def test_to_grayscale_accepts_supported_channel_layouts(shape):
    image = np.zeros(shape, dtype=np.uint8)

    result = to_grayscale(image)

    assert result.shape == (5, 7)
    assert result.dtype == np.uint8
