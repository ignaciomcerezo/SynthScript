from pathlib import Path

import cv2
import numpy as np

from synthscript.shared.image_handling import (
    is_grayscale_png,
    load_image_grayscale,
    read_image_shape,
    write_encoded_image_grayscale,
)


def _encode_png(image: np.ndarray) -> bytes:
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded
    return buffer.tobytes()


def test_grayscale_png_is_written_without_reencoding(tmp_path: Path) -> None:
    encoded_image = _encode_png(np.arange(16, dtype=np.uint8).reshape(4, 4))
    destination = tmp_path / "grayscale.png"

    assert is_grayscale_png(encoded_image)

    write_encoded_image_grayscale(destination, encoded_image)

    assert destination.read_bytes() == encoded_image


def test_color_png_is_converted_to_grayscale(tmp_path: Path) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    encoded_image = _encode_png(image)
    destination = tmp_path / "converted.png"

    assert not is_grayscale_png(encoded_image)

    write_encoded_image_grayscale(destination, encoded_image)

    assert is_grayscale_png(destination.read_bytes())
    np.testing.assert_array_equal(
        load_image_grayscale(destination),
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    )


def test_load_image_grayscale_supports_unicode_paths(tmp_path: Path) -> None:
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    path = tmp_path / "imágen.png"
    path.write_bytes(_encode_png(image))

    np.testing.assert_array_equal(load_image_grayscale(path), image)


def test_read_image_shape_from_png_header(tmp_path: Path) -> None:
    path = tmp_path / "dimensions.png"
    path.write_bytes(_encode_png(np.zeros((7, 11), dtype=np.uint8)))

    assert read_image_shape(path) == (7, 11)
