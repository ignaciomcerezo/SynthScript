from pathlib import Path

import cv2
import numpy as np

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_CHUNK_LENGTH = b"\x00\x00\x00\r"
_PNG_IHDR_CHUNK_TYPE = b"IHDR"
_PNG_DIMENSION_HEADER_LENGTH = 24
_PNG_COLOR_TYPE_OFFSET = 25
_PNG_GRAYSCALE_COLOR_TYPE = 0


def is_grayscale_png(encoded_image: bytes) -> bool:
    """Return whether encoded bytes have a single-channel grayscale PNG header."""
    return (
        len(encoded_image) > _PNG_COLOR_TYPE_OFFSET
        and encoded_image.startswith(_PNG_SIGNATURE)
        and encoded_image[8:12] == _PNG_IHDR_CHUNK_LENGTH
        and encoded_image[12:16] == _PNG_IHDR_CHUNK_TYPE
        and encoded_image[_PNG_COLOR_TYPE_OFFSET] == _PNG_GRAYSCALE_COLOR_TYPE
    )


def decode_image_grayscale(encoded_image: bytes | np.ndarray) -> np.ndarray:
    """Decode encoded image data into a two-dimensional grayscale array."""
    if isinstance(encoded_image, bytes):
        encoded_image = np.frombuffer(encoded_image, dtype=np.uint8)

    image = cv2.imdecode(encoded_image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Encoded image could not be decoded.")
    return image


def load_image_grayscale(path: Path) -> np.ndarray:
    """Load an image as grayscale through a Windows-path-safe byte decode."""
    return decode_image_grayscale(np.fromfile(path, dtype=np.uint8))


def read_image_shape(path: Path) -> tuple[int, int]:
    """Read an image height and width using the PNG header if possible"""
    with path.open("rb") as image_file:
        header = image_file.read(_PNG_DIMENSION_HEADER_LENGTH)

    if (
        len(header) == _PNG_DIMENSION_HEADER_LENGTH
        and header.startswith(_PNG_SIGNATURE)
        and header[8:12] == _PNG_IHDR_CHUNK_LENGTH
        and header[12:16] == _PNG_IHDR_CHUNK_TYPE
    ):
        width = int.from_bytes(header[16:20], byteorder="big")
        height = int.from_bytes(header[20:24], byteorder="big")
        if width > 0 and height > 0:
            return height, width

    return load_image_grayscale(path).shape[:2]


def write_encoded_image_grayscale(path: Path, encoded_image: bytes) -> None:
    """Write encoded image data, converting it only when not grayscale PNG."""
    if path.suffix.lower() == ".png" and is_grayscale_png(encoded_image):
        path.write_bytes(encoded_image)
        return

    image = decode_image_grayscale(encoded_image)
    encoded, buffer = cv2.imencode(path.suffix, image)
    if not encoded:
        raise ValueError(f"Image could not be encoded as {path.suffix!r}.")
    path.write_bytes(buffer.tobytes())
