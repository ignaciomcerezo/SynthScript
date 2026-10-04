import cv2
import numpy as np


def letterbox(
    img: np.ndarray, size: int, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resizes `img` to fit inside a (size, size) canvas preserving aspect ratio
    and padding with 'color'."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - new_w, size - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return canvas, r, (left, top)


def letterbox_mask(
    mask: np.ndarray, size: int, r: float, pad: tuple[int, int]
) -> np.ndarray:
    """Applies the exact scale/pad computed by _letterbox for the paired image to
    a mask, using nearest-neighbour interpolation."""
    h, w = mask.shape[:2]
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    left, top = pad
    right, bottom = size - new_w - left, size - new_h - top
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
