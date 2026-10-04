from __future__ import annotations

import cv2
import numpy as np
import torch
from shapely.affinity import affine_transform
from shapely.geometry import box
from torch.utils.data import Dataset

from synthscript.datasets.segmentation.formatters import _polygon_to_mask
from synthscript.datasets.segmentation.segmentation_dataset import SegmentationDataset
from synthscript.training.yolo.helpers import letterbox


class _SegmentationLineDataset(Dataset):
    """
    Wraps one `SegmentationDataset` sample -- (np.ndarray, list[Polygon]) -- into
    the per-image dict Ultralytics' segmentation loss expects.
    """

    def __init__(
        self, seg_dataset: SegmentationDataset, imgsz: int = 640, mask_ratio: int = 4
    ):
        self._ds = seg_dataset
        self._imgsz = imgsz
        self._mask_ratio = mask_ratio

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> dict:
        image, polygons = self._ds[index]
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h0, w0 = img_bgr.shape[:2]

        canvas, r, pad = letterbox(img_bgr, self._imgsz)

        mh = mw = self._imgsz // self._mask_ratio
        cls_list, bbox_list, mask_list = [], [], []
        left, top = pad
        canvas_bounds = box(0, 0, self._imgsz, self._imgsz)

        for poly in polygons:
            letterboxed_poly = affine_transform(poly, (r, 0, 0, r, left, top))
            if letterboxed_poly.area == 0:
                letterboxed_poly = letterboxed_poly.buffer(1.5)

            visible_poly = letterboxed_poly.intersection(canvas_bounds)
            if visible_poly.is_empty:
                continue

            lb_mask = _polygon_to_mask(letterboxed_poly, self._imgsz, self._imgsz)

            min_x, min_y, max_x, max_y = visible_poly.bounds
            x0 = max(0, int(min_x))
            y0 = max(0, int(min_y))
            x1 = min(self._imgsz - 1, int(max_x))
            y1 = min(self._imgsz - 1, int(max_y))
            if x0 > x1 or y0 > y1:
                continue

            cls_list.append(0.0)
            bbox_list.append(
                (
                    (x0 + x1) / 2 / self._imgsz,
                    (y0 + y1) / 2 / self._imgsz,
                    (x1 - x0 + 1) / self._imgsz,
                    (y1 - y0 + 1) / self._imgsz,
                )
            )
            mask_list.append(
                cv2.resize(lb_mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
            )

        n = len(cls_list)
        cls = torch.tensor(cls_list, dtype=torch.float32).reshape(n, 1)
        bboxes = torch.tensor(bbox_list, dtype=torch.float32).reshape(n, 4)
        masks = (
            torch.from_numpy(np.stack(mask_list)).float()
            if mask_list
            else torch.zeros((0, mh, mw), dtype=torch.float32)
        )

        return {
            "img": torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))),
            "cls": cls,
            "bboxes": bboxes,
            "masks": masks,
            "im_file": f"synthetic_{index}.jpg",
            "ori_shape": (h0, w0),
            "resized_shape": (self._imgsz, self._imgsz),
            "ratio_pad": ((r, r), pad),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Mirrors Ultralytics' own YOLODataset.collate_fn: images stack normally,
        but per-instance targets (cls/bboxes/masks) are concatenated across the
        whole batch and tied back to their source image via a `batch_idx` column,
        instead of being padded per-image. Ultralytics' loss functions expect this
        flattened representation."""
        batch_idx = torch.cat(
            [torch.full((len(b["cls"]),), i) for i, b in enumerate(batch)]
        )
        return {
            "img": torch.stack([b["img"] for b in batch]),
            "batch_idx": batch_idx.float(),
            "cls": torch.cat([b["cls"] for b in batch]),
            "bboxes": torch.cat([b["bboxes"] for b in batch]),
            "masks": torch.cat([b["masks"] for b in batch]),
            "im_file": [b["im_file"] for b in batch],
            "ori_shape": [b["ori_shape"] for b in batch],
            "resized_shape": [b["resized_shape"] for b in batch],
            "ratio_pad": [b["ratio_pad"] for b in batch],
        }


# ---------------------------------------------------------------------------
# Custom trainer
# ---------------------------------------------------------------------------
