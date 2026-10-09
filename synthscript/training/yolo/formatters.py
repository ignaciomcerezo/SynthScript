from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box
from ultralytics.utils.ops import (  # ty: ignore[unresolved-import]
    xyxyxyxy2xywhr,
)

from synthscript.datasets.segmentation.formatters import _polygon_to_mask
from synthscript.training.yolo.helpers import letterbox


class _YOLOFormatter:
    def __init__(self, imgsz: int = 640):
        self.imgsz = imgsz

    def _prepare_sample(
        self, image: np.ndarray, polygons: Sequence[Polygon], index: int
    ) -> tuple[dict, list[Polygon]]:
        h0, w0 = image.shape[:2]
        canvas, r, pad = letterbox(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), self.imgsz)
        left, top = pad
        sample = {
            "img": torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))),
            "im_file": f"synthetic_{index}.jpg",
            "ori_shape": (h0, w0),
            "resized_shape": (self.imgsz, self.imgsz),
            "ratio_pad": ((r, r), pad),
        }
        return sample, [
            affine_transform(poly, (r, 0, 0, r, left, top)) for poly in polygons
        ]

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Stack images and flatten instance targets with their source image indices."""
        result = {}
        for key in batch[0]:
            values = [sample[key] for sample in batch]
            if key in {"img", "sem_masks"}:
                result[key] = torch.stack(values)
            elif key in {"cls", "bboxes", "masks"}:
                result[key] = torch.cat(values)
            else:
                result[key] = values
        result["batch_idx"] = torch.cat(
            [torch.full((len(sample["cls"]),), i) for i, sample in enumerate(batch)]
        ).float()
        return result


class YOLOSegmentationFormatter(_YOLOFormatter):
    """Format rendered line polygons for Ultralytics instance segmentation."""

    def __init__(
        self, imgsz: int = 640, mask_ratio: int = 4, overlap_mask: bool = False
    ):
        super().__init__(imgsz)
        if not 1 <= mask_ratio <= imgsz:
            raise ValueError("mask_ratio must be between 1 and imgsz.")
        self.mask_ratio = mask_ratio
        self.overlap_mask = overlap_mask

    def __call__(
        self, image: np.ndarray, polygons: Sequence[Polygon], index: int = 0
    ) -> dict:
        sample, polygons = self._prepare_sample(image, polygons, index)
        mh = mw = self.imgsz // self.mask_ratio
        bbox_list, mask_list = [], []
        canvas_bounds = box(0, 0, self.imgsz, self.imgsz)

        for poly in polygons:
            if poly.area == 0:
                poly = poly.buffer(1.5)
            visible_poly = poly.intersection(canvas_bounds)
            if visible_poly.is_empty or visible_poly.area <= 0:
                continue
            min_x, min_y, max_x, max_y = visible_poly.bounds
            x0 = max(0, int(min_x))
            y0 = max(0, int(min_y))
            x1 = min(self.imgsz - 1, int(max_x))
            y1 = min(self.imgsz - 1, int(max_y))
            if x0 > x1 or y0 > y1:
                continue
            bbox_list.append(
                (
                    (x0 + x1) / 2 / self.imgsz,
                    (y0 + y1) / 2 / self.imgsz,
                    (x1 - x0 + 1) / self.imgsz,
                    (y1 - y0 + 1) / self.imgsz,
                )
            )
            mask_list.append(
                cv2.resize(
                    _polygon_to_mask(poly, self.imgsz, self.imgsz),
                    (mw, mh),
                    interpolation=cv2.INTER_NEAREST,
                )
            )

        n = len(bbox_list)
        bboxes = torch.tensor(bbox_list, dtype=torch.float32).reshape(n, 4)
        masks = (
            torch.from_numpy(np.stack(mask_list)).float()
            if mask_list
            else torch.zeros((0, mh, mw), dtype=torch.float32)
        )
        if self.overlap_mask:
            # Smaller instances overwrite larger ones, with boxes in the same order.
            order = torch.argsort(masks.sum(dim=(1, 2)), descending=True)
            bboxes = bboxes[order]
            overlap = torch.zeros((1, mh, mw), dtype=torch.float32)
            for i, mask in enumerate(masks[order]):
                overlap[0][mask.bool()] = i + 1
            masks = overlap

        sample.update(
            cls=torch.zeros((n, 1), dtype=torch.float32),
            bboxes=bboxes,
            masks=masks,
            # All instances belong to the single "line" class (class index zero).
            sem_masks=torch.zeros((mh, mw), dtype=torch.float32),
        )
        return sample


class YOLOOBBFormatter(_YOLOFormatter):
    """Format rendered line polygons as normalized xywhr boxes for Ultralytics."""

    def __call__(
        self, image: np.ndarray, polygons: Sequence[Polygon], index: int = 0
    ) -> dict:

        sample, polygons = self._prepare_sample(image, polygons, index)
        canvas_bounds = box(0, 0, self.imgsz, self.imgsz)
        corners = []
        for poly in polygons:
            visible_poly = poly.intersection(canvas_bounds)
            if visible_poly.is_empty or visible_poly.area <= 0:
                continue
            rectangle = visible_poly.minimum_rotated_rectangle
            corners.append(np.asarray(rectangle.exterior.coords[:4], dtype=np.float32))

        n = len(corners)
        if corners:
            bboxes = xyxyxyxy2xywhr(torch.from_numpy(np.stack(corners)))
            # Normalize spatial coordinates only; the angle stays in radians.
            bboxes[:, :4] /= self.imgsz
        else:
            bboxes = torch.zeros((0, 5), dtype=torch.float32)
        sample.update(cls=torch.zeros((n, 1), dtype=torch.float32), bboxes=bboxes)
        return sample
