from typing import Literal

from ultralytics.models.yolo.segment.train import (  # ty: ignore[unresolved-import]
    SegmentationTrainer,
)

from synthscript.datasets.segmentation.segmentation_dataset import SegmentationDataset
from synthscript.training.yolo.formatters import YOLOSegmentationFormatter


class SegmentationDatasetTrainer(SegmentationTrainer):
    """
    A `SegmentationTrainer` that pulls train/val batches from two
    `SegmentationDataset` instances instead of a YOLO-format images/labels folder
    described by a data.yaml.
    A YOLOSegmentationFormatter is assigned when dataset.formatter is None; an
    explicitly assigned formatter is preserved.

    Example usage:
        trainer = SegmentationDatasetTrainer(overrides=dict(
            model="yolo11n-seg.pt",
            epochs=100, imgsz=640, batch=32, device=0, workers=4,
            lr0=0.01, patience=15,
            project="/kaggle/working/handwriting", name="line_seg_run",
            plots=False,
            overlap_mask=False,
            fliplr=0.0, flipud=0.0,
            mosaic=0.0, mixup=0.0, copy_paste=0.0,
        ))
        trainer.train_seg_dataset = train # SegmentationDataset
        trainer.val_seg_dataset = test # SegmentationDataset
        trainer.train()
    """

    train_seg_dataset: SegmentationDataset | None = None
    val_seg_dataset: SegmentationDataset | None = None

    def get_dataset(self):
        # short-circuits Ultralytics check_det_dataset(), which expects a YAML
        # pointing at image or label directories on disk and will error out
        self.data = {
            "train": "train",
            "val": "val",
            "nc": 1,
            "names": {0: "line"},
            "channels": 3,
        }
        return self.data

    def build_dataset(
        self, img_path: str, mode: Literal["train", "val"] = "train", batch=None
    ):
        seg_ds = self.train_seg_dataset if mode == "train" else self.val_seg_dataset
        if seg_ds is None:
            raise RuntimeError(
                f"trainer.{mode}_seg_dataset is not set -- assign your "
                "SegmentationDataset instances before calling .train()."
            )
        if seg_ds.formatter is None:
            seg_ds.formatter = YOLOSegmentationFormatter(
                imgsz=self.args.imgsz,
                mask_ratio=self.args.mask_ratio,
                overlap_mask=self.args.overlap_mask,
            )
        return seg_ds

    def plot_training_labels(self):
        """Skip static label plots because synthetic labels are generated on access."""

    def final_eval(self):
        print("Skipping final_eval re-validation.")
