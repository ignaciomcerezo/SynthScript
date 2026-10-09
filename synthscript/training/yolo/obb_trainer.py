from typing import Literal

from ultralytics.models.yolo.obb.train import (  # ty: ignore[unresolved-import]
    OBBTrainer,
)

from synthscript.datasets.segmentation.segmentation_dataset import SegmentationDataset
from synthscript.training.yolo.formatters import YOLOOBBFormatter


class OBBDatasetTrainer(OBBTrainer):
    """Train Ultralytics OBB models directly from SegmentationDataset instances.

    Polygons are converted to minimum rotated rectangles, so datasets with either
    value of return_bounding_boxes can be used. Configure each dataset's orders
    and transforms before training; augmentation is provided by those datasets.
    A YOLOOBBFormatter is assigned when dataset.formatter is None; an explicitly
    assigned formatter is preserved.

    Example usage:
        trainer = OBBDatasetTrainer(overrides=dict(
            model="yolo11n-obb.pt",
            epochs=100, imgsz=640, batch=32, device=0, workers=4,
            project="/kaggle/working/handwriting", name="line_obb_run",
            plots=False,
        ))
        trainer.train_seg_dataset = train  # SegmentationDataset
        trainer.val_seg_dataset = test  # SegmentationDataset
        trainer.train()
    """

    train_seg_dataset: SegmentationDataset | None = None
    val_seg_dataset: SegmentationDataset | None = None

    def get_dataset(self):
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
            seg_ds.formatter = YOLOOBBFormatter(imgsz=self.args.imgsz)
        return seg_ds

    def plot_training_labels(self):
        """Skip static label plots because synthetic labels are generated on access."""

    def final_eval(self):
        print("Skipping final_eval re-validation.")
