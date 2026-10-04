from dataclasses import dataclass

from synthscript.datasets.image_transform_pack import ImageTransformPack
from synthscript.datasets.ocr_transform_pack import OCRTransformPack


@dataclass
class TransformParams:

    @classmethod
    def from_transform_packs(
        cls,
        ocr_transform_pack: OCRTransformPack | None = None,
        image_transform_pack: ImageTransformPack | None = None,
    ):
        object = cls.__new__(cls)
