from dataclasses import dataclass, field

from synthscript.datasets.image_transform_pack import ImageTransformPack
from synthscript.datasets.ocr_transform_pack import OCRTransformPack
from synthscript.transforms.transforms import ImageTransform, OCRTransform


@dataclass
class TransformInfo:
    name: str
    arguments: dict[str, str]
    probability: float


@dataclass
class TransformParams:
    line: list[TransformInfo] = field(default_factory=list)
    paragraph: list[TransformInfo] = field(default_factory=list)
    page: list[TransformInfo] = field(default_factory=list)
    stroke: list[TransformInfo] = field(default_factory=list)
    background: list[TransformInfo] = field(default_factory=list)
    global_image: list[TransformInfo] = field(default_factory=list)

    @staticmethod
    def _transform_info(transform: OCRTransform | ImageTransform) -> TransformInfo:
        arguments = {
            name.lstrip("_"): repr(value)
            for name, value in vars(transform).items()
            if name
            not in {"_rng", "_probability", "may_cause_intersections", "type2map"}
        }
        return TransformInfo(
            name=type(transform).__name__,
            arguments=arguments,
            probability=transform.probability,
        )

    @classmethod
    def from_transform_packs(
        cls,
        ocr_transform_pack: OCRTransformPack | None = None,
        image_transform_pack: ImageTransformPack | None = None,
    ) -> "TransformParams":
        params = cls()

        if ocr_transform_pack is not None:
            params.line = [
                cls._transform_info(transform) for transform in ocr_transform_pack._line
            ]
            params.paragraph = [
                cls._transform_info(transform)
                for transform in ocr_transform_pack._paragraph
            ]
            params.page = [
                cls._transform_info(transform) for transform in ocr_transform_pack._page
            ]

        if image_transform_pack is not None:
            params.stroke = [
                cls._transform_info(transform)
                for transform in image_transform_pack._stroke
            ]
            params.background = [
                cls._transform_info(transform)
                for transform in image_transform_pack._background
            ]
            params.global_image = [
                cls._transform_info(transform)
                for transform in image_transform_pack._global_image
            ]

        return params
