import cv2
from tqdm.auto import tqdm

from synthscript.loading.external_interfaces.external_interface import ExternalInterface
from synthscript.shared.default_parameters import (
    DATASET_LONGEST_SIZE_PX,
    PROCESSING_LONGEST_SIDE_PX,
)
from synthscript.shared.image_processing import separate_background_and_stroke
from synthscript.shared.path_bundle import PathBundle


class ImageSeparationInterface(ExternalInterface):
    def __init__(self):
        pass

    def __repr__(self):
        return "<ImageSeparationInterface'>."

    def parts_required(self):
        return {"raw_images"}

    def parts_managed(self):
        return {"background_images", "stroke_images"}

    def setup(self, paths: PathBundle):
        for raw_image_path in tqdm(
            list(paths.raw_images_path.iterdir()),
            desc="Stroke/background separation...",
        ):

            raw_image = paths.load_image_grayscale_np(raw_image_path)

            if raw_image is None:
                raise ValueError("Tried to separate nonexistent image.")

            if paths.has_processed_images(raw_image_path.stem):
                continue

            background, stroke = separate_background_and_stroke(
                raw_image,
                out_longest_side=DATASET_LONGEST_SIZE_PX,
                processing_longest_side=PROCESSING_LONGEST_SIDE_PX,
            )
            cv2.imwrite(
                PathBundle.change_image_category_path(raw_image_path, "stroke"), stroke
            )
            cv2.imwrite(
                PathBundle.change_image_category_path(raw_image_path, "background"),
                background,
            )
