import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, model_validator

from synthscript.shared.path_bundle import PathBundle


def _validate_path_and_existance(path):

    if not isinstance(path, (Path, str)):
        raise ValueError(
            f"Failed path validation: Incorrect path format (path='{path}')."
        )

    path = Path(path) if not isinstance(path, Path) else path

    if not path.exists():
        raise ValueError(f"Failed path validation: path does not exist ('{path}').")


def _load_path_content(path: Path):
    return json.loads(path.read_text())


def _validate_polygons(
    polygons_list: Any,
) -> list[list[tuple[float, float]]]:

    if not isinstance(polygons_list, list):
        raise ValueError(
            f"Failed polygon path validation: the outer element must be a list, found {type(polygons_list)}."
        )

    for polyElement in polygons_list:

        if not isinstance(polyElement, list):
            raise ValueError(
                f"Failed polygon path validation: an element of the outer list is not a list itself, found {type(polyElement)}."
            )
        for xyElement in polyElement:
            if not isinstance(xyElement, (tuple, list)):
                raise ValueError(
                    f"Failed polygon path validation: an element of the inner list is not a tuple or list, found {type(xyElement)}."
                )
            if not (len(xyElement) == 2):
                raise ValueError(
                    f"Failed polygon path validation: one of the tuples in the inner list is of length {len(xyElement)} != 2."
                )

            if not (
                isinstance(xyElement[0], (float, int))
                and isinstance(xyElement[1], (float, int))
            ):
                raise ValueError(
                    f"Failed polygon path validation: one of the tuples contains non-float non-int objects: {[type(z) for z in xyElement]}"
                )
    return polygons_list


def _validate_transcriptions(transcriptions: Any) -> list[str]:

    if not isinstance(transcriptions, list):
        raise ValueError(
            f"Failed transcriptions path validation: the outer element must be a list, found {type(transcriptions)}."
        )

    if not all(isinstance(transcription, str) for transcription in transcriptions):
        raise ValueError(
            f"Failed transcrpitions path: expected all elements to be str, found types {set(type(x) for x in transcriptions)}"
        )
    return transcriptions


def _validate_rotations(rotations: Any) -> list[int | float]:

    if not isinstance(rotations, list):
        raise ValueError(
            f"Failed rotations path validation: the outer element must be a list, found {type(rotations)}."
        )

    if not all(isinstance(rotation, (int, float)) for rotation in rotations):
        raise ValueError(
            f"Failed rotations path: expected all elements to be float or int, found {set(type(x) for x in rotations)}"
        )

    return rotations


def _validate_ids(ids: Any) -> list[str]:

    if not isinstance(ids, list):
        raise ValueError(
            f"Failed ids path validation: the outer element must be a list, found {type(ids)}."
        )

    if not all(isinstance(id_i, str) for id_i in ids):
        raise ValueError(
            f"Failed ids path: expected all elements to be str, found {set(type(id_i) for id_i in ids)}"
        )

    return ids


def _validate_image_path(image_path: Path) -> Path:
    stroke_path = PathBundle.change_image_category_path(image_path, "raw")
    background_path = PathBundle.change_image_category_path(image_path, "background")
    if not image_path.exists():
        raise ValueError("The raw image does not exist.")
    if not stroke_path.exists():
        print(stroke_path)
        raise ValueError("The stroke image does not exist.")
    if not background_path.exists():
        print(background_path)
        raise ValueError("The background image does not exist.")

    return image_path


class PageSampleMetadata(BaseModel):
    page: str
    task_id: int
    subindex: int
    completer: str
    updater: str
    ann_id: int
    order: int
    image_path: Annotated[Path, _validate_image_path]
    ids_path: Path
    transcriptions_path: Path
    polygons_path: Path
    rotations_path: Path
    polygons_are_in_percentage: bool
    source: str

    @model_validator(mode="after")
    def setup_check_booleans(self):
        self.invalidate_caches()
        return self

    def invalidate_caches(self) -> None:
        self._polygons: list[list[tuple[float, float]]] | None = None
        self._transcriptions: list[str] | None = None
        self._rotations: list[int | float] | None = None
        self._ids: list[str] | None = None

    def load_transcriptions(self) -> list[str]:
        if self._transcriptions is None:
            transcriptions = _load_path_content(self.transcriptions_path)
            self._transcriptions: list[str] = _validate_transcriptions(transcriptions)
        return self._transcriptions

    def load_polygon_coords(self) -> list[list[tuple[float, float]]]:
        if self._polygons is None:
            polygons_list = _load_path_content(self.polygons_path)
            self._polygons: list[list[tuple[float, float]]] = _validate_polygons(
                polygons_list
            )
        return self._polygons

    def load_rotations(self) -> list[float | int]:
        if self._rotations is None:
            rotations = _load_path_content(self.rotations_path)
            self._rotations = _validate_rotations(rotations)
        return self._rotations

    def load_ids(self) -> list[str]:
        if self._ids is None:
            ids = _load_path_content(self.ids_path)
            self._ids = _validate_ids(ids)
        return self._ids

    def load_raw(self) -> np.ndarray:
        path = PathBundle.change_image_category_path(self.image_path, "raw")
        return PathBundle.load_image_grayscale_np(path)

    def load_stroke(self) -> np.ndarray:
        path = PathBundle.change_image_category_path(self.image_path, "stroke")
        return PathBundle.load_image_grayscale_np(path)

    def load_background(self) -> np.ndarray:
        path = PathBundle.change_image_category_path(self.image_path, "background")
        return PathBundle.load_image_grayscale_np(path)
