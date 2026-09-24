import functools
import json
import operator
from collections import defaultdict
from collections.abc import Callable, Collection
from pathlib import Path

from tqdm.auto import tqdm

from synthscript.loading.page_metadata import PageSampleMetadata
from synthscript.ocr_units.ocr_page import OCRPage
from synthscript.shared.path_bundle import PathBundle


def load_pages(
    paths: PathBundle,
    *,
    pages: Collection[str | int] | None = None,
    tasks: Collection[int] | None = None,
    combine_same_page_annotations: bool = True,
    length: int | None = None,
    transcription_homogenizer: Callable[[str], str] | None = None,
) -> list[OCRPage]:
    """
    Uses the information stored in paths.metadata_path to access the appropriate
    images, transcriptions, polygons, ids and rotations and creates AnnotatedPage
    instances.
    """

    tasks: set[int] | None = (
        set([task for task in tasks]) if isinstance(tasks, Collection) else None
    )
    pages: set[str] | None = (
        set([str(page) for page in pages]) if isinstance(pages, Collection) else None
    )

    def _acceptable(page, task_id):
        if (pages is None) and (tasks is None):
            return True
        if tasks is None:
            return page in pages  # ty: ignore[unsupported-operator]
        if pages is None:
            return task_id in tasks

        return (task_id in tasks) or (page in pages)

    taskid2annpage: dict[int, list[OCRPage]] = defaultdict(lambda: list())

    k = 0
    for metadata_filepath in tqdm(
        list(Path(paths.metadata_path).iterdir()),
        desc="Building OCRPage objects from disk...",
    ):
        if length is not None and k > length:
            break
        metadata = PageSampleMetadata.model_validate(
            json.loads(metadata_filepath.read_text())
        )

        page = metadata.page
        task_id = metadata.task_id

        if not _acceptable(page, task_id):
            # print(f"Skipping {task_id=}/{page=} (looking for {tasks=} or {pages=})")
            continue

        completer: str = metadata.completer
        updater: str = metadata.updater
        # subindex: int = metadata_content["subindex"]
        # ann_id  = metadata_content["ann_id"]
        # order = metadata_content["order"]

        polygons_are_in_percentage: bool = metadata.polygons_are_in_percentage

        transcriptions = metadata.load_transcriptions()
        if transcription_homogenizer is not None:
            transcriptions = [
                transcription_homogenizer(transcription)
                for transcription in transcriptions
            ]
        polygon_coords = metadata.load_polygon_coords()
        rotations = metadata.load_rotations()
        ids = metadata.load_ids()
        image_path = metadata.image_path

        stroke = paths.load_stroke_image(image_path.stem)
        background = paths.load_background_image(image_path.stem)

        if (stroke is None) or (background is None):
            raise ValueError(
                f"Stroke or background images could not be loaded for task {task_id}/page {page}:\n"
                f"background: {paths.get_background_image_path(image_path.stem)}\n"
                f"stroke: {paths.get_stroke_image_path(image_path.stem)}"
            )

        taskid2annpage[task_id].append(
            OCRPage(
                transcriptions=transcriptions,
                polygon_coords=polygon_coords,
                line_ids=ids,
                rotations=rotations,
                task_id=int(task_id),
                page=page,
                stroke=stroke,
                background=background,
                completer=completer,
                updater=updater,
                polygons_are_in_percentage=polygons_are_in_percentage,
            )
        )
        k += 1

    if combine_same_page_annotations:
        for page, annotations in taskid2annpage.items():
            taskid2annpage[page] = [OCRPage.combine_annotations(*annotations)]

    return functools.reduce(operator.iadd, taskid2annpage.values(), [])
