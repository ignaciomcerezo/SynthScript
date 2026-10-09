from collections import defaultdict
from collections.abc import Callable, Collection
from pathlib import Path

from tqdm.auto import tqdm

from synthscript.loading.page_xml.parser import (
    load_page_xml_document,
    page_xml_identity,
)
from synthscript.loading.page_xml.validation import parse_xml_document
from synthscript.metric.homogenizer.ast_homogenizer import ASTHomogenizer
from synthscript.metric.homogenizer.homogenizer import TextHomogenizer
from synthscript.ocr_units.ocr_page import OCRPage
from synthscript.shared.path_bundle import PathBundle


def load_pages(
    *,
    root_path: PathBundle | Path | str | None = None,
    pages: Collection[str | int] | None = None,
    tasks: Collection[int] | None = None,
    combine_same_page_annotations: bool = True,
    length: int | None = None,
    transcription_homogenizer: TextHomogenizer | Callable[[str], str] | None = None,
) -> list[OCRPage]:
    """Load PAGE-XML annotations from disk into OCRPage objects."""
    paths = (
        PathBundle(root_path) if not isinstance(root_path, PathBundle) else root_path
    )
    wanted_tasks = set(tasks) if tasks is not None else None
    wanted_pages = {str(page) for page in pages} if pages is not None else None
    homogenizer = transcription_homogenizer or ASTHomogenizer()

    def acceptable(page_id: str, task_id: int | None) -> bool:
        matches_page = wanted_pages is not None and page_id in wanted_pages
        matches_task = wanted_tasks is not None and task_id in wanted_tasks
        if wanted_pages is None and wanted_tasks is None:
            return True
        return matches_page or matches_task

    pages_by_id: dict[str, list[OCRPage]] = defaultdict(list)
    accepted_count = 0
    xml_paths = sorted(paths.page_xml_path.glob("*.xml"))
    for xml_path in tqdm(xml_paths, desc="Building OCRPage objects from PAGE-XML..."):
        if length is not None and accepted_count >= length:
            break
        document = parse_xml_document(xml_path)
        page_id, task_id = page_xml_identity(document)
        if not acceptable(page_id, task_id):
            continue
        page = load_page_xml_document(
            document,
            xml_path,
            transcription_homogenizer=homogenizer,
        )
        pages_by_id[page.page_id].append(page)
        accepted_count += 1

    result: list[OCRPage] = []
    for annotations in pages_by_id.values():
        if combine_same_page_annotations and len(annotations) > 1:
            result.append(OCRPage.combine_annotations(*annotations))
        else:
            result.extend(annotations)
    return result
