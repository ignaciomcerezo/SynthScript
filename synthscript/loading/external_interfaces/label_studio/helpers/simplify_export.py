from tqdm.auto import tqdm

from synthscript.loading.external_interfaces.label_studio.ls_typed_dicts import (
    LabelStudioTask,
    PolygonResult,
    RectangleResult,
    RelationResult,
    ResultItem,
    SimplifiedAnnotation,
    SimplifiedResultItem,
    SimplifiedTask,
    SimplifiedTextCorrectionResult,
    TextCorrectionResult,
    TextRegionResult,
)

newline = "\n"
tab = "\t"


def resolve_text_for_group(group: list[ResultItem], full_text: str) -> list[str]:
    correction_res_list: list[TextCorrectionResult] = [
        r for r in group if isinstance(r, TextCorrectionResult)
    ]
    collected_corrections = []

    for res in correction_res_list:
        val = res.value.text
        if isinstance(val, list):
            valid_texts = [v for v in val if isinstance(v, str) and v.strip()]
            collected_corrections.extend(valid_texts)
        elif isinstance(val, str) and val.strip():
            collected_corrections.append(val)

    if collected_corrections:
        return collected_corrections

    label_res: TextRegionResult | None = next(
        (
            TextRegionResult.model_validate(r)
            for r in group
            if r.type in ["labels", "hypertextlabels"]
        ),
        None,
    )
    if label_res:
        val = label_res.value
        if val.text:
            t = val.text
            return t if isinstance(t, list) else [t]

        if full_text:
            try:
                return [full_text[int(val.start) : int(val.end)]]
            except Exception:
                pass
    return []


def convert_result_raw(
    obj: (
        dict
        | RelationResult
        | TextCorrectionResult
        | PolygonResult
        | RectangleResult
        | TextRegionResult
    ),
) -> (
    RelationResult
    | PolygonResult
    | TextCorrectionResult
    | RectangleResult
    | TextRegionResult
):
    if not isinstance(obj, dict):
        return obj
    match obj.get("type"):
        case "relation":
            return RelationResult.model_validate(obj)
        case "polygonlabels":
            return PolygonResult.model_validate(obj)
        case "textarea":
            return TextCorrectionResult.model_validate(obj)
        case "rectanglelabels":
            return RectangleResult.model_validate(obj)
        case "labels" | "hypertextlabels":
            return TextRegionResult.model_validate(obj)
        case _:
            raise ValueError(f"Unknown result type: {obj.get('type')}")


def simplify_task(task_input: dict | LabelStudioTask) -> SimplifiedTask:
    task = (
        LabelStudioTask.model_validate(task_input)
        if isinstance(task_input, dict)
        else task_input
    )
    full_text = task.data.transcription
    empty_boxes_on_page = 0
    simplified_annotations: list[SimplifiedAnnotation] = []

    for ann in task.annotations:
        new_results: list[SimplifiedResultItem] = []
        results_by_id: dict[str, list[ResultItem]] = {}
        relations: list[RelationResult] = []

        for res in (convert_result_raw(r) for r in ann.result):
            if isinstance(res, RelationResult):
                relations.append(res)
                continue
            results_by_id.setdefault(res.id, []).append(res)

        for rid, group in results_by_id.items():
            box_res = next(
                (
                    item
                    for item in group
                    if isinstance(item, (RectangleResult, PolygonResult))
                ),
                None,
            )

            if box_res is not None:
                new_results.append(box_res)
            else:
                final_text_list = resolve_text_for_group(group, full_text)
                if not final_text_list:
                    empty_boxes_on_page += 1
                    continue

                if len(final_text_list) > 1:
                    label_res = next(
                        (
                            TextRegionResult.model_validate(r)
                            for r in group
                            if r.type in ["labels", "hypertextlabels"]
                        ),
                        None,
                    )
                    original_text = (
                        str(label_res.value.text)
                        if label_res is not None
                        else "<sin etiqueta original>"
                    )
                    print(
                        f"Task {task.id}: segment '{original_text}' has multiple corrections."
                    )

                final_text = final_text_list[0]
                if final_text and final_text.strip():
                    synthetic_res = SimplifiedTextCorrectionResult.model_validate(
                        {
                            "id": rid,
                            "type": "textarea",
                            "value": {"text": [final_text]},
                            "from_name": "text_adapter",
                            "to_name": "image",
                        }
                    )
                    new_results.append(synthetic_res)
                else:
                    empty_boxes_on_page += 1

        new_results.extend(relations)
        simplified_annotations.append(
            SimplifiedAnnotation.model_validate(
                {**ann.model_dump(), "result": new_results}
            )
        )

    if empty_boxes_on_page > 0:
        print(f"Task {task.id}: ignored {empty_boxes_on_page} empty text segments.")

    return SimplifiedTask.model_validate(
        {**task.model_dump(), "annotations": simplified_annotations}
    )


def simplify_tasks(
    tasks: list[dict | LabelStudioTask],
) -> list[SimplifiedTask]:
    simplified_tasks = []

    for task in tqdm(tasks):
        simplified_tasks.append(simplify_task(task))

    return simplified_tasks
