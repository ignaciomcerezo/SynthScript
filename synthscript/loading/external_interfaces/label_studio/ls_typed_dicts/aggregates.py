from typing import Annotated, Any

from pydantic import BaseModel, Field

from synthscript.loading.external_interfaces.label_studio.ls_typed_dicts.results import (
    PolygonResult,
    RectangleResult,
    RelationResult,
    TextCorrectionResult,
    TextRegionResult,
)

ResultItem = Annotated[
    TextRegionResult
    | TextCorrectionResult
    | RectangleResult
    | PolygonResult
    | RelationResult,
    Field(discriminator="type"),
]

ResultItemNotRelation = (
    TextRegionResult | TextCorrectionResult | RectangleResult | PolygonResult
)


class RawAnnotation(BaseModel):
    id: int
    completed_by: int
    result: list[ResultItem]
    result_count: int
    was_cancelled: bool
    ground_truth: bool
    created_at: str
    updated_at: str
    lead_time: float
    unique_id: str
    bulk_created: bool
    task: int
    project: int
    updated_by: int

    # Optional fields based on trace
    draft_created_at: str | None = None
    import_id: int | None = None
    last_action: Any | None = None
    last_created_by: Any | None = None
    parent_annotation: int | None = None
    parent_prediction: Any | None = None
    prediction: dict = {}


class TaskData(BaseModel):
    image_url: str
    transcription: str


class LabelStudioTask(BaseModel):
    id: int
    inner_id: int
    file_upload: str
    created_at: str
    updated_at: str
    project: int
    updated_by: int

    data: TaskData
    annotations: list[RawAnnotation]
    drafts: list[Any]
    predictions: list[Any]
    meta: dict

    total_annotations: int
    cancelled_annotations: int
    total_predictions: int

    comment_authors: list[Any]
    comment_count: int
    unresolved_comment_count: int
    last_comment_updated_at: Any | None = None
