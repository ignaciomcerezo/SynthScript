import os
from pathlib import Path
from urllib.parse import unquote as url_unquote

from label_studio_sdk import Client
from shapely import make_valid
from shapely.geometry import Polygon

from synthscript.loading.page_xml import save_page_xml
from synthscript.loading.page_xml.constants import (
    LABEL_STUDIO_ANNOTATION_ID,
    LABEL_STUDIO_COMPLETER,
    LABEL_STUDIO_CREATED_AT,
    LABEL_STUDIO_GROUND_TRUTH,
    LABEL_STUDIO_LEAD_TIME,
    LABEL_STUDIO_PROJECT,
    LABEL_STUDIO_TASK_ID,
    LABEL_STUDIO_UNIQUE_ID,
    LABEL_STUDIO_UPDATED_AT,
    LABEL_STUDIO_UPDATER,
    SYNTHSCRIPT_SOURCE,
    SYNTHSCRIPT_SUBINDEX,
)
from synthscript.ocr_units import OCRPage
from synthscript.shared.geometry_processing import calculate_reading_angle
from synthscript.shared.path_bundle import PathBundle

from ..external_interface import ExternalInterface
from .helpers.json_conversor import (
    extract_bounds,
    pair_lines,
)
from .helpers.simplify_export import (
    simplify_tasks,
)
from .ls_typed_dicts import (
    RectangleResult,
    SimplifiedTask,
)


class _LSUsersManager:
    def __init__(self):
        self.usernames: list[str] | None = None

    def __getitem__(self, index):
        if self.usernames is None:
            return "Offline/Unknown"
        elif index < len(self.usernames):
            return self.usernames[index]
        else:
            return "Impossible username"

    def update_usernames(self, usernames: list[str]):
        self.usernames = usernames

    def __repr__(self):
        return f"<_LSUsersManager with usernames={self.usernames}.>"


class LabelStudioInterface(ExternalInterface):
    """Manages Label Studio integration by pulling remote tasks and
    transforming them directly into serialized dataset artifacts.
    """

    slots = (
        "project_id",
        "server_url",
        "token",
        "online",
        "paths",
        "usernames",
    )

    def __init__(
        self,
        server_url: str,
        token: str,
        project_id: int = 4,
        online: bool = True,
    ):
        self.online = online
        self.project_id = project_id
        self.token = token
        self.url = server_url
        self._usernames: _LSUsersManager = _LSUsersManager()

    @classmethod
    def from_env(
        cls,
        online: bool = True,
        project_id: int = 4,
        ls_token: str | None = None,
        ls_server_url: str | None = None,
        token_env_var: str = "LS_TOKEN",
        url_env_var: str = "LS_URL",
    ) -> "LabelStudioInterface":
        if (token_env_var not in os.environ) and (ls_token is None):
            raise ValueError(
                f"{token_env_var} no está presente en las variables de entorno."
            )
        if (url_env_var not in os.environ) and (ls_server_url is None):
            raise ValueError(
                f"{url_env_var} no está presente en las variables de entorno."
            )

        token = ls_token if ls_token is not None else str(os.getenv(token_env_var))
        url = (
            ls_server_url if ls_server_url is not None else str(os.getenv(url_env_var))
        )

        obj = cls(url, token, project_id, online)
        return obj

    def __repr__(self):
        return f"<LabelStudioInterface with URL {self.url}>"

    def test_connection_successful(self) -> bool:
        try:
            client = Client(url=self.url, api_key=self.token)
            client.check_connection()
            print("LSI Connection successful.")
            return True
        except Exception:
            print("LSI Connection unsuccessful.")
            return False

    def _update_usernames(self, ls_client: Client | None = None) -> None:
        if ls_client is None:
            ls_client = Client(url=self.url, api_key=self.token)

        users = ls_client.get_users()
        user_ids = [user.id for user in users]
        ordered_usernames: list[str] = []
        if user_ids:
            for x in range(max(user_ids) + 1):
                matching = [u.username for u in users if u.id == x]
                if matching:
                    ordered_usernames.append(matching[0])
                else:
                    ordered_usernames.append("Impossible LS user")
        self._usernames.update_usernames(ordered_usernames)

    def fetch_simplified_tasks(self) -> list[SimplifiedTask]:
        """Pulls task data from Label Studio and simplifies."""
        if not self.online:
            print(f"LSI configured with online={self.online}; skipping remote fetch.")
            return []

        ls_client = Client(url=self.url, api_key=self.token)
        project = ls_client.get_project(id=self.project_id)

        self._update_usernames()
        print("Downloading and simplifying tasks from Label Studio...")
        raw_tasks_data = project.export_tasks()
        raw_tasks_data.sort(
            key=lambda t: t.get("id", 0) if isinstance(t, dict) else t.id
        )

        return simplify_tasks(raw_tasks_data)

    def users(self) -> _LSUsersManager:
        return self._usernames

    def parts_managed(self):
        return {"annotations"}

    def parts_required(self):
        return {"raw_images", "background_images", "stroke_images"}

    def setup(self, paths: PathBundle) -> None:
        """Fetch Label Studio annotations and write canonical PAGE-XML."""
        if not self.online:
            print(
                f"LSI configured with online={self.online}; "
                "keeping local generated data."
            )
            return

        tasks = self.fetch_simplified_tasks()

        for task in tasks:
            image_url = task.data.image_url
            task_id = task.id
            page = Path(url_unquote(image_url)).stem
            raw_image_path = paths.get_raw_image_path(page)
            stroke_image_path = paths.get_stroke_image_path(page)
            background_image_path = paths.get_background_image_path(page)
            stroke = paths.load_stroke_image(page)
            background = paths.load_background_image(page)
            if stroke.shape != background.shape:
                raise ValueError(
                    f"Stroke and background images for page {page!r} must have "
                    "identical dimensions."
                )
            height, width = stroke.shape[:2]

            for subindex, simplified_ann in enumerate(task.annotations):
                transcriptions: list[str] = []
                poly_coords: list[list[tuple[float, float]]] = []
                ids: list[str] = []
                rotations: list[float] = []

                completer: str = self._usernames[simplified_ann.completed_by]
                updater: str = self._usernames[simplified_ann.updated_by]
                ann_id = simplified_ann.id

                results = simplified_ann.result
                box2text, id2boxres, id2txtres = pair_lines(results)

                trios = [
                    (
                        id2boxres[key],
                        id2txtres[box2text[key]],
                        f"{key}-{box2text[key]}",
                    )
                    for key in id2boxres
                    if key in box2text and box2text[key] in id2txtres
                ]

                for trio in trios:
                    box_result = trio[0]
                    txt_result = trio[1]
                    transcription = txt_result.value.text
                    assert len(transcription) == 1
                    transcriptions.append(transcription[0])

                    percentage_bounds = extract_bounds(box_result)
                    assert len(percentage_bounds) > 3
                    pixel_bounds = [
                        (
                            min(max(float(x), 0.0), 100.0) * width / 100.0,
                            min(max(float(y), 0.0), 100.0) * height / 100.0,
                        )
                        for x, y in percentage_bounds
                    ]
                    polygon = Polygon(pixel_bounds)
                    if not polygon.is_valid:
                        repaired = make_valid(polygon)
                        polygon = (
                            repaired
                            if isinstance(repaired, Polygon)
                            else repaired.convex_hull
                        )
                    if polygon.is_empty or not isinstance(polygon, Polygon):
                        raise ValueError(
                            f"Label Studio region {trio[2]!r} has no usable polygon."
                        )
                    poly_coords.append(list(polygon.exterior.coords)[:-1])

                    ids.append(trio[2])

                    if isinstance(box_result, RectangleResult):
                        rotations.append(box_result.value.rotation)
                    else:
                        rotations.append(calculate_reading_angle(polygon))

                metadata = {
                    LABEL_STUDIO_TASK_ID: task_id,
                    LABEL_STUDIO_COMPLETER: completer,
                    LABEL_STUDIO_UPDATER: updater,
                    LABEL_STUDIO_ANNOTATION_ID: ann_id,
                    LABEL_STUDIO_PROJECT: simplified_ann.project,
                    LABEL_STUDIO_UNIQUE_ID: simplified_ann.unique_id,
                    LABEL_STUDIO_GROUND_TRUTH: simplified_ann.ground_truth,
                    LABEL_STUDIO_CREATED_AT: simplified_ann.created_at,
                    LABEL_STUDIO_UPDATED_AT: simplified_ann.updated_at,
                    LABEL_STUDIO_LEAD_TIME: simplified_ann.lead_time,
                    SYNTHSCRIPT_SUBINDEX: subindex,
                    SYNTHSCRIPT_SOURCE: "Label Studio",
                }
                ocr_page = OCRPage(
                    transcriptions=transcriptions,
                    polygon_coords=poly_coords,
                    line_ids=ids,
                    rotations=rotations,
                    page_id=page,
                    stroke=stroke,
                    background=background,
                    metadata={key: str(value) for key, value in metadata.items()},
                )
                save_page_xml(
                    ocr_page,
                    paths.get_page_xml_path(page, subindex),
                    raw_image_path=raw_image_path,
                    stroke_image_path=stroke_image_path,
                    background_image_path=background_image_path,
                    metadata=metadata,
                )
