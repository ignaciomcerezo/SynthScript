from collections.abc import Iterator

import numpy as np

from synthscript.ocr_units.ocr_line import OCRLine
from synthscript.shared.graph_utilities import (
    is_path_graph,
)


class OCRParagraph:
    __slots__ = (
        "_index",
        "avg_rotation",
        "centroid",
        "line_ids",
        "lines",
        "task_id",
        "total_words",
    )

    def __init__(
        self,
        *,
        lines: list[OCRLine],
        task_id: int,
        subgraph: dict[str, set[str]],
        index: int | None = None,
    ):

        if not lines:
            raise ValueError("Lines cannot be empty.")

        if len(subgraph) != len(lines):
            raise ValueError(
                "The length of the subgraph passed to an OCRParagraph must be equal to"
                " the number of lines it contains."
            )

        self.lines = lines
        self.task_id: int | None = task_id
        self._index: int | None = index

        self._set_geometric_and_topological_properties(subgraph)

        self._sort_lines_using_centroid_and_subgraph(subgraph)

        self.line_ids = [line.id for line in self.lines]

        if set(subgraph.keys()) != set(self.line_ids):
            raise ValueError("The subgraph keys does not match the lines passed.")

        for line in self.lines:
            if line.paragraph_index is not None:
                raise ValueError("Line with paragraph already set.")
            line.paragraph_index = index

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value: int):
        self._index = value
        for line in self.lines:
            line.paragraph_index = value

    def __iter__(self) -> Iterator[OCRLine]:
        for x in self.lines:
            yield x

    def __getitem__(self, index) -> OCRLine:
        return self.lines[index]

    def __lt__(self, other: "OCRParagraph"):
        return (self.top, self.left) < (other.top, other.left)

    def __gt__(self, other: "OCRParagraph"):
        return (self.top, self.left) > (other.top, other.left)

    def transcription(self, separator: str = " "):
        return separator.join([fragment.text for fragment in self.lines])

    def __len__(self):
        return len(self.line_ids)

    def __repr__(self):
        return f"<{self.index}-th paragraph of order {len(self)} contained in AnnotatedPage of task ({self.task_id})>"

    @property
    def top(self) -> float:
        return min(line.top for line in self.lines)

    @property
    def left(self) -> float:
        return min(line.left for line in self.lines)

    @property
    def right(self) -> float:
        return max(line.right for line in self.lines)

    @property
    def bot(self) -> float:
        return max(line.bot for line in self.lines)

    def _set_geometric_and_topological_properties(
        self, subgraph: dict[str, set[str]]
    ) -> None:
        self.centroid: np.ndarray = np.zeros((2,))
        self.total_words: int = 0
        total_area = 0

        areas = [line.polygon.area for line in self.lines]

        for line, area in zip(self.lines, areas, strict=True):
            self.total_words += len(line.text.split())

            self.centroid += np.array(line.centroid()) * area
            total_area += area

        assert self.total_words > 0, "Se ha pasado un párrafo sin palabras."

        self.centroid /= total_area

        rotations = [line.rotation for line in self.lines]
        angles_in_radians = np.radians(rotations)
        sum_sin = np.sum(np.sin(angles_in_radians) * np.array(areas))
        sum_cos = np.sum(np.cos(angles_in_radians) * np.array(areas))
        self.avg_rotation = -float(np.degrees(np.arctan2(sum_sin, sum_cos)))

    def _sort_lines_using_centroid_and_subgraph(
        self,
        subgraph: dict[str, set[str]],
    ) -> None:

        theta_rad = -np.radians(-self.avg_rotation)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        cx_para, cy_para = self.centroid

        for line in self.lines:
            cx, cy = line.centroid()

            dx = cx - cx_para
            dy = cy - cy_para

            corrected_x = dx * cos_theta - dy * sin_theta + cx_para
            corrected_y = dx * sin_theta + dy * cos_theta + cy_para

            line.corrected_centroid = (
                corrected_x,
                corrected_y,
            )

        if not is_path_graph(
            subgraph
        ):  # if it is not a path graph, we use the reading order given by the projections
            self.lines = sorted(
                self.lines,
                key=lambda line: (
                    line.corrected_centroid[1],
                    line.corrected_centroid[0],
                ),
            )
            return

        if len(self.lines) == 1:
            return

        terminal_vertices = [line for line in self.lines if len(subgraph[line.id]) == 1]
        assert len(terminal_vertices) == 2

        top_line = min(
            terminal_vertices,
            key=lambda line: (
                line.corrected_centroid[1],
                line.corrected_centroid[0],
            ),
        )

        lines_by_id = {line.id: line for line in self.lines}
        ordered_lines = [top_line]
        visited = {top_line.id}
        previous_id: str | None = None
        current_id = top_line.id

        while len(ordered_lines) < len(self.lines):
            next_candidates = [
                neighbor_id
                for neighbor_id in subgraph[current_id]
                if neighbor_id != previous_id and neighbor_id not in visited
            ]
            assert len(next_candidates) == 1

            next_id = next_candidates[0]
            ordered_lines.append(lines_by_id[next_id])
            visited.add(next_id)
            previous_id, current_id = current_id, next_id

        self.lines = ordered_lines
