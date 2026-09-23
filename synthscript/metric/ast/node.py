from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeKind(Enum):
    ROOT = "root"

    TEXT = "text"
    COMMENT = "comment"
    MATH = "math"
    SYMBOL = "symbol"

    MACRO = "macro"
    GROUP = "group"
    ENVIRONMENT = "environment"
    ARGUMENTS = "arguments"

    FRACTION = "fraction"
    SQRT = "sqrt"

    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"
    SUBSUP = "subsup"

    STYLE = "style"


class MetricNodeKind(Enum):
    """Node kinds used by the tree-edit metric.

    The metric tree mirrors the canonical tree except that canonical text
    nodes are replaced by char nodes (one per character).
    """

    ROOT = "root"

    CHAR = "char"
    COMMENT = "comment"
    MATH = "math"
    SYMBOL = "symbol"

    MACRO = "macro"
    GROUP = "group"
    ENVIRONMENT = "environment"
    ARGUMENTS = "arguments"

    FRACTION = "fraction"
    SQRT = "sqrt"

    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"
    SUBSUP = "subsup"

    STYLE = "style"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CanonicalNode:
    kind: NodeKind
    value: str | None = None
    children: tuple[CanonicalNode, ...] = ()

    span: SourceSpan | None = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class MetricNode:
    kind: MetricNodeKind
    value: str | None = None
    children: tuple[MetricNode, ...] = ()


def _to_metric_forest(node: CanonicalNode) -> tuple[MetricNode, ...]:
    if node.kind == NodeKind.TEXT:
        return tuple(
            MetricNode(kind=MetricNodeKind.CHAR, value=char)
            for char in node.value or ""
        )

    children = tuple(
        metric_child
        for child in node.children
        for metric_child in _to_metric_forest(child)
    )

    return (
        MetricNode(
            kind=MetricNodeKind(node.kind.value),
            value=node.value,
            children=children,
        ),
    )


def to_metric_tree(node: CanonicalNode) -> MetricNode:
    """Convert a canonical subtree into one rooted metric tree.

    Text nodes expand into sibling character nodes. A synthetic root contains
    those siblings when ``node`` is not already a canonical root.
    """
    if node.kind == NodeKind.ROOT:
        children = tuple(
            metric_child
            for child in node.children
            for metric_child in _to_metric_forest(child)
        )
        return MetricNode(
            kind=MetricNodeKind.ROOT,
            value=node.value,
            children=children,
        )

    return MetricNode(
        kind=MetricNodeKind.ROOT,
        children=_to_metric_forest(node),
    )
