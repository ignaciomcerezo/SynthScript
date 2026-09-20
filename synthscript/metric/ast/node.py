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
    )  # TODO: this will only be used for testing, can be removed for training
