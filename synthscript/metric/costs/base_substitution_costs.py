from Levenshtein import distance as levenshtein_distance

from synthscript.metric.ast.node import MetricNode, MetricNodeKind
from synthscript.metric.costs.mass import mass


def levenshtein_cost(
    a: MetricNode,
    b: MetricNode,
    *,
    strict_text: bool = False,
) -> float:
    if strict_text and not (a.kind == b.kind == MetricNodeKind.CHAR):
        raise ValueError(
            "Cannot compare two non-text nodes using lehvenstein when strict_text=True"
        )
    if a.value is None:
        return 0 if b.value is None else len(b.value)

    elif b.value is None:
        return 0 if a.value is None else len(a.value)

    else:
        return levenshtein_distance(a.value, b.value)


def max_cost(x: MetricNode, y: MetricNode):
    return mass(x) + mass(y)
