from collections.abc import Callable

from Levenshtein import distance as levenshtein_distance

from synthscript.metric.ast.node import MetricNode, MetricNodeKind
from synthscript.metric.costs.base_substitution_costs import levenshtein_cost
from synthscript.metric.costs.deletion_cost import (
    mass_deletion_cost,
)
from synthscript.metric.costs.insertion_cost import mass_insertion_cost
from synthscript.metric.costs.mass import weight


def are_compatible(a: MetricNode, b: MetricNode) -> bool:
    if a.kind == b.kind:
        return True

    compatible_pairs = {
        (MetricNodeKind.SUBSCRIPT, MetricNodeKind.SUPERSCRIPT),
        (MetricNodeKind.SUBSCRIPT, MetricNodeKind.SUBSUP),
        (MetricNodeKind.SUPERSCRIPT, MetricNodeKind.SUBSUP),
        # Perhaps:
        (MetricNodeKind.MACRO, MetricNodeKind.STYLE),
    }

    return (a.kind, b.kind) in compatible_pairs or (b.kind, a.kind) in compatible_pairs


def _compatible_substitution_cost(
    a: MetricNode,
    b: MetricNode,
    *,
    deletion_cost: Callable[[MetricNode], float] = mass_deletion_cost,
    insertion_cost: Callable[[MetricNode], float] = mass_insertion_cost,
    equal_cost: Callable[[MetricNode, MetricNode], float] = levenshtein_cost,
) -> float:
    # Exact same node label
    if a.kind == b.kind and a.value == b.value:
        return 0.0

    if a.kind == b.kind:

        return equal_cost(a, b)

    pair = frozenset((a.kind, b.kind))

    if pair == {MetricNodeKind.SUBSCRIPT, MetricNodeKind.SUPERSCRIPT}:
        return 1.0

    if pair in {
        frozenset((MetricNodeKind.SUBSCRIPT, MetricNodeKind.SUBSUP)),
        frozenset((MetricNodeKind.SUPERSCRIPT, MetricNodeKind.SUBSUP)),
    }:
        return 0.5

    if pair == {MetricNodeKind.STYLE, MetricNodeKind.MACRO}:
        return 1.0

    structural = {
        MetricNodeKind.FRACTION,
        MetricNodeKind.SQRT,
        MetricNodeKind.SUBSCRIPT,
        MetricNodeKind.SUPERSCRIPT,
        MetricNodeKind.SUBSUP,
    }

    if a.kind in structural and b.kind in structural:
        return weight(a) + weight(b)

    return deletion_cost(a) + insertion_cost(b)


def composite_substitution_cost(
    source: MetricNode,
    target: MetricNode,
    *,
    deletion_cost: Callable[[MetricNode], float] = mass_deletion_cost,
    insertion_cost: Callable[[MetricNode], float] = mass_insertion_cost,
) -> float:
    if source.kind == target.kind and source.value == target.value:
        return 0.0

    if source.kind == target.kind == MetricNodeKind.CHAR:
        return levenshtein_distance(
            source.value or "",
            target.value or "",
        )

    if are_compatible(source, target):
        return _compatible_substitution_cost(
            source, target, deletion_cost=deletion_cost, insertion_cost=insertion_cost
        )

    return deletion_cost(source) + insertion_cost(target)
