from __future__ import annotations

from collections.abc import Callable

from zss import distance as zss_distance

from synthscript.metric.ast.node import CanonicalNode, MetricNode, to_metric_tree


def tree_edit_distance(
    source: CanonicalNode,
    target: CanonicalNode,
    *,
    insertion_cost: Callable[[MetricNode], float],
    deletion_cost: Callable[[MetricNode], float],
    substitution_cost: Callable[[MetricNode, MetricNode], float],
) -> float:
    """
    Compute a weighted ordered tree edit distance.

    - insertion_cost(x): cost of inserting node x
    - deletion_cost(x): cost of deleting node x
    - substitution_cost(x, y): cost of replacing node x with node y

    The underlying dynamic program is the Zhang-Shasha ordered tree edit
    distance algorithm.
    """
    metric_source = to_metric_tree(source)
    metric_target = to_metric_tree(target)

    return float(
        zss_distance(
            metric_source,
            metric_target,
            get_children=lambda node: node.children,
            insert_cost=insertion_cost,
            remove_cost=deletion_cost,
            update_cost=substitution_cost,
        )
    )
