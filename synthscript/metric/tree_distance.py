from __future__ import annotations

from collections.abc import Callable

from zss import distance as zss_distance

from synthscript.metric.ast.node import CanonicalNode, MetricNode, to_metric_tree
from synthscript.metric.costs.parameters import (
    DEFAULT_METRIC_PARAMETERS,
    MetricParameters,
)


def tree_edit_distance(
    source: CanonicalNode | MetricNode,
    target: CanonicalNode | MetricNode,
    *,
    parameters: MetricParameters | None = None,
    insertion_cost: Callable[[MetricNode], float] | None = None,
    deletion_cost: Callable[[MetricNode], float] | None = None,
    substitution_cost: Callable[[MetricNode, MetricNode], float] | None = None,
) -> float:
    """
    Compute a weighted ordered tree edit distance.

    parameters supplies all costs, where individual arguments can
    override them behavior.

    The underlying dynamic program is the Zhang-Shasha ordered tree edit
    distance algorithm.
    """
    metric_parameters = parameters or DEFAULT_METRIC_PARAMETERS
    insertion = insertion_cost or metric_parameters.mass_cost
    deletion = deletion_cost or metric_parameters.mass_cost
    substitution = substitution_cost or metric_parameters.substitution_cost

    metric_source = (
        to_metric_tree(source) if isinstance(source, CanonicalNode) else source
    )
    metric_target = (
        to_metric_tree(target) if isinstance(target, CanonicalNode) else target
    )

    return float(
        zss_distance(
            metric_source,
            metric_target,
            get_children=lambda node: node.children,
            insert_cost=insertion,
            remove_cost=deletion,
            update_cost=substitution,
        )
    )
