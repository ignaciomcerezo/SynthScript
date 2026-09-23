from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from Levenshtein import distance as levenshtein_distance

from synthscript.metric.ast.node import (
    CanonicalNode,
    MetricNode,
    MetricNodeKind,
    NodeKind,
)

Node = CanonicalNode | MetricNode
UnaryCost = Callable[[MetricNode], float]
BinaryCost = Callable[[MetricNode, MetricNode], float]


@dataclass(frozen=True, slots=True)
class MetricParameters:
    """
    Weights and edit costs used by the metric, and the metric implementations
    themselves
    """

    root_weight: float = 0
    character_weight: float = 1
    comment_weight: float = 1
    comment_character_weight: float = 1
    math_weight: float = 0.9  # prev. 1
    symbol_weight: float = 1
    macro_weight: float = 1
    group_weight: float = 0.4  # prev. 0.5
    environment_weight: float = 1
    arguments_weight: float = 0
    fraction_weight: float = 1.9  # prev. 2
    sqrt_weight: float = 1.9  # prev 2
    superscript_weight: float = 0.9  # prev 1.
    subscript_weight: float = 0.9  # prev. 1
    subsup_weight: float = 2
    style_weight: float = 0.9  # prev 1.

    mass_factor: float = 0.9  # prev 1.

    subscript_superscript_cost: float = 1.1  # prev 1.
    script_subsup_cost: float = 0.5
    style_macro_cost: float = 1
    macro_substitution_cost: float = 1
    symbol_substitution_cost: float = 1
    style_substitution_cost: float = 1.5
    environment_substitution_cost: float = 1.5

    def weight(self, node: Node) -> float:
        """Return the importance of one node, excluding its children."""
        match node.kind:
            case NodeKind.ROOT | MetricNodeKind.ROOT:
                return self.root_weight
            case NodeKind.TEXT:
                return self.character_weight * len(node.value or "")
            case MetricNodeKind.CHAR:
                return self.character_weight
            case NodeKind.COMMENT | MetricNodeKind.COMMENT:
                return self.comment_weight + self.comment_character_weight * len(
                    node.value or ""
                )
            case NodeKind.MATH | MetricNodeKind.MATH:
                return self.math_weight
            case NodeKind.SYMBOL | MetricNodeKind.SYMBOL:
                return self.symbol_weight
            case NodeKind.MACRO | MetricNodeKind.MACRO:
                return self.macro_weight
            case NodeKind.GROUP | MetricNodeKind.GROUP:
                return self.group_weight
            case NodeKind.ENVIRONMENT | MetricNodeKind.ENVIRONMENT:
                return self.environment_weight
            case NodeKind.ARGUMENTS | MetricNodeKind.ARGUMENTS:
                return self.arguments_weight
            case NodeKind.FRACTION | MetricNodeKind.FRACTION:
                return self.fraction_weight
            case NodeKind.SQRT | MetricNodeKind.SQRT:
                return self.sqrt_weight
            case NodeKind.SUPERSCRIPT | MetricNodeKind.SUPERSCRIPT:
                return self.superscript_weight
            case NodeKind.SUBSCRIPT | MetricNodeKind.SUBSCRIPT:
                return self.subscript_weight
            case NodeKind.SUBSUP | MetricNodeKind.SUBSUP:
                return self.subsup_weight
            case NodeKind.STYLE | MetricNodeKind.STYLE:
                return self.style_weight
            case _:
                raise ValueError(f"Unsupported node kind: {node.kind}")

    def mass(self, node: Node) -> float:
        """Return the total weight of a subtree."""
        return self.weight(node) + sum(self.mass(child) for child in node.children)

    def unit_cost(self, node: MetricNode) -> float:
        return 1

    def mass_cost(self, node: MetricNode) -> float:
        return self.mass_factor * self.mass(node)

    @staticmethod
    def are_compatible(source: MetricNode, target: MetricNode) -> bool:
        if source.kind == target.kind:
            return True
        compatible_pairs = {
            frozenset((MetricNodeKind.SUBSCRIPT, MetricNodeKind.SUPERSCRIPT)),
            frozenset((MetricNodeKind.SUBSCRIPT, MetricNodeKind.SUBSUP)),
            frozenset((MetricNodeKind.SUPERSCRIPT, MetricNodeKind.SUBSUP)),
            frozenset((MetricNodeKind.MACRO, MetricNodeKind.STYLE)),
        }
        return frozenset((source.kind, target.kind)) in compatible_pairs

    @staticmethod
    def levenshtein_cost(
        source: MetricNode,
        target: MetricNode,
        *,
        strict_text: bool = False,
    ) -> float:
        if strict_text and not (source.kind == target.kind == MetricNodeKind.CHAR):
            raise ValueError("Cannot compare non-character nodes with strict_text=True")
        return float(levenshtein_distance(source.value or "", target.value or ""))

    def maximum_substitution_cost(
        self, source: MetricNode, target: MetricNode
    ) -> float:
        return self.mass(source) + self.mass(target)

    def compatible_substitution_cost(
        self,
        source: MetricNode,
        target: MetricNode,
        *,
        deletion_cost: UnaryCost | None = None,
        insertion_cost: UnaryCost | None = None,
        equal_cost: BinaryCost | None = None,
    ) -> float:
        deletion = deletion_cost or self.mass_cost
        insertion = insertion_cost or self.mass_cost
        equal = equal_cost or self.levenshtein_cost

        if source.kind == target.kind and source.value == target.value:
            return 0.0
        if source.kind == target.kind:
            basic_costs = {
                MetricNodeKind.MACRO: self.macro_substitution_cost,
                MetricNodeKind.SYMBOL: self.symbol_substitution_cost,
                MetricNodeKind.STYLE: self.style_substitution_cost,
                MetricNodeKind.ENVIRONMENT: self.environment_substitution_cost,
            }
            if source.kind in basic_costs:
                return basic_costs[source.kind]
            if source.kind == MetricNodeKind.COMMENT:
                return equal(source, target)
            return deletion(source) + insertion(target)

        pair = frozenset((source.kind, target.kind))
        if pair == {MetricNodeKind.SUBSCRIPT, MetricNodeKind.SUPERSCRIPT}:
            return self.subscript_superscript_cost
        if pair in {
            frozenset((MetricNodeKind.SUBSCRIPT, MetricNodeKind.SUBSUP)),
            frozenset((MetricNodeKind.SUPERSCRIPT, MetricNodeKind.SUBSUP)),
        }:
            return self.script_subsup_cost
        if pair == {MetricNodeKind.STYLE, MetricNodeKind.MACRO}:
            return self.style_macro_cost

        structural = {
            MetricNodeKind.FRACTION,
            MetricNodeKind.SQRT,
            MetricNodeKind.SUBSCRIPT,
            MetricNodeKind.SUPERSCRIPT,
            MetricNodeKind.SUBSUP,
        }
        if source.kind in structural and target.kind in structural:
            return self.weight(source) + self.weight(target)
        return deletion(source) + insertion(target)

    def substitution_cost(
        self,
        source: MetricNode,
        target: MetricNode,
        *,
        deletion_cost: UnaryCost | None = None,
        insertion_cost: UnaryCost | None = None,
    ) -> float:
        deletion = deletion_cost or self.mass_cost
        insertion = insertion_cost or self.mass_cost

        if source.kind == target.kind and source.value == target.value:
            return 0.0
        if source.kind == target.kind == MetricNodeKind.CHAR:
            return self.levenshtein_cost(source, target)
        if self.are_compatible(source, target):
            return self.compatible_substitution_cost(
                source,
                target,
                deletion_cost=deletion,
                insertion_cost=insertion,
            )
        return deletion(source) + insertion(target)


DEFAULT_METRIC_PARAMETERS = MetricParameters()
