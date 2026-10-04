from abc import ABC, abstractmethod
from collections.abc import Callable

from synthscript.metric.ast.node import CanonicalNode
from synthscript.metric.ast.parser import LatexParser
from synthscript.metric.ast.representation import flatten_latex
from synthscript.metric.character_distances import CER
from synthscript.metric.node_metrics import ConfiguredNodeMetrics
from synthscript.metric.tree_distance import tree_edit_distance


class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(self, reference: str, prediction: str) -> dict[str, float]:
        raise NotImplementedError


class OCREvaluator(BaseEvaluator):
    def __init__(self, *metrics: Callable[[str, str], float]):
        self.metrics = metrics

    def evaluate(self, reference: str, prediction: str) -> dict[str, float]:
        return {str(metric): metric(reference, prediction) for metric in self.metrics}


class StandardEvaluator(BaseEvaluator):
    def __init__(
        self,
        parser: LatexParser | None = None,
        configured_node_metrics: ConfiguredNodeMetrics | None = None,
    ):
        self.parser = parser if parser is not None else LatexParser()

        self.configured_metrics = (
            ConfiguredNodeMetrics()
            if configured_node_metrics is None
            else configured_node_metrics
        )

    def ted(self, ref_tree: CanonicalNode, pred_tree: CanonicalNode) -> float:
        return tree_edit_distance(
            ref_tree, pred_tree, parameters=self.configured_metrics
        )

    @staticmethod
    def canonical_cer(ref_tree: CanonicalNode, pred_tree: CanonicalNode) -> float:
        return CER(flatten_latex(ref_tree), flatten_latex(pred_tree))

    def evaluate(self, reference: str, prediction: str) -> dict[str, float]:
        ref_tree = self.parser.parse(reference)
        pred_tree = self.parser.parse(prediction)

        return {
            "TED": self.ted(ref_tree, pred_tree),
            "CERc": self.canonical_cer(ref_tree, pred_tree),
            "CER": CER(reference, prediction),
            "RefMass": self.configured_metrics.mass(ref_tree),
            "PredMass": self.configured_metrics.mass(pred_tree),
            "RefMath%": self.configured_metrics.math_percentage(ref_tree),
            "PredMath%": self.configured_metrics.math_percentage(pred_tree),
        }
