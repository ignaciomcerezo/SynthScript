from synthscript.metric.ast.node import MetricNode
from synthscript.metric.costs.mass import mass


def unit_deletion_cost(x: MetricNode) -> float:
    return 1


def mass_deletion_cost(x: MetricNode) -> float:
    return mass(x)
