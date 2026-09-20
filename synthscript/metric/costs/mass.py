from synthscript.metric.ast.node import MetricNode, MetricNodeKind


def mass(node: MetricNode):
    """
    Importance of a subtree rooted at node.
    """
    return weight(node) + sum(mass(child) for child in node.children)


def weight(node: MetricNode) -> float | int:
    """
    Importance of a single node (not its children).
    """
    match node.kind:
        case MetricNodeKind.ROOT:
            return 0

        case MetricNodeKind.CHAR:
            return 1

        case MetricNodeKind.COMMENT:
            return 1 + len(node.value) if node.value is not None else 1

        case MetricNodeKind.MATH:
            return 1

        case MetricNodeKind.MACRO:
            return 1  # pending revision

        case MetricNodeKind.GROUP:
            return 0.5

        case MetricNodeKind.ENVIRONMENT:
            return 1

        case MetricNodeKind.ARGUMENTS:
            return 0

        case MetricNodeKind.FRACTION:
            return 2

        case MetricNodeKind.SQRT:
            return 2

        case MetricNodeKind.SUPERSCRIPT:
            return 1

        case MetricNodeKind.SUBSCRIPT:
            return 1

        case MetricNodeKind.SUBSUP:
            return 2

        case MetricNodeKind.STYLE:
            return 1

        case MetricNodeKind.SYMBOL:
            return 1

        case _:

            raise ValueError(f"Unsupported nodekind: {node.kind}")
