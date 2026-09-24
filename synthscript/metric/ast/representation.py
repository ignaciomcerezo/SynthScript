import ast

from synthscript.metric.ast.node import CanonicalNode, NodeKind

FlatToken = tuple[str, str | None, int]


def _label(node: CanonicalNode) -> str:
    """Format one node as a single diagram line."""
    if node.value is None:
        return node.kind.value
    return f"{node.kind.value}: {node.value!r}"


def render_vertical(node: CanonicalNode, *, ascii_only: bool = False) -> str:
    """Draw a vertical tree in canonical child order.

    Use ``ascii_only`` for terminals without box drawing character support.
    """
    lines = [_label(node)]
    stack = [
        (child, "", index == len(node.children) - 1)
        for index, child in reversed(list(enumerate(node.children)))
    ]

    while stack:
        current, prefix, is_last = stack.pop()
        branch = (
            ("`-- " if is_last else "|-- ")
            if ascii_only
            else ("└── " if is_last else "├── ")
        )
        lines.append(f"{prefix}{branch}{_label(current)}")
        child_prefix = prefix + (
            "    " if is_last else ("|   " if ascii_only else "│   ")
        )
        stack.extend(
            (child, child_prefix, index == len(current.children) - 1)
            for index, child in reversed(list(enumerate(current.children)))
        )

    return "\n".join(lines)


def flatten(node: CanonicalNode) -> tuple[FlatToken, ...]:
    tokens: list[FlatToken] = []
    stack = [node]
    while stack:
        current = stack.pop()
        tokens.append((current.kind.value, current.value, len(current.children)))
        stack.extend(reversed(current.children))
    return tuple(tokens)


def _delimiters_latex_flatten(
    value: str | None, default: tuple[str, str]
) -> tuple[str, str]:
    if value is None:
        return default
    pair = ast.literal_eval(value)
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError(f"Invalid delimiter pair: {value!r}")
    return pair


def _argument_latex_flatten(child: CanonicalNode) -> str:
    if child.kind == NodeKind.GROUP:
        return flatten_latex(child)
    return "{" + flatten_latex(child) + "}"


def _sequence_latex_flatten(
    children: tuple[CanonicalNode, ...], *, math: bool = False
) -> str:
    parts: list[str] = []
    previous: CanonicalNode | None = None
    for child in children:
        part = flatten_latex(child)
        if (
            previous is not None
            and previous.kind in {NodeKind.MACRO, NodeKind.STYLE}
            and not previous.children
            and previous.value
            and previous.value[-1].isalpha()
            and part[:1].isalpha()
        ):
            # TeX would otherwise read the following letters as part of the macro.
            parts.append(" ")
        parts.append(part)
        previous = child
    return "".join(parts)


def flatten_latex(current: CanonicalNode) -> str:
    """
    Flattens the AST into renderable latex
    """
    kind = current.kind
    children = current.children
    if kind in (NodeKind.ROOT, NodeKind.ARGUMENTS):
        return _sequence_latex_flatten(children)
    if kind in (NodeKind.TEXT, NodeKind.COMMENT):
        return current.value or ""
    if kind == NodeKind.MATH:
        left, right = _delimiters_latex_flatten(current.value, ("$", "$"))
        return left + _sequence_latex_flatten(children, math=True) + right
    if kind == NodeKind.GROUP:
        left, right = _delimiters_latex_flatten(current.value, ("{", "}"))
        return left + _sequence_latex_flatten(children) + right
    if kind == NodeKind.SYMBOL:
        return (current.value or "") + "".join(
            _argument_latex_flatten(c) for c in children
        )
    if kind in {NodeKind.MACRO, NodeKind.STYLE}:
        return (
            "\\"
            + (current.value or "")
            + "".join(
                (
                    flatten_latex(c)
                    if c.kind == NodeKind.GROUP
                    else _argument_latex_flatten(c)
                )
                for c in children
            )
        )
    if kind == NodeKind.ENVIRONMENT:
        if not current.value:
            raise ValueError("Environment node requires a name")
        args = (
            children[0].children
            if children and children[0].kind == NodeKind.ARGUMENTS
            else ()
        )
        body = (
            children[1:]
            if args or (children and children[0].kind == NodeKind.ARGUMENTS)
            else children
        )
        return (
            "\\begin{"
            + current.value
            + "}"
            + "".join(
                (
                    flatten_latex(c)
                    if c.kind == NodeKind.GROUP
                    else _argument_latex_flatten(c)
                )
                for c in args
            )
            + _sequence_latex_flatten(body)
            + "\\end{"
            + current.value
            + "}"
        )
    if kind == NodeKind.FRACTION:
        if len(children) != 2:
            raise ValueError("Fraction node requires two children")
        return "\\frac" + "".join(_argument_latex_flatten(c) for c in children)
    if kind == NodeKind.SQRT:
        if len(children) not in (1, 2):
            raise ValueError("Square root node requires one or two children")
        index = "[" + flatten_latex(children[1]) + "]" if len(children) == 2 else ""
        return "\\sqrt" + index + _argument_latex_flatten(children[0])
    if kind in (NodeKind.SUBSCRIPT, NodeKind.SUPERSCRIPT, NodeKind.SUBSUP):
        expected = 3 if kind == NodeKind.SUBSUP else 2
        if len(children) != expected:
            raise ValueError(f"{kind.value} node requires {expected} children")
        base = flatten_latex(children[0])
        if kind == NodeKind.SUBSUP:
            return (
                base
                + "_"
                + _argument_latex_flatten(children[1])
                + "^"
                + _argument_latex_flatten(children[2])
            )
        marker = "_" if kind == NodeKind.SUBSCRIPT else "^"
        return base + marker + _argument_latex_flatten(children[1])

    raise ValueError(f"Unsupported node kind: {kind!r}")
