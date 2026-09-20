import re
from collections.abc import Collection
from dataclasses import dataclass, field

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexNode,
    LatexSpecialsNode,
)

from synthscript.metric.ast.macro_groups import (
    FRACTION_MACROS,
    GROUP_SPLITTING_MACROS,
    MACRO_ALIASES,
    STYLE_DECLARATIONS,
    STYLE_MACROS,
    TRANSPARENT_MACROS,
)
from synthscript.metric.ast.node import CanonicalNode, NodeKind, SourceSpan


@dataclass(frozen=True)
class CanonicalizationConfig:
    """
    Ignored LaTeX syntax differences
    """

    normalize_whitespace: bool = True
    remove_comments: bool = True
    normalize_math_delimiters: bool = True
    normalize_fraction_style: bool = True
    macro_aliases: dict[str, str] = field(default_factory=lambda: dict(MACRO_ALIASES))


class LatexCanonicalizer:
    def __init__(self, config: CanonicalizationConfig | None = None) -> None:
        self.config = config if config is not None else CanonicalizationConfig()

    def canonicalize(self, nodes: Collection[LatexNode]) -> CanonicalNode:
        """Convert a sequence of pylatexenc nodes into a rooted canonical tree"""
        return CanonicalNode(
            kind=NodeKind.ROOT,
            children=tuple(self._visit_many(nodes)),
        )

    def _visit_many(self, nodes: Collection[LatexNode]) -> list[CanonicalNode]:
        """Convert siblings, join adjacent text and pair neighboring super/subscripts"""
        result: list[CanonicalNode] = []
        for node in nodes:
            converted = self._visit(node)
            if converted is None:
                continue
            if isinstance(converted, list):
                result.extend(converted)
            else:
                result.append(converted)

        result = self._combine_scripts(self._merge_text_nodes(result))
        result = self._normalize_style_declarations(result)
        return self._normalize_group_splitting(result)

    def _visit(self, node: LatexNode) -> CanonicalNode | list[CanonicalNode] | None:
        """Processes one pylatexenc node into its canonical representation."""
        if isinstance(node, LatexCommentNode):
            if self.config.remove_comments:
                return None
            return CanonicalNode(
                kind=NodeKind.COMMENT,
                value=f"%{node.comment}{node.comment_post_space}",
                span=self._span(node),
            )
        if isinstance(node, LatexCharsNode):
            return self._chars(node)
        if isinstance(node, LatexGroupNode):
            return self._group(node)
        if isinstance(node, LatexMacroNode):
            return self._macro(node)
        if isinstance(node, LatexMathNode):
            return self._math(node)
        if isinstance(node, LatexEnvironmentNode):
            return self._environment(node)
        if isinstance(node, LatexSpecialsNode):
            return self._special(node)
        raise TypeError(f"Unsupported pylatexenc node: {type(node)}")

    @staticmethod
    def _span(node: LatexNode) -> SourceSpan:
        """Return the span covered by a pylatexenc node."""
        return SourceSpan(
            node.pos,  # ty: ignore[invalid-argument-type]
            node.pos + node.len,  # ty: ignore[unsupported-operator]
        )

    def _chars(self, node: LatexCharsNode) -> CanonicalNode | None:
        """Normalize whitespace using math or text mode depending on the node state."""
        text = node.chars
        if self.config.normalize_whitespace:
            parsing_state = getattr(node, "parsing_state", None)
            if getattr(parsing_state, "in_math_mode", False):
                text = re.sub(r"\s+", "", text)
            else:
                text = re.sub(r"\s+", " ", text)
        if not text:
            return None
        return CanonicalNode(kind=NodeKind.TEXT, value=text, span=self._span(node))

    def _merge_text_nodes(self, nodes: list[CanonicalNode]) -> list[CanonicalNode]:
        result: list[CanonicalNode] = []
        for node in nodes:
            if result and result[-1].kind == node.kind == NodeKind.TEXT:
                previous = result.pop()
                value = (previous.value or "") + (node.value or "")
                if self.config.normalize_whitespace:
                    value = re.sub(r"\s+", " ", value)
                span = (
                    SourceSpan(previous.span.start, node.span.end)
                    if previous.span is not None and node.span is not None
                    else None
                )
                result.append(CanonicalNode(NodeKind.TEXT, value=value, span=span))
            else:
                result.append(node)
        return result

    @staticmethod
    def _combine_scripts(nodes: list[CanonicalNode]) -> list[CanonicalNode]:
        """Attach scripts to their base and normalize sub/superscript order."""
        result: list[CanonicalNode] = []
        for node in nodes:
            if node.kind not in (NodeKind.SUBSCRIPT, NodeKind.SUPERSCRIPT):
                result.append(node)
                continue

            if (
                result
                and (result[-1].kind, node.kind)
                in (
                    (NodeKind.SUBSCRIPT, NodeKind.SUPERSCRIPT),
                    (NodeKind.SUPERSCRIPT, NodeKind.SUBSCRIPT),
                )
                and len(result[-1].children) == 2
            ):
                previous = result.pop()
                subscript, superscript = (
                    (previous.children[1], node.children[0])
                    if previous.kind == NodeKind.SUBSCRIPT
                    else (node.children[0], previous.children[1])
                )
                result.append(
                    CanonicalNode(
                        kind=NodeKind.SUBSUP,
                        children=(previous.children[0], subscript, superscript),
                        span=(
                            SourceSpan(previous.span.start, node.span.end)
                            if previous.span is not None and node.span is not None
                            else None
                        ),
                    )
                )
                continue

            if not result:
                result.append(
                    CanonicalNode(
                        kind=NodeKind.SYMBOL,
                        value="^" if node.kind == NodeKind.SUPERSCRIPT else "_",
                        children=node.children,
                        span=node.span,
                    )
                )
                continue

            base = LatexCanonicalizer._take_script_base(result)
            result.append(
                CanonicalNode(
                    kind=node.kind,
                    children=(base, node.children[0]),
                    span=(
                        SourceSpan(base.span.start, node.span.end)
                        if base.span is not None and node.span is not None
                        else None
                    ),
                )
            )
        return result

    @staticmethod
    def _take_script_base(nodes: list[CanonicalNode]) -> CanonicalNode:
        """Take the final text character, or a whole non-text node, as the base."""
        base = nodes.pop()
        if base.kind != NodeKind.TEXT or not base.value or len(base.value) == 1:
            return base

        prefix_span = (
            SourceSpan(base.span.start, base.span.end - 1)
            if base.span is not None
            else None
        )
        base_span = (
            SourceSpan(base.span.end - 1, base.span.end)
            if base.span is not None
            else None
        )
        nodes.append(
            CanonicalNode(NodeKind.TEXT, value=base.value[:-1], span=prefix_span)
        )
        return CanonicalNode(NodeKind.TEXT, value=base.value[-1], span=base_span)

    @staticmethod
    def _is_plain_macro(node: CanonicalNode, names: Collection[str]) -> bool:
        """Return whether a node is an argumentless macro in the given names."""
        return node.kind == NodeKind.MACRO and not node.children and node.value in names

    def _normalize_style_declarations(
        self, nodes: list[CanonicalNode]
    ) -> list[CanonicalNode]:
        """Apply legacy style declarations to following siblings in their scope."""
        result: list[CanonicalNode] = []
        segment: list[CanonicalNode] = []
        active_style: str | None = None

        def flush_segment() -> None:
            if not segment:
                return
            if active_style is None:
                result.extend(segment)
            else:
                result.append(
                    CanonicalNode(
                        kind=NodeKind.STYLE,
                        value=active_style,
                        children=(
                            CanonicalNode(
                                kind=NodeKind.GROUP,
                                children=tuple(segment),
                            ),
                        ),
                    )
                )
            segment.clear()

        for node in nodes:
            if self._is_plain_macro(node, STYLE_DECLARATIONS):
                flush_segment()
                source_name = node.value
                assert source_name is not None
                active_style = STYLE_DECLARATIONS[source_name]
            elif self._is_plain_macro(node, GROUP_SPLITTING_MACROS):
                flush_segment()
                result.append(node)
            else:
                segment.append(node)
        flush_segment()
        return result

    @staticmethod
    def _sequence_group(nodes: list[CanonicalNode]) -> CanonicalNode:
        """Keep a multi-node operand together as one canonical group."""
        if len(nodes) == 1 and nodes[0].kind == NodeKind.GROUP:
            return nodes[0]
        return CanonicalNode(kind=NodeKind.GROUP, children=tuple(nodes))

    def _fraction_operand(self, nodes: list[CanonicalNode]) -> CanonicalNode:
        """Use a single operand directly and group a longer sequence."""
        if len(nodes) == 1:
            return self._unwrap_group(nodes[0])
        return self._sequence_group(nodes)

    def _normalize_group_splitting(
        self, nodes: list[CanonicalNode]
    ) -> list[CanonicalNode]:
        """Replace one infix splitter with its canonical two-argument node."""
        split_indexes = [
            index
            for index, node in enumerate(nodes)
            if self._is_plain_macro(node, GROUP_SPLITTING_MACROS)
        ]
        if len(split_indexes) != 1:
            return nodes

        index = split_indexes[0]
        left = nodes[:index]
        right = nodes[index + 1 :]
        if not left or not right:
            return nodes

        source_name = nodes[index].value
        assert source_name is not None
        canonical_name = GROUP_SPLITTING_MACROS[source_name]
        if canonical_name == "frac":
            return [
                CanonicalNode(
                    kind=NodeKind.FRACTION,
                    children=(
                        self._fraction_operand(left),
                        self._fraction_operand(right),
                    ),
                )
            ]

        return [
            CanonicalNode(
                kind=NodeKind.MACRO,
                value=canonical_name,
                children=(
                    self._sequence_group(left),
                    self._sequence_group(right),
                ),
            )
        ]

    def _group(self, node: LatexGroupNode) -> CanonicalNode | list[CanonicalNode]:
        """Remove grouping syntax consumed by declarations or infix macros."""
        direct_macros = {
            child.macroname
            for child in node.nodelist
            if isinstance(child, LatexMacroNode)
        }
        contains_declaration = bool(direct_macros & STYLE_DECLARATIONS.keys())
        contains_splitter = bool(direct_macros & GROUP_SPLITTING_MACROS.keys())

        value = None if node.delimiters == ("{", "}") else repr(node.delimiters)
        children = tuple(self._visit_many(node.nodelist))
        if value is None and contains_declaration:
            return list(children)
        if value is None and contains_splitter and len(children) == 1:
            return children[0]
        if (
            value is None
            and len(children) == 1
            and children[0].kind == NodeKind.GROUP
            and children[0].value is None
        ):
            children = children[0].children
        return CanonicalNode(
            kind=NodeKind.GROUP,
            value=value,
            children=children,
            span=self._span(node),
        )

    @staticmethod
    def _unwrap_group(
        node: CanonicalNode, *, allow_optional: bool = False
    ) -> CanonicalNode:
        """
        Remove a single child argument group when its delimiters don't add structure
        """
        while (
            node.kind == NodeKind.GROUP
            and (node.value is None or allow_optional)
            and len(node.children) == 1
        ):
            node = node.children[0]
        return node

    def _argument(self, node) -> CanonicalNode:
        """Keep one parsed argument together if a conversion expands its content"""
        converted = self._visit(node)
        if isinstance(converted, CanonicalNode) and converted.kind == NodeKind.GROUP:
            return converted
        if isinstance(converted, CanonicalNode):
            converted = [converted]
        return CanonicalNode(
            kind=NodeKind.GROUP,
            children=tuple(converted or ()),
            span=self._span(node),
        )

    def _macro_arguments(self, node) -> list[CanonicalNode]:
        """Convert a macro/environment's arguments in source order."""
        nodeargd = getattr(node, "nodeargd", None)
        if nodeargd is None or not nodeargd.argnlist:
            return []
        return [self._argument(arg) for arg in nodeargd.argnlist if arg is not None]

    def _math(self, node: LatexMathNode) -> CanonicalNode:
        """Convert math content and optionally retain its delimiter spelling."""
        return CanonicalNode(
            kind=NodeKind.MATH,
            value=(
                None if self.config.normalize_math_delimiters else repr(node.delimiters)
            ),
            children=tuple(self._visit_many(node.nodelist)),
            span=self._span(node),
        )

    def _fraction(
        self, node: LatexMacroNode, arguments: list[CanonicalNode]
    ) -> CanonicalNode:
        """Store a fraction: numerator and denominator"""
        if not self.config.normalize_fraction_style or len(arguments) != 2:
            return CanonicalNode(
                kind=NodeKind.MACRO,
                value=node.macroname,
                children=tuple(arguments),
                span=self._span(node),
            )
        return CanonicalNode(
            kind=NodeKind.FRACTION,
            children=(
                self._unwrap_group(arguments[0]),
                self._unwrap_group(arguments[1]),
            ),
            span=self._span(node),
        )

    def _sqrt(
        self, node: LatexMacroNode, arguments: list[CanonicalNode]
    ) -> CanonicalNode:
        """Store a square root: radicand first, then an optional index."""
        if len(arguments) not in (1, 2):
            return CanonicalNode(
                kind=NodeKind.MACRO,
                value=node.macroname,
                children=tuple(arguments),
                span=self._span(node),
            )
        radicand = self._unwrap_group(arguments[-1])
        children = (radicand,)
        if len(arguments) == 2:
            children += (self._unwrap_group(arguments[0], allow_optional=True),)
        return CanonicalNode(
            kind=NodeKind.SQRT, children=children, span=self._span(node)
        )

    @staticmethod
    def _transparent_macro(
        arguments: list[CanonicalNode],
    ) -> list[CanonicalNode] | None:
        """Discard while keeping its content."""
        # TODO: this maybe has more to do with the metric and is not a good idea to just
        # remove them... it needs thinking
        if len(arguments) != 1:
            return None
        argument = arguments[0]
        if argument.kind == NodeKind.GROUP and argument.value is None:
            return list(argument.children)
        return [argument]

    def _macro(self, node: LatexMacroNode) -> CanonicalNode | list[CanonicalNode]:
        """Apply aliases and special macro rules, preserving other macros."""
        if node.macroname in STYLE_DECLARATIONS or node.macroname in (
            GROUP_SPLITTING_MACROS
        ):
            name = node.macroname
        else:
            name = self.config.macro_aliases.get(node.macroname, node.macroname)
        arguments = self._macro_arguments(node)
        if name in FRACTION_MACROS:
            return self._fraction(node, arguments)
        if name == "sqrt":
            return self._sqrt(node, arguments)
        if name in TRANSPARENT_MACROS:
            transparent = self._transparent_macro(arguments)
            if transparent is not None:
                return transparent
        kind = NodeKind.STYLE if name in STYLE_MACROS else NodeKind.MACRO

        return CanonicalNode(
            kind=kind,
            value=name,
            children=tuple(arguments),
            span=self._span(node),
        )

    def _environment(self, node: LatexEnvironmentNode) -> CanonicalNode:
        """Keep present environment arguments separate from its body nodes."""
        arguments = self._macro_arguments(node)
        children: tuple[CanonicalNode, ...] = tuple(self._visit_many(node.nodelist))
        if arguments:
            children = (
                CanonicalNode(kind=NodeKind.ARGUMENTS, children=tuple(arguments)),
                *children,
            )
        return CanonicalNode(
            kind=NodeKind.ENVIRONMENT,
            value=node.environmentname,
            children=children,
            span=self._span(node),
        )

    def _special(self, node: LatexSpecialsNode) -> CanonicalNode:
        """Represent math scripts and other special macros as symbols."""
        arguments = self._macro_arguments(node)
        in_math = getattr(getattr(node, "parsing_state", None), "in_math_mode", False)
        if in_math and node.specials_chars in ("^", "_") and len(arguments) == 1:
            kind = (
                NodeKind.SUPERSCRIPT
                if node.specials_chars == "^"
                else NodeKind.SUBSCRIPT
            )
            return CanonicalNode(
                kind=kind,
                children=(self._unwrap_group(arguments[0]),),
                span=self._span(node),
            )
        return CanonicalNode(
            kind=NodeKind.SYMBOL,
            value=node.specials_chars,
            children=tuple(arguments),
            span=self._span(node),
        )
