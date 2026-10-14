import pytest

from synthscript.metric.ast.canonicalizer import CanonicalizationConfig
from synthscript.metric.ast.node import (
    CanonicalNode,
    MetricNode,
    MetricNodeKind,
    NodeKind,
    SourceSpan,
    to_metric_tree,
)
from synthscript.metric.ast.parser import LatexParser
from synthscript.metric.ast.representation import (
    flatten,
    flatten_latex,
    render_vertical,
)
from synthscript.metric.node_metrics import ConfiguredNodeMetrics
from synthscript.metric.tree_distance import tree_edit_distance


def parse(source: str, config: CanonicalizationConfig | None = None) -> CanonicalNode:
    return LatexParser(config).parse(source)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (r"$a_2^3$", r"$a^3_2$"),
        (r"$x^2$", r"$x^{2}$"),
        (r"$x_i$", r"$x_{i}$"),
        (r"$\frac{a}{b}$", r"$\dfrac a b$"),
        (r"$\frac{a}{b}$", r"$\tfrac{a}{b}$"),
        (r"$a  b$", r"$ab$"),
        ("a\t\nb", "a b"),
        (r"$x$", r"\(x\)"),
        (r"$x$", r"$$x$$"),
        (r"$x$", r"\[x\]"),
    ],
)
def test_equivalent_syntax(left: str, right: str) -> None:
    assert parse(left) == parse(right)


def test_explicit_styles_are_preserved() -> None:
    assert parse(r"$\mathrm{ab}$") != parse(r"$ab$")
    assert parse(r"$\mathit{a}b$") != parse(r"$ab$")


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("to", "rightarrow"),
        ("lnot", "neg"),
        ("gets", "leftarrow"),
        ("le", "leq"),
        ("ge", "geq"),
        ("ne", "neq"),
        ("lor", "vee"),
        ("land", "wedge"),
    ],
)
def test_macro_aliases(alias: str, canonical: str) -> None:
    assert parse(f"$\\{alias}$") == parse(f"$\\{canonical}$")


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        (r"$a\mathop{x}$", r"$a{x}$"),
        (r"$a\quad b$", r"$ab$"),
        (r"$a\qquad b$", r"$ab$"),
        (r"$a\:b$", r"$ab$"),
        (r"$a\;b$", r"$ab$"),
        (r"$a\s b$", r"$ab$"),
        (r"$a\!b$", r"$ab$"),
        (r"$a\medskip b$", r"$ab$"),
        (r"$a\bigskip b$", r"$ab$"),
        (r"$a\smallskip b$", r"$ab$"),
        (r"$a\break b$", r"$ab$"),
        (r"$a\big(b)$", r"$a(b)$"),
        (r"$\left(x\right)$", r"$(x)$"),
        (r"$a\mid b$", r"$ab$"),
    ],
)
def test_irrelevant_macros_are_removed(source: str, canonical: str) -> None:
    assert parse(source) == parse(canonical)


def test_operatorname_keeps_its_argument_group() -> None:
    tree = parse(r"$\operatorname{Autext}$")
    assert tree == parse(r"$\mathrm{Autext}$")
    assert flatten_latex(tree) == r"$\mathrm{Autext}$"
    assert parse(flatten_latex(tree)) == tree


def test_custom_aliases_replace_defaults() -> None:
    config = CanonicalizationConfig(macro_aliases={"foo": "quux"})
    assert parse(r"$\foo$", config) == parse(r"$\quux$", config)
    assert parse(r"$\to$", config) != parse(r"$\rightarrow$", config)


def test_macro_arguments_keep_one_canonical_group() -> None:
    canonical = parse(r"$\tilde{x}$")
    assert parse(r"$\tilde x$") == canonical
    assert parse(r"$\tilde{{{{x}}}}$") == canonical
    argument = canonical.children[0].children[0].children[0]
    assert argument.kind == NodeKind.GROUP
    assert argument.children == (CanonicalNode(NodeKind.TEXT, "x"),)


@pytest.mark.parametrize(
    ("nested", "minimal"),
    [
        (r"$\frac{{{{a}}}}{b}$", r"$\frac{a}{b}$"),
        (r"$\sqrt{{{{x}}}}$", r"$\sqrt{x}$"),
        (r"$x_{{{{i}}}}$", r"$x_i$"),
    ],
)
def test_structural_parents_absorb_redundant_groups(nested: str, minimal: str) -> None:
    assert parse(nested) == parse(minimal)


def test_redundant_math_group_is_removed_after_script_binding() -> None:
    grouped = parse(r"${x}$")
    assert grouped == parse(r"$x$")
    assert parse(r"$a{x}b$") == parse(r"$axb$")
    assert flatten_latex(grouped) == r"$x$"

    # These braces still determine that the entire ``xy`` sequence is the base.
    assert parse(r"${xy}^2$") != parse(r"$xy^2$")


def test_script_shape_and_negative_cases() -> None:
    math = parse(r"$a_2^3$").children[0]
    assert math.children == (
        CanonicalNode(
            NodeKind.SUBSUP,
            children=(
                CanonicalNode(NodeKind.TEXT, value="a"),
                CanonicalNode(NodeKind.TEXT, value="2"),
                CanonicalNode(NodeKind.TEXT, value="3"),
            ),
        ),
    )
    assert parse(r"$a_2^3$") != parse(r"$a_3^2$")
    assert parse(r"$a+b$") != parse(r"$b+a$")
    assert parse(r"a_2^3") != parse(r"a^3_2")


def test_script_binds_to_last_math_character() -> None:
    source = r"$\mathcal{E} \simeq \prod_{i \in I} \mathrm{Ens}(G_i)$"
    tree = parse(source)
    math = tree.children[0]
    assert math.children[2].kind == NodeKind.SUBSCRIPT
    assert math.children[2].children[0] == CanonicalNode(NodeKind.MACRO, "prod")
    assert math.children[3:] == (
        CanonicalNode(
            NodeKind.STYLE,
            "mathrm",
            children=(
                CanonicalNode(
                    NodeKind.GROUP,
                    children=(CanonicalNode(NodeKind.TEXT, "Ens"),),
                ),
            ),
        ),
        CanonicalNode(NodeKind.TEXT, "("),
        CanonicalNode(
            NodeKind.SUBSCRIPT,
            children=(
                CanonicalNode(NodeKind.TEXT, "G"),
                CanonicalNode(NodeKind.TEXT, "i"),
            ),
        ),
        CanonicalNode(NodeKind.TEXT, ")"),
    )
    assert "|-- text: 'G'" in render_vertical(tree, ascii_only=True)


def test_flatten_preserves_tree_shape_without_source_spans() -> None:
    assert flatten(parse(r"$a_2^3$")) == flatten(parse(r"$a^3_2$"))
    assert flatten(CanonicalNode(NodeKind.TEXT, "x", span=SourceSpan(0, 1))) == flatten(
        CanonicalNode(NodeKind.TEXT, "x", span=SourceSpan(9, 10))
    )
    assert flatten(
        CanonicalNode(NodeKind.ROOT, children=(CanonicalNode(NodeKind.TEXT, "ab"),))
    ) != flatten(
        CanonicalNode(
            NodeKind.ROOT,
            children=(
                CanonicalNode(NodeKind.TEXT, "a"),
                CanonicalNode(NodeKind.TEXT, "b"),
            ),
        )
    )


def test_fraction_shape_and_style_switch() -> None:
    fraction = parse(r"$\frac{a}{b}$").children[0].children[0]
    assert fraction == CanonicalNode(
        NodeKind.FRACTION,
        children=(
            CanonicalNode(NodeKind.TEXT, value="a"),
            CanonicalNode(NodeKind.TEXT, value="b"),
        ),
    )
    assert parse(r"$\frac{a}{b}$") != parse(r"$\frac{b}{a}$")
    config = CanonicalizationConfig(normalize_fraction_style=False)
    assert parse(r"$\frac{a}{b}$", config) != parse(r"$\dfrac{a}{b}$", config)
    assert (
        parse(r"$\frac{a}{b}$", config).children[0].children[0].kind == NodeKind.MACRO
    )


def test_square_root_index_and_argument_order() -> None:
    root = parse(r"$\sqrt[3]{x}$").children[0].children[0]
    assert root == CanonicalNode(
        NodeKind.SQRT,
        children=(
            CanonicalNode(NodeKind.TEXT, value="x"),
            CanonicalNode(NodeKind.TEXT, value="3"),
        ),
    )
    assert parse(r"$\sqrt[3]{x}$") == parse(r"$\sqrt[3]x$")
    assert parse(r"$\sqrt[2]{x}$") == parse(r"$\sqrt{x}$")
    assert flatten_latex(parse(r"$\sqrt[2]{x}$")) == r"$\sqrt{x}$"
    assert parse(r"$\sqrt{x}$") != parse(r"$\sqrt[3]{x}$")
    assert parse(r"$\sqrt[3]{x}$") != parse(r"$\sqrt[4]{x}$")


def test_comments_and_comment_switch() -> None:
    source = "a%ignored\nb"
    assert parse(source) == parse("ab")
    config = CanonicalizationConfig(remove_comments=False)
    tree = parse(source, config)
    assert tree != parse("ab", config)
    assert tree.children[1].kind == NodeKind.COMMENT
    assert tree.children[1].value == "%ignored\n"


def test_whitespace_switch_and_text_inside_math() -> None:
    assert parse(r"$\text{a  b}$") == parse(r"$\text{a b}$")
    assert parse(r"$\text{a b}$") != parse(r"$\text{ab}$")
    config = CanonicalizationConfig(normalize_whitespace=False)
    assert parse("a  b", config) != parse("a b", config)
    assert parse(r"$a b$", config) != parse(r"$ab$", config)


def test_math_delimiter_switch() -> None:
    config = CanonicalizationConfig(normalize_math_delimiters=False)
    assert parse(r"$x$", config) != parse(r"\(x\)", config)
    assert parse(r"$x$", config) != parse(r"\[x\]", config)


def test_environment_arguments_body_and_specials() -> None:
    tree = parse(r"\begin{array}{cc}a&b\end{array}")
    environment = tree.children[0]
    assert environment.kind == NodeKind.ENVIRONMENT
    assert environment.value == "array"
    assert environment.children[0].kind == NodeKind.ARGUMENTS
    assert environment.children[1].value == "a"
    assert environment.children[2] == CanonicalNode(NodeKind.SYMBOL, value="&")
    assert parse(r"\begin{array}{cc}a&b\end{array}") != parse(
        r"\begin{array}{c}a&b\end{array}"
    )
    assert parse(r"\begin{array}{cc}a&b\end{array}") != parse(
        r"\begin{array}{cc}a+b\end{array}"
    )


def test_environment_omits_empty_arguments_node() -> None:
    environment = parse(r"\begin{equation}x\end{equation}").children[0]
    assert environment.kind == NodeKind.ENVIRONMENT
    assert all(child.kind != NodeKind.ARGUMENTS for child in environment.children)
    assert environment.children == (CanonicalNode(NodeKind.TEXT, "x"),)


def test_source_spans_do_not_change_equality() -> None:
    left = CanonicalNode(NodeKind.TEXT, value="x", span=SourceSpan(0, 1))
    right = CanonicalNode(NodeKind.TEXT, value="x", span=SourceSpan(10, 11))
    assert left == right


@pytest.mark.parametrize(
    "source",
    [
        r"$a_2^3$",
        r"$\frac{a}{b}+\sqrt[3]{x}$",
        r"$\mathcal{E} \simeq \prod_{i \in I} \mathrm{Ens}(G_i)$",
        r"\begin{array}{cc}a&b\end{array}",
        r"$\alpha x$",
        r"\(x\)",
        "a%ignored\nb",
    ],
)
def canonization_is_idempotent(source: str) -> None:
    tree = parse(source)
    assert parse(flatten_latex(tree)) == tree


def test_flatten_latex_preserves_nondefault_delimiters_and_comments() -> None:
    configs = (
        CanonicalizationConfig(normalize_math_delimiters=False),
        CanonicalizationConfig(remove_comments=False),
    )
    sources = (r"\[x\]", "a% comment\nb")
    for source, config in zip(sources, configs, strict=True):
        tree = parse(source, config)
        assert parse(flatten_latex(tree), config) == tree


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        (r"$22 \over 33$", r"$\frac{22}{33}$"),
        ("$" + r"{22 \over 33}$", r"$\frac{22}{33}$"),
        ("$" + r"{2 \choose 3}$", r"$\binom{2}{3}$"),
        (
            r"$2 \over {3 \over 4}$",
            r"$\frac{2}{\frac{3}{4}}$",
        ),
    ],
)
def test_group_splitting_macros(legacy: str, canonical: str) -> None:
    tree = parse(legacy)
    assert tree == parse(canonical)
    assert parse(flatten_latex(tree)) == tree


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        (
            "$" + r"{\cal Text \bf Text2}$",
            r"$\mathcal{Text}\mathbf{Text2}$",
        ),
        (
            "$" + r"{\cal A \bf B \cal C}$",
            r"$\mathcal{A}\mathbf{B}\mathcal{C}$",
        ),
        (
            "$" + r"{\cal a \over b}$",
            r"$\frac{\mathcal{a}}{\mathcal{b}}$",
        ),
    ],
)
def test_scoped_style_declarations(legacy: str, canonical: str) -> None:
    tree = parse(legacy)
    assert tree == parse(canonical)
    assert parse(flatten_latex(tree)) == tree


def test_style_declaration_stops_at_its_group_boundary() -> None:
    scoped = parse("$" + r"{\cal A}B$")
    assert scoped == parse(r"$\mathcal{A}B$")
    assert scoped != parse(r"$\mathcal{AB}$")


@pytest.mark.parametrize(
    ("declaration", "text_command", "math_command"),
    [
        ("bf", "textbf", "mathbf"),
        ("it", "textit", "mathit"),
        ("rm", "textrm", "mathrm"),
        ("sf", "textsf", "mathsf"),
        ("tt", "texttt", "mathtt"),
    ],
)
def test_style_declarations_are_mode_dependent(
    declaration: str, text_command: str, math_command: str
) -> None:
    text_declaration = parse(rf"{{\{declaration} text}}")
    math_declaration = parse(rf"${{\{declaration} x}}$")

    assert text_declaration == parse(rf"\{text_command}{{text}}")
    assert math_declaration == parse(rf"$\{math_command}{{x}}$")


def test_malformed_repeated_infix_is_not_reassociated() -> None:
    math = parse(r"$2 \over 3 \over 4$").children[0]
    assert (
        sum(
            child.kind == NodeKind.MACRO and child.value == "over"
            for child in math.children
        )
        == 2
    )


def test_metric_text_nodes_expand_into_character_nodes() -> None:
    canonical = CanonicalNode(NodeKind.TEXT, "a b")

    assert to_metric_tree(canonical) == MetricNode(
        MetricNodeKind.ROOT,
        children=(
            MetricNode(MetricNodeKind.CHAR, "a"),
            MetricNode(MetricNodeKind.CHAR, " "),
            MetricNode(MetricNodeKind.CHAR, "b"),
        ),
    )


def test_metric_text_expands_in_place_around_structural_nodes() -> None:
    canonical = parse(r"left $x$ right")
    metric_root = to_metric_tree(canonical)

    assert metric_root.kind == MetricNodeKind.ROOT
    assert tuple(child.kind for child in metric_root.children) == (
        *(MetricNodeKind.CHAR for _ in "left "),
        MetricNodeKind.MATH,
        *(MetricNodeKind.CHAR for _ in " right"),
    )
    math = metric_root.children[len("left ")]
    assert math.children == (MetricNode(MetricNodeKind.CHAR, "x"),)


def test_tree_edit_distance_uses_character_metric_tree() -> None:
    assert (
        tree_edit_distance(
            parse(r"ab $x$ cd"),
            parse(r"aZ $x$ cY"),
        )
        == 2
    )


def test_metric_parameters_customize_weights_and_costs() -> None:
    parameters = ConfiguredNodeMetrics(
        character_weight=2,
        fraction_weight=7,
        weight_factor=3,
        subscript_superscript_cost=5,
    )
    character = MetricNode(MetricNodeKind.CHAR, "x")
    fraction = MetricNode(
        MetricNodeKind.FRACTION,
        children=(character,),
    )

    assert parameters.weight(character) == 2
    assert parameters.mass(fraction) == 9
    assert parameters.weight_cost(fraction) == 21
    assert (
        parameters.substitution_cost(
            MetricNode(MetricNodeKind.SUBSCRIPT),
            MetricNode(MetricNodeKind.SUPERSCRIPT),
        )
        == 5
    )


@pytest.mark.parametrize(
    ("kind", "parameter", "cost"),
    [
        (MetricNodeKind.MACRO, "macro_substitution_cost", 1.25),
        (MetricNodeKind.SYMBOL, "symbol_substitution_cost", 1.5),
        (MetricNodeKind.STYLE, "style_substitution_cost", 1.75),
        (MetricNodeKind.ENVIRONMENT, "environment_substitution_cost", 2.0),
    ],
)
def test_non_character_names_have_atomic_substitution_costs(
    kind: MetricNodeKind, parameter: str, cost: float
) -> None:
    parameters = ConfiguredNodeMetrics(**{parameter: cost})
    source = MetricNode(kind, "a-very-long-name")
    target = MetricNode(kind, "x")

    assert parameters.substitution_cost(source, target) == cost


def test_macro_name_length_does_not_scale_tree_distance() -> None:
    assert tree_edit_distance(parse(r"$\epsilon$"), parse(r"$\xi$")) == 1


def test_tree_edit_distance_uses_metric_parameters_by_default() -> None:
    character = MetricNode(MetricNodeKind.CHAR, "x")
    source = MetricNode(
        MetricNodeKind.ROOT,
        children=(character,),
    )
    target = MetricNode(MetricNodeKind.ROOT)

    default_parameters = ConfiguredNodeMetrics()
    assert tree_edit_distance(source, target) == pytest.approx(
        default_parameters.weight_cost(character)
    )

    custom_parameters = ConfiguredNodeMetrics(character_weight=3)
    assert tree_edit_distance(
        source,
        target,
        parameters=custom_parameters,
    ) == pytest.approx(custom_parameters.weight_cost(character))
