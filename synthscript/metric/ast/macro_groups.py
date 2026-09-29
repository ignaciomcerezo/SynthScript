from synthscript.metric.ast.node import CanonicalNode, NodeKind

# different spellings for one single symbol
# macro synonym: macro to be replaced with
MACRO_ALIASES = {
    "to": "rightarrow",
    "longrightarrow": "rightarrow",
    "longleftarrow": "leftarrow",
    "longleftrightarrow": "leftrightarrow",
    "iff": "Leftrightarrow",
    "implies": "Rightarrow",
    "impliedby": "Leftarrow",
    "Longrightarrow": "Rightarrow",
    "Longleftarrow": "Leftarrow",
    "Longleftrightarrow": "Leftrightarrow",
    "lnot": "neg",
    "gets": "leftarrow",
    "le": "leq",
    "ge": "geq",
    "ne": "neq",
    "lor": "vee",
    "land": "wedge",
    "operatorname": "mathrm",
    "text": "mathrm",
    "widehat": "hat",
    # TODO: expand this list
}

CanonicalReplacement = CanonicalNode | tuple[CanonicalNode, ...] | None


def _styled_text(style: str, text: str) -> CanonicalNode:
    return CanonicalNode(
        kind=NodeKind.STYLE,
        value=style,
        children=(
            CanonicalNode(
                kind=NodeKind.GROUP,
                children=(CanonicalNode(NodeKind.TEXT, value=text),),
            ),
        ),
    )


# argumentless commands where values may be subtrees or tuples of siblign subtrees
ARGUMENTLESS_MACRO_REPLACEMENTS: dict[str, CanonicalReplacement] = {
    "E": _styled_text("mathcal", "E"),
    "U": _styled_text("mathcal", "U"),
}

# replacements selected and triggered by a macro name and its single plain-text argument
SINGLE_TEXT_ARGUMENT_MACRO_REPLACEMENTS: dict[tuple[str, str], CanonicalReplacement] = {
    ("not", "="): CanonicalNode(NodeKind.MACRO, value="neq"),
}

# mode dependent macro and special replacements. If None, removes the matched
# node.
TEXT_MODE_MACRO_REPLACEMENTS: dict[str, CanonicalReplacement] = {
    " ": CanonicalNode(NodeKind.TEXT, value=" "),
}
MATH_MODE_MACRO_REPLACEMENTS: dict[str, CanonicalReplacement] = {
    " ": None,
}
TEXT_MODE_SPECIAL_REPLACEMENTS: dict[str, CanonicalReplacement] = {
    "~": CanonicalNode(NodeKind.TEXT, value=" "),
}
MATH_MODE_SPECIAL_REPLACEMENTS: dict[str, CanonicalReplacement] = {
    "~": None,
}

# custom one-argument commands from our corpus
SINGLE_ARGUMENT_MACROS = {
    "operatorname",
}

DOUBLE_ARGUMENT_MACROS = {
    "stackrel",
}

# their argument is left in the comparison tree.
TRANSPARENT_MACROS = {
    # "mathit",
    "textit",
    "textbf",
    "textrm",
    "textsf",
    "textsl",
    "texttt",
    "underline",
    "emph",
    "footnote",
}

# These argumentless commands are discarded.
CONTENT_INDEPENDENT_MACROS = {
    "mathop",
    "quad",
    "qquad",
    ",",
    ":",
    ";",
    "s",
    "!",
    "medskip",
    "bigskip",
    "smallskip",
    "break",
    "big",
    "left",
    "right",
    "mid",
}

# Both command and its parsed arguments are discarded. The values are
# pylatexenc argument specifications, so adding an entry also teaches the parser
# which source tokens belong to the discarded command.
CONTENT_INDEPENDENT_MACROS_WITH_ARGS = {
    "tag": "{",
}

# wrapper is discarded but the environment body is preserved.
TRANSPARENT_ENVIRONMENTS = {
    "center",
}

# both wrapper and body are discarded.
CONTENT_INDEPENDENT_ENVIRONMENTS: set[str] = set()

# they differ in display style but have the same structure.
FRACTION_MACROS = {
    "frac",
    "dfrac",
    "tfrac",
}

STYLE_MACROS = {
    "mathrm",
    "mathit",
    "mathbf",
    "mathcal",
    "mathbb",
    "mathscr",
    "mathfrak",
    "mathsf",
    "mathtt",
}

# infix commands
GROUP_SPLITTING_MACROS = {"over": "frac", "choose": "binom"}

# Style declarations select different explicit commands depending on whether
# they occur in text or math. Keeping both mappings here lets downstream users
# extend the supported declarations without changing the canonicalizer.
TEXT_STYLE_DECLARATIONS = {
    "cal": "mathcal",
    "bf": "textbf",
    "it": "textit",
    "mit": "mathit",
    "rm": "textrm",
    "sf": "textsf",
    "tt": "texttt",
}

MATH_STYLE_DECLARATIONS = {
    "cal": "mathcal",
    "bf": "mathbf",
    "it": "mathit",
    "mit": "mathit",
    "rm": "mathrm",
    "sf": "mathsf",
    "tt": "mathtt",
}
