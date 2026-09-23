from pylatexenc.latexwalker import LatexWalker, get_default_latex_context_db
from pylatexenc.macrospec import MacroSpec, MacroStandardArgsParser, SpecialsSpec

from synthscript.metric.ast.canonicalizer import (
    CanonicalizationConfig,
    LatexCanonicalizer,
)
from synthscript.metric.ast.macro_groups import (
    FRACTION_MACROS,
    GROUP_SPLITTING_MACROS,
)
from synthscript.metric.ast.node import CanonicalNode


class LatexParser:
    """
    Parse latex code into a canonical AST.
    """

    def __init__(self, config: CanonicalizationConfig | None = None) -> None:
        self._canonicalizer = LatexCanonicalizer(config)
        self._latex_context = get_default_latex_context_db()
        self._latex_context.add_context_category(
            "synthscript",
            macros=[
                MacroSpec(name, "{{")
                for name in sorted(
                    FRACTION_MACROS | set(GROUP_SPLITTING_MACROS.values())
                )
            ],
            specials=[
                SpecialsSpec("^", args_parser=MacroStandardArgsParser("{")),
                SpecialsSpec("_", args_parser=MacroStandardArgsParser("{")),
            ],
            prepend=True,
        )

    def parse(self, source: str) -> CanonicalNode:
        """Parses a latex string and returns its canonical tree form."""
        nodes, _, _ = LatexWalker(
            source,
            latex_context=self._latex_context,
        ).get_latex_nodes()
        return self._canonicalizer.canonicalize(nodes)
