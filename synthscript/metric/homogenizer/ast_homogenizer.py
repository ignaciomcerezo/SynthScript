import re
from collections.abc import Callable

from synthscript.metric.ast.node import CanonicalNode
from synthscript.metric.ast.parser import LatexParser
from synthscript.metric.ast.representation import flatten_latex
from synthscript.metric.homogenizer.homogenizer import TextHomogenizer
from synthscript.metric.homogenizer.plain_text_replacements import REPLACEMENTS


class ASTHomogenizer(TextHomogenizer):
    def __init__(
        self,
        parser: LatexParser | None = None,
        flattener: Callable[[CanonicalNode], str] | None = None,
        plain_text_replacements: tuple[tuple[re.Pattern, str], ...] | None = None,
        plain_text_supplement_replacer: Callable[[str], str] | None = None,
    ):
        """
        Text homogenizer that uses:
            1. Usual regex replacement first, and
            2. an AST parser and flattener that canonicalizes the text with known
            structure.
        """
        self.plain_text_replacements = (
            plain_text_replacements
            if plain_text_replacements is not None
            else tuple(REPLACEMENTS)
        )
        self.parser = parser if parser is not None else LatexParser()
        self.flattener = flattener if flattener is not None else flatten_latex
        self.plain_text_supplement_replacer = plain_text_supplement_replacer

    def _replace_plain(self, text: str) -> str:

        for pattern, replacement in self.plain_text_replacements:
            text = pattern.sub(replacement, text)

        if self.plain_text_supplement_replacer is not None:
            text = self.plain_text_supplement_replacer(text)

        return text

    def _replace_canonical(self, text: str) -> str:
        return self.flattener(self.parser.parse(text))

    def homogenize(self, text: str) -> str:
        """
        Homogenizes the text by applying the plain text replacements first,
        then the AST canonicalization and projection using the given parser and
        flattener.
        """
        return self._replace_canonical(self._replace_plain(text)).strip()
