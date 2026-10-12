"""Source objects and zero-width DVI markers; no word is put in a TeX box."""

import re
from dataclasses import dataclass
from typing import Literal

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexSpecialsNode,
    LatexWalker,
)


class UnsupportedLatexError(ValueError):
    """The source or its rendered layout is outside the MVP's supported subset."""


@dataclass
class SourceObject:
    """
    Token of latex source
    """

    id: str
    paragraph: int
    kind: Literal["text", "inline_math", "display", "diagram"]
    text: str
    start: int
    end: int
    space_before: bool


_DISPLAYS = {"equation", "equation*", "align", "align*", "gather", "gather*"}
_TEXT_MACROS = {
    "textbf",
    "textit",
    "textrm",
    "textsf",
    "texttt",
    "emph",
    "underline",
    "LaTeX",
    "TeX",
    "%",
    "&",
    "#",
    "$",
    "_",
    "{",
    "}",
    "'",
    '"',
    "`",
    "^",
    "~",
    "=",
    ".",
    "c",
    "v",
    "H",
    "u",
    "b",
    "d",
    "-",
}
_PACKAGES = {
    "amsmath",
    "amssymb",
    "amsfonts",
    "tikz",
    "geometry",
    "inputenc",
    "fontenc",
    "lmodern",
}


def _remove_comments(source: str) -> tuple[str, list[int]]:
    # TeX comments consume the newline without inserting a word separator.
    keep = [True] * len(source)
    for match in re.finditer(r"%", source):
        pos = match.start()
        slashes = 0
        # lines that have \% are not comments
        while pos > 0 and source[pos - 1] == "\\":
            slashes += 1
            pos -= 1
        # lines that do not, are true comments - we remove them
        if slashes % 2 == 0:
            end = source.find("\n", match.start())
            end = len(source) if end == -1 else end + 1
            keep[match.start() : end] = [False] * (end - match.start())
    positions = [i for i, retained in enumerate(keep) if retained]
    return "".join(source[i] for i in positions), positions


def clean_and_tokenize(source: str) -> tuple[str, list[SourceObject]]:
    """
    Transforms a piece of source LaTeX into a renderable LaTeX code, placing special
    characters to divide content into visual pieces, and the visual pieces themselves.
    """
    """Accept a body fragment or an article document and retain source offsets."""
    if not isinstance(source, str) or not source.strip():
        raise UnsupportedLatexError("Expected a non-empty LaTeX source string.")
    clean, positions = _remove_comments(source)

    # check for includes, they are managed separately
    if re.search(
        r"\\(?:input|include|includegraphics|write|openout|read|special|catcode|"
        r"newcommand|renewcommand|def|let|csname|directlua|usepackage)\b",
        clean,
    ):
        body_check = clean.split(r"\begin{document}")[-1]
        if re.search(
            r"\\(?:input|include|includegraphics|write|openout|read|special|catcode|"
            r"newcommand|renewcommand|def|let|csname|directlua|usepackage)\b",
            body_check,
        ):
            # if there are includes and inputs outside of the preamble, we error
            raise UnsupportedLatexError(
                "External files and custom TeX commands are unsupported."
            )

    begindoc, enddoc = r"\begin{document}", r"\end{document}"
    if begindoc in clean or enddoc in clean:
        if clean.count(begindoc) != 1 or clean.count(enddoc) != 1:
            raise UnsupportedLatexError("Expected exactly one document environment.")

        offset = clean.index(begindoc) + len(begindoc)
        stop = clean.index(enddoc)
        if stop < offset or clean[stop + len(enddoc) :].strip():
            raise UnsupportedLatexError("Unexpected content outside the document.")
        preamble = clean[: offset - len(begindoc)]

        # we compact the preamble into a standarized format, keeping it short
        remainder = re.sub(r"\\documentclass(?:\[[^]]*\])?\{article\}", "", preamble)
        if remainder == preamble or "twocolumn" in preamble:
            raise UnsupportedLatexError(
                "Only single-column article documents are supported."
            )

        for package in re.finditer(r"\\usepackage(?:\[[^]]*\])?\{([^}]+)\}", remainder):
            if not set(package[1].split(",")).issubset(_PACKAGES):
                raise UnsupportedLatexError(f"Unsupported packages: {package[1]}.")
        # package and other information removal
        remainder = re.sub(r"\\usepackage(?:\[[^]]*\])?\{[^}]+\}", "", remainder)
        remainder = re.sub(
            r"\\setlength\{\\(?:textwidth|textheight|parindent|parskip)\}\{[^}]+\}",
            "",
            remainder,
        )
        # if something survives, it is unsupported.
        if remainder.strip():
            raise UnsupportedLatexError("Unsupported preamble commands.")
        body = clean[offset:stop]
    else:
        # no preamble case (the main one)
        offset = 0
        body = clean
        preamble = r"\documentclass[a4paper]{article}"

    try:
        nodes, _, _ = LatexWalker(body, tolerant_parsing=False).get_latex_nodes()
    except Exception as exc:
        raise UnsupportedLatexError(f"Malformed LaTeX: {exc}") from exc

    objects: list[SourceObject] = []
    parts: list[str] = []
    paragraph = 0
    spaced = False
    pending_start: int | None = None
    pending_end = 0
    display_end = ""

    def flush(
        kind: Literal["text", "inline_math", "display", "diagram"] = "text",
    ) -> None:
        """
        Gives the pending content an ID, generates a source object and adds
        rendendering markers to it. Sets pending_start as None, as this is signal
        that a new token is being parsed.
        """
        nonlocal pending_start, spaced, display_end
        if pending_start is None:
            return
        identifier = f"ss{len(objects):06d}"
        fragment = body[pending_start:pending_end]
        objects.append(
            SourceObject(
                identifier,
                paragraph,
                kind,
                fragment,
                positions[offset + pending_start],
                positions[offset + pending_end - 1] + 1,
                spaced,
            )
        )
        # the markers ask dvisvgm to report the bounding box of the rendered object
        end_marker = (
            r"\special{dvisvgm:raw </g>}"
            + rf"\special{{dvisvgm:raw <g data-ss-object='{identifier}' "
            + rf"data-bbox='{{?bbox {identifier}}}' data-y='{{?y}}'/>}}"
        )
        start_marker = (
            display_end
            + rf"\special{{dvisvgm:bbox new {identifier}}}"
            + rf"\special{{dvisvgm:raw <g data-ss-start='{identifier}' "
            + r"data-y='{?y}'>}"
        )
        if kind == "display":
            # TeX manages display latex different inside environments
            opening = re.match(r"\\begin\{[^}]+\}|\\\[|\$\$", fragment)
            assert opening is not None
            parts.append(opening[0] + start_marker + fragment[opening.end() :])
        else:
            parts.append(r"\leavevmode" + start_marker + fragment + end_marker)

        display_end = end_marker if kind == "display" else ""
        pending_start = None
        spaced = False

    def update_window(start: int, stop: int) -> None:
        """
        Updates (by extension) the window of content that will turn into the next token

        This mechanism allows us to concatenate content that can be split among
        different nodes into a single token.
        """
        nonlocal pending_start, pending_end
        if pending_start is None:
            pending_start = start
        pending_end = stop

    for node in nodes:
        if isinstance(node, LatexCharsNode):
            # normal text is split into words
            for match in re.finditer(r"\s+|[^\s]+", node.chars):
                if match[0].isspace():
                    flush()
                    parts.append(match[0])
                    spaced = True
                    if re.search(r"\n\s*\n", match[0]):
                        paragraph += 1
                else:
                    update_window(node.pos + match.start(), node.pos + match.end())

        elif isinstance(node, LatexMathNode):
            flush()
            update_window(node.pos, node.pos + node.len)
            flush("display" if node.displaytype == "display" else "inline_math")
        elif isinstance(node, LatexEnvironmentNode):
            flush()
            if node.environmentname not in _DISPLAYS | {"tikzpicture"}:
                raise UnsupportedLatexError(
                    f"Unsupported environment: {node.environmentname}."
                )
            update_window(node.pos, node.pos + node.len)
            flush("diagram" if node.environmentname == "tikzpicture" else "display")
        elif isinstance(node, LatexMacroNode) and node.macroname in {
            "par",
            "noindent",
            "\\",
        }:
            flush()
            parts.append(body[node.pos : node.pos + node.len])
            if node.macroname == "par":
                paragraph += 1
            spaced = True
        elif isinstance(node, LatexSpecialsNode) and node.specials_chars in {
            "~",
            "--",
            "---",
            "``",
            "''",
        }:
            update_window(node.pos, node.pos + node.len)
        elif isinstance(node, LatexGroupNode) and not node.nodelist:
            # an empty group terminates a linebreak's optional argument scan,
            # as in \\{}[ABC]
            if pending_start is None:
                parts.append(body[node.pos : node.pos + node.len])
            else:
                update_window(node.pos, node.pos + node.len)
        elif isinstance(node, (LatexMacroNode, LatexGroupNode)):
            if isinstance(node, LatexMacroNode) and node.macroname not in _TEXT_MACROS:
                raise UnsupportedLatexError(
                    f"Unsupported text command: \\{node.macroname}."
                )
            fragment = body[node.pos : node.pos + node.len]
            if re.search(r"\s", fragment.strip()):
                raise UnsupportedLatexError(
                    "Multi-word text groups are unsupported in the MVP."
                )
            update_window(node.pos, node.pos + node.len)
        else:
            raise UnsupportedLatexError(
                f"Unsupported source node: {type(node).__name__}."
            )
    flush()

    if display_end:
        parts.append(r"\par" + display_end)
    if not objects:
        raise UnsupportedLatexError("The source contains no renderable objects.")

    # once everything has been tokenized and marked, a document is built
    document = (
        r"\def\pgfsysdriver{pgfsys-dvisvgm.def}"
        + "\n"
        + preamble
        + "\n"
        + r"\usepackage{amsmath,amssymb,tikz}"
        + "\n"
        + begindoc
        + "\n"
        + r"\pagestyle{empty}"
        + "\n"
        + "".join(parts)
        + "\n"
        + enddoc
        + "\n"
    )
    return document, objects
