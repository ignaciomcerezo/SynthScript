import re


def _deproto(proto: list[tuple[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    return [
        (re.compile(re.escape(old)), new.replace("\\", r"\\")) for old, new in proto
    ]


ENCODING_ARTIFACTS = [
    (re.compile(re.escape(" e0")), "à"),
    (re.compile(re.escape(" e9")), "é"),
    (re.compile(re.escape(" f9")), "ù"),
]


LATEX_LITERAL_REPLACEMENTS = _deproto(
    [
        (r"\nexists", r"\not \exists"),
        (r"\dots", "..."),
        (r"\ldots", "..."),
        (r"\colon", ":"),
        (r"\varprojlim", r"\lim_{\leftarrow}"),
        (r"\varinjlim", r"\lim_{\to}"),
        (r"\/", ""),
    ]
)


_accented_characters = {
    ("'", "a"): "à",
    ("`", "a"): "à",
    ("^", "a"): "â",
    ('"', "a"): "ä",
    ("'", "e"): "é",
    ("`", "e"): "è",
    ("^", "e"): "ê",
    ('"', "e"): "ë",
    ("'", "i"): "í",
    ("`", "i"): "ì",
    ("^", "i"): "î",
    ('"', "i"): "ï",
    ("'", "o"): "ò",
    ("`", "o"): "ò",
    ("^", "o"): "ô",
    ('"', "o"): "ö",
    ("'", "u"): "ú",
    ("`", "u"): "ù",
    ("^", "u"): "û",
    ('"', "u"): "ü",
    (",", "c"): "ç",
}

LATEX_ACCENT_REPLACEMENTS = _deproto(
    [
        (source, replacement)
        for (accent, letter), replacement in _accented_characters.items()
        for source in (
            f"\\{accent}{{{letter}}}",
            f"\\{accent}{letter}",
        )
    ]
    + [(r"\c{c}", "ç")]
)


MATH_ENVIRONMENT_REPLACEMENTS = _deproto(
    [
        replacement
        for environment in ("equation", "equation*", "align", "align*")
        for replacement in (
            (rf"\begin{{{environment}}}", "$"),
            (rf"\end{{{environment}}}", "$"),
        )
    ]
)


ANNOTATION_CORRECTIONS = _deproto(
    [
        (
            r"\text{catégoriel U}",
            r"\mathcal{U}",
        ),  # TODO: solve this problem in the annotations (search and correct it)
        ("á", "à"),
        ("ó", "ò"),
    ]
)

_proto_unicode_artifacts = [
    ("``", '"'),
    ("''", '"'),
    ("«", '"'),
    ("»", '"'),
    ("\N{NON-BREAKING HYPHEN}", "-"),
    ("\N{RIGHT SINGLE QUOTATION MARK}", "'"),
    ("“", '"'),
    ("”", '"'),
    ("§", r"\S"),
    ("…", "..."),
    ("\N{EN DASH}", "-"),
    ("—", "-"),
]

UNICODE_ARTIFACTS = _deproto(_proto_unicode_artifacts)

OCR_ARTIFACTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\\n(?!(?:ot|ew|ode|u|eq|exists|ewpage|oindent|atural|eg|earrow|warrow|abla|obreak|otag)(?![a-zA-Z]))"
        ),
        " ",
    ),
    (re.compile(r"(?<!\\)`"), "'"),
]

PUNCTUATION_SPACING_CORRECTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!\\)[ \t]+(?=[.,:;])"), ""),
]

REPLACEMENTS: list[tuple[re.Pattern[str], str]] = (
    ENCODING_ARTIFACTS
    + UNICODE_ARTIFACTS
    + LATEX_LITERAL_REPLACEMENTS
    + LATEX_ACCENT_REPLACEMENTS
    + MATH_ENVIRONMENT_REPLACEMENTS
    + ANNOTATION_CORRECTIONS
    + OCR_ARTIFACTS
    + PUNCTUATION_SPACING_CORRECTIONS
)
