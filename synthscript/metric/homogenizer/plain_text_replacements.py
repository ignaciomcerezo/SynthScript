import re


def _deproto(proto: list[tuple[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(re.escape(a)), b) for (a, b) in proto]


MATH_ENVS_TO_DOLLAR = [
    (re.compile(re.escape(r"\begin{equation}")), r"$"),
    (re.compile(re.escape(r"\end{equation}")), r"$"),
    (re.compile(re.escape(r"\begin{equation*}")), r"$"),
    (re.compile(re.escape(r"\end{equation*}")), r"$"),
    (re.compile(re.escape(r"\begin{align}")), r"$"),
    (re.compile(re.escape(r"\end{align}")), r"$"),
    (re.compile(re.escape(r"\begin{align*}")), r"$"),
    (re.compile(re.escape(r"\end{align*}")), r"$"),
    (re.compile(re.escape(r"\(")), r"$"),
    (re.compile(re.escape(r"\)")), r"$"),
    (re.compile(re.escape(r"\[")), r"$"),
    (re.compile(re.escape(r"\]")), r"$"),
    (re.compile(re.escape(r"$$")), r"$"),
]

ENCODING_ARTIFACTS = [
    (re.compile(re.escape(" e0")), "à"),
    (re.compile(re.escape(" e9")), "é"),
    (re.compile(re.escape(" f9")), "ù"),
]


_proto_macro_replacements: list[tuple[str, str]] = [
    (r"\nexists", r"\not \exists"),
    (r"\dots", "..."),
    (r"\ldots", "..."),
    (r"\colon", ":"),
    (r"\varprojlim", r"\lim_{\leftarrow}"),
    (r"\varinjlim", r"\lim_{\to}"),
    ("—", "-"),
    (r"\/", r""),
    (r"\nobreak", ""),
    (
        r"\text{catégoriel U}",
        r"\mathcal{U}",
    ),  # TODO: solve this problem in the annotations (search and correct it)
    ("~", " "),
    ("á", "à"),
    ("ó", "ò"),
]
MACRO_REPLACEMENTS = _deproto(_proto_macro_replacements)

_proto_unicode_artifacts = [
    ("``", '"'),
    ("''", '"'),
    ("«", '"'),
    ("»", '"'),
    ("‑", "-"),
    ("’", "'"),
    ("“", '"'),
    ("”", '"'),
    ("`", "'"),
    ("§", r"\S"),
    ("…", "..."),
    ("–", "-"),
]

UNICODE_ARTIFACTS = _deproto(_proto_unicode_artifacts)

REGEX_MATH_MACROS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\U\b"), r"\\mathcal U"),
    (re.compile(r"\\E\b"), r"\\mathcal E"),
    (re.compile(r"\\tag\{\d*\}"), ""),
    (re.compile(r"\\not *="), r"\neq"),
]

STRUCTURAL_PADDING = [(re.compile(re.escape("\ ")), " ")]

REPLACEMENTS: list[tuple[re.Pattern[str], str]] = (
    MATH_ENVS_TO_DOLLAR
    + ENCODING_ARTIFACTS
    + MACRO_REPLACEMENTS
    + UNICODE_ARTIFACTS
    + REGEX_MATH_MACROS
    + STRUCTURAL_PADDING
)
