import re

french_latex_characters: set[str] = {
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ö",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "é",
    "à",
    "è",
    "ù",
    "â",
    "ê",
    "î",
    "ô",
    "û",
    "ë",
    "ï",
    "ü",
    "ÿ",
    "ç",
    "œ",
    "æ",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "É",
    "À",
    "È",
    "Ù",
    "Â",
    "Ê",
    "Î",
    "Ô",
    "Û",
    "Ë",
    "Ï",
    "Ü",
    "Ÿ",
    "Ç",
    "Œ",
    "Æ",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    " ",
    "\t",
    "\n",
    "\r",
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "-",
    "_",
    "/",
    "\\",
    "|",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "+",
    "=",
    "<",
    ">",
    "'",
    '"',
}

MACRO_REPLACEMENTS: list[tuple[str, str]] = [
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

ENCODING_ARTIFACTS: list[tuple[str, str]] = [
    (" e0", "à"),
    (" e9", "é"),
    (" f9", "ù"),
]

UNICODE_ARTIFACTS: list[tuple[str, str]] = [
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

REPLACEMENTS_ENVS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    ((r"\textit{", "}"), ("", "")),
    ((r"\textbf{", "}"), ("", "")),
    ((r"\textsl{", "}"), ("", "")),
    ((r"\underline{", "}"), ("", "")),
    ((r"\emph{", "}"), ("", "")),
    ((r"\footnote{", "}"), ("", "")),
    ((r"\begin{center}", r"\end{center}"), ("", "")),
]

text_nobracket_p = r"[^{}]"
bracketed_block = r"\{" + text_nobracket_p + r"*\}"
cmm_p = r"\\[a-zA-Z]+"
for _ in range(10):
    bracketed_block = r"\{(?:" + text_nobracket_p + r"|" + bracketed_block + r")*\}"
scriptable_block = rf"({cmm_p}|\w|{bracketed_block})"

# TODO: the text should be corrected instead of having all of this.
REGEX_MATH_MACROS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\U\b"), r"\\mathcal U"),
    (re.compile(r"\\E\b"), r"\\mathcal E"),
    (
        re.compile(rf"\{{\\rm *({bracketed_block}|{text_nobracket_p}+)\}}"),
        r"\\mathrm{\1}",
    ),
    (re.compile(r"\\tag\{\d*\}"), ""),
    (re.compile(r"\\not *="), r"\neq"),
    (re.compile(r"\{\\cal *"), r"\\mathcal{"),
    (re.compile(r"\\cal"), r"\\mathcal"),
    (re.compile(r"\{\\frak *"), r"\\mathfrak{"),
    (re.compile(r"\\frak"), r"\\mathfrak"),
    (re.compile(rf"\\textrm({bracketed_block})"), r"\\text\1"),
    (re.compile(rf"\\operatorname({bracketed_block})"), r"\\text\1"),
]

TASK_SPECIFIC_REGEX_REPLACEMENTS: dict[int, list[tuple[re.Pattern, str]]] = {
    438: [(re.compile(r"\{\\ U\\\}"), r"{U}")],
}

accent_latex_mispellings = {
    ("'", "a"): "á",
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
    ("'", "o"): "ó",
    ("`", "o"): "ò",
    ("^", "o"): "ô",
    ('"', "o"): "ö",
    ("'", "u"): "ú",
    ("`", "u"): "ù",
    ("^", "u"): "û",
    ('"', "u"): "ü",
    (",", "c"): "ç",
}

accent_latex_mispellings_regex = re.compile(
    r"""
    \\([`'^",])
    \{?
    ([A-Za-z])
    \}?
    """,
    re.VERBOSE,
)

# maybe this is okay to keep....
MATH_ENVS_TO_DOLLAR = [
    (r"\begin{equation}", r"$"),
    (r"\end{equation}", r"$"),
    (r"\begin{equation*}", r"$"),
    (r"\end{equation*}", r"$"),
    (r"\begin{align}", r"$"),
    (r"\end{align}", r"$"),
    (r"\begin{align*}", r"$"),
    (r"\end{align*}", r"$"),
    (r"\(", r"$"),
    (r"\)", r"$"),
    (r"\[", r"$"),
    (r"\]", r"$"),
    (r"$$", r"$"),
]

math_chars = r""

REGEX_MATH_SPACING: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r" *(\\[a-zA-Z]+(?![a-zA-Z{}^_])) *"),
        r" \1 ",
    ),  # "\mathbf{Text} $\abc\def\ghi\frac{a}{b}$" -> "\mathbf{Text} $ \abc \def \ghi \frac{a}{b}$"
    (re.compile(r" *= *"), r" = "),  # "a=b" -> "a = b"
    (
        re.compile(
            r"\{\s*([\\a-zA-Z_^()\[\]+/%-]+(?: +[\\a-zA-Z_^()\[\]+/%-]+)*)\s*\}"
        ),
        r"{\1}",
    ),  # "\mathbf{Text} $ \abc \def \ghi \frac{a}{b}$" -> "\mathbf{ Text } $ \abc \def \ghi \frac{ a }{ b }$"
]

REGEX_NEWLINES_AND_SPACING: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\\n(?!(?:ot|ew|ode|u|eq|exists|ewpage|oindent|atural|eg|earrow|warrow|abla|obreak|otag)(?![a-zA-Z]))"
        ),
        " ",
    ),
    (re.compile(r" *\\\, *"), " "),
    (re.compile(r" *\\\; *"), " "),
    (re.compile(r" *\\\: *"), " "),
    (re.compile(r" *\\quad *"), " "),
    (re.compile(r" *\\qquad *"), " "),
    (re.compile(r" {2,}"), " "),
    (re.compile(r"(?<![ \t\n`'(\[{\~])(?=\$)"), " "),
    (re.compile(r"(?<=\$)(?![ \t\n.,;:?!)\]}''-])"), " "),
]


def replace_latex_accents(text: str) -> str:
    """Converts LaTeX accent macros to their unicode equivalents."""

    def repl(match):
        accent = match.group(1)
        letter = match.group(2)
        return accent_latex_mispellings.get((accent, letter), match.group(0))

    return accent_latex_mispellings_regex.sub(repl, text)


def replace_env_balanced(
    text: str,
    opener: str,
    closer: str = "}",
    new_opener: str = "",
    new_closer: str = "",
) -> str:
    """
    Safely strips or replaces environments in LaTeX, handling nested braces.
    """
    start_search_pos = 0

    while True:
        idx_inicio = text.find(opener, start_search_pos)
        if idx_inicio == -1:
            break

        idx_contenido = idx_inicio + len(opener)
        balance = 1
        idx_fin = -1
        i = idx_contenido

        while i < len(text):
            char = text[i]
            if char == "{":
                balance += 1
            elif char == "}":
                balance -= 1
                if balance == 0:
                    idx_fin = i
                    break
            i += 1

        if idx_fin != -1:
            parte_anterior = text[:idx_inicio]
            raw_content = text[idx_contenido:idx_fin]

            if opener.strip().endswith(("bf", "it", "sl")):
                contenido_interno = raw_content.lstrip()
            else:
                contenido_interno = raw_content

            text = (
                parte_anterior
                + new_opener
                + contenido_interno
                + new_closer
                + text[idx_fin + len(closer) :]
            )

            start_search_pos = (
                len(parte_anterior)
                + len(new_opener)
                + len(contenido_interno)
                + len(new_closer)
            )
        else:
            start_search_pos = idx_inicio + 1

    return text


def strip_latex_environments(text: str) -> str:
    for (opener, closer), (new_opener, new_closer) in REPLACEMENTS_ENVS:
        text = replace_env_balanced(text, opener, closer, new_opener, new_closer)
    return text


def apply_string_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def apply_regex_replacements(
    text: str, regex_list: list[tuple[re.Pattern, str]]
) -> str:
    for pattern, replacement in regex_list:
        text = pattern.sub(replacement, text)
    return text


def normalize_math_envs_to_dollar(text: str) -> str:
    for env, replacement in MATH_ENVS_TO_DOLLAR:
        text = text.replace(env, replacement)
    return text


def regularize_text(text: str, task_id: int | None = None) -> str:
    text = apply_string_replacements(text, ENCODING_ARTIFACTS)

    text = apply_string_replacements(text, UNICODE_ARTIFACTS)

    text = replace_latex_accents(text)

    text = strip_latex_environments(text)

    text = apply_string_replacements(text, MACRO_REPLACEMENTS)

    text = normalize_math_envs_to_dollar(text)

    text = apply_regex_replacements(text, REGEX_MATH_MACROS)

    if task_id is not None and task_id in TASK_SPECIFIC_REGEX_REPLACEMENTS:
        text = apply_regex_replacements(text, TASK_SPECIFIC_REGEX_REPLACEMENTS[task_id])

    text = apply_regex_replacements(text, REGEX_MATH_SPACING)

    text = apply_regex_replacements(text, REGEX_NEWLINES_AND_SPACING)

    return text


MULTI_SPACE_REGEX = re.compile(r"\s+")


def regularize_line(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(MULTI_SPACE_REGEX, " ", text)
    return text
