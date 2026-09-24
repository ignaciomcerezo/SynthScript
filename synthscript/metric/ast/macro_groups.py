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

# their argument is left in the comparison tree.
TRANSPARENT_MACROS = {
    # "mathit",
}

# removed during canonization
CONTENT_INDEPENDENT_MACROS = {
    "mathop",
    "quad",
    "qquad",
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

# they differ in display style but have the same structure.
FRACTION_MACROS = {
    "frac",
    "dfrac",
    "tfrac",
}

SPACING = {"quad", "qquad", "!", ",", ";", ":"}

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

# style declarations
STYLE_DECLARATIONS = {
    "cal": "mathcal",
    "bf": "mathbf",
    "it": "mathit",
    "mit": "mathit",
    "rm": "mathrm",
    "sf": "mathsf",
    "tt": "mathtt",
}
