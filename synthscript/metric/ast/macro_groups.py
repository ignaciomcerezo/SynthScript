# different spellings for one single symbol
# macro synonym: macro to be replaced with
MACRO_ALIASES = {
    "to": "rightarrow",
    "lnot": "neg",
    "gets": "leftarrow",
    "le": "leq",
    "ge": "geq",
    "ne": "neq",
    "lor": "vee",
    "land": "wedge",
    "operatorname": "mathrm",
    "text": "mathrm",
    # TODO: expand this list
}

# their argument is left in the comparison tree.
TRANSPARENT_MACROS = {
    "mathop",
    # "mathit",
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
