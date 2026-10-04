from rapidfuzz.distance.metrics_py import levenshtein_distance


def levenshtein(s1: str, s2: str) -> float:
    """
    Calculates the levenshtein distance between two strings using rapidfuzz
    """
    return levenshtein_distance(s1, s2)


def CER(reference: str, prediction: str) -> float:
    """
    Calculates the CER between two strings, taking 'reference' as the reference
    """
    return levenshtein(reference, prediction) / len(reference)


def sCER(s1: str, s2: str) -> float:
    """
    Calculates the average CER between two strings, that is
        1/2 * (CER(s1, s2) + CER(s2, s1))
    """
    return 2 * CER(s1, s2) / len(s2)
