from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def jaro_winkler_similarity(name1, name2):
    if not name1 or not name2:
        return 0.0

    return JaroWinkler.normalized_similarity(
        str(name1),
        str(name2)
    )


def levenshtein_similarity(name1, name2):
    if not name1 or not name2:
        return 0.0

    return fuzz.ratio(
        str(name1),
        str(name2)
    ) / 100.0


def token_set_jaccard(tokens1, tokens2):
    if tokens1 is None or tokens2 is None:
        return 0.0

    set1 = set(tokens1)
    set2 = set(tokens2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union


def containment_similarity(name1, name2):
    if not name1 or not name2:
        return 0.0

    a = str(name1).strip().lower()
    b = str(name2).strip().lower()

    if not a or not b:
        return 0.0

    if a in b or b in a:
        return 1.0

    return 0.0


def prefix_similarity(name1, name2):
    if not name1 or not name2:
        return 0.0

    a = str(name1).strip().lower()
    b = str(name2).strip().lower()

    if not a or not b:
        return 0.0

    max_length = max(len(a), len(b))

    if max_length == 0:
        return 0.0

    common_length = 0

    for char1, char2 in zip(a, b):
        if char1 != char2:
            break
        common_length += 1

    return common_length / max_length


def phonetic_code(value):
    """
    Simple Soundex-style phonetic code.

    Used only as a lightweight blocking/matching feature.
    """

    if not value:
        return ""

    text = "".join(
        char for char in str(value).upper()
        if char.isalpha()
    )

    if not text:
        return ""

    first = text[0]

    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2",
        "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    digits = []

    previous = mapping.get(first, "")

    for char in text[1:]:
        code = mapping.get(char, "")

        if code and code != previous:
            digits.append(code)

        previous = code

    digits = "".join(digits)

    return (first + digits + "000")[:4]


def phonetic_match(name1, name2):
    if not name1 or not name2:
        return 0.0

    code1 = phonetic_code(name1)
    code2 = phonetic_code(name2)

    if not code1 or not code2:
        return 0.0

    return float(code1 == code2)
