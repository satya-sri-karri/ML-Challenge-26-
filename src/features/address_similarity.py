from rapidfuzz import fuzz


def address_token_jaccard(tokens1, tokens2):
    if tokens1 is None or tokens2 is None:
        return 0.0

    set1 = set(tokens1)
    set2 = set(tokens2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union


def numeric_token_tolerant_match(numbers1, numbers2, tolerance=2):
    nums1 = set()
    nums2 = set()

    for value in numbers1 if numbers1 is not None else []:
        try:
            nums1.add(int(value))
        except (ValueError, TypeError):
            continue

    for value in numbers2 if numbers2 is not None else []:
        try:
            nums2.add(int(value))
        except (ValueError, TypeError):
            continue

    if not nums1 or not nums2:
        return 0.0

    for a in nums1:
        for b in nums2:
            if abs(a - b) <= tolerance:
                return 1.0

    return 0.0


def address_string_similarity(address1, address2):
    if not address1 or not address2:
        return 0.0

    return fuzz.ratio(
        str(address1),
        str(address2)
    ) / 100.0
