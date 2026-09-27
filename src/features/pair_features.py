from src.features.similarity import (
    jaro_winkler_similarity,
    levenshtein_similarity,
    token_set_jaccard,
    containment_similarity,
    prefix_similarity,
    phonetic_match,
)

from src.features.address_similarity import (
    address_token_jaccard,
    numeric_token_tolerant_match,
    address_string_similarity,
)

from src.features.text_similarity import (
    char_ngram_cosine_similarity,
)


def build_pair_features(record1, record2):
    name1 = record1.get("name_core", "")
    name2 = record2.get("name_core", "")

    address1 = record1.get("address_clean", "")
    address2 = record2.get("address_clean", "")

    name_tokens1 = record1.get("name_tokens", [])
    name_tokens2 = record2.get("name_tokens", [])

    address_tokens1 = record1.get("address_tokens", [])
    address_tokens2 = record2.get("address_tokens", [])

    numbers1 = record1.get("address_numbers", [])
    numbers2 = record2.get("address_numbers", [])

    country1 = str(
        record1.get("country_norm", "")
    ).strip().lower()

    country2 = str(
        record2.get("country_norm", "")
    ).strip().lower()

    source2 = str(
        record2.get("entity_source", "")
    ).strip().lower()

    return {
        "name_jaro_winkler": jaro_winkler_similarity(name1, name2),

        "name_levenshtein": levenshtein_similarity(
            name1, name2
        ),

        "name_token_jaccard": token_set_jaccard(
            name_tokens1, name_tokens2
        ),

        "name_containment": containment_similarity(
            name1, name2
        ),

        "name_prefix_similarity": prefix_similarity(
            name1, name2
        ),

        "name_char_ngram_cosine": char_ngram_cosine_similarity(
            name1, name2
        ),

        "name_phonetic_match": phonetic_match(
            name1, name2
        ),

        "address_token_jaccard": address_token_jaccard(
            address_tokens1, address_tokens2
        ),

        "address_number_tolerant_match":
            numeric_token_tolerant_match(
                numbers1,
                numbers2
            ),

        "address_string_similarity":
            address_string_similarity(
                address1,
                address2
            ),

        "country_match": int(
            bool(country1)
            and bool(country2)
            and country1 == country2
        ),

        "name1_empty": int(
            bool(record1.get("name_is_empty", False))
        ),

        "name2_empty": int(
            bool(record2.get("name_is_empty", False))
        ),

        "address1_empty": int(
            bool(record1.get("address_is_empty", False))
        ),

        "address2_empty": int(
            bool(record2.get("address_is_empty", False))
        ),

        "is_source2_pair": int(
            source2 in {"s2", "source2"}
        ),

        "is_source3_pair": int(
            source2 in {"s3", "source3"}
        ),
    }
