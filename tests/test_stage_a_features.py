from src.features.similarity import (
    prefix_similarity,
    phonetic_match,
)

from src.features.text_similarity import (
    char_ngram_cosine_similarity,
)

from src.features.pair_features import build_pair_features


def test_prefix_similarity():
    assert prefix_similarity(
        "atlantic",
        "atlantic pinnacle deli"
    ) > 0


def test_phonetic_match():
    assert phonetic_match(
        "wayne",
        "wanye"
    ) == 1.0


def test_char_ngram_similarity():
    score = char_ngram_cosine_similarity(
        "williams",
        "wilblims"
    )

    assert score > 0


def test_source2_pair_flag():
    record1 = {
        "name_core": "abc",
        "address_clean": "123 main street",
        "name_tokens": ["abc"],
        "address_tokens": ["123", "main", "street"],
        "address_numbers": ["123"],
        "country_norm": "us",
        "entity_source": "S1",
    }

    record2 = {
        "name_core": "abc",
        "address_clean": "123 main street",
        "name_tokens": ["abc"],
        "address_tokens": ["123", "main", "street"],
        "address_numbers": ["123"],
        "country_norm": "us",
        "entity_source": "S2",
    }

    features = build_pair_features(
        record1,
        record2
    )

    assert features["is_source2_pair"] == 1
    assert features["is_source3_pair"] == 0
