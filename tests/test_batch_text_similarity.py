from src.features.batch_text_similarity import (
    build_char_tfidf,
    pair_char_ngram_cosine,
)


def test_batch_tfidf_similarity():

    texts = [
        "williams",
        "wilblims",
        "completely different",
    ]

    _, matrix = build_char_tfidf(texts)

    scores = pair_char_ngram_cosine(
        matrix,
        matrix,
        [0, 0],
        [1, 2]
    )

    assert scores[0] > 0
    assert scores[1] >= 0
