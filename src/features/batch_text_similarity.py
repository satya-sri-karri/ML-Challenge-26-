import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_char_tfidf(texts):
    cleaned = [
        str(text).strip().lower()
        if text is not None
        else ""
        for text in texts
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1
    )

    matrix = vectorizer.fit_transform(cleaned)

    return vectorizer, matrix


def pair_char_ngram_cosine(
    source1_matrix,
    candidate_matrix,
    source1_indices,
    candidate_indices,
    chunk_size=50000
):
    """
    Calculate cosine similarity for candidate pairs in chunks.

    Because TF-IDF vectors are L2-normalized, the sparse
    dot product is equivalent to cosine similarity.
    """

    source1_indices = np.asarray(
        source1_indices,
        dtype=np.int64
    )

    candidate_indices = np.asarray(
        candidate_indices,
        dtype=np.int64
    )

    if len(source1_indices) != len(candidate_indices):
        raise ValueError(
            "source1_indices and candidate_indices "
            "must have the same length"
        )

    scores = np.empty(
        len(source1_indices),
        dtype=float
    )

    for start in range(
        0,
        len(source1_indices),
        chunk_size
    ):
        end = min(
            start + chunk_size,
            len(source1_indices)
        )

        source_rows = source1_matrix[
            source1_indices[start:end]
        ]

        candidate_rows = candidate_matrix[
            candidate_indices[start:end]
        ]

        chunk_scores = source_rows.multiply(
            candidate_rows
        ).sum(axis=1)

        scores[start:end] = np.asarray(
            chunk_scores
        ).ravel()

    return scores