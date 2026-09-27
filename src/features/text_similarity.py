from sklearn.feature_extraction.text import TfidfVectorizer


def char_ngram_cosine_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0

    text1 = str(text1).strip().lower()
    text2 = str(text2).strip().lower()

    if not text1 or not text2:
        return 0.0

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4)
    )

    matrix = vectorizer.fit_transform([
        text1,
        text2
    ])

    similarity = matrix[0].dot(
        matrix[1].T
    ).toarray()[0][0]

    return float(similarity)
