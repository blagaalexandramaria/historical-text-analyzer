from similarity import extract_years, jaccard_similarity, tf_idf_cosine_similarity


def test_jaccard_similarity_uses_unique_word_overlap() -> None:
    words1 = ["romania", "war", "war", "alliance"]
    words2 = ["romania", "revolution", "alliance"]

    assert jaccard_similarity(words1, words2) == 0.5


def test_extract_years_counts_repeated_four_digit_tokens() -> None:
    words = ["romania", "1914", "1918", "1918", "war", "99"]

    assert extract_years(words) == [("1918", 2), ("1914", 1)]


def test_tfidf_cosine_similarity_is_one_for_identical_terms() -> None:
    words = ["archive", "document", "romania"]

    assert tf_idf_cosine_similarity(words, words) == 1.0
