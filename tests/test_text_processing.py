from text_processing import clean_text, remove_stop_words, tokenize


def test_clean_text_keeps_years_and_normalizes_punctuation() -> None:
    text = "Romania entered World War I in 1916; victory came later!"

    assert clean_text(text) == "romania entered world war i in 1916 victory came later"


def test_tokenize_and_remove_stop_words_keep_historical_years() -> None:
    tokens = tokenize(clean_text("The war in Romania lasted until 1918."))

    assert remove_stop_words(tokens) == ["war", "romania", "lasted", "1918"]
