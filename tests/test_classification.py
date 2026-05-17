from classification import classify_propaganda, classify_propaganda_ml


def test_rule_based_classifier_detects_high_rhetoric_text() -> None:
    words = (
        "glory hero heroic sacrifice duty destiny honor victory enemy traitor "
        "nation patriot revolution leader party regime liberation freedom people"
    ).split()

    result = classify_propaganda(words, " ".join(words))

    assert result["label"] == "propaganda"
    assert result["rule_based_label"] == "propaganda"
    assert result["rhetoric_score"] >= 8


def test_rule_based_classifier_accepts_neutral_academic_text() -> None:
    words = (
        "analysis source archive document evidence data method interpretation "
        "context research report bibliography chronology statistic study"
    ).split()

    result = classify_propaganda(words, " ".join(words))

    assert result["label"] == "neutral"
    assert result["neutral_count"] > result["propaganda_count"]


def test_ml_classifier_returns_comparable_result_shape() -> None:
    result = classify_propaganda_ml(
        "The heroic nation follows the leader to defeat the enemy."
    )

    assert result["label"] in {"propaganda", "neutral", "unavailable"}
    assert "method" in result
    assert "top_terms" in result
