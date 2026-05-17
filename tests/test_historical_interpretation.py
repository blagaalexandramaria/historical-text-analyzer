from historical_interpretation import build_historical_interpretation


def test_historical_interpretation_summarizes_differences() -> None:
    interpretation = build_historical_interpretation(
        themes1={"war": 5, "politics": 1},
        themes2={"war": 1, "politics": 4},
        unique1=[("army", 3), ("front", 2)],
        unique2=[("revolution", 4), ("party", 2)],
        years1=[("1916", 2)],
        years2=[("1917", 3)],
        classification1={
            "label": "propaganda",
            "rule_based_label": "propaganda",
            "ml_label": "propaganda",
            "rhetoric_score": 9.0,
        },
        classification2={
            "label": "neutral",
            "rule_based_label": "neutral",
            "ml_label": "neutral",
            "rhetoric_score": 2.0,
        },
        sentiment1={"label": "negative historical framing"},
        sentiment2={"label": "mixed or neutral framing"},
        similarity=25.0,
    )

    assert interpretation["dominant_theme1"] == "war"
    assert interpretation["dominant_theme2"] == "politics"
    assert interpretation["observations"]
