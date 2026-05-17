"""
Historical interpretation helpers.

The functions in this module turn numeric NLP signals into short,
human-readable observations suitable for the GUI and web report.
"""

from typing import Any


def build_historical_interpretation(
    *,
    themes1: dict[str, int],
    themes2: dict[str, int],
    unique1: list[tuple[str, int]],
    unique2: list[tuple[str, int]],
    years1: list[tuple[str, int]],
    years2: list[tuple[str, int]],
    classification1: dict[str, object],
    classification2: dict[str, object],
    sentiment1: dict[str, object],
    sentiment2: dict[str, object],
    similarity: float,
) -> dict[str, Any]:
    """
    Builds a concise interpretation of thematic and rhetorical differences.

    The output is intentionally cautious: it says what the extracted signals
    suggest, not what the texts definitively prove.
    """
    dominant_theme1, score1 = _dominant_theme(themes1)
    dominant_theme2, score2 = _dominant_theme(themes2)
    unique_terms1 = _format_terms(unique1)
    unique_terms2 = _format_terms(unique2)
    years_focus1 = _format_terms(years1, limit=5)
    years_focus2 = _format_terms(years2, limit=5)
    rhetoric1 = _format_rhetoric(classification1)
    rhetoric2 = _format_rhetoric(classification2)

    observations: list[str] = []
    if dominant_theme1 and dominant_theme2:
        if dominant_theme1 == dominant_theme2:
            observations.append(
                "Both texts are led by the same broad theme "
                f"({dominant_theme1}), which suggests a shared historical focus."
            )
        else:
            observations.append(
                "The dominant themes differ: "
                f"File 1 leans toward {dominant_theme1}, while File 2 leans "
                f"toward {dominant_theme2}."
            )

    observations.append(
        "Distinctive vocabulary suggests File 1 emphasizes "
        f"{unique_terms1}, while File 2 emphasizes {unique_terms2}."
    )
    observations.append(
        "Rhetorical framing differs as follows: "
        f"File 1 is {rhetoric1}; File 2 is {rhetoric2}."
    )
    observations.append(
        "The timeline signals focus on "
        f"{years_focus1} for File 1 and {years_focus2} for File 2."
    )

    if similarity >= 60:
        summary = (
            "The texts share a strong vocabulary base but still differ in emphasis."
        )
    elif similarity >= 30:
        summary = "The texts overlap moderately, with visible thematic differences."
    else:
        summary = "The texts use substantially different vocabularies and frames."

    return {
        "summary": summary,
        "dominant_theme1": dominant_theme1,
        "dominant_theme2": dominant_theme2,
        "dominant_theme_score1": score1,
        "dominant_theme_score2": score2,
        "vocabulary_focus1": unique_terms1,
        "vocabulary_focus2": unique_terms2,
        "timeline_focus1": years_focus1,
        "timeline_focus2": years_focus2,
        "rhetorical_framing1": rhetoric1,
        "rhetorical_framing2": rhetoric2,
        "sentiment1": sentiment1.get("label", "mixed or neutral framing"),
        "sentiment2": sentiment2.get("label", "mixed or neutral framing"),
        "observations": observations,
    }


def _dominant_theme(themes: dict[str, int]) -> tuple[str, int]:
    """Returns the highest-scoring theme and its score."""
    if not themes:
        return "no dominant theme", 0
    theme, score = max(themes.items(), key=lambda item: item[1])
    if score <= 0:
        return "no dominant theme", 0
    return theme, score


def _format_terms(items: list[tuple[str, int]], limit: int = 5) -> str:
    """Formats ranked terms for a short report sentence."""
    terms = [term for term, _ in items[:limit]]
    if not terms:
        return "no strong extracted terms"
    return ", ".join(terms)


def _format_rhetoric(classification: dict[str, object]) -> str:
    """Formats rule-based and ML propaganda signals side by side."""
    rule_label = classification.get("rule_based_label") or classification.get("label")
    ml_label = classification.get("ml_label", "unavailable")
    rhetoric_score = classification.get("rhetoric_score", 0)
    if isinstance(rhetoric_score, (int, float)):
        score_text = f"{float(rhetoric_score):.2f}"
    else:
        score_text = "n/a"

    if ml_label == "unavailable":
        return f"{rule_label} by rules, with rhetoric score {score_text}"

    return (
        f"{rule_label} by rules and {ml_label} by ML, "
        f"with rhetoric score {score_text}"
    )
