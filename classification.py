"""
Module responsible for propaganda detection in historical texts.

Uses a rule-based approach based on keyword frequency and ratios.
"""

from collections import Counter


# Keywords associated with propaganda language
# Includes both English and Romanian terms
PROPAGANDA_KEYWORDS = {
    # emotional / ideological language
    "glory", "hero", "heroic", "sacrifice", "duty", "destiny",
    "honor", "victory", "enemy", "traitor", "nation", "patriot",
    "revolution", "leader", "party", "regime", "liberation",
    "freedom", "people", "masses", "truth", "historic",
    "great", "strength", "victorious", "invincible",
    "struggle", "fight", "blood", "loyalty", "mission", "ideology",

    # military / mobilization language
    "war", "army", "soldiers", "regiment", "front",
    "mobilize", "mobilization",

    # Romanian equivalents
    "glorie", "erou", "eroi", "sacrificiu", "datorie",
    "destin", "onoare", "dusman", "tradator", "victorie",
    "natiune", "patrie", "revolutie", "conducator",
    "partid", "regim", "eliberare", "libertate",
    "popor", "mase", "adevar", "istoric",
    "maret", "etern", "putere", "neinvins",
    "lupta", "razboi", "soldati", "armata",
}


# Keywords associated with neutral / academic language
# Used to balance propaganda detection
NEUTRAL_KEYWORDS = {
    "analysis", "source", "archive", "document", "evidence",
    "data", "method", "interpretation", "context", "research",
    "report", "bibliography", "chronology", "statistic",
    "study", "reference",

    # Romanian equivalents
    "analiza", "sursa", "arhiva", "documente", "dovada",
    "metoda", "interpretare", "context", "cercetare",
    "raport", "bibliografie", "cronologie", "statistica",
    "studiu", "referinta",
}


def classify_propaganda(words: list[str]) -> dict[str, object]:
    """
    Classifies a text as 'propaganda' or 'neutral'.

    Uses:
    - keyword frequency
    - ratio between propaganda and neutral terms
    - density of propaganda words

    Returns a dictionary with metrics and classification label.
    """
    counter = Counter(words)
    total = sum(counter.values())

    # Count occurrences of propaganda-related terms
    propaganda_count = sum(
        counter[word] for word in PROPAGANDA_KEYWORDS if word in counter
    )

    # Count occurrences of neutral/academic terms
    neutral_count = sum(
        counter[word] for word in NEUTRAL_KEYWORDS if word in counter
    )

    # Ratio helps compare dominance of propaganda vs neutral language
    # +1 smoothing avoids division by zero
    ratio = (propaganda_count + 1) / (neutral_count + 1)

    # Density = how much of the text is propaganda-related
    density = propaganda_count / max(1, total)

    # Heuristic classification rule:
    # text is propaganda if:
    # - enough propaganda terms exist AND
    # - ratio or density passes threshold
    label = (
        "propaganda"
        if propaganda_count >= 8 and (ratio >= 1.5 or density >= 0.02)
        else "neutral"
    )

    # Extract most frequent propaganda terms
    top_terms = [
        (word, counter[word])
        for word in PROPAGANDA_KEYWORDS
        if word in counter
    ]

    # Sort descending by frequency
    top_terms.sort(key=lambda item: item[1], reverse=True)

    return {
        "label": label,
        "propaganda_count": propaganda_count,
        "neutral_count": neutral_count,
        "ratio": ratio,
        "density": density,
        "top_terms": top_terms[:10],
    }