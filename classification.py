"""
Module responsible for propaganda detection in historical texts.

Uses a rule-based approach based on keyword frequency and ratios.
"""

from collections import Counter
from functools import lru_cache
from typing import Any

# Keywords associated with propaganda language
# Includes both English and Romanian terms
PROPAGANDA_KEYWORDS = {
    # emotional / ideological language
    "glory",
    "hero",
    "heroic",
    "sacrifice",
    "duty",
    "destiny",
    "honor",
    "victory",
    "enemy",
    "traitor",
    "nation",
    "patriot",
    "revolution",
    "leader",
    "party",
    "regime",
    "liberation",
    "freedom",
    "people",
    "masses",
    "truth",
    "historic",
    "great",
    "strength",
    "victorious",
    "invincible",
    "struggle",
    "fight",
    "blood",
    "loyalty",
    "mission",
    "ideology",
    # military / mobilization language
    "war",
    "army",
    "soldiers",
    "regiment",
    "front",
    "mobilize",
    "mobilization",
    # Romanian equivalents
    "glorie",
    "erou",
    "eroi",
    "sacrificiu",
    "datorie",
    "destin",
    "onoare",
    "dusman",
    "tradator",
    "victorie",
    "natiune",
    "patrie",
    "revolutie",
    "conducator",
    "partid",
    "regim",
    "eliberare",
    "libertate",
    "popor",
    "mase",
    "adevar",
    "istoric",
    "maret",
    "etern",
    "putere",
    "neinvins",
    "lupta",
    "razboi",
    "soldati",
    "armata",
}


ML_TRAINING_EXAMPLES = (
    (
        "The glorious nation must unite behind the leader and defeat the enemy.",
        "propaganda",
    ),
    (
        "Heroic soldiers sacrifice their blood for victory, honor and destiny.",
        "propaganda",
    ),
    (
        "The party and the people march together in an historic struggle.",
        "propaganda",
    ),
    (
        "Traitors and oppressors threaten our sacred mission of liberation.",
        "propaganda",
    ),
    (
        "All citizens must fight for freedom and the invincible army.",
        "propaganda",
    ),
    (
        "Poporul trebuie sa urmeze conducatorul pentru victorie si glorie.",
        "propaganda",
    ),
    (
        "Natiunea mareata lupta impotriva dusmanului si tradatorilor.",
        "propaganda",
    ),
    (
        "Armata eroica are o misiune istorica de eliberare a patriei.",
        "propaganda",
    ),
    (
        "The archive document compares policy changes using dated sources.",
        "neutral",
    ),
    (
        "The study analyzes demographic data, evidence and historical context.",
        "neutral",
    ),
    (
        "Researchers describe the chronology of events and cite bibliography.",
        "neutral",
    ),
    (
        "The report summarizes documents from several archives and statistics.",
        "neutral",
    ),
    (
        "This interpretation compares economic reforms and administrative data.",
        "neutral",
    ),
    (
        "Analiza surselor prezinta contextul, metoda si cronologia evenimentelor.",
        "neutral",
    ),
    (
        "Raportul foloseste documente, date si dovezi pentru interpretare.",
        "neutral",
    ),
    (
        "Studiul compara referinte istorice fara apeluri emotionale.",
        "neutral",
    ),
)


INTENSIFIER_KEYWORDS = {
    "glorious",
    "tragic",
    "eternal",
    "sacred",
    "heroic",
    "evil",
    "pure",
    "great",
    "historic",
    "invincible",
    "maret",
    "tragic",
    "etern",
    "sacru",
}


GENERALIZATION_KEYWORDS = {
    "all",
    "never",
    "always",
    "everyone",
    "nobody",
    "entire",
    "whole",
    "tot",
    "toti",
    "niciodata",
    "mereu",
    "intotdeauna",
}


POLARIZATION_KEYWORDS = {
    "enemy",
    "traitor",
    "betray",
    "liberator",
    "oppressor",
    "tyranny",
    "dusman",
    "tradator",
    "asupritor",
    "eliberator",
    "tiranie",
}


# Keywords associated with neutral / academic language
# Used to balance propaganda detection
NEUTRAL_KEYWORDS = {
    "analysis",
    "source",
    "archive",
    "document",
    "evidence",
    "data",
    "method",
    "interpretation",
    "context",
    "research",
    "report",
    "bibliography",
    "chronology",
    "statistic",
    "study",
    "reference",
    # Romanian equivalents
    "analiza",
    "sursa",
    "arhiva",
    "documente",
    "dovada",
    "metoda",
    "interpretare",
    "context",
    "cercetare",
    "raport",
    "bibliografie",
    "cronologie",
    "statistica",
    "studiu",
    "referinta",
}


@lru_cache(maxsize=1)
def _train_ml_classifier() -> tuple[Any | None, str]:
    """Trains a small supervised TF-IDF classifier for portfolio comparison."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
    except Exception as exc:
        return (
            None,
            f"install scikit-learn for ML classifier ({exc.__class__.__name__})",
        )

    texts = [text for text, _ in ML_TRAINING_EXAMPLES]
    labels = [label for _, label in ML_TRAINING_EXAMPLES]
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline, "TF-IDF + Logistic Regression trained on demo examples"


def classify_propaganda_ml(text: str, limit: int = 8) -> dict[str, object]:
    """
    Classifies propaganda with a small supervised model.

    The model is intentionally lightweight: it demonstrates an ML baseline
    that can be compared with the transparent rule-based classifier.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return {
            "label": "unavailable",
            "confidence": None,
            "method": "not enough text for ML classification",
            "top_terms": [],
        }

    pipeline, method = _train_ml_classifier()
    if pipeline is None:
        return {
            "label": "unavailable",
            "confidence": None,
            "method": method,
            "top_terms": [],
        }

    prediction = str(pipeline.predict([cleaned_text])[0])
    probabilities = pipeline.predict_proba([cleaned_text])[0]
    classes = [str(label) for label in pipeline.named_steps["model"].classes_]
    confidence = float(probabilities[classes.index(prediction)])

    return {
        "label": prediction,
        "confidence": confidence,
        "method": method,
        "top_terms": _top_ml_terms(pipeline, cleaned_text, prediction, limit),
    }


def _top_ml_terms(
    pipeline: Any,
    text: str,
    predicted_label: str,
    limit: int,
) -> list[dict[str, object]]:
    """Returns input terms that contributed most to the predicted ML label."""
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    classes = [str(label) for label in model.classes_]

    if len(classes) == 2:
        positive_class = classes[1]
        direction = 1 if predicted_label == positive_class else -1
        weights = model.coef_[0] * direction
    else:
        class_index = classes.index(predicted_label)
        weights = model.coef_[class_index]

    row = matrix[0]
    scored_terms: list[tuple[str, float]] = []
    for feature_index, value in zip(row.indices, row.data):
        contribution = float(value * weights[feature_index])
        if contribution > 0:
            scored_terms.append((str(feature_names[feature_index]), contribution))

    scored_terms.sort(key=lambda item: item[1], reverse=True)
    return [
        {"term": term, "weight": round(weight, 4)}
        for term, weight in scored_terms[:limit]
    ]


def classify_propaganda(
    words: list[str],
    raw_text: str | None = None,
) -> dict[str, object]:
    """
    Classifies a text as 'propaganda' or 'neutral'.

    Uses:
    - keyword frequency
    - ratio between propaganda and neutral terms
    - density of propaganda words

    Returns a dictionary with metrics and both rule-based and ML labels.
    """
    counter = Counter(words)
    total = sum(counter.values())

    # Count occurrences of propaganda-related terms
    propaganda_count = sum(
        counter[word] for word in PROPAGANDA_KEYWORDS if word in counter
    )
    intensifier_count = sum(
        counter[word] for word in INTENSIFIER_KEYWORDS if word in counter
    )
    generalization_count = sum(
        counter[word] for word in GENERALIZATION_KEYWORDS if word in counter
    )
    polarization_count = sum(
        counter[word] for word in POLARIZATION_KEYWORDS if word in counter
    )

    # Count occurrences of neutral/academic terms
    neutral_count = sum(counter[word] for word in NEUTRAL_KEYWORDS if word in counter)

    # Ratio helps compare dominance of propaganda vs neutral language
    # +1 smoothing avoids division by zero
    ratio = (propaganda_count + 1) / (neutral_count + 1)

    # Density = how much of the text is propaganda-related
    density = propaganda_count / max(1, total)
    rhetoric_score = (
        propaganda_count
        + (intensifier_count * 1.25)
        + (generalization_count * 1.5)
        + (polarization_count * 1.5)
    )

    # Heuristic classification rule:
    # text is propaganda if:
    # - enough propaganda terms exist AND
    # - ratio or density passes threshold
    label = (
        "propaganda"
        if rhetoric_score >= 8 and (ratio >= 1.5 or density >= 0.02)
        else "neutral"
    )

    # Extract most frequent propaganda terms
    top_terms = [
        (word, counter[word]) for word in PROPAGANDA_KEYWORDS if word in counter
    ]

    # Sort descending by frequency
    top_terms.sort(key=lambda item: item[1], reverse=True)
    ml_result = classify_propaganda_ml(raw_text or " ".join(words))
    ml_label = ml_result["label"]
    model_agreement = ml_label == label if ml_label != "unavailable" else None

    return {
        "label": label,
        "rule_based_label": label,
        "propaganda_count": propaganda_count,
        "neutral_count": neutral_count,
        "ratio": ratio,
        "density": density,
        "rhetoric_score": rhetoric_score,
        "intensifier_count": intensifier_count,
        "generalization_count": generalization_count,
        "polarization_count": polarization_count,
        "top_terms": top_terms[:10],
        "ml_label": ml_label,
        "ml_confidence": ml_result["confidence"],
        "ml_method": ml_result["method"],
        "ml_top_terms": ml_result["top_terms"],
        "model_agreement": model_agreement,
    }
