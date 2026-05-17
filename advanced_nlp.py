"""
Advanced NLP features for historical text analysis.

The project should remain usable even when heavy NLP libraries are missing.
For that reason, this module uses optional integrations first and local
fallbacks second:
- spaCy for lemmatization and named entities
- scikit-learn for TF-IDF n-grams and LDA topics
- sentence-transformers for semantic similarity
- YAKE for keyword extraction
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from text_processing import clean_text, load_stop_words, remove_stop_words, tokenize

POSITIVE_TERMS = {
    "victory",
    "unification",
    "freedom",
    "liberation",
    "reform",
    "progress",
    "independence",
    "success",
    "peace",
    "stability",
    "rights",
    "alliance",
    "victorie",
    "unire",
    "libertate",
    "eliberare",
    "progres",
    "pace",
}

NEGATIVE_TERMS = {
    "defeat",
    "occupation",
    "loss",
    "crisis",
    "war",
    "violence",
    "death",
    "collapse",
    "repression",
    "enemy",
    "invasion",
    "suffering",
    "conflict",
    "infrangere",
    "ocupatie",
    "pierdere",
    "criza",
    "razboi",
    "moarte",
}


KNOWN_HISTORICAL_ENTITIES = {
    "GPE": (
        "Romania",
        "Russia",
        "Germany",
        "France",
        "Britain",
        "Italy",
        "Bulgaria",
        "Serbia",
        "Transylvania",
        "Bessarabia",
        "Bucharest",
        "Moscow",
        "Petrograd",
        "St Petersburg",
        "Vienna",
    ),
    "ORG": (
        "Allied Powers",
        "Central Powers",
        "Ottoman Empire",
        "Austro-Hungarian Empire",
        "Austria-Hungary",
        "Russian Empire",
        "Romanian Army",
        "Red Army",
        "Bolshevik Party",
        "Bolsheviks",
    ),
    "NORP": (
        "Romanian",
        "Romanians",
        "Russian",
        "Russians",
        "German",
        "Germans",
        "French",
        "British",
        "Bolshevik",
    ),
}


@lru_cache(maxsize=1)
def _load_spacy_model():
    """Loads an English spaCy model if available."""
    try:
        import spacy  # type: ignore

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def lemmatize_text(text: str) -> dict[str, Any]:
    """
    Converts words to lemmas, with a lightweight fallback.

    The fallback is intentionally conservative. It handles a few common English
    inflections without pretending to be a complete lemmatizer.
    """
    nlp = _load_spacy_model()
    stop_words = load_stop_words()

    if nlp:
        doc = nlp(text)
        lemmas: list[str] = []
        for token in doc:
            lemma = token.lemma_.lower().strip()
            if not lemma or token.is_space or token.is_punct:
                continue
            if token.like_num and len(token.text) == 4:
                lemmas.append(token.text)
                continue
            if token.is_alpha and lemma not in stop_words and len(lemma) > 2:
                lemmas.append(lemma)
        return {"tokens": lemmas, "method": "spaCy lemmatization"}

    words = remove_stop_words(tokenize(clean_text(text)))
    lemmas = [_simple_lemma(word) for word in words]
    return {"tokens": lemmas, "method": "fallback rule-based lemmatization"}


def _simple_lemma(word: str) -> str:
    """Small fallback for frequent English historical-text inflections."""
    irregular = {
        "fought": "fight",
        "fighting": "fight",
        "romanians": "romanian",
        "powers": "power",
        "armies": "army",
        "countries": "country",
        "revolutions": "revolution",
    }
    if word in irregular:
        return irregular[word]
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def extract_ngrams(
    words: list[str], min_n: int = 2, max_n: int = 3, limit: int = 12
) -> list[tuple[str, int]]:
    """Extracts frequent bigrams and trigrams from normalized tokens."""
    counts: Counter[str] = Counter()
    for n in range(min_n, max_n + 1):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if any(part.isdigit() for part in phrase.split()):
                continue
            counts[phrase] += 1
    return counts.most_common(limit)


def sklearn_tfidf_cosine(raw_text1: str, raw_text2: str) -> dict[str, Any]:
    """
    Computes upgraded TF-IDF cosine similarity with n-grams and filtering.

    Uses sublinear TF scaling plus df filtering when scikit-learn is present.
    For a two-document comparison, max_df is kept at 2 so shared historical
    terms are not removed before cosine similarity is computed.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_df=2,
            min_df=1,
        )
        matrix = vectorizer.fit_transform([raw_text1, raw_text2])
        score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        return {"score": score, "method": "scikit-learn TF-IDF n-grams"}
    except Exception:
        return {"score": None, "method": "install scikit-learn for upgraded TF-IDF"}


def semantic_similarity(raw_text1: str, raw_text2: str) -> dict[str, Any]:
    """Computes semantic similarity with sentence-transformers if installed."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        try:
            model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except TypeError:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = model.encode([raw_text1, raw_text2])
        score = _cosine(vectors[0], vectors[1])
        return {"score": score, "method": "sentence-transformers all-MiniLM-L6-v2"}
    except Exception:
        return {"score": None, "method": "install sentence-transformers for embeddings"}


def _cosine(vec1: Any, vec2: Any) -> float:
    """Cosine similarity for lists or numpy-like vectors."""
    dot = float(sum(float(a) * float(b) for a, b in zip(vec1, vec2)))
    norm1 = math.sqrt(sum(float(a) * float(a) for a in vec1))
    norm2 = math.sqrt(sum(float(b) * float(b) for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def extract_entities(text: str, limit: int = 15) -> dict[str, Any]:
    """Extracts people, places, organizations, national groups and years."""
    nlp = _load_spacy_model()
    if not nlp:
        return {
            "entities": _fallback_entities(text, limit),
            "method": "fallback regex entity extraction",
        }

    doc = nlp(text)
    wanted = {"PERSON", "GPE", "LOC", "ORG", "NORP", "EVENT", "DATE"}
    counter: Counter[tuple[str, str]] = Counter()
    for ent in doc.ents:
        entity_text = ent.text.strip()
        if ent.label_ in wanted and entity_text:
            counter[(entity_text, ent.label_)] += 1

    for year in _extract_year_entities(text):
        counter[(year, "YEAR")] += 1

    entities = [
        {"text": entity_text, "label": label, "count": count}
        for (entity_text, label), count in counter.most_common(limit)
    ]
    return {"entities": entities, "method": "spaCy NER + regex years"}


def _extract_year_entities(text: str) -> list[str]:
    """Finds historical-looking 4-digit years."""
    return re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", text)


def _fallback_entities(text: str, limit: int) -> list[dict[str, Any]]:
    """Small regex fallback for historical entities when spaCy is unavailable."""
    counter: Counter[tuple[str, str]] = Counter()

    for year in _extract_year_entities(text):
        counter[(year, "YEAR")] += 1

    for label, names in KNOWN_HISTORICAL_ENTITIES.items():
        for name in names:
            pattern = r"\b" + re.escape(name).replace(r"\ ", r"\s+") + r"\b"
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                counter[(name, label)] += len(matches)

    entities = [
        {"text": entity_text, "label": label, "count": count}
        for (entity_text, label), count in counter.most_common(limit)
    ]
    return entities


def topic_modeling(
    texts: list[str], topic_count: int = 3, terms_per_topic: int = 6
) -> dict[str, Any]:
    """Discovers automatic themes with LDA when scikit-learn is available."""
    try:
        from sklearn.decomposition import LatentDirichletAllocation  # type: ignore
        from sklearn.feature_extraction.text import CountVectorizer  # type: ignore

        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        matrix = vectorizer.fit_transform(texts)
        if matrix.shape[1] == 0:
            return {"topics": [], "method": "not enough vocabulary for LDA"}

        n_components = min(
            topic_count, max(1, matrix.shape[0]), max(1, matrix.shape[1])
        )
        lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
        lda.fit(matrix)
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for index, topic in enumerate(lda.components_, start=1):
            top_indexes = topic.argsort()[-terms_per_topic:][::-1]
            topics.append(
                {
                    "topic": f"Topic {index}",
                    "terms": [str(feature_names[i]) for i in top_indexes],
                }
            )
        return {"topics": topics, "method": "scikit-learn LDA"}
    except Exception:
        return {"topics": [], "method": "install scikit-learn for LDA topic modeling"}


def analyze_sentiment(words: list[str]) -> dict[str, Any]:
    """Rule-based sentiment tuned for historical narratives."""
    counter = Counter(words)
    positive = sum(counter[word] for word in POSITIVE_TERMS if word in counter)
    negative = sum(counter[word] for word in NEGATIVE_TERMS if word in counter)
    total = max(1, positive + negative)
    score = (positive - negative) / total
    if score > 0.2:
        label = "positive historical framing"
    elif score < -0.2:
        label = "negative historical framing"
    else:
        label = "mixed or neutral framing"
    return {
        "label": label,
        "score": score,
        "positive_count": positive,
        "negative_count": negative,
    }


def extract_keywords(text: str, words: list[str], limit: int = 10) -> dict[str, Any]:
    """Extracts key phrases with YAKE, falling back to ranked n-grams."""
    try:
        import yake  # type: ignore

        extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=limit)
        keywords = [
            {"keyword": keyword, "score": round(float(score), 4)}
            for keyword, score in extractor.extract_keywords(text)
        ]
        return {"keywords": keywords, "method": "YAKE"}
    except Exception:
        ngrams = extract_ngrams(words, 1, 3, limit * 2)
        keywords = [
            {"keyword": phrase, "score": count}
            for phrase, count in ngrams
            if not re.fullmatch(r"\d+", phrase)
        ][:limit]
        return {"keywords": keywords, "method": "fallback frequency-ranked n-grams"}
