"""
Module responsible for orchestrating the full NLP analysis pipeline.

This file combines:
- text preprocessing
- similarity metrics
- year extraction
- theme analysis
- propaganda classification

It also includes a small LRU-style cache to avoid recomputing
results for the same input text.
"""

from collections import OrderedDict
from typing import Any

from advanced_nlp import (
    analyze_sentiment,
    extract_entities,
    extract_keywords,
    extract_ngrams,
    lemmatize_text,
    semantic_similarity,
    sklearn_tfidf_cosine,
    topic_modeling,
)
from classification import classify_propaganda
from historical_interpretation import build_historical_interpretation
from similarity import (
    extract_years,
    jaccard_similarity,
    tf_idf_cosine_similarity,
    top_common_words,
    top_words,
    unique_words,
)
from text_processing import clean_text, remove_stop_words, tokenize
from theme_analysis import analyze_themes

# Maximum number of cached text analyses
_CACHE_LIMIT = 32

# Stores recent analysis results:
# key = text hash, value = computed statistics
_text_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def _cache_get(key: str) -> dict[str, Any] | None:
    """
    Retrieves cached analysis result if available.

    Also moves the item to the end of the OrderedDict,
    so it behaves like an LRU (least recently used) cache.
    """
    if key in _text_cache:
        _text_cache.move_to_end(key)
        return _text_cache[key]
    return None


def _cache_set(key: str, value: dict[str, Any]) -> None:
    """
    Stores a new analysis result in cache.

    If cache exceeds the maximum size, removes the oldest item.
    """
    _text_cache[key] = value
    _text_cache.move_to_end(key)

    # Remove oldest cached result if limit is exceeded
    if len(_text_cache) > _CACHE_LIMIT:
        _text_cache.popitem(last=False)


def _hash_text(text: str) -> str:
    """
    Generates a stable SHA-256 hash for raw input text.

    Used as cache key so repeated analyses of the same text
    do not need to be recomputed.
    """
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _analyze_single_text(raw_text: str) -> dict[str, Any]:
    """
    Runs the full analysis pipeline for a single text.

    Steps:
    - check cache
    - preprocess text
    - compute top words
    - extract years
    - analyze themes
    - classify propaganda patterns
    """
    cache_key = _hash_text(raw_text)
    cached = _cache_get(cache_key)

    # Return cached result if already analyzed
    if cached:
        return cached

    # Preprocessing pipeline:
    # clean text -> tokenize -> remove stop words -> lemmatize when available
    words = remove_stop_words(tokenize(clean_text(raw_text)))
    lemma_data = lemmatize_text(raw_text)
    lemma_words = lemma_data["tokens"] or words

    stats = {
        "words": lemma_words,
        "original_words": words,
        "lemmatization_method": lemma_data["method"],
        "top": top_words(lemma_words, n=10),
        "years": extract_years(words),
        "themes": analyze_themes(lemma_words),
        "classification": classify_propaganda(lemma_words, raw_text),
        "ngrams": extract_ngrams(lemma_words),
        "entities": extract_entities(raw_text),
        "sentiment": analyze_sentiment(lemma_words),
        "keywords": extract_keywords(raw_text, lemma_words),
    }

    # Save analysis in cache for future reuse
    _cache_set(cache_key, stats)
    return stats


def analyze_raw_texts(raw_text1: str, raw_text2: str) -> dict[str, Any]:
    """
    Compares two raw historical texts and returns a full analysis report.

    Includes:
    - Jaccard similarity
    - TF-IDF cosine similarity
    - common and unique words
    - key years
    - theme distributions
    - propaganda classification
    """
    stats1 = _analyze_single_text(raw_text1)
    stats2 = _analyze_single_text(raw_text2)

    words1 = stats1["words"]
    words2 = stats2["words"]

    # Similarity scores are converted to percentages for display
    similarity = jaccard_similarity(words1, words2) * 100
    tfidf_similarity = tf_idf_cosine_similarity(words1, words2) * 100
    advanced_tfidf = sklearn_tfidf_cosine(raw_text1, raw_text2)
    semantic = semantic_similarity(raw_text1, raw_text2)
    topics = topic_modeling([raw_text1, raw_text2])

    # Simple alphabetical list of all common words
    common = sorted(set(words1).intersection(set(words2)))

    # Most important terms
    top1 = stats1["top"]
    top2 = stats2["top"]
    top_common = top_common_words(words1, words2, n=10)
    unique1, unique2 = unique_words(words1, words2, n=10)

    # Historical timeline extraction
    years1 = stats1["years"]
    years2 = stats2["years"]

    # Theme distributions
    themes1 = stats1["themes"]
    themes2 = stats2["themes"]

    # Propaganda classification
    classification1 = stats1["classification"]
    classification2 = stats2["classification"]
    historical_interpretation = build_historical_interpretation(
        themes1=themes1,
        themes2=themes2,
        unique1=unique1,
        unique2=unique2,
        years1=years1,
        years2=years2,
        classification1=classification1,
        classification2=classification2,
        sentiment1=stats1["sentiment"],
        sentiment2=stats2["sentiment"],
        similarity=similarity,
    )

    return {
        "words1": words1,
        "words2": words2,
        "similarity": similarity,
        "tfidf_similarity": tfidf_similarity,
        "advanced_tfidf_similarity": (
            advanced_tfidf["score"] * 100
            if advanced_tfidf["score"] is not None
            else None
        ),
        "advanced_tfidf_method": advanced_tfidf["method"],
        "semantic_similarity": (
            semantic["score"] * 100 if semantic["score"] is not None else None
        ),
        "semantic_method": semantic["method"],
        "common": common,
        "top1": top1,
        "top2": top2,
        "top_common": top_common,
        "unique1": unique1,
        "unique2": unique2,
        "years1": years1,
        "years2": years2,
        "themes1": themes1,
        "themes2": themes2,
        "classification1": classification1,
        "classification2": classification2,
        "lemmatization_method1": stats1["lemmatization_method"],
        "lemmatization_method2": stats2["lemmatization_method"],
        "ngrams1": stats1["ngrams"],
        "ngrams2": stats2["ngrams"],
        "entities1": stats1["entities"],
        "entities2": stats2["entities"],
        "sentiment1": stats1["sentiment"],
        "sentiment2": stats2["sentiment"],
        "keywords1": stats1["keywords"],
        "keywords2": stats2["keywords"],
        "topics": topics,
        "historical_interpretation": historical_interpretation,
    }
