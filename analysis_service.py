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

from classification import classify_propaganda
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
    # clean text -> tokenize -> remove stop words
    words = remove_stop_words(tokenize(clean_text(raw_text)))

    stats = {
        "words": words,
        "top": top_words(words, n=10),
        "years": extract_years(words),
        "themes": analyze_themes(words),
        "classification": classify_propaganda(words),
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

    return {
        "words1": words1,
        "words2": words2,
        "similarity": similarity,
        "tfidf_similarity": tfidf_similarity,
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
    }