"""
Module responsible for similarity and frequency analysis.

Includes:
- Jaccard similarity
- TF-IDF cosine similarity
- top/common/unique words
- extraction of relevant historical years
"""

from collections import Counter
import math


def jaccard_similarity(words1: list[str], words2: list[str]) -> float:
    """
    Computes Jaccard similarity between two texts.

    Formula:
    similarity = intersection / union

    Measures how many unique words are shared between the two texts.
    """
    set1 = set(words1)
    set2 = set(words2)

    # Avoid division by zero if both texts are empty
    if not set1 and not set2:
        return 0.0

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    return len(intersection) / len(union)


def common_words(words1: list[str], words2: list[str]) -> list[str]:
    """
    Returns all common words between the two texts, sorted alphabetically.
    """
    return sorted(set(words1).intersection(set(words2)))


def top_words(words: list[str], n: int = 10) -> list[tuple[str, int]]:
    """
    Returns the most frequent words in a text.
    """
    counter = Counter(words)
    return counter.most_common(n)


def top_common_words(words1: list[str], words2: list[str], n: int = 10) -> list[tuple[str, int]]:
    """
    Returns the most relevant common words between two texts.

    Words are ranked by combined frequency:
    frequency in text 1 + frequency in text 2
    """
    counter1 = Counter(words1)
    counter2 = Counter(words2)

    common = set(counter1).intersection(set(counter2))

    # Combine frequencies from both texts for ranking
    ranked = [(word, counter1[word] + counter2[word]) for word in common]
    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked[:n]


def unique_words(words1: list[str], words2: list[str], n: int = 10) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Returns the most frequent words that are unique to each text.

    Useful for highlighting thematic differences between documents.
    """
    counter1 = Counter(words1)
    counter2 = Counter(words2)

    only1 = [(word, freq) for word, freq in counter1.items() if word not in counter2]
    only2 = [(word, freq) for word, freq in counter2.items() if word not in counter1]

    only1.sort(key=lambda x: x[1], reverse=True)
    only2.sort(key=lambda x: x[1], reverse=True)

    return only1[:n], only2[:n]


def extract_years(words: list[str]) -> list[tuple[str, int]]:
    """
    Extracts 4-digit numbers from text tokens.

    In historical texts, these are usually years
    (for example: 1914, 1918, 1920).
    """
    years = [word for word in words if word.isdigit() and len(word) == 4]
    counter = Counter(years)
    return counter.most_common()


def tf_idf_cosine_similarity(words1: list[str], words2: list[str]) -> float:
    """
    Computes cosine similarity between two texts using a simple TF-IDF approach.

    Steps:
    - count term frequencies
    - compute inverse document frequency (IDF)
    - build TF-IDF vectors
    - compute cosine similarity

    This captures similarity more precisely than Jaccard,
    because it also considers term importance and frequency.
    """
    if not words1 and not words2:
        return 0.0

    tf1 = Counter(words1)
    tf2 = Counter(words2)

    # Vocabulary = all unique terms from both texts
    vocab = set(tf1.keys()).union(tf2.keys())

    total1 = sum(tf1.values()) or 1
    total2 = sum(tf2.values()) or 1

    # Smoothed IDF:
    # terms appearing in both texts get lower weight,
    # terms that are more specific get higher weight
    idf: dict[str, float] = {}
    for term in vocab:
        df = (1 if term in tf1 else 0) + (1 if term in tf2 else 0)
        idf[term] = math.log((2 + 1) / (df + 1)) + 1.0

    def tfidf_vector(tf: Counter[str], total: int) -> dict[str, float]:
        """
        Converts raw term counts into a TF-IDF vector.
        """
        return {term: (count / total) * idf[term] for term, count in tf.items()}

    v1 = tfidf_vector(tf1, total1)
    v2 = tfidf_vector(tf2, total2)

    # Dot product of the two vectors
    dot = sum(v1.get(term, 0.0) * v2.get(term, 0.0) for term in vocab)

    # Vector norms
    norm1 = math.sqrt(sum(val * val for val in v1.values()))
    norm2 = math.sqrt(sum(val * val for val in v2.values()))

    # Avoid division by zero
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)