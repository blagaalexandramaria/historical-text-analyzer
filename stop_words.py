"""
Fallback stop-word list for text preprocessing.

This module is used when stop_words.json is missing or cannot be loaded.
It contains common words that are removed before NLP analysis.
"""
STOP_WORDS = {
    # articles
    "a", "an", "the",

    # pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our",
    "mine", "yours", "hers", "ours", "theirs",

    # demonstratives
    "this", "that", "these", "those",

    # auxiliary verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did", "its",

    # modal verbs
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",

    # negations
    "not", "no", "nor",

    # prepositions
    "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "out", "off",
    "over", "under", "however",

    # conjunctions
    "and", "or", "but", "if", "because", "as", "until", "while",

    # common adverbs
    "very", "too", "also", "just", "only", "even",
    "more", "most", "some", "such",

    # relative / interrogative words
    "which", "who", "whom", "whose", "what", "where", "when", "why", "how",

    # general filler words
    "there", "here", "then", "than",
    "all", "any", "both", "each", "few", "many",

    # frequent but low-value words in historical texts
    "made", "make", "making",
    "used", "use",
    "part", "parts",
    "one", "two", "first", "second",
    "new", "old",
    "much", "well",

    # artifacts from text parsing
    "s", "t"
}