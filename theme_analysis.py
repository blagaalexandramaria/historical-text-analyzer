"""
Module responsible for thematic analysis of historical texts.

Assigns scores to predefined themes (e.g., war, politics)
based on keyword frequency.
"""

from collections import Counter
import json
import os

from themes import THEMES


# Get absolute path to themes.json
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
THEMES_PATH = os.path.join(ROOT_DIR, "themes.json")


def load_themes() -> dict[str, set[str]]:
    """
    Loads themes and their associated keywords from JSON file.

    Falls back to hardcoded THEMES if file cannot be read.
    """
    try:
        with open(THEMES_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

            # Convert keyword lists to sets for faster lookup
            return {theme: set(map(str, keywords)) for theme, keywords in data.items()}
    except Exception:
        # Fallback if JSON is missing or invalid
        return THEMES


def analyze_themes(words: list[str]) -> dict[str, int]:
    """
    Computes a score for each theme based on keyword frequency.

    For each theme:
    - count how many times its keywords appear in the text

    Returns:
    dictionary of theme → score
    """
    counter = Counter(words)
    result: dict[str, int] = {}

    for theme, keywords in load_themes().items():

        # Score = total frequency of all keywords belonging to that theme
        score = sum(counter[word] for word in keywords if word in counter)

        result[theme] = score

    return result