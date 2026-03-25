"""
Fallback theme definitions for historical text analysis.

This module is used when themes.json is missing or can not be loaded.
Each theme contains a set of keywords used for thematic sorting.
"""
THEMES = {
    "war": {"war", "battle", "army", "front", "military", "offensive", "troops"},
    "politics": {"government", "political", "state", "power", "parliament", "regime"},
    "revolution": {"revolution", "bolsheviks", "soviet", "rebellion", "mutiny"},
    "nationalism": {"nation", "national", "romanian", "romanians", "hungarian", "transylvania"},
    "diplomacy": {"treaty", "alliance", "peace", "conference", "negotiations"}
}