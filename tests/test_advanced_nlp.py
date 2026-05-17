import advanced_nlp


def test_fallback_entities_extract_years_places_and_organizations(monkeypatch) -> None:
    monkeypatch.setattr(advanced_nlp, "_load_spacy_model", lambda: None)

    result = advanced_nlp.extract_entities(
        "Romania entered the war in 1916 with the Allied Powers."
    )
    entities = {(item["text"], item["label"]) for item in result["entities"]}

    assert ("1916", "YEAR") in entities
    assert ("Romania", "GPE") in entities
    assert ("Allied Powers", "ORG") in entities
