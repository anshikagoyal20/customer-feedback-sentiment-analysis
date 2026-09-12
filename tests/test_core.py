from datetime import date

import pandas as pd

from src.nlp.text_preprocessing import clean_text, prepare_model_text


def test_text_cleaning_removes_urls_and_preserves_negation():
    assert clean_text("Not good! Visit https://example.com") == "not good visit"
    assert "not" in prepare_model_text("Bad", "This is not useful")


def test_rating_derived_labels_are_expected():
    ratings = pd.Series([1, 2, 3, 4, 5])
    labels = ratings.map(lambda rating: "Negative" if rating in (1, 2) else "Neutral" if rating == 3 else "Positive")
    assert labels.tolist() == ["Negative", "Negative", "Neutral", "Positive", "Positive"]


def test_date_value_is_serializable_for_storage():
    parsed = pd.to_datetime("2019-03-25T00:00:00.000Z").date()
    assert parsed == date(2019, 3, 25)