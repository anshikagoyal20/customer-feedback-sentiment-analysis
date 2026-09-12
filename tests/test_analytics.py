import pandas as pd

from src.analysis.analytics import add_percentages, find_negative_terms


def test_analytics_percentages_use_counts():
    frames = {
        "overall": pd.DataFrame({"sentiment": ["Negative", "Positive"], "review_count": [2, 8]}),
        "category": pd.DataFrame({"category": ["A"], "review_count": [10], "negative_count": [2], "neutral_count": [1], "positive_count": [7]}),
        "brand": pd.DataFrame({"brand": ["B"], "review_count": [10], "negative_count": [2], "neutral_count": [1], "positive_count": [7]}),
        "product": pd.DataFrame({"product": ["P"], "review_count": [10], "negative_count": [2], "neutral_count": [1], "positive_count": [7]}),
        "monthly": pd.DataFrame({"month": ["2020-01"], "review_count": [10], "negative_count": [2], "neutral_count": [1], "positive_count": [7]}),
    }
    result = add_percentages(frames)
    assert result["overall"].set_index("sentiment").loc["Positive", "percentage"] == 80
    assert result["category"].loc[0, "negative_percentage"] == 20


def test_negative_terms_reuse_cleaning_and_count_phrases():
    terms = find_negative_terms(["Very poor product", "Poor product", "not good product"])
    assert "poor" in terms["term"].tolist()
    assert "poor product" in terms["term"].tolist()


def test_negative_terms_support_zero_one_and_two_documents():
    assert find_negative_terms([]).empty
    assert "poor" in find_negative_terms(["Very poor product"])["term"].tolist()
    assert not find_negative_terms(["Very poor product", "Poor product"]).empty