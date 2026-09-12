"""Create dataset-backed figures and summary tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_figures(reviews: pd.DataFrame, predictions: pd.DataFrame, output_dir: str | Path, top_terms_path: str | Path = "outputs/reports/top_terms.csv") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    _bar(reviews["sentiment"].value_counts().reindex(["Negative", "Neutral", "Positive"]), "Overall Sentiment Distribution", "Sentiment", output_dir / "sentiment_distribution.png")
    _bar(reviews["rating"].value_counts().sort_index(), "Rating Distribution", "Rating", output_dir / "rating_distribution.png")

    category = pd.crosstab(reviews["category"], reviews["sentiment"]).reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
    category = category.loc[category.sum(axis=1).sort_values(ascending=False).index]
    category.head(12).plot(kind="bar", stacked=True, figsize=(12, 6), color=["#d95f02", "#7570b3", "#1b9e77"])
    plt.title("Sentiment by Primary Category (Top 12 by Volume)")
    plt.xlabel("Primary category")
    plt.ylabel("Review count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_by_category.png", dpi=150)
    plt.close()

    monthly = reviews.assign(month=pd.to_datetime(reviews["review_date"]).dt.to_period("M")).pivot_table(index="month", columns="sentiment", values="review_id", aggfunc="count", fill_value=0).reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
    monthly.plot(figsize=(12, 6), color=["#d95f02", "#7570b3", "#1b9e77"])
    plt.title("Sentiment Over Time")
    plt.xlabel("Review month")
    plt.ylabel("Review count")
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_over_time.png", dpi=150)
    plt.close()

    pd.crosstab(reviews["rating"], reviews["sentiment"]).reindex(index=[1, 2, 3, 4, 5], columns=["Negative", "Neutral", "Positive"], fill_value=0).plot(kind="bar", figsize=(10, 6), color=["#d95f02", "#7570b3", "#1b9e77"])
    plt.title("Rating vs Derived Sentiment")
    plt.xlabel("Rating")
    plt.ylabel("Review count")
    plt.tight_layout()
    plt.savefig(output_dir / "rating_vs_sentiment.png", dpi=150)
    plt.close()

    matrix = pd.crosstab(predictions["sentiment"], predictions["predicted_sentiment"]).reindex(index=["Negative", "Neutral", "Positive"], columns=["Negative", "Neutral", "Positive"], fill_value=0)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted sentiment")
    plt.ylabel("Actual sentiment")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    top_terms = pd.read_csv(top_terms_path)
    top_terms = top_terms[top_terms["sentiment"].isin(["Negative", "Neutral", "Positive"])].copy()
    top_terms["term_label"] = top_terms["sentiment"] + ": " + top_terms["term"]
    top_terms.sort_values("coefficient").tail(30).plot(kind="barh", x="term_label", y="coefficient", figsize=(10, 9), legend=False, color="#2c7fb8")
    plt.title("Top TF-IDF Terms by Sentiment Class")
    plt.xlabel("Logistic Regression coefficient")
    plt.ylabel("Sentiment and term")
    plt.tight_layout()
    plt.savefig(output_dir / "top_terms.png", dpi=150)
    plt.close()


def _bar(values: pd.Series, title: str, xlabel: str, path: Path) -> None:
    values.plot(kind="bar", figsize=(8, 5), color="#2c7fb8")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Review count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_analysis_tables(reviews: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    category = reviews.groupby("category").agg(review_count=("review_id", "count"), average_rating=("rating", "mean"), negative_count=("sentiment", lambda value: (value == "Negative").sum())).reset_index()
    category["negative_share"] = category["negative_count"] / category["review_count"]
    category.query("review_count >= 30").sort_values(["negative_share", "review_count"], ascending=[False, False]).to_csv(output_dir / "category_summary.csv", index=False)
    brand = reviews.groupby("brand").agg(review_count=("review_id", "count"), average_rating=("rating", "mean"), negative_count=("sentiment", lambda value: (value == "Negative").sum())).reset_index()
    brand["negative_share"] = brand["negative_count"] / brand["review_count"]
    brand.query("review_count >= 30").sort_values("negative_share", ascending=False).to_csv(output_dir / "brand_summary.csv", index=False)
    reviews.assign(month=pd.to_datetime(reviews["review_date"]).dt.to_period("M").astype(str)).groupby(["month", "sentiment"]).size().unstack(fill_value=0).to_csv(output_dir / "monthly_sentiment.csv")