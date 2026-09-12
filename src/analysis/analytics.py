"""Database-backed business analytics for stored review predictions."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from src.database.config import mysql_config
from src.nlp.text_preprocessing import prepare_model_text


MIN_GROUP_REVIEWS = 30
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]


def fetch_analytics() -> dict[str, pd.DataFrame]:
    """Run aggregations in MySQL and return complete result sets as DataFrames."""
    import mysql.connector

    queries = {
        "overall": """
            SELECT sentiment, COUNT(*) AS review_count
            FROM review_predictions
            GROUP BY sentiment
        """,
        "category": """
            SELECT category, COUNT(*) AS review_count,
                   SUM(sentiment = 'Negative') AS negative_count,
                   SUM(sentiment = 'Neutral') AS neutral_count,
                   SUM(sentiment = 'Positive') AS positive_count,
                   ROUND(AVG(rating), 2) AS average_rating
            FROM review_predictions
            GROUP BY category
            HAVING COUNT(*) >= %s
            ORDER BY negative_count / review_count DESC
        """,
        "brand": """
            SELECT brand, COUNT(*) AS review_count,
                   SUM(sentiment = 'Negative') AS negative_count,
                   SUM(sentiment = 'Neutral') AS neutral_count,
                   SUM(sentiment = 'Positive') AS positive_count,
                   ROUND(AVG(rating), 2) AS average_rating
            FROM review_predictions
            GROUP BY brand
            HAVING COUNT(*) >= %s
            ORDER BY negative_count / review_count DESC
        """,
        "product": """
            SELECT product, brand, category, COUNT(*) AS review_count,
                   SUM(sentiment = 'Negative') AS negative_count,
                   SUM(sentiment = 'Neutral') AS neutral_count,
                   SUM(sentiment = 'Positive') AS positive_count,
                   ROUND(AVG(rating), 2) AS average_rating
            FROM review_predictions
            GROUP BY product, brand, category
            HAVING COUNT(*) >= %s
            ORDER BY negative_count / review_count DESC
        """,
        "monthly": """
            SELECT DATE_FORMAT(review_date, '%Y-%m') AS month,
                   COUNT(*) AS review_count,
                   SUM(sentiment = 'Negative') AS negative_count,
                   SUM(sentiment = 'Neutral') AS neutral_count,
                   SUM(sentiment = 'Positive') AS positive_count
            FROM review_predictions
            GROUP BY month
            HAVING COUNT(*) >= %s
            ORDER BY month
        """,
        "negative_text": """
            SELECT review_text
            FROM review_predictions
            WHERE sentiment = 'Negative' AND review_text IS NOT NULL
        """,
    }
    connection = mysql.connector.connect(**mysql_config())
    try:
        frames = {}
        for name, query in queries.items():
            cursor = connection.cursor(dictionary=True)
            parameters = (MIN_GROUP_REVIEWS,) if name in {"category", "brand", "product", "monthly"} else None
            cursor.execute(query, parameters)
            frames[name] = pd.DataFrame(cursor.fetchall())
            cursor.close()
        return frames
    finally:
        connection.close()


def add_percentages(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add percentage columns to count-based result tables."""
    overall = frames["overall"].copy()
    overall["review_count"] = pd.to_numeric(overall["review_count"])
    total = overall["review_count"].sum()
    overall["percentage"] = overall["review_count"] / total * 100
    frames["overall"] = overall
    for name in ("category", "brand", "product"):
        frame = frames[name].copy()
        for column in ("review_count", "negative_count", "neutral_count", "positive_count", "average_rating"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column])
        frame["negative_percentage"] = frame["negative_count"] / frame["review_count"] * 100
        frame["neutral_percentage"] = frame["neutral_count"] / frame["review_count"] * 100
        frame["positive_percentage"] = frame["positive_count"] / frame["review_count"] * 100
        frames[name] = frame
    monthly = frames["monthly"].copy()
    for column in ("review_count", "negative_count", "neutral_count", "positive_count"):
        monthly[column] = pd.to_numeric(monthly[column])
    monthly["negative_percentage"] = monthly["negative_count"] / monthly["review_count"] * 100
    monthly["neutral_percentage"] = monthly["neutral_count"] / monthly["review_count"] * 100
    monthly["positive_percentage"] = monthly["positive_count"] / monthly["review_count"] * 100
    frames["monthly"] = monthly
    return frames


def find_negative_terms(texts: Iterable[str], limit: int = 25) -> pd.DataFrame:
    """Find frequent unigrams/bigrams using shared cleaning and safe sample sizing."""
    cleaned = [prepare_model_text("", text) for text in texts if text]
    if not cleaned:
        return pd.DataFrame(columns=["term", "count"])

    # Requiring two documents is useful for the overall corpus but invalid for a
    # product with one negative review. Keep the same tokenizer and n-grams while
    # adapting only the document-frequency threshold to the sample size.
    min_document_frequency = 2 if len(cleaned) >= 2 else 1
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=min_document_frequency,
        max_df=1.0,
        max_features=5000,
    )
    if not any(vectorizer.build_analyzer()(text) for text in cleaned):
        return pd.DataFrame(columns=["term", "count"])
    matrix = vectorizer.fit_transform(cleaned)
    counts = matrix.sum(axis=0).A1
    terms = pd.DataFrame({"term": vectorizer.get_feature_names_out(), "count": counts})
    return terms.sort_values(["count", "term"], ascending=[False, True]).head(limit).reset_index(drop=True)


def write_reports(frames: dict[str, pd.DataFrame], negative_terms: pd.DataFrame, report_dir: str | Path) -> None:
    """Write machine-readable tables and a concise business insight report."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(report_dir / f"analytics_{name}.csv", index=False)
    negative_terms.to_csv(report_dir / "analytics_negative_terms.csv", index=False)

    overall = frames["overall"].set_index("sentiment")
    category = frames["category"]
    brand = frames["brand"]
    product = frames["product"]
    monthly = frames["monthly"]
    report_lines = [
        "# Customer Feedback Business Insights",
        "",
        f"Source: MySQL `review_predictions`; category, brand, product, and monthly comparisons require at least {MIN_GROUP_REVIEWS} reviews per group.",
        "",
        "## Overall sentiment",
        f"- Total reviews analyzed: {int(overall['review_count'].sum()):,}",
    ]
    for sentiment in SENTIMENT_ORDER:
        row = overall.loc[sentiment] if sentiment in overall.index else {"review_count": 0, "percentage": 0}
        report_lines.append(f"- {sentiment}: {int(row['review_count']):,} ({row['percentage']:.2f}%)")
    if not category.empty:
        worst_category = category.iloc[0]
        best_category = category.sort_values("positive_percentage", ascending=False).iloc[0]
        report_lines.extend([
            "",
            "## Category findings",
            f"- Highest negative-feedback proportion: {worst_category['category']} ({worst_category['negative_percentage']:.2f}% across {int(worst_category['review_count']):,} reviews).",
            f"- Highest positive-feedback proportion among qualifying categories: {best_category['category']} ({best_category['positive_percentage']:.2f}% across {int(best_category['review_count']):,} reviews).",
        ])
    if not brand.empty:
        worst_brand = brand.iloc[0]
        report_lines.extend([
            "",
            "## Brand findings",
            f"- Highest negative-feedback proportion: {worst_brand['brand']} ({worst_brand['negative_percentage']:.2f}% across {int(worst_brand['review_count']):,} reviews).",
        ])
    if not product.empty:
        worst_product = product.iloc[0]
        best_product = product.sort_values("positive_percentage", ascending=False).iloc[0]
        report_lines.extend([
            "",
            "## Product findings",
            f"- Highest negative-feedback proportion: {worst_product['product']} ({worst_product['negative_percentage']:.2f}% across {int(worst_product['review_count']):,} reviews).",
            f"- Highest positive-feedback proportion among qualifying products: {best_product['product']} ({best_product['positive_percentage']:.2f}% across {int(best_product['review_count']):,} reviews).",
        ])
    if not monthly.empty:
        first, last = monthly.iloc[0], monthly.iloc[-1]
        change = last["negative_percentage"] - first["negative_percentage"]
        direction = "increased" if change > 0 else "decreased" if change < 0 else "was unchanged"
        report_lines.extend([
            "",
            "## Time trend",
            f"- Qualifying monthly data runs from {first['month']} to {last['month']}.",
            f"- Negative sentiment {direction} by {abs(change):.2f} percentage points between those qualifying endpoint months.",
            "- This comparison describes association over time and does not establish causation.",
        ])
    report_lines.extend([
        "",
        "## Common negative-review terms",
        "- " + ", ".join(f"{row.term} ({int(row.count)})" for row in negative_terms.head(10).itertuples()),
        "",
        "These are frequent terms in stored Negative reviews after the project's standard text cleaning; they are not causal explanations by themselves.",
    ])
    (report_dir / "business_insights.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def generate_analytics_figures(frames: dict[str, pd.DataFrame], negative_terms: pd.DataFrame, output_dir: str | Path) -> None:
    """Create simple business-facing charts from database aggregates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    overall = frames["overall"].set_index("sentiment").reindex(SENTIMENT_ORDER).fillna(0)
    _save_bar(overall["review_count"], "Overall Sentiment", "Sentiment", output_dir / "analytics_overall_sentiment.png", rotation=0)

    category = frames["category"].sort_values("negative_percentage", ascending=False).head(12)
    category.set_index("category")["negative_percentage"].plot(kind="bar", figsize=(11, 6), color="#d95f02")
    plt.title("Negative Feedback by Category (Minimum 30 Reviews)")
    plt.xlabel("Category")
    plt.ylabel("Negative reviews (%)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "analytics_category_negative_share.png", dpi=150)
    plt.close()

    monthly = frames["monthly"].set_index("month")
    monthly[["negative_percentage", "neutral_percentage", "positive_percentage"]].plot(figsize=(12, 6), color=["#d95f02", "#7570b3", "#1b9e77"])
    plt.title("Monthly Sentiment Percentages (Minimum 30 Reviews)")
    plt.xlabel("Review month")
    plt.ylabel("Reviews (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "analytics_sentiment_trend.png", dpi=150)
    plt.close()

    negative_terms.head(15).sort_values("count").plot(kind="barh", x="term", y="count", figsize=(10, 7), legend=False, color="#d95f02")
    plt.title("Common Terms in Negative Reviews")
    plt.xlabel("Occurrences")
    plt.ylabel("Term")
    plt.tight_layout()
    plt.savefig(output_dir / "analytics_negative_terms.png", dpi=150)
    plt.close()


def _save_bar(values: pd.Series, title: str, xlabel: str, path: Path, rotation: int) -> None:
    values.plot(kind="bar", figsize=(8, 5), color=["#d95f02", "#7570b3", "#1b9e77"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Review count")
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()