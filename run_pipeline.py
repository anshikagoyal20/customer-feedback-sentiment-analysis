"""Run the complete local pipeline from raw CSV to reports and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.visualizations import generate_figures, write_analysis_tables
from src.data_processing.spark_pipeline import create_spark_session, dataframe_to_pandas, load_and_clean_reviews
from src.models.train_model import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
    parser.add_argument("--mysql", action="store_true", help="Write predictions to MySQL after local validation")
    args = parser.parse_args()

    spark = create_spark_session()
    try:
        reviews = dataframe_to_pandas(load_and_clean_reviews(spark, args.data).cache())
    finally:
        spark.stop()

    reports = Path("outputs/reports")
    metrics, predictions, _, _ = train_and_evaluate(reviews, "models", reports)
    generate_figures(reviews, predictions, "outputs/figures", reports / "top_terms.csv")
    write_analysis_tables(reviews, reports)
    (reports / "sentiment_distribution.json").write_text(json.dumps(reviews["sentiment"].value_counts().to_dict(), indent=2), encoding="utf-8")

    if args.mysql:
        from src.database.mysql_writer import write_predictions
        print(f"Processed {write_predictions(predictions)} predictions in MySQL (inserted or updated).")
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"classification_report", "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()