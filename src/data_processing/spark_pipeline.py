"""Load and clean the canonical review export with PySpark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


CANONICAL_COLUMNS = [
    "id",
    "reviews.id",
    "name",
    "brand",
    "categories",
    "primaryCategories",
    "reviews.date",
    "reviews.rating",
    "reviews.title",
    "reviews.text",
    "reviews.numHelpful",
    "reviews.doRecommend",
    "reviews.didPurchase",
    "reviews.username",
]


def _source_column(name: str):
    """Reference a CSV header literally, including headers containing dots."""
    return F.col(f"`{name}`")


def create_spark_session() -> SparkSession:
    """Create a small local Spark session suitable for a normal laptop."""
    return (
        SparkSession.builder.appName("CustomerSentimentAnalysis")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def load_and_clean_reviews(spark: SparkSession, csv_path: str | Path) -> DataFrame:
    """Read the CSV, remove unusable/duplicate rows, and derive sentiment."""
    source = str(Path(csv_path).resolve())
    raw = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(source)
    )

    selected = raw.select(*[_source_column(column) for column in CANONICAL_COLUMNS])
    cleaned = (
        selected.withColumn("review_text", F.trim(F.coalesce(_source_column("reviews.text"), F.lit(""))))
        .withColumn("review_title", F.trim(F.coalesce(_source_column("reviews.title"), F.lit(""))))
        .withColumn("product", F.trim(F.coalesce(_source_column("name"), F.lit(""))))
        .withColumn("brand", F.trim(F.coalesce(_source_column("brand"), F.lit(""))))
        .withColumn("category", F.trim(F.coalesce(_source_column("primaryCategories"), F.lit(""))))
        .withColumn("rating", _source_column("reviews.rating").cast("double"))
        .withColumn("review_date", F.to_date(F.to_timestamp(_source_column("reviews.date"))))
        .filter((F.length("review_text") > 0) & F.col("rating").isin([1, 2, 3, 4, 5]))
        .filter(F.col("review_date").isNotNull())
        .filter(F.length("product") > 0)
        .dropDuplicates(CANONICAL_COLUMNS)
        .withColumn(
            "sentiment",
            F.when(F.col("rating").isin([1, 2]), F.lit("Negative"))
            .when(F.col("rating") == 3, F.lit("Neutral"))
            .otherwise(F.lit("Positive")),
        )
    )

    review_key_columns = ["id", "product", "review_title", "review_text", "rating", "review_date"]
    return cleaned.withColumn(
        "review_id",
        F.sha2(F.concat_ws("||", *[F.col(column).cast("string") for column in review_key_columns]), 256),
    )


def dataframe_to_pandas(reviews: DataFrame):
    """Collect the cleaned, moderate-sized dataset for scikit-learn."""
    # Avoid Spark's pandas bridge, which imports removed distutils on Python 3.12.
    return pd.DataFrame([row.asDict(recursive=True) for row in reviews.collect()])