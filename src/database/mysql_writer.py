"""Optional MySQL persistence using environment-based credentials."""

from __future__ import annotations

import mysql.connector
import pandas as pd

from src.database.config import mysql_config


def write_predictions(reviews: pd.DataFrame) -> int:
    """Insert or update predictions and return the number processed."""
    connection = mysql.connector.connect(**mysql_config())
    cursor = connection.cursor()
    query = """INSERT INTO review_predictions
        (review_id, product, brand, category, review_date, rating, review_title,
         review_text, sentiment, predicted_sentiment)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            product = VALUES(product),
            brand = VALUES(brand),
            category = VALUES(category),
            review_date = VALUES(review_date),
            rating = VALUES(rating),
            review_title = VALUES(review_title),
            review_text = VALUES(review_text),
            sentiment = VALUES(sentiment),
            predicted_sentiment = VALUES(predicted_sentiment)
    """
    columns = [
        "review_id",
        "product",
        "brand",
        "category",
        "review_date",
        "rating",
        "review_title",
        "review_text",
        "sentiment",
        "predicted_sentiment",
    ]
    rows = [tuple(row) for row in reviews[columns].itertuples(index=False, name=None)]
    try:
        if rows:
            cursor.executemany(query, rows)
            connection.commit()
        return len(rows)
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()