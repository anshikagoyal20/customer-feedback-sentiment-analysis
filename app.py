"""Lightweight Streamlit presentation layer for the sentiment project."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis.analytics import (
    SENTIMENT_ORDER,
    add_percentages,
    fetch_analytics,
    find_negative_terms,
)
from src.database.config import mysql_config


st.set_page_config(page_title="Customer Sentiment Analysis", page_icon="📊", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    return add_percentages(fetch_analytics())


def query_database(query: str, parameters: tuple = (), dictionary: bool = True) -> list:
    """Run one read-only query with project-managed credentials."""
    import mysql.connector

    connection = mysql.connector.connect(**mysql_config())
    cursor = connection.cursor(dictionary=dictionary)
    try:
        cursor.execute(query, parameters)
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


@st.cache_data(ttl=300, show_spinner=False)
def load_products() -> list[str]:
    rows = query_database("SELECT DISTINCT product FROM review_predictions ORDER BY product")
    return [row["product"] for row in rows]


@st.cache_data(ttl=300, show_spinner=False)
def load_product_data(product: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_rows = query_database(
        """
        SELECT DISTINCT brand, category
        FROM review_predictions
        WHERE product = %s
        ORDER BY brand, category
        """,
        (product,),
    )
    overview_rows = query_database(
        """
        SELECT sentiment, COUNT(*) AS review_count
        FROM review_predictions
        WHERE product = %s
        GROUP BY sentiment
        """,
        (product,),
    )
    review_rows = []
    for sentiment in SENTIMENT_ORDER:
        review_rows.extend(
            query_database(
                """
                SELECT review_id, sentiment, review_title, review_text, rating, review_date
                FROM review_predictions
                WHERE product = %s AND sentiment = %s
                ORDER BY review_date DESC, review_id
                LIMIT 3
                """,
                (product, sentiment),
            )
        )
    return pd.DataFrame(metadata_rows), pd.DataFrame(overview_rows), pd.DataFrame(review_rows)


@st.cache_data(ttl=300, show_spinner=False)
def load_all_product_negative_text(product: str) -> list[str]:
    rows = query_database(
        """
        SELECT review_text
        FROM review_predictions
        WHERE product = %s AND sentiment = 'Negative' AND review_text IS NOT NULL
        """,
        (product,),
    )
    return [row["review_text"] for row in rows]


def render_dashboard() -> None:
    st.header("Business Dashboard")
    st.caption("Live aggregates from the MySQL review_predictions table. Group comparisons require at least 30 reviews.")
    frames = load_dashboard_data()
    overall = frames["overall"].set_index("sentiment").reindex(SENTIMENT_ORDER).fillna(0)
    total = int(overall["review_count"].sum())
    metrics = st.columns(4)
    metrics[0].metric("Total reviews", f"{total:,}")
    for column, sentiment in zip(metrics[1:], ["Positive", "Negative", "Neutral"]):
        row = overall.loc[sentiment]
        column.metric(f"{sentiment} reviews", f"{int(row['review_count']):,}", f"{row['percentage']:.2f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Overall sentiment")
        st.bar_chart(overall["review_count"])
    with right:
        st.subheader("Negative feedback by category")
        category = frames["category"].head(12).set_index("category")
        st.bar_chart(category["negative_percentage"])
        if not category.empty:
            st.caption(f"Highest qualifying negative proportion: {category.index[0]} ({category.iloc[0]['negative_percentage']:.2f}%).")

    st.subheader("Monthly sentiment trend")
    monthly = frames["monthly"].set_index("month")[["negative_percentage", "neutral_percentage", "positive_percentage"]]
    monthly.columns = ["Negative", "Neutral", "Positive"]
    st.line_chart(monthly)

    left, right = st.columns(2)
    with left:
        st.subheader("Brand comparison")
        brand = frames["brand"].set_index("brand")[["negative_percentage", "positive_percentage"]]
        brand.columns = ["Negative %", "Positive %"]
        st.dataframe(brand, use_container_width=True)
        if not brand.empty:
            st.caption(f"Highest qualifying negative proportion: {brand.index[0]} ({brand.iloc[0]['Negative %']:.2f}%).")
    with right:
        st.subheader("Highest-negative product")
        products = frames["product"]
        if products.empty:
            st.info("No qualifying products are available.")
        else:
            worst_product = products.iloc[0]
            st.write(worst_product["product"])
            st.metric("Negative feedback", f"{worst_product['negative_percentage']:.2f}%", f"{int(worst_product['review_count']):,} reviews")
            st.caption(f"Brand: {worst_product['brand']} | Category: {worst_product['category']}")

    st.subheader("Top negative-feedback terms")
    terms = find_negative_terms(frames["negative_text"]["review_text"].tolist())
    st.bar_chart(terms.set_index("term").head(15)["count"])
    st.dataframe(terms.head(15), hide_index=True, use_container_width=True)


def render_product_analysis() -> None:
    st.header("Product Analysis")
    st.caption("Products are loaded dynamically from MySQL; no product names are hardcoded.")
    try:
        products = load_products()
    except Exception as error:
        st.error(f"Could not load products from MySQL: {error}")
        return
    if not products:
        st.info("No products are available in review_predictions.")
        return
    product = st.selectbox("Select a product", products)
    try:
        metadata, overview, representative_reviews = load_product_data(product)
        negative_text = load_all_product_negative_text(product)
    except Exception as error:
        st.error(f"Could not load product data from MySQL: {error}")
        return
    if overview.empty:
        st.info("This product has no reviews.")
        return

    overview = overview.set_index("sentiment").reindex(SENTIMENT_ORDER).fillna(0)
    total = int(overview["review_count"].sum())
    st.subheader("Product information")
    st.write(f"**Product:** {product}")
    if metadata.empty:
        st.write("**Brand:** Not available in database")
        st.write("**Category:** Not available in database")
    else:
        brands = [str(value) for value in metadata["brand"].dropna().unique() if str(value).strip()]
        categories = [str(value) for value in metadata["category"].dropna().unique() if str(value).strip()]
        st.write(f"**Brand:** {'; '.join(brands) if brands else 'Not available in database'}")
        st.write(f"**Category:** {'; '.join(categories) if categories else 'Not available in database'}")
    st.metric("Total reviews", f"{total:,}")

    st.subheader("Sentiment summary")
    metrics = st.columns(3)
    for column, sentiment in zip(metrics, ["Positive", "Neutral", "Negative"]):
        count = int(overview.loc[sentiment, "review_count"])
        column.metric(f"{sentiment}", f"{count:,}", f"{count / total * 100:.2f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment distribution")
        st.bar_chart(overview["review_count"])
    with right:
        negative_percentage = overview.loc["Negative", "review_count"] / total * 100
        st.subheader("Product insight")
        st.info(f"This product has {total:,} reviews. {negative_percentage:.2f}% of its reviews are negative.")
        negative_count = int(overview.loc["Negative", "review_count"])
        st.subheader("Negative-review terms" if negative_count < 3 else "Common negative terms")
        if negative_count == 0:
            st.write("No negative reviews were found for this product.")
        else:
            if negative_count < 3:
                st.caption(f"Based on {negative_count} negative review(s); treat these terms as indicative rather than representative.")
            terms = find_negative_terms(negative_text)
            if terms.empty:
                st.write("No usable repeated terms were found in these negative reviews.")
            else:
                st.dataframe(terms.head(15), hide_index=True, use_container_width=True)

    st.subheader("Representative recent reviews")
    st.caption("Up to three most recent reviews per sentiment, ordered by review date and review ID.")
    if representative_reviews.empty:
        st.info("No reviews are available for this product.")
    else:
        review_tabs = st.tabs([f"{sentiment} ({int(overview.loc[sentiment, 'review_count'])})" for sentiment in SENTIMENT_ORDER])
        for tab, sentiment in zip(review_tabs, SENTIMENT_ORDER):
            with tab:
                sentiment_reviews = representative_reviews[representative_reviews["sentiment"] == sentiment]
                available = int(overview.loc[sentiment, "review_count"])
                if sentiment_reviews.empty:
                    st.info(f"No {sentiment.lower()} reviews were found for this product.")
                else:
                    if available < 3:
                        st.caption(f"Only {available} {sentiment.lower()} review(s) exist; showing all available reviews.")
                    for review in sentiment_reviews.to_dict("records"):
                        title = review.get("review_title") or "Untitled review"
                        st.markdown(f"**{title}** | Rating: {review.get('rating')} | Date: {review.get('review_date')}")
                        st.write(review.get("review_text") or "(No review text)")
                        st.divider()


def render_prediction() -> None:
    st.header("New Review Prediction")
    st.caption("Uses the saved TF-IDF vectorizer and Logistic Regression model. No retraining is performed.")
    review = st.text_area("Enter a customer review", height=160, placeholder="The product stopped working after two days.")
    if st.button("Predict Sentiment", type="primary"):
        if not review.strip():
            st.warning("Enter a review before predicting.")
            return
        try:
            from predict import predict_sentiment

            sentiment = predict_sentiment(review.strip())
            st.success(f"Predicted Sentiment: {sentiment}")
        except Exception as error:
            st.error(f"Prediction failed: {error}")


def main() -> None:
    st.title("Customer Sentiment Analysis & Trend Analysis")
    st.caption("A live business view of customer feedback, product issues, and new-review sentiment.")
    dashboard, products, prediction = st.tabs(["📊 Business Dashboard", "🔎 Product Analysis", "🤖 New Review Prediction"])
    with dashboard:
        try:
            render_dashboard()
        except Exception as error:
            st.error(f"Could not load the business dashboard from MySQL: {error}")
    with products:
        render_product_analysis()
    with prediction:
        render_prediction()


if __name__ == "__main__":
    main()