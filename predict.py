"""Predict sentiment for one new customer review using saved model artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.nlp.text_preprocessing import prepare_model_text


PROJECT_ROOT = Path(__file__).resolve().parent
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.joblib"


def predict_sentiment(review: str) -> str:
    """Transform and classify a new review without fitting either artifact."""
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(MODEL_PATH)

    # Keep preprocessing identical to training: title is empty for this body-only input.
    cleaned_review = prepare_model_text("", review)
    review_features = vectorizer.transform([cleaned_review])
    return str(classifier.predict(review_features)[0])


def main() -> None:
    if not VECTORIZER_PATH.is_file() or not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Saved model artifacts were not found. Run the training pipeline first."
        )

    review = input("Enter a customer review: ").strip()
    if not review:
        raise ValueError("The customer review cannot be empty.")

    print(f"\nPredicted Sentiment: {predict_sentiment(review)}")


if __name__ == "__main__":
    main()