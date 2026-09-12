"""Train and evaluate a TF-IDF plus multiclass logistic regression model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.nlp.text_preprocessing import prepare_model_text


LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def train_and_evaluate(reviews: pd.DataFrame, model_dir: str | Path, report_dir: str | Path):
    """Fit only on training text and return predictions plus real evaluation metrics."""
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    reviews = reviews.copy()
    reviews["model_text"] = [
        prepare_model_text(title, body)
        for title, body in zip(reviews["review_title"], reviews["review_text"])
    ]
    train_data, test_data = train_test_split(
        reviews,
        test_size=0.2,
        random_state=42,
        stratify=reviews["sentiment"],
    )

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_data["model_text"])
    test_matrix = vectorizer.transform(test_data["model_text"])
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    classifier.fit(train_matrix, train_data["sentiment"])
    predictions = classifier.predict(test_matrix)

    metrics = {
        "test_rows": int(len(test_data)),
        "train_rows": int(len(train_data)),
        "accuracy": float(accuracy_score(test_data["sentiment"], predictions)),
        "precision_macro": float(precision_score(test_data["sentiment"], predictions, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(test_data["sentiment"], predictions, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(test_data["sentiment"], predictions, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(test_data["sentiment"], predictions, labels=LABEL_ORDER, average="weighted", zero_division=0)),
        "classification_report": classification_report(test_data["sentiment"], predictions, labels=LABEL_ORDER, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test_data["sentiment"], predictions, labels=LABEL_ORDER).tolist(),
        "labels": LABEL_ORDER,
        "vectorizer_features": int(len(vectorizer.get_feature_names_out())),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    joblib.dump(classifier, model_dir / "logistic_regression.joblib")

    test_data = test_data.copy()
    test_data["predicted_sentiment"] = predictions
    test_data.to_csv(report_dir / "test_predictions.csv", index=False)
    all_matrix = vectorizer.transform(reviews["model_text"])
    all_predictions = reviews.drop(columns=["model_text"]).copy()
    all_predictions["predicted_sentiment"] = classifier.predict(all_matrix)
    all_predictions.to_csv(report_dir / "predictions.csv", index=False)
    _write_top_terms(vectorizer, classifier, report_dir / "top_terms.csv")
    return metrics, all_predictions, vectorizer, classifier


def _write_top_terms(vectorizer: TfidfVectorizer, classifier: LogisticRegression, path: Path) -> None:
    """Save high-weight terms for each one-vs-rest class for interpretation."""
    terms = vectorizer.get_feature_names_out()
    rows = []
    for label, weights in zip(classifier.classes_, classifier.coef_):
        for index in weights.argsort()[-20:][::-1]:
            rows.append({"sentiment": label, "term": terms[index], "coefficient": float(weights[index])})
    pd.DataFrame(rows).to_csv(path, index=False)