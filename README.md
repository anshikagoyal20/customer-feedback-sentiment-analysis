# Customer Sentiment Analysis & Trend Analysis

An interview-friendly machine learning project that loads Amazon customer reviews with PySpark, derives sentiment from ratings, trains a TF-IDF plus multiclass Logistic Regression model, produces dataset-backed analysis, and optionally stores predictions in MySQL.

## Problem and Objective

Customer reviews contain useful feedback but are difficult to summarize manually. This project classifies reviews into Negative, Neutral, and Positive sentiment and supports trend analysis by rating, product category, brand, and date.

The canonical input is `data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` (28,332 rows, 24 columns). The raw CSV is never modified.

The raw CSV files are intentionally excluded from Git because they are large local data assets. Obtain the verified Datafiniti Amazon customer-review export separately and place the canonical May 2019 file at the path above before running the pipeline. The repository includes `DATASET_ANALYSIS.md` with the inspected schema, quality checks, and file-selection rationale.

Important fields include `reviews.text`, `reviews.title`, `reviews.rating`, `reviews.date`, `name`, `brand`, `primaryCategories`, and `categories`. The selected file contains 65 products, 3 brands, and 9 primary categories.

## Important Label Limitation

The dataset has no human sentiment-label column. Labels are derived from the customer rating:

- Ratings 1-2: `Negative`
- Rating 3: `Neutral`
- Ratings 4-5: `Positive`

These are rating-derived proxy labels, not independently human-annotated sentiment labels. The dataset is highly imbalanced: Positive is the majority class. The model uses a stratified split and `class_weight="balanced"`; metrics include macro and weighted scores.

## Architecture

```text
Raw CSV
  -> PySpark load and cleaning
  -> date parsing, deduplication, rating-derived labels
  -> title + review text normalization
  -> stratified train/test split
  -> training-only TF-IDF fit
  -> multiclass Logistic Regression
  -> evaluation and predictions
  -> figures and trend tables
  -> optional MySQL storage
```

## Technologies

Python, PySpark, scikit-learn, TF-IDF, Logistic Regression, pandas, Matplotlib, Seaborn, MySQL Connector/Python, pytest.

## Preprocessing and Modeling

PySpark reads only the May 2019 CSV, selects useful columns, removes unusable rows and exact duplicates, parses `reviews.date` to a date, and generates a stable SHA-256 review key. Missing review text is treated as empty and filtered if it remains empty.

Titles and bodies are combined. Text is lowercased, URLs and punctuation are removed, whitespace is normalized, and common stopwords are removed while preserving negation words such as `not`, `never`, and `no`. No aggressive stemming or lemmatization is used.

TF-IDF uses unigrams and bigrams, `min_df=2`, `max_features=20000`, and sublinear term frequency. The vectorizer is fit only on training text to prevent leakage. Logistic Regression uses the `lbfgs` solver, `max_iter=1000`, L2 regularization by default, `class_weight="balanced"`, and a reproducible random seed. SciPy is pinned to `1.15.3` because scikit-learn `1.6.1` passes an `iprint` option to L-BFGS that newer SciPy releases reject with an `OptimizeWarning`; this keeps the existing model configuration and behavior unchanged.

## Installation

Use Python 3.12 and Java 8 or 11+ for local Spark. From the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Run the Project

```powershell
python run_pipeline.py
```

This creates real local artifacts under `models/`, `outputs/figures/`, and `outputs/reports/`, including `metrics.json`, `test_predictions.csv`, category summaries, monthly sentiment counts, and six figures. Generated artifacts are ignored by Git.

Run the database-backed business analytics separately after the predictions have been stored in MySQL:

```powershell
python run_analytics.py
```

This uses SQL aggregation over `review_predictions` and writes `outputs/reports/business_insights.md`, CSV summary tables, and `analytics_*.png` figures. Category, brand, product, and qualifying-month comparisons use a minimum of 30 reviews to reduce small-sample distortion. The report describes associations in the data and does not claim causation.

## Streamlit Application

The lightweight Streamlit presentation layer reads live values from the existing MySQL `review_predictions` table and does not alter the training pipeline or database schema. It has three tabs:

- Business Dashboard: overall sentiment, category, brand, product, monthly trend, and negative-term views.
- Product Analysis: dynamically populated product selector, sentiment breakdown, real negative reviews, and product-specific terms.
- New Review Prediction: inference with the saved TF-IDF vectorizer and Logistic Regression model without retraining.

Configure the existing project-root `.env` with the MySQL variables described below, then launch from the repository root:

```powershell
streamlit run app.py
```

The app handles database connection failures with an in-app message and never displays credentials. Product and analytics values are queried dynamically; no report values are hardcoded.

Run tests with:

```powershell
python -m pytest -q
```

## Predict a New Review

After the model artifacts exist, run the root-level inference script:

```powershell
python predict.py
```

Enter one new review when prompted. The script reuses the saved TF-IDF vectorizer and Logistic Regression model, applies the same preprocessing as training, and uses `transform()` only; it does not retrain the model.

## MySQL Integration

MySQL is optional and is not required to train or evaluate the model. Copy `.env.example` to `.env`, fill in the connection values, start MySQL, and run `sql/schema.sql`. Then load the environment variables in the shell and run:

```powershell
python run_pipeline.py --mysql
```

The writer uses `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`, `MYSQL_USER`, and `MYSQL_PASSWORD`. Credentials are never stored in source code. Useful demonstration queries are in `sql/analysis_queries.sql`.

## Project Structure

```text
data/                 immutable raw CSV files
src/data_processing/  PySpark loading, cleaning, and labels
src/nlp/              reusable text normalization
src/models/           TF-IDF and Logistic Regression training
src/analysis/         figures and trend tables
src/database/         optional MySQL writer
sql/                  schema and analysis queries
tests/                focused transformation tests
app.py                Streamlit presentation layer
models/               locally generated model artifacts
outputs/              locally generated figures and reports
```

## Limitations and Future Improvements

The target labels reflect star ratings, so the model measures agreement with the rating-derived scheme rather than independent sentiment truth. Positive reviews dominate, and repeated generic review text requires conservative deduplication. Future work could add human-reviewed labels, time-based validation, calibrated probabilities, richer aspect-level feedback extraction, and scheduled database refreshes.

No accuracy or trend result is claimed in this README. Run the pipeline to calculate metrics from the actual local dataset.