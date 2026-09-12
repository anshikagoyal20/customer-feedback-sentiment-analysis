# 📊 Customer Sentiment Analysis & Trend Analysis

A local ML pipeline that classifies Amazon customer reviews as Negative, Neutral, or Positive and turns them into business insights by product, brand, category, and date. Built for data teams, product managers, or anyone who wants to convert raw customer feedback into sentiment trends and predictions without standing up a hosted service.

## ✨ Key Features

- **Rating-Derived Sentiment Classification** — Automatically labels reviews as Negative (1–2 stars), Neutral (3 stars), or Positive (4–5 stars) based on rating data.
- **PySpark Data Pipeline** — Loads, cleans, parses dates, filters, and deduplicates large CSV review datasets at scale.
- **TF-IDF + Logistic Regression Model** — Uses unigram/bigram TF-IDF vectorization with a balanced multiclass Logistic Regression classifier for sentiment prediction.
- **Business Analytics Reports** — Generates CSV reports, JSON metrics, confusion matrices, sentiment charts, and negative-term analysis by category, brand, and month.
- **Interactive Streamlit Dashboard** — Explore a Business Dashboard, Product Analysis view, and live New Review Prediction tool.
- **Optional MySQL Persistence** — Store predictions in a database for historical, query-driven analytics.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend & ML** | Python 3.12, PySpark 3.5.6, pandas, scikit-learn, joblib |
| **Frontend** | Streamlit 1.41.1, Matplotlib, Seaborn |
| **Database/Services** | MySQL, MySQL Connector/Python, python-dotenv |

## 📦 Installation & Setup

### Prerequisites

- Python 3.12
- Java 8 or newer (required for local Spark)
- MySQL (only needed for the dashboard, database persistence, or database-backed analytics)
- The canonical dataset placed at `data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv`

### On Windows (PowerShell):

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# (Optional) Configure MySQL
Copy-Item .env.example .env
# Edit .env with your MySQL credentials, then:
mysql -u your_username -p < sql/schema.sql
```

### On macOS/Linux:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# (Optional) Configure MySQL
cp .env.example .env
# Edit .env with your MySQL credentials, then:
mysql -u your_username -p < sql/schema.sql
```

## 🚀 Quick Start

Run the local pipeline to train the model and generate reports:

```bash
python run_pipeline.py
```

To also persist predictions to MySQL:

```bash
python run_pipeline.py --mysql
```

Launch the dashboard (requires MySQL populated with the `review_predictions` table):

```bash
streamlit run app.py
```

Access it at the local URL printed in your terminal (typically `http://localhost:8501`). There is currently no hosted deployment — the project runs entirely locally.

### First Time Using It

1. Run `python run_pipeline.py` to clean the data, train the model, and generate reports and figures.
2. Run `python -m pytest -q` to confirm the test suite passes (6 tests).
3. If you want database features, configure `.env` and load the schema via `sql/schema.sql`.
4. Run `python run_pipeline.py --mysql` to write predictions into MySQL.
5. Launch the dashboard with `streamlit run app.py`.
6. In **Business Dashboard**, review overall sentiment and category/month trends.
7. In **Product Analysis**, select a product to inspect sentiment counts, negative terms, and recent reviews, then try **New Review Prediction** on a custom review.

### Using Custom Data

The pipeline expects the raw CSV to follow the Datafiniti Amazon Reviews export format. Key columns used include:

| Column Name | Description |
|-------------|-------------|
| `reviews.rating` | Star rating (1–5), used to derive the sentiment label |
| `reviews.text` | Full text of the customer review |
| `reviews.date` | Date the review was submitted |
| `name` / `brand` | Product name and brand, used for grouping in analytics |
| `categories` | Product category, used for category-level trend reports |

Raw CSV files should be treated as immutable input data and are excluded from Git due to size. Place your file at `data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` before running the pipeline.

## 📁 Project Structure

```
customer-feedback-sentiment-analysis/
├── app.py
├── predict.py
├── run_pipeline.py
├── run_analytics.py
├── requirements.txt
├── DATASET_ANALYSIS.md
├── data/
├── data_processing/
├── nlp/
├── models/
├── analysis/
├── database/
├── sql/
│   ├── schema.sql
│   └── analysis_queries.sql
├── tests/
├── outputs/
└── notebooks/
```
