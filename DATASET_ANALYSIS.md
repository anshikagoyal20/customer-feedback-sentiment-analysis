# Dataset Analysis — Customer Sentiment Analysis & Trend Analysis

This report is based on a read-only inspection of all three CSV files in `data/`. The raw files were not modified. CSV fields are read as text in the source files; rating and date types below are based on validated parsing.

## Dataset selected for the project

**Canonical input:** `data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv`  
**Size:** 28,332 rows × 24 columns (265.6 MB)

The repository contains three related Datafiniti Amazon-review exports. The May 2019 file is recommended because it is large enough for an interview-scale ML project and has complete review text, ratings, titles, product, brand, category, and date fields. The files should **not** be merged without a separately designed cross-file de-duplication strategy, as they are related exports and could overlap.

Other available files:

| File | Rows | Notes |
| --- | ---: | --- |
| `1429_1.csv` | 34,660 | Missing ratings (33), dates (39), review text (1), and product names (6,760). |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv` | 5,000 | 95 exact duplicate rows and 615 duplicate non-empty review texts. |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` | 28,332 | Recommended; metadata needed for the planned analysis is complete. |

The May 2019 export is recommended because it has the required review text, title, rating, date, product, brand, and category fields without missing values in those core columns. The 1429 export lacks `primaryCategories` and has missing core values; the 5,000-row export contains exact duplicates.

## Important columns

| Project field | Source column | Availability / notes |
| --- | --- | --- |
| Review text | `reviews.text` | Complete; string data. |
| Review title | `reviews.title` | Complete; string data. |
| Rating | `reviews.rating` | Complete integer values from 1 to 5. |
| Review date | `reviews.date` | Complete ISO-8601 timestamps. |
| Product | `name` | Complete; 65 distinct products. |
| Brand | `brand` | Complete; 3 distinct brands. |
| Category | `primaryCategories`, `categories` | Complete; 9 primary categories and 60 detailed category strings. |
| Product/review identifier | `id`, `reviews.id` | `id` identifies the product record (65 distinct values), not a unique review. `reviews.id` is missing for most rows, so a pipeline-generated review key will be needed later. |

Other potentially useful metadata includes `asins`, `reviews.numHelpful`, `reviews.doRecommend`, `reviews.didPurchase`, and `reviews.username`.

## Data types and missing values

The source is CSV, so textual metadata is loaded as strings; `reviews.rating` is an integer, `reviews.numHelpful` is numeric, and recommendation/purchase fields are boolean-like. Date fields should be explicitly parsed in the pipeline rather than relying on CSV type inference.

Fields with missing values in the selected file:

| Column | Missing rows |
| --- | ---: |
| `reviews.didPurchase` | 28,323 |
| `reviews.id` | 28,291 |
| `reviews.doRecommend` | 12,246 |
| `reviews.numHelpful` | 12,217 |
| `reviews.username` | 5 |

All core modeling and trend-analysis fields (`reviews.text`, `reviews.title`, `reviews.rating`, `reviews.date`, `name`, `brand`, `categories`, and `primaryCategories`) have **0 missing values**.

The optional fields `reviews.didPurchase`, `reviews.id`, `reviews.doRecommend`, `reviews.numHelpful`, and `reviews.username` are incomplete. `id` is a product identifier with 65 distinct values, while `reviews.id` has only 41 non-empty distinct values and cannot serve as a reliable review key.

## Rating and derived-sentiment distribution

No existing sentiment-label column is present. Sentiment should therefore be documented and created from ratings:

| Rating | Rows | Derived sentiment | Rows | Share |
| ---: | ---: | --- | ---: | ---: |
| 1 | 965 | Negative (1–2) | 1,581 | 5.58% |
| 2 | 616 | Neutral (3) | 1,206 | 4.26% |
| 3 | 1,206 | Positive (4–5) | 25,545 | 90.17% |
| 4 | 5,648 |  |  |  |
| 5 | 19,897 |  |  |  |

These are **rating-derived labels**, not independently human-annotated sentiment labels. The target is strongly imbalanced toward Positive, which must be reflected in the later train/test split and multiclass evaluation.

## Dates, duplicates, and data-quality findings

- All 28,332 review dates parse successfully using an ISO-8601/mixed-format parser. The range is **2009-02-26 to 2019-03-25**.
- There are no exact full-row duplicates.
- There are 5 duplicates when matching product, review text, title, rating, and review date.
- 7,034 distinct non-empty review texts are repeated; 17,198 rows belong to those repeated-text groups, meaning 10,164 are extra occurrences beyond the first copy. Many are short generic reviews (for example, “good” or “Great”). After grouping additionally by product and rating, 1,399 repeated review texts remain. These should be reviewed and de-duplicated on a conservative review-level key during preprocessing to reduce training leakage and repeated-record bias.
- `reviews.id` is absent for most rows, and `id` is a product identifier rather than a unique review identifier. A stable generated review key will be required for database storage.
- Purchase and recommendation metadata are too incomplete to treat as core analysis fields without explicit missing-value handling.

## Preprocessing recommendations

1. Read only the selected May 2019 CSV as immutable raw input; write cleaned outputs separately.
2. Parse `reviews.date` explicitly and retain it for time trends.
3. Keep all rows with review text and ratings (both are complete); generate the rating-derived target label.
4. Remove the 5 confirmed review-level duplicates. Assess repeated text conservatively using product, rating, title, and date rather than removing every generic text globally.
5. Build model text from `reviews.title` plus `reviews.text`; lowercase, remove URLs and unnecessary punctuation, normalize whitespace, and use stop-word removal carefully so negation is preserved.
6. Use a stratified split and report macro/weighted precision, recall, and F1 in addition to accuracy because Positive reviews dominate the data.
7. Retain `name`, `brand`, `primaryCategories`, `categories`, `reviews.rating`, and parsed date for later trend reporting. Treat sparse optional metadata separately.
