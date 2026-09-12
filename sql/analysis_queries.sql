USE customer_sentiment;

SELECT sentiment, COUNT(*) AS review_count, ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM review_predictions GROUP BY sentiment;

SELECT category, sentiment, COUNT(*) AS review_count
FROM review_predictions GROUP BY category, sentiment ORDER BY category, sentiment;

SELECT category, COUNT(*) AS review_count, ROUND(AVG(rating), 2) AS average_rating
FROM review_predictions GROUP BY category HAVING COUNT(*) >= 30 ORDER BY average_rating;

SELECT YEAR(review_date) AS review_year, MONTH(review_date) AS review_month, sentiment, COUNT(*) AS review_count
FROM review_predictions GROUP BY YEAR(review_date), MONTH(review_date), sentiment
ORDER BY review_year, review_month, sentiment;

SELECT brand, COUNT(*) AS review_count,
       ROUND(100 * SUM(sentiment = 'Negative') / COUNT(*), 2) AS negative_percentage
FROM review_predictions GROUP BY brand HAVING COUNT(*) >= 30
ORDER BY negative_percentage DESC;

SELECT product, category, COUNT(*) AS review_count
FROM review_predictions GROUP BY product, category ORDER BY review_count DESC;