CREATE DATABASE IF NOT EXISTS customer_sentiment;
USE customer_sentiment;

CREATE TABLE IF NOT EXISTS review_predictions (
    review_id CHAR(64) PRIMARY KEY,
    product VARCHAR(500) NOT NULL,
    brand VARCHAR(100),
    category VARCHAR(200),
    review_date DATE,
    rating DECIMAL(2,1) NOT NULL,
    review_title TEXT,
    review_text TEXT NOT NULL,
    sentiment ENUM('Negative', 'Neutral', 'Positive') NOT NULL,
    predicted_sentiment ENUM('Negative', 'Neutral', 'Positive'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category (category),
    INDEX idx_review_date (review_date),
    INDEX idx_sentiment (sentiment)
);