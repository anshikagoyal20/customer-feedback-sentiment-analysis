"""Conservative text normalization for review titles and bodies."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


NEGATION_WORDS = {"no", "not", "nor", "never", "none", "neither", "cannot"}
STOP_WORDS = ENGLISH_STOP_WORDS.difference(NEGATION_WORDS)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize text without stemming or removing sentiment-bearing negations."""
    text = "" if value is None else str(value)
    text = URL_PATTERN.sub(" ", text.lower())
    text = NON_WORD_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def combine_review_text(title: object, body: object) -> str:
    """Combine title and body, preserving title terms as part of the review."""
    return clean_text(f"{title or ''} {body or ''}")


def remove_stopwords(text: str) -> str:
    """Remove common function words while retaining common negation words."""
    return " ".join(token for token in text.split() if token not in STOP_WORDS)


def prepare_model_text(title: object, body: object) -> str:
    return remove_stopwords(combine_review_text(title, body))