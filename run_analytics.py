"""Run business analytics from the existing MySQL review_predictions table."""

from src.analysis.analytics import (
    add_percentages,
    fetch_analytics,
    find_negative_terms,
    generate_analytics_figures,
    write_reports,
)


def main() -> None:
    frames = add_percentages(fetch_analytics())
    negative_terms = find_negative_terms(frames.pop("negative_text")["review_text"].tolist())
    write_reports(frames, negative_terms, "outputs/reports")
    generate_analytics_figures(frames, negative_terms, "outputs/figures")
    print("Analytics completed from MySQL review_predictions.")
    print("Reports: outputs/reports/business_insights.md")
    print("Figures: outputs/figures/analytics_*.png")


if __name__ == "__main__":
    main()