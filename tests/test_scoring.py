from datetime import datetime, timedelta

from scraper_project.models import ContentItem, Engagement, ScoringSettings
from scraper_project.pipeline.scoring import score_items


def make_item(score: int) -> ContentItem:
    return ContentItem(
        id=f"item-{score}",
        source_id="test",
        url_canonical=f"https://example.com/{score}",
        collected_at=datetime.utcnow(),
        published_at=datetime.utcnow() - timedelta(hours=score),
        engagement_raw=Engagement(like=score, comment=score // 2, share=score // 3, view=score * 10),
    )


def test_score_items_orders_by_engagement():
    items = [make_item(1), make_item(5), make_item(10)]
    scored = score_items(items, ScoringSettings())
    viral_scores = [item.viral_score for item in scored]
    assert viral_scores[0] < viral_scores[-1]
