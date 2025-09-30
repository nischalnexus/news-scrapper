from datetime import datetime

from scraper_project.models import ContentItem, Engagement
from scraper_project.pipeline.dedupe import dedupe_items


def make_item(item_id: str) -> ContentItem:
    return ContentItem(
        id=item_id,
        source_id="source",
        url_canonical=f"https://example.com/{item_id}",
        collected_at=datetime.utcnow(),
        engagement_raw=Engagement(),
    )


def test_dedupe_items():
    items = [make_item("1"), make_item("1"), make_item("2")]
    deduped = dedupe_items(items)
    assert len(deduped) == 2
    assert {item.id for item in deduped} == {"1", "2"}
