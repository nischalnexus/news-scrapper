from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

from ..models import ContentItem, Settings
from ..utils.logging import get_logger
from ..utils.async_tools import ensure_proactor_event_loop_policy
from .dedupe import dedupe_items
from .fetcher import run_connectors
from .normalizer import normalize_items
from .scoring import score_items


def _item_timestamp_utc(item: ContentItem) -> datetime | None:
    """Return the best available timestamp for recency filtering."""

    ts = item.published_at or item.collected_at
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


async def ingest_async(settings: Settings) -> List[ContentItem]:
    items = await run_connectors(settings.sources)
    items = normalize_items(items)
    items = dedupe_items(items)
    items = score_items(items, settings.scoring)
    get_logger(__name__).info("Ingestion complete", extra={"count": len(items)})
    return items


def filter_items_by_hours(items: List[ContentItem], hours: float) -> List[ContentItem]:
    if not hours:
        return items
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    filtered: List[ContentItem] = []
    for item in items:
        ts = _item_timestamp_utc(item)
        if ts is not None and ts < cutoff:
            continue
        filtered.append(item)
    return filtered


def ingest(settings: Settings) -> List[ContentItem]:
    ensure_proactor_event_loop_policy()
    return asyncio.run(ingest_async(settings))
