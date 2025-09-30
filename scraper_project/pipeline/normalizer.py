from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List

from ..models import ContentItem


def normalize_items(items: Iterable[ContentItem]) -> List[ContentItem]:
    normalized: List[ContentItem] = []
    for item in items:
        item.title = (item.title or item.text[:120] if item.text else None)
        item.summary = item.summary or (item.text[:5000] if item.text else None)
        if item.published_at:
            if item.published_at.tzinfo is None:
                item.published_at = item.published_at.replace(tzinfo=timezone.utc)
            else:
                item.published_at = item.published_at.astimezone(timezone.utc)
        normalized.append(item)
    return normalized
