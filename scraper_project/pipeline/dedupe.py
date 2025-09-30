from __future__ import annotations

from typing import Iterable, List

from ..models import ContentItem


def dedupe_items(items: Iterable[ContentItem]) -> List[ContentItem]:
    seen = {}
    for item in items:
        seen[item.id] = item
    return list(seen.values())
