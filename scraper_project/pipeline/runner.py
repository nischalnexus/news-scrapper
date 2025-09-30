from __future__ import annotations

import asyncio
from typing import List

from ..models import ContentItem, Settings
from ..utils.logging import get_logger
from .dedupe import dedupe_items
from .fetcher import run_connectors
from .normalizer import normalize_items
from .scoring import score_items


async def ingest_async(settings: Settings) -> List[ContentItem]:
    items = await run_connectors(settings.sources)
    items = normalize_items(items)
    items = dedupe_items(items)
    items = score_items(items, settings.scoring)
    get_logger(__name__).info("Ingestion complete", extra={"count": len(items)})
    return items


def ingest(settings: Settings) -> List[ContentItem]:
    return asyncio.run(ingest_async(settings))
