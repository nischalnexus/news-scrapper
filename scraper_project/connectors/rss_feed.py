from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncGenerator, Dict

import feedparser
import httpx
from dateutil import parser as date_parser

from ..models import ContentItem, Engagement
from ..utils.ids import uuid_from_url
from ..utils.logging import get_logger
from ..utils.text import extract_hashtags, extract_keywords
from .base import BaseConnector


class RSSFeedConnector(BaseConnector):
    """Fetch entries from a standard RSS or Atom feed."""

    async def fetch(self) -> AsyncGenerator[Dict[str, Any], None]:
        if not self.source.url:
            raise ValueError("RSS connector requires source.url")

        headers = {"User-Agent": "OpenScraper/1.0"}
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(str(self.source.url))
            response.raise_for_status()
        parsed = feedparser.parse(response.text)
        for entry in parsed.entries:
            yield {"feed": parsed.feed, "entry": entry}

    async def normalize(self, payload: Dict[str, Any]) -> ContentItem:
        entry = payload["entry"]
        link = entry.get("link")
        if not link:
            raise ValueError("RSS entry missing link field")

        published = entry.get("published") or entry.get("updated")
        published_at = None
        if published:
            try:
                published_at = date_parser.parse(published)
            except (ValueError, OverflowError) as exc:
                get_logger(__name__).warning("Unable to parse published timestamp", extra={"error": exc})

        summary = entry.get("summary") or entry.get("description")
        text_blob = " ".join(filter(None, [summary, entry.get("title")]))
        hashtags = extract_hashtags(text_blob)
        keywords = extract_keywords(text_blob)

        return ContentItem(
            id=uuid_from_url(link),
            source_id=self.source_id,
            url_canonical=link,
            author=(entry.get("author") or payload["feed"].get("title")),
            title=entry.get("title"),
            text=summary,
            summary=summary,
            published_at=published_at,
            media_urls=[media.get("url") for media in entry.get("media_content", []) if media.get("url")],
            hashtags=hashtags,
            keywords=keywords,
            engagement_raw=Engagement(),
            metadata={"tags": self.source.tags},
        )
