from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncGenerator, Dict

import httpx

from ..models import ContentItem, Engagement
from ..utils.ids import uuid_from_url
from ..utils.text import extract_hashtags, extract_keywords
from .base import BaseConnector


class RedditPublicConnector(BaseConnector):
    """Fetch posts from a public subreddit JSON feed."""

    BASE_URL = "https://www.reddit.com{path}.json"

    async def fetch(self) -> AsyncGenerator[Dict[str, Any], None]:
        if not self.source.path:
            raise ValueError("Reddit connector requires path, e.g. /r/Python/")
        params = {"limit": self.source.crawl.get("limit", 25)}
        headers = {"User-Agent": "OpenScraper/1.0"}
        async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
            response = await client.get(self.BASE_URL.format(path=self.source.path.rstrip("/")), params=params)
            response.raise_for_status()
            payload = response.json()
            for child in payload.get("data", {}).get("children", []):
                yield child.get("data", {})

    async def normalize(self, payload: Dict[str, Any]) -> ContentItem:
        url = payload.get("url") or payload.get("permalink")
        if not url:
            raise ValueError("Reddit post missing url")
        canonical = url if url.startswith("http") else f"https://www.reddit.com{url}"

        text = payload.get("selftext", "")
        title = payload.get("title")
        hashtags = extract_hashtags(text + " " + (title or ""))
        keywords = extract_keywords(text + " " + (title or ""))

        created = payload.get("created_utc")
        published_at = datetime.utcfromtimestamp(created) if created else None

        engagement = Engagement(
            like=int(payload.get("ups", 0)),
            comment=int(payload.get("num_comments", 0)),
            share=int(payload.get("crosspost_parent_list", []).__len__()),
            view=int(payload.get("view_count", 0) or 0),
        )

        media_urls = []
        if payload.get("preview"):
            for image in payload["preview"].get("images", []):
                source = image.get("source")
                if source and source.get("url"):
                    media_urls.append(source["url"].replace("&amp;", "&"))

        return ContentItem(
            id=uuid_from_url(canonical),
            source_id=self.source_id,
            url_canonical=canonical,
            author=payload.get("author"),
            title=title,
            text=text,
            summary=text[:5000],
            published_at=published_at,
            media_urls=media_urls,
            hashtags=hashtags,
            keywords=keywords,
            engagement_raw=engagement,
            metadata={"subreddit": payload.get("subreddit"), "tags": self.source.tags},
        )
