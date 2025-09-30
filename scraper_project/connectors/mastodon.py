from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List

import httpx

from ..models import ContentItem, Engagement
from ..utils.ids import uuid_from_url
from ..utils.logging import get_logger
from ..utils.text import extract_hashtags, extract_keywords
from .base import BaseConnector


class MastodonConnector(BaseConnector):
    """Fetch public posts for a Mastodon account."""

    API_BASE = "https://{instance}"

    async def fetch(self) -> AsyncGenerator[Dict[str, Any], None]:
        if not self.source.instance or not self.source.handle:
            raise ValueError("Mastodon connector requires instance and handle")

        account = await self._lookup_account()
        if not account:
            return

        params = {"limit": self.source.crawl.get("limit", 20), "exclude_reblogs": True}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(self._url(f"/api/v1/accounts/{account['id']}/statuses"), params=params)
            response.raise_for_status()
            for status in response.json():
                yield status

    async def normalize(self, payload: Dict[str, Any]) -> ContentItem:
        url = payload.get("url")
        if not url:
            raise ValueError("Mastodon status missing url")

        content = payload.get("content") or ""
        text = _strip_html(content)
        hashtags = [tag.get("name", "").lower() for tag in payload.get("tags", []) if tag.get("name")]
        keywords = extract_keywords(text)

        published_at = datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00"))

        engagement = Engagement(
            like=payload.get("favourites_count", 0),
            comment=payload.get("replies_count", 0),
            share=payload.get("reblogs_count", 0),
            view=payload.get("replies_count", 0) * 10,
        )

        return ContentItem(
            id=uuid_from_url(url),
            source_id=self.source_id,
            url_canonical=url,
            author=payload.get("account", {}).get("display_name") or payload.get("account", {}).get("username"),
            title=payload.get("spoiler_text") or text[:120],
            text=text,
            summary=text[:5000],
            published_at=published_at,
            media_urls=[attachment.get("url") for attachment in payload.get("media_attachments", []) if attachment.get("url")],
            hashtags=sorted(set(hashtags + extract_hashtags(text))),
            keywords=keywords,
            engagement_raw=engagement,
            metadata={"visibility": payload.get("visibility")},
        )

    async def _lookup_account(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(self._url("/api/v1/accounts/lookup"), params={"acct": self.source.handle.lstrip("@").strip()})
            response.raise_for_status()
            return response.json()

    def _url(self, path: str) -> str:
        return self.API_BASE.format(instance=self.source.instance.strip("/")) + path


def _strip_html(html: str) -> str:
    # simple HTML tags removal for Mastodon content
    import re

    text = re.sub(r"<br\s*/?>", "\n", html)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
