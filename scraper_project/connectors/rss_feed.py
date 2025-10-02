from __future__ import annotations

import asyncio
import warnings
from typing import Any, AsyncGenerator, Dict, Set

import feedparser
import httpx
from dateutil import parser as date_parser, tz as date_tz

from ..models import ContentItem, Engagement
from ..utils.ids import uuid_from_url
from ..utils.logging import get_logger
from ..utils.text import extract_hashtags, extract_keywords
from .base import BaseConnector




FAILED_FEED_URLS: Set[str] = set()

_KNOWN_TZINFOS = {
    "UTC": date_tz.UTC,
    "GMT": date_tz.UTC,
    "Z": date_tz.UTC,
    "PST": date_tz.gettz("America/Los_Angeles"),
    "PDT": date_tz.gettz("America/Los_Angeles"),
    "MST": date_tz.gettz("America/Denver"),
    "MDT": date_tz.gettz("America/Denver"),
    "CST": date_tz.gettz("America/Chicago"),
    "CDT": date_tz.gettz("America/Chicago"),
    "EST": date_tz.gettz("America/New_York"),
    "EDT": date_tz.gettz("America/New_York"),
    "AKST": date_tz.gettz("America/Anchorage"),
    "AKDT": date_tz.gettz("America/Anchorage"),
    "HST": date_tz.gettz("Pacific/Honolulu"),
    "BST": date_tz.gettz("Europe/London"),
    "IST": date_tz.gettz("Asia/Kolkata"),
    "CET": date_tz.gettz("Europe/Paris"),
    "CEST": date_tz.gettz("Europe/Paris"),
    "AEST": date_tz.gettz("Australia/Sydney"),
    "AEDT": date_tz.gettz("Australia/Sydney"),
}

_DEFAULT_HEADER_VARIANTS = [
    {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/124.0 Safari/537.36 (compatible; OpenScraper/1.0))"),
        "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7",
        "Accept-Language": "en-US,en;q=0.8",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 \
        (KHTML, like Gecko) Version/17.4 Safari/605.1.15"),
        "Accept": "application/xml, text/xml, */*;q=0.8",
        "Accept-Language": "en-US,en;q=0.7",
        "Connection": "keep-alive",
    },
]

def _header_variants_for(source_headers: Dict[str, Any] | None) -> list[Dict[str, str]]:
    variants = [variant.copy() for variant in _DEFAULT_HEADER_VARIANTS]
    if source_headers:
        normalized = {str(k): str(v) for k, v in source_headers.items()}
        variants.insert(0, {**variants[0], **normalized})
    return variants

class RSSFeedConnector(BaseConnector):
    """Fetch entries from a standard RSS or Atom feed."""

    async def fetch(self) -> AsyncGenerator[Dict[str, Any], None]:
        if not self.source.url:
            raise ValueError("RSS connector requires source.url")

        crawl_opts = getattr(self.source, "crawl", {}) or {}
        custom_headers = crawl_opts.get("headers") if isinstance(crawl_opts, dict) else None
        header_variants = _header_variants_for(custom_headers if isinstance(custom_headers, dict) else None)

        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, http2=True) as client:
            response, error = await _fetch_with_header_variants(client, str(self.source.url), header_variants)

        parsed = None
        if response is not None:
            parsed = feedparser.parse(response.text)
        else:
            fallback_headers = header_variants[0] if header_variants else {}
            try:
                parsed = await asyncio.to_thread(feedparser.parse, str(self.source.url), request_headers=fallback_headers)
            except Exception as exc:  # pragma: no cover - defensive fallback
                error = exc

        entries = getattr(parsed, "entries", None) if parsed else None
        if not parsed or not entries:
            url_str = str(self.source.url)
            if url_str not in FAILED_FEED_URLS:
                FAILED_FEED_URLS.add(url_str)
                reason = "no entries returned"
                if error:
                    reason = f"{type(error).__name__}: {error}"
                elif parsed:
                    bozo_exc = getattr(parsed, "bozo_exception", None)
                    if bozo_exc:
                        reason = f"{type(bozo_exc).__name__}: {bozo_exc}"
                    else:
                        status = getattr(parsed, "status", None)
                        if status:
                            reason = f"feedparser status {status}"
                get_logger(__name__).warning("RSS fetch failed for %s (%s)", url_str, reason)
            return

        for entry in entries:
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
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="tzname .+ identified but not understood",
                        module="dateutil.parser",
                    )
                    published_at = date_parser.parse(published, tzinfos=_KNOWN_TZINFOS)
            except (ValueError, OverflowError) as exc:
                get_logger(__name__).warning(
                    "Unable to parse published timestamp for %s: %s",
                    link,
                    exc,
                )
            else:
                if getattr(published_at, "tzinfo", None):
                    published_at = published_at.astimezone(date_tz.UTC)

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
            metadata=self.metadata(),
        )


async def _fetch_with_header_variants(
    client: httpx.AsyncClient, url: str, header_variants: list[Dict[str, str]]
) -> tuple[httpx.Response | None, Exception | None]:
    last_error: Exception | None = None
    for headers in header_variants:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response, None
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            if status in {401, 403, 404, 429, 503}:
                # try next header variant; many feeds gate based on UA
                continue
            break
        except httpx.HTTPError as exc:
            last_error = exc
    return None, last_error
