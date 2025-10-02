from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from ..models import ContentItem, Engagement
from ..utils.ids import uuid_from_url
from ..utils.logging import get_logger
from ..utils.text import extract_hashtags, extract_keywords
from .base import BaseConnector

_PLAYWRIGHT_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    from playwright.async_api import async_playwright

    _PLAYWRIGHT_AVAILABLE = True
except Exception:  # pylint: disable=broad-except
    _PLAYWRIGHT_AVAILABLE = False


class WebSpiderConnector(BaseConnector):
    """Lightweight crawler for websites, with optional Playwright rendering."""

    async def fetch(self) -> AsyncGenerator[Dict[str, Any], None]:
        if not self.source.url:
            raise ValueError("web connector requires source.url")

        depth_limit = int(self.source.crawl.get("depth", 0))
        max_pages = int(self.source.crawl.get("max_pages", 10))
        queue: deque[Tuple[str, int]] = deque([(str(self.source.url), 0)])
        seen: Set[str] = set()

        async with httpx.AsyncClient(timeout=30.0, headers=self._headers) as client:
            while queue and len(seen) < max_pages:
                url, depth = queue.popleft()
                if url in seen:
                    continue
                seen.add(url)

                html = await self._fetch_page(client, url)
                if not html:
                    continue

                yield {"url": url, "html": html}

                if depth < depth_limit:
                    for link in self._extract_links(url, html):
                        if link not in seen:
                            queue.append((link, depth + 1))

    async def normalize(self, payload: Dict[str, Any]) -> ContentItem:
        url = payload["url"]
        html = payload["html"]

        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        author = None
        author_tag = soup.find(attrs={"name": "author"})
        if author_tag:
            author = author_tag.get("content")

        extracted = trafilatura.extract(html, include_links=False, include_images=False)
        text = extracted or soup.get_text(separator=" ", strip=True)

        hashtags = extract_hashtags(text)
        keywords = extract_keywords(text)

        published_at = self._extract_date(soup)

        return ContentItem(
            id=uuid_from_url(url),
            source_id=self.source_id,
            url_canonical=url,
            author=author,
            title=title,
            text=text,
            summary=text[:5000],
            published_at=published_at,
            media_urls=[img.get("src") for img in soup.find_all("img") if img.get("src")],
            hashtags=hashtags,
            keywords=keywords,
            engagement_raw=Engagement(),
            metadata=self.metadata(),
        )

    @property
    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": "OpenScraper/1.0"}

    async def _fetch_page(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        use_render = bool(self.source.crawl.get("render", False)) and _PLAYWRIGHT_AVAILABLE
        if use_render:
            html = await self._render_with_playwright(url)
            if html:
                return html
        try:
            response = await client.get(url)
            response.raise_for_status()
            if _is_html(response):
                return response.text
        except httpx.HTTPError as exc:
            get_logger(__name__).warning("HTTP fetch failed", extra={"url": url, "error": str(exc)})
        if use_render:
            return await self._render_with_playwright(url)
        return None

    async def _render_with_playwright(self, url: str) -> Optional[str]:
        if not _PLAYWRIGHT_AVAILABLE:
            return None
        try:  # pragma: no cover - requires external browser binaries
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                except Exception as exc:  # pylint: disable=broad-except
                    get_logger(__name__).info("Playwright goto fallback", extra={"url": url, "error": str(exc)})
                html = await page.content()
                await browser.close()
                return html
        except Exception as exc:  # pylint: disable=broad-except
            get_logger(__name__).warning("Playwright render failed", extra={"url": url, "error": str(exc)})
            return None

    def _extract_links(self, base_url: str, html: str) -> Set[str]:
        soup = BeautifulSoup(html, "lxml")
        links: Set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if href.startswith("#"):
                continue
            absolute = urljoin(base_url, href)
            if urlparse(absolute).scheme in {"http", "https"}:
                links.add(absolute)
        return links

    def _extract_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            try:
                return datetime.fromisoformat(time_tag["datetime"].replace("Z", "+00:00"))
            except ValueError:
                pass
        meta_date = soup.find(attrs={"property": "article:published_time"})
        if meta_date and meta_date.get("content"):
            try:
                return datetime.fromisoformat(meta_date["content"].replace("Z", "+00:00"))
            except ValueError:
                pass
        return None


def _is_html(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "")
    return "text/html" in content_type
