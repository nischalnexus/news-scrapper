from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import httpx
import pandas as pd
import tldextract
import typer
from bs4 import BeautifulSoup
from trafilatura import fetch_url as tf_fetch_url, extract as tf_extract
from trafilatura.settings import use_config
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser

EXCEL_PATH = Path("top100_tech_feeds_4cats.csv")
LOGS_DIR = Path("_scrape_test_logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
        "(compatible; TechScraper/1.0)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.7",
    "Connection": "keep-alive",
}

RETRY_STATUS = {429, 500, 502, 503, 504, 520, 522, 524}
MAX_RETRIES = 4
BASE_BACKOFF = 0.8
GLOBAL_CONCURRENCY = 10
PER_HOST_CONCURRENCY = 2
HTTP2_DEFAULT = False
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 20.0

CANDIDATE_FEEDS = (
    "feed",
    "rss",
    "rss.xml",
    "atom.xml",
    "index.xml",
    "feeds",
    "feed/",
    "category/technology/rss",
)

TRAF_CONF = use_config()
TRAF_CONF.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

_robot_cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}
_host_locks: Dict[str, asyncio.Semaphore] = {}


def registrable_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join([p for p in [ext.domain, ext.suffix] if p])


def _host_lock(host: str) -> asyncio.Semaphore:
    if host not in _host_locks:
        _host_locks[host] = asyncio.Semaphore(PER_HOST_CONCURRENCY)
    return _host_locks[host]


async def politeness_allowed(url: str, user_agent: str = "TechScraper/1.0") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    if robots_url not in _robot_cache:
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception:
            rp = None
        _robot_cache[robots_url] = rp
    rp = _robot_cache[robots_url]
    return True if rp is None else rp.can_fetch(user_agent, url)


async def politeness_delay() -> None:
    await asyncio.sleep(0.5)


def make_async_client(
    *,
    http2: bool = HTTP2_DEFAULT,
    verify: bool = True,
    follow_redirects: bool = True,
    max_connections: int = 40,
    max_keepalive_connections: int = 20,
    trust_env: bool = True,
) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=30.0,
    )
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=20.0,
        pool=30.0,
    )
    return httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        http2=http2,
        limits=limits,
        verify=verify,
        follow_redirects=follow_redirects,
        timeout=timeout,
        trust_env=trust_env,
    )


async def backoff_sleep(attempt: int) -> None:
    await asyncio.sleep(BASE_BACKOFF * (2 ** attempt) * random.uniform(0.5, 1.5))


async def get_with_retries(client: httpx.AsyncClient, url: str) -> httpx.Response:
    host = httpx.URL(url).host or ""
    lock = _host_lock(host)
    last_exc: Optional[Exception] = None
    async with lock:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.get(url)
                if response.status_code in RETRY_STATUS:
                    raise httpx.HTTPStatusError("retryable", request=response.request, response=response)
                return response
            except (
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError,
            ) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    await backoff_sleep(attempt)
                    continue
                raise last_exc


async def discover_feed(client: httpx.AsyncClient, base_url: str) -> Optional[str]:
    base = base_url.rstrip("/")
    for candidate in CANDIDATE_FEEDS:
        test_url = urljoin(base + "/", candidate)
        try:
            response = await client.get(test_url)
            content_type = response.headers.get("content-type", "").lower()
            if response.status_code == 200 and ("xml" in content_type or response.text.strip().startswith("<?xml")):
                return str(response.request.url)
        except Exception:
            continue
    try:
        response = await client.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.select("link[rel='alternate']"):
                link_type = (link.get("type") or "").lower()
                href = link.get("href")
                if href and ("rss" in link_type or "atom" in link_type or href.endswith((".xml", "/rss"))):
                    return urljoin(base_url, href)
    except Exception:
        pass
    return None


async def fetch_rss(feed_url: str) -> Tuple[str, List[Dict[str, Any]]]:
    def _parse() -> feedparser.FeedParserDict:
        return feedparser.parse(feed_url)

    parsed = await asyncio.to_thread(_parse)
    if getattr(parsed, "bozo", False) and not parsed.entries:
        raise RuntimeError(f"unable to parse feed: {feed_url}")
    feed_title = getattr(parsed.feed, "title", "")
    items: List[Dict[str, Any]] = []
    for entry in parsed.entries:
        items.append({
            "title": entry.get("title"),
            "link": entry.get("link"),
        })
    return feed_title, items


async def extract_with_trafilatura(url: str) -> Optional[str]:
    if not await politeness_allowed(url):
        return None
    await politeness_delay()
    downloaded = await asyncio.to_thread(tf_fetch_url, url, None, False, True)
    if not downloaded:
        return None
    text = await asyncio.to_thread(tf_extract, downloaded, False, False, TRAF_CONF)
    return text or None


async def bs4_title_probe(client: httpx.AsyncClient, url: str) -> Optional[str]:
    response = await get_with_retries(client, url)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


@dataclass
class SiteResult:
    brand: str
    url: str
    method: str
    ok: bool
    reason: str = ""
    items: int = 0
    title: str = ""


async def test_one_site(client: httpx.AsyncClient, brand: str, url: str) -> SiteResult:
    try:
        feed_url = await discover_feed(client, url)
        if feed_url or url.endswith(('.xml', '.rss', '.atom')) or 'feed' in url.lower():
            target_feed = feed_url or url
            try:
                feed_title, items = await fetch_rss(target_feed)
                return SiteResult(brand, url, method=f"rss:{target_feed}", ok=True, items=len(items), title=feed_title)
            except Exception:
                if feed_url:
                    pass
        try:
            extracted = await extract_with_trafilatura(url)
            if extracted and extracted.strip():
                return SiteResult(brand, url, method="trafilatura", ok=True, items=1, title=brand)
        except Exception:
            pass
        try:
            title = await bs4_title_probe(client, url)
            if title:
                return SiteResult(brand, url, method="httpx+bs4", ok=True, items=1, title=title)
        except Exception as exc:
            return SiteResult(brand, url, method="httpx+bs4", ok=False, reason=type(exc).__name__)
        return SiteResult(brand, url, method="none", ok=False, reason="no_feed_no_html")
    except Exception as exc:
        return SiteResult(brand, url, method="error", ok=False, reason=type(exc).__name__)


def clean_root(url: str) -> str:
    cleaned = url.strip().split('#')[0].split('?')[0]
    if cleaned.startswith("http://"):
        cleaned = "https://" + cleaned[len("http://"):]
    if "reddit.com/r/" in cleaned:
        return "https://www.reddit.com/"
    return cleaned


def load_catalog(max_sites: Optional[int] = None) -> List[Tuple[str, str]]:
    if EXCEL_PATH.exists():
        df = pd.read_csv(EXCEL_PATH)
        columns = {c.lower(): c for c in df.columns}
        brand_col = columns.get("brand") or columns.get("brand name") or list(df.columns)[0]
        url_col = columns.get("link") or columns.get("url") or list(df.columns)[1]
        df[url_col] = df[url_col].map(clean_root)
        df["_domain"] = df[url_col].map(registrable_domain)
        df = df.drop_duplicates("_domain")
        pairs = [(str(row[brand_col]), str(row[url_col])) for _, row in df.iterrows()]
    else:
        pairs = [
            ("TechCrunch", clean_root("https://techcrunch.com/")),
            ("The Verge", clean_root("https://www.theverge.com/")),
            ("Ars Technica", clean_root("https://arstechnica.com/")),
            ("Reddit", clean_root("https://www.reddit.com/")),
            ("GitHub", clean_root("https://github.com/")),
        ]
    if max_sites:
        pairs = pairs[:max_sites]
    return pairs


def save_results(results: List[SiteResult]) -> Dict[str, Any]:
    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    pd.DataFrame([r.__dict__ for r in results]).to_csv(LOGS_DIR / "summary.csv", index=False)
    return {
        "tested": len(results),
        "ok": ok_count,
        "fail": fail_count,
        "sample_fails": [r.__dict__ for r in results if not r.ok][:10],
    }


def log_result(result: SiteResult) -> None:
    with (LOGS_DIR / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")


async def run_catalog_test(max_sites: Optional[int] = None, http2: bool = HTTP2_DEFAULT) -> Dict[str, Any]:
    sites = load_catalog(max_sites)
    results: List[SiteResult] = []
    global_sem = asyncio.Semaphore(GLOBAL_CONCURRENCY)

    async with make_async_client(http2=http2) as client:
        async def _task(brand: str, url: str) -> SiteResult:
            async with global_sem:
                return await test_one_site(client, brand, url)

        tasks = [asyncio.create_task(_task(brand, url)) for brand, url in sites]
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
            log_result(result)

    return save_results(results)


def main(
    max_sites: Optional[int] = typer.Option(None, help="Limit number of sites to test"),
    http2: bool = typer.Option(False, help="Enable HTTP/2 support"),
) -> None:
    summary = asyncio.run(run_catalog_test(max_sites=max_sites, http2=http2))
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)
