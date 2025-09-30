from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import pandas as pd
from pydantic import ValidationError

from ..models import SourceConfig
from ..utils.logging import get_logger

REQUIRED_COLUMNS = ["Brand Name", "URL", "Category"]



RSS_FEED_OVERRIDES = {
    "wsj.com": "https://feeds.a.dj.com/rss/RSSWSJD.xml",
    "nytimes.com": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "reuters.com": "https://www.reuters.com/markets/technology/rss",
    "bbc.com": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "ft.com": "https://www.ft.com/feed",
}

_FEED_KEYWORDS = (
    "feed",
    "rss",
    "atom",
    "xml",
)

def _looks_like_feed(url: str) -> bool:
    lower = url.lower()
    return any(keyword in lower for keyword in _FEED_KEYWORDS)


RSS_FEED_OVERRIDES = {
    "wsj.com": "https://feeds.a.dj.com/rss/RSSWSJD.xml",
    "nytimes.com": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "reuters.com": "https://www.reuters.com/markets/technology/rss",
    "bbc.com": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "ft.com": "https://www.ft.com/rss/companies/technology",
}

def _normalize_domain(value: str) -> str:
    parsed = urlparse(value)
    host = (parsed.netloc or parsed.path).lower()
    if host.startswith("www."):
        host = host[4:]
    return host.rstrip("/")


def _candidate_domains(domain: str) -> List[str]:
    parts = [part for part in domain.split(".") if part]
    return [".".join(parts[i:]) for i in range(len(parts)) if len(parts[i:]) >= 2]


def _empty_catalog() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)




def _rename_catalog_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    fallback_by_index = {0: "Brand Name", 1: "URL", 2: "Category"}

    for idx, column in enumerate(df.columns):
        raw = column.strip()
        key = raw.lower()
        mapped = None
        if key in {"brand name", "brand", "source", "site", "brand_name"}:
            mapped = "Brand Name"
        elif key in {"url", "link", "website", "feed", "rss"}:
            mapped = "URL"
        elif key in {"category", "categories", "topic"}:
            mapped = "Category"
        elif raw == "" or key.startswith("unnamed"):
            mapped = fallback_by_index.get(idx)
        if mapped:
            rename_map[column] = mapped

    df = df.rename(columns=rename_map)

    for idx, required in enumerate(REQUIRED_COLUMNS):
        if required not in df.columns:
            source_col = fallback_by_index.get(idx)
            if source_col and source_col in df.columns:
                df[required] = df[source_col]
            else:
                df[required] = ""

    ordered_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered_cols]
    return df[ordered_cols + remaining]


def load_catalog_table(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return _empty_catalog()
    resolved = Path(path)
    if not resolved.exists():
        get_logger(__name__).warning("Catalog path not found", extra={"path": str(resolved)})
        return _empty_catalog()
    try:
        if resolved.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(resolved)
        elif resolved.suffix.lower() == ".csv":
            df = pd.read_csv(resolved)
        else:
            get_logger(__name__).warning("Unsupported catalog format", extra={"path": str(resolved)})
            return _empty_catalog()
    except Exception as exc:  # pragma: no cover - defensive
        get_logger(__name__).error("Failed to load catalog", extra={"path": str(resolved), "error": str(exc)})
        return _empty_catalog()

    df = _rename_catalog_columns(df)
    df = df.fillna("")
    for column in ["Brand Name", "URL", "Category"]:
        df[column] = df[column].astype(str).str.strip()
    return df.reset_index(drop=True)


def save_catalog_table(df: pd.DataFrame, path: str) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    df = _rename_catalog_columns(df)
    df = df.copy()
    df["URL_normalized"] = df["URL"].str.lower()
    df = df.drop_duplicates(subset="URL_normalized", keep="last")
    df = df.drop(columns=["URL_normalized"], errors="ignore")
    if resolved.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(resolved, index=False)
    elif resolved.suffix.lower() == ".csv":
        df.to_csv(resolved, index=False)
    else:
        raise ValueError(f"Unsupported catalog output format: {resolved.suffix}")


def ensure_category_row(df: pd.DataFrame, category: str) -> pd.DataFrame:
    category = (category or "").strip()
    if not category:
        return df
    mask = df["Category"].str.lower() == category.lower()
    if mask.any():
        return df
    row = {"Brand Name": "", "URL": "", "Category": category}
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def normalize_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if not parsed.scheme:
        value = f"https://{value}"
        parsed = urlparse(value)
    if not parsed.netloc:
        return value
    return value.rstrip("/")


def upsert_catalog_entry(df: pd.DataFrame, brand: str, url: str, category: str) -> pd.DataFrame:
    catalog = _rename_catalog_columns(df).copy()
    normalized_url = normalize_url(url)
    if not normalized_url:
        raise ValueError("URL is required for catalog entries")
    category_value = (category or "").strip()
    if not category_value:
        raise ValueError("Category is required for catalog entries")
    brand_value = (brand or "").strip()

    catalog["URL_lower"] = catalog["URL"].str.lower()
    mask = catalog["URL_lower"] == normalized_url.lower()
    row = {"Brand Name": brand_value, "URL": normalized_url, "Category": category_value}
    if mask.any():
        catalog.loc[mask, ["Brand Name", "URL", "Category"]] = [brand_value, normalized_url, category_value]
    else:
        catalog = pd.concat([catalog.drop(columns=["URL_lower"], errors="ignore"), pd.DataFrame([row])], ignore_index=True)
        catalog["URL_lower"] = catalog["URL"].str.lower()
    catalog = catalog.drop(columns=["URL_lower"], errors="ignore")
    catalog = ensure_category_row(catalog, category_value)
    return catalog.reset_index(drop=True)


def remove_catalog_entries(df: pd.DataFrame, urls: Sequence[str]) -> pd.DataFrame:
    if not urls:
        return df
    catalog = _rename_catalog_columns(df).copy()
    normalized = {normalize_url(url).lower() for url in urls if normalize_url(url)}
    if not normalized:
        return catalog
    catalog["URL_lower"] = catalog["URL"].str.lower()
    catalog = catalog[~catalog["URL_lower"].isin(normalized)]
    catalog = catalog.drop(columns=["URL_lower"], errors="ignore")
    return catalog.reset_index(drop=True)


def remove_catalog_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    category_value = (category or "").strip().lower()
    if not category_value:
        return df
    catalog = _rename_catalog_columns(df).copy()
    mask = catalog["Category"].str.lower() != category_value
    return catalog[mask].reset_index(drop=True)


@lru_cache(maxsize=8)
def load_category_map(path: Optional[str] = None) -> Dict[str, Dict[str, Optional[str]]]:
    if not path:
        return {}
    df = load_catalog_table(path)
    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    for _, row in df.iterrows():
        url = (row.get("URL") or "").strip()
        if not url:
            continue
        domain = _normalize_domain(url)
        if not domain:
            continue
        info = {
            "category": (row.get("Category") or "").strip() or None,
            "brand_name": (row.get("Brand Name") or "").strip() or None,
        }
        for candidate in _candidate_domains(domain):
            mapping.setdefault(candidate, info)
    return mapping




def _lookup_info(mapping: Dict[str, Dict[str, Optional[str]]], url: str) -> Optional[Dict[str, Optional[str]]]:
    domain = _normalize_domain(url)
    if not domain:
        return None
    for candidate in _candidate_domains(domain):
        info = mapping.get(candidate)
        if info:
            return info
    return None


def _feed_override(url: str) -> Optional[str]:
    domain = _normalize_domain(url)
    if not domain:
        return None
    for candidate in _candidate_domains(domain):
        feed = RSS_FEED_OVERRIDES.get(candidate)
        if feed:
            return feed
    return None


def enrich_sources_with_categories(
    sources: Iterable[SourceConfig],
    path: Optional[str] = None,
) -> List[SourceConfig]:
    mapping = load_category_map(path)
    enriched: List[SourceConfig] = []
    for source in sources:
        update: Dict[str, Optional[str]] = {}
        if source.url:
            info = _lookup_info(mapping, str(source.url))
        else:
            info = None
        if info:
            if info.get("category") and not getattr(source, "category", None):
                update["category"] = info["category"]
            if info.get("brand_name") and not getattr(source, "brand_name", None):
                update["brand_name"] = info["brand_name"]
        if update:
            enriched.append(source.model_copy(update=update))
        else:
            enriched.append(source)
    return enriched



def sources_from_catalog(
    path: Optional[str],
    *,
    default_crawl: Optional[Dict[str, int]] = None,
    default_tags: Optional[List[str]] = None,
) -> List[SourceConfig]:
    if not path:
        return []
    df = load_catalog_table(path)
    if df.empty:
        return []
    crawl_defaults = default_crawl or {"depth": 1, "max_pages": 10, "render": False}
    base_tags = default_tags or ["catalog"]
    sources: List[SourceConfig] = []
    for _, row in df.iterrows():
        raw_url = (row.get("URL") or "").strip()
        if not raw_url:
            continue
        url_value = normalize_url(raw_url)
        brand_value = (row.get("Brand Name") or "").strip() or None
        category_value = (row.get("Category") or "").strip() or None

        tags = list(base_tags)
        if category_value:
            tags.append(category_value.lower())

        if _looks_like_feed(url_value):
            payload = {
                "type": "rss",
                "url": url_value,
                "tags": tags + ["rss"],
                "brand_name": brand_value,
                "category": category_value,
            }
        else:
            feed_url = _feed_override(url_value)
            if feed_url:
                payload = {
                    "type": "rss",
                    "url": feed_url,
                    "tags": tags + ["rss"],
                    "brand_name": brand_value,
                    "category": category_value,
                }
            else:
                payload = {
                    "type": "web",
                    "url": url_value,
                    "crawl": crawl_defaults,
                    "tags": tags + ["web"],
                    "brand_name": brand_value,
                    "category": category_value,
                }
        try:
            sources.append(SourceConfig(**payload))
        except ValidationError as exc:  # pragma: no cover - defensive logging
            get_logger(__name__).warning(
                "Skipping invalid catalog entry",
                extra={"url": url_value, "error": str(exc)}
            )
    return sources
