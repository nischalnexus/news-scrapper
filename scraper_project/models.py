from __future__ import annotations

import json

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class SourceConfig(BaseModel):
    """Configuration for a single content source."""

    type: str = Field(..., description="Connector name such as rss, web, mastodon, reddit")
    url: Optional[HttpUrl] = Field(None, description="Primary URL for crawlers")
    crawl: Dict[str, Any] = Field(default_factory=dict, description="Crawl options (depth, schedule, etc.)")
    instance: Optional[str] = Field(None, description="Mastodon instance domain")
    handle: Optional[str] = Field(None, description="Mastodon account handle")
    path: Optional[str] = Field(None, description="Path component for Reddit or web connectors")
    tags: List[str] = Field(default_factory=list, description="Optional tags to attach to generated items")
    brand_name: Optional[str] = Field(None, description="Human-friendly source label")
    category: Optional[str] = Field(None, description="Category grouping label")
    fetch_interval_minutes: int = Field(60, ge=5, description="Suggested refresh cadence in minutes")


class FetchSettings(BaseModel):
    user_agent: str = Field("OpenScraper/1.0 (+contact@example.com)", description="HTTP User-Agent")
    max_concurrency: int = Field(10, ge=1, le=128)
    timeout_s: int = Field(20, ge=1, le=120)
    respect_robots: bool = True


class StorageSettings(BaseModel):
    engine: str = Field("duckdb", description="duckdb or sqlite")
    path: str = Field("data/db.duckdb", description="Database file path")
    write_parquet: bool = True
    parquet_dir: str = Field("data/exports", description="Directory for Parquet exports")


class ScoringWeights(BaseModel):
    like: float = 1.0
    comment: float = 3.0
    share: float = 5.0
    view: float = 0.01


class ScoringSettings(BaseModel):
    lambda_decay: float = Field(0.05, ge=0, description="Decay factor for freshness")
    weights: ScoringWeights = Field(default_factory=ScoringWeights)


class Settings(BaseModel):
    sources: List[SourceConfig] = Field(default_factory=list)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    categories_path: Optional[str] = Field("top100_tech_feeds_4cats.csv", description="Path to site/category catalog (csv or xlsx)")


class Engagement(BaseModel):
    like: int = 0
    comment: int = 0
    share: int = 0
    view: int = 0


class ContentItem(BaseModel):
    id: str
    source_id: str
    url_canonical: HttpUrl
    author: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    summary: Optional[str] = None
    published_at: Optional[datetime] = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    media_urls: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    engagement_raw: Engagement = Field(default_factory=Engagement)
    viral_score: float = 0.0
    trendy_score: float = 0.0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def dict_for_storage(self) -> Dict[str, Any]:
        """Flatten nested structures for storage layers that require primitives."""

        payload = self.model_dump(mode="python")
        payload["url_canonical"] = str(self.url_canonical)
        payload["source_id"] = str(self.source_id)
        payload["media_urls"] = ",".join(self.media_urls)
        payload["hashtags"] = ",".join(self.hashtags)
        payload["keywords"] = ",".join(self.keywords)
        payload["entities"] = ",".join(self.entities)
        engagement = self.engagement_raw
        payload["engagement_like"] = engagement.like
        payload["engagement_comment"] = engagement.comment
        payload["engagement_share"] = engagement.share
        payload["engagement_view"] = engagement.view
        payload["metadata"] = json.dumps(self.metadata)
        payload.pop("engagement_raw", None)
        return payload
