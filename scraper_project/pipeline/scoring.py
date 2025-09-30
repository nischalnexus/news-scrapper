from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable, List

from ..models import ContentItem, ScoringSettings


def score_items(items: Iterable[ContentItem], settings: ScoringSettings) -> List[ContentItem]:
    scored: List[ContentItem] = []
    now = datetime.now(timezone.utc)
    for item in items:
        recency_hours = _recency_hours(item, now)
        engagement = item.engagement_raw
        weights = settings.weights
        base = (
            engagement.like * weights.like
            + engagement.comment * weights.comment
            + engagement.share * weights.share
            + engagement.view * weights.view
        )
        viral = base * math.exp(-settings.lambda_decay * recency_hours)
        trendy = _momentum(engagement)
        quality = _quality_score(item)
        item.viral_score = round(float(viral), 2)
        item.trendy_score = round(trendy, 2)
        item.quality_score = round(quality, 2)
        scored.append(item)
    return scored


def _recency_hours(item: ContentItem, now: datetime) -> float:
    if item.published_at:
        published = item.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return max((now - published).total_seconds() / 3600.0, 0.0)
    return 24.0


def _momentum(engagement) -> float:
    return engagement.like * 0.4 + engagement.comment * 0.8 + engagement.share * 1.2


def _quality_score(item: ContentItem) -> float:
    text_score = min(len(item.text or "") / 280.0, 1.0) * 40
    media_bonus = 20 if item.media_urls else 0
    keyword_bonus = min(len(item.keywords), 10) * 2
    entity_bonus = min(len(item.entities), 10) * 2
    return text_score + media_bonus + keyword_bonus + entity_bonus
