from __future__ import annotations

import abc
from typing import Any, Dict, Iterable, List, Optional

from ..models import ContentItem, SourceConfig


class BaseConnector(abc.ABC):
    """Abstract base connector."""

    def __init__(self, source: SourceConfig):
        self.source = source

    @property
    def source_id(self) -> str:
        return self.source.handle or self.source.path or (str(self.source.url) if self.source.url else "unknown")

    def metadata(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if getattr(self.source, "tags", None):
            meta["tags"] = self.source.tags
        if getattr(self.source, "category", None):
            meta["category"] = self.source.category
        if getattr(self.source, "brand_name", None):
            meta["brand_name"] = self.source.brand_name
        if extra:
            meta.update(extra)
        return meta

    @abc.abstractmethod
    async def fetch(self) -> Iterable[Dict[str, Any]]:
        """Return raw payloads for downstream parsing."""

    @abc.abstractmethod
    async def normalize(self, payload: Dict[str, Any]) -> ContentItem:
        """Convert a raw payload into a ContentItem."""

    async def run(self) -> List[ContentItem]:
        items: List[ContentItem] = []
        async for payload in _iter_async(self.fetch()):
            try:
                item = await self.normalize(payload)
                items.append(item)
            except Exception as exc:  # pragma: no cover - defensive fallback
                from ..utils.logging import get_logger

                get_logger(__name__).exception("Failed to normalize payload", extra={"error": exc})
        return items


async def _iter_async(value: Iterable[Any]):
    if hasattr(value, "__aiter__"):
        async for item in value:  # type: ignore[attr-defined]
            yield item
    else:
        for item in value:
            yield item
