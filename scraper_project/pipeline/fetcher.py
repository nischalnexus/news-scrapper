from __future__ import annotations

import asyncio
from typing import Dict, Iterable, List, Type

from ..models import ContentItem, SourceConfig
from ..utils.logging import get_logger
from .parser import parse_items

from ..connectors.base import BaseConnector
from ..connectors.mastodon import MastodonConnector
from ..connectors.reddit_public import RedditPublicConnector
from ..connectors.rss_feed import RSSFeedConnector
from ..connectors.web_spider import WebSpiderConnector

CONNECTOR_REGISTRY: Dict[str, Type[BaseConnector]] = {
    "rss": RSSFeedConnector,
    "web": WebSpiderConnector,
    "mastodon": MastodonConnector,
    "reddit": RedditPublicConnector,
}


async def run_connectors(sources: Iterable[SourceConfig]) -> List[ContentItem]:
    tasks = []
    for source in sources:
        connector_cls = CONNECTOR_REGISTRY.get(source.type)
        if not connector_cls:
            get_logger(__name__).warning("No connector registered", extra={"source_type": source.type})
            continue
        connector = connector_cls(source)
        tasks.append(asyncio.create_task(_run_connector(connector)))

    results: List[ContentItem] = []
    for task in asyncio.as_completed(tasks):
        results.extend(await task)
    return parse_items(results)


async def _run_connector(connector: BaseConnector) -> List[ContentItem]:
    get_logger(__name__).info("Running connector", extra={"connector": connector.__class__.__name__})
    items = await connector.run()
    return items
