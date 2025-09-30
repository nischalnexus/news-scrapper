from __future__ import annotations

from typing import Iterable, List

from ..models import ContentItem
from ..utils.logging import get_logger
from ..utils.text import extract_hashtags, extract_keywords

_NLP = None
try:  # pragma: no cover - optional heavy dependency
    import spacy  # type: ignore

    _NLP = spacy.load("en_core_web_sm")  # type: ignore
except Exception:  # pylint: disable=broad-except
    get_logger(__name__).info("spaCy model not available; entity extraction disabled")


def parse_items(items: Iterable[ContentItem]) -> List[ContentItem]:
    enriched: List[ContentItem] = []
    for item in items:
        if not item.hashtags:
            item.hashtags = extract_hashtags(item.text or "")
        if not item.keywords:
            item.keywords = extract_keywords(item.text or "")
        if _NLP:
            item.entities = _extract_entities(item.text or "")
        enriched.append(item)
    return enriched


def _extract_entities(text: str) -> List[str]:
    if not _NLP:
        return []
    doc = _NLP(text)
    return sorted({ent.text for ent in doc.ents if len(ent.text) > 2})
