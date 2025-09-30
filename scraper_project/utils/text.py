from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Tuple

HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
WORD_RE = re.compile(r"[a-zA-Z0-9_]{3,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "your",
    "have",
    "will",
    "about",
    "http",
    "https",
    "www",
}


def extract_hashtags(text: str) -> List[str]:
    if not text:
        return []
    return sorted({tag.lower() for tag in HASHTAG_RE.findall(text)})


def extract_keywords(text: str, top_k: int = 15) -> List[str]:
    if not text:
        return []
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    filtered = [t for t in tokens if t not in STOPWORDS]
    ranking = Counter(filtered).most_common(top_k)
    return [word for word, _ in ranking]


def trending_growth(series: Iterable[int]) -> Tuple[float, float]:
    """Return (percent_growth, velocity) for a simple integer series."""

    values = list(series)
    if len(values) < 2:
        return 0.0, float(values[-1] if values else 0)
    previous, current = values[-2], values[-1]
    growth = 0.0
    if previous > 0:
        growth = ((current - previous) / previous) * 100.0
    velocity = current - previous
    return growth, float(velocity)
