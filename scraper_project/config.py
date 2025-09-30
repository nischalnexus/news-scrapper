from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Iterable

from .models import Settings, SourceConfig
from .utils.categories import enrich_sources_with_categories, load_category_map, sources_from_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_sources(path: Path) -> Iterable[SourceConfig]:
    payload = load_yaml(path)
    sources = payload.get("sources", [])
    for entry in sources:
        yield SourceConfig(**entry)


def load_settings(sources_path: Path, settings_path: Path) -> Settings:
    config = load_yaml(settings_path)
    settings = Settings(**config)

    raw_sources = list(load_sources(sources_path))
    category_path = settings.categories_path
    if category_path:
        category_path_obj = Path(category_path)
        if not category_path_obj.is_absolute():
            category_path_obj = (PROJECT_ROOT / category_path_obj).resolve()
        category_path = str(category_path_obj)
    settings.categories_path = category_path
    if category_path:
        load_category_map.cache_clear()

    combined_sources: list[SourceConfig] = []
    existing_keys: set[str] = set()
    for source in raw_sources:
        combined_sources.append(source)
        if source.url:
            existing_keys.add(str(source.url).lower())

    catalog_sources = sources_from_catalog(category_path)
    for source in catalog_sources:
        key = str(source.url).lower() if source.url else ""
        if key and key in existing_keys:
            continue
        combined_sources.append(source)
        if key:
            existing_keys.add(key)

    settings.sources = enrich_sources_with_categories(combined_sources, category_path)
    return settings
