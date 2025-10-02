from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

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

def _source_key(source: SourceConfig) -> str:
    if source.url:
        return f"url:{str(source.url).lower()}"
    if source.handle:
        return f"handle:{str(source.handle).lower()}@{(source.instance or '').lower()}"
    if source.path:
        return f"path:{str(source.path).lower()}"
    return f"type:{source.type.lower()}"



def load_settings(
    sources_path: Path,
    settings_path: Path,
    *,
    include_catalog: bool = True,
    include_custom: bool = True,
    custom_sources_path: Path | None = None,
) -> Settings:
    config = load_yaml(settings_path)
    settings = Settings(**config)

    raw_sources = list(load_sources(sources_path))
    if include_custom:
        custom_path = custom_sources_path
        if custom_path is None:
            custom_path = PROJECT_ROOT / "config/custom_category.yml"
        if not custom_path.is_absolute():
            custom_path = (PROJECT_ROOT / custom_path).resolve()
        else:
            custom_path = custom_path.resolve()
        if custom_path.exists():
            raw_sources.extend(load_sources(custom_path))

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
    existing_keys: Set[str] = set()
    for source in raw_sources:
        key = _source_key(source)
        if key in existing_keys:
            continue
        combined_sources.append(source)
        existing_keys.add(key)

    if include_catalog:
        catalog_sources = sources_from_catalog(category_path)
        for source in catalog_sources:
            key = _source_key(source)
            if key in existing_keys:
                continue
            combined_sources.append(source)
            existing_keys.add(key)

    settings.sources = enrich_sources_with_categories(combined_sources, category_path)
    return settings
