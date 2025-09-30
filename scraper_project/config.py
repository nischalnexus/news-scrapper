from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Iterable

from .models import Settings, SourceConfig


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
    settings.sources = list(load_sources(sources_path))
    return settings
