from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer

from ..config import load_settings
from ..models import ContentItem, Settings
from ..pipeline.runner import ingest_async
from ..pipeline.scoring import score_items
from ..pipeline.storage import export_items, load_items, load_dataframe, write_items
from ..utils.logging import configure_logging
from ..utils.async_tools import ensure_proactor_event_loop_policy

app = typer.Typer(add_completion=False, help="CLI utilities for the scraper project")


@app.command()
def ingest(
    sources: Path = typer.Option(Path("config/sources.yml"), help="Path to sources configuration"),
    settings: Path = typer.Option(Path("config/settings.yml"), help="Path to runtime settings"),
    hours: int = typer.Option(24, min=1, help="Only keep items published within the last N hours"),
    include_catalog: bool = typer.Option(True, help="Include catalog-derived sources in addition to the explicit list"),
    include_custom: bool = typer.Option(True, help="Include custom category sources"),
    custom_sources: Optional[Path] = typer.Option(None, help="Optional path to custom sources file"),
    dry_run: bool = typer.Option(False, help="Do not persist results to storage"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Run connectors, compute scores, and store results."""

    configure_logging(log_level)
    cfg = load_settings(
        sources,
        settings,
        include_catalog=include_catalog,
        include_custom=include_custom,
        custom_sources_path=custom_sources,
    )

    async def _run() -> list[ContentItem]:
        items = await ingest_async(cfg)
        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            items = [item for item in items if not item.published_at or item.published_at >= cutoff]
        if dry_run:
            typer.echo(f"Fetched {len(items)} items (dry-run, not persisted)")
        else:
            write_items(items, cfg.storage)
            typer.echo(f"Persisted {len(items)} items to {cfg.storage.path}")
        return items

    ensure_proactor_event_loop_policy()
    asyncio.run(_run())


@app.command()
def export(
    out: Path = typer.Argument(..., help="Destination file (csv, json, parquet)"),
    settings: Path = typer.Option(Path("config/settings.yml"), help="Settings for storage connection"),
    query: str = typer.Option("SELECT * FROM items", help="DuckDB SQL query to run before export"),
):
    """Export stored items to a file."""

    cfg = load_settings(Path("config/sources.yml"), settings)
    export_items(cfg.storage, out, query)
    typer.echo(f"Exported data to {out}")


@app.command()
def rescore(
    settings: Path = typer.Option(Path("config/settings.yml"), help="Settings for storage connection"),
    query: str = typer.Option("SELECT * FROM items", help="Subset of items to rescore"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Recompute viral, trendy, and quality scores using current settings."""

    configure_logging(log_level)
    cfg = load_settings(Path("config/sources.yml"), settings)
    items = load_items(cfg.storage, query)
    rescored = score_items(items, cfg.scoring)
    write_items(rescored, cfg.storage)
    typer.echo(f"Rescored {len(rescored)} items")


@app.command()
def show(
    settings: Path = typer.Option(Path("config/settings.yml"), help="Settings for storage connection"),
    limit: int = typer.Option(10, help="Number of rows to display"),
    order_by: str = typer.Option("viral_score DESC", help="Order clause"),
):
    """Quickly preview top results in the terminal."""

    cfg = load_settings(Path("config/sources.yml"), settings)
    df = load_dataframe(cfg.storage, f"SELECT * FROM items ORDER BY {order_by} LIMIT {limit}")
    if df.empty:
        typer.echo("No items available")
    else:
        typer.echo(df[["title", "viral_score", "trendy_score", "quality_score"]])


if __name__ == "__main__":  # pragma: no cover
    app()
