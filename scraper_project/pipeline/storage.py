from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import json
import duckdb
import pandas as pd

from ..models import ContentItem, Engagement, StorageSettings
from ..utils.logging import get_logger


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            url_canonical TEXT,
            author TEXT,
            title TEXT,
            text TEXT,
            summary TEXT,
            published_at TIMESTAMP,
            collected_at TIMESTAMP,
            media_urls TEXT,
            hashtags TEXT,
            keywords TEXT,
            entities TEXT,
            engagement_like INTEGER,
            engagement_comment INTEGER,
            engagement_share INTEGER,
            engagement_view INTEGER,
            viral_score DOUBLE,
            trendy_score DOUBLE,
            quality_score DOUBLE,
            metadata TEXT
        )
        """
    )


def write_items(items: Iterable[ContentItem], settings: StorageSettings) -> int:
    items = list(items)
    if not items:
        return 0
    db_path = Path(settings.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [item.dict_for_storage() for item in items]
    df = pd.DataFrame(payload)
    columns = [
        "id",
        "source_id",
        "url_canonical",
        "author",
        "title",
        "text",
        "summary",
        "published_at",
        "collected_at",
        "media_urls",
        "hashtags",
        "keywords",
        "entities",
        "engagement_like",
        "engagement_comment",
        "engagement_share",
        "engagement_view",
        "viral_score",
        "trendy_score",
        "quality_score",
        "metadata",
    ]
    df = df.reindex(columns=columns)

    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)

    for item_id in df["id"].tolist():
        conn.execute("DELETE FROM items WHERE id = ?", [item_id])

    conn.register("items_df", df)
    conn.execute(
        """
        INSERT INTO items SELECT * FROM items_df
        """
    )
    conn.unregister("items_df")
    conn.close()

    if settings.write_parquet:
        export_dir = Path(settings.parquet_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = export_dir / "items.parquet"
        df.to_parquet(parquet_path, index=False)
        get_logger(__name__).info("Wrote Parquet export", extra={"path": str(parquet_path)})

    return len(df)


def export_items(settings: StorageSettings, out_path: Path, query: str = "SELECT * FROM items") -> None:
    df = load_dataframe(settings, query)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    elif out_path.suffix == ".json":
        df.to_json(out_path, orient="records", lines=False)
    elif out_path.suffix in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False)
    else:
        raise ValueError("Unsupported export format")
    get_logger(__name__).info("Exported dataset", extra={"rows": len(df), "path": str(out_path)})


def load_dataframe(settings: StorageSettings, query: str = "SELECT * FROM items") -> pd.DataFrame:
    conn = duckdb.connect(str(settings.path))
    df = conn.execute(query).df()
    conn.close()
    return df


def load_items(settings: StorageSettings, query: str = "SELECT * FROM items") -> List[ContentItem]:
    df = load_dataframe(settings, query)
    items: List[ContentItem] = []
    for row in df.to_dict(orient="records"):
        engagement = Engagement(
            like=int(row.get("engagement_like") or 0),
            comment=int(row.get("engagement_comment") or 0),
            share=int(row.get("engagement_share") or 0),
            view=int(row.get("engagement_view") or 0),
        )
        metadata_value = row.get("metadata")
        try:
            metadata = json.loads(metadata_value) if metadata_value else {}
        except json.JSONDecodeError:
            metadata = {}
        item = ContentItem(
            id=row["id"],
            source_id=row.get("source_id", ""),
            url_canonical=row["url_canonical"],
            author=row.get("author"),
            title=row.get("title"),
            text=row.get("text"),
            summary=row.get("summary"),
            published_at=row.get("published_at"),
            collected_at=row.get("collected_at"),
            media_urls=(row.get("media_urls") or "").split(",") if row.get("media_urls") else [],
            hashtags=(row.get("hashtags") or "").split(",") if row.get("hashtags") else [],
            keywords=(row.get("keywords") or "").split(",") if row.get("keywords") else [],
            entities=(row.get("entities") or "").split(",") if row.get("entities") else [],
            engagement_raw=engagement,
            viral_score=row.get("viral_score", 0.0),
            trendy_score=row.get("trendy_score", 0.0),
            quality_score=row.get("quality_score", 0.0),
            metadata=metadata,
        )
        items.append(item)
    return items
