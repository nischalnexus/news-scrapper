from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import json
from urllib.parse import urlparse
import html
import re

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from scraper_project.config import load_settings
from scraper_project.pipeline.runner import ingest_async
from scraper_project.pipeline.storage import load_dataframe, write_items

from scraper_project.utils.categories import (
    ensure_category_row,
    load_catalog_table,
    load_category_map,
    remove_catalog_category,
    remove_catalog_entries,
    save_catalog_table,
    upsert_catalog_entry,
)

def _brand_from_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path
    host = host.split("@")[-1]
    return host.lstrip("www.")

@dataclass(frozen=True)
class ScoreBadge:
    label: str
    color: str
    tooltip: str

_SCORE_BUCKETS = [
    (750, ScoreBadge("Ultra", "#f0abfc", "Ultra: score >= 750 - breakout viral signal")),
    (400, ScoreBadge("High", "#c7d2fe", "High: score 400-749 - strong viral momentum")),
    (200, ScoreBadge("Medium", "#bbf7d0", "Medium: score 200-399 - consistent engagement")),
    (50, ScoreBadge("Warm", "#fde68a", "Warm: score 50-199 - early traction")),
    (0, ScoreBadge("Cool", "#bae6fd", "Cool: score < 50 - low momentum")),
]

_NEUTRAL_BADGE = ScoreBadge("Neutral", "#e5e7eb", "No score yet - awaiting processing")

def _text_color_for_background(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return "#111827"
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#111827" if luminance > 0.6 else "#f9fafb"

def _score_badge_meta(score: float) -> ScoreBadge:
    if score is None or pd.isna(score):
        return _NEUTRAL_BADGE
    for threshold, badge in _SCORE_BUCKETS:
        if score >= threshold:
            return badge
    return _SCORE_BUCKETS[-1][1]

def _score_badge(score: float) -> str:
    return _score_badge_meta(score).label

_SCORE_BAND_HELP = (
    ":purple_circle: **Ultra**  score = 750 (breakout viral signal)\n"
    ":blue_circle: **High**  score 400-749 (strong viral momentum)\n"
    ":green_circle: **Medium**  score 200-399 (consistent engagement)\n"
    ":yellow_circle: **Warm**  score 50-199 (early traction)\n"
    ":white_circle: **Cool**  score < 50 (low momentum)\n"
)

_TAG_RE = re.compile(r"<[^>]+>")

def _clean_text_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text_value = html.unescape(str(value))
    cleaned = _TAG_RE.sub("", text_value)
    return cleaned.strip()


def _parse_metadata_cell(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        payload = value.strip()
        if not payload:
            return {}
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:  # pragma: no cover - defensive parsing
            return {}
    return {}


def _metadata_tags(meta: dict[str, object]) -> str:
    tags = meta.get("tags")
    if isinstance(tags, list):
        return ",".join(str(tag) for tag in tags)
    if isinstance(tags, str):
        return tags
    return ""


def _category_label(value: str | None) -> str:
    return value or "Uncategorized"


def _styled_scores_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()

    if "url_canonical" in display.columns:
        display = display.rename(columns={"url_canonical": "url"})
    if "brand_name" in display.columns:
        display = display.rename(columns={"brand_name": "brand"})
    if "source_id" in display.columns:
        display = display.rename(columns={"source_id": "source"})
    if "published_at_display" in display.columns and "published_at" not in display.columns:
        display = display.rename(columns={"published_at_display": "published_at"})

    text_like_cols = [
        col
        for col in display.columns
        if display[col].dtype == object or col in {"title", "brand", "source", "published_at", "url"}
    ]
    for col in text_like_cols:
        display[col] = display[col].apply(_clean_text_cell)

    if "viral_score" in display.columns:
        display["viral_band"] = display["viral_score"].apply(_score_badge)
    if "trendy_score" in display.columns:
        display["trendy_band"] = display["trendy_score"].apply(_score_badge)

    preferred_order = [
        "title",
        "brand",
        "source",
        "published_at",
        "viral_score",
        "viral_band",
        "trendy_score",
        "trendy_band",
        "quality_score",
        "url",
    ]
    ordered_columns = [col for col in preferred_order if col in display.columns]
    ordered_columns += [col for col in display.columns if col not in ordered_columns]

    return display.loc[:, ordered_columns]

def _score_table_column_config() -> dict[str, st.column_config.Column]:
    return {
        "url": st.column_config.LinkColumn("Link", display_text="Open", max_chars=60),
        "viral_band": st.column_config.TextColumn("Viral Band", help=_SCORE_BAND_HELP),
        "trendy_band": st.column_config.TextColumn("Trendy Band", help=_SCORE_BAND_HELP),
    }

def _ingest_and_store(cfg, ingest_hours: int) -> None:
    items = asyncio.run(ingest_async(cfg))
    if ingest_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ingest_hours)
        items = [item for item in items if not item.published_at or item.published_at >= cutoff]
    write_items(items, cfg.storage)

st.set_page_config(page_title="Open Scraper Dashboard", layout="wide")
st.title("Open Source Web & Social Data Scraper")

settings_path = Path(st.sidebar.text_input("Settings path", "config/settings.yml"))
sources_path = Path(st.sidebar.text_input("Sources path", "config/sources.yml"))
limit = st.sidebar.slider("Rows to load", 100, 10000, 1000, step=100)

cfg = load_settings(sources_path, settings_path)

catalog_path = getattr(cfg, "categories_path", None)
catalog_df = load_catalog_table(catalog_path)

raw_category_values = set()
has_uncategorized = False
for source in cfg.sources:
    value = getattr(source, "category", None)
    if value:
        raw_category_values.add(str(value))
    else:
        has_uncategorized = True

if not catalog_df.empty:
    catalog_categories = catalog_df["Category"].astype(str).str.strip()
    raw_category_values.update(cat for cat in catalog_categories if cat)
    if (catalog_categories == "").any():
        has_uncategorized = True

category_options = sorted(raw_category_values)
if has_uncategorized:
    if "Uncategorized" not in category_options:
        category_options.append("Uncategorized")
if not category_options:
    category_options = ["Uncategorized"]
prev_category_selection = st.session_state.get("category_filter_selection")
if prev_category_selection:
    category_default = [value for value in prev_category_selection if value in category_options]
    if not category_default:
        category_default = category_options
else:
    category_default = category_options
category_filter_selection = st.sidebar.multiselect("Categories", category_options, default=category_default)
selected_categories = category_filter_selection or category_options

if catalog_path:
    with st.sidebar.expander("Manage categories & sites"):
        st.caption(f"Catalog file: {Path(catalog_path).name}")
        st.markdown("Add or update websites for each category.")
        new_category_input = st.text_input("Category name", key="catalog_category_name")
        new_brand_input = st.text_input("Brand / source name", key="catalog_brand_name")
        new_url_input = st.text_input("Website URL", key="catalog_site_url")
        manage_cols = st.columns(2)
        with manage_cols[0]:
            if st.button("Add / Update site", key="catalog_add_site_btn"):
                try:
                    updated_catalog = upsert_catalog_entry(catalog_df, new_brand_input, new_url_input, new_category_input)
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    save_catalog_table(updated_catalog, catalog_path)
                    load_category_map.cache_clear()
                    st.success("Catalog updated.")
                    st.experimental_rerun()
        with manage_cols[1]:
            if st.button("Add category", key="catalog_add_category_btn"):
                if not new_category_input.strip():
                    st.warning("Provide a category name to add.")
                else:
                    updated_catalog = ensure_category_row(catalog_df, new_category_input)
                    save_catalog_table(updated_catalog, catalog_path)
                    load_category_map.cache_clear()
                    st.success("Category added.")
                    st.experimental_rerun()
        existing_categories = sorted({cat for cat in catalog_df["Category"].astype(str).str.strip() if cat})
        if existing_categories:
            delete_category_choice = st.selectbox("Select category", options=existing_categories, key="catalog_delete_category_select")
            category_rows = catalog_df[catalog_df["Category"].astype(str).str.lower() == delete_category_choice.lower()]
            site_options = {
                f"{(row['Brand Name'] or row['URL'])} ({row['URL']})": row['URL']
                for _, row in category_rows.iterrows()
                if str(row.get('URL', '')).strip()
            }
            selected_sites_to_delete = st.multiselect("Websites to delete", list(site_options.keys()), key="catalog_delete_sites_select")
            if st.button("Delete selected sites", key="catalog_delete_sites_btn"):
                updated_catalog = remove_catalog_entries(catalog_df, [site_options[label] for label in selected_sites_to_delete])
                save_catalog_table(updated_catalog, catalog_path)
                load_category_map.cache_clear()
                st.success("Selected sites removed.")
                st.experimental_rerun()
            if st.button("Delete category", key="catalog_delete_category_btn"):
                updated_catalog = remove_catalog_category(catalog_df, delete_category_choice)
                save_catalog_table(updated_catalog, catalog_path)
                load_category_map.cache_clear()
                st.success("Category removed.")
                st.experimental_rerun()
        else:
            st.info("No categories defined yet; add one above.")
else:
    st.sidebar.info("Set a catalog path in settings to manage categories.")


stored_filters = st.session_state.get("filters_apply", {})
keyword_default = st.session_state.get("keyword_filter", "")
hashtag_default = st.session_state.get("hashtag_filter", "")

ingest_default = int(stored_filters.get("ingest_hours", 24))
ingest_hours = st.sidebar.slider("Fetch window (hours)", 1, 168, ingest_default)

time_unit_default = stored_filters.get("time_unit", "Hours")
time_unit = st.sidebar.selectbox(
    "Time window unit",
    ["Hours", "Minutes"],
    index=0 if time_unit_default == "Hours" else 1,
)

if time_unit == "Hours":
    time_value_default = int(stored_filters.get("time_value", 24))
    time_window_value = st.sidebar.slider("Time window (hours)", 1, 168, time_value_default)
else:
    time_value_default = int(stored_filters.get("time_value", 60))
    time_window_value = st.sidebar.slider("Time window (minutes)", 5, 720, time_value_default)

keyword_filter = st.sidebar.text_input("Keyword contains", value=keyword_default)
hashtag_filter = st.sidebar.text_input("Hashtag contains", value=hashtag_default)

if st.sidebar.button("Apply"):
    st.session_state["filters_apply"] = {
        "ingest_hours": ingest_hours,
        "time_unit": time_unit,
        "time_value": time_window_value,
    }
    st.session_state["keyword_filter"] = keyword_filter
    st.session_state["hashtag_filter"] = hashtag_filter
    st.session_state["category_filter_selection"] = selected_categories
    try:
        effective_categories = set(selected_categories or category_options)
        sources_for_ingest = [
            source
            for source in cfg.sources
            if _category_label(getattr(source, "category", None)) in effective_categories
        ]
        if not sources_for_ingest:
            raise RuntimeError("No sources match the selected categories.")
        cfg_ingest = cfg.model_copy(deep=True)
        cfg_ingest.sources = sources_for_ingest
        with st.spinner("Fetching fresh content..."):
            _ingest_and_store(cfg_ingest, ingest_hours)
        st.session_state["ingest_success"] = True
    except Exception as exc:  # pragma: no cover - defensive UI handling
        st.session_state["ingest_error"] = str(exc)
    st.session_state["last_ingest"] = datetime.now(timezone.utc).isoformat()

if st.session_state.pop("ingest_success", False):
    st.success("Latest content ingested.")
    st.caption(f"Last ingest: {st.session_state.get('last_ingest', 'unknown')}")
if err := st.session_state.pop("ingest_error", None):
    st.error(f"Ingestion failed: {err}")

df = load_dataframe(cfg.storage, f"SELECT * FROM items ORDER BY published_at DESC LIMIT {limit}")

if df.empty:
    st.info("No data available. Click Apply to ingest content.")
    st.stop()

now_utc = pd.Timestamp.now(tz="UTC")

metadata_series = pd.Series([{}] * len(df), index=df.index)
if "metadata" in df.columns:
    metadata_series = df["metadata"].apply(_parse_metadata_cell)
df["category"] = metadata_series.apply(lambda meta: _category_label(meta.get("category")))
df["source_tags"] = metadata_series.apply(_metadata_tags)
brand_from_meta = metadata_series.apply(lambda meta: (meta.get("brand_name") or "").strip())

if "published_at" in df.columns:
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["published_hour"] = df["published_at"].dt.floor("h")
    df["published_at_display"] = df["published_at"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
else:
    df["published_at_display"] = ""

if "collected_at" in df.columns:
    df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce", utc=True)

if "published_at_display" in df.columns:
    df["published_at_display"] = df["published_at_display"].fillna("")

if "url_canonical" in df.columns:
    df["brand_name"] = df["url_canonical"].apply(_brand_from_url)
else:
    df["brand_name"] = ""
mask_brand_meta = brand_from_meta != ""
if mask_brand_meta.any():
    df.loc[mask_brand_meta, "brand_name"] = brand_from_meta[mask_brand_meta]

if "title" in df.columns:
    df["display_title"] = df["title"].fillna("")
else:
    df["display_title"] = ""

missing_mask = df["display_title"].astype(str).str.strip() == ""
df.loc[missing_mask, "display_title"] = df.loc[missing_mask, "brand_name"]
if "source_id" in df.columns:
    still_missing = df["display_title"].astype(str).str.strip() == ""
    df.loc[still_missing, "display_title"] = df.loc[still_missing, "source_id"].fillna(df.loc[still_missing, "brand_name"])

if time_unit == "Hours":
    past_window = pd.Timedelta(hours=time_window_value)
else:
    past_window = pd.Timedelta(minutes=time_window_value)
recent_cutoff = now_utc - past_window

filtered = df.copy()
if selected_categories:
    filtered = filtered[filtered["category"].isin(selected_categories)]
elif category_options:
    filtered = filtered[filtered["category"].isin(category_options)]
if keyword_filter and "keywords" in filtered.columns:
    filtered = filtered[filtered["keywords"].str.contains(keyword_filter, case=False, na=False)]
if hashtag_filter and "hashtags" in filtered.columns:
    filtered = filtered[filtered["hashtags"].str.contains(hashtag_filter, case=False, na=False)]
if "published_at" in filtered.columns:
    filtered = filtered[filtered["published_at"] >= recent_cutoff]

trending_tab, trends_tab, heatmap_tab, predictive_tab = st.tabs([
    "Trending Now",
    "Trends Over Time",
    "Keyword Heatmap",
    "Predictive View",
])

with trending_tab:
    st.subheader("Top Posts")
    if "viral_score" not in filtered.columns:
        st.info("Viral scores unavailable.")
    else:
        top_columns = [
            "display_title",
            "brand_name",
            "source_id",
            "category",
            "published_at_display",
            "viral_score",
            "trendy_score",
            "quality_score",
            "url_canonical",
        ]
        available_top_columns = [col for col in top_columns if col in filtered.columns]
        top_v = (
            filtered.sort_values("viral_score", ascending=False)
            .head(20)[available_top_columns]
            .rename(
                columns={
                    "display_title": "title",
                    "brand_name": "brand",
                    "published_at_display": "published_at",
                }
            )
        )
        display_columns = [
            col
            for col in ["title", "brand", "source_id", "category", "published_at", "viral_score", "trendy_score", "quality_score", "url_canonical"]
            if col in top_v.columns
        ]
        top_v_display = top_v.reindex(columns=display_columns)
        if {"viral_score", "trendy_score"}.issubset(top_v_display.columns):
            st.dataframe(
                _styled_scores_table(top_v_display),
                width='stretch',
                hide_index=True,
                column_config=_score_table_column_config(),
            )
        else:
            st.dataframe(top_v_display, width='stretch', hide_index=True)

    st.subheader("Leaderboard (avg viral score)")
    if {"source_id", "viral_score"}.issubset(filtered.columns) and not filtered.empty:
        leaderboard = (
            filtered.groupby("source_id")["viral_score"].mean().sort_values(ascending=False).head(10)
        )
        st.bar_chart(leaderboard)
    else:
        st.info("Source or viral score data unavailable for leaderboard.")

with trends_tab:
    st.subheader("Engagement Velocity per Keyword")
    if (
        "published_hour" in filtered.columns
        and "keywords" in filtered.columns
        and not filtered.empty
    ):
        exploded = filtered.assign(keywords=filtered["keywords"].str.split(",")).explode("keywords")
        exploded["keywords"] = exploded["keywords"].str.strip()
        exploded = exploded[exploded["keywords"].notnull() & (exploded["keywords"] != "")]
        if not exploded.empty:
            keyword_counts = (
                exploded.groupby(["published_hour", "keywords"]).size().reset_index(name="count")
            )
            chart_data = keyword_counts.pivot(index="published_hour", columns="keywords", values="count").fillna(0)
            st.line_chart(chart_data)
        else:
            st.info("No keyword counts available for the selected window.")
    else:
        st.info("Published timestamps or keywords unavailable for trend chart")

with heatmap_tab:
    st.subheader("Hashtags Heating Up")
    if "hashtags" in filtered.columns and not filtered.empty:
        exploded_tags = filtered.assign(hashtags=filtered["hashtags"].str.split(",")).explode("hashtags")
        exploded_tags["hashtags"] = exploded_tags["hashtags"].str.strip()
        exploded_tags = exploded_tags[exploded_tags["hashtags"].notnull() & (exploded_tags["hashtags"] != "")]
        if not exploded_tags.empty and "published_at" in exploded_tags.columns:
            recent = exploded_tags[exploded_tags["published_at"] >= recent_cutoff]
            previous = exploded_tags[
                (exploded_tags["published_at"] < recent_cutoff)
                & (exploded_tags["published_at"] >= (recent_cutoff - past_window))
            ]
            recent_counts = recent.groupby("hashtags").size() if not recent.empty else pd.Series(dtype=int)
            previous_counts = previous.groupby("hashtags").size() if not previous.empty else pd.Series(dtype=int)
            growth = (recent_counts - previous_counts).fillna(0).sort_values(ascending=False).head(25)
            heatmap_df = pd.DataFrame({"growth": growth})
            st.bar_chart(heatmap_df)
        else:
            st.info("No hashtags available yet")
    else:
        st.info("No hashtags available yet")

with predictive_tab:
    st.subheader("High Potential Viral Posts")
    threshold = st.slider("Potential viral score >=", 0.0, 500.0, 80.0)
    if "viral_score" not in df.columns:
        st.info("Viral scores unavailable.")
    else:
        if "published_at" in df.columns:
            candidates = df[df["viral_score"] >= threshold].sort_values("published_at", ascending=False)
        else:
            candidates = df[df["viral_score"] >= threshold]
        if candidates.empty:
            st.info("No candidates above the selected threshold.")
        else:
            candidate_columns = [
                "display_title",
                "brand_name",
                "source_id",
                "category",
                "published_at_display",
                "viral_score",
                "trendy_score",
                "quality_score",
                "url_canonical",
            ]
            available_candidate_columns = [col for col in candidate_columns if col in candidates.columns]
            candidates_table = (
                candidates[available_candidate_columns]
                .rename(
                    columns={
                        "display_title": "title",
                        "brand_name": "brand",
                        "published_at_display": "published_at",
                    }
                )
            )
            display_columns = [
                col
                for col in ["title", "brand", "source_id", "category", "published_at", "viral_score", "trendy_score", "quality_score", "url_canonical"]
                if col in candidates_table.columns
            ]
            candidates_display = candidates_table.reindex(columns=display_columns)
            if {"viral_score", "trendy_score"}.issubset(candidates_display.columns):
                st.dataframe(
                    _styled_scores_table(candidates_display),
                    width='stretch',
                    hide_index=True,
                    column_config=_score_table_column_config(),
                )
            else:
                st.dataframe(candidates_display, width='stretch', hide_index=True)

    st.subheader("Daily Digest Preview")
    if "published_at" not in df.columns:
        st.info("No posts in the current digest window.")
    else:
        digest_source = df[df["published_at"] >= recent_cutoff]
        digest = (
            digest_source.sort_values(["viral_score", "trendy_score"], ascending=False)
            .head(10)[
                [
                    col
                    for col in [
                        "display_title",
                        "brand_name",
                        "source_id",
                        "published_at_display",
                        "viral_score",
                        "trendy_score",
                        "url_canonical",
                    ]
                    if col in digest_source.columns
                ]
            ]
            .rename(
                columns={
                    "display_title": "title",
                    "brand_name": "brand",
                    "published_at_display": "published_at",
                }
            )
        )
        if digest.empty:
            st.info("No posts in the current digest window.")
        else:
            digest_columns = [
                col
                for col in ["title", "brand", "source_id", "category", "published_at", "viral_score", "trendy_score", "url_canonical"]
                if col in digest.columns
            ]
            digest_display = digest.reindex(columns=digest_columns)
            if {"viral_score", "trendy_score"}.issubset(digest_display.columns):
                st.dataframe(
                    _styled_scores_table(digest_display),
                    width='stretch',
                    hide_index=True,
                    column_config=_score_table_column_config(),
                )
            else:
                st.dataframe(digest_display, width='stretch', hide_index=True)


