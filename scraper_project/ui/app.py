from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
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
    try:
        with st.spinner("Fetching fresh content..."):
            _ingest_and_store(cfg, ingest_hours)
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
            for col in ["title", "brand", "source_id", "published_at", "viral_score", "trendy_score", "quality_score", "url_canonical"]
            if col in top_v.columns
        ]
        top_v_display = top_v.reindex(columns=display_columns)
        if {"viral_score", "trendy_score"}.issubset(top_v_display.columns):
            st.dataframe(
                _styled_scores_table(top_v_display),
                use_container_width=True,
                hide_index=True,
                column_config=_score_table_column_config(),
            )
        else:
            st.dataframe(top_v_display, use_container_width=True, hide_index=True)

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
                for col in ["title", "brand", "source_id", "published_at", "viral_score", "trendy_score", "quality_score", "url_canonical"]
                if col in candidates_table.columns
            ]
            candidates_display = candidates_table.reindex(columns=display_columns)
            if {"viral_score", "trendy_score"}.issubset(candidates_display.columns):
                st.dataframe(
                    _styled_scores_table(candidates_display),
                    use_container_width=True,
                    hide_index=True,
                    column_config=_score_table_column_config(),
                )
            else:
                st.dataframe(candidates_display, use_container_width=True, hide_index=True)

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
                for col in ["title", "brand", "source_id", "published_at", "viral_score", "trendy_score", "url_canonical"]
                if col in digest.columns
            ]
            digest_display = digest.reindex(columns=digest_columns)
            if {"viral_score", "trendy_score"}.issubset(digest_display.columns):
                st.dataframe(
                    _styled_scores_table(digest_display),
                    use_container_width=True,
                    hide_index=True,
                    column_config=_score_table_column_config(),
                )
            else:
                st.dataframe(digest_display, use_container_width=True, hide_index=True)

