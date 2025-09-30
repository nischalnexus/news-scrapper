# Open Source Web & Social Data Scraper Development Guide

This document describes how to build an end-to-end Python-based scraping framework using only open-source tools. The system will crawl public websites and social endpoints, normalize the data, extract clean text and media, score content, and present results through a CLI and a Streamlit dashboard.

---

## Tech Stack (Chosen)

- Scrapy: Base crawling framework (spiders, scheduling, pipelines)
- Playwright (Python): Fallback renderer for dynamic or JS-heavy pages
- trafilatura / readability-lxml: Clean article text extraction
- Beautiful Soup / lxml: Structured HTML parsing
- httpx or aiohttp: Async fetchers for connectors outside Scrapy
- requests-cache / diskcache: Local caching of HTTP GET requests
- DuckDB or SQLite: Local storage and analytics
- Typer / Click: CLI interface
- Streamlit: Lightweight dashboard UI

---

## Project Structure

```
scraper-project/
|-- connectors/
|   |-- base.py
|   |-- rss_feed.py
|   |-- web_spider.py
|   |-- mastodon.py
|   |-- reddit_public.py
|-- pipeline/
|   |-- fetcher.py
|   |-- parser.py
|   |-- normalizer.py
|   |-- scoring.py
|   |-- dedupe.py
|   |-- storage.py
|-- ui/
|   |-- cli.py
|   |-- app.py
|-- config/
|   |-- sources.yml
|   |-- settings.yml
|-- tests/
|   |-- test_spiders.py
|   |-- test_scoring.py
|-- data/
|   |-- db.duckdb
|   |-- cache/
|-- README.md
```

---

## Configuration

**`config/sources.yml`**
```yaml
sources:
  - type: rss
    url: https://example.com/feed
  - type: web
    url: https://example.com/blog
    crawl: { depth: 1 }
  - type: mastodon
    instance: mastodon.social
    handle: "@user"
  - type: reddit
    path: "/r/Python/"
```

**`config/settings.yml`**
```yaml
fetch:
  user_agent: "OpenScraper/1.0 (+contact@example.com)"
  max_concurrency: 20
  timeout_s: 20
  respect_robots: true

storage:
  engine: duckdb
  path: data/db.duckdb
  write_parquet: true

scoring:
  lambda_decay: 0.05
  weights:
    like: 1
    comment: 3
    share: 5
    view: 0.01
```

---

## Core Components

### 1. Crawling (Scrapy)
- Define spiders per source type (news sites, blogs, social endpoints)
- Use Scrapy pipelines for cleaning, deduplication, and storage
- Integrate Playwright middleware for rendering JS-heavy pages on demand

### 2. Parsing and Extraction
- trafilatura for full-article clean text
- Beautiful Soup or lxml for structured data (titles, authors, media)
- Normalize fields into a unified schema (title, text, author, timestamp, media)

### 3. Async Connectors
- httpx or aiohttp to fetch feeds and APIs (Reddit JSON, Mastodon, Hacker News)
- Apply common retry, rate-limiting, and error-handling policies

### 4. Caching
- requests-cache or diskcache to avoid redundant requests
- Configure cache expiry in `config/settings.yml`

### 5. Storage
- Persist normalized data in DuckDB or SQLite
- Support Parquet, CSV, and JSON exports for downstream tools

### 6. Scoring
- Viral score blends engagement volume with recency decay
- Trendy score measures short-term engagement momentum
- Quality score captures readability, media presence, and domain trust

### 7. CLI (Typer)
```bash
# Run ingestion
python -m ui.cli ingest --hours 24 --sources config/sources.yml

# Export filtered dataset
python -m ui.cli export --since 2025-09-20 --types post,news --out out/posts.csv

# Recompute scores
python -m ui.cli rescore --settings config/settings.yml
```

### 8. Streamlit UI
- Filters: timeframe, keywords, content types (post, image, news)
- Sliders: Viral score threshold, Trendy score threshold
- Table view with export buttons (CSV, JSON, Parquet)

---

## Trend Analysis Features

### Time-series analysis
- Use DuckDB or Pandas to compute rolling averages (likes per hour, shares per hour)
- Detect spikes with z-scores or rolling mean versus rolling standard deviation

### Hashtag or keyword trending
- Extract hashtags and keywords from normalized text fields
- Track frequency growth over time (example: "#AI" mentions up 150 percent in last 24 hours)

### Entity-based trends
- Use spaCy named entity recognition to extract brands, people, and places
- Rank entities by engagement growth and share momentum across connectors

---

## Viral Prediction Techniques (Open Source)

- Early engagement velocity: monitor interactions received within the first hours after publish time
- Similarity to past viral content: use TF-IDF vectors or sentence-transformer embeddings to compare against historical viral examples
- Clustering for hot topics: cluster embeddings (BERTopic, k-means) to reveal rapidly expanding topics and surface their top posts

---

## Streamlit Dashboard Extensions

- Add a Trending Now tab with a leaderboard of top posts by viral and trendy scores
- Trend over time charts: plot engagement velocity curves per keyword or topic
- Keyword heatmap: visualize which hashtags or keywords are heating up or cooling down
- Predictive view: highlight posts with high potential viral scores even when they are newly ingested

---

## Optional Add-ons

- Topic clustering via BERTopic for automatic content grouping and tagging
- Word clouds to visualize trending terms or entities
- Alerting: CLI flag or Streamlit notification when a post reaches Viral score greater than or equal to 80
- Daily digest export: generate a CSV or Markdown summary of the top 10 viral candidates in the last 24 hours

---

## Unified Data Schema

**Table: items**
- `id` (UUID)
- `source_id`
- `url_canonical`
- `author`
- `title`
- `text`
- `published_at`
- `media_urls`
- `engagement_raw`
- `viral_score`
- `trendy_score`
- `quality_score`

**Table: sources**
- `id`
- `type`
- `url_or_handle`
- `last_crawled_at`

---

## Testing

- pytest for unit and integration tests
- Recorded fixtures for spiders (VCR.py or betamax)
- Golden tests for scoring functions
- Smoke test on Hacker News plus two RSS feeds

---

## Roadmap

- v0.1: Scrapy plus trafilatura plus DuckDB plus CLI
- v0.2: Playwright integration plus Streamlit UI
- v0.3: Async connectors (Mastodon, Reddit, Hacker News)
- v0.4: Trend and viral scoring improvements, clustering

---

## Acceptance Criteria

- [ ] Ingest at least three connector types (RSS, web, Hacker News)
- [ ] Normalize and deduplicate with more than 95 percent accuracy
- [ ] Compute viral, trendy, and quality scores
- [ ] Export CSV, JSON, and Parquet from the CLI
- [ ] Streamlit UI supports interactive filters

---

## License

MIT License
