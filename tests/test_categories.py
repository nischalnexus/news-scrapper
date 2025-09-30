import pandas as pd

from scraper_project.models import SourceConfig
from scraper_project.utils.categories import (
    ensure_category_row,
    enrich_sources_with_categories,
    load_catalog_table,
    remove_catalog_entries,
    save_catalog_table,
    sources_from_catalog,
    upsert_catalog_entry,
)



def test_enrich_sources_with_direct_domain(tmp_path):
    csv_path = tmp_path / "categories.csv"
    csv_path.write_text(
        "Brand Name,URL,Category\nTechCrunch,https://techcrunch.com/,Startups\n",
        encoding="utf-8",
    )

    source = SourceConfig(type="web", url="https://techcrunch.com/")
    enriched = enrich_sources_with_categories([source], str(csv_path))

    assert enriched[0].category == "Startups"
    assert enriched[0].brand_name == "TechCrunch"


def test_enrich_sources_with_subdomain_match(tmp_path):
    csv_path = tmp_path / "categories.csv"
    csv_path.write_text(
        "Brand Name,URL,Category\nArs Technica,https://arstechnica.com/,Tech Publication\n",
        encoding="utf-8",
    )

    source = SourceConfig(type="rss", url="https://feeds.arstechnica.com/arstechnica/index/")
    enriched = enrich_sources_with_categories([source], str(csv_path))

    assert enriched[0].category == "Tech Publication"
    assert enriched[0].brand_name == "Ars Technica"


def test_enrich_sources_without_match(tmp_path):
    csv_path = tmp_path / "categories.csv"
    csv_path.write_text(
        "Brand Name,URL,Category\nTechCrunch,https://techcrunch.com/,Startups\n",
        encoding="utf-8",
    )

    source = SourceConfig(type="web", url="https://unknown.example.com/")
    enriched = enrich_sources_with_categories([source], str(csv_path))

    assert enriched[0].category is None
    assert enriched[0].brand_name is None
def test_sources_from_catalog_builds_configs(tmp_path):
    df = pd.DataFrame(
        [
            {"Brand Name": "TechCrunch", "URL": "https://techcrunch.com/", "Category": "Startups"},
            {"Brand Name": "The Verge", "URL": "https://www.theverge.com/", "Category": "Gadgets"},
        ]
    )
    catalog_path = tmp_path / "catalog.xlsx"
    save_catalog_table(df, str(catalog_path))

    loaded = load_catalog_table(str(catalog_path))
    assert set(loaded["Category"]) == {"Startups", "Gadgets"}

    sources = sources_from_catalog(str(catalog_path))

    assert len(sources) == 2
    tech_source = next(s for s in sources if s.brand_name == "TechCrunch")
    verge_source = next(s for s in sources if s.brand_name == "The Verge")

    assert tech_source.type == "web"
    assert "startups" in tech_source.tags
    assert tech_source.category == "Startups"

    assert verge_source.type == "web"
    assert "gadgets" in verge_source.tags
    assert verge_source.category == "Gadgets"

def test_catalog_upsert_and_remove(tmp_path):
    df = pd.DataFrame(columns=["Brand Name", "URL", "Category"])
    df = upsert_catalog_entry(df, "TechCrunch", "techcrunch.com", "Startups")
    assert df.iloc[0]["URL"].startswith("https://")

    df = ensure_category_row(df, "Analytics")
    assert "Analytics" in set(df["Category"].astype(str))

    df = remove_catalog_entries(df, ["https://techcrunch.com/"])
    assert not df["URL"].str.contains("techcrunch", case=False).any()
def test_sources_from_catalog_uses_feed_override(tmp_path):
    df = pd.DataFrame(
        [
            {"Brand Name": "WSJ Technology", "URL": "https://www.wsj.com/news/technology", "Category": "Media"},
            {"Brand Name": "TechCrunch", "URL": "https://techcrunch.com/", "Category": "Startups"},
        ]
    )
    catalog_path = tmp_path / "catalog_override.xlsx"
    save_catalog_table(df, str(catalog_path))

    sources = sources_from_catalog(str(catalog_path))
    rss_source = next(source for source in sources if source.brand_name == "WSJ Technology")
    web_source = next(source for source in sources if "techcrunch" in str(source.url))

    assert rss_source.type == "rss"
    assert str(rss_source.url) == "https://feeds.a.dj.com/rss/RSSWSJD.xml"
    assert "rss" in rss_source.tags

    assert web_source.type == "web"
    assert "web" in web_source.tags


def test_sources_from_catalog_with_direct_feed(tmp_path):
    df = pd.DataFrame([
        {"Brand": "Example Feed", "Link": "https://example.com/feed.xml", "Category": "Example"}
    ])
    csv_path = tmp_path / "feeds.csv"
    df.to_csv(csv_path, index=False)

    sources = sources_from_catalog(str(csv_path))

    assert len(sources) == 1
    source = sources[0]
    assert source.type == "rss"
    assert str(source.url) == "https://example.com/feed.xml"
    assert source.brand_name == "Example Feed"
    assert source.category == "Example"
    assert "rss" in source.tags
