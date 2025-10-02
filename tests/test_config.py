from scraper_project.config import load_settings


def test_load_settings_skip_catalog(tmp_path):
    sources_path = tmp_path / "sources.yml"
    sources_path.write_text(
        """sources:
  - type: rss
    url: https://example.com/feed""",
        encoding="utf-8",
    )

    settings_path = tmp_path / "settings.yml"
    settings_path.write_text("categories_path: null\n", encoding="utf-8")

    cfg = load_settings(sources_path, settings_path, include_catalog=False, include_custom=False)

    urls = [str(source.url) for source in cfg.sources]
    assert urls == ["https://example.com/feed"]


def test_load_settings_with_custom_sources(tmp_path):
    sources_path = tmp_path / "sources.yml"
    sources_path.write_text(
        """sources:
  - type: rss
    url: https://example.com/feed""",
        encoding="utf-8",
    )

    settings_path = tmp_path / "settings.yml"
    settings_path.write_text("categories_path: null\n", encoding="utf-8")

    custom_path = tmp_path / "custom.yml"
    custom_path.write_text(
        """sources:
  - type: rss
    url: https://custom.example.com/feed
    category: custom""",
        encoding="utf-8",
    )

    cfg = load_settings(
        sources_path,
        settings_path,
        include_catalog=False,
        include_custom=True,
        custom_sources_path=custom_path,
    )

    urls = [str(source.url) for source in cfg.sources]
    assert urls == ["https://example.com/feed", "https://custom.example.com/feed"]
    categories = [getattr(source, "category", None) for source in cfg.sources]
    assert categories == [None, "custom"]
