from scraper_project.utils.text import extract_hashtags, extract_keywords


def test_extract_hashtags():
    text = "Learning about #AI and #ML with #Python"
    assert extract_hashtags(text) == ["ai", "ml", "python"]


def test_extract_keywords():
    text = "Python enables rapid prototyping for data pipelines and scraping"
    keywords = extract_keywords(text, top_k=10)
    assert "python" in keywords
    assert "scraping" in keywords
