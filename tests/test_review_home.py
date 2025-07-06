import importlib


def test_home_renders_summary(tmp_path):
    review = importlib.import_module("dream")
    summary = tmp_path / "summary.md"
    summary.write_text("# Hello\nWorld")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/"):
        html = review.home()
    assert "Hello" in html
