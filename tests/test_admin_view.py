import importlib


def test_admin_page(tmp_path):
    review = importlib.import_module("convey")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin"):
        html = review.admin_page()
    assert "Admin" in html or "Journal Configuration" in html
