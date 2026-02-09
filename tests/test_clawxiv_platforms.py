import importlib.util
from pathlib import Path

from swarm.research.platforms import ClawxivClient, Paper


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_paper_from_dict_reads_files_and_authors():
    data = {
        "paper_id": "clawxiv.2602.00038",
        "title": "Diversity as Defense",
        "abstract": "Abstract",
        "categories": ["cs.MA"],
        "files": {"source": "SRC", "bib": "BIB", "images": {"fig.png": "abc"}},
        "authors": [{"name": "SWARMSafety", "isBot": True}],
        "created_at": "2026-02-08T17:01:29.724371Z",
        "updated_at": None,
    }
    paper = Paper.from_dict(data)
    assert paper.source == "SRC"
    assert paper.bib == "BIB"
    assert paper.images["fig.png"] == "abc"
    assert paper.authors == ["SWARMSafety"]
    assert paper.created_at.tzinfo is not None


def test_clawxiv_search_builds_get_params(monkeypatch):
    client = ClawxivClient(api_key="clx_test")
    captured = {}

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        payload = {
            "papers": [
                {
                    "paper_id": "clawxiv.2602.00038",
                    "title": "Diversity as Defense",
                    "abstract": "Abstract",
                    "categories": ["cs.MA"],
                    "authors": [{"name": "SWARMSafety", "isBot": True}],
                    "created_at": "2026-02-08T17:01:29.724371Z",
                    "updated_at": "2026-02-08T17:01:29.724371Z",
                }
            ],
            "total": 1,
        }
        return DummyResponse(payload)

    monkeypatch.setattr(client, "_request", fake_request)

    result = client.search(
        "llm",
        category="cs.AI",
        page=2,
        limit=25,
        sort_by="date",
        sort_order="desc",
    )

    assert captured["method"] == "GET"
    assert captured["url"].endswith("/search")
    assert captured["params"]["query"] == "llm"
    assert captured["params"]["category"] == "cs.AI"
    assert captured["params"]["page"] == 2
    assert captured["params"]["limit"] == 25
    assert captured["params"]["sort_by"] == "date"
    assert captured["params"]["sort_order"] == "desc"
    assert result.total_count == 1
    assert result.papers[0].authors == ["SWARMSafety"]


def test_clawxiv_submit_includes_bib_and_images(monkeypatch):
    client = ClawxivClient(api_key="clx_test")
    captured = {}

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return DummyResponse({"paper_id": "clawxiv.2602.00038"})

    monkeypatch.setattr(client, "_request", fake_request)

    paper = Paper(
        title="Title",
        abstract="Abstract",
        categories=["cs.MA"],
        source="SRC",
        bib="BIB",
        images={"figure.png": "abc"},
    )
    client.submit(paper)

    files = captured["json"]["files"]
    assert captured["method"] == "POST"
    assert files["source"] == "SRC"
    assert files["bib"] == "BIB"
    assert files["images"]["figure.png"] == "abc"


def test_clawxiv_update_includes_bib_and_images(monkeypatch):
    client = ClawxivClient(api_key="clx_test")
    captured = {}

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return DummyResponse({"paper_id": "clawxiv.2602.00038", "version": 2})

    monkeypatch.setattr(client, "_request", fake_request)

    paper = Paper(
        title="Title",
        abstract="Abstract",
        categories=["cs.MA"],
        source="SRC",
        bib="BIB",
        images={"figure.png": "abc"},
    )
    client.update("clawxiv.2602.00038", paper)

    files = captured["json"]["files"]
    assert captured["method"] == "PUT"
    assert files["source"] == "SRC"
    assert files["bib"] == "BIB"
    assert files["images"]["figure.png"] == "abc"


def _load_export_history_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "clawxiv"
        / "export_history.py"
    )
    spec = importlib.util.spec_from_file_location("export_history", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_history_refuses_api_key_for_non_clawxiv_url():
    mod = _load_export_history_module()
    status = mod._post_json("https://example.com/api", {"ok": True}, "clx_test", 1)
    assert status == 2


def test_export_history_sends_x_api_key_for_safe_url(monkeypatch):
    mod = _load_export_history_module()
    captured = {}

    class DummyHTTPResponse:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyOpener:
        def open(self, req, timeout=0):
            captured["headers"] = req.headers
            return DummyHTTPResponse()

    monkeypatch.setattr(
        mod.urllib.request, "build_opener", lambda *args, **kwargs: DummyOpener()
    )

    status = mod._post_json(
        "https://www.clawxiv.org/api/v1/metrics",
        {"ok": True},
        "clx_test",
        1,
    )
    assert status == 200
    assert any(k.lower() == "x-api-key" for k in captured["headers"])
