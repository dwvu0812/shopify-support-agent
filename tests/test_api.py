"""Tests for the shop support agent API.

Run:  pytest -q
No API key and no Chroma database are needed: the LLM provider and
the retriever are replaced with fakes, and order lookup is
deterministic (it reads data/orders.json, never the LLM).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from main import RetrievedChunk, app, extract_order_id, get_order, order_answer

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _point_at_seed_data(monkeypatch):
    """Every test reads the shipped seed data, wherever pytest runs from."""
    monkeypatch.setattr(main.settings, "data_dir", str(ROOT / "data"))
    main._shop_data.clear()
    yield
    main._shop_data.clear()


class FakeProvider:
    name = "fake:test"

    def generate(self, system_prompt: str, question: str) -> str:
        return f"Fake answer for: {question}"


class FakeRetriever:
    def __init__(self, chunks=None):
        self._chunks = chunks or []

    def available(self) -> bool:
        return bool(self._chunks)

    def retrieve(self, question: str):
        return self._chunks


def make_client(monkeypatch, chunks=None, provider=None):
    if provider is not None:
        monkeypatch.setattr(main, "get_llm_provider", lambda: provider)
    app.dependency_overrides[main.get_retriever] = lambda: FakeRetriever(chunks)
    return TestClient(app)


def test_health_ok(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "X-Request-ID" in resp.headers


def test_ready_reports_retrieval(monkeypatch):
    client = make_client(
        monkeypatch, chunks=[RetrievedChunk("docs/faq.md", 0, "hello")]
    )
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["retrieval_available"] is True


def test_chat_blank_question_rejected_without_key(monkeypatch):
    # No provider override at all: order lookup does not need a key,
    # and blank questions are rejected before any provider is resolved.
    client = make_client(monkeypatch)
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "   "}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_chat_order_lookup_needs_no_llm(monkeypatch):
    """Order questions answer from the order record, never the LLM."""
    def _boom():
        raise AssertionError("LLM provider must not be called for order lookup")

    monkeypatch.setattr(main, "get_llm_provider", _boom)
    client = make_client(monkeypatch)
    resp = client.post("/chat", json={"question": "Where is my order #1002?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "order-lookup"
    assert body["order"]["id"] == "1002"
    assert body["order"]["status"] == "shipped"
    assert "shipped" in body["answer"]
    assert "784512369014" in body["answer"]  # real tracking number, not invented
    assert body["citations"][0]["source"] == "orders/1002"


def test_chat_order_lookup_variants(monkeypatch):
    def _boom():
        raise AssertionError("LLM provider must not be called for order lookup")

    monkeypatch.setattr(main, "get_llm_provider", _boom)
    client = make_client(monkeypatch)
    for question, order_id in [
        ("order 1007 status please", "1007"),
        ("#1001", "1001"),
        ("Hi, where is my order #1004?", "1004"),
    ]:
        body = client.post("/chat", json={"question": question}).json()
        assert body["order"]["id"] == order_id


def test_chat_unknown_order_is_a_clear_answer_not_an_error(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.post("/chat", json={"question": "Where is my order #9999?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["order"] is None
    assert "couldn't find order #9999" in body["answer"]


def test_chat_rag_returns_citations_and_product_card(monkeypatch):
    chunks = [
        RetrievedChunk(
            "docs/products.md", 0,
            "Trailhead Hiking Backpack 40L — $89.99. In stock: 24.",
        )
    ]
    client = make_client(monkeypatch, chunks=chunks, provider=FakeProvider())
    resp = client.post(
        "/chat", json={"question": "How much is the Trailhead Hiking Backpack?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved"] is True
    assert body["citations"][0]["source"] == "docs/products.md"
    assert len(body["products"]) == 1
    assert body["products"][0]["id"] == "P001"
    assert body["products"][0]["price"] == 89.99


def test_chat_rag_without_key_returns_500(monkeypatch):
    monkeypatch.setattr(main.settings, "openai_api_key", None)
    monkeypatch.setattr(main.settings, "gemini_api_key", None)
    client = make_client(monkeypatch, chunks=[])
    resp = client.post("/chat", json={"question": "How long is shipping?"})
    assert resp.status_code == 500


def test_get_order_endpoint(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.get("/orders/1002")
    assert resp.status_code == 200
    assert resp.json()["status"] == "shipped"
    assert client.get("/orders/9999").status_code == 404


def test_list_products(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.get("/products")
    assert resp.status_code == 200
    products = resp.json()
    assert len(products) == 12
    base_layer = next(p for p in products if p["id"] == "P004")
    assert base_layer["stock"] == 0  # out of stock on purpose


def test_extract_order_id():
    assert extract_order_id("Where is my order #1002?") == "1002"
    assert extract_order_id("order 1007 status") == "1007"
    assert extract_order_id("How long does shipping take?") is None
    assert extract_order_id("Do you have the jacket in size M?") is None


def test_order_answer_wording():
    assert "was delivered on 2026-11-20" in order_answer(get_order("1001"))
    assert "out for delivery today" in order_answer(get_order("1007"))
    assert "was cancelled" in order_answer(get_order("1006"))
    assert "Estimated delivery: 2026-12-02" in order_answer(get_order("1003"))


def test_admin_upload_doc_reindexes(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "docs_dir", str(tmp_path))
    monkeypatch.setattr(main.settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(main.settings, "admin_token", None)
    monkeypatch.setattr(main, "embed_text", lambda text: [1.0, 0.0, 0.0])
    main._retriever = None
    client = make_client(monkeypatch)
    resp = client.post(
        "/admin/docs",
        json={"filename": "policy.md", "content": "Free returns within 30 days."},
    )
    assert resp.status_code == 200
    assert resp.json()["chunks"] >= 1
    assert (tmp_path / "policy.md").read_text() == "Free returns within 30 days."
    # Uploading again replaces the doc instead of duplicating chunks.
    resp2 = client.post(
        "/admin/docs",
        json={"filename": "policy.md", "content": "Free returns within 60 days."},
    )
    assert resp2.status_code == 200
    main._retriever = None


def test_admin_upload_doc_rejects_bad_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "docs_dir", str(tmp_path))
    client = make_client(monkeypatch)
    assert client.post(
        "/admin/docs", json={"filename": "../secret.md", "content": "x"}
    ).status_code == 422
    assert client.post(
        "/admin/docs", json={"filename": "notes.txt", "content": "x"}
    ).status_code == 422


def test_admin_upload_doc_requires_token_when_set(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "docs_dir", str(tmp_path))
    monkeypatch.setattr(main.settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(main.settings, "admin_token", "secret-token")
    monkeypatch.setattr(main, "embed_text", lambda text: [1.0, 0.0, 0.0])
    main._retriever = None
    client = make_client(monkeypatch)
    assert client.post(
        "/admin/docs", json={"filename": "a.md", "content": "hi"}
    ).status_code == 401
    assert client.post(
        "/admin/docs",
        json={"filename": "a.md", "content": "hi"},
        headers={"X-Admin-Token": "secret-token"},
    ).status_code == 200
    main._retriever = None
