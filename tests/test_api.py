import pytest
from fastapi.testclient import TestClient

from app import main
from app.rag import RagService


@pytest.fixture
def client(monkeypatch, fake_vector_store, fake_llm):
    fake_service = RagService(vector_store=fake_vector_store, llm=fake_llm)

    def fake_builder(settings):
        return fake_service

    monkeypatch.setattr(main, "build_rag_service", fake_builder)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with TestClient(main.app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(client, fake_llm):
    response = client.post("/query", json={"question": "What is an eigenvalue?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_llm.canned_answer
    assert isinstance(body["sources"], list)
    assert len(body["sources"]) >= 1
    for s in body["sources"]:
        assert "source" in s
        assert "snippet" in s


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_respects_k_param(client, fake_vector_store):
    response = client.post("/query", json={"question": "determinant", "k": 1})
    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 1
