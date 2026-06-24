from fastapi.testclient import TestClient

from scripts import api


class FakeResult:
    def __init__(self, **payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


def test_health_reports_validated_brain():
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["brain"] == "scripts.answer.answer"


def test_chat_uses_same_rag_answer_function(monkeypatch):
    calls = []

    def fake_answer(**kwargs):
        calls.append(kwargs)
        return FakeResult(
            answer="PPAP Level 3 requires a full submission package. [1]",
            confidence="medium",
            action_required=True,
            insufficient_evidence=False,
            sources=["ppap_level_requirements.md"],
            retrieved_sources=["ppap_level_requirements.md"],
            retrieved_context=[],
            risk_level="not_applicable",
        )

    monkeypatch.setattr(api, "rag_answer", fake_answer)
    monkeypatch.setattr(api, "build_where_filter", api_filter_builder)
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={
            "question": "What does PPAP Level 3 require?",
            "family": "Electronics",
            "risk": "ppap",
            "session_id": "test",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["sources"] == ["ppap_level_requirements.md"]
    assert calls[0]["question"] == "What does PPAP Level 3 require?"
    assert calls[0]["session_id"] == "test"
    assert calls[0]["where_filter"] == {
        "$and": [
            {"risk_domain": {"$eq": "ppap"}},
            {
                "$or": [
                    {"commodity": {"$eq": "Electronics"}},
                    {"commodity": {"$eq": "GENERAL"}},
                ]
            },
        ]
    }


def test_chat_stream_emits_status_result_and_done(monkeypatch):
    def fake_answer(**_kwargs):
        return FakeResult(
            answer="Use the supplier quality guidance. [1]",
            confidence="medium",
            action_required=False,
            insufficient_evidence=False,
            sources=["supplier_quality_manual.md"],
            retrieved_sources=["supplier_quality_manual.md"],
            retrieved_context=[],
            risk_level="not_applicable",
        )

    monkeypatch.setattr(api, "rag_answer", fake_answer)
    monkeypatch.setattr(api, "build_where_filter", api_filter_builder)
    client = TestClient(api.app)

    response = client.post(
        "/chat/stream",
        json={"question": "What should the supplier do?"},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: status" in body
    assert "running_sicc_brain" in body
    assert "event: result" in body
    assert "supplier_quality_manual.md" in body
    assert "event: done" in body


def test_chat_is_rate_limited(monkeypatch):
    from scripts import rate_limit

    # Tight limit + clean bucket so the test is deterministic and isolated.
    monkeypatch.setattr(rate_limit, "_RATE_LIMIT", 2)
    rate_limit._hits.clear()

    monkeypatch.setattr(api, "rag_answer", lambda **_kwargs: FakeResult(
        answer="ok [1]", confidence="low", action_required=False,
        insufficient_evidence=False, sources=["s.md"],
        retrieved_sources=[], retrieved_context=[], risk_level="not_applicable",
    ))
    monkeypatch.setattr(api, "build_where_filter", api_filter_builder)
    client = TestClient(api.app)

    body = {"question": "hello"}
    assert client.post("/chat", json=body).status_code == 200
    assert client.post("/chat", json=body).status_code == 200
    blocked = client.post("/chat", json=body)
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers

    rate_limit._hits.clear()


def api_filter_builder(risk=None, family=None):
    clauses = []
    if risk:
        clauses.append({"risk_domain": {"$eq": risk}})
    if family:
        clauses.append(
            {
                "$or": [
                    {"commodity": {"$eq": family}},
                    {"commodity": {"$eq": "GENERAL"}},
                ]
            }
        )
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
