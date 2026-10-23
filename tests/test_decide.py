"""Regression tests for the live 503: Gemini overload must not fail propose.

No real Gemini calls are made; the model client and decide_batch are faked.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from google.genai import errors

import api.index as api
import decide
from store import MailroomStore
from test_api import make_request


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    """Fails with a provider 503 for the first model, succeeds for the next."""

    def __init__(self, ok_text):
        self.ok_text = ok_text
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        if len(self.calls) == 1:
            raise errors.ServerError(
                503, {"error": {"code": 503, "status": "UNAVAILABLE"}}
            )
        return FakeResponse(self.ok_text)


class FakeClient:
    def __init__(self, ok_text):
        self.models = FakeModels(ok_text)


def decision(dossier_id):
    return {
        "dossierId": dossier_id,
        "action": "no_action",
        "target": None,
        "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "REF-1"},
        "evidence": ["line-1"],
    }


def two_dossier_request(public_key):
    request = make_request(public_key)
    second = dict(request["dossiers"][0], dossierId="demo-dossier-002")
    request["dossiers"].append(second)
    request["corpus"]["stableCount"] = 2
    return request


def main():
    decide.time.sleep = lambda seconds: None

    # 1. A provider 503 on the first model falls through to the next model.
    client = FakeClient('{"proposals":[{"dossierId":"d1"}]}')
    result = decide.decide_chunk(
        client, [{"dossierId": "d1"}], time.monotonic() + 30
    )
    assert result == [{"dossierId": "d1"}]
    assert client.models.calls == decide.MODELS[:2]
    print("PASS: provider 503 retries with the next model")

    # 2. A chunk that cannot be decided before the deadline returns nothing
    #    instead of raising.
    assert decide.decide_chunk(client, [{"dossierId": "d1"}], 0) == []
    print("PASS: exhausted chunk returns no decisions instead of raising")

    public_key = b"\x01" * 32

    with tempfile.TemporaryDirectory() as temp:
        api.store = MailroomStore(Path(temp) / "decide-test.db")
        http = TestClient(api.app)
        calls = []

        # 3. Model answers only one of two dossiers, even after the retry pass:
        #    propose still returns 200, the other gets the safe fallback,
        #    and only the real model decision is cached.
        def partial_decide(dossiers, budget_seconds=None):
            calls.append([d["dossierId"] for d in dossiers])
            return [
                decision(d["dossierId"])
                for d in dossiers
                if d["dossierId"] == "demo-dossier-001"
            ]

        api.decide_batch = partial_decide
        request = two_dossier_request(public_key)
        response = http.post("/a2a/", json=request)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/json"
        actions = {p["dossierId"]: p["action"] for p in response.json()["proposals"]}
        assert actions == {
            "demo-dossier-001": "no_action",
            "demo-dossier-002": "request_confirmation",
        }
        assert calls == [
            ["demo-dossier-001", "demo-dossier-002"],
            ["demo-dossier-002"],
        ]
        print("PASS: partial model failure still returns 200 with safe fallback")

        # The next evaluation reuses the cached decision and only asks the
        # model about the dossier that fell back.
        calls.clear()
        request["evaluationId"] = "offline-evaluation-002"
        response = http.post("/a2a/", json=request)
        assert response.status_code == 200, response.text
        assert calls[0] == ["demo-dossier-002"]
        print("PASS: fallback decisions are not cached")

        # 4. A total outage with nothing cached is still reported as 503.
        api.decide_batch = lambda dossiers, budget_seconds=None: []
        outage = make_request(public_key)
        outage["evaluationId"] = "offline-evaluation-003"
        outage["dossiers"][0]["dossierId"] = "demo-dossier-outage"
        response = http.post("/a2a/", json=outage)
        assert response.status_code == 503, response.text
        assert response.json() == {"error": "Decision model unavailable"}
        print("PASS: total model outage returns 503 JSON")

        # 5. A missing API key is a clear configuration error.
        def no_key(dossiers, budget_seconds=None):
            raise decide.ModelConfigError("GEMINI_API_KEY is missing from environment")

        api.decide_batch = no_key
        outage["evaluationId"] = "offline-evaluation-004"
        response = http.post("/a2a/", json=outage)
        assert response.status_code == 503, response.text
        assert "GEMINI_API_KEY" in response.json()["error"]
        print("PASS: missing API key reported clearly")

        http.close()

    print("All decision resilience tests passed")


if __name__ == "__main__":
    main()
