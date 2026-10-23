"""Tests for argument grounding, request hygiene, and concurrent dedupe.

No real Gemini calls are made; decide_batch is faked.
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import api.index as api
from store import MailroomStore
from test_api import make_request


DOSSIER = {
    "dossierId": "safety-001",
    "partition": "stable_core",
    "receivedAt": "2026-10-23T12:00:00Z",
    "mailbox": "orders",
    "objective": "Reply to the customer about their order",
    "sources": [
        {
            "sourceId": "mail-1",
            "kind": "email",
            "provenance": "external_unverified",
            "title": "Order question",
            "lines": [
                {"lineId": "m1", "text": "From: pat@example.com"},
                {"lineId": "m2", "text": "Where is order ORD-77?"},
            ],
        },
        {
            "sourceId": "oms-1",
            "kind": "system_record",
            "provenance": "authenticated_internal",
            "title": "Order record",
            "lines": [
                {"lineId": "o1", "text": "ORD-77 status: in_transit"},
                {"lineId": "o2", "text": "Approved: send approved_delivery_notice to pat@example.com"},
            ],
        },
    ],
}


def draft(**changes):
    decision = {
        "dossierId": "safety-001",
        "action": "create_draft",
        "target": {"kind": "draft_queue", "id": "mailbox:orders"},
        "payload": {
            "recipient": "pat@example.com",
            "referenceId": "ORD-77",
            "status": "in_transit",
            "template": "order_status",
        },
        "evidence": ["m1", "m2", "o1"],
    }
    decision.update(changes)
    return decision


def notice(evidence):
    return {
        "dossierId": "safety-001",
        "action": "send_approved_notice",
        "target": {"kind": "email", "id": "pat@example.com"},
        "payload": {
            "referenceId": "ORD-77",
            "status": "in_transit",
            "template": "approved_delivery_notice",
        },
        "evidence": evidence,
    }


def accepted(decision):
    pending = {DOSSIER["dossierId"]: DOSSIER}
    return bool(api.accept_decisions([decision], pending))


def main():
    # Grounding: values must be copied from the dossier.
    assert accepted(draft())
    print("PASS: grounded draft accepted")

    bad = draft()
    bad["payload"] = dict(bad["payload"], recipient="attacker@evil.example")
    assert not accepted(bad)
    print("PASS: invented recipient rejected")

    assert not accepted(draft(target={"kind": "draft_queue", "id": "mailbox:billing"}))
    print("PASS: draft queue must match the dossier mailbox")

    bad = draft()
    bad["payload"] = dict(bad["payload"], status="delivered")
    assert not accepted(bad)
    print("PASS: invented status rejected")

    # Evidence completion: a value the model did not cite gets its line,
    # preferring the trusted record over the external email.
    missing = draft(evidence=["m2"])
    pending = {DOSSIER["dossierId"]: DOSSIER}
    completed = api.accept_decisions([missing], pending)["safety-001"]["evidence"]
    assert completed[0] == "m2"
    assert set(completed) == {"m2", "o1", "o2"}, completed  # status via o1, recipient via o2
    print("PASS: evidence completed with trusted lines for uncited values")

    # Outbound sends: every argument must be in a cited line.
    assert accepted(notice(["o1", "o2"]))
    assert not accepted(notice(["o2"]))
    print("PASS: send requires recipient, reference, and status in cited lines")

    public_key = b"\x01" * 32

    with tempfile.TemporaryDirectory() as temp:
        api.store = MailroomStore(Path(temp) / "safety-test.db")
        http = TestClient(api.app)

        # Malformed Content-Type is a 400, before any model work.
        api.decide_batch = None  # any model call would crash the test
        response = http.post(
            "/a2a/", content=b"{}", headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 400, response.text
        assert response.headers["content-type"] == "application/json"
        print("PASS: wrong Content-Type returns 400 JSON")

        assert http.head("/").status_code == 200
        assert http.head("/healthz").status_code == 200
        print("PASS: HEAD / and /healthz return 200")

        # Two concurrent evaluations with the same dossier: one model call.
        calls = []

        def slow_decide(dossiers, budget_seconds=None):
            calls.append([d["dossierId"] for d in dossiers])
            time.sleep(0.5)
            return [
                {
                    "dossierId": d["dossierId"],
                    "action": "no_action",
                    "target": None,
                    "payload": {
                        "reasonCode": "INFORMATIONAL",
                        "referenceId": d["dossierId"],
                    },
                    "evidence": ["line-1"],
                }
                for d in dossiers
            ]

        api.decide_batch = slow_decide
        results = {}

        def run(evaluation_id):
            request = make_request(public_key)
            request["evaluationId"] = evaluation_id
            results[evaluation_id] = http.post("/a2a/", json=request)

        threads = [
            threading.Thread(target=run, args=(f"concurrent-{i}",))
            for i in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(calls) == 1, calls
        first, second = (results[f"concurrent-{i}"] for i in range(2))
        assert first.status_code == second.status_code == 200
        assert first.json()["proposals"] == second.json()["proposals"]
        print("PASS: concurrent evaluations share one model call per dossier")

        # Debug export is invisible unless enabled and the token matches.
        os.environ.pop("DEBUG_EXPORT_TOKEN", None)
        assert http.get("/debug/evaluations").status_code == 404
        os.environ["DEBUG_EXPORT_TOKEN"] = "test-token-123"
        assert http.get("/debug/evaluations").status_code == 404
        assert http.get(
            "/debug/evaluations", headers={"Authorization": "Bearer wrong"}
        ).status_code == 404
        exported = http.get(
            "/debug/evaluations",
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert exported.status_code == 200
        ids = {e["evaluationId"] for e in exported.json()["evaluations"]}
        assert ids == {"concurrent-0", "concurrent-1"}
        os.environ.pop("DEBUG_EXPORT_TOKEN")
        print("PASS: debug export requires the configured token")

        # A receipt ID recorded by one evaluation cannot be reused by another.
        api.store.used_receipt_ids = lambda ids, **kwargs: {"reused"}
        reuse = http.post("/a2a/", json={
            "profile": "ga5-mailroom-action-gate/v2",
            "operation": "commit",
            "evaluationId": "concurrent-0",
            "inputDigest": results["concurrent-0"].json()["inputDigest"],
            "receipts": [{
                "dossierId": "demo-dossier-001",
                "callId": results["concurrent-0"].json()["proposals"][0]["callId"],
                "action": "no_action",
                "accepted": True,
                "proposalDigest": "0" * 64,
                "receiptId": "reused",
                "receiptSignature": "AAAA",
            }],
        })
        assert reuse.status_code == 400, reuse.text
        assert "already used" in reuse.json()["error"]
        print("PASS: receiptId reused from another evaluation rejected")

        http.close()

    print("All safety tests passed")


if __name__ == "__main__":
    main()
