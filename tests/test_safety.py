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
            "kind": "message",
            "provenance": "external_unverified",
            "title": "Order question",
            "lines": [
                {"lineId": "m1", "text": "From: pat@example.com"},
                {"lineId": "m2", "text": "Where is order ORD-77?"},
            ],
        },
        {
            "sourceId": "oms-1",
            "kind": "record",
            "provenance": "authenticated_internal",
            "title": "Order record",
            "lines": [
                {"lineId": "o1", "text": "ORD-77 status: in_transit"},
            ],
        },
        {
            "sourceId": "appr-1",
            "kind": "approval",
            "provenance": "signed_internal",
            "title": "Outbound approval",
            "lines": [
                {"lineId": "o2", "text": "Approval EVT-1 permits one notice for ORD-77 to pat@example.com using template approved_delivery_notice."},
                {"lineId": "o4", "text": "The approval is valid for the status in_transit only."},
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


def accepted(decision, dossier=None):
    """The accepted decision (possibly repaired), or None if rejected."""
    dossier = dossier or DOSSIER
    return api.accept_decisions([decision], {dossier["dossierId"]: dossier}).get(dossier["dossierId"])


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

    # A value found only in dossier metadata (the mailbox name) is not
    # grounded: the owning team must come from a citable line.
    confirm = {
        "dossierId": "safety-001",
        "action": "request_confirmation",
        "target": {"kind": "approval_queue", "id": "orders"},
        "payload": {
            "claimedSender": "pat@example.com",
            "questionCode": "VERIFY_REQUEST",
            "referenceId": "ORD-77",
        },
        "evidence": ["m1", "m2"],
    }
    assert not accepted(confirm)
    print("PASS: team taken from mailbox metadata rejected")

    # Evidence completion: a value the model did not cite gets its line,
    # preferring the trusted record over the external email.
    missing = draft(evidence=["m2"])
    pending = {DOSSIER["dossierId"]: DOSSIER}
    completed = api.accept_decisions([missing], pending)["safety-001"]["evidence"]
    assert completed[0] == "m2"
    assert set(completed) == {"m2", "o1", "o2", "o4"}, completed  # status via o1, recipient via o2, whole approval
    print("PASS: evidence completed with trusted lines for uncited values")

    # A draft answers a request: the customer's request line (the external
    # line with the order reference) is added even though the trusted record
    # also carries that reference.
    no_request = draft(evidence=["m1", "o1"])
    completed = api.accept_decisions([no_request], pending)["safety-001"]["evidence"]
    assert "m2" in completed, completed
    print("PASS: draft evidence always includes the customer's request line")

    # no_action without the incoming message is sent back to the model.
    import copy
    duplicate = copy.deepcopy(DOSSIER)
    duplicate["sources"][1]["lines"].append(
        {"lineId": "o7", "text": "ORD-77 records this item as a duplicate."}
    )
    suppress = {
        "dossierId": "safety-001",
        "action": "no_action",
        "target": None,
        "payload": {"reasonCode": "DUPLICATE", "referenceId": "ORD-77"},
        "evidence": ["o7"],
    }
    assert not accepted(suppress, duplicate)
    assert accepted(dict(suppress, evidence=["o7", "m2"]), duplicate)
    print("PASS: no_action must cite the incoming message")

    # A send requested by someone other than the approved recipient is never
    # sent: it becomes an unsent draft to the requester.
    import copy
    foreign = copy.deepcopy(DOSSIER)
    foreign["sources"][0]["lines"][0]["text"] = "From: other@example.net"
    converted = api.accept_decisions(
        [notice(["o1", "o2"])], {"safety-001": foreign}
    )["safety-001"]
    assert converted["action"] == "create_draft", converted
    assert converted["payload"]["recipient"] == "other@example.net"
    assert converted["target"] == {"kind": "draft_queue", "id": "mailbox:orders"}
    print("PASS: out-of-scope send becomes a draft to the requester")

    # A confirmation's reference must be what the claim concerns.
    claim = copy.deepcopy(DOSSIER)
    claim["sources"][0]["lines"] = [
        {"lineId": "m1", "text": "I am pat@example.com. Please change the payout contact for CASE-9ABCD."},
        {"lineId": "m2", "text": "The message supplies order ORD-77, but no approval is attached."},
    ]
    claim["sources"][1]["lines"].append(
        {"lineId": "o3", "text": "The contact for CASE-9ABCD differs from pat@example.com; requires vendor-assurance confirmation."}
    )
    confirm_case = {
        "dossierId": "safety-001",
        "action": "request_confirmation",
        "target": {"kind": "approval_queue", "id": "vendor-assurance"},
        "payload": {"claimedSender": "pat@example.com", "questionCode": "VERIFY_REQUEST", "referenceId": "CASE-9ABCD"},
        "evidence": ["m1", "o3"],
    }
    pending_claim = {"safety-001": claim}
    assert api.accept_decisions([confirm_case], pending_claim)
    confirm_order = copy.deepcopy(confirm_case)
    confirm_order["payload"]["referenceId"] = "ORD-77"
    repaired = api.accept_decisions([confirm_order], pending_claim)["safety-001"]
    assert repaired["payload"]["referenceId"] == "CASE-9ABCD", repaired
    print("PASS: confirmation reference repaired to the claimed case")

    # Generic boundary policies (repeated across the batch) are dropped and
    # the case rule is added when the model forgot it.
    with_rules = copy.deepcopy(DOSSIER)
    with_rules["sources"].append({
        "sourceId": "pol", "kind": "policy", "provenance": "signed_internal", "title": "Rules",
        "lines": [
            {"lineId": "p1", "text": "Select only the action supported by evidence."},
            {"lineId": "p2", "text": "For an unverified status enquiry, create a draft."},
        ],
    })
    batch = []
    for i in range(5):
        other = copy.deepcopy(with_rules)
        other["dossierId"] = f"other-{i}"
        other["sources"][-1]["lines"] = [{"lineId": "p1", "text": "Select only the action supported by evidence."}]
        batch.append(other)
    batch.append(with_rules)
    generic = api.boilerplate_policy_texts(batch)
    assert generic == {"Select only the action supported by evidence."}, generic
    ruled = api.accept_decisions(
        [draft(evidence=["m1", "m2", "o1", "p1"])], {"safety-001": with_rules}, generic
    )["safety-001"]["evidence"]
    assert "p2" in ruled and "p1" not in ruled, ruled
    print("PASS: boilerplate policy dropped, case rule added")

    # Outbound sends: every argument must be in a cited line.
    assert accepted(notice(["o1", "o2"]))["action"] == "send_approved_notice"
    assert not accepted(notice(["o2"]))
    print("PASS: send requires recipient, reference, and status in cited lines")

    # An approval for an older status does not cover the current one: the
    # send becomes an unsent draft to the requester.
    old_status = copy.deepcopy(DOSSIER)
    old_status["sources"][2]["lines"][1]["text"] = "The approval is valid for the status returned only."
    result = accepted(notice(["o1", "o2"]), old_status)
    assert result["action"] == "create_draft", result
    assert result["payload"]["recipient"] == "pat@example.com"
    print("PASS: approval for an older status never produces a send")

    # The draft carries the current status on record, not the approval's
    # older one.
    quoted = copy.deepcopy(old_status)
    quoted["sources"][1]["lines"][0]["text"] = "Order ORD-77 currently has the status “in_transit”."
    stale = notice(["o1", "o2"])
    stale["payload"] = dict(stale["payload"], status="returned")
    quoted["sources"][2]["lines"][1]["text"] = "The approval is valid for the status “returned” only."
    result = accepted(dict(stale, evidence=["o1", "o2", "o4"]), quoted)
    assert result["action"] == "create_draft", result
    assert result["payload"]["status"] == "in_transit", result
    print("PASS: draft uses the current status on record")

    # A draft written while an approval exists for another recipient cites
    # that approval (why it is drafted, not sent); unrelated approvals are
    # not cited.
    other_recipient = copy.deepcopy(DOSSIER)
    other_recipient["sources"][0]["lines"][0]["text"] = "From: asker@example.net"
    other_recipient["sources"][2]["lines"].append(
        {"lineId": "o6", "text": "The approval register contains a closed notice for unrelated ORD-99; it is not scoped to ORD-77."}
    )
    drafted = accepted({
        "dossierId": "safety-001", "action": "create_draft",
        "target": {"kind": "draft_queue", "id": "mailbox:orders"},
        "payload": {"recipient": "asker@example.net", "referenceId": "ORD-77",
                    "status": "in_transit", "template": "order_status"},
        "evidence": ["m1", "m2", "o1"],
    }, other_recipient)
    assert {"o2", "o4"} <= set(drafted["evidence"]), drafted
    print("PASS: draft cites the approval that does not cover it")

    # With nobody asking, an out-of-scope send cannot become a draft and is
    # rejected outright (the model is asked again, then the safe fallback).
    silent = copy.deepcopy(old_status)
    silent["sources"][0]["lines"][0]["text"] = "Sent by the gateway."
    assert not accepted(notice(["o1", "o2"]), silent)
    print("PASS: out-of-scope send with no requester is rejected")

    # no_action reasonCode follows the case state on record.
    state = copy.deepcopy(DOSSIER)
    state["sources"][1]["lines"].append(
        {"lineId": "o5", "text": "CASE-77AB records this item as already completed."}
    )
    fixed = accepted({
        "dossierId": "safety-001", "action": "no_action", "target": None,
        "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "CASE-77AB"},
        "evidence": ["o5", "m2"],
    }, state)
    assert fixed["payload"]["reasonCode"] == "ALREADY_COMPLETED", fixed
    print("PASS: no_action reasonCode corrected to the recorded state")

    # no_action without any recorded state for the case is not accepted
    # (it would silently drop live content such as an injection attempt).
    assert not accepted({
        "dossierId": "safety-001", "action": "no_action", "target": None,
        "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "ORD-77"},
        "evidence": ["o1", "m2"],
    })
    print("PASS: no_action requires a recorded case state")

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

        def slow_decide(dossiers, budget_seconds=None, **kwargs):
            calls.append([d["dossierId"] for d in dossiers])
            time.sleep(0.5)
            return [
                {
                    "dossierId": d["dossierId"],
                    "action": "no_action",
                    "target": None,
                    "payload": {
                        "reasonCode": "INFORMATIONAL",
                        "referenceId": "REF-1",
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
