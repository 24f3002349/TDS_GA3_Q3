import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from store import MailroomStore

with tempfile.TemporaryDirectory() as temp:
    store = MailroomStore(Path(temp) / "test.db")

    proposal = {
        "dossierId": "demo-001",
        "callId": "mailroom:123456789",
        "action": "no_action",
        "target": None,
        "payload": {
            "reasonCode": "INFORMATIONAL",
            "referenceId": "example-001",
        },
        "evidence": ["line-1"],
        "proposalDigest": "a" * 64,
    }

    created = store.save_evaluation(
        evaluation_id="evaluation-001",
        input_digest="b" * 64,
        request={
            "profile": "ga5-mailroom-action-gate/v2",
            "operation": "propose",
            "evaluationId": "evaluation-001",
        },
        verifier={"algorithm": "Ed25519", "publicKeyJwk": {}},
        propose_response={"status": "awaiting_receipts"},
        proposals=[proposal],
    )

    assert created is True
    assert store.get_evaluation("evaluation-001") is not None
    assert len(store.get_proposals("evaluation-001")) == 1

    # Exact evaluation replay must not create a second evaluation.
    replay = store.save_evaluation(
        evaluation_id="evaluation-001",
        input_digest="b" * 64,
        request={
            "profile": "ga5-mailroom-action-gate/v2",
            "operation": "propose",
            "evaluationId": "evaluation-001",
        },
        verifier={"algorithm": "Ed25519", "publicKeyJwk": {}},
        propose_response={"status": "awaiting_receipts"},
        proposals=[proposal],
    )
    assert replay is False

    print("PASS: evaluation persisted")
    print("PASS: proposals persisted")
    print("PASS: exact replay does not duplicate records")

print("All storage tests passed.")
