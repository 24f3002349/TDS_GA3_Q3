
import base64
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import api.index as api
from store import MailroomStore
from core import PROFILE, ALLOWED_ACTIONS, canonical_json


def make_request(public_key):
    public_key_b64 = base64.urlsafe_b64encode(public_key).decode("ascii").rstrip("=")

    dossier = {
        "dossierId": "demo-dossier-001",
        "partition": "stable_core",
        "receivedAt": "2026-10-23T12:00:00Z",
        "mailbox": "support",
        "objective": "Classify this informational message",
        "sources": [
            {
                "sourceId": "source-1",
                "kind": "email",
                "provenance": "untrusted_sender",
                "title": "Example message",
                "lines": [
                    {
                        "lineId": "line-1",
                        "text": "This is a synthetic informational message.",
                    }
                ],
            }
        ],
    }

    return {
        "profile": PROFILE,
        "operation": "propose",
        "evaluationId": "offline-evaluation-001",
        "receiptVerifier": {
            "algorithm": "Ed25519",
            "publicKeyJwk": {
                "kty": "OKP",
                "crv": "Ed25519",
                "x": public_key_b64,
            },
        },
        "corpus": {
            "coreId": "offline-core",
            "auditId": "offline-audit",
            "stableCount": 1,
            "freshCount": 0,
        },
        "allowedActions": sorted(ALLOWED_ACTIONS),
        "dossiers": [dossier],
    }


def make_signed_receipt(private_key, evaluation_id, input_digest, proposal, accepted):
    receipt = {
        "dossierId": proposal["dossierId"],
        "callId": proposal["callId"],
        "action": proposal["action"],
        "accepted": accepted,
        "proposalDigest": proposal["proposalDigest"],
        "receiptId": "offline-receipt-001",
    }

    signed_message = {
        "profile": PROFILE,
        "evaluationId": evaluation_id,
        "inputDigest": input_digest,
        "receipt": receipt,
    }

    signature = private_key.sign(
        canonical_json(signed_message).encode("utf-8")
    )

    receipt["receiptSignature"] = base64.b64encode(signature).decode("ascii")
    return receipt


def main():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    with tempfile.TemporaryDirectory() as temp:
        api.store = MailroomStore(Path(temp) / "api-test.db")
        client = TestClient(api.app)
        model_calls = {"count": 0}

        def fake_decide_batch(dossiers, budget_seconds=None):
            model_calls["count"] += 1
            return [
                {
                    "dossierId": dossier["dossierId"],
                    "action": "no_action",
                    "target": None,
                    "payload": {
                        "reasonCode": "INFORMATIONAL",
                        "referenceId": "REF-1",
                    },
                    "evidence": ["line-1"],
                }
                for dossier in dossiers
            ]

        api.decide_batch = fake_decide_batch

        request_data = make_request(public_key)

        first = client.post("/a2a/", json=request_data)
        assert first.status_code == 200, first.text
        first_json = first.json()
        assert first_json["status"] == "awaiting_receipts"
        assert len(first_json["proposals"]) == 1
        assert "proposalDigest" not in first_json["proposals"][0]
        assert model_calls["count"] == 1
        print("PASS: valid propose request")

        replay = client.post("/a2a/", json=request_data)
        assert replay.status_code == 200, replay.text
        assert replay.json() == first_json
        assert model_calls["count"] == 1
        print("PASS: exact propose replay does not call the model again")

        changed = make_request(public_key)
        changed["dossiers"][0]["objective"] = "Changed synthetic objective"
        conflict = client.post("/a2a/", json=changed)
        assert conflict.status_code == 409, conflict.text
        print("PASS: changed propose request under same evaluation ID rejected")

        invalid = make_request(public_key)
        invalid["profile"] = "wrong-profile"
        bad_response = client.post("/a2a/", json=invalid)
        assert bad_response.status_code == 400, bad_response.text
        assert model_calls["count"] == 1
        print("PASS: invalid profile rejected before model call")

        # Retrieve the internal proposal, which contains its proposalDigest.
        proposal_rows = api.store.get_proposals(request_data["evaluationId"])
        assert len(proposal_rows) == 1
        proposal = json.loads(proposal_rows[0]["proposal_json"])

        def make_commit(accepted):
            receipt = make_signed_receipt(
                private_key=private_key,
                evaluation_id=request_data["evaluationId"],
                input_digest=first_json["inputDigest"],
                proposal=proposal,
                accepted=accepted,
            )
            return {
                "profile": PROFILE,
                "operation": "commit",
                "evaluationId": request_data["evaluationId"],
                "inputDigest": first_json["inputDigest"],
                "receipts": [receipt],
            }

        # Invalid signature must not complete or partially persist the commit.
        invalid_commit = make_commit(accepted=False)
        bad_signature = invalid_commit["receipts"][0]["receiptSignature"]
        invalid_commit["receipts"][0]["receiptSignature"] = (
            ("A" if bad_signature[0] != "A" else "B") + bad_signature[1:]
        )

        rejected_signature = client.post("/a2a/", json=invalid_commit)
        assert rejected_signature.status_code == 400, rejected_signature.text
        evaluation = api.store.get_evaluation(request_data["evaluationId"])
        assert evaluation["status"] == "awaiting_receipts"

        with api.store.connect() as db:
            saved_receipts = db.execute(
                "SELECT COUNT(*) FROM receipts"
            ).fetchone()[0]
        assert saved_receipts == 0
        print("PASS: invalid signature rejected without saving receipts or completing evaluation")

        # A correctly signed receipt with accepted=False must yield rejected.
        valid_commit = make_commit(accepted=False)
        committed = client.post("/a2a/", json=valid_commit)
        assert committed.status_code == 200, committed.text
        committed_json = committed.json()
        assert committed_json["status"] == "completed"
        assert len(committed_json["outcomes"]) == 1
        assert committed_json["outcomes"][0]["status"] == "rejected"
        print("PASS: valid signed receipt commits with rejected outcome when accepted=False")

        # Exact commit replay must return the same result.
        commit_replay = client.post("/a2a/", json=valid_commit)
        assert commit_replay.status_code == 200, commit_replay.text
        assert commit_replay.json() == committed_json
        print("PASS: exact commit replay returns the saved response")

        # A different request under the already-committed evaluation ID must conflict.
        changed_commit = make_commit(accepted=True)
        changed_replay = client.post("/a2a/", json=changed_commit)
        assert changed_replay.status_code == 409, changed_replay.text
        print("PASS: changed commit under the same evaluation ID rejected")

        client.close()

    print("All offline API tests passed")
    print("No Gemini API calls made")


if __name__ == "__main__":
    main()
