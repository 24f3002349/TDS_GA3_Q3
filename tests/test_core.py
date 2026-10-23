import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (
    canonical_json,
    content_fingerprint,
    input_digest,
    make_call_id,
    proposal_digest,
    validate_dossiers,
    validate_proposal,
)

dossier = {
    "dossierId": "demo-001",
    "partition": "stable_core",
    "sources": [{
        "sourceId": "source-1",
        "lines": [
            {"lineId": "line-1", "text": "Example informational message."}
        ],
    }],
}

validate_dossiers([dossier])

proposal = {
    "dossierId": "demo-001",
    "callId": make_call_id(dossier),
    "action": "no_action",
    "target": None,
    "payload": {
        "reasonCode": "INFORMATIONAL",
        "referenceId": "example-001",
    },
    "evidence": ["line-1"],
}

validate_proposal(proposal, dossier)

assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
assert input_digest([dossier]) == input_digest([dossier])
assert content_fingerprint(dossier) == content_fingerprint(dossier)
assert proposal_digest(proposal) == proposal_digest(proposal)
assert len(input_digest([dossier])) == 64
assert len(proposal_digest(proposal)) == 64
assert make_call_id(dossier) == make_call_id(dossier)

print("PASS: canonical JSON")
print("PASS: SHA-256 input and proposal digests")
print("PASS: stable call ID")
print("PASS: dossier and evidence validation")
print("All core tests passed.")

