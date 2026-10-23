import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actions import validate_action_schema

valid = {
    "dossierId": "demo-001",
    "callId": "mailroom:123456789",
    "action": "no_action",
    "target": None,
    "payload": {
        "reasonCode": "INFORMATIONAL",
        "referenceId": "example-001",
    },
    "evidence": ["line-1"],
}

validate_action_schema(valid)
print("PASS: valid no_action schema")

invalid = {
    **valid,
    "payload": {
        **valid["payload"],
        "unexpected": "must not be allowed",
    },
}

try:
    validate_action_schema(invalid)
except ValueError:
    print("PASS: unexpected payload fields rejected")
else:
    raise AssertionError("Invalid payload was accepted")

print("All action schema tests passed.")

