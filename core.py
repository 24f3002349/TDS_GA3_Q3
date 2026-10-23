import hashlib
import json
import re
from typing import Any

PROFILE = "ga5-mailroom-action-gate/v2"

ALLOWED_ACTIONS = {
    "create_draft",
    "update_internal_record",
    "send_approved_notice",
    "request_confirmation",
    "quarantine_item",
    "no_action",
}

CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{12,128}$")


def canonical_json(value: Any) -> str:
    """Canonical compact JSON with recursively sorted object keys."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_fingerprint(dossier: dict) -> str:
    """Fingerprint a dossier independently of its evaluation ID."""
    return sha256_hex(canonical_json(dossier).encode("utf-8"))


def make_call_id(dossier: dict) -> str:
    """Stable ID for the same dossier ID and content."""
    material = {
        "dossierId": dossier["dossierId"],
        "fingerprint": content_fingerprint(dossier),
    }
    return "mailroom:" + sha256_hex(
        canonical_json(material).encode("utf-8")
    )


def input_digest(dossiers: list[dict]) -> str:
    """Hash the canonical JSON encoding of the entire dossier array."""
    return sha256_hex(canonical_json(dossiers).encode("utf-8"))


def proposal_digest(proposal: dict) -> str:
    """Hash exactly the six fields specified by the contract."""
    fields = (
        "dossierId",
        "callId",
        "action",
        "target",
        "payload",
        "evidence",
    )

    if any(field not in proposal for field in fields):
        raise ValueError("Proposal is missing a required field")

    normalized = {field: proposal[field] for field in fields}
    normalized["evidence"] = sorted(normalized["evidence"])

    return sha256_hex(canonical_json(normalized).encode("utf-8"))


def validate_dossiers(dossiers: Any) -> None:
    """Reject malformed dossier batches before any model calls."""
    if not isinstance(dossiers, list):
        raise ValueError("dossiers must be a list")

    seen_dossiers = set()

    for dossier in dossiers:
        if not isinstance(dossier, dict):
            raise ValueError("Each dossier must be an object")

        dossier_id = dossier.get("dossierId")
        if not isinstance(dossier_id, str) or not dossier_id:
            raise ValueError("Invalid dossierId")

        if dossier_id in seen_dossiers:
            raise ValueError("Duplicate dossierId")
        seen_dossiers.add(dossier_id)

        if not isinstance(dossier.get("sources"), list):
            raise ValueError("sources must be a list")

        seen_lines = set()

        for source in dossier["sources"]:
            if not isinstance(source, dict):
                raise ValueError("Each source must be an object")

            lines = source.get("lines")
            if not isinstance(lines, list):
                raise ValueError("source lines must be a list")

            for line in lines:
                if not isinstance(line, dict):
                    raise ValueError("Each line must be an object")

                line_id = line.get("lineId")
                if not isinstance(line_id, str) or not line_id:
                    raise ValueError("Invalid lineId")

                if line_id in seen_lines:
                    raise ValueError("Duplicate lineId within dossier")

                if not isinstance(line.get("text"), str):
                    raise ValueError("Line text must be a string")

                seen_lines.add(line_id)


def validate_proposal(proposal: Any, dossier: dict) -> None:
    """Validate contract fields, evidence ownership, and action schema."""
    from actions import validate_action_schema

    if not isinstance(proposal, dict):
        raise ValueError("Proposal must be an object")

    required = {
        "dossierId", "callId", "action",
        "target", "payload", "evidence",
    }

    if set(proposal) != required:
        raise ValueError("Proposal fields do not match the contract")

    if proposal["dossierId"] != dossier["dossierId"]:
        raise ValueError("Proposal dossierId mismatch")

    if not isinstance(proposal["callId"], str):
        raise ValueError("Invalid callId")

    if not CALL_ID_PATTERN.fullmatch(proposal["callId"]):
        raise ValueError("Invalid callId format")

    if proposal["action"] not in ALLOWED_ACTIONS:
        raise ValueError("Action is not allowed")

    evidence = proposal["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Evidence must be a non-empty list")

    if any(not isinstance(item, str) or not item for item in evidence):
        raise ValueError("Evidence IDs must be non-empty strings")

    if len(set(evidence)) != len(evidence):
        raise ValueError("Duplicate evidence IDs")

    available = {
        line["lineId"]
        for source in dossier["sources"]
        for line in source["lines"]
    }

    if not set(evidence).issubset(available):
        raise ValueError("Evidence references an unknown lineId")

    validate_action_schema(proposal)
