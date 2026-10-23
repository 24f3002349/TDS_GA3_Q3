from typing import Any

ACTIONS = {
    "create_draft": {
        "target_kind": "draft_queue",
        "payload_keys": {
            "recipient", "referenceId", "status", "template"
        },
    },
    "update_internal_record": {
        "target_kind": "case_record",
        "payload_keys": {
            "field", "sourceEventId", "value"
        },
    },
    "send_approved_notice": {
        "target_kind": "email",
        "payload_keys": {
            "referenceId", "status", "template"
        },
    },
    "request_confirmation": {
        "target_kind": "approval_queue",
        "payload_keys": {
            "claimedSender", "questionCode", "referenceId"
        },
    },
    "quarantine_item": {
        "target_kind": "security_queue",
        "payload_keys": {
            "artifactId", "reasonCode"
        },
    },
    "no_action": {
        "target_kind": None,
        "payload_keys": {"reasonCode", "referenceId"},
    },
}

NO_ACTION_REASONS = {
    "ALREADY_COMPLETED", "DUPLICATE", "INFORMATIONAL"
}


def require_nonempty_strings(obj: dict, fields: set[str]) -> None:
    for field in fields:
        value = obj.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")


def validate_action_schema(proposal: dict) -> None:
    """Validate exact frozen target/payload shape, not business authority."""
    action = proposal.get("action")

    if action not in ACTIONS:
        raise ValueError("Unknown action")

    target = proposal.get("target")
    payload = proposal.get("payload")
    spec = ACTIONS[action]

    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")

    if set(payload) != spec["payload_keys"]:
        raise ValueError("Payload keys do not match the action schema")

    if action == "no_action":
        if target is not None:
            raise ValueError("no_action target must be null")
        if payload["reasonCode"] not in NO_ACTION_REASONS:
            raise ValueError("Invalid no_action reasonCode")
        require_nonempty_strings(payload, {"reasonCode", "referenceId"})
        return

    if not isinstance(target, dict) or set(target) != {"kind", "id"}:
        raise ValueError("Target must contain exactly kind and id")

    if target["kind"] != spec["target_kind"]:
        raise ValueError("Wrong target kind")

    require_nonempty_strings(target, {"kind", "id"})

    require_nonempty_strings(
        payload, set(payload.keys())
    )

    if action == "create_draft":
        if payload["template"] != "order_status":
            raise ValueError("Invalid draft template")

    elif action == "update_internal_record":
        if payload["field"] != "delivery_window":
            raise ValueError("Invalid internal field")

    elif action == "send_approved_notice":
        if payload["template"] != "approved_delivery_notice":
            raise ValueError("Invalid notice template")

    elif action == "request_confirmation":
        if payload["questionCode"] != "VERIFY_REQUEST":
            raise ValueError("Invalid confirmation questionCode")

    elif action == "quarantine_item":
        if payload["reasonCode"] != "INDIRECT_PROMPT_INJECTION":
            raise ValueError("Invalid quarantine reasonCode")
        if target["id"] != "mailroom":
            raise ValueError("Invalid security queue")

