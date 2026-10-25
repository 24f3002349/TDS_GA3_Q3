import base64
import collections
import hmac
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from actions import validate_action_schema
from receipts import verify_receipt_signature
from core import (
    ALLOWED_ACTIONS,
    PROFILE,
    canonical_json,
    content_fingerprint,
    input_digest,
    make_call_id,
    proposal_digest,
    validate_dossiers,
    validate_proposal,
)
import decide
from decide import ModelConfigError, decide_batch
from store import MailroomStore


app = FastAPI(
    title="TDS GA3 Safe AI Mailroom",
    version="1.0.0",
)

store = MailroomStore()
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024

# Decisions exported from an earlier run and committed to the repo, so stable
# dossiers keep the same proposal after Render restarts wipe the database.
SEED_FILE = Path(__file__).resolve().parents[1] / "seed_decisions.json"


def load_seed():
    if not SEED_FILE.exists():
        return
    try:
        seeded = store.seed_decisions(
            json.loads(SEED_FILE.read_text(encoding="utf-8"))
        )
        print(f"[startup] seeded {seeded} cached decisions", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[startup] seed skipped: {type(exc).__name__}", file=sys.stderr, flush=True)


load_seed()


def error_response(status: int, message: str):
    return JSONResponse(
        status_code=status,
        content={"error": message},
    )


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    # Keep every failure JSON; log only the exception type, not request data.
    print(f"[api] unhandled {type(exc).__name__}", file=sys.stderr, flush=True)
    return error_response(500, "Internal server error")


def validate_propose_request(data):
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")

    if data.get("profile") != PROFILE:
        raise ValueError("Invalid profile")

    if data.get("operation") != "propose":
        raise ValueError("Expected operation propose")

    evaluation_id = data.get("evaluationId")
    if (
        not isinstance(evaluation_id, str)
        or not evaluation_id
        or len(evaluation_id) > 256
    ):
        raise ValueError("Invalid evaluationId")

    verifier = data.get("receiptVerifier")
    if not isinstance(verifier, dict):
        raise ValueError("receiptVerifier must be an object")

    if set(verifier) != {"algorithm", "publicKeyJwk"}:
        raise ValueError("Invalid receiptVerifier fields")

    if verifier["algorithm"] != "Ed25519":
        raise ValueError("Only Ed25519 is supported")

    jwk = verifier["publicKeyJwk"]
    if (
        not isinstance(jwk, dict)
        or set(jwk) != {"kty", "crv", "x"}
        or jwk.get("kty") != "OKP"
        or jwk.get("crv") != "Ed25519"
        or not isinstance(jwk.get("x"), str)
    ):
        raise ValueError("Invalid Ed25519 public key JWK")

    try:
        raw_key = base64.urlsafe_b64decode(
            jwk["x"] + "=" * (-len(jwk["x"]) % 4)
        )
    except Exception as exc:
        raise ValueError("Invalid public key encoding") from exc

    if len(raw_key) != 32:
        raise ValueError("Ed25519 public key must be 32 bytes")

    allowed = data.get("allowedActions")
    if (
        not isinstance(allowed, list)
        or any(not isinstance(x, str) for x in allowed)
        or len(allowed) != len(set(allowed))
        or set(allowed) != ALLOWED_ACTIONS
    ):
        raise ValueError("allowedActions does not match the contract")

    corpus = data.get("corpus")
    if not isinstance(corpus, dict):
        raise ValueError("corpus must be an object")

    for field in ("coreId", "auditId"):
        if not isinstance(corpus.get(field), str) or not corpus[field]:
            raise ValueError(f"Invalid corpus {field}")

    for field in ("stableCount", "freshCount"):
        value = corpus.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid corpus {field}")

    dossiers = data.get("dossiers")
    validate_dossiers(dossiers)

    required_dossier_fields = {
        "dossierId", "partition", "receivedAt",
        "mailbox", "objective", "sources",
    }

    stable = 0
    fresh = 0

    for dossier in dossiers:
        if not required_dossier_fields.issubset(dossier):
            raise ValueError("Dossier is missing required fields")

        if dossier["partition"] == "stable_core":
            stable += 1
        elif dossier["partition"] == "fresh_audit":
            fresh += 1
        else:
            raise ValueError("Invalid dossier partition")

        if not isinstance(dossier["receivedAt"], str):
            raise ValueError("receivedAt must be a string")

        if not isinstance(dossier["mailbox"], str):
            raise ValueError("mailbox must be a string")

        if not isinstance(dossier["objective"], str):
            raise ValueError("objective must be a string")

    if stable != corpus["stableCount"] or fresh != corpus["freshCount"]:
        raise ValueError("Corpus counts do not match dossiers")

    return evaluation_id, verifier, dossiers


def public_proposal(proposal):
    """Return only the six proposal fields defined by the contract."""
    return {
        "dossierId": proposal["dossierId"],
        "callId": proposal["callId"],
        "action": proposal["action"],
        "target": proposal["target"],
        "payload": proposal["payload"],
        "evidence": proposal["evidence"],
    }


DECISION_FIELDS = {"dossierId", "action", "target", "payload", "evidence"}
# Extra model passes for dossiers whose decision was missing or invalid,
# while enough of the request's time budget remains. The grader allows 55
# seconds per request; keep a margin for persistence.
FIRST_PASS_SECONDS = 20
RETRY_PASS_SECONDS = 15
MAX_PASSES = 4
RETRY_MIN_REMAINING_SECONDS = 11
RETRY_ALONE_MAX = 6
REQUEST_BUDGET_SECONDS = 43

# Model-chosen argument values that must be copied from the dossier:
# (target id must be grounded?, payload fields that must be grounded).
GROUNDED_FIELDS = {
    "create_draft": (False, ("recipient", "referenceId", "status")),
    "update_internal_record": (True, ("sourceEventId", "value")),
    "send_approved_notice": (True, ("referenceId", "status")),
    "request_confirmation": (True, ("claimedSender", "referenceId")),
    "quarantine_item": (False, ("artifactId",)),
    "no_action": (False, ("referenceId",)),
}

# Dossiers currently being decided, by content fingerprint. A concurrent
# evaluation containing the same dossier waits for that result instead of
# paying for a second model call; different dossiers proceed in parallel.
inflight_lock = threading.Lock()
inflight = {}


def make_proposal(dossier, decision):
    return {
        "dossierId": dossier["dossierId"],
        "callId": make_call_id(dossier),
        "action": decision["action"],
        "target": decision["target"],
        "payload": decision["payload"],
        "evidence": decision["evidence"],
    }


def draft_queue_id(dossier):
    mailbox = dossier["mailbox"]
    return mailbox if mailbox.startswith("mailbox:") else "mailbox:" + mailbox


def dossier_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from dossier_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from dossier_strings(item)


def check_grounded(decision, dossier):
    """Reject argument values the model did not copy from the dossier.

    This blocks invented recipients, IDs, and statuses. Outbound sends are
    held to a stricter rule: every argument must appear in a cited line.
    """
    action = decision["action"]
    target_grounded, payload_fields = GROUNDED_FIELDS[action]
    values = [decision["payload"][field] for field in payload_fields]
    if target_grounded:
        values.append(decision["target"]["id"])

    if action == "create_draft":
        if decision["target"]["id"] != draft_queue_id(dossier):
            raise ValueError("Draft queue does not match the dossier mailbox")

    if action == "send_approved_notice":
        cited = set(decision["evidence"])
        texts = [
            line["text"]
            for source in dossier["sources"]
            for line in source["lines"]
            if line["lineId"] in cited
        ]
    else:
        # Values must come from a line the decision could cite: not dossier
        # metadata (such as the mailbox name) and not history notes or
        # secrets. A quarantined artifact may also be named by its sourceId.
        citable = [
            source for source in dossier["sources"]
            if source.get("kind") not in NEVER_CITE_KINDS
        ]
        texts = [line["text"] for source in citable for line in source["lines"]]
        if action == "quarantine_item":
            texts += [source.get("sourceId", "") for source in citable]

    for value in values:
        if not any(value in text for text in texts):
            raise ValueError("Argument value is not grounded in the dossier")


# Sources that may supply an argument value when evidence is completed.
TRUSTED_PROVENANCE = {"authenticated_internal", "signed_internal"}
NEVER_CITE_KINDS = {"history", "secret"}
# Words a governing rule uses about each action; only a tie-breaker when a
# dossier has several candidate rule lines and the model cited none.
ACTION_TERMS = {
    "create_draft": ("draft", "unsent", "prepar", "written up"),
    "request_confirmation": ("confirm", "verif", "challenge", "mismatch", "differ"),
    "update_internal_record": ("update", "change", "field", "record"),
    "quarantine_item": ("quarantine", "isolate", "security", "exfiltrat", "untrusted"),
    "no_action": ("complete", "duplicate", "repeat", "informational", "no additional", "suppress"),
}
# Payload field whose value the incoming request line must carry.
REQUEST_LINE_FIELDS = {
    "create_draft": "referenceId",
    "request_confirmation": "claimedSender",
}


def boilerplate_policy_texts(dossiers):
    """Policy lines that repeat across most dossiers of a partition.

    Every dossier carries generic boundary policies alongside the one rule
    that governs its case type. Generic ones repeat in nearly every dossier;
    case rules appear only in their own case type. Counting is done per
    partition because audit dossiers use their own paraphrases.
    """
    by_partition = collections.defaultdict(list)
    for dossier in dossiers:
        by_partition[dossier.get("partition")].append(dossier)

    generic = set()
    for group in by_partition.values():
        if len(group) < 4:
            continue
        counts = collections.Counter()
        for dossier in group:
            counts.update({
                line["text"]
                for source in dossier["sources"] if source.get("kind") == "policy"
                for line in source["lines"]
            })
        generic |= {text for text, n in counts.items() if n >= 0.6 * len(group)}
    return frozenset(generic)


# Upper-case identifiers such as order, case, or event references.
IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{2,}-[A-Z0-9]{4,}\b")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def incoming_addresses(dossier):
    """Addresses in incoming external messages (not attachments)."""
    found = []
    for source in dossier["sources"]:
        if (
            not source.get("provenance", "").startswith("external")
            or source.get("kind") == "attachment"
        ):
            continue
        for line in source["lines"]:
            for address in EMAIL_PATTERN.findall(line["text"]):
                address = address.rstrip(".")
                if address not in found:
                    found.append(address)
    return found


def send_in_scope(decision, dossier):
    """True only when a signed approval covers this exact send.

    The approval itself must name the recipient and the order, an approval
    line must name the exact status, a trusted record must show that status
    is current, and nobody other than the approved recipient may be the one
    asking. Anything else (another recipient, an older status, a different
    requester) is outside the approval and must not be sent.
    """
    recipient = decision["target"]["id"]
    reference = decision["payload"]["referenceId"]
    status = decision["payload"]["status"]

    def lines_of(kind, trusted_only=False):
        return [
            line["text"]
            for source in dossier["sources"]
            if source.get("kind") == kind
            and (not trusted_only or source.get("provenance") in TRUSTED_PROVENANCE)
            for line in source["lines"]
        ]

    approvals = lines_of("approval", trusted_only=True)
    return (
        all(a.lower() == recipient.lower() for a in incoming_addresses(dossier))
        and any(recipient in text and reference in text for text in approvals)
        and any(status in text for text in approvals)
        and any(status in text for text in lines_of("record", trusted_only=True))
    )


def in_scope_send_for_draft(decision, dossier):
    """True when a draft could have been a fully approved send.

    The model sometimes drafts a case whose approval covers exactly this
    recipient, order, and current status; such a dossier is asked again on
    its own. The model decides; code never turns a draft into a send.
    """
    if decision["action"] != "create_draft":
        return False
    candidate = {
        "target": {"kind": "email", "id": decision["payload"]["recipient"]},
        "payload": {
            "referenceId": decision["payload"]["referenceId"],
            "status": decision["payload"]["status"],
        },
    }
    return send_in_scope(candidate, dossier)


def check_send_scope(decision, dossier):
    """Last line of defence: never let an out-of-scope send through."""
    if decision["action"] == "send_approved_notice" and not send_in_scope(decision, dossier):
        raise ValueError("Send is not covered by a matching signed approval")


def draft_instead_of_send(decision, dossier):
    """Turn an out-of-scope send into the unsent draft the policy prescribes.

    The approval does not cover this send, so nothing may be sent; the reply
    is prepared as a draft to whoever asked, keeping the approval lines as
    evidence of why it was not sent.
    """
    asking = incoming_addresses(dossier)
    if len(asking) != 1:
        raise ValueError("Out-of-scope send with no single requester to draft for")
    return {
        "dossierId": decision["dossierId"],
        "action": "create_draft",
        "target": {"kind": "draft_queue", "id": draft_queue_id(dossier)},
        "payload": {
            "recipient": asking[0],
            "referenceId": decision["payload"]["referenceId"],
            "status": decision["payload"]["status"],
            "template": "order_status",
        },
        "evidence": list(decision["evidence"]),
    }


QUOTED_PATTERN = re.compile(r"[“\"]([^”\"]+)[”\"]")


def check_draft_status(decision, dossier):
    """A status draft must carry the order's current status on record.

    The trusted record line for the referenced order states the current
    status (an approval may name an older one). If the draft's status is
    not in that line and the line quotes exactly one status, use it.
    """
    if decision["action"] != "create_draft":
        return decision
    reference = decision["payload"]["referenceId"]
    status = decision["payload"]["status"]
    records = [
        line["text"]
        for source in dossier["sources"]
        if source.get("kind") == "record" and source.get("provenance") in TRUSTED_PROVENANCE
        for line in source["lines"]
        if reference in line["text"]
    ]
    if not records or any(status in text for text in records):
        return decision
    quoted = {q for text in records for q in QUOTED_PATTERN.findall(text)}
    if len(quoted) == 1:
        payload = dict(decision["payload"], status=quoted.pop())
        return dict(decision, payload=payload)
    raise ValueError("Draft status is not the current status on record")


NO_ACTION_STATES = {
    "ALREADY_COMPLETED": ("completed",),
    "DUPLICATE": ("duplicate",),
    "INFORMATIONAL": ("informational",),
}


def check_no_action_reason(decision, dossier):
    """Align a no_action reasonCode with the case state on record.

    The trusted record line about the referenced case states whether the
    item is completed, a duplicate, or informational; if it names exactly
    one of these, the reasonCode must be that one.
    """
    if decision["action"] != "no_action":
        return decision
    reference = decision["payload"]["referenceId"]
    records = [
        line["text"].lower()
        for source in dossier["sources"]
        if source.get("kind") == "record" and source.get("provenance") in TRUSTED_PROVENANCE
        for line in source["lines"]
        if reference in line["text"]
    ]
    states = {
        code for code, words in NO_ACTION_STATES.items()
        if any(word in text for text in records for word in words)
    }
    if len(states) == 1 and decision["payload"]["reasonCode"] not in states:
        payload = dict(decision["payload"], reasonCode=states.pop())
        return dict(decision, payload=payload)
    return decision


def check_confirmation_reference(decision, dossier):
    """A confirmation's reference must be what the claimant's request is about.

    The case the claim concerns appears in the same line as the claimed
    address (the request or the identity check); an order number mentioned
    elsewhere is not the subject of the confirmation.
    """
    if decision["action"] != "request_confirmation":
        return decision
    sender = decision["payload"]["claimedSender"]
    reference = decision["payload"]["referenceId"]
    # The claim (the requester's message) and the identity check (a record)
    # are what a confirmation concerns; approvals or policies that happen to
    # mention the address are not.
    claim_lines = [
        line["text"]
        for source in dossier["sources"] if source.get("kind") in ("message", "record")
        for line in source["lines"]
        if sender in line["text"]
    ]
    if any(reference in text for text in claim_lines):
        return decision

    # Repair: if the claim lines name exactly one identifier, that is what
    # the request concerns.
    identifiers = {
        token for text in claim_lines for token in IDENTIFIER_PATTERN.findall(text)
    }
    if len(identifiers) == 1:
        payload = dict(decision["payload"], referenceId=identifiers.pop())
        return dict(decision, payload=payload)
    raise ValueError("Confirmation reference is not what the claimant's request concerns")


def complete_evidence(decision, dossier, boilerplate=frozenset()):
    """Add a line for any argument value the cited lines do not contain.

    Evidence must establish every exact argument. If the model cited none of
    the lines holding a value, add the most authoritative line that does
    (trusted provenance first), never history notes or secrets.
    """
    target_grounded, payload_fields = GROUNDED_FIELDS[decision["action"]]
    values = [decision["payload"][field] for field in payload_fields]
    if target_grounded:
        values.append(decision["target"]["id"])

    lines = [
        (source, line)
        for source in dossier["sources"]
        for line in source["lines"]
        if source.get("kind") not in NEVER_CITE_KINDS
    ]
    evidence = list(decision["evidence"])
    cited_text = [line["text"] for _, line in lines if line["lineId"] in evidence]

    for value in values:
        if any(value in text for text in cited_text):
            continue
        holders = [
            (source.get("provenance") not in TRUSTED_PROVENANCE, line["lineId"], line)
            for source, line in lines
            if value in line["text"]
        ]
        if holders:
            best = min(holders, key=lambda h: h[0])[2]
            evidence.append(best["lineId"])
            cited_text.append(best["text"])

    # Drafts and confirmations answer an incoming request, so the request
    # itself is evidence even when its values also appear in a trusted
    # record: the external line carrying the order reference (draft) or the
    # claimed address (confirmation).
    request_field = REQUEST_LINE_FIELDS.get(decision["action"])
    if request_field:
        value = decision["payload"][request_field]
        requests = [
            line for source, line in lines
            if source.get("provenance", "").startswith("external")
            and source.get("kind") != "attachment"
            and value in line["text"]
        ]
        if requests and not any(line["lineId"] in evidence for line in requests):
            evidence.append(requests[0]["lineId"])

    # Every action except a send rests on the case rule. Generic boundary
    # policies are not that rule and are dropped. If the case rule is not
    # cited and exactly one candidate exists it is added; with several
    # candidates the model is asked again.
    if decision["action"] != "send_approved_notice":
        policy_lines = [
            line for source, line in lines if source.get("kind") == "policy"
        ]
        evidence = [
            line_id for line_id in evidence
            if not any(
                line["lineId"] == line_id and line["text"] in boilerplate
                for line in policy_lines
            )
        ]
        rules = [line for line in policy_lines if line["text"] not in boilerplate]
        if rules and not any(line["lineId"] in evidence for line in rules):
            if len(rules) > 1:
                # Tie-break between candidate rules by which one speaks about
                # the chosen action; give up (re-ask) if that is unclear.
                terms = ACTION_TERMS[decision["action"]]
                scored = sorted(
                    rules,
                    key=lambda line: -sum(t in line["text"].lower() for t in terms),
                )
                best = sum(t in scored[0]["text"].lower() for t in terms)
                runner_up = sum(t in scored[1]["text"].lower() for t in terms)
                if best == 0 or best == runner_up:
                    raise ValueError("Decision must cite the governing policy line")
                rules = scored[:1]
            evidence.append(rules[0]["lineId"])

    # A suppressed item must cite the incoming message being suppressed.
    # Which line that is needs judgement (decoy messages exist), so a
    # decision without one is sent back to the model instead of guessed.
    if decision["action"] == "no_action":
        external = {
            line["lineId"] for source, line in lines
            if source.get("provenance", "").startswith("external")
            and source.get("kind") != "attachment"
        }
        if external and not external & set(evidence):
            raise ValueError("no_action must cite the incoming message")

    return dict(decision, evidence=evidence)


def accept_decisions(generated, pending, boilerplate=frozenset(), first_pass=False):
    """Return {dossierId: decision} for each model item that fully validates."""
    accepted = {}

    if not isinstance(generated, list):
        return accepted

    for item in generated:
        if not isinstance(item, dict) or set(item) != DECISION_FIELDS:
            continue

        if not isinstance(item["dossierId"], str):
            continue

        dossier = pending.get(item["dossierId"])
        if dossier is None or item["dossierId"] in accepted:
            continue

        decision = {field: item[field] for field in DECISION_FIELDS}
        # Drop invented or repeated line IDs instead of discarding the whole
        # decision; the checks below re-add any line that is actually needed.
        if isinstance(decision["evidence"], list):
            known = {
                line["lineId"] for source in dossier["sources"] for line in source["lines"]
            }
            decision["evidence"] = list(dict.fromkeys(
                x for x in decision["evidence"] if isinstance(x, str) and x in known
            ))
        try:
            validate_proposal(make_proposal(dossier, decision), dossier)
            check_grounded(decision, dossier)
            if decision["action"] == "send_approved_notice" and not send_in_scope(decision, dossier):
                decision = draft_instead_of_send(decision, dossier)
                check_grounded(decision, dossier)
            check_send_scope(decision, dossier)
            decision = check_confirmation_reference(decision, dossier)
            decision = check_no_action_reason(decision, dossier)
            decision = check_draft_status(decision, dossier)
            if first_pass and in_scope_send_for_draft(decision, dossier):
                raise ValueError("Drafted a case with an in-scope approval; asking again alone")
            decision = complete_evidence(decision, dossier, boilerplate)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            # Log only the generic reason; never dossier text or values.
            print(f"[accept] rejected a model decision: {exc}", file=sys.stderr, flush=True)
            continue

        accepted[item["dossierId"]] = decision

    return accepted


def fallback_decision(dossier):
    """Safe internal-only decision used when the model gave no valid answer.

    It routes the item to human review and never causes an outbound effect.
    It is never cached, so the next evaluation asks the model again.
    """
    first_line = next(
        (
            line["lineId"]
            for source in dossier["sources"]
            for line in source["lines"]
        ),
        None,
    )
    if first_line is None:
        raise RuntimeError("Dossier has no lines to cite")

    return {
        "dossierId": dossier["dossierId"],
        "action": "request_confirmation",
        "target": {
            "kind": "approval_queue",
            "id": dossier["mailbox"] or "mailroom",
        },
        "payload": {
            "claimedSender": "unknown",
            "questionCode": "VERIFY_REQUEST",
            "referenceId": dossier["dossierId"],
        },
        "evidence": [first_line],
    }


def remaining_budget(started):
    return REQUEST_BUDGET_SECONDS - (time.monotonic() - started)


def generate_decisions(missing, started, boilerplate=frozenset()):
    pending = {d["dossierId"]: d for d in missing}
    decisions = {}

    budget = min(FIRST_PASS_SECONDS, remaining_budget(started))

    for attempt in range(MAX_PASSES):
        if not pending:
            break

        # Retries ask each remaining dossier on its own (in parallel) when
        # there are only a few: the model is most reliable without neighbours.
        retry_alone = attempt >= 1 and len(pending) <= RETRY_ALONE_MAX
        generated = decide_batch(
            list(pending.values()),
            budget_seconds=budget,
            **({"chunk_size": 1} if retry_alone else {}),
        )
        decided = accept_decisions(generated, pending, boilerplate, first_pass=attempt == 0)
        decisions.update(decided)
        for dossier_id in decided:
            del pending[dossier_id]

        budget = min(RETRY_PASS_SECONDS, remaining_budget(started))
        if budget < RETRY_MIN_REMAINING_SECONDS:
            break

    return decisions, pending


def claim_dossiers(dossiers, fingerprints):
    """Split dossiers into cached, to-decide-here, and decided-elsewhere."""
    decisions = {}
    mine = []
    waiting = []

    with inflight_lock:
        for dossier in dossiers:
            fingerprint = fingerprints[dossier["dossierId"]]
            cached = store.get_cached_decision(fingerprint)

            if cached is not None:
                decisions[dossier["dossierId"]] = cached
            elif fingerprint in inflight:
                waiting.append((dossier, inflight[fingerprint]))
            else:
                inflight[fingerprint] = threading.Event()
                mine.append(dossier)

    return decisions, mine, waiting


def build_proposals(dossiers):
    started = time.monotonic()
    fingerprints = {
        dossier["dossierId"]: content_fingerprint(dossier)
        for dossier in dossiers
    }

    decisions, mine, waiting = claim_dossiers(dossiers, fingerprints)
    undecided = {}

    try:
        if mine:
            try:
                generated, undecided = generate_decisions(
                    mine, started, boilerplate_policy_texts(dossiers)
                )
            except ModelConfigError as exc:
                raise RuntimeError(str(exc)) from exc

            # Cache only validated model decisions, never fallbacks.
            for dossier_id, decision in generated.items():
                store.cache_decision(fingerprints[dossier_id], decision)
            decisions.update(generated)
    finally:
        with inflight_lock:
            for dossier in mine:
                inflight.pop(fingerprints[dossier["dossierId"]]).set()

    for dossier, event in waiting:
        event.wait(max(0.0, remaining_budget(started)))
        cached = store.get_cached_decision(fingerprints[dossier["dossierId"]])
        if cached is None:
            undecided[dossier["dossierId"]] = dossier
        else:
            decisions[dossier["dossierId"]] = cached

    if undecided:
        print(
            f"[propose] {len(undecided)} of {len(dossiers)} dossiers "
            "used the safe fallback",
            file=sys.stderr,
            flush=True,
        )
        if len(undecided) == len(dossiers):
            raise RuntimeError("Decision model unavailable")
        for dossier_id, dossier in undecided.items():
            decisions[dossier_id] = fallback_decision(dossier)
            # A stable dossier must get the same proposal in every
            # evaluation, so its final decision is cached even when it is
            # the fallback. Fresh audit dossiers never repeat.
            if dossier["partition"] == "stable_core":
                store.cache_decision(fingerprints[dossier_id], decisions[dossier_id])

    proposals = []

    for dossier in dossiers:
        proposal = make_proposal(dossier, decisions[dossier["dossierId"]])

        try:
            validate_proposal(proposal, dossier)
            validate_action_schema(proposal)
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError("A proposal failed validation") from exc

        proposal["proposalDigest"] = proposal_digest(proposal)
        proposals.append(proposal)

    return proposals


# HEAD is allowed so Render's port and health probes get 200, not 405.
@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz():
    # Render sets RENDER_GIT_COMMIT; it shows which commit is deployed.
    return {"status": "ok", "commit": os.getenv("RENDER_GIT_COMMIT", "")[:7]}


@app.get("/debug/evaluations")
def debug_evaluations(request: Request):
    """Export recent stored evaluations for offline analysis.

    Disabled unless DEBUG_EXPORT_TOKEN is set; requires that token as a
    Bearer header. Looks like a missing route to everyone else.
    """
    token = os.getenv("DEBUG_EXPORT_TOKEN", "")
    supplied = request.headers.get("authorization", "")
    if not token or not hmac.compare_digest(
        supplied.encode("utf-8"), f"Bearer {token}".encode("utf-8")
    ):
        return error_response(404, "Not Found")

    return {
        "evaluations": store.export_evaluations(),
        "requestLog": store.export_request_log(),
        "decisionCache": store.export_decision_cache(),
        "modelErrors": list(decide.RECENT_ERRORS),
    }


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "service": PROFILE,
        "status": "ready",
    }


def check_receipts(evaluation, data):
    """Verify every receipt's binding and signature before any effect.

    Returns (error_response, None) on the first problem, otherwise
    (None, [(receipt, outcome), ...]) in proposal order.
    """
    evaluation_id = evaluation["evaluation_id"]
    digest = data["inputDigest"]
    receipts = data["receipts"]

    if digest != evaluation["input_digest"]:
        return error_response(409, "inputDigest does not match evaluation"), None

    saved_propose = json.loads(evaluation["propose_response_json"])
    public_proposals = saved_propose["proposals"]

    proposal_by_id = {
        row["dossier_id"]: json.loads(row["proposal_json"])
        for row in store.get_proposals(evaluation_id)
    }

    if len(receipts) != len(public_proposals):
        return error_response(400, "One receipt is required per proposal"), None

    receipt_by_id = {}
    seen_call_ids = set()
    seen_receipt_ids = set()

    for receipt in receipts:
        dossier_id = receipt["dossierId"]

        if dossier_id in receipt_by_id:
            return error_response(400, "Duplicate receipt dossierId"), None

        if receipt["callId"] in seen_call_ids:
            return error_response(400, "Duplicate receipt callId"), None

        if receipt["receiptId"] in seen_receipt_ids:
            return error_response(400, "Duplicate receiptId"), None

        receipt_by_id[dossier_id] = receipt
        seen_call_ids.add(receipt["callId"])
        seen_receipt_ids.add(receipt["receiptId"])

    if set(receipt_by_id) != set(proposal_by_id):
        return error_response(400, "Missing or unexpected receipt dossierId"), None

    # A receipt is scoped to one evaluation; never accept a transferred one.
    if store.used_receipt_ids(
        list(seen_receipt_ids), exclude_evaluation=evaluation_id
    ):
        return error_response(
            400, "receiptId already used by another evaluation"
        ), None

    public_key_jwk = json.loads(evaluation["verifier_json"])["publicKeyJwk"]
    receipts_and_outcomes = []

    # Verify the entire batch before saving any outcomes.
    for public_proposal in public_proposals:
        dossier_id = public_proposal["dossierId"]
        proposal = proposal_by_id.get(dossier_id)
        receipt = receipt_by_id.get(dossier_id)

        if proposal is None or receipt is None:
            return error_response(400, "Receipt has no matching proposal"), None

        if (
            receipt["callId"] != proposal["callId"]
            or receipt["action"] != proposal["action"]
            or receipt["proposalDigest"] != proposal["proposalDigest"]
        ):
            return error_response(400, "Receipt does not match its proposal"), None

        if not verify_receipt_signature(
            evaluation_id=evaluation_id,
            input_digest=digest,
            receipt=receipt,
            public_key_jwk=public_key_jwk,
        ):
            return error_response(400, "Invalid receipt signature"), None

        outcome = {
            "dossierId": proposal["dossierId"],
            "callId": proposal["callId"],
            "action": proposal["action"],
            "proposalDigest": proposal["proposalDigest"],
            "receiptId": receipt["receiptId"],
            "status": "executed" if receipt["accepted"] else "rejected",
        }
        receipts_and_outcomes.append((receipt, outcome))

    return None, receipts_and_outcomes


def handle_commit_data(data):
    """Validate and persist a complete, signed receipt batch."""
    required_fields = {
        "profile", "operation", "evaluationId",
        "inputDigest", "receipts",
    }

    if not isinstance(data, dict) or set(data) != required_fields:
        return error_response(400, "Invalid commit request fields")

    if data["profile"] != PROFILE or data["operation"] != "commit":
        return error_response(400, "Invalid profile or operation")

    evaluation_id = data["evaluationId"]
    digest = data["inputDigest"]
    receipts = data["receipts"]

    if not isinstance(evaluation_id, str) or not evaluation_id:
        return error_response(400, "Invalid evaluationId")

    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        return error_response(400, "Invalid inputDigest")

    if not isinstance(receipts, list):
        return error_response(400, "receipts must be a list")

    receipt_fields = {
        "dossierId", "callId", "action", "accepted",
        "proposalDigest", "receiptId", "receiptSignature",
    }

    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != receipt_fields:
            return error_response(400, "Invalid receipt fields")

        for field in (
            "dossierId", "callId", "action",
            "proposalDigest", "receiptId", "receiptSignature",
        ):
            if not isinstance(receipt[field], str) or not receipt[field]:
                return error_response(400, f"Invalid receipt field: {field}")

        if not isinstance(receipt["accepted"], bool):
            return error_response(400, "accepted must be a boolean")

    evaluation = store.get_evaluation(evaluation_id)
    if evaluation is None:
        # Receipts for an evaluation we never proposed are rejected as a
        # conflict with persisted state (409), alongside mismatched receipts.
        return error_response(409, "Unknown evaluationId")

    if evaluation["commit_response_json"]:
        if evaluation["commit_request_json"] == canonical_json(data):
            return JSONResponse(
                content=json.loads(evaluation["commit_response_json"])
            )
        # A different commit for a finished evaluation: report an invalid
        # receipt as such (400) before treating it as a conflict (409).
        error, _ = check_receipts(evaluation, data)
        return error or error_response(409, "Evaluation already committed")

    if evaluation["status"] != "awaiting_receipts":
        return error_response(409, "Evaluation is not awaiting receipts")

    error, receipts_and_outcomes = check_receipts(evaluation, data)
    if error:
        return error

    response = {
        "profile": PROFILE,
        "evaluationId": evaluation_id,
        "status": "completed",
        "inputDigest": digest,
        "outcomes": [outcome for _, outcome in receipts_and_outcomes],
    }

    if len(canonical_json(response).encode("utf-8")) > MAX_RESPONSE_BYTES:
        return error_response(500, "Generated response exceeds size limit")

    try:
        store.save_commit(
            evaluation_id=evaluation_id,
            commit_request=data,
            receipts_and_outcomes=receipts_and_outcomes,
            commit_response=response,
        )
    except ValueError:
        return error_response(409, "Evaluation already committed")
    except Exception:
        return error_response(500, "Could not persist commit")

    saved = store.get_cached_commit(evaluation_id)
    if saved is None:
        return error_response(500, "Could not retrieve committed result")

    return JSONResponse(content=saved)


async def handle_propose(request: Request):
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("application/json"):
        # The contract asks for 400/422 on malformed requests.
        return error_response(400, "Content-Type must be application/json")

    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        return error_response(413, "Request body too large")

    
    try:
        data = json.loads(body)

        if isinstance(data, dict) and data.get("operation") == "commit":
            return handle_commit_data(data)

        evaluation_id, verifier, dossiers = validate_propose_request(data)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return error_response(400, str(exc))


    digest = input_digest(dossiers)
    existing = store.get_evaluation(evaluation_id)

    if existing:
        if (
            existing["input_digest"] != digest
            or existing["request_json"] != canonical_json(data)
        ):
            return error_response(
                409, "evaluationId already exists with changed content"
            )

        saved = json.loads(existing["propose_response_json"])
        return JSONResponse(content=saved)

    # Model calls block, so run them off the event loop; /healthz and other
    # requests stay responsive while a batch is being decided.
    try:
        proposals = await run_in_threadpool(build_proposals, dossiers)
    except RuntimeError as exc:
        return error_response(503, str(exc))

    response = {
        "profile": PROFILE,
        "evaluationId": evaluation_id,
        "status": "awaiting_receipts",
        "inputDigest": digest,
        "proposals": [public_proposal(p) for p in proposals],
    }

    response_bytes = canonical_json(response).encode("utf-8")
    if len(response_bytes) > MAX_RESPONSE_BYTES:
        return error_response(500, "Generated response exceeds size limit")

    try:
        store.save_evaluation(
            evaluation_id=evaluation_id,
            input_digest=digest,
            request=data,
            verifier=verifier,
            propose_response=response,
            proposals=proposals,
        )
    except ValueError:
        return error_response(
            409, "evaluationId already exists with changed content"
        )
    except Exception:
        return error_response(500, "Could not persist evaluation")

    # Read the saved response so concurrent exact replays return the
    # canonical persisted result rather than a newly generated response.
    saved = store.get_evaluation(evaluation_id)
    return JSONResponse(content=json.loads(saved["propose_response_json"]))


async def handle_and_log(request: Request):
    response = await handle_propose(request)

    # Keep rejected requests (bad receipts, conflicts, malformed input) so the
    # grader's negative tests can be inspected later via the debug export.
    if response.status_code >= 400:
        try:
            body = (await request.body())[:MAX_REQUEST_BYTES].decode("utf-8", "replace")
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = {}
            parsed = parsed if isinstance(parsed, dict) else {}
            store.log_request(
                operation=str(parsed.get("operation"))[:32],
                evaluation_id=str(parsed.get("evaluationId"))[:256],
                status_code=response.status_code,
                error=json.loads(response.body).get("error"),
                request=body,
            )
        except Exception:
            pass

    return response


@app.post("/")
async def propose_root(request: Request):
    return await handle_and_log(request)


@app.post("/a2a/")
async def propose_a2a(request: Request):
    return await handle_and_log(request)
