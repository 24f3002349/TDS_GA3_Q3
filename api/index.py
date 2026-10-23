import base64
import json
import sys
import threading
import time

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
from decide import ModelConfigError, decide_batch
from store import MailroomStore


app = FastAPI(
    title="TDS GA3 Safe AI Mailroom",
    version="1.0.0",
)

store = MailroomStore()
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024


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
# One extra model pass for dossiers whose first decision was missing or
# invalid, but only while enough of the request's time budget remains.
# The grader allows 55 seconds per request; keep a margin for persistence.
FIRST_PASS_SECONDS = 30
RETRY_MIN_REMAINING_SECONDS = 12
REQUEST_BUDGET_SECONDS = 43

# Serialize model work so concurrent evaluations sharing the same core
# dossiers reuse one set of cached decisions instead of paying twice.
decision_lock = threading.Lock()


def make_proposal(dossier, decision):
    return {
        "dossierId": dossier["dossierId"],
        "callId": make_call_id(dossier),
        "action": decision["action"],
        "target": decision["target"],
        "payload": decision["payload"],
        "evidence": decision["evidence"],
    }


def accept_decisions(generated, pending):
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
        try:
            validate_proposal(make_proposal(dossier, decision), dossier)
        except (ValueError, TypeError, KeyError, AttributeError):
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


def generate_decisions(missing, started):
    pending = {d["dossierId"]: d for d in missing}
    decisions = {}

    budget = FIRST_PASS_SECONDS

    for _ in range(2):
        if not pending:
            break

        generated = decide_batch(list(pending.values()), budget_seconds=budget)
        decided = accept_decisions(generated, pending)
        decisions.update(decided)
        for dossier_id in decided:
            del pending[dossier_id]

        budget = REQUEST_BUDGET_SECONDS - (time.monotonic() - started)
        if budget < RETRY_MIN_REMAINING_SECONDS:
            break

    return decisions, pending


def build_proposals(dossiers):
    started = time.monotonic()
    fingerprints = {
        dossier["dossierId"]: content_fingerprint(dossier)
        for dossier in dossiers
    }

    with decision_lock:
        decisions = {}
        missing = []

        for dossier in dossiers:
            fingerprint = fingerprints[dossier["dossierId"]]
            cached = store.get_cached_decision(fingerprint)

            if cached is None:
                missing.append(dossier)
            else:
                decisions[dossier["dossierId"]] = cached

        undecided = {}
        if missing:
            try:
                generated, undecided = generate_decisions(missing, started)
            except ModelConfigError as exc:
                raise RuntimeError(str(exc)) from exc

            # Cache only validated model decisions, never fallbacks.
            for dossier_id, decision in generated.items():
                store.cache_decision(fingerprints[dossier_id], decision)
            decisions.update(generated)

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


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "service": PROFILE,
        "status": "ready",
    }


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
        return error_response(404, "Unknown evaluationId")

    if evaluation["commit_response_json"]:
        if evaluation["commit_request_json"] != canonical_json(data):
            return error_response(409, "Evaluation already committed")
        return JSONResponse(
            content=json.loads(evaluation["commit_response_json"])
        )

    if digest != evaluation["input_digest"]:
        return error_response(409, "inputDigest does not match evaluation")

    if evaluation["status"] != "awaiting_receipts":
        return error_response(409, "Evaluation is not awaiting receipts")

    saved_propose = json.loads(evaluation["propose_response_json"])
    public_proposals = saved_propose["proposals"]

    proposal_rows = store.get_proposals(evaluation_id)
    proposal_by_id = {
        row["dossier_id"]: json.loads(row["proposal_json"])
        for row in proposal_rows
    }

    if len(receipts) != len(public_proposals):
        return error_response(400, "One receipt is required per proposal")

    receipt_by_id = {}
    seen_call_ids = set()
    seen_receipt_ids = set()

    for receipt in receipts:
        dossier_id = receipt["dossierId"]

        if dossier_id in receipt_by_id:
            return error_response(400, "Duplicate receipt dossierId")

        if receipt["callId"] in seen_call_ids:
            return error_response(400, "Duplicate receipt callId")

        if receipt["receiptId"] in seen_receipt_ids:
            return error_response(400, "Duplicate receiptId")

        receipt_by_id[dossier_id] = receipt
        seen_call_ids.add(receipt["callId"])
        seen_receipt_ids.add(receipt["receiptId"])

    if set(receipt_by_id) != set(proposal_by_id):
        return error_response(400, "Missing or unexpected receipt dossierId")

    verifier = json.loads(evaluation["verifier_json"])
    public_key_jwk = verifier["publicKeyJwk"]
    receipts_and_outcomes = []

    # Verify the entire batch before saving any outcomes.
    for public_proposal in public_proposals:
        dossier_id = public_proposal["dossierId"]
        proposal = proposal_by_id.get(dossier_id)
        receipt = receipt_by_id.get(dossier_id)

        if proposal is None or receipt is None:
            return error_response(400, "Receipt has no matching proposal")

        if (
            receipt["callId"] != proposal["callId"]
            or receipt["action"] != proposal["action"]
            or receipt["proposalDigest"] != proposal["proposalDigest"]
        ):
            return error_response(400, "Receipt does not match its proposal")

        if not verify_receipt_signature(
            evaluation_id=evaluation_id,
            input_digest=digest,
            receipt=receipt,
            public_key_jwk=public_key_jwk,
        ):
            return error_response(400, "Invalid receipt signature")

        outcome = {
            "dossierId": proposal["dossierId"],
            "callId": proposal["callId"],
            "action": proposal["action"],
            "proposalDigest": proposal["proposalDigest"],
            "receiptId": receipt["receiptId"],
            "status": "executed" if receipt["accepted"] else "rejected",
        }

        receipts_and_outcomes.append((receipt, outcome))

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
        return error_response(415, "Content-Type must be application/json")

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


@app.post("/")
async def propose_root(request: Request):
    return await handle_propose(request)


@app.post("/a2a/")
async def propose_a2a(request: Request):
    return await handle_propose(request)
