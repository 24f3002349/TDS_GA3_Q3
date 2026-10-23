import base64
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from core import PROFILE, canonical_json

RECEIPT_FIELDS = {
    "dossierId",
    "callId",
    "action",
    "accepted",
    "proposalDigest",
    "receiptId",
    "receiptSignature",
}


def decode_base64url(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("Invalid base64url value")

    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded)
    except Exception as exc:
        raise ValueError("Invalid base64url encoding") from exc


def verify_receipt_signature(
    evaluation_id: str,
    input_digest: str,
    receipt: dict[str, Any],
    public_key_jwk: dict[str, Any],
) -> bool:
    """Verify the exact signed receipt envelope specified by the contract."""
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        return False

    if not isinstance(receipt.get("accepted"), bool):
        return False

    if not isinstance(public_key_jwk, dict):
        return False

    if (
        public_key_jwk.get("kty") != "OKP"
        or public_key_jwk.get("crv") != "Ed25519"
    ):
        return False

    try:
        public_key_bytes = decode_base64url(public_key_jwk["x"])
        encoded_signature = receipt["receiptSignature"]
        if not isinstance(encoded_signature, str):
            return False

        signature = base64.b64decode(encoded_signature, validate=True)

        # Accept only the canonical standard-base64 spelling. Otherwise a
        # modified signature string (url-safe alphabet, altered padding
        # bits) could decode to the same bytes and still verify.
        if base64.b64encode(signature).decode("ascii") != encoded_signature:
            return False

        if len(public_key_bytes) != 32 or len(signature) != 64:
            return False

        signed_receipt = {
            key: value
            for key, value in receipt.items()
            if key != "receiptSignature"
        }

        envelope = {
            "profile": PROFILE,
            "evaluationId": evaluation_id,
            "inputDigest": input_digest,
            "receipt": signed_receipt,
        }

        message = canonical_json(envelope).encode("utf-8")

        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        public_key.verify(signature, message)
        return True

    except (
        KeyError,
        TypeError,
        ValueError,
        InvalidSignature,
    ):
        return False
    except Exception:
        return False
