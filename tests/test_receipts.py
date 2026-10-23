import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core import PROFILE, canonical_json
from receipts import verify_receipt_signature

evaluation_id = "local-evaluation-001"
input_digest = "a" * 64

private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

public_bytes = public_key.public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)

b64url = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")

jwk = {
    "kty": "OKP",
    "crv": "Ed25519",
    "x": b64url(public_bytes),
}

receipt = {
    "dossierId": "demo-001",
    "callId": "mailroom:123456789",
    "action": "no_action",
    "accepted": True,
    "proposalDigest": "b" * 64,
    "receiptId": "local-receipt-001",
}

envelope = {
    "profile": PROFILE,
    "evaluationId": evaluation_id,
    "inputDigest": input_digest,
    "receipt": receipt,
}

signature = private_key.sign(
    canonical_json(envelope).encode("utf-8")
)

signed_receipt = {
    **receipt,
    "receiptSignature": base64.b64encode(signature).decode(),
}

assert verify_receipt_signature(
    evaluation_id, input_digest, signed_receipt, jwk
)
print("PASS: valid signature accepted")

tampered_receipt = {**signed_receipt, "accepted": False}

assert not verify_receipt_signature(
    evaluation_id, input_digest, tampered_receipt, jwk
)
print("PASS: tampered receipt rejected")

assert not verify_receipt_signature(
    evaluation_id, input_digest, signed_receipt,
    {**jwk, "x": b64url(b"z" * 32)}
)
print("PASS: wrong public key rejected")

good = signed_receipt["receiptSignature"]
urlsafe = good.replace("+", "-").replace("/", "_")
if urlsafe != good:  # ~6% of signatures contain neither "+" nor "/"
    assert not verify_receipt_signature(
        evaluation_id, input_digest,
        {**signed_receipt, "receiptSignature": urlsafe}, jwk,
    )
assert not verify_receipt_signature(
    evaluation_id, input_digest,
    {**signed_receipt, "receiptSignature": good.rstrip("=")}, jwk,
)
print("PASS: non-canonical signature encodings rejected")

assert not verify_receipt_signature(
    "other-evaluation", input_digest, signed_receipt, jwk
)
assert not verify_receipt_signature(
    evaluation_id, input_digest,
    {**signed_receipt, "dossierId": "demo-002"}, jwk,
)
print("PASS: signature bound to its evaluation and receipt")

print("All receipt signature tests passed.")

