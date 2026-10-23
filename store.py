import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core import canonical_json

# Point MAILROOM_DB_PATH at a persistent disk (e.g. a Render disk mount)
# so the decision cache survives restarts and redeploys.
DEFAULT_DB = Path(
    os.getenv("MAILROOM_DB_PATH")
    or Path(__file__).resolve().parent / "mailroom.db"
)


class MailroomStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = str(db_path)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(
            self.db_path,
            timeout=10,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self):
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    input_digest TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    verifier_json TEXT NOT NULL,
                    propose_response_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    commit_request_json TEXT,
                    commit_response_json TEXT
                );

                CREATE TABLE IF NOT EXISTS proposals (
                    evaluation_id TEXT NOT NULL,
                    dossier_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    proposal_digest TEXT NOT NULL,
                    PRIMARY KEY (evaluation_id, dossier_id),
                    UNIQUE (evaluation_id, call_id),
                    FOREIGN KEY (evaluation_id)
                        REFERENCES evaluations(evaluation_id)
                );

                CREATE TABLE IF NOT EXISTS receipts (
                    evaluation_id TEXT NOT NULL,
                    dossier_id TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    outcome_json TEXT NOT NULL,
                    PRIMARY KEY (evaluation_id, dossier_id),
                    FOREIGN KEY (evaluation_id, dossier_id)
                        REFERENCES proposals(evaluation_id, dossier_id)
                );

                CREATE TABLE IF NOT EXISTS decision_cache (
                    fingerprint TEXT PRIMARY KEY,
                    proposal_json TEXT NOT NULL
                );
            """)

    def get_evaluation(self, evaluation_id: str):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
            return dict(row) if row else None

    def save_evaluation(
        self,
        evaluation_id: str,
        input_digest: str,
        request: dict[str, Any],
        verifier: dict[str, Any],
        propose_response: dict[str, Any],
        proposals: list[dict[str, Any]],
    ):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute(
                    "SELECT * FROM evaluations WHERE evaluation_id = ?",
                    (evaluation_id,),
                ).fetchone()

                request_json = canonical_json(request)

                if existing:
                    if (
                        existing["input_digest"] != input_digest
                        or existing["request_json"] != request_json
                    ):
                        raise ValueError(
                            "evaluationId already exists with changed content"
                        )
                    db.commit()
                    return False

                db.execute(
                    """INSERT INTO evaluations (
                        evaluation_id, input_digest, request_json,
                        verifier_json, propose_response_json, status
                    ) VALUES (?, ?, ?, ?, ?, 'awaiting_receipts')""",
                    (
                        evaluation_id,
                        input_digest,
                        request_json,
                        canonical_json(verifier),
                        canonical_json(propose_response),
                    ),
                )

                for proposal in proposals:
                    db.execute(
                        """INSERT INTO proposals (
                            evaluation_id, dossier_id, call_id,
                            action, proposal_json, proposal_digest
                        ) VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            evaluation_id,
                            proposal["dossierId"],
                            proposal["callId"],
                            proposal["action"],
                            canonical_json(proposal),
                            proposal["proposalDigest"],
                        ),
                    )

                db.commit()
                return True
            except Exception:
                db.rollback()
                raise

    def get_proposals(self, evaluation_id: str):
        with self.connect() as db:
            rows = db.execute(
                """SELECT * FROM proposals
                   WHERE evaluation_id = ? ORDER BY dossier_id""",
                (evaluation_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_cached_decision(self, fingerprint: str):
        with self.connect() as db:
            row = db.execute(
                "SELECT proposal_json FROM decision_cache WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            return json.loads(row["proposal_json"]) if row else None

    def cache_decision(self, fingerprint: str, proposal: dict[str, Any]):
        with self.connect() as db:
            db.execute(
                """INSERT INTO decision_cache (fingerprint, proposal_json)
                   VALUES (?, ?)
                   ON CONFLICT(fingerprint) DO NOTHING""",
                (fingerprint, canonical_json(proposal)),
            )

    def used_receipt_ids(self, receipt_ids: list[str]) -> set[str]:
        """Receipt IDs already recorded by any committed evaluation."""
        if not receipt_ids:
            return set()
        placeholders = ",".join("?" for _ in receipt_ids)
        with self.connect() as db:
            rows = db.execute(
                f"""SELECT json_extract(receipt_json, '$.receiptId') AS rid
                    FROM receipts
                    WHERE json_extract(receipt_json, '$.receiptId')
                          IN ({placeholders})""",
                list(receipt_ids),
            ).fetchall()
            return {row["rid"] for row in rows}

    def export_evaluations(self, limit: int = 4):
        """Most recent evaluations, for the token-protected debug export."""
        with self.connect() as db:
            rows = db.execute(
                """SELECT evaluation_id, request_json, propose_response_json,
                          status, commit_request_json, commit_response_json
                   FROM evaluations ORDER BY rowid DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        def load(text):
            return json.loads(text) if text else None

        return [
            {
                "evaluationId": row["evaluation_id"],
                "status": row["status"],
                "request": load(row["request_json"]),
                "proposeResponse": load(row["propose_response_json"]),
                "commitRequest": load(row["commit_request_json"]),
                "commitResponse": load(row["commit_response_json"]),
            }
            for row in rows
        ]

    def get_cached_commit(self, evaluation_id: str):
        with self.connect() as db:
            row = db.execute(
                """SELECT commit_response_json FROM evaluations
                   WHERE evaluation_id = ?""",
                (evaluation_id,),
            ).fetchone()

            if not row or not row["commit_response_json"]:
                return None

            return json.loads(row["commit_response_json"])
    
    def save_commit(
        self,
        evaluation_id: str,
        commit_request: dict[str, Any],
        receipts_and_outcomes: list[tuple[dict[str, Any], dict[str, Any]]],
        commit_response: dict[str, Any],
    ):
        """Persist a complete commit atomically; never save partial outcomes."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT status, commit_request_json, commit_response_json "
                    "FROM evaluations WHERE evaluation_id = ?",
                    (evaluation_id,),
                ).fetchone()

                if row is None:
                    raise ValueError("Unknown evaluationId")

                request_json = canonical_json(commit_request)

                if row["commit_response_json"]:
                    if row["commit_request_json"] != request_json:
                        raise ValueError(
                            "evaluationId already committed with different content"
                        )
                    db.commit()
                    return False

                for receipt, outcome in receipts_and_outcomes:
                    db.execute(
                        """INSERT INTO receipts (
                            evaluation_id, dossier_id, receipt_json, outcome_json
                        ) VALUES (?, ?, ?, ?)""",
                        (
                            evaluation_id,
                            receipt["dossierId"],
                            canonical_json(receipt),
                            canonical_json(outcome),
                        ),
                    )

                db.execute(
                    """UPDATE evaluations
                       SET status = 'committed',
                           commit_request_json = ?,
                           commit_response_json = ?
                       WHERE evaluation_id = ?""",
                    (
                        request_json,
                        canonical_json(commit_response),
                        evaluation_id,
                    ),
                )

                db.commit()
                return True
            except Exception:
                db.rollback()
                raise

