"""Download the grader requests stored by the live service for analysis.

Reads DEBUG_EXPORT_TOKEN from .env (never printed) and saves the export to
captures/ (git-ignored). Run from the project root:

    .\.venv\Scripts\python.exe tools\export_evaluations.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
URL = "https://tds-ga3-q3-bqjf.onrender.com/debug/evaluations"

load_dotenv(ROOT / ".env")
token = os.getenv("DEBUG_EXPORT_TOKEN")
if not token:
    sys.exit("DEBUG_EXPORT_TOKEN is missing from .env")

request = urllib.request.Request(URL, headers={"Authorization": f"Bearer {token}"})
try:
    with urllib.request.urlopen(request, timeout=90) as response:
        data = json.loads(response.read())
except urllib.error.HTTPError as exc:
    sys.exit(
        f"HTTP {exc.code}: is DEBUG_EXPORT_TOKEN set on Render with the "
        "same value as in .env?"
    )

out_dir = ROOT / "captures"
out_dir.mkdir(exist_ok=True)
out = out_dir / f"evaluations-{time.strftime('%Y%m%d-%H%M%S')}.json"
out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

evaluations = data["evaluations"]
print(f"Saved {len(evaluations)} evaluations to {out}")
for evaluation in evaluations:
    dossiers = evaluation["request"]["dossiers"]
    print(f"  {evaluation['evaluationId']}: {len(dossiers)} dossiers, "
          f"status {evaluation['status']}")
