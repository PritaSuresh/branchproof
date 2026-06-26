"""Publication guards and shared-candidate accounting, without model requests."""

import importlib.util
from pathlib import Path

import pytest

from branchproof.paired_eval import schedule, summarize

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paired_publisher", ROOT / "scripts/summarize_evaluation.py"
)
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


def fixture():
    manifest = {
        "sha256": "frozen",
        "cases": [
            {
                "id": "demo",
                "identity": "test:demo",
                "mutants": [
                    {"id": "mutation", "kind": "return", "before": "False", "after": "None"}
                ],
            }
        ],
    }
    jobs = schedule(manifest)
    for job in jobs:
        job.update(
            status="valid_candidate",
            grades={
                k: {"valid": True, "detected": 0, "mutations": []} for k in ("initial", "final")
            },
            trace=[{"usage_known": True, "prompt_tokens": 100, "completion_tokens": 50}],
            generation_seconds=1,
        )
    study = {
        "status": "complete",
        "manifest_sha256": "frozen",
        "jobs": jobs,
        "profile": {
            "pricing_reference": {
                "meters": [
                    {"meterName": "gpt Inp global", "retailPrice": 0.5},
                    {"meterName": "gpt Opt global", "retailPrice": 4},
                ]
            }
        },
    }
    study["summary"] = summarize(manifest, study)
    return manifest, study


def test_paired_initial_usage_counted_once_not_twice():
    result = PUBLISHER.report(*fixture())
    assert result["usage"]["initial"]["calls"] == 3
    assert result["usage"]["continuation"]["calls"] == 0
    assert not result["actual_invoice_verified"]


def test_reject_partial_grading():
    manifest, study = fixture()
    del study["jobs"][0]["grades"]["final"]
    with pytest.raises(ValueError, match="complete"):
        PUBLISHER.report(manifest, study)


def test_reject_stale_stored_summary():
    manifest, study = fixture()
    study["summary"]["initial"]["successful"] = 99
    with pytest.raises(ValueError, match="recompute"):
        PUBLISHER.report(manifest, study)


def test_zero_pool_families_are_not_measured():
    result = PUBLISHER.report(*fixture())
    assert result["by_operator"]["initial"]["comparison"]["scheduled"] == 0
    assert result["by_operator"]["initial"]["return"]["scheduled"] == 3
