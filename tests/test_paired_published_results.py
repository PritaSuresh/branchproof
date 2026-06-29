"""Published paired results retain exact shared candidates and all scheduled failures."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paired_publisher_check", ROOT / "scripts/summarize_evaluation.py"
)
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


def test_published_paired_metrics_recompute_without_runtime_files():
    result = json.loads((ROOT / "benchmarks/results.json").read_text())
    manifest = json.loads((ROOT / "benchmarks/dataset.json").read_text())
    recomputed = PUBLISHER.report(manifest, result)
    for key in ("summary", "by_operator", "usage"):
        assert recomputed[key] == result[key]
    assert len(result["jobs"]) == 30
    assert sum(j["status"] == "invalid_initial_action" for j in result["jobs"]) == 3
    for job in result["jobs"]:
        generated = job.get("generation", {})
        initial = generated.get("initial")
        if initial and initial["original_valid"]:
            assert generated["final"] == initial
            assert job["grades"]["initial"] == job["grades"]["final"]
    assert result["summary"]["rescued_valid"] == 6
    assert result["summary"]["rescued_successful"] == 4
