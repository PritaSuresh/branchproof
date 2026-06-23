"""Execution-based evaluation calibration; every provider here is API-free."""

import copy
import json
import os
import subprocess

import pytest

from branchproof.agent import AgentAction, ProviderError
from branchproof.cli import main
from branchproof.sandbox import execution_image, resolve_image
from branchproof.strengthen import CallBudget
from branchproof.test_study import (
    StudyBudget,
    combine,
    digest,
    prepare,
    run,
    score,
    summary,
    validate,
    validate_jobs,
)

SOURCE = "def positive(x):\n    return x > 0\n\ndef large(x):\n    return x >= 10\n"
EXISTING = ("from bank import positive, large\n\ndef test_ordinary():\n"
            "    assert positive(1)\n    assert not positive(-1)\n"
            "    assert large(20)\n    assert not large(0)\n")
ADDED = ("from bank import positive, large\n\ndef test_boundaries():\n"
         "    assert not positive(0)\n    assert large(10)\n")


class Script:
    def next_action(self, messages):
        # Additional grading outcomes must never become generation feedback.
        assert not any("additional_mutants" in m["content"] for m in messages)
        return AgentAction(tool="edit_test", args={
            "path": "tests/test_generated.py", "content": ADDED}), 10, 5


@pytest.fixture
def prepared(sample_repo, env, tmp_path):
    (sample_repo / "bank.py").write_text(SOURCE)
    (sample_repo / "tests/test_bank.py").write_text(EXISTING)
    subprocess.run(["git", "add", "."], cwd=sample_repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "study"],
                   cwd=sample_repo, check=True)
    repos = {"fixture": sample_repo}
    directory, manifest = prepare(repos, [("fixture", "bank.py")], env, tmp_path / "prepare",
                                  candidate_limit=20, per_module=1, additional_limit=1)
    return repos, directory, manifest


def test_preparation_freezes_disjoint_function_pools(prepared):
    repos, directory, manifest = prepared
    assert len(manifest["cases"]) == 1
    entry = manifest["modules"][0]
    assert entry["shown_functions"] == ["1:positive"]
    assert len(entry["additional_mutants"]) == 1
    assert entry["coverage"]["available"]
    assert entry["coverage"]["num_statements"] > 0
    assert json.loads((directory / "manifest.json").read_text())["sha256"] == manifest["sha256"]
    validate(manifest, repos)


def test_combining_preserves_every_target_and_rejects_duplicate_modules(prepared, tmp_path):
    repos, directory, original = prepared
    source = directory / "manifest.json"
    _, merged = combine([source], repos, tmp_path / "combined")
    assert merged["cases"] == original["cases"]
    assert merged["modules"] == original["modules"]
    assert merged["source_preparations"][0]["sha256"] == original["sha256"]
    validate(merged, repos)
    with pytest.raises(ValueError, match="duplicate modules"):
        combine([source, source], repos, tmp_path / "bad")


def test_paired_generation_and_independent_grading(prepared, env, tmp_path):
    repos, _, manifest = prepared
    directory, result = run(manifest, repos, env, Script(), tmp_path / "runs",
                            max_total_calls=20, max_total_tokens=100000,
                            completion_cap=100, provider_metadata={"kind": "scripted"})
    assert result["status"] == "complete"
    assert result["model_calls"] == 2
    assert result["prompt_tokens"] == 20
    assert result["completion_tokens"] == 10
    assert all(r["accepted"] for r in result["rows"])
    grades = score(manifest, repos, env, directory)
    for grade in grades["grades"]:
        assert grade["accepted"]
        assert grade["additional"][0]["outcome"] == "killed"
        assert grade["coverage_after"]["available"]
    assert grades["summary"]["initial"]["eligible"] == 1
    assert grades["summary"]["feedback"]["accepted"] == 1
    # Modifying a valid exported file must not silently replace the measured patch.
    row = result["rows"][0]
    artifact = directory / row["artifact"]
    target = json.loads((artifact / "strengthen.json").read_text())["targets"][0]
    patch = artifact / target["patch"]
    patch.write_text(patch.read_text() + "\n")
    audited = score(manifest, repos, env, directory)
    changed = next(g for g in audited["grades"] if g["arm"] == row["arm"])
    assert not changed["accepted"]
    assert "differs from generation" in changed["error"]
    assert grades["grading_directory"] != audited["grading_directory"]


def test_budget_keeps_unattempted_jobs_in_denominators(prepared, env, tmp_path):
    repos, _, manifest = prepared
    _, result = run(manifest, repos, env, Script(), tmp_path / "runs", max_total_calls=1,
                    max_total_tokens=100000, completion_cap=100, provider_metadata={})
    assert result["status"] == "budget_stopped"
    assert result["model_calls"] == 1
    assert len([r for r in result["rows"] if r["status"] == "pending"]) == 1
    assert all(result["summary"][arm]["eligible"] == 1 for arm in ("initial", "feedback"))


def test_single_arm_schedule_and_independent_grading(prepared, env, tmp_path):
    repos, _, manifest = prepared
    directory, result = run(
        manifest, repos, env, Script(), tmp_path / "single", max_total_calls=5,
        max_total_tokens=100000, completion_cap=100, provider_metadata={},
        arms=("initial",), per_job_tokens=100000)
    assert result["status"] == "complete"
    assert len(result["rows"]) == 1
    assert result["rows"][0]["accepted"]
    assert result["limits"]["arms"] == ["initial"]
    grades = score(manifest, repos, env, directory)
    assert grades["summary"]["initial"]["accepted"] == 1
    assert grades["summary"]["feedback"]["eligible"] == 0


def test_per_job_token_exhaustion_does_not_starve_next_job(prepared, env, tmp_path):
    repos, _, manifest = prepared
    _, result = run(
        manifest, repos, env, Script(), tmp_path / "reserved", max_total_calls=10,
        max_total_tokens=10000, completion_cap=100, provider_metadata={}, per_job_tokens=5000)
    assert result["status"] == "complete"
    assert len(result["rows"]) == 2
    assert all(r["status"] == "complete" and r["job_token_budget_stopped"]
               and not r["accepted"] for r in result["rows"])
    assert result["model_calls"] == 0


def test_per_job_reservations_must_fit_global_allowance(prepared, env, tmp_path):
    repos, _, manifest = prepared
    with pytest.raises(ValueError, match="reserve every scheduled job"):
        run(manifest, repos, env, Script(), tmp_path / "bad-reservation", max_total_calls=10,
            max_total_tokens=10000, completion_cap=100, provider_metadata={},
            per_job_tokens=6000)


def test_artifact_error_does_not_hide_model_spend(prepared, env, tmp_path, monkeypatch):
    repos, _, manifest = prepared

    def failed_export(repo, env, module, provider, *args, **kwargs):
        provider.next_action([{"role": "user", "content": "fixture"}])
        raise OSError("artifact export failed")

    monkeypatch.setattr("branchproof.test_study.strengthen", failed_export)
    _, result = run(manifest, repos, env, Script(), tmp_path / "runs", max_total_calls=20,
                    max_total_tokens=100000, completion_cap=100, provider_metadata={})
    assert result["status"] == "infrastructure_stopped"
    assert result["model_calls"] == 1
    assert result["prompt_tokens"] == 10
    assert sum(r.get("prompt_tokens", 0) for r in result["rows"]) == 10
    assert sorted(r["status"] for r in result["rows"]) == ["error", "pending"]


def test_source_inspection_is_not_credited_as_behavioral_test(prepared, env, tmp_path):
    class SourceInspection:
        def next_action(self, messages):
            return AgentAction(tool="edit_test", args={"path": "tests/test_generated.py",
                "content": "import bank\nfrom pathlib import Path\n\ndef test_source():\n"
                           "    assert 'x > 0' in Path(bank.__file__).read_text()\n"}), 10, 5

    repos, _, manifest = prepared
    directory, result = run(manifest, repos, env, SourceInspection(), tmp_path / "runs",
                            max_total_calls=20, max_total_tokens=100000,
                            completion_cap=100, provider_metadata={})
    assert all(r["accepted"] for r in result["rows"])  # original pass / mutant assertion fail
    grades = score(manifest, repos, env, directory)
    assert all(not g["accepted"] for g in grades["grades"])
    assert all("target execution" in g["reason"] for g in grades["grades"])


def test_unknown_provider_usage_stops_further_spend(prepared, env, tmp_path):
    class Lost:
        def next_action(self, messages):
            raise ProviderError("response lost", usage_known=False)

    repos, _, manifest = prepared
    _, result = run(manifest, repos, env, Lost(), tmp_path / "runs", max_total_calls=20,
                    max_total_tokens=100000, completion_cap=100, provider_metadata={})
    assert result["model_calls"] == 1
    assert not result["usage_complete"]
    assert result["status"] == "budget_stopped"


def test_token_reservation_stops_before_provider_call():
    provider = StudyBudget(Script(), 20, 1, 100)
    with pytest.raises(ProviderError, match="reservation"):
        provider.next_action([])
    assert provider.calls == 0
    assert provider.blocked


def test_known_provider_error_usage_is_not_discarded():
    class Invalid:
        def next_action(self, messages):
            raise ProviderError("invalid JSON", prompt_tokens=12, completion_tokens=3,
                                usage_known=True)

    provider = CallBudget(Invalid(), 1)
    with pytest.raises(ProviderError):
        provider.next_action([])
    assert provider.trace[0]["usage_known"]
    assert provider.trace[0]["prompt_tokens"] == 12


def test_manifest_and_worktree_changes_are_rejected(prepared):
    repos, _, manifest = prepared
    broken = copy.deepcopy(manifest)
    broken["cases"][0]["mutant"]["after"] = "bad"
    with pytest.raises(ValueError, match="fingerprint"):
        validate(broken, repos)
    broken["sha256"] = digest({k: v for k, v in broken.items() if k != "sha256"})
    with pytest.raises(ValueError, match="source mutation"):
        validate(broken, repos)
    broken = copy.deepcopy(manifest)
    broken["modules"][0]["additional_mutants"] = [broken["cases"][0]["mutant"]]
    broken["sha256"] = digest({k: v for k, v in broken.items() if k != "sha256"})
    with pytest.raises(ValueError, match="overlap"):
        validate(broken, repos)
    path = repos["fixture"] / "bank.py"
    path.write_text(SOURCE + "# dirty\n")
    with pytest.raises(ValueError, match="clean state"):
        validate(manifest, repos)


def test_cli_dry_run_never_creates_provider(prepared, tmp_path, monkeypatch, capsys):
    repos, directory, _ = prepared

    def forbidden(*args):
        raise AssertionError("dry run must not create a model client")

    monkeypatch.setattr("branchproof.cli._provider", forbidden)
    assert main(["test-study", "--repo-map", f"fixture={repos['fixture']}",
                 "--manifest", str(directory / "manifest.json"),
                 "--output", str(tmp_path / "unused"), "--unsafe-local"]) == 0
    assert '"paid_calls": false' in capsys.readouterr().out
    assert not (tmp_path / "unused").exists()


def test_repeats_do_not_inflate_primary_score():
    rows = [{"arm": "initial", "repeat": repeat, "status": "complete", "accepted": repeat > 0,
             "repository": "fixture", "module": "bank.py"} for repeat in range(3)]
    result = summary(rows)["initial"]
    assert result["eligible"] == 1
    assert result["accepted"] == 0
    assert result["repeat_accepted"] == 2


def test_missing_or_duplicate_result_rows_cannot_shrink_frozen_denominator():
    manifest = {"cases": [{"id": "target"}], "repeat_cases": ["target"]}
    record = {"limits": {"extra_repeats": 0}, "rows": [
        {"id": "target", "arm": arm, "repeat": 0} for arm in ("initial", "feedback")]}
    validate_jobs(manifest, record)
    broken = copy.deepcopy(record)
    broken["rows"].pop()
    with pytest.raises(ValueError, match="complete frozen"):
        validate_jobs(manifest, broken)
    broken = copy.deepcopy(record)
    broken["rows"].append(broken["rows"][0])
    with pytest.raises(ValueError, match="complete frozen"):
        validate_jobs(manifest, broken)


@pytest.mark.skipif(not os.getenv("HARDENER_DOCKER_TESTS"), reason="opt-in real Docker study")
def test_docker_coverage_report_roundtrip(sample_repo, env):
    from branchproof.test_study import coverage

    image = resolve_image(os.getenv("HARDENER_SANDBOX_IMAGE", "hardener-sandbox:tests-v1"))
    with execution_image(image):
        report = coverage(sample_repo, env, "bank.py")
    assert report["available"], report
    assert report["covered_lines"] > 0
    assert report["num_branches"] > 0


@pytest.mark.skipif(not os.getenv("HARDENER_DOCKER_TESTS"), reason="opt-in paired Docker study")
def test_complete_paired_study_inside_docker(prepared, env, tmp_path):
    repos, _, _ = prepared
    image = resolve_image(os.getenv("HARDENER_SANDBOX_IMAGE", "hardener-sandbox:tests-v1"))
    with execution_image(image):
        _, manifest = prepare(repos, [("fixture", "bank.py")], env, tmp_path / "docker-prepare",
                              candidate_limit=20, per_module=1, additional_limit=1)
        directory, result = run(manifest, repos, env, Script(), tmp_path / "docker-study",
                                max_total_calls=20, max_total_tokens=100000,
                                completion_cap=100, provider_metadata={"kind": "scripted"})
        graded = score(manifest, repos, env, directory)
    assert result["status"] == "complete"
    for arm in ("initial", "feedback"):
        assert graded["summary"][arm]["accepted"] == 1
        assert graded["summary"][arm]["additional_unique_killed"] == 1
