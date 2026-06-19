"""Real pytest execution on trusted fixtures; no API calls."""

import json
import os
import subprocess

import pytest

from branchproof.agent import AgentAction, ProviderError, ScriptedProvider
from branchproof.sandbox import execution_image, resolve_image
from branchproof.strengthen import CallBudget, strengthen

CONTENT = ("from bank import can_withdraw\n\ndef test_boundary():\n"
           "    assert can_withdraw(100, 100)\n"
           "    assert not can_withdraw(100, 0)\n"
           "    assert not can_withdraw(100, -1)\n")


def test_shared_budget_counts_failed_calls():
    class Broken:
        def next_action(self, messages):
            raise ProviderError("network failed")

    provider = CallBudget(Broken(), 1)
    with pytest.raises(ProviderError, match="network failed"):
        provider.next_action([])
    with pytest.raises(ProviderError, match="budget exhausted"):
        provider.next_action([])
    assert provider.calls == 1


def test_exports_applicable_patch_without_editing_checkout(sample_repo, env, tmp_path):
    provider = ScriptedProvider([AgentAction(tool="edit_test", args={
        "path": "tests/test_boundary.py", "content": CONTENT})])
    directory, report = strengthen(sample_repo, env, "bank.py", provider, tmp_path / "runs",
                                   limit=1, stability_runs=1)
    assert report["accepted"] == 1
    assert report["baseline"]["passed"]
    assert report["eligible_targets"] == 1
    assert report["model_calls"] == 1
    assert report["usage_complete"]
    assert report["trace"][0]["action"]["tool"] == "edit_test"
    assert report["targets"][0]["replay"]["accepted"]
    assert json.loads((directory / "strengthen.json").read_text())["accepted"] == 1
    patch = directory / report["targets"][0]["patch"]
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=sample_repo, check=True)
    assert not (sample_repo / "tests/test_boundary.py").exists()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=sample_repo)


def test_export_preserves_missing_final_newline(sample_repo, env, tmp_path):
    provider = ScriptedProvider([AgentAction(tool="edit_test", args={
        "path": "tests/test_boundary.py", "content": CONTENT.rstrip("\n")})])
    directory, result = strengthen(sample_repo, env, "bank.py", provider, tmp_path / "runs",
                                   limit=1, stability_runs=1)
    assert result["accepted"] == 1
    assert result["targets"][0]["replay"]["accepted"]
    assert (directory / result["targets"][0]["patch"]).exists()


def test_no_survivor_does_not_call_model(sample_repo, env, tmp_path, monkeypatch):
    monkeypatch.setattr("branchproof.strengthen.discover_in_file", lambda *args: [])
    _, result = strengthen(sample_repo, env, "bank.py", ScriptedProvider([]), tmp_path / "runs")
    assert result["eligible_targets"] == result["accepted"] == result["model_calls"] == 0
    assert result["status"] == "complete"


def test_cli_screening_needs_no_provider(sample_repo, tmp_path, monkeypatch, capsys):
    from branchproof.cli import main

    def forbidden(*args):
        raise AssertionError("screening must not create a model client")

    monkeypatch.setattr("branchproof.cli._provider", forbidden)
    assert main(["strengthen", "--repo", str(sample_repo), "--module", "bank.py",
                 "--output", str(tmp_path / "runs"), "--screen-only", "--limit", "1",
                 "--unsafe-local"]) == 0
    assert "no model calls" in capsys.readouterr().out


def test_cli_refuses_missing_docker_without_fallback(sample_repo, tmp_path, monkeypatch):
    from branchproof.cli import main

    def unavailable(image):
        raise ValueError("Sandbox unavailable")

    monkeypatch.setattr("branchproof.cli.resolve_image", unavailable)
    with pytest.raises(SystemExit, match="Sandbox unavailable"):
        main(["strengthen", "--repo", str(sample_repo), "--module", "bank.py",
              "--output", str(tmp_path / "runs"), "--screen-only"])
    assert not (tmp_path / "runs").exists()


def test_dirty_checkout_rejected_before_output_or_model(sample_repo, env, tmp_path):
    (sample_repo / "bank.py").write_text("uncommitted")
    with pytest.raises(ValueError, match="clean committed"):
        strengthen(sample_repo, env, "bank.py", ScriptedProvider([]), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_failed_baseline_persists_evidence_without_model(sample_repo, env, tmp_path):
    (sample_repo / "tests/test_failure.py").write_text("def test_failure():\n    assert False\n")
    subprocess.run(["git", "add", "."], cwd=sample_repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "bad"],
                   cwd=sample_repo, check=True)
    with pytest.raises(ValueError, match="Baseline suite"):
        strengthen(sample_repo, env, "bank.py", ScriptedProvider([]), tmp_path / "runs")
    result = json.loads(next((tmp_path / "runs").glob("*/strengthen.json")).read_text())
    assert result["status"] == "error"
    assert result["model_calls"] == 0
    assert not result["baseline"]["passed"]


def test_limits_and_paths_are_checked_before_generation(sample_repo, env, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        strengthen(sample_repo, env, "bank.py", ScriptedProvider([]), tmp_path, max_calls=0)
    with pytest.raises(ValueError):
        strengthen(sample_repo, env, "../outside.py", ScriptedProvider([]), tmp_path)
    with pytest.raises(ValueError, match="outside the input"):
        strengthen(sample_repo, env, "bank.py", ScriptedProvider([]), sample_repo / "runs")
    with pytest.raises(ValueError, match="production module"):
        strengthen(sample_repo, env, "tests/test_bank.py", ScriptedProvider([]), tmp_path)
    with pytest.raises(ValueError, match="repository root"):
        strengthen(sample_repo / "tests", env, "test_bank.py", ScriptedProvider([]), tmp_path)


def test_failed_independent_replay_does_not_count_as_accepted(
    sample_repo, env, tmp_path, monkeypatch,
):
    from branchproof.models import CheckResult, Verdict

    monkeypatch.setattr("branchproof.strengthen.verify_candidate", lambda *args, **kwargs:
                        Verdict(False, (CheckResult("replay", False, "rejected fixture"),)))
    provider = ScriptedProvider([AgentAction(tool="edit_test", args={
        "path": "tests/test_boundary.py", "content": CONTENT})])
    directory, result = strengthen(sample_repo, env, "bank.py", provider, tmp_path / "runs",
                                   limit=1, stability_runs=1)
    assert result["accepted"] == 0
    row = result["targets"][0]
    assert row["status"] == "error"
    assert not row["replay"]["accepted"]
    assert "patch" not in row
    assert (directory / row["candidate_patch"]).exists()  # retain rejected evidence


@pytest.mark.skipif(not os.getenv("HARDENER_DOCKER_TESTS"), reason="opt-in real Docker test")
def test_strengthen_in_real_sandbox(sample_repo, env, tmp_path):
    image = resolve_image(os.getenv("HARDENER_SANDBOX_IMAGE", "hardener-sandbox:py312"))
    provider = ScriptedProvider([AgentAction(tool="edit_test", args={
        "path": "tests/test_boundary.py", "content": CONTENT})])
    with execution_image(image):
        directory, report = strengthen(sample_repo, env, "bank.py", provider, tmp_path / "runs",
                                       limit=1, stability_runs=1)
    assert report["accepted"] == 1
    assert json.loads((directory / "metadata.json").read_text())["sandbox_image"] == image
