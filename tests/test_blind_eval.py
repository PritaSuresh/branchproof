"""The scored mutation pool must never enter generation or feedback."""

import ast
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from branchproof import blind_eval
from branchproof.agent import AgentAction, ProviderError
from branchproof.workspace import worktree


class Scripted:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.seen = []

    def next_action(self, messages, system=None):
        self.seen.append({"messages": [dict(m) for m in messages], "system": system})
        return next(self.actions), 0, 0


def edit(content, path="tests/test_branchproof_demo.py"):
    return AgentAction(tool="edit_test", args={"path": path, "content": content})


GOOD = "from bank import can_withdraw\ndef test_equal():\n    assert can_withdraw(100,100)\n"
WRONG = "from bank import can_withdraw\ndef test_equal():\n    assert not can_withdraw(100,100)\n"


def test_context_has_original_source_not_mutant(sample_repo):
    text = blind_eval.initial_context(
        sample_repo, "bank.py", "can_withdraw", ["tests/test_bank.py"]
    )
    assert "return balance >= amount" in text
    assert "The change:" not in text
    assert " -> " not in text
    assert "tests/test_bank.py" in text


def test_oversized_context_is_rejected_not_truncated(sample_repo):
    (sample_repo / "large.py").write_text("# " + "x" * 40000)
    with pytest.raises(ValueError, match="context exceeds"):
        blind_eval.initial_context(sample_repo, "large.py", "target", [])


def test_mutations_in_defaults_and_docstrings_not_in_body_pool():
    source = 'def target(value=1):\n    """Example 123."""\n    return value + 2\n'
    node = ast.parse(source).body[0]
    assert all(m.line == 3 for m in blind_eval.body_mutants(source, "target.py", node))


def test_private_functions_are_not_selected():
    nodes = list(blind_eval.function_nodes("def _private():\n    return 1\n"))
    assert nodes == []


def test_generation_refuses_hidden_task_fields(sample_repo, env):
    with worktree(sample_repo) as space:
        with pytest.raises(ValueError, match="only public"):
            blind_eval.generate(
                space, env, {"id": "demo", "context": "x", "mutants": []}, Scripted([]), "agent"
            )


def test_generation_never_calls_mutant_verifier(sample_repo, env, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Hidden grading called during generation")

    monkeypatch.setattr(blind_eval, "verify_candidate", forbidden)
    provider = Scripted([edit(GOOD)])
    with worktree(sample_repo) as space:
        result = blind_eval.generate(
            space, env, {"id": "demo", "context": "original code"}, provider, "agent"
        )
    assert result["status"] == "valid_candidate"
    assert len(provider.seen) == 1


def test_agent_repairs_invalid_test_with_only_original_feedback(sample_repo, env):
    provider = Scripted([edit(WRONG), edit(GOOD)])
    with worktree(sample_repo) as space:
        result = blind_eval.generate(
            space, env, {"id": "demo", "context": "original code"}, provider, "agent"
        )
    assert result["status"] == "valid_candidate"
    assert result["submissions"] == 2
    feedback = provider.seen[1]["messages"][-1]["content"]
    assert "Candidate on original code" in feedback
    assert "mutant" not in feedback.lower()


def test_direct_cannot_retry_wrong_test(sample_repo, env):
    provider = Scripted([edit(WRONG), edit(GOOD)])
    with worktree(sample_repo) as space:
        result = blind_eval.generate(
            space, env, {"id": "demo", "context": "original code"}, provider, "direct"
        )
    assert result["status"] == "no_valid_candidate"
    assert len(provider.seen) == 1


def test_direct_and_agent_share_same_initial_user_context(sample_repo, env):
    providers = []
    for arm in ("direct", "agent"):
        provider = Scripted([edit(GOOD)])
        with worktree(sample_repo) as space:
            blind_eval.generate(
                space, env, {"id": "demo", "context": "same original source"}, provider, arm
            )
        providers.append(provider)
    assert providers[0].seen[0]["messages"] == providers[1].seen[0]["messages"]


def test_cannot_replace_upstream_tests(sample_repo, env):
    provider = Scripted([edit(GOOD, "tests/test_bank.py")])
    original = (sample_repo / "tests/test_bank.py").read_text()
    with worktree(sample_repo) as space:
        result = blind_eval.generate(space, env, {"id": "demo", "context": "x"}, provider, "direct")
        assert (space.root / "tests/test_bank.py").read_text() == original
    assert result["status"] == "no_valid_candidate"


def test_schedule_is_paired_and_repeats_are_preselected():
    manifest = {"cases": [{"id": "a"}, {"id": "b"}], "repeat_cases": ["b"]}
    jobs = blind_eval.schedule(manifest)
    assert jobs == blind_eval.schedule(manifest)
    assert len(jobs) == 6
    assert {(j["id"], j["arm"]) for j in jobs if not j["repeat"]} == {
        (name, arm) for name in ("a", "b") for arm in ("direct", "agent")
    }


def test_empty_or_incomplete_manifest_cannot_run():
    with pytest.raises((ValueError, KeyError)):
        blind_eval.validate_manifest({"sha256": "wrong"}, Path("/unused"))


def test_failed_jobs_remain_in_detection_denominator():
    manifest = {"cases": [{"id": "a", "mutants": [{}, {}]}], "repeat_cases": []}
    jobs = blind_eval.schedule(manifest)
    jobs[0]["status"] = "token_budget_error"
    record = blind_eval.summarize(manifest, {"jobs": jobs})
    for arm in ("direct", "agent"):
        assert record[arm]["scheduled_targets"] == 1
        assert record[arm]["hidden_mutations_scheduled"] == 2
        assert record[arm]["hidden_mutations_detected"] == 0


def test_missing_job_cannot_shrink_published_denominator():
    manifest = {"cases": [{"id": "a", "mutants": [{}]}], "repeat_cases": []}
    with pytest.raises(ValueError, match="denominators"):
        blind_eval.summarize(manifest, {"jobs": blind_eval.schedule(manifest)[:1]})


def test_operator_breakdown_retains_failed_jobs_and_separates_repeats():
    manifest = {
        "cases": [
            {"id": "a", "mutants": [{"id": "r", "kind": "return"}, {"id": "c", "kind": "constant"}]}
        ],
        "repeat_cases": ["a"],
    }
    jobs = blind_eval.schedule(manifest)
    for job in jobs:
        job["status"] = "complete"
        if job["arm"] == "agent" or job["repeat"]:
            job["grade"] = {
                "valid": True,
                "detected": 1,
                "mutations": [{"id": "r", "detected": True}, {"id": "c", "detected": False}],
            }
    summary = blind_eval.summarize(manifest, {"jobs": jobs})
    direct = summary["direct"]["primary_by_operator"]
    agent = summary["agent"]["primary_by_operator"]
    assert direct["return"] == {"scheduled": 1, "measured": 0, "detected": 0, "detection_rate": 0.0}
    assert agent["return"] == {"scheduled": 1, "measured": 1, "detected": 1, "detection_rate": 1.0}
    assert agent["constant"]["detected"] == 0
    assert agent["comparison"]["scheduled"] == 0
    assert agent["comparison"]["detection_rate"] is None


def test_operator_breakdown_rejects_duplicate_grading_rows():
    manifest = {
        "cases": [{"id": "a", "mutants": [{"id": "r", "kind": "return"}]}],
        "repeat_cases": [],
    }
    jobs = blind_eval.schedule(manifest)
    jobs[0]["grade"] = {
        "valid": True,
        "mutations": [{"id": "r", "detected": True}, {"id": "r", "detected": True}],
    }
    with pytest.raises(ValueError, match="duplicate"):
        blind_eval.summarize(manifest, {"jobs": jobs})


def test_unknown_usage_stops_pending_jobs_without_automatic_retry(
    sample_repo, env, tmp_path, monkeypatch
):
    class UnknownUsage:
        calls = 0

        def next_action(self, messages, system=None):
            self.calls += 1
            raise ProviderError("unknown usage", usage_known=False)

    monkeypatch.setattr(blind_eval, "validate_manifest", lambda *args: None)
    manifest = {
        "sha256": "frozen",
        "cases": [
            {"id": "a", "repository": sample_repo.name, "context": "original code", "mutants": [{}]}
        ],
        "repeat_cases": [],
    }
    provider = UnknownUsage()
    _, record = blind_eval.run(
        manifest,
        sample_repo.parent,
        env,
        provider,
        tmp_path / "study",
        {"reviewed_manifest_sha256": "frozen"},
    )
    assert provider.calls == 1
    assert record["status"] == "stopped_no_automatic_resume"
    assert sum(j["status"] == "pending" for j in record["jobs"]) == 1
    assert sum(a["scheduled_targets"] for a in record["summary"].values()) == 2


def test_complete_scripted_run_exports_applicable_fresh_verified_patches(
    sample_repo, env, tmp_path, monkeypatch
):
    from branchproof.mutation import discover

    monkeypatch.setattr(blind_eval, "validate_manifest", lambda *args: None)
    mutant = next(
        m
        for m in discover((sample_repo / "bank.py").read_text(), "bank.py")
        if m.before == ">=" and m.after == ">"
    )
    manifest = {
        "sha256": "scripted-fixture-only",
        "repeat_cases": [],
        "cases": [
            {
                "id": "demo",
                "repository": sample_repo.name,
                "context": "original code",
                "mutants": [asdict(mutant)],
            }
        ],
    }
    directory, result = blind_eval.run(
        manifest,
        sample_repo.parent,
        env,
        Scripted([edit(GOOD), edit(GOOD)]),
        tmp_path / "scripted-study",
        {"reviewed_manifest_sha256": "scripted-fixture-only"},
    )
    assert result["status"] == "generation_complete"
    assert all(row["grade"]["valid"] for row in result["jobs"])
    assert all(row["successful_patches"] == 1 for row in result["summary"].values())
    patches = list(directory.glob("patch-*.diff"))
    assert len(patches) == 2
    for patch in patches:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=sample_repo, check=True)
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=sample_repo)
