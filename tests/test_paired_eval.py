"""Shared candidate identity, feedback boundaries and fixed denominators."""

import pytest
from test_blind_eval import Scripted, edit

from branchproof import paired_eval
from branchproof.agent import AgentAction
from branchproof.blind_eval import grade
from branchproof.workspace import worktree


class Space:
    def __init__(self, root):
        self.root = root
        self.writes = []

    def write(self, path, content):
        self.writes.append((path, content))


TASK = {"id": "demo", "context": "Original function, no mutation"}


def test_valid_first_candidate_is_exact_same_both_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(paired_eval, "original_checks", lambda *a: (True, "passes"))
    provider = Scripted([edit("assert True")])
    result = paired_eval.generate_pair(Space(tmp_path), tmp_path, TASK, provider)
    assert result["initial"] == result["final"]
    assert len(provider.seen) == 1
    assert "Direct control" not in provider.seen[0]["system"]


def test_invalid_initial_receives_only_original_feedback_and_is_preserved(tmp_path, monkeypatch):
    checks = iter([(False, "Original assertion failed: expected 3, got 2"), (True, "passes")])
    monkeypatch.setattr(paired_eval, "original_checks", lambda *a: next(checks))
    provider = Scripted([edit("assert 2 == 3"), edit("assert 2 == 2")])
    result = paired_eval.generate_pair(Space(tmp_path), tmp_path, TASK, provider)
    assert result["initial"]["content"] == "assert 2 == 3"
    assert not result["initial"]["original_valid"]
    assert result["final"]["original_valid"]
    assert provider.seen[1]["messages"][-1]["content"].startswith("Original assertion failed")


def test_no_resample_of_wrong_first_action(tmp_path):
    provider = Scripted([AgentAction(tool="read_file", args={"path": "source.py"})])
    result = paired_eval.generate_pair(Space(tmp_path), tmp_path, TASK, provider)
    assert result["status"] == "invalid_initial_action"
    assert len(provider.seen) == 1


def test_three_submissions_maximum(tmp_path, monkeypatch):
    monkeypatch.setattr(paired_eval, "original_checks", lambda *a: (False, "wrong expectation"))
    result = paired_eval.generate_pair(
        Space(tmp_path), tmp_path, TASK, Scripted([edit("assert False")] * 3)
    )
    assert result["submissions"] == 3
    assert result["status"] == "no_valid_candidate"


def test_reject_hidden_task_data():
    with pytest.raises(ValueError, match="Only public"):
        paired_eval.request({**TASK, "mutants": []})


def test_all_failures_keep_full_schedule_and_mutation_denominator():
    manifest = {
        "cases": [{"id": "demo", "mutants": [{"id": "mutant", "before": "False", "after": "None"}]}]
    }
    jobs = paired_eval.schedule(manifest)
    assert len(jobs) == 3
    result = paired_eval.summarize(manifest, {"jobs": jobs})
    assert result["initial"]["scheduled"] == 3
    assert result["initial"]["hidden_scheduled"] == 3
    assert result["initial"]["false_to_none_scheduled"] == 3
    assert result["final"]["successful"] == 0
    with pytest.raises(ValueError, match="schedule"):
        paired_eval.summarize(manifest, {"jobs": jobs[:2]})


def test_real_execution_repairs_wrong_initial_without_changing_it(sample_repo, env):
    wrong = (
        "from bank import can_withdraw\ndef test_boundary():\n"
        "    assert not can_withdraw(100,100)\n"
    )
    good = "from bank import can_withdraw\ndef test_boundary():\n    assert can_withdraw(100,100)\n"
    provider = Scripted([edit(wrong), edit(good)])
    with worktree(sample_repo) as space:
        result = paired_eval.generate_pair(space, env, TASK, provider)
    assert result["initial"]["content"] == wrong
    assert result["final"]["content"] == good
    assert not result["initial"]["original_valid"]
    assert result["final"]["original_valid"]
    assert "Candidate on original code" in provider.seen[1]["messages"][-1]["content"]
    # Independent replay confirms validity; no hidden mutation enters generation.
    task = {"id": "demo", "mutants": []}
    assert not grade(sample_repo, env, task, wrong)["valid"]
    assert grade(sample_repo, env, task, good)["valid"]
