"""Mutation results must measure behavior without damaging existing artifacts."""

import ast

import pytest

from branchproof import context, verify
from branchproof.agent import AgentAction, ProviderError, ScriptedProvider
from branchproof.coverage_map import _test_context, context_files
from branchproof.loop import harden_mutant
from branchproof.mutation import apply_mutant, discover, discover_in_file
from branchproof.runner import RunResult
from branchproof.workspace import worktree

GOOD = """from bank import can_withdraw


def test_boundary():
    assert can_withdraw(100, 100) is True
"""


def boundary_mutant(root):
    return next(
        m for m in discover_in_file(root, "bank.py") if m.before == ">=" and m.after == ">"
    )


@pytest.mark.parametrize("literal", ["0xFF", "0b101", "0o17", "1_000"])
def test_integer_mutation_preserves_actual_source_spelling(literal):
    source = f"def f():\n    return {literal}\n"
    mutant = next(m for m in discover(source, "m.py") if m.kind == "constant")
    assert mutant.before == literal
    changed = apply_mutant(source, mutant)
    values = [n.value for n in ast.walk(ast.parse(changed)) if isinstance(n, ast.Constant)]
    assert values == [int(literal, 0) + 1]


def test_every_mutation_uses_character_spans_after_unicode():
    source = "def f(é, x):\n    return é >= x + 0x10 and True\n"
    mutants = discover(source, "m.py")
    assert {"comparison", "arithmetic", "constant", "boolean", "return"} <= {
        m.kind for m in mutants
    }
    for mutant in mutants:
        changed = apply_mutant(source, mutant)
        ast.parse(changed)
        assert changed.splitlines()[0] == source.splitlines()[0]


def test_context_preserves_pipe_in_parameter_id():
    node = "tests/test_parser.py::test_parse[a|b]"
    assert _test_context(node + "|run") == node
    assert _test_context(node) == node
    assert context_files([node + "|run", node]) == ["tests/test_parser.py"]


def test_context_accepts_node_ids_and_ignores_outside_symlinks(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "test_local.py").write_text("def test_local(): pass\n")
    outside = tmp_path / "test_private.py"
    outside.write_text("PRIVATE_OUTSIDE_CONTENT")
    (root / "test_link.py").symlink_to(outside)
    result = context.existing_tests(
        root, ["test_local.py::test_local|run", "test_link.py", "../test_private.py"]
    )
    assert "def test_local" in result
    assert "PRIVATE_OUTSIDE_CONTENT" not in result


@pytest.mark.parametrize("invalid", [
    RunResult(-1, "", "timed out", timed_out=True),
    RunResult(2, "ERROR collecting tests/test_new.py", ""),
    RunResult(4, "", "usage error"),
    RunResult(1, "1 failed, 1 error", ""),
    RunResult(1, "1 error", ""),
])
def test_invalid_mutant_execution_is_not_a_kill(sample_repo, env, monkeypatch, invalid):
    with worktree(sample_repo) as space:
        mutant = boundary_mutant(space.root)
        original = space.read("bank.py")
        space.write("tests/test_new.py", GOOD)
        results = iter([RunResult(0, "1 passed", ""), RunResult(0, "4 passed", ""), invalid])
        monkeypatch.setattr(verify, "run_pytest", lambda *a, **kw: next(results))
        verdict = verify.verify_candidate(space, env, "tests/test_new.py", mutant, [])
        assert not verdict.accepted
        assert verdict.first_failure.name == "mutant_execution_valid"
        assert "not a valid test failure" in verdict.first_failure.detail
        assert space.read("bank.py") == original


def test_candidate_alone_must_kill_mutant(sample_repo, env, monkeypatch):
    calls = []

    def fake_run(root, environment, targets=None, **kwargs):
        calls.append(targets)
        if len(calls) == 3:
            # An old test failing must not be attributed to the generated candidate.
            if len(targets) > 1:
                return RunResult(1, "1 failed", "")
            return RunResult(0, "1 passed", "")
        return RunResult(0, "1 passed", "")

    monkeypatch.setattr(verify, "run_pytest", fake_run)
    with worktree(sample_repo) as space:
        space.write("tests/test_new.py", GOOD)
        verdict = verify.verify_candidate(
            space, env, "tests/test_new.py", boundary_mutant(space.root), ["tests/test_bank.py"]
        )
    assert not verdict.accepted
    assert calls[2] == ["tests/test_new.py"]


def test_candidate_execution_cannot_change_production(sample_repo, env):
    with worktree(sample_repo) as space:
        mutant = boundary_mutant(space.root)
        space.write("tests/test_new.py", """from pathlib import Path


def test_changes_source():
    Path('bank.py').write_text('def can_withdraw(*args): return True\\n')
    assert True
""")
        verdict = verify.verify_candidate(space, env, "tests/test_new.py", mutant, [])
    assert not verdict.accepted
    assert verdict.first_failure.name == "stayed_in_bounds"
    assert "test execution changed: bank.py" in verdict.reason


def test_loop_preserves_existing_tests_and_keeps_success_content(sample_repo, env):
    provider = ScriptedProvider([
        AgentAction(tool="edit_test", args={"path": "tests/test_bank.py", "content": GOOD}),
        AgentAction(tool="edit_test", args={"path": "tests/test_new.py", "content": GOOD}),
    ])
    with worktree(sample_repo) as space:
        original = space.read("tests/test_bank.py")
        run = harden_mutant(
            space, env, boundary_mutant(space.root), provider, ["tests/test_bank.py"],
            stability_runs=1,
        )
        assert space.read("tests/test_bank.py") == original
        assert run.status == "success", run.attempts
    assert run.accepted_test_content == GOOD
    assert run.to_dict()["accepted_test_content"] == GOOD


def test_refused_write_cannot_be_verified_as_candidate(sample_repo, env, monkeypatch):
    def cannot_verify(*args, **kwargs):
        pytest.fail("A refused write must never reach the verifier")

    monkeypatch.setattr("branchproof.loop.verify_candidate", cannot_verify)
    provider = ScriptedProvider([
        AgentAction(tool="edit_test", args={"path": "tests/test_bank.py", "content": GOOD}),
    ])
    with worktree(sample_repo) as space:
        original = space.read("tests/test_bank.py")
        run = harden_mutant(space, env, boundary_mutant(space.root), provider, [])
        assert space.read("tests/test_bank.py") == original
        assert run.status == "failed"


def test_provider_error_is_recorded_without_aborting_batch(sample_repo, env):
    class BrokenProvider:
        def next_action(self, messages):
            raise ProviderError("The model returned an invalid action.")

    with worktree(sample_repo) as space:
        run = harden_mutant(space, env, boundary_mutant(space.root), BrokenProvider(), [])
    assert run.status == "error"
    assert run.failure_class == "provider error"
    assert "invalid action" in run.attempts[0].reason
