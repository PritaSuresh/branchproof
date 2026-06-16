"""The verifier is what stops a bad test being kept, so each rejection path is checked."""

import pytest

from branchproof.mutation import discover_in_file
from branchproof.verify import verify_candidate
from branchproof.workspace import worktree

SCOPED = ["tests/test_bank.py"]

GOOD = """from bank import can_withdraw


def test_exact_balance_is_allowed():
    assert can_withdraw(100, 100) is True
"""

WRONG_EXPECTATION = """from bank import can_withdraw


def test_exact_balance_is_refused():
    assert can_withdraw(100, 100) is False
"""

MISSES_THE_BOUNDARY = """from bank import can_withdraw


def test_far_below_balance():
    assert can_withdraw(100, 1) is True
"""

BAD_IMPORT = """from bank import withdraw_money


def test_something():
    assert withdraw_money(1, 1)
"""


def boundary_mutant(root):
    return next(
        m for m in discover_in_file(root, "bank.py")
        if m.before == ">=" and m.after == ">"
    )


@pytest.fixture
def space_and_mutant(sample_repo):
    with worktree(sample_repo) as space:
        yield space, boundary_mutant(space.root)


def run(space, env, content, mutant, path="tests/test_new.py"):
    space.write(path, content)
    return verify_candidate(space, env, path, mutant, SCOPED, stability_runs=2)


def test_a_test_that_covers_the_boundary_is_accepted(space_and_mutant, env):
    space, mutant = space_and_mutant
    verdict = run(space, env, GOOD, mutant)
    assert verdict.accepted, verdict.reason
    assert [c.name for c in verdict.checks if c.passed] == [
        "stayed_in_bounds",
        "candidate_collects",
        "passes_on_original",
        "suite_still_green",
        "kills_mutant",
        "stable_across_runs",
    ]


def test_a_test_that_fails_on_the_real_code_is_rejected(space_and_mutant, env):
    space, mutant = space_and_mutant
    verdict = run(space, env, WRONG_EXPECTATION, mutant)
    assert not verdict.accepted
    assert verdict.first_failure.name == "passes_on_original"


def test_a_test_that_misses_the_boundary_is_rejected(space_and_mutant, env):
    space, mutant = space_and_mutant
    verdict = run(space, env, MISSES_THE_BOUNDARY, mutant)
    assert not verdict.accepted
    assert verdict.first_failure.name == "kills_mutant"


def test_a_test_that_cannot_be_imported_is_rejected(space_and_mutant, env):
    space, mutant = space_and_mutant
    verdict = run(space, env, BAD_IMPORT, mutant)
    assert not verdict.accepted
    assert verdict.first_failure.name == "candidate_collects"


def test_editing_the_source_is_rejected_before_anything_runs(space_and_mutant, env):
    space, mutant = space_and_mutant
    (space.root / "bank.py").write_text(
        "def can_withdraw(balance, amount):\n    return True\n\n\ndef fee(a):\n    return a * 2\n"
    )
    space.write("tests/test_new.py", GOOD)
    verdict = verify_candidate(space, env, "tests/test_new.py", mutant, SCOPED, stability_runs=2)
    assert not verdict.accepted
    assert verdict.first_failure.name == "stayed_in_bounds"


def test_the_source_is_put_back_after_the_mutant_run(space_and_mutant, env):
    space, mutant = space_and_mutant
    before = (space.root / "bank.py").read_text()
    run(space, env, GOOD, mutant)
    assert (space.root / "bank.py").read_text() == before


def test_a_flaky_mutant_kill_is_rejected_and_source_restored(space_and_mutant, env, monkeypatch):
    from branchproof.runner import RunResult

    space, mutant = space_and_mutant
    original = (space.root / "bank.py").read_text()
    mutant_runs = 0

    def execute(root, *args, **kwargs):
        nonlocal mutant_runs
        mutated = (root / "bank.py").read_text() != original
        if mutated:
            mutant_runs += 1
        # The initial kill is real, but every repeated mutant run passes.
        if mutated and mutant_runs == 1:
            return RunResult(1, "", "", test_counts={"failed": 1, "passed": 0})
        return RunResult(0, "", "", test_counts={"passed": 1})

    monkeypatch.setattr("branchproof.verify.run_pytest", execute)
    verdict = run(space, env, GOOD, mutant)
    assert not verdict.accepted
    assert verdict.first_failure.name == "stable_across_runs"
    assert mutant_runs == 3
    assert (space.root / "bank.py").read_text() == original


def test_repeated_mutant_exception_still_restores_source(space_and_mutant, env, monkeypatch):
    from branchproof.runner import RunResult

    space, mutant = space_and_mutant
    original = (space.root / "bank.py").read_text()
    mutant_runs = 0

    def execute(root, *args, **kwargs):
        nonlocal mutant_runs
        if (root / "bank.py").read_text() != original:
            mutant_runs += 1
            if mutant_runs == 2:
                raise RuntimeError("infrastructure lost during repeated kill")
            return RunResult(1, "", "", test_counts={"failed": 1})
        return RunResult(0, "", "", test_counts={"passed": 1})

    monkeypatch.setattr("branchproof.verify.run_pytest", execute)
    with pytest.raises(RuntimeError, match="infrastructure lost"):
        run(space, env, GOOD, mutant)
    assert (space.root / "bank.py").read_text() == original
