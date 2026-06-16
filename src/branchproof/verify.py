"""Decides whether a generated test is worth keeping.

The model does not get a say in this. A candidate is kept only if every check
below passes, and the checks run in the order that fails fastest and cheapest.

  1. no production file was touched
  2. the candidate is collectable and passes on the real code
  3. the existing suite is still green
  4. the targeted mutant now fails
  5. repeated original passes AND repeated behavioral mutant failures

Check 2 matters most. A test that fails on the real code has written down the
wrong expected behaviour, and keeping it would be worse than having no test.
"""

from __future__ import annotations

from pathlib import Path

from .models import CheckResult, Mutant, Verdict
from .mutation import write_mutant
from .runner import run_pytest
from .workspace import Workspace, hash_all, resolve_inside


def _unchanged(space: Workspace, before: dict[str, str], checks: list[CheckResult]) -> bool:
    """Executing a candidate must not change source, tests, or configuration."""
    after = hash_all(space.root)
    changed = sorted(
        name for name in before.keys() | after.keys() if before.get(name) != after.get(name)
    )
    if changed:
        checks.append(CheckResult(
            "stayed_in_bounds", False, "test execution changed: " + ", ".join(changed[:5])
        ))
    return not changed


def verify_candidate(
    space: Workspace,
    env: Path,
    test_path: str,
    mutant: Mutant,
    scoped_tests: list[str],
    stability_runs: int = 3,
    test_timeout: int = 120,
    suite_timeout: int = 300,
) -> Verdict:
    checks: list[CheckResult] = []

    changed = space.violations()
    checks.append(
        CheckResult(
            "stayed_in_bounds",
            not changed,
            "" if not changed else "changed: " + ", ".join(changed[:5]),
        )
    )
    if changed:
        return Verdict(False, tuple(checks))

    before = hash_all(space.root)
    on_original = run_pytest(space.root, env, [test_path], timeout=test_timeout)
    if not _unchanged(space, before, checks):
        return Verdict(False, tuple(checks))
    if on_original.collection_error:
        checks.append(CheckResult("candidate_collects", False, on_original.tail(25)))
        return Verdict(False, tuple(checks))
    checks.append(CheckResult("candidate_collects", True))
    checks.append(
        CheckResult(
            "passes_on_original",
            on_original.passed,
            "" if on_original.passed else on_original.tail(25),
        )
    )
    if not on_original.passed:
        return Verdict(False, tuple(checks))

    suite = run_pytest(space.root, env, timeout=suite_timeout)
    if not _unchanged(space, before, checks):
        return Verdict(False, tuple(checks))
    checks.append(
        CheckResult("suite_still_green", suite.passed, "" if suite.passed else suite.tail(25))
    )
    if not suite.passed:
        return Verdict(False, tuple(checks))

    try:
        original_source = write_mutant(space.root, mutant)
    except (OSError, ValueError) as error:
        checks.append(CheckResult(
            "mutant_execution_valid", False, f"could not apply mutant: {error}"
        ))
        return Verdict(False, tuple(checks))
    try:
        mutated = hash_all(space.root)
        # Attribute the kill to this candidate, not another scoped test's failure.
        with_mutant = run_pytest(space.root, env, [test_path], timeout=test_timeout)
        unchanged = _unchanged(space, mutated, checks)
        if not with_mutant.passed and not with_mutant.behavioral_failure:
            checks.append(CheckResult(
                "mutant_execution_valid", False,
                "mutant run was not a valid test failure: " + with_mutant.tail(25),
            ))
            return Verdict(False, tuple(checks))
        killed = with_mutant.behavioral_failure and unchanged
        detail = "" if killed else "the mutant still passes, so the test does not cover it"
        checks.append(
            CheckResult(
                "kills_mutant", killed, detail,
            )
        )
    finally:
        resolve_inside(space.root, mutant.path).write_text(original_source)
    if not killed:
        return Verdict(False, tuple(checks))

    # Check both sides of the differential, not only original-code passes.
    # Repetition is evidence of stability, never proof that a test is non-flaky.
    outcomes = []
    mutant_outcomes = []
    for _ in range(max(1, stability_runs)):
        outcomes.append(run_pytest(space.root, env, [test_path], timeout=test_timeout).passed)
        if not _unchanged(space, before, checks):
            return Verdict(False, tuple(checks))
        original_source = write_mutant(space.root, mutant)
        try:
            mutated = hash_all(space.root)
            repeated = run_pytest(space.root, env, [test_path], timeout=test_timeout)
            unchanged = _unchanged(space, mutated, checks)
            mutant_outcomes.append(repeated.behavioral_failure and unchanged)
        finally:
            resolve_inside(space.root, mutant.path).write_text(original_source)
        if not unchanged:
            return Verdict(False, tuple(checks))
    stable = all(outcomes) and all(mutant_outcomes)
    checks.append(
        CheckResult(
            "stable_across_runs",
            stable,
            "" if stable else f"original passes={outcomes}; behavioral mutant failures="
            f"{mutant_outcomes}",
        )
    )
    return Verdict(stable, tuple(checks))
