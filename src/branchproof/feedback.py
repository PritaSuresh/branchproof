"""Turns a rejection into something the model can act on.

Telling it "rejected" is useless. Telling it the import that failed, or that the
test passed on both versions, gives it something specific to change.
"""

from __future__ import annotations

import re

from .models import Verdict

ASSERTION = re.compile(r"^E\s+(.*)$", re.MULTILINE)


def _assertion_lines(output: str, limit: int = 8) -> str:
    hits = ASSERTION.findall(output)
    return "\n".join(hits[:limit]) if hits else ""


def describe(verdict: Verdict, mutant_label: str) -> str:
    if verdict.accepted:
        return "Accepted. The test passes on the real code and fails on the mutant."

    failure = verdict.first_failure
    if failure is None:
        return "Rejected, but no failing check was recorded."

    if failure.name == "stayed_in_bounds":
        return (
            "Rejected. A file outside the test directory was modified. "
            "Only test files may change. Undo the source edit and put the behaviour "
            "in the test instead.\n"
            f"{failure.detail}"
        )

    if failure.name == "candidate_collects":
        assertions = _assertion_lines(failure.detail)
        return (
            "The test file could not be collected, so nothing ran. This is usually a bad "
            "import or a syntax error. Check what the module actually exports.\n\n"
            f"{assertions or failure.detail}"
        )

    if failure.name == "passes_on_original":
        assertions = _assertion_lines(failure.detail)
        return (
            "The test failed on the real, unmodified code. That means the expected value "
            "written in the test is wrong, not the code. Work out what the function "
            "actually returns and assert that.\n\n"
            f"{assertions or failure.detail}"
        )

    if failure.name == "suite_still_green":
        return (
            "The new test passes, but it broke tests that were passing before. Something "
            "in it leaks into other tests, such as shared state, a fixture, or a file it "
            "writes.\n\n"
            f"{failure.detail}"
        )

    if failure.name == "kills_mutant":
        return (
            f"The test passes on the real code and also passes with the change applied "
            f"({mutant_label}). The assertion does not distinguish the two versions. "
            "Check which implementation actually runs, including subclass overrides and "
            "explicit arguments that bypass changed defaults. Pick inputs where the "
            "two versions give observably different answers, then assert the real one."
        )

    if failure.name == "stable_across_runs":
        return (
            "The test gave different results: original passes or behavioral mutant "
            "failures were not stable. "
            "Remove whatever varies, "
            "such as randomness, the clock, dictionary ordering, or a temporary file."
        )

    return f"Rejected: {verdict.reason}"
