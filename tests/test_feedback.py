"""A rejection is only useful if it says what to change."""

from branchproof.feedback import describe
from branchproof.models import CheckResult, Verdict


def verdict(name, detail=""):
    return Verdict(False, (CheckResult(name, False, detail),))


def test_a_wrong_expectation_is_named_as_such():
    text = describe(verdict("passes_on_original", "E  assert 5 == 6"), "m.py:1 > -> >=")
    assert "failed on the real" in text
    assert "assert 5 == 6" in text


def test_a_useless_test_says_the_mutant_survived():
    text = describe(verdict("kills_mutant"), "m.py:4 >= -> >")
    assert "m.py:4 >= -> >" in text
    assert "different answers" in text


def test_an_import_error_is_passed_through():
    detail = "E   ImportError: cannot import name 'withdraw' from 'bank'"
    text = describe(verdict("candidate_collects", detail), "m.py:1 > -> >=")
    assert "could not be collected" in text
    assert "cannot import name" in text


def test_editing_source_is_explained():
    text = describe(verdict("stayed_in_bounds", "changed: bank.py"), "m.py:1 > -> >=")
    assert "Only test files may change" in text
    assert "bank.py" in text


def test_an_unstable_test_is_explained():
    text = describe(verdict("stable_across_runs"), "m.py:1 > -> >=")
    assert "different results" in text


def test_acceptance_says_so():
    text = describe(Verdict(True, (CheckResult("kills_mutant", True),)), "m.py:1 > -> >=")
    assert "Accepted" in text
