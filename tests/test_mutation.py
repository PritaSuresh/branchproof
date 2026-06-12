import pytest

from branchproof.mutation import apply_mutant, discover

SOURCE = """def check(balance, amount):
    if amount <= 0:
        return False
    return balance >= amount
"""


def kinds(source, path="m.py"):
    return {m.kind for m in discover(source, path)}


def test_finds_comparison_boundary_and_direction():
    mutants = [m for m in discover(SOURCE, "m.py") if m.before == ">="]
    assert {m.after for m in mutants} == {">", "<="}


def test_finds_the_main_kinds():
    source = "def f(a, b):\n    return a + b > 1 and a is not None\n"
    assert {"arithmetic", "comparison", "boolean", "constant"} <= kinds(source)


def test_applying_a_mutant_changes_only_that_span():
    mutant = next(m for m in discover(SOURCE, "m.py") if m.before == ">=" and m.after == ">")
    changed = apply_mutant(SOURCE, mutant)
    assert changed.count("\n") == SOURCE.count("\n")
    assert "balance > amount" in changed
    assert "amount <= 0" in changed


def test_applying_to_the_wrong_source_is_refused():
    mutant = next(m for m in discover(SOURCE, "m.py") if m.before == ">=")
    with pytest.raises(ValueError):
        apply_mutant("def other():\n    pass\n", mutant)


def test_multiline_expressions_are_skipped():
    source = "def f(a, b):\n    return (a\n            > b)\n"
    assert not [m for m in discover(source, "m.py") if m.kind == "comparison"]


def test_ids_are_stable():
    first = [m.id for m in discover(SOURCE, "m.py")]
    second = [m.id for m in discover(SOURCE, "m.py")]
    assert first == second
