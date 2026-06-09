"""The tools are the only things the model can do, so the limits are checked here."""

from branchproof.coverage_map import covered_mutants
from branchproof.coverage_map import tests_for_line as lookup_tests
from branchproof.models import Mutant
from branchproof.tools import ToolBox
from branchproof.workspace import worktree


def box(space, env):
    return ToolBox(space=space, env=env)


def test_reading_a_file_shows_line_numbers(sample_repo, env):
    with worktree(sample_repo) as space:
        text = box(space, env).read_file("bank.py")
    assert "can_withdraw" in text
    assert "    1  " in text


def test_reading_a_missing_file_says_so(sample_repo, env):
    with worktree(sample_repo) as space:
        assert "does not exist" in box(space, env).read_file("nope.py")


def test_search_returns_file_and_line(sample_repo, env):
    with worktree(sample_repo) as space:
        hits = box(space, env).search_code("can_withdraw")
    assert "bank.py:1" in hits


def test_search_with_no_match_says_so(sample_repo, env):
    with worktree(sample_repo) as space:
        assert "No match" in box(space, env).search_code("zzzz_not_here")


def test_writing_a_test_works(sample_repo, env):
    with worktree(sample_repo) as space:
        result = box(space, env).edit_test("tests/test_a.py", "def test_a():\n    assert True\n")
        assert "Wrote" in result
        assert (space.root / "tests" / "test_a.py").exists()


def test_writing_to_the_source_is_refused(sample_repo, env):
    with worktree(sample_repo) as space:
        result = box(space, env).edit_test("bank.py", "x = 1\n")
        assert "Refused" in result
        assert space.changed_sources() == []


def test_running_tests_reports_the_outcome(sample_repo, env):
    with worktree(sample_repo) as space:
        assert box(space, env).run_tests("tests/test_bank.py").startswith("passed")


def test_existing_test_protection_also_applies_to_targeted_replacements(sample_repo, env):
    with worktree(sample_repo) as space:
        tools = ToolBox(space=space, env=env, protect_existing=True)
        before = space.read("tests/test_bank.py")
        result = tools.replace_in_file("tests/test_bank.py", "100, 500", "100, 50")
        assert "Refused" in result
        assert space.read("tests/test_bank.py") == before


def mutant(line):
    return Mutant(f"m{line}", "bank.py", line, 0, 1, "comparison", ">=", ">")


def test_an_empty_line_map_keeps_every_mutant():
    """A broken coverage run must not silently leave the module untested."""
    mutants = [mutant(1), mutant(2)]
    assert covered_mutants(mutants, {}) == mutants


def test_a_line_map_narrows_to_covered_lines():
    assert [m.line for m in covered_mutants([mutant(1), mutant(2)], {2: ["t"]})] == [2]


def test_tests_for_a_line_use_the_map_when_it_has_one(sample_repo):
    assert lookup_tests(sample_repo, "bank.py", 4, {4: ["tests/test_bank.py::x"]}) == [
        "tests/test_bank.py::x"
    ]


def test_tests_for_a_line_fall_back_to_matching_by_name(sample_repo):
    assert lookup_tests(sample_repo, "bank.py", 4, {}) == ["tests/test_bank.py"]
