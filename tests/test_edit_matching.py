"""Edit matching, which is where most model patches used to die.

A model copies code out of a numbered read and loses a space, or writes the block
at the indentation it remembers rather than the one in the file. The edit was
still the right edit. These tests pin down what the tools accept, what they
refuse, and what they say back, because the reply is the model's next input.
"""

from branchproof.tools import ToolBox
from branchproof.workspace import SOURCE_ONLY, worktree


def box(space, env):
    """Source edits are what repair does, so these run under the repair policy."""
    space.enter(SOURCE_ONLY)
    return ToolBox(space=space, env=env)


def test_indentation_that_does_not_match_is_still_matched(sample_repo, env):
    with worktree(sample_repo) as space:
        tools = box(space, env)
        result = tools.replace_in_file("bank.py", "return balance >= amount",
                                       "return balance >= amount + 1")
        assert "Replaced" in result
        assert "return balance >= amount + 1" in space.read("bank.py")


def test_a_block_copied_at_the_wrong_indent_is_reindented(sample_repo, env):
    with worktree(sample_repo) as space:
        tools = box(space, env)
        result = tools.replace_in_file(
            "bank.py",
            "if amount <= 0:\n    return False",
            "if amount <= 0:\n    return None",
        )
        assert "Replaced" in result
        assert "ignoring" in result
        assert "        return None" in space.read("bank.py")
        assert "def can_withdraw" in space.read("bank.py")


def test_tabs_in_the_model_text_match_a_file_that_uses_spaces(sample_repo, env):
    with worktree(sample_repo) as space:
        tools = box(space, env)
        result = tools.replace_in_file("bank.py", "\treturn amount * 2", "\treturn amount * 3")
        assert "Replaced" in result
        assert "return amount * 3" in space.read("bank.py")


def test_missing_text_comes_back_with_the_closest_real_lines(sample_repo, env):
    with worktree(sample_repo) as space:
        result = box(space, env).replace_in_file("bank.py", "return amount * 22", "x")
    assert "not in bank.py" in result
    assert "closest lines" in result
    assert "return amount * 2" in result


def test_text_in_several_places_reports_the_line_numbers(sample_repo, env):
    with worktree(sample_repo) as space:
        tools = box(space, env)
        space.write("bank.py", "def a():\n    return 1\n\n\ndef b():\n    return 1\n")
        result = tools.replace_in_file("bank.py", "    return 1", "    return 2")
    assert "2 times" in result
    assert "lines 2, 6" in result
    assert "replace_lines" in result


def test_an_applied_edit_shows_the_new_line_numbers(sample_repo, env):
    with worktree(sample_repo) as space:
        result = box(space, env).replace_in_file("bank.py", "return amount * 2",
                                                 "return amount * 3")
    assert "now reads" in result
    assert "    8  " in result


def test_an_edit_that_breaks_the_file_is_refused_and_changes_nothing(sample_repo, env):
    with worktree(sample_repo) as space:
        before = space.read("bank.py")
        result = box(space, env).replace_in_file("bank.py", "return amount * 2", "return (((")
        assert "Not applied" in result
        assert "unparseable" in result
        assert space.read("bank.py") == before


def test_replace_lines_edits_by_position(sample_repo, env):
    with worktree(sample_repo) as space:
        result = box(space, env).replace_lines("bank.py", 8, 8, "    return amount * 3")
        assert "Replaced lines 8-8" in result
        assert "return amount * 3" in space.read("bank.py")


def test_replace_lines_refuses_a_range_outside_the_file(sample_repo, env):
    with worktree(sample_repo) as space:
        result = box(space, env).replace_lines("bank.py", 400, 401, "x = 1")
    assert "Invalid range" in result


def test_replace_lines_will_not_leave_the_file_unparseable(sample_repo, env):
    with worktree(sample_repo) as space:
        before = space.read("bank.py")
        result = box(space, env).replace_lines("bank.py", 8, 8, "    return (((")
        assert "Not applied" in result
        assert space.read("bank.py") == before


def test_a_long_read_says_where_it_stopped(sample_repo, env):
    with worktree(sample_repo) as space:
        tools = box(space, env)
        space.write("bank.py", "".join(f"x{n} = {n}\n" for n in range(4000)))
        text = tools.read_file("bank.py")
    assert "output stops here" in text
    assert "4000 lines" in text
