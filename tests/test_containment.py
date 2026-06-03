"""Paths come from the model, so they are checked before anything opens them."""

import pytest

from branchproof.tools import ToolBox
from branchproof.workspace import (
    SOURCE_ONLY,
    OutsideWorkspace,
    is_test_path,
    resolve_inside,
    worktree,
)

ESCAPES = [
    "../../../tmp/test_evil.py",
    "/tmp/test_evil.py",
    "tests/../../../etc/passwd",
    "tests/../../test_sneaky.py",
]


@pytest.mark.parametrize("path", ESCAPES)
def test_paths_leaving_the_workspace_are_refused(sample_repo, path):
    with worktree(sample_repo) as space:
        with pytest.raises(OutsideWorkspace):
            resolve_inside(space.root, path)


def test_ordinary_paths_are_allowed(sample_repo):
    with worktree(sample_repo) as space:
        assert resolve_inside(space.root, "tests/test_bank.py").exists()


@pytest.mark.parametrize("path", ESCAPES)
def test_a_traversal_that_looks_like_a_test_file_is_still_refused(sample_repo, path):
    """The name check alone is not enough: ../../tmp/test_evil.py passes it."""
    with worktree(sample_repo) as space:
        with pytest.raises((OutsideWorkspace, ValueError)):
            space.write(path, "def test_x():\n    assert True\n")


def test_the_name_check_on_its_own_would_have_let_these_through():
    for path in ["../../../tmp/test_evil.py", "/tmp/test_evil.py"]:
        assert is_test_path(path), "name matching alone cannot be the containment check"


@pytest.mark.parametrize("path", ESCAPES)
def test_reading_outside_the_workspace_is_refused(sample_repo, env, path):
    with worktree(sample_repo) as space:
        assert "Refused" in ToolBox(space=space, env=env).read_file(path)


@pytest.mark.parametrize("path", ESCAPES)
def test_writing_outside_the_workspace_is_refused(sample_repo, env, path):
    with worktree(sample_repo) as space:
        result = ToolBox(space=space, env=env).edit_test(path, "x = 1\n")
        assert "Refused" in result
        assert space.changed_sources() == []


def test_running_tests_outside_the_workspace_is_refused(sample_repo, env):
    with worktree(sample_repo) as space:
        assert "Refused" in ToolBox(space=space, env=env).run_tests("/tmp/test_evil.py")


def test_nothing_was_written_outside(sample_repo, env, tmp_path):
    stray = tmp_path / "test_evil.py"
    with worktree(sample_repo) as space:
        ToolBox(space=space, env=env).edit_test(str(stray), "x = 1\n")
    assert not stray.exists()


def test_search_does_not_read_symlinks_outside_repository(sample_repo, env, tmp_path):
    outside = tmp_path / "private.py"
    outside.write_text("private_marker = 'never return this'\n")
    with worktree(sample_repo) as space:
        (space.root / "external.py").symlink_to(outside)
        tools = ToolBox(space=space, env=env)
        assert "No match" in tools.search_code("private_marker")
        assert "Refused" in tools.read_file("external.py")
        assert "external.py" in space.violations()


def test_edits_cannot_follow_symlink_aliases(sample_repo):
    with worktree(sample_repo, policy=SOURCE_ONLY) as space:
        (space.root / "alias.py").symlink_to(space.root / "bank.py")
        with pytest.raises(ValueError):
            space.write("alias.py", "overwrite = True\n")
        assert "can_withdraw" in space.read("bank.py")


@pytest.mark.parametrize("target", ["--help", "-p=arbitrary_plugin", "--rootdir=/tmp"])
def test_model_cannot_supply_pytest_options(sample_repo, env, target):
    with worktree(sample_repo) as space:
        assert "Refused" in ToolBox(space=space, env=env).run_tests(target)


@pytest.mark.parametrize("path", ["", " ", "tests/te\x00st.py"])
def test_malformed_paths_are_refused(sample_repo, path):
    with worktree(sample_repo) as space:
        with pytest.raises(OutsideWorkspace):
            resolve_inside(space.root, path)
