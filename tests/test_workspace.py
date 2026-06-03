import pytest

from branchproof.workspace import (
    FROZEN,
    SOURCE_ONLY,
    TESTS_ONLY,
    hash_sources,
    is_test_path,
    worktree,
)


def test_test_paths_are_recognised():
    assert is_test_path("tests/test_a.py")
    assert is_test_path("conftest.py")
    assert is_test_path("pkg/tests/test_b.py")
    assert not is_test_path("src/pkg/core.py")


def test_hashes_skip_test_files(sample_repo):
    digests = hash_sources(sample_repo)
    assert "bank.py" in digests
    assert "tests/test_bank.py" not in digests


def test_editing_source_is_detected(sample_repo):
    with worktree(sample_repo) as space:
        assert space.changed_sources() == []
        (space.root / "bank.py").write_text("def can_withdraw(a, b):\n    return True\n")
        assert space.changed_sources() == ["bank.py"]


def test_writing_a_test_is_not_a_source_change(sample_repo):
    with worktree(sample_repo) as space:
        space.write("tests/test_new.py", "def test_x():\n    assert True\n")
        assert space.changed_sources() == []


def test_writing_outside_tests_is_refused(sample_repo):
    with worktree(sample_repo) as space:
        try:
            space.write("bank.py", "")
        except ValueError:
            return
        raise AssertionError("writing to a source file should have been refused")


@pytest.mark.parametrize("policy", [TESTS_ONLY, SOURCE_ONLY])
@pytest.mark.parametrize("path", [".git", "pyproject.toml", "tests/pytest.ini", ".git/hook.py"])
def test_write_policies_refuse_metadata_and_non_python_files(sample_repo, policy, path):
    with worktree(sample_repo, policy=policy) as space:
        with pytest.raises(ValueError):
            space.write(path, "changed = true\n")


def test_source_policy_cannot_be_tricked_by_dotdot(sample_repo):
    with worktree(sample_repo, policy=SOURCE_ONLY) as space:
        assert not space.may_write("tests/../tests/test_bank.py")
        assert space.may_write("tests/../bank.py")


@pytest.mark.parametrize("policy", [TESTS_ONLY, SOURCE_ONLY, FROZEN])
def test_non_python_config_changes_are_integrity_violations(sample_repo, policy):
    with worktree(sample_repo, policy=policy) as space:
        (space.root / "pytest.ini").write_text("[pytest]\npython_files = nothing.py\n")
        assert "pytest.ini" in space.violations()


def test_unknown_policy_does_not_enable_source_edits(sample_repo):
    with worktree(sample_repo) as space:
        with pytest.raises(ValueError, match="Unknown write policy"):
            space.enter("typo")


def test_integrity_scan_hashes_symlink_identity_without_reading_target(sample_repo, tmp_path):
    with worktree(sample_repo) as space:
        link = space.root / "outside.py"
        link.symlink_to(tmp_path / "missing.py")
        assert "outside.py" in space.changed_sources()
