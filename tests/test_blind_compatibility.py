"""Context and setup corrections cannot leak outcomes or silently drop doctests."""

import pytest

from branchproof import blind_eval
from branchproof.runner import (
    UPSTREAM_COLLECTION,
    run_pytest,
    upstream_collection_args,
    with_upstream_collection,
)


def test_equivalent_github_origins_are_allowed_but_other_repos_are_not():
    assert blind_eval.canonical_origin(
        "https://github.com/Owner/Repo.git"
    ) == blind_eval.canonical_origin("https://github.com/owner/repo")
    assert blind_eval.canonical_origin(
        "https://github.com/owner/other"
    ) != blind_eval.canonical_origin("https://github.com/owner/repo")
    with pytest.raises(ValueError):
        blind_eval.canonical_origin("https://example.com/owner/repo")


def test_reexported_function_selects_only_complete_relevant_test_file(tmp_path):
    (tmp_path / "validation.py").write_text(
        "def first():\n    return 1\ndef second():\n    return 2\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_first.py").write_text("from package import first\n")
    (tmp_path / "tests/test_second.py").write_text("from package import second\n")
    (tmp_path / "tests/conftest.py").write_text("# applicable fixture\n")
    paths = ["tests/test_first.py", "tests/test_second.py", "tests/conftest.py"]
    selected = blind_eval.relevant_tests(tmp_path, "validation.py", paths, "first")
    assert selected == ["tests/conftest.py", "tests/test_first.py"]


def test_initial_reservation_uses_identical_generation_request():
    _, system, messages = blind_eval.initial_request("demo", "original code", "direct")
    import json

    assert blind_eval.first_reservation("demo", "original code") == (
        len(json.dumps(messages).encode()) + len(system.encode()) + 8192
    )


def test_collection_preserves_doctests_not_coverage_gates(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        'addopts = ["--doctest-modules", "--cov=x", "--cov-fail-under=100"]\n'
    )
    assert upstream_collection_args(tmp_path) == ["--doctest-modules"]


def test_collection_mode_restored_even_on_error():
    @with_upstream_collection
    def fails():
        assert UPSTREAM_COLLECTION.get()
        raise ValueError("failure")

    assert not UPSTREAM_COLLECTION.get()
    with pytest.raises(ValueError):
        fails()
    assert not UPSTREAM_COLLECTION.get()


def test_failing_upstream_doctest_is_really_executed(tmp_path, env):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = ["--doctest-modules"]\n'
    )
    (tmp_path / "example.py").write_text('"""\n>>> 1 + 1\n3\n"""\n')
    (tmp_path / "test_example.py").write_text("def test_normal():\n    assert True\n")
    result = with_upstream_collection(run_pytest)(tmp_path, env)
    assert result.behavioral_failure
    assert result.counts()["failed"] == 1
    assert result.counts()["passed"] == 1


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef entropy(n):\n    return os.urandom(n)\n",
        "from os import urandom as entropy_source\ndef entropy(n):\n    return entropy_source(n)\n",
        "import secrets as s\ndef entropy(n):\n    return s.token_hex(n)\n",
    ],
)
def test_explicit_entropy_never_qualifies_as_deterministic(source):
    nodes = list(blind_eval.function_nodes(source))
    assert len(nodes) == 1
    assert nodes[0][1] == "external-state operation"
