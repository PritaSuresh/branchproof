"""Execution evidence must come from pytest outcomes, not printed text."""

import pytest

from branchproof.runner import collect_only, normalize_test_target, run_pytest


def test_failure_counts_cannot_be_overridden_by_printed_summary(tmp_path, env):
    (tmp_path / "test_example.py").write_text(
        "def test_bug():\n"
        "    print('999 passed, 0 failed')\n"
        "    assert 1 == 2\n"
    )
    result = run_pytest(tmp_path, env)
    assert result.behavioral_failure
    assert result.counts()["failed"] == 1
    assert result.counts()["passed"] == 0
    assert result.failure_messages


def test_fixture_error_is_not_a_behavioral_failure(tmp_path, env):
    (tmp_path / "test_example.py").write_text(
        "def test_bug(missing_fixture):\n    assert False\n"
    )
    result = run_pytest(tmp_path, env)
    assert not result.behavioral_failure
    assert result.collection_error
    assert result.counts()["error"] == 1


def test_collection_error_is_reported_without_parsing_stdout(tmp_path, env):
    (tmp_path / "test_example.py").write_text("import certainly_missing_package_123456\n")
    result = run_pytest(tmp_path, env)
    assert not result.behavioral_failure
    assert result.collection_error
    assert result.test_counts is not None


def test_teardown_error_cannot_count_as_clean_test_failure(tmp_path, env):
    (tmp_path / "test_example.py").write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def cleanup():\n"
        "    yield\n"
        "    raise RuntimeError('broken cleanup')\n"
        "def test_bug(cleanup):\n"
        "    assert False\n"
    )
    result = run_pytest(tmp_path, env)
    assert result.counts()["failed"] == 1
    assert result.counts()["error"] == 1
    assert not result.behavioral_failure


def test_parent_credentials_and_pytest_options_are_not_inherited(tmp_path, env, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-secret-not-a-real-key")
    monkeypatch.setenv("UNRELATED_PRIVATE_TOKEN", "another-dummy-secret")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--this-would-break-pytest")
    (tmp_path / "test_environment.py").write_text(
        "import os\n"
        "def test_environment():\n"
        "    assert 'AZURE_OPENAI_API_KEY' not in os.environ\n"
        "    assert 'UNRELATED_PRIVATE_TOKEN' not in os.environ\n"
        "    assert os.environ['HARDENER_TEST_SETTING'] == 'explicit'\n"
    )
    result = run_pytest(tmp_path, env, env_extra={"HARDENER_TEST_SETTING": "explicit"})
    assert result.passed, result.tail()


@pytest.mark.parametrize("target", ["--help", "-p=plugin", "../test_outside.py", ""])
def test_invalid_test_targets_are_rejected_before_running(tmp_path, env, target):
    with pytest.raises(ValueError):
        run_pytest(tmp_path, env, [target])


def test_node_id_is_kept_while_file_path_is_normalized(tmp_path):
    assert normalize_test_target(tmp_path, "tests/../test_a.py::test_x[param]") == (
        "test_a.py::test_x[param]"
    )


def test_collect_only_reports_collection_success(tmp_path, env):
    (tmp_path / "test_example.py").write_text("def test_bug():\n    assert False\n")
    result = collect_only(tmp_path, env, "test_example.py")
    assert result.passed, result.tail()
    assert result.counts()["failed"] == 0


def test_timeout_is_never_a_behavioral_failure(tmp_path, env):
    (tmp_path / "test_example.py").write_text(
        "import time\ndef test_wait():\n    time.sleep(30)\n"
    )
    result = run_pytest(tmp_path, env, timeout=1)
    assert result.timed_out
    assert not result.behavioral_failure
    assert not result.passed


def test_all_skipped_candidate_is_not_a_passing_repair(tmp_path, env):
    (tmp_path / "test_skip.py").write_text(
        "import pytest\ndef test_bug():\n    pytest.skip('not actually tested')\n"
    )
    result = run_pytest(tmp_path, env)
    assert result.returncode == 0
    assert result.counts()["skipped"] == 1
    assert not result.passed


def test_model_feedback_preserves_failure_when_warnings_fill_the_tail():
    from branchproof.runner import RunResult

    result = RunResult(
        1, "warning\n" * 300, "", test_counts={"failed": 1},
        failure_messages=("UnboundLocalError: ParserError was referenced before assignment",),
    )
    assert "UnboundLocalError" not in result.tail(25)
    assert "UnboundLocalError" in result.feedback()
    assert len(result.feedback()) < 6100
