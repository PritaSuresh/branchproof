"""The CLI exposes only test-strengthening workflows."""

import tomllib
from pathlib import Path

import pytest

from branchproof.cli import build_parser, repository_map


def test_distribution_and_cli_are_branchproof():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["name"] == "branchproof"
    assert project["scripts"] == {"branchproof": "branchproof.cli:main"}
    help_text = build_parser().format_help()
    assert "usage: branchproof" in help_text
    for command in ("strengthen", "test-prepare", "test-study", "test-score"):
        assert command in help_text


@pytest.mark.parametrize("command", ["intake", "resolve", "solve", "compare", "triage"])
def test_retired_commands_are_not_callable(command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


@pytest.mark.parametrize("value", ["missing", "=path", "../repo=path", "repo="])
def test_repository_map_rejects_invalid_entries(value):
    with pytest.raises(ValueError):
        repository_map([value])


def test_repository_map_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        repository_map(["repo=one", "repo=two"])
