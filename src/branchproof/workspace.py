"""Gives every task its own copy of the repository.

Tasks write test files and apply mutants, so they must not share a directory.
Each task gets a git worktree at a chosen commit. Production files are hashed on
the way in and checked on the way out, so a task that edits source instead of
tests is caught rather than trusted.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

TEST_MARKERS = ("tests/", "test_", "_test.py", "conftest.py")


class OutsideWorkspace(ValueError):
    """A path that would land outside the task's own copy of the repository."""


def resolve_inside(root: Path, relative: str) -> Path:
    """Turn a path from the model into a real one, or refuse it.

    Joining a root to "../.." or to an absolute path escapes the root, so the
    result is resolved and checked against the root before anything touches it.
    """
    if not isinstance(relative, str) or not relative.strip() or "\x00" in relative:
        raise OutsideWorkspace("empty or malformed path")
    try:
        candidate = (root / relative).resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise OutsideWorkspace("malformed or cyclic path") from error
    base = root.resolve()
    if candidate != base and base not in candidate.parents:
        raise OutsideWorkspace(f"{relative} is outside the workspace")
    return candidate


def is_test_path(relative: str) -> bool:
    path = Path(relative)
    name = path.name
    return (
        any(part in {"tests", "test"} for part in path.parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


SKIP_DIRS = {".git", ".venv", "__pycache__", "tests", "test", "build", "dist", "docs", "examples"}
SKIP_NAMES = {"setup.py", "conf.py", "__init__.py", "__main__.py", "_version.py", "version.py"}


def is_real_module(relative: str) -> bool:
    """Code worth writing tests for. Packaging, docs and re-export files are not."""
    path = Path(relative)
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if path.name in SKIP_NAMES or path.name.startswith("test"):
        return False
    return path.suffix == ".py" and not path.name.endswith("_test.py")


def hash_sources(root: Path) -> dict[str, str]:
    """Hash every python file that is not a test."""
    return {
        name: digest for name, digest in hash_all(root).items()
        if Path(name).suffix == ".py" and not is_test_path(name)
    }


# What a stage of work is allowed to change. The point is that a job which writes
# a failing test must not be able to edit the code to make it pass, and a job that
# edits code must not be able to edit the test instead.
TESTS_ONLY = "tests"
SOURCE_ONLY = "source"
FROZEN = "frozen"

# Interpreter/test artifacts do not belong to a proposed patch. Everything else,
# including configuration and non-Python files, participates in integrity checks.
ARTIFACT_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".tox", ".nox", "build", "dist", "htmlcov",
}


def repository_files(root: Path) -> Iterator[Path]:
    """Walk repository files without following symlinks or reading their targets."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in sorted(dirs):
            path = parent / name
            if path.is_symlink():
                yield path
        dirs[:] = sorted(
            name for name in dirs
            if name not in ARTIFACT_DIRS and not (parent / name).is_symlink()
        )
        for name in sorted(files):
            if name == ".coverage" or name.startswith(".coverage.") or name.endswith(".pyc"):
                continue
            yield parent / name


def _editable_python(relative: str) -> bool:
    path = Path(relative)
    return path.suffix == ".py" and not any(
        part in ARTIFACT_DIRS or part.startswith(".") for part in path.parts
    )


def hash_all(root: Path) -> dict[str, str]:
    """Hash repository content, including config, deletions and symlink identity."""
    digests: dict[str, str] = {}
    for path in repository_files(root):
        content = (
            b"symlink\0" + os.fsencode(os.readlink(path))
            if path.is_symlink() else b"file\0" + path.read_bytes()
        )
        digests[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
    return digests


@dataclass
class Workspace:
    root: Path
    repo: str
    commit: str
    _baseline: dict[str, str]
    policy: str = TESTS_ONLY

    def changed_sources(self) -> list[str]:
        """Production files that differ from when the task started."""
        current = hash_all(self.root)
        changed = [
            name for name, digest in current.items()
            if self._baseline.get(name) != digest and not is_test_path(name)
        ]
        removed = [
            name for name in self._baseline
            if name not in current and not is_test_path(name)
        ]
        return sorted(changed + removed)

    def violations(self) -> list[str]:
        """Files this task was not allowed to touch but did.

        A job that writes tests must not edit the code under test, or it can make
        any test pass. A job that fixes code must not edit the tests, or it can
        make any failure go away. Same check, opposite side.
        """
        current = hash_all(self.root)
        touched = sorted(
            name for name, digest in current.items() if self._baseline.get(name) != digest
        )
        touched += sorted(name for name in self._baseline if name not in current)
        if self.policy == FROZEN:
            return touched
        return [name for name in touched if not self.may_write(name)]

    def may_write(self, relative: str) -> bool:
        if self.policy not in {TESTS_ONLY, SOURCE_ONLY}:
            return False
        try:
            target = resolve_inside(self.root, relative)
        except OutsideWorkspace:
            return False
        requested = self.root / relative
        # A symlink may change after an earlier check. Do not use symlink aliases
        # for edits, even when their current target is within the worktree.
        while requested != self.root and self.root in requested.parents:
            if requested.is_symlink():
                return False
            requested = requested.parent
        canonical = str(target.relative_to(self.root.resolve()))
        if self.policy == TESTS_ONLY and Path(canonical).name in {"conftest.py", "__init__.py"}:
            return False
        # Resolve aliases before classification: tests/../package.py is source.
        return _editable_python(canonical) and (
            is_test_path(canonical) == (self.policy == TESTS_ONLY)
        )

    def snapshot(self) -> dict[str, str]:
        """Contents of editable production Python files for repair rollback."""
        saved: dict[str, str] = {}
        for name in hash_all(self.root):
            if not _editable_python(name) or is_test_path(name):
                continue
            path = self.root / name
            if not path.is_symlink():
                saved[name] = path.read_text()
        return saved

    def restore(self, saved: dict[str, str]) -> None:
        """Put the source back. A failed repair leaves the file broken, and the
        next attempt would otherwise start from the damage."""
        for name, content in saved.items():
            target = resolve_inside(self.root, name)
            if not target.exists() or target.read_text() != content:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        for name in hash_all(self.root):
            if _editable_python(name) and not is_test_path(name) and name not in saved:
                (self.root / name).unlink(missing_ok=True)

    def enter(self, policy: str) -> None:
        """Move to the next stage and take its baseline from where we are now."""
        if policy not in {TESTS_ONLY, SOURCE_ONLY, FROZEN}:
            raise ValueError(f"Unknown write policy: {policy}")
        self.policy = policy
        self.rebaseline()

    def rebaseline(self) -> None:
        self._baseline = hash_all(self.root)

    def write(self, relative: str, content: str) -> Path:
        # Containment is checked before the name, because a traversal path can end
        # in something that looks like a perfectly ordinary test file.
        target = resolve_inside(self.root, relative)
        if not isinstance(content, str):
            raise ValueError("file content must be a string")
        if not self.may_write(relative):
            wanted = {
                TESTS_ONLY: "a test file",
                SOURCE_ONLY: "a source file",
            }.get(self.policy, "writable at this stage")
            raise ValueError(f"{relative} is not {wanted}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def read(self, relative: str) -> str:
        return resolve_inside(self.root, relative).read_text()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _materialize_submodules(clone: Path, checkout: Path, commit: str) -> None:
    """Copy pinned submodule blobs from an operator-initialized local cache.

    Never fetch URLs from an untrusted checkout or run its setup scripts. Missing
    submodules fail closed: otherwise pytest can silently collect fewer tests.
    Only regular files are copied; archive paths and total size are bounded.
    """
    links = []
    for record in _git(clone, "ls-tree", "-rz", commit).split("\0"):
        if record.startswith("160000 "):
            header, path = record.split("\t", 1)
            links.append((path, header.split()[2]))
    if not links:
        return
    config = _git(clone, "config", f"--blob={commit}:.gitmodules",
                  "--get-regexp", r"^submodule\..*\.path$")
    names = {}
    for line in config.splitlines():
        key, path = line.split(None, 1)
        names[path] = key[len("submodule."):-len(".path")]
    common = Path(_git(clone, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = clone / common
    for relative, revision in links:
        if relative not in names:
            raise ValueError(f"Submodule configuration missing: {relative}")
        database = resolve_inside(common / "modules", names[relative])
        if not database.is_dir():
            raise ValueError(f"Initialize the trusted submodule cache before running: {relative}")
        tree = subprocess.run(
            ["git", f"--git-dir={database.resolve()}", "ls-tree", "-r", revision],
            capture_output=True, text=True, check=True,
        ).stdout
        if any(line.startswith("160000 ") for line in tree.splitlines()):
            raise ValueError("Nested submodules are not supported; do not silently omit them.")
        archive = subprocess.run(
            ["git", f"--git-dir={database.resolve()}", "archive", "--format=tar", revision],
            capture_output=True, check=True,
        ).stdout
        if len(archive) > 32 * 1024 * 1024:
            raise ValueError("Submodule archive exceeds 32 MiB.")
        destination = resolve_inside(checkout, relative)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            for member in bundle:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("Submodule archives must contain only regular files.")
                target = resolve_inside(destination, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source:
                    target.write_bytes(source.read())


@contextmanager
def worktree(
    clone: Path, commit: str = "HEAD", keep: bool = False, policy: str = TESTS_ONLY
) -> Iterator[Workspace]:
    """Check out a commit into a throwaway directory."""
    target = Path(tempfile.mkdtemp(prefix="hardener-"))
    checkout = target / "repo"
    _git(clone, "worktree", "add", "--detach", str(checkout), commit)
    try:
        _materialize_submodules(clone, checkout, commit)
        space = Workspace(
            root=checkout,
            repo=clone.name,
            commit=_git(checkout, "rev-parse", "HEAD"),
            _baseline=hash_all(checkout),
            policy=policy,
        )
        yield space
    finally:
        if not keep:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(checkout)],
                cwd=clone, capture_output=True, text=True,
            )
            shutil.rmtree(target, ignore_errors=True)
