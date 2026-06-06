"""Runs pytest inside a workspace and reports what happened."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from .sandbox import IMAGE, run_container
from .workspace import repository_files, resolve_inside

SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors|skipped)")
UPSTREAM_COLLECTION = ContextVar("upstream_collection", default=False)


def with_upstream_collection(function):
    """Opt-in for blind evaluation; historical workflows keep their defaults."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        token = UPSTREAM_COLLECTION.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            UPSTREAM_COLLECTION.reset(token)

    return wrapped


def upstream_collection_args(root):
    """Keep doctest/selection options, not coverage gates or reporting flags."""
    import configparser

    options = []
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text())
        value = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", [])
        options += shlex.split(value) if isinstance(value, str) else value
    for filename, section in (("pytest.ini", "pytest"), ("setup.cfg", "tool:pytest")):
        path = root / filename
        if path.exists():
            config = configparser.ConfigParser(interpolation=None)
            config.read(path)
            options += shlex.split(config.get(section, "addopts", fallback=""))
    return [
        option
        for option in options
        if option == "--doctest-modules"
        or option.startswith(("--doctest-glob=", "--ignore=", "--ignore-glob=", "--import-mode="))
    ]


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    test_counts: dict[str, int] | None = None
    failure_messages: tuple[str, ...] = ()
    report_errors: tuple[str, ...] = ()
    report_error: str | None = None
    collection_only: bool = False
    coverage_report: dict | None = None

    @property
    def passed(self) -> bool:
        executed = (
            self.test_counts is None
            or self.collection_only
            or self.test_counts.get("passed", 0) > 0
        )
        clean = not self.report_errors and not (
            self.test_counts and (self.test_counts.get("failed") or self.test_counts.get("error"))
        )
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.report_error
            and executed
            and clean
        )

    @property
    def collection_error(self) -> bool:
        if self.test_counts is not None or self.report_error:
            return bool(self.report_errors or self.report_error)
        text = self.stdout + self.stderr
        return "ERROR collecting" in text or "INTERNALERROR" in text

    @property
    def behavioral_failure(self) -> bool:
        """A test body failed, with no collection, fixture, or execution errors.

        This is execution evidence, not proof that the failure matches a reported
        bug. The evaluator must check that separately against hidden ground truth.
        """
        return (
            self.returncode == 1
            and not self.timed_out
            and not self.collection_error
            and self.counts().get("failed", 0) > 0
            and not self.counts().get("error", 0)
        )

    def counts(self) -> dict[str, int]:
        if self.test_counts is not None:
            return dict(self.test_counts)
        found: dict[str, int] = {}
        for number, label in SUMMARY.findall(self.stdout):
            found[label.rstrip("s")] = int(number)
        return found

    def tail(self, lines: int = 40) -> str:
        text = (self.stdout + "\n" + self.stderr).strip().splitlines()
        return "\n".join(text[-lines:])

    def feedback(self) -> str:
        """Keep actual failure evidence even when warning summaries fill stdout.

        Structured outcomes remain the authority; this is bounded model feedback.
        Do not infer the cause just from a failed test's name.
        """
        messages = [*self.failure_messages, *self.report_errors]
        if self.report_error:
            messages.append(self.report_error)
        detail = "\n".join(messages)[:4000]
        return ("Structured failure details:\n" + detail + "\n\n" if detail else "") + (
            self.tail(18)[-2000:]
        )


def python_for(env: Path) -> Path:
    """Interpreter inside a virtualenv, always absolute.

    Subprocesses run with cwd set to the repository being tested, so a relative
    path here would be looked up in the wrong place.
    """
    env = env.resolve()
    candidate = env / "bin" / "python"
    return candidate if candidate.exists() else env / "Scripts" / "python.exe"


def normalize_test_target(root: Path, target: str) -> str:
    """Preserve pytest node IDs while ensuring their file part is a local path."""
    if not isinstance(target, str) or not target.strip() or target.lstrip().startswith("-"):
        raise ValueError("a test target must be a repository path, not a pytest option")
    filename, separator, node = target.partition("::")
    path = resolve_inside(root, filename)
    relative = path.relative_to(root.resolve()).as_posix()
    return relative + (separator + node if separator else "")


def _test_environment(root: Path, extra: dict[str, str] | None) -> dict[str, str]:
    """Give test processes system settings without inheriting provider credentials.

    Tests still run with this user's OS permissions. This reduces accidental
    credential exposure; it is not filesystem or network isolation.
    """
    allowed = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TMP",
        "TEMP",
        "TMPDIR",
        "LANG",
        "LANGUAGE",
        "TZ",
    }
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in allowed or name.startswith("LC_")
    }
    environment["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        environment.update(extra)
    return environment


def _read_test_report(path: Path) -> tuple[dict[str, int], tuple[str, ...], tuple[str, ...]]:
    """Read pytest's built-in JUnit report rather than matching captured output."""
    return _parse_test_report(path.read_text())


def _parse_test_report(xml: str) -> tuple[dict[str, int], tuple[str, ...], tuple[str, ...]]:
    report = ET.fromstring(xml)
    if report.tag not in {"testsuites", "testsuite"}:
        raise ValueError("unrecognised pytest report")
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    failures: list[str] = []
    errors: list[str] = []
    for case in report.iter("testcase"):
        failed = case.findall("failure")
        errored = case.findall("error")
        skipped = case.findall("skipped")
        if failed:
            counts["failed"] += 1
            failures.extend(item.get("message", "") for item in failed)
        if errored:
            counts["error"] += 1
            errors.extend(item.get("message", "") for item in errored)
        if skipped:
            counts["skipped"] += 1
        if not failed and not errored and not skipped:
            counts["passed"] += 1
    return counts, tuple(failures), tuple(errors)


def run_pytest(
    root: Path,
    env: Path,
    targets: list[str] | None = None,
    timeout: int = 300,
    extra: list[str] | None = None,
    env_extra: dict[str, str] | None = None,
) -> RunResult:
    root = root.resolve()
    if UPSTREAM_COLLECTION.get():
        extra = list(dict.fromkeys(upstream_collection_args(root) + (extra or [])))
    normalized = [normalize_test_target(root, target) for target in targets or []]
    if IMAGE.get():
        try:
            result = run_container(root, normalized, extra or [], timeout, env_extra)
            if result["timed_out"]:
                return RunResult(-1, result["stdout"], result["stderr"], timed_out=True)
            counts, failures, errors = _parse_test_report(result["xml"])
            return RunResult(
                result["returncode"],
                result["stdout"],
                result["stderr"],
                test_counts=counts,
                failure_messages=failures,
                report_errors=errors,
                report_error="Tests modified the copied repository."
                if result["files_changed"]
                else None,
                collection_only="--collect-only" in (extra or []),
                coverage_report=result.get("coverage_report"),
            )
        except (OSError, RuntimeError, ValueError, KeyError, ET.ParseError) as error:
            return RunResult(
                -1, "", str(error), report_error="Sandbox did not return valid evidence."
            )
    # Projects put things like --cov-fail-under=99 in their pytest addopts. Running
    # one test file then fails the coverage gate and pytest exits non zero, which
    # looks exactly like the test failing. Clear the project's addopts and pass only
    # what this harness asks for.
    command = [
        str(python_for(env)),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:randomly",
        "--no-header",
        "-o",
        "addopts=",
    ]
    if extra:
        command += extra
    # The virtualenv has the package installed from the original clone. Putting the
    # workspace first means we test the checked out code, not the installed copy.
    environment = _test_environment(root, env_extra)
    with tempfile.TemporaryDirectory(prefix="hardener-pytest-") as temporary:
        report = Path(temporary) / "results.xml"
        command += [f"--junitxml={report}", "-o", "junit_family=xunit2"]
        if normalized:
            command += ["--", *normalized]
        try:
            finished = subprocess.run(
                command, cwd=root, capture_output=True, text=True, timeout=timeout, env=environment
            )
        except subprocess.TimeoutExpired as expired:
            partial = expired.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            return RunResult(returncode=-1, stdout=partial, stderr="timed out", timed_out=True)
        try:
            counts, failures, errors = _read_test_report(report)
        except (OSError, ET.ParseError, ValueError) as error:
            return RunResult(
                finished.returncode,
                finished.stdout,
                finished.stderr,
                report_error=f"pytest did not produce a valid outcome report: {error}",
            )
        return RunResult(
            finished.returncode,
            finished.stdout,
            finished.stderr,
            test_counts=counts,
            failure_messages=failures,
            report_errors=errors,
            collection_only="--collect-only" in (extra or []),
        )


def collect_only(root: Path, env: Path, target: str, timeout: int = 120) -> RunResult:
    return run_pytest(root, env, [target], timeout=timeout, extra=["--collect-only"])


def tests_touching(root: Path, module: str) -> list[str]:
    """Test files that mention the module, used when no coverage map exists."""
    stem = Path(module).stem
    package = Path(module).parent.name
    hits: list[str] = []
    for path in sorted(repository_files(root)):
        if not path.name.startswith("test_") or path.suffix != ".py" or path.is_symlink():
            continue
        text = path.read_text(errors="replace")
        if stem in text or (package and package in text):
            hits.append(str(path.relative_to(root)))
    return hits


def load_coverage_map(path: Path) -> dict[str, dict[int, list[str]]]:
    if not path.exists():
        return {}
    return {
        module: {int(line): tests for line, tests in lines.items()}
        for module, lines in json.loads(path.read_text()).items()
    }
