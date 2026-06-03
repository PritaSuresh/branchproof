from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Mutant:
    """One change to one line of source."""

    id: str
    path: str
    line: int
    col: int
    end_col: int
    kind: str
    before: str
    after: str

    @property
    def label(self) -> str:
        return f"{self.path}:{self.line} {self.before} -> {self.after}"


@dataclass(frozen=True)
class MutantStatus:
    mutant: Mutant
    killed: bool
    tests_run: tuple[str, ...]
    detail: str = ""


@dataclass(frozen=True)
class CheckResult:
    """One acceptance check and whether the candidate passed it."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    checks: tuple[CheckResult, ...]

    @property
    def first_failure(self) -> CheckResult | None:
        for check in self.checks:
            if not check.passed:
                return check
        return None

    @property
    def reason(self) -> str:
        failure = self.first_failure
        return "accepted" if failure is None else f"{failure.name}: {failure.detail}"


@dataclass(frozen=True)
class Action:
    """One step the model asked for."""

    tool: Literal["read_file", "search_code", "edit_test", "run_tests", "run_mutant", "finish"]
    args: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class Attempt:
    number: int
    action: str
    args_summary: str
    accepted: bool
    reason: str
    feedback: str = ""


@dataclass
class TaskRun:
    """Everything that happened while working on one mutant."""

    task_id: str
    repo: str
    mutant: Mutant
    attempts: list[Attempt] = field(default_factory=list)
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    accepted_test_path: str | None = None
    accepted_test_content: str | None = None
    status: Literal["success", "failed", "error"] = "failed"
    failure_class: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalBug:
    """A commit where the maintainers fixed something and added a test for it."""

    id: str
    repo: str
    fix_commit: str
    parent_commit: str
    module: str
    added_tests: tuple[str, ...]
    subject: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
