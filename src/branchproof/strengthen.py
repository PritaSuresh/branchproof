"""A bounded, patch-only front door to mutation-guided test generation.

Each target is independent: patches are not claimed to be compatible as a batch.
The caller selects Docker execution before invoking this workflow.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from .agent import Provider, ProviderError
from .artifacts import new_run_directory
from .loop import harden_mutant
from .models import Mutant
from .mutation import discover_in_file, write_mutant
from .runner import run_pytest, tests_touching
from .verify import verify_candidate
from .workspace import hash_all, is_test_path, resolve_inside, worktree


def input_hashes(root: Path) -> dict[str, str]:
    """Git worktree pointer files differ between identical source snapshots."""
    return {name: value for name, value in hash_all(root).items() if name != ".git"}


class CallBudget:
    """Shared application-call limit; SDK transport retries are not model decisions."""

    def __init__(self, provider: Provider | None, limit: int):
        self.provider = provider
        self.limit = limit
        self.calls = 0
        self.trace: list[dict] = []

    @property
    def last_response_metadata(self):
        return getattr(self.provider, "last_response_metadata", {})

    def next_action(self, messages, system=None):
        if self.provider is None:
            raise ProviderError("No model provider selected", usage_known=True)
        if self.calls >= self.limit:
            raise ProviderError("Shared model-call budget exhausted", usage_known=True,
                                kind="call_budget")
        self.calls += 1
        entry = {"call": self.calls, "messages": [dict(m) for m in messages]}
        self.trace.append(entry)
        try:
            result = (self.provider.next_action(messages) if system is None else
                      self.provider.next_action(messages, system=system))
        except Exception as error:
            entry["error"] = str(error)
            entry["usage_known"] = getattr(error, "usage_known", False)
            entry["prompt_tokens"] = getattr(error, "prompt_tokens", 0)
            entry["completion_tokens"] = getattr(error, "completion_tokens", 0)
            response = getattr(error, "response", "")
            if response:
                entry["response"] = response
            raise
        action, prompt_tokens, completion_tokens = result
        entry.update(action=action.model_dump(), prompt_tokens=prompt_tokens,
                     completion_tokens=completion_tokens, usage_known=True)
        entry["response_metadata"] = dict(self.last_response_metadata)
        return result


def strengthen(
    repo: Path, env: Path, module: str, provider: Provider | None, output: Path,
    *, limit: int = 3, candidate_limit: int = 20, max_attempts: int = 3,
    max_calls: int = 30, stability_runs: int = 3, screen_only: bool = False,
    provider_metadata: dict | None = None,
    frozen_targets: list[Mutant] | None = None,
    expected_revision: str | None = None, expected_hashes: dict | None = None,
    screen_timeout: int = 30, suite_timeout: int = 300, test_timeout: int = 120,
) -> tuple[Path, dict]:
    """Screen bounded mutations with the full suite, then export independent tests.

    The committed HEAD is the input; dirty checkouts are rejected to avoid
    silently evaluating a different version than the user supplied.
    """
    if min(limit, candidate_limit, max_attempts, max_calls, stability_runs,
           screen_timeout, suite_timeout, test_timeout) < 1:
        raise ValueError("All limits must be positive")
    if provider is None and not screen_only:
        raise ValueError("A provider is required unless screen_only is selected")
    repo = repo.resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=repo,
                         capture_output=True, text=True, check=True).stdout.strip()
    if Path(top).resolve() != repo:
        raise ValueError("--repo must name the Git repository root, not a subdirectory")
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                            cwd=repo, capture_output=True, text=True, check=True)
    if status.stdout.strip():
        raise ValueError("Use a clean committed repository; no files were changed")
    source = resolve_inside(repo, module)
    if not source.is_file() or source.suffix != ".py":
        raise ValueError("Module must be a Python source file inside the repository")
    module = source.relative_to(repo).as_posix()
    if is_test_path(module):
        raise ValueError("Choose a production module, not an existing test file")
    output = output.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Output must be outside the input repository")
    directory = new_run_directory(output)
    budget = CallBudget(provider, max_calls)
    report = {"schema_version": 1, "repo": str(repo), "module": module,
              "revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                         capture_output=True, text=True, check=True).stdout.strip(),
              "status": "running", "screened": [], "targets": [],
              "provider": provider_metadata or {"kind": "unspecified"},
              "settings": {"limit": limit, "candidate_limit": candidate_limit,
                           "max_attempts": max_attempts, "max_calls": max_calls,
                           "stability_runs": stability_runs, "screen_only": screen_only},
              "scope": "Independent patches; shown-mutant detection, not real-bug accuracy"}
    report["settings"].update(screen_timeout=screen_timeout, suite_timeout=suite_timeout,
                              test_timeout=test_timeout)
    try:
        with worktree(repo) as space:
            report["input_hashes"] = input_hashes(space.root)
            if ((expected_revision and space.commit != expected_revision)
                    or (expected_hashes is not None and report["input_hashes"] != expected_hashes)):
                raise ValueError("Frozen repository input changed before screening")
            baseline = run_pytest(space.root, env, timeout=suite_timeout)
            report["baseline"] = {"passed": baseline.passed, "counts": baseline.counts(),
                                  "detail": baseline.feedback()}
            if not baseline.passed:
                raise ValueError("Baseline suite must pass and execute tests before generation")
            available = discover_in_file(space.root, module)
            if frozen_targets is not None:
                if (not frozen_targets or len({m.id for m in frozen_targets}) != len(frozen_targets)
                        or any(m.path != module or m not in available for m in frozen_targets)):
                    raise ValueError("Frozen mutations must exactly match module candidates")
                candidates = frozen_targets
            else:
                candidates = available
            report["discovered"] = len(candidates)
            survivors = []
            for mutant in candidates[:candidate_limit]:
                original = write_mutant(space.root, mutant)
                try:
                    result = run_pytest(space.root, env, timeout=screen_timeout)
                finally:
                    resolve_inside(space.root, mutant.path).write_text(original)
                outcome = ("survived" if result.passed else
                           "killed" if result.behavioral_failure else "invalid")
                report["screened"].append({"mutant": asdict(mutant), "outcome": outcome,
                                           "detail": result.feedback()})
                if result.passed:
                    survivors.append(mutant)
                elif frozen_targets is not None:
                    raise ValueError("Frozen target no longer survives the original suite")
                if len(survivors) >= limit:
                    break
        if screen_only:
            report["status"] = "screened"
        for index, mutant in enumerate([] if screen_only else survivors, 1):
            if budget.calls >= max_calls:
                report["status"] = "budget_exhausted"
                break
            with worktree(repo) as space:
                if input_hashes(space.root) != report["input_hashes"]:
                    raise ValueError("Repository input changed before generation")
                nearby = tests_touching(space.root, module)
                run = harden_mutant(space, env, mutant, budget, nearby,
                                    max_attempts=max_attempts, stability_runs=stability_runs,
                                    test_timeout=test_timeout, suite_timeout=suite_timeout)
                record = asdict(run)
                if run.status == "success":
                    content = run.accepted_test_content
                    patch = "".join(difflib.unified_diff(
                        [], content.splitlines(keepends=True),
                        fromfile="/dev/null", tofile="b/" + run.accepted_test_path))
                    if content and not content.endswith("\n"):
                        patch += "\n\\ No newline at end of file\n"
                    patch_name = f"target-{index:03d}.patch"
                    (directory / patch_name).write_text(patch)
                    record["patch_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
                    # Re-open the exported patch in a fresh workspace. Do not
                    # let generation's in-memory accepted flag grade the artifact.
                    with worktree(repo) as replay:
                        if input_hashes(replay.root) != report["input_hashes"]:
                            raise ValueError("Repository input changed before patch replay")
                        applied = subprocess.run(
                            ["git", "apply", str(directory / patch_name)], cwd=replay.root,
                            capture_output=True, text=True)
                        if applied.returncode:
                            record["replay"] = {"accepted": False, "error": applied.stderr}
                        else:
                            verdict = verify_candidate(
                                replay, env, run.accepted_test_path, mutant, [],
                                stability_runs=stability_runs, test_timeout=test_timeout,
                                suite_timeout=suite_timeout)
                            record["replay"] = asdict(verdict)
                    record["candidate_patch"] = patch_name
                    if record["replay"]["accepted"]:
                        record["patch"] = patch_name
                    else:
                        record["status"] = "error"
                        record["failure_class"] = "exported patch failed independent replay"
                report["targets"].append(record)
        if report["status"] == "running":
            report["status"] = "complete"
        report["accepted"] = sum(r["status"] == "success" for r in report["targets"])
        report["eligible_targets"] = len(survivors)
        return directory, report
    except Exception as error:
        report["status"] = "error"
        report["error"] = str(error)
        raise
    finally:
        report["model_calls"] = budget.calls
        report["trace"] = budget.trace
        report["usage_complete"] = all(t["usage_known"] for t in budget.trace)
        report["prompt_tokens"] = sum(t.get("prompt_tokens", 0) for t in budget.trace)
        report["completion_tokens"] = sum(t.get("completion_tokens", 0) for t in budget.trace)
        (directory / "strengthen.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n")
