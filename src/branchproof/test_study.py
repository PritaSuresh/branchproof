"""Frozen, execution-graded paired study of test generation. No model judge.

Preparation is model-free. Generation and additional-mutation scoring are
separate phases so withheld execution outcomes cannot enter model feedback.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

from .agent import SYSTEM_PROMPT, Provider, ProviderError
from .artifacts import new_run_directory
from .models import Mutant
from .mutation import discover_in_file, write_mutant
from .runner import run_pytest
from .sandbox import IMAGE
from .strengthen import CallBudget, input_hashes, strengthen
from .workspace import hash_all, resolve_inside, worktree

ARMS = {"initial": (1, 5), "feedback": (3, 15)}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def function_key(source: str, line: int) -> str:
    nodes = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        and n.lineno <= line <= n.end_lineno
    ]
    node = min(nodes, key=lambda n: n.end_lineno - n.lineno) if nodes else None
    return f"{node.lineno}:{node.name}" if node else "0:module"


def coverage(
    root: Path, env: Path, module: str, targets: list[str] | None = None, *, timeout: int = 300
) -> dict:
    """Measure module line/branch counts; absent reports are missing, never zero."""
    with tempfile.TemporaryDirectory(prefix="hardener-coverage-") as temporary:
        path = (
            Path("/tmp/hardener-coverage.json")
            if IMAGE.get()
            else Path(temporary) / "coverage.json"
        )
        result = run_pytest(
            root,
            env,
            targets,
            timeout=timeout,
            extra=[
                "--cov=" + str(Path(module).parent),
                "--cov-branch",
                "--cov-report=json:" + str(path),
            ],
        )
        if not result.passed:
            return {"available": False, "error": result.feedback()}
        try:
            report = result.coverage_report if IMAGE.get() else json.loads(path.read_text())
            matches = [
                v
                for k, v in report["files"].items()
                if k == module or k.replace("\\", "/").endswith("/" + module)
            ]
            if len(matches) != 1:
                raise ValueError("Coverage report did not uniquely identify the source module")
            return {
                "available": True,
                "executed_lines": matches[0]["executed_lines"],
                **{
                    k: matches[0]["summary"][k]
                    for k in ("covered_lines", "num_statements", "covered_branches", "num_branches")
                },
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {"available": False, "error": str(error)}


def prepare(
    repositories: dict[str, Path],
    modules: list[tuple[str, str]],
    env: Path,
    output: Path,
    *,
    candidate_limit: int = 20,
    per_module: int = 2,
    additional_limit: int = 2,
    seed: int = 17,
    development: bool = True,
    screen_timeout: int = 30,
) -> tuple[Path, dict]:
    if not modules or len(set(modules)) != len(modules):
        raise ValueError("Supply distinct NAME:module.py entries")
    if min(candidate_limit, per_module, screen_timeout) < 1 or additional_limit < 0:
        raise ValueError("Invalid preparation limits")
    if any(name not in repositories for name, _ in modules):
        raise ValueError("Unknown repository in module inventory")
    output = output.resolve()
    if any(output == p.resolve() or p.resolve() in output.parents for p in repositories.values()):
        raise ValueError("Study output must be outside every input repository")
    directory = new_run_directory(output)
    manifest = {
        "schema_version": 1,
        "purpose": "development" if development else "evaluation",
        "seed": seed,
        "sandbox_image": IMAGE.get(),
        "cases": [],
        "modules": [],
        "settings": {
            "candidate_limit": candidate_limit,
            "per_module": per_module,
            "additional_limit": additional_limit,
            "screen_timeout": screen_timeout,
        },
        "selection": "Declared module order; alternate whole-function pools in "
        "source order before generation; first eligible mutations per pool",
    }
    for name, module in modules:
        entry = {"repository": name, "module": module}
        manifest["modules"].append(entry)
        try:
            artifact, screen = strengthen(
                repositories[name],
                env,
                module,
                None,
                directory / "screening",
                screen_only=True,
                limit=candidate_limit,
                candidate_limit=candidate_limit,
                screen_timeout=screen_timeout,
            )
            entry.update(
                revision=screen["revision"],
                input_hashes=screen["input_hashes"],
                evidence=str(artifact.relative_to(directory)),
                baseline=screen["baseline"],
                screened=screen["screened"],
            )
            entry["screen_timeout"] = screen_timeout
            with worktree(repositories[name]) as space:
                source = resolve_inside(space.root, module).read_text()
                candidates = discover_in_file(space.root, module)[:candidate_limit]
                groups = sorted(
                    {function_key(source, m.line) for m in candidates},
                    key=lambda key: int(key.split(":", 1)[0]),
                )
                shown = set(groups[::2])
                survivors = [r["mutant"] for r in screen["screened"] if r["outcome"] == "survived"]
                primary = [m for m in survivors if function_key(source, m["line"]) in shown]
                extra = [m for m in survivors if function_key(source, m["line"]) not in shown]
                entry["coverage"] = coverage(space.root, env, module)
            entry["shown_functions"] = sorted(shown)
            entry["additional_mutants"] = extra[:additional_limit]
            entry["eligible"] = bool(primary) and entry["coverage"]["available"]
            for mutant in primary[:per_module] if entry["eligible"] else []:
                manifest["cases"].append(
                    {
                        "id": name + "@" + mutant["id"],
                        "repository": name,
                        "module": module,
                        "revision": screen["revision"],
                        "mutant": mutant,
                    }
                )
            if not primary:
                entry["exclusion"] = "No original-suite survivor in the shown-function pool"
            elif not entry["coverage"]["available"]:
                entry["exclusion"] = "Baseline coverage unavailable in this execution image"
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            entry.update(eligible=False, exclusion=str(error))
        save(directory / "preparation.json", manifest)
    manifest["repeat_cases"] = sorted(
        (c["id"] for c in manifest["cases"]), key=lambda key: digest([seed, key])
    )[:12]
    manifest["sha256"] = digest(manifest)
    save(directory / "manifest.json", manifest)
    return directory, manifest


def combine(paths: list[Path], repositories: dict[str, Path], output: Path) -> tuple[Path, dict]:
    """Union complete model-free preparations, retaining exclusions and provenance."""
    if not paths:
        raise ValueError("Supply at least one preparation manifest")
    parts = [json.loads(p.read_text()) for p in paths]
    for part in parts:
        validate(part, repositories)
    if len({(p["seed"], p["purpose"]) for p in parts}) != 1:
        raise ValueError("Preparations must share seed and development/evaluation purpose")
    keys = [(e["repository"], e["module"]) for p in parts for e in p["modules"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Cannot merge duplicate modules or select among competing preparations")
    output = output.resolve()
    if any(output == p.resolve() or p.resolve() in output.parents for p in repositories.values()):
        raise ValueError("Study output must be outside every input repository")
    directory = new_run_directory(output)
    result = {
        "schema_version": 1,
        "purpose": parts[0]["purpose"],
        "seed": parts[0]["seed"],
        "sandbox_image": parts[0]["sandbox_image"],
        "source_preparations": [
            {"sha256": p["sha256"], "path": str(path.resolve())}
            for path, p in zip(paths, parts, strict=True)
        ],
        "settings": {"parts": [p["settings"] for p in parts]},
        "selection": "Union of declared preparations before generation; no target filtering",
        "modules": [e for p in parts for e in p["modules"]],
        "cases": [c for p in parts for c in p["cases"]],
    }
    result["repeat_cases"] = sorted(
        (c["id"] for c in result["cases"]), key=lambda key: digest([result["seed"], key])
    )[:12]
    result["sha256"] = digest(result)
    save(directory / "manifest.json", result)
    return directory, result


def validate(manifest: dict, repositories: dict[str, Path]) -> None:
    content = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest.get("sha256") != digest(content):
        raise ValueError("Manifest fingerprint mismatch")
    if not manifest["cases"] or len({c["id"] for c in manifest["cases"]}) != len(manifest["cases"]):
        raise ValueError("Manifest requires distinct eligible targets")
    if manifest["sandbox_image"] != IMAGE.get():
        raise ValueError("Use the same pinned execution image as preparation")
    keys = {(e["repository"], e["module"]) for e in manifest["modules"] if e.get("eligible")}
    if any((c["repository"], c["module"]) not in keys for c in manifest["cases"]):
        raise ValueError("Target refers to an ineligible or absent module")
    for entry in manifest["modules"]:
        if not entry.get("eligible"):
            continue
        if not entry.get("coverage", {}).get("available"):
            raise ValueError("Baseline coverage must be available before generation")
        repo = repositories[entry["repository"]]
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        if status.strip() or revision != entry["revision"]:
            raise ValueError("Repository revision or clean state changed")
        with worktree(repo) as space:
            if input_hashes(space.root) != entry["input_hashes"]:
                raise ValueError("Frozen source/test/configuration hashes changed")
            available = discover_in_file(space.root, entry["module"])
            primary = [
                c
                for c in manifest["cases"]
                if c["repository"] == entry["repository"] and c["module"] == entry["module"]
            ]
            source = resolve_inside(space.root, entry["module"]).read_text()
            shown_functions = {function_key(source, c["mutant"]["line"]) for c in primary}
            for case in primary:
                if case["revision"] != revision or Mutant(**case["mutant"]) not in available:
                    raise ValueError("Frozen target is not a source mutation at this revision")
            for extra in entry["additional_mutants"]:
                if Mutant(**extra) not in available or function_key(
                    source, extra["line"]
                ) in shown_functions | set(entry["shown_functions"]):
                    raise ValueError("Additional mutations overlap shown functions or changed")


class StudyBudget(CallBudget):
    """Conservative pre-call reservation; not a provider billing guarantee."""

    def __init__(self, provider: Provider, calls: int, tokens: int, completion_cap: int):
        super().__init__(provider, calls)
        if min(calls, tokens, completion_cap) < 1:
            raise ValueError("Positive call/token/completion ceilings required")
        self.tokens = tokens
        self.completion_cap = completion_cap
        self.blocked = False

    def next_action(self, messages, system=None):
        known = sum(t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in self.trace)
        reservation = (
            len(json.dumps(messages).encode())
            + len((system or SYSTEM_PROMPT).encode())
            + self.completion_cap
            + 4096
        )
        if any(not t["usage_known"] for t in self.trace) or known + reservation > self.tokens:
            self.blocked = True
            raise ProviderError(
                "Study stopped: token reservation exhausted or usage unknown",
                usage_known=True,
                kind=(
                    "usage_unknown"
                    if any(not t["usage_known"] for t in self.trace)
                    else "token_budget"
                ),
            )
        return super().next_action(messages, system)


def summary(rows: list[dict]) -> dict:
    result = {}
    for arm in ARMS:
        primary = [r for r in rows if r["arm"] == arm and r["repeat"] == 0]
        counts = {
            "eligible": len(primary),
            "accepted": sum(r.get("accepted", False) for r in primary),
            "attempted": sum(r["status"] != "pending" for r in primary),
            "model_calls": sum(r.get("model_calls", 0) for r in primary),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in primary),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in primary),
        }
        counts["accepted_pct"] = (
            round(100 * counts["accepted"] / len(primary), 1) if primary else None
        )
        counts["first_proposal_accepted"] = sum(
            r.get("first_proposal_accepted", False) and r.get("accepted", False) for r in primary
        )
        repeated = [r for r in rows if r["arm"] == arm and r["repeat"] > 0]
        counts["repeat_jobs"] = len(repeated)
        counts["repeat_accepted"] = sum(r.get("accepted", False) for r in repeated)
        counts["repeat_model_calls"] = sum(r.get("model_calls", 0) for r in repeated)
        counts["seconds"] = round(sum(r.get("seconds", 0) for r in primary), 3)
        counts["known_tokens_per_accepted"] = (
            round((counts["prompt_tokens"] + counts["completion_tokens"]) / counts["accepted"])
            if counts["accepted"]
            else None
        )
        counts["usage_complete"] = all(r.get("usage_complete", True) for r in primary)
        counts["failure_classes"] = dict(
            Counter(r.get("failure_class") or r["status"] for r in primary if not r.get("accepted"))
        )
        counts["by_module"] = {}
        counts["by_repository"] = {}
        for row in primary:
            key = row["repository"] + ":" + row["module"]
            group = counts["by_module"].setdefault(key, {"eligible": 0, "accepted": 0})
            group["eligible"] += 1
            group["accepted"] += bool(row.get("accepted"))
            repo = counts["by_repository"].setdefault(
                row["repository"], {"eligible": 0, "accepted": 0}
            )
            repo["eligible"] += 1
            repo["accepted"] += bool(row.get("accepted"))
        result[arm] = counts
    return result


def validate_jobs(manifest: dict, record: dict) -> None:
    """A partial/truncated result must not quietly shrink the denominator."""
    extra = record["limits"]["extra_repeats"]
    if extra not in {0, 2}:
        raise ValueError("Unexpected repeat schedule")
    arms = record["limits"].get("arms", list(ARMS))
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("Invalid declared study arms")
    expected = {
        (case["id"], arm, repeat)
        for case in manifest["cases"]
        for arm in arms
        for repeat in (range(1 + extra) if case["id"] in manifest["repeat_cases"] else range(1))
    }
    actual = {(r["id"], r["arm"], r["repeat"]) for r in record["rows"]}
    if actual != expected or len(actual) != len(record["rows"]):
        raise ValueError("Study rows differ from the complete frozen job schedule")


def run(
    manifest: dict,
    repositories: dict[str, Path],
    env: Path,
    provider: Provider,
    output: Path,
    *,
    max_total_calls: int,
    max_total_tokens: int,
    completion_cap: int,
    provider_metadata: dict,
    extra_repeats: int = 0,
    arms: tuple[str, ...] = ("initial", "feedback"),
    per_job_tokens: int | None = None,
) -> tuple[Path, dict]:
    validate(manifest, repositories)
    if extra_repeats not in {0, 2}:
        raise ValueError("Use zero or the predeclared two additional repeats")
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("Invalid declared study arms")
    if per_job_tokens is not None and per_job_tokens < 1:
        raise ValueError("Positive per-job token allowance required")
    job_count = sum(
        1 + (extra_repeats if c["id"] in manifest["repeat_cases"] else 0) for c in manifest["cases"]
    ) * len(arms)
    if per_job_tokens is not None and job_count * per_job_tokens > max_total_tokens:
        raise ValueError("Global tokens must reserve every scheduled job's allowance")
    budget = StudyBudget(provider, max_total_calls, max_total_tokens, completion_cap)
    output = output.resolve()
    if any(output == p.resolve() or p.resolve() in output.parents for p in repositories.values()):
        raise ValueError("Study output must be outside every input repository")
    directory = new_run_directory(output)
    rows = []
    for case in manifest["cases"]:
        repeats = range(1 + extra_repeats) if case["id"] in manifest["repeat_cases"] else range(1)
        for repeat in repeats:
            for arm in arms:
                rows.append(
                    {
                        "id": case["id"],
                        "repository": case["repository"],
                        "module": case["module"],
                        "arm": arm,
                        "repeat": repeat,
                        "status": "pending",
                        "accepted": False,
                    }
                )
    random.Random(manifest["seed"]).shuffle(rows)
    rows.sort(key=lambda row: row["repeat"] > 0)  # primary measurement before extra spending
    record = {
        "schema_version": 1,
        "manifest_sha256": manifest["sha256"],
        "provider": provider_metadata,
        "rows": rows,
        "status": "running",
        "limits": {
            "calls": max_total_calls,
            "tokens": max_total_tokens,
            "completion_cap": completion_cap,
            "extra_repeats": extra_repeats,
            "arms": list(arms),
            "per_job_tokens": per_job_tokens,
        },
        "scope": "Shown-mutant test generation; unequal-cost arms, not real-bug accuracy",
    }
    record["transport_usage_scope"] = (
        "Reported decision usage; SDK transport retries can have additional unreported billing"
    )

    def checkpoint():
        record["summary"] = summary(rows)
        record["model_calls"] = budget.calls
        record["usage_complete"] = all(t["usage_known"] for t in budget.trace)
        record["prompt_tokens"] = sum(t.get("prompt_tokens", 0) for t in budget.trace)
        record["completion_tokens"] = sum(t.get("completion_tokens", 0) for t in budget.trace)
        save(directory / "study.json", record)

    checkpoint()
    for index, row in enumerate(rows):
        if budget.calls >= max_total_calls or budget.blocked:
            record["status"] = "budget_stopped"
            break
        case = next(c for c in manifest["cases"] if c["id"] == row["id"])
        entry = next(
            e
            for e in manifest["modules"]
            if e["repository"] == case["repository"] and e["module"] == case["module"]
        )
        attempts, calls = ARMS[row["arm"]]
        row["status"] = "in_flight"
        checkpoint()  # interruption stays visible; never silently replay a billed job
        started = time.monotonic()
        before = budget.calls
        trace_start = len(budget.trace)
        job_budget = (
            StudyBudget(
                budget, min(calls, max_total_calls - budget.calls), per_job_tokens, completion_cap
            )
            if per_job_tokens is not None
            else budget
        )
        try:
            artifact, result = strengthen(
                repositories[case["repository"]],
                env,
                case["module"],
                job_budget,
                directory / f"job-{index:04d}",
                limit=1,
                candidate_limit=1,
                max_attempts=attempts,
                max_calls=min(calls, max_total_calls - budget.calls),
                frozen_targets=[Mutant(**case["mutant"])],
                provider_metadata=provider_metadata,
                expected_revision=case["revision"],
                expected_hashes=entry["input_hashes"],
                screen_timeout=entry.get("screen_timeout", 300),
            )
            row.update(
                status="complete",
                accepted=result["accepted"] == 1,
                artifact=str(artifact.relative_to(directory)),
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                usage_complete=result["usage_complete"],
                failure_class=(
                    result["targets"][0]["failure_class"] if result["targets"] else "no candidate"
                ),
                patch_sha256=(
                    result["targets"][0].get("patch_sha256") if result["targets"] else None
                ),
                first_proposal_accepted=bool(
                    result["accepted"]
                    and result["targets"]
                    and result["targets"][0]["attempts"]
                    and result["targets"][0]["attempts"][0]["accepted"]
                ),
            )
            row["job_token_budget_stopped"] = bool(
                per_job_tokens is not None and job_budget.blocked
            )
        except Exception as error:
            row.update(status="error", error=str(error))
            record["status"] = "infrastructure_stopped"
        finally:
            row["model_calls"] = budget.calls - before
            trace = budget.trace[trace_start:]
            row["prompt_tokens"] = sum(t.get("prompt_tokens", 0) for t in trace)
            row["completion_tokens"] = sum(t.get("completion_tokens", 0) for t in trace)
            row["usage_complete"] = all(t["usage_known"] for t in trace)
            row["seconds"] = round(time.monotonic() - started, 3)
            checkpoint()
        if not record["usage_complete"]:
            budget.blocked = True
        if row["status"] == "error":
            break
    if record["status"] == "running":
        record["status"] = (
            "complete" if all(r["status"] != "pending" for r in rows) else "budget_stopped"
        )
    checkpoint()
    return directory, record


def score(manifest: dict, repositories: dict[str, Path], env: Path, study: Path) -> dict:
    """Replay accepted exports and grade withheld mutations after all generation."""
    validate(manifest, repositories)
    record = json.loads((study / "study.json").read_text())
    if record["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Study/manifest mismatch")
    validate_jobs(manifest, record)
    if any(r["status"] == "in_flight" for r in record["rows"]):
        raise ValueError("Interrupted generation requires an accounting audit before scoring")
    grading = new_run_directory(study / "grading")
    from .verify import verify_candidate

    grades = []
    for row in record["rows"]:
        grade = {
            "id": row["id"],
            "arm": row["arm"],
            "repeat": row["repeat"],
            "accepted": False,
            "additional": [],
        }
        grades.append(grade)
        if not row.get("accepted"):
            continue
        case = next(c for c in manifest["cases"] if c["id"] == row["id"])
        entry = next(
            e
            for e in manifest["modules"]
            if e["repository"] == case["repository"] and e["module"] == case["module"]
        )
        phase = "primary"
        try:
            artifact = resolve_inside(study, row["artifact"])
            result = json.loads((artifact / "strengthen.json").read_text())
            target = result["targets"][0]
            patch = resolve_inside(artifact, target["patch"])
            patch_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
            if patch_hash != row["patch_sha256"] or target["mutant"] != case["mutant"]:
                raise ValueError("Exported patch or target differs from generation record")
            with worktree(repositories[case["repository"]]) as space:
                before = hash_all(space.root)
                subprocess.run(
                    ["git", "apply", str(patch)], cwd=space.root, check=True, capture_output=True
                )
                after = hash_all(space.root)
                changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
                if (
                    changed != {target["accepted_test_path"]}
                    or target["accepted_test_path"] in before
                ):
                    raise ValueError("Patch must add exactly one new test and change nothing else")
                verdict = verify_candidate(
                    space, env, target["accepted_test_path"], Mutant(**case["mutant"]), []
                )
                grade.update(
                    accepted=verdict.accepted, patch_sha256=patch_hash, reason=verdict.reason
                )
                if verdict.accepted:
                    candidate_coverage = coverage(
                        space.root, env, case["module"], [target["accepted_test_path"]]
                    )
                    grade["candidate_coverage"] = candidate_coverage
                    if (
                        not candidate_coverage["available"]
                        or case["mutant"]["line"] not in candidate_coverage["executed_lines"]
                    ):
                        grade.update(
                            accepted=False, reason="Candidate target execution not established"
                        )
                        continue
                    grade["coverage_before"] = entry["coverage"]
                    grade["coverage_after"] = coverage(space.root, env, case["module"])
            if not grade["accepted"]:
                continue
            phase = "additional"
            for extra in entry["additional_mutants"]:
                with worktree(repositories[case["repository"]]) as space:
                    subprocess.run(
                        ["git", "apply", str(patch)],
                        cwd=space.root,
                        check=True,
                        capture_output=True,
                    )
                    write_mutant(space.root, Mutant(**extra))
                    before_hashes = hash_all(space.root)
                    execution = run_pytest(space.root, env, [target["accepted_test_path"]])
                    valid = hash_all(space.root) == before_hashes
                    outcome = (
                        "killed"
                        if execution.behavioral_failure and valid
                        else "survived"
                        if execution.passed and valid
                        else "invalid"
                    )
                    grade["additional"].append({"id": extra["id"], "outcome": outcome})
        except Exception as error:
            if phase == "primary":
                grade.update(accepted=False, error=str(error))
            else:
                grade["additional_error"] = str(error)
        save(grading / "grades.partial.json", {"grades": grades})
    rows = [
        {
            **row,
            "accepted": grade["accepted"],
            "failure_class": (grade.get("error") or grade.get("reason") or row.get("failure_class"))
            if not grade["accepted"]
            else "",
        }
        for row, grade in zip(record["rows"], grades, strict=True)
    ]
    graded = {
        "manifest_sha256": manifest["sha256"],
        "grades": grades,
        "summary": summary(rows),
        "scope": record["scope"],
        "generation_status": record["status"],
        "grading_directory": str(grading),
    }
    pool = {
        (e["repository"], m["id"])
        for e in manifest["modules"]
        if e.get("eligible")
        for m in e["additional_mutants"]
    }
    for arm in ARMS:
        primary = [g for g in grades if g["arm"] == arm and g["repeat"] == 0]
        killed = {
            (next(c["repository"] for c in manifest["cases"] if c["id"] == g["id"]), m["id"])
            for g in primary
            for m in g["additional"]
            if m["outcome"] == "killed"
        }
        data = graded["summary"][arm]
        data.update(
            additional_unique_available=len(pool),
            additional_unique_killed=len(killed),
            additional_invalid_runs=sum(
                m["outcome"] == "invalid" for g in primary for m in g["additional"]
            ),
            additional_measurement_errors=sum("additional_error" in g for g in primary),
        )
        for key, label in (("covered_lines", "line"), ("covered_branches", "branch")):
            gains = [
                g["coverage_after"][key] - g["coverage_before"][key]
                for g in primary
                if g.get("coverage_after", {}).get("available")
                and g.get("coverage_before", {}).get("available")
            ]
            data["coverage_measured_accepted_patches"] = len(gains)
            data[f"median_{label}_gain_per_accepted_patch"] = (
                statistics.median(gains) if gains else None
            )
    save(grading / "grades.json", graded)
    return graded
