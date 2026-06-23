"""Blind repository evaluation: generation never receives mutation feedback."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

from .agent import ProviderError, action_error
from .artifacts import new_run_directory
from .models import Mutant
from .mutation import discover, write_mutant
from .real_cases import FORBIDDEN, git
from .runner import run_pytest, with_upstream_collection
from .sandbox import IMAGE
from .strengthen import input_hashes
from .test_study import StudyBudget, coverage, digest, save
from .tools import ToolBox
from .verify import verify_candidate
from .workspace import hash_all, is_test_path, repository_files, resolve_inside, worktree

INVENTORY = (
    ("inflection", "https://github.com/jpvanhal/inflection.git"),
    ("python-stringcase", "https://github.com/okunishinishi/python-stringcase.git"),
    ("humps", "https://github.com/nficano/humps.git"),
)
CONTEXT_BYTES = 40000
PER_JOB_TOKENS = 80000
EXTERNAL_OPERATIONS = FORBIDDEN | {
    "urandom",
    "token_bytes",
    "token_hex",
    "token_urlsafe",
    "uuid1",
    "uuid4",
    "SystemRandom",
    "choice",
    "choices",
    "randint",
    "randrange",
    "getrandbits",
    "uniform",
    "shuffle",
    "sample",
    "seed",
    "perf_counter",
    "monotonic",
    "utcnow",
}
SYSTEM = """Write meaningful pytest tests for the named original Python function.
Find overlooked behavior using its source, documentation and existing tests.
Only create the designated new test file; never modify existing files or source.
Assert behavior through calls, not source inspection, bytecode, coverage artifacts,
monkeypatching test infrastructure, or test-runner manipulation. Do not guess
expected results. You may use the permitted tools to inspect and execute tests.
Return one JSON action: {"tool":"edit_test","args":{"path":"designated path",
"content":"complete Python test file"},"note":"reason"}.
Tools: read_file(path,start?,end?), search_code(query,path?), run_tests(path?),
edit_test(path,content), finish(reason). There is no shell or mutation tool.
After a submission, original-code tests are checked. If they pass and preserve
the existing suite, that submission is final. Hidden grading is not feedback.
"""


def function_nodes(source):
    """Structural scope, independent of generation or mutation outcomes."""
    tree = ast.parse(source)
    entropy_aliases = set()
    for imported in tree.body:
        if isinstance(imported, ast.Import):
            entropy_aliases.update(
                alias.asname or alias.name
                for alias in imported.names
                if alias.name.split(".")[0] in {"random", "secrets", "uuid"}
            )
        elif isinstance(imported, ast.ImportFrom):
            if imported.module in {"random", "secrets", "uuid"}:
                entropy_aliases.update(alias.asname or alias.name for alias in imported.names)
            elif imported.module == "os":
                entropy_aliases.update(
                    alias.asname or alias.name
                    for alias in imported.names
                    if alias.name == "urandom"
                )
    for node in sorted(
        (n for n in tree.body if isinstance(n, ast.FunctionDef)), key=lambda n: n.name
    ):
        if node.name.startswith("_"):
            continue
        text = "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
        reason = ""
        if node.decorator_list or any(
            isinstance(n, (ast.AsyncFunctionDef, ast.Yield, ast.YieldFrom, ast.ClassDef))
            or (isinstance(n, ast.FunctionDef) and n is not node)
            for n in ast.walk(node)
        ):
            reason = "decorated/nested/async/generator"
        elif sum(bool(s.strip()) for s in text.splitlines()) > 100:
            reason = "function line cap"
        elif any(
            (isinstance(n, ast.Name) and n.id in EXTERNAL_OPERATIONS | entropy_aliases)
            or (isinstance(n, ast.Attribute) and n.attr in EXTERNAL_OPERATIONS)
            for n in ast.walk(node)
        ):
            reason = "external-state operation"
        yield node, reason


def body_mutants(source, module, node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    executable = {
        n.lineno for statement in body for n in ast.walk(statement) if hasattr(n, "lineno")
    }
    return [m for m in discover(source, module) if m.line in executable][:8]


def initial_context(root, module, function, tests):
    """No mutant argument exists: context cannot encode the hidden transformation."""
    chunks = [
        f"Strengthen tests for {function} in {module}.",
        "Original source and existing upstream tests follow.",
        f"# {module}\n" + resolve_inside(root, module).read_text(),
    ]
    for path in sorted(tests):
        chunks.append(f"# {path}\n" + resolve_inside(root, path).read_text())
    text = "\n\n".join(chunks)
    # Leave explicit room for the identical system and designated-path contract.
    if len((SYSTEM + text).encode()) + 1000 > CONTEXT_BYTES:
        raise ValueError("complete initial context exceeds 40000-byte cap")
    return text


def relevant_tests(root, module, tests, function=None):
    """Select complete relevant files without inspecting generation outcomes."""
    stem = Path(module).stem.lstrip("_")
    names = (
        {function}
        if function
        else {n.name for n, _ in function_nodes(resolve_inside(root, module).read_text())}
    )
    selected = set()
    for relative in tests:
        if Path(relative).name == "conftest.py":
            continue
        match = stem in Path(relative).stem or bool(function and function in Path(relative).stem)
        try:
            tree = ast.parse(resolve_inside(root, relative).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    match |= any(a.name.rsplit(".", 1)[-1] == stem for a in node.names)
                if isinstance(node, ast.ImportFrom):
                    match |= bool(node.module and node.module.rsplit(".", 1)[-1] == stem)
                    match |= any(a.name == stem for a in node.names)
                    match |= any(a.name in names for a in node.names)
        except (OSError, SyntaxError):
            continue
        if match:
            selected.add(relative)
    for relative in tests:
        if Path(relative).name == "conftest.py" and any(
            Path(relative).parent in Path(p).parents for p in selected
        ):
            selected.add(relative)
    return sorted(selected)


def initial_request(task_id, context, arm):
    path = f"tests/test_branchproof_{task_id}.py"
    contract = f"\nCreate only {path}. Use edit_test with the complete Python file."
    system = SYSTEM + contract
    if arm == "direct":
        system += "\nDirect control: submit edit_test now; no tools or feedback are available."
    return path, system, [{"role": "user", "content": context + contract}]


def first_reservation(task_id, context):
    _, system, messages = initial_request(task_id, context, "direct")
    return len(json.dumps(messages).encode()) + len(system.encode()) + 4096 + 4096


def canonical_origin(url):
    """Equivalent public HTTPS GitHub URLs may omit the .git suffix."""
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError("Expected a public HTTPS GitHub origin")
    if parsed.query or parsed.fragment:
        raise ValueError("Unexpected origin query/fragment")
    return parsed.path.rstrip("/").removesuffix(".git").lower()


@with_upstream_collection
def prepare(clones: Path, output: Path, env: Path, *, inventory=INVENTORY, compact_context=False):
    if IMAGE.get() is None:
        raise ValueError("Docker sandbox is required")
    directory = new_run_directory(output)
    manifest = {
        "schema_version": 1,
        "protocol": "blind_repository_v1",
        "seed": 73,
        "sandbox_image": IMAGE.get(),
        "inventory": {},
        "cases": [],
        "exclusions": [],
        "screening": [],
        "ready": False,
        "compact_context": compact_context,
        "per_job_token_reservation": PER_JOB_TOKENS,
    }
    # Commit/url/hash/licence records are written before any Python source scan.
    for name, url in inventory:
        root = clones / name
        if git(root, "status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("Input clone is dirty: " + name)
        observed_origin = git(root, "remote", "get-url", "origin")
        if canonical_origin(observed_origin) != canonical_origin(url):
            raise ValueError("Input origin mismatch: " + name)
        manifest["inventory"][name] = {
            "url": url,
            "observed_origin": observed_origin,
            "revision": git(root, "rev-parse", "HEAD"),
            "input_hashes": input_hashes(root),
            "licences": {
                p.name: p.read_text()
                for p in root.iterdir()
                if p.is_file() and p.name.lower().startswith(("license", "copying"))
            },
        }
    save(directory / "inventory.json", manifest["inventory"])
    for name, _ in inventory:
        root = clones / name
        eligible = inspected = 0
        with worktree(root) as space:
            originals = [run_pytest(space.root, env, timeout=30) for _ in range(2)]
            manifest["inventory"][name]["baseline"] = [
                {"passed": r.passed, "counts": r.counts(), "detail": r.feedback()}
                for r in originals
            ]
            if not all(r.passed for r in originals):
                manifest["exclusions"].append(
                    {"repository": name, "reason": "original full suite not green twice"}
                )
                save(directory / "selection.partial.json", manifest)
                continue
            tests = [
                p.relative_to(space.root).as_posix()
                for p in repository_files(space.root)
                if p.suffix == ".py" and is_test_path(p.relative_to(space.root).as_posix())
            ]
            modules = sorted(
                p
                for p in repository_files(space.root)
                if p.suffix == ".py"
                and not is_test_path(p.relative_to(space.root).as_posix())
                and p.name not in {"setup.py", "conf.py", "__main__.py"}
                and not any(
                    s in {"docs", "examples", "build"} or s.endswith("_tests")
                    for s in p.relative_to(space.root).parts
                )
            )
            for path in modules:
                module = path.relative_to(space.root).as_posix()
                source = path.read_text()
                if compact_context and len(source.encode()) + len(SYSTEM.encode()) >= CONTEXT_BYTES:
                    manifest["exclusions"].append(
                        {
                            "repository": name,
                            "module": module,
                            "reason": "source module alone exceeds context cap",
                        }
                    )
                    continue
                try:
                    nodes = list(function_nodes(source))
                except SyntaxError:
                    continue
                if not nodes:
                    continue
                if not any(
                    not reason and body_mutants(source, module, node) for node, reason in nodes
                ):
                    manifest["exclusions"].append(
                        {
                            "repository": name,
                            "module": module,
                            "reason": "no in-scope operator-bearing functions",
                        }
                    )
                    continue
                module_tests = (
                    relevant_tests(space.root, module, tests) if compact_context else tests
                )
                if compact_context and not module_tests:
                    manifest["exclusions"].append(
                        {"repository": name, "module": module, "reason": "no relevant test module"}
                    )
                    continue
                cov = coverage(space.root, env, module, timeout=30)
                if not cov.get("available"):
                    manifest["exclusions"].append(
                        {"repository": name, "module": module, "reason": "coverage unavailable"}
                    )
                    continue
                for node, reason in nodes:
                    if inspected >= 60 or eligible >= 8 or len(manifest["cases"]) >= 20:
                        break
                    inspected += 1
                    identity = f"{name}:{module}:{node.name}"
                    if not reason:
                        try:
                            case_tests = (
                                relevant_tests(space.root, module, tests, node.name)
                                if compact_context
                                else module_tests
                            )
                            if not case_tests:
                                raise ValueError("no relevant test module for target function")
                            context = initial_context(space.root, module, node.name, case_tests)
                            if (
                                compact_context
                                and first_reservation("0" * 12, context) > PER_JOB_TOKENS
                            ):
                                raise ValueError(
                                    "initial context exceeds conservative token reservation"
                                )
                        except ValueError as error:
                            reason = str(error)
                    if reason:
                        manifest["exclusions"].append({"id": identity, "reason": reason})
                        continue
                    survivors = []
                    for mutant in body_mutants(source, module, node):
                        row = {"target": identity, "mutant": asdict(mutant)}
                        if mutant.line not in cov["executed_lines"]:
                            row["outcome"] = "uncovered"
                        else:
                            before = write_mutant(space.root, mutant)
                            try:
                                results = [run_pytest(space.root, env, timeout=30)]
                                if results[0].passed:
                                    results.append(run_pytest(space.root, env, timeout=30))
                                row["runs"] = [
                                    {
                                        "passed": r.passed,
                                        "behavioral_failure": r.behavioral_failure,
                                        "timed_out": r.timed_out,
                                    }
                                    for r in results
                                ]
                                if len(results) == 2 and all(r.passed for r in results):
                                    row["outcome"] = "survived_twice"
                                    survivors.append(asdict(mutant))
                                elif results[0].behavioral_failure:
                                    row["outcome"] = "killed"
                                else:
                                    row["outcome"] = "invalid_or_unstable"
                            finally:
                                path.write_text(before)
                        manifest["screening"].append(row)
                        save(directory / "selection.partial.json", manifest)
                    if not survivors:
                        manifest["exclusions"].append(
                            {"id": identity, "reason": "no covered survivor in cap"}
                        )
                        continue
                    task_id = hashlib.sha256(identity.encode()).hexdigest()[:12]
                    manifest["cases"].append(
                        {
                            "id": task_id,
                            "identity": identity,
                            "repository": name,
                            "module": module,
                            "function": node.name,
                            "tests": sorted(case_tests),
                            "context": context,
                            "context_bytes": len(context.encode()),
                            "context_token_estimate": math.ceil(len(context.encode()) / 3),
                            "mutants": survivors[:3],
                        }
                    )
                    eligible += 1
                    save(directory / "selection.partial.json", manifest)
                if inspected >= 60 or eligible >= 8 or len(manifest["cases"]) >= 20:
                    break
            manifest["inventory"][name]["inspected_functions"] = inspected
    cases = manifest["cases"]
    ordered = sorted(cases, key=lambda c: digest([73, c["id"]]))
    manifest["repeat_cases"] = [c["id"] for c in ordered[:3]]
    manifest["ready"] = len(cases) >= 6 and len({c["repository"] for c in cases}) >= 2
    manifest["sha256"] = digest(manifest)
    save(directory / "manifest.json", manifest)
    return directory, manifest


def original_checks(space, env, path):
    """Only this output is eligible for generation feedback; no mutants here."""
    before = hash_all(space.root)
    candidate = run_pytest(space.root, env, [path], timeout=30)
    suite = run_pytest(space.root, env, timeout=30) if candidate.passed else None
    intact = hash_all(space.root) == before and not space.violations()
    valid = candidate.passed and suite is not None and suite.passed and intact
    feedback = "Candidate on original code:\n" + candidate.feedback()
    if suite is not None:
        feedback += "\nExisting full suite:\n" + suite.feedback()
    if not intact:
        feedback += "\nRejected: execution or edits changed protected files."
    return valid, feedback


@with_upstream_collection
def generate(space, env, task, provider, arm):
    """Receives a PUBLIC task only. Hidden mutations are not accepted here."""
    if set(task) != {"id", "context"}:
        raise ValueError("Generation requires only public task id and context")
    if arm not in {"direct", "agent"}:
        raise ValueError("Unknown arm")
    path = f"tests/test_branchproof_{task['id']}.py"
    if resolve_inside(space.root, path).exists():
        raise ValueError("Designated test path already exists")
    _, system, messages = initial_request(task["id"], task["context"], arm)
    toolbox = ToolBox(space, env, test_timeout=30, protect_existing=True)
    record = {"status": "no_valid_candidate", "path": path, "actions": [], "submissions": 0}
    for _ in range(1 if arm == "direct" else 15):
        try:
            action, _, _ = provider.next_action(messages, system=system)
        except ProviderError as error:
            record.update(status=error.kind + "_error", usage_known=error.usage_known)
            break
        messages.append({"role": "assistant", "content": action.model_dump_json()})
        entry = {"action": action.model_dump()}
        record["actions"].append(entry)
        error = action_error(action)
        if error:
            feedback = "Rejected action: " + error
        elif action.tool == "finish":
            break
        elif action.tool == "edit_test":
            if action.args["path"] != path:
                feedback = "Refused: write only the designated new test file: " + path
            else:
                record["submissions"] += 1
                # Revising this one generated file is allowed; upstream files never are.
                space.write(path, action.args["content"])
                valid, feedback = original_checks(space, env, path)
                if valid:
                    record.update(status="valid_candidate", content=space.read(path))
                    entry["feedback"] = feedback
                    break
                if arm == "direct" or record["submissions"] >= 3:
                    entry["feedback"] = feedback
                    break
        elif arm == "agent" and action.tool in {"read_file", "search_code", "run_tests"}:
            feedback = getattr(toolbox, action.tool)(**action.args)
        else:
            feedback = "Refused: tool not permitted in this arm."
        entry["feedback"] = feedback
        messages.append({"role": "user", "content": feedback})
    record["messages"] = messages
    return record


@with_upstream_collection
def grade(repo, env, task, content):
    """Fresh replay, then hidden grading. Never called by generation."""
    result = {"valid": False, "detected": 0, "total": len(task["mutants"]), "mutations": []}
    path = f"tests/test_branchproof_{task['id']}.py"
    with worktree(repo) as space:
        space.write(path, content)
        checks = [original_checks(space, env, path)[0] for _ in range(2)]
        result["valid"] = all(checks)
        if not result["valid"]:
            return result
        for specification in task["mutants"]:
            verdict = verify_candidate(
                space,
                env,
                path,
                Mutant(**specification),
                [],
                stability_runs=2,
                test_timeout=30,
                suite_timeout=30,
            )
            result["mutations"].append(
                {
                    "id": specification["id"],
                    "detected": verdict.accepted,
                    "checks": [asdict(c) for c in verdict.checks],
                }
            )
            result["detected"] += int(verdict.accepted)
    return result


def schedule(manifest):
    jobs = [
        {"id": case["id"], "arm": arm, "repeat": repeat, "status": "pending"}
        for case in manifest["cases"]
        for arm in ("direct", "agent")
        for repeat in range(2 if case["id"] in manifest["repeat_cases"] else 1)
    ]
    random.Random(73).shuffle(jobs)
    jobs.sort(key=lambda job: job["repeat"])
    return jobs


def summarize(manifest, record):
    """Full scheduled denominators, including failed and pending jobs."""

    def key(job):
        return job["id"], job["arm"], job["repeat"]

    expected = {key(j) for j in schedule(manifest)}
    actual = [key(j) for j in record["jobs"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate schedule cannot shrink denominators")
    sizes = {c["id"]: len(c["mutants"]) for c in manifest["cases"]}
    result = {}
    for arm in ("direct", "agent"):
        primary = [j for j in record["jobs"] if j["arm"] == arm and j["repeat"] == 0]
        repeats = [j for j in record["jobs"] if j["arm"] == arm and j["repeat"] == 1]
        by_operator = {
            kind: {"scheduled": 0, "measured": 0, "detected": 0}
            for kind in ("comparison", "arithmetic", "boolean", "constant", "return")
        }
        for job in primary:
            case = next(c for c in manifest["cases"] if c["id"] == job["id"])
            specifications = {
                m.get("id", str(index)): m.get("kind", "unspecified")
                for index, m in enumerate(case["mutants"])
            }
            for kind in specifications.values():
                by_operator.setdefault(kind, {"scheduled": 0, "measured": 0, "detected": 0})[
                    "scheduled"
                ] += 1
            grading = job.get("grade", {})
            rows = grading.get("mutations", []) if grading.get("valid") else []
            ids = [row["id"] for row in rows]
            if len(ids) != len(set(ids)) or not set(ids) <= specifications.keys():
                raise ValueError("Unknown or duplicate graded mutation")
            for row in rows:
                counts = by_operator[specifications[row["id"]]]
                counts["measured"] += 1
                counts["detected"] += int(row["detected"])
        for counts in by_operator.values():
            counts["detection_rate"] = (
                counts["detected"] / counts["scheduled"] if counts["scheduled"] else None
            )
        result[arm] = {
            "primary_by_operator": by_operator,
            "headline_metric": "successful_patches / scheduled_targets",
            "scheduled_targets": len(primary),
            "attempted_targets": sum(j["status"] != "pending" for j in primary),
            "valid_patches": sum(j.get("grade", {}).get("valid", False) for j in primary),
            "targets_with_detection": sum(
                j.get("grade", {}).get("detected", 0) > 0 for j in primary
            ),
            "successful_patches": sum(
                j.get("grade", {}).get("valid", False) and j.get("grade", {}).get("detected", 0) > 0
                for j in primary
            ),
            "hidden_mutations_detected": sum(
                j.get("grade", {}).get("detected", 0) for j in primary
            ),
            "hidden_mutations_scheduled": sum(sizes[j["id"]] for j in primary),
            "primary_model_calls": sum(j.get("calls", 0) for j in primary),
            "primary_prompt_tokens": sum(j.get("prompt_tokens", 0) for j in primary),
            "primary_completion_tokens": sum(j.get("completion_tokens", 0) for j in primary),
            "repeat_jobs": len(repeats),
            "repeat_valid_patches": sum(j.get("grade", {}).get("valid", False) for j in repeats),
            "both_runs_detect_something": sum(
                j.get("grade", {}).get("detected", 0) > 0
                and any(
                    p["id"] == j["id"] and p.get("grade", {}).get("detected", 0) > 0
                    for p in primary
                )
                for j in repeats
            ),
        }
    return result


def validate_manifest(manifest, clones):
    if digest({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
        raise ValueError("Frozen manifest changed")
    if not manifest["ready"] or manifest["sandbox_image"] != IMAGE.get():
        raise ValueError("Dataset not ready or sandbox image mismatch")
    if manifest.get("per_job_token_reservation") != PER_JOB_TOKENS:
        raise ValueError("Frozen per-job token reservation changed")
    for case in manifest["cases"]:
        reservation = first_reservation(case["id"], case["context"])
        if reservation > PER_JOB_TOKENS:
            raise ValueError(
                "Initial context cannot fit conservative per-job reservation: " + case["id"]
            )
    for name, item in manifest["inventory"].items():
        root = clones / name
        if (
            git(root, "rev-parse", "HEAD") != item["revision"]
            or input_hashes(root) != item["input_hashes"]
            or git(root, "status", "--porcelain", "--untracked-files=normal")
        ):
            raise ValueError("Frozen input changed: " + name)


def run(manifest, clones, env, provider, output, profile):
    """Single frozen run; unknown usage stops scheduling, never automatic reruns."""
    validate_manifest(manifest, clones)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Study output already exists; no automatic paid rerun")
    if profile.get("reviewed_manifest_sha256") != manifest["sha256"]:
        raise ValueError("User-reviewed frozen manifest is required before generation")
    directory = new_run_directory(output)
    record = {
        "manifest_sha256": manifest["sha256"],
        "profile": profile,
        "status": "running",
        "jobs": schedule(manifest),
    }
    save(directory / "study.json", record)
    for index, job in enumerate(record["jobs"]):
        task = next(c for c in manifest["cases"] if c["id"] == job["id"])
        budget = StudyBudget(provider, 1 if job["arm"] == "direct" else 15, PER_JOB_TOKENS, 4096)
        job.update(status="in_flight")
        save(directory / "study.json", record)
        started = time.monotonic()
        try:
            with worktree(clones / task["repository"]) as space:
                generated = generate(
                    space, env, {"id": task["id"], "context": task["context"]}, budget, job["arm"]
                )
            save(directory / f"generation-{index:03d}.json", generated)
            job.update(
                status=generated["status"], generation_artifact=f"generation-{index:03d}.json"
            )
        except Exception as error:
            job.update(status="infrastructure_error", error_type=type(error).__name__)
        job.update(
            calls=budget.calls,
            trace=budget.trace,
            prompt_tokens=sum(t.get("prompt_tokens", 0) for t in budget.trace),
            completion_tokens=sum(t.get("completion_tokens", 0) for t in budget.trace),
            usage_complete=all(t["usage_known"] for t in budget.trace),
            generation_seconds=round(time.monotonic() - started, 3),
        )
        save(directory / "study.json", record)
        if not job["usage_complete"] or job["status"] == "infrastructure_error":
            record["status"] = "stopped_no_automatic_resume"
            break
    else:
        record["status"] = "generation_complete"
    # All generation is terminal before any hidden result is computed.
    save(directory / "study.json", record)
    for index, job in enumerate(record["jobs"]):
        if job["status"] != "valid_candidate":
            continue
        task = next(c for c in manifest["cases"] if c["id"] == job["id"])
        generated = json.loads((directory / f"generation-{index:03d}.json").read_text())
        job["grade"] = grade(clones / task["repository"], env, task, generated["content"])
        path = generated["path"]
        patch = "".join(
            difflib.unified_diff(
                [],
                generated["content"].splitlines(keepends=True),
                fromfile="/dev/null",
                tofile="b/" + path,
            )
        )
        (directory / f"patch-{index:03d}.diff").write_text(patch)
        save(directory / "study.json", record)
    record["summary"] = summarize(manifest, record)
    save(directory / "study.json", record)
    return directory, record
