"""Bounded, model-free extraction of attributed upstream utility functions."""

from __future__ import annotations

import ast
import builtins
import doctest
import hashlib
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

from .artifacts import new_run_directory
from .mutation import discover, write_mutant
from .runner import run_pytest
from .sandbox import IMAGE
from .test_study import coverage, digest, save
from .workspace import resolve_inside, worktree

STANDARD = {
    "math",
    "re",
    "datetime",
    "decimal",
    "fractions",
    "itertools",
    "functools",
    "operator",
    "string",
    "collections",
    "bisect",
    "statistics",
    "numbers",
    "typing",
}
FORBIDDEN = {
    "open",
    "input",
    "print",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "read",
    "write",
    "now",
    "today",
    "random",
    "time",
    "sleep",
    "getenv",
    "environ",
    "system",
    "popen",
    "exit",
}
EXCLUDED = {
    "boltons/" + name + ".py"
    for name in ("mathutils", "strutils", "iterutils", "dictutils", "setutils", "queueutils")
}
EXCLUDED |= {
    "tomlkit/" + name + ".py"
    for name in ("api", "items", "container", "parser", "source", "_utils")
}


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def extract(tree, node, text):
    if node.decorator_list or any(
        isinstance(n, (ast.AsyncFunctionDef, ast.Yield, ast.YieldFrom, ast.ClassDef))
        for n in ast.walk(node)
    ):
        raise ValueError("decorated, asynchronous, generator or nested class")
    if any(isinstance(n, ast.FunctionDef) and n is not node for n in ast.walk(node)):
        raise ValueError("nested function")
    source = "\n".join(text.splitlines()[node.lineno - 1 : node.end_lineno]) + "\n"
    if sum(bool(line.strip()) for line in source.splitlines()) > 100:
        raise ValueError("source line cap")
    examples = doctest.DocTestParser().get_examples(ast.get_docstring(node) or "")
    expected = [i for i, e in enumerate(examples) if e.want]
    if len(expected) < 2:
        raise ValueError("fewer than two upstream expected-output examples")
    imports = {}
    for item in tree.body:
        if isinstance(item, ast.Import):
            for alias in item.names:
                if alias.name.split(".")[0] in STANDARD:
                    imports[alias.asname or alias.name.split(".")[0]] = ast.unparse(
                        ast.Import(names=[alias])
                    )
        elif (
            isinstance(item, ast.ImportFrom)
            and item.module
            and item.module.split(".")[0] in STANDARD
        ):
            for alias in item.names:
                if alias.name != "*":
                    imports[alias.asname or alias.name] = ast.unparse(
                        ast.ImportFrom(module=item.module, names=[alias], level=0)
                    )
    loaded = {
        n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    bound = {n.arg for n in ast.walk(node) if isinstance(n, ast.arg)} | {node.name}
    bound |= {
        n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    unknown = loaded - bound - (set(dir(builtins)) - FORBIDDEN) - imports.keys()
    if unknown or loaded & FORBIDDEN:
        raise ValueError("non-self-contained or forbidden names: " + ",".join(sorted(unknown)))
    if any(isinstance(n, ast.Attribute) and n.attr in FORBIDDEN for n in ast.walk(node)):
        raise ValueError("external-state attribute")
    for example in examples:
        for item in ast.walk(ast.parse(example.source)):
            if isinstance(item, ast.Name) and item.id in FORBIDDEN:
                raise ValueError("unsafe doctest name")
            if isinstance(item, ast.Attribute) and item.attr in FORBIDDEN:
                raise ValueError("unsafe doctest attribute")
            if isinstance(item, ast.Import) and any(
                a.name.split(".")[0] not in STANDARD for a in item.names
            ):
                raise ValueError("unsupported doctest import")
            if isinstance(item, ast.ImportFrom) and (
                not item.module or item.module.split(".")[0] not in STANDARD
            ):
                raise ValueError("unsupported doctest import")
    prefix = "from __future__ import annotations\n"
    prefix += "\n".join(sorted({imports[n] for n in loaded & imports.keys()})) + "\n\n"
    return prefix + source, expected[0] + 1, source


def test_source(module, function, count=None):
    selection = "examples" if count is None else f"examples[:{count}]"
    return (
        f"import doctest\nimport {module} as subject\n\n"
        "def test_upstream_examples():\n"
        f"    examples = doctest.DocTestParser().get_examples(subject.{function}.__doc__)\n"
        f"    test = doctest.DocTest({selection}, subject.__dict__, 'upstream', __file__, 0, '')\n"
        "    result = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS).run(test)\n"
        "    assert result.attempted > 0 and result.failed == 0\n"
    )


def initialise(root):
    for args in (
        ("init", "-q"),
        ("add", "."),
        (
            "-c",
            "user.name=BranchProof",
            "-c",
            "user.email=local@example.invalid",
            "commit",
            "-qm",
            "Pinned upstream fixture",
        ),
    ):
        subprocess.run(["git", *args], cwd=root, capture_output=True, check=True)


def prepare(
    clones: Path,
    output: Path,
    env: Path,
    repository_names=("boltons", "humanize", "python-slugify", "tomlkit"),
):
    directory = new_run_directory(output)
    inventories = {}
    roots = {}
    for name in sorted(repository_names):
        root = clones / name
        if git(root, "status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("Selection requires clean clones: " + name)
        roots[name] = root
        licences = sorted(
            p
            for p in root.iterdir()
            if p.is_file() and p.name.lower().startswith(("license", "copying"))
        )
        inventories[name] = {
            "revision": git(root, "rev-parse", "HEAD"),
            "url": git(root, "remote", "get-url", "origin"),
            "licences": {p.name: p.read_text() for p in licences},
        }
    save(directory / "inventory.json", inventories)  # before any function scan
    inspected = executed = 0
    eligible = []
    exclusions = []
    stop = False
    for name, root in roots.items():
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if (
                relative in EXCLUDED
                or any(
                    part in {".git", "tests", "test", "docs", "examples"}
                    for part in path.relative_to(root).parts
                )
                or path.name.startswith("test")
            ):
                continue
            text = path.read_text(errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in sorted(
                (n for n in tree.body if isinstance(n, ast.FunctionDef)), key=lambda n: n.name
            ):
                if inspected >= 200 or executed >= 30 or len(eligible) >= 12:
                    stop = True
                    break
                inspected += 1
                identity = f"{name}:{relative}:{node.name}"
                try:
                    source, smoke_count, original = extract(tree, node, text)
                except (ValueError, SyntaxError) as error:
                    exclusions.append({"id": identity, "reason": str(error)})
                    continue
                executed += 1
                probe = directory / f"probe-{executed:03d}"
                (probe / "tests").mkdir(parents=True)
                (probe / "upstream.py").write_text(source)
                (probe / "tests/test_smoke.py").write_text(
                    test_source("upstream", node.name, smoke_count)
                )
                initialise(probe)
                evidence = {"id": identity, "survivors": [], "screened": []}
                try:
                    with worktree(probe) as space:
                        if not run_pytest(space.root, env, timeout=10).passed:
                            raise ValueError("extracted smoke does not pass")
                        reference = test_source("upstream", node.name)
                        reference_path = space.root / "tests/test_reference.py"
                        reference_path.write_text(reference)
                        good = run_pytest(space.root, env, ["tests/test_reference.py"], timeout=10)
                        reference_path.unlink()
                        if not good.passed:
                            raise ValueError("upstream reference examples do not all pass")
                        cov = coverage(space.root, env, "upstream.py")
                        if not cov.get("available"):
                            raise ValueError("baseline coverage unavailable")
                        for mutant in discover(source, "upstream.py")[:8]:
                            if mutant.line not in cov["executed_lines"]:
                                evidence["screened"].append(
                                    {"id": mutant.id, "outcome": "uncovered"}
                                )
                                continue
                            saved = write_mutant(space.root, mutant)
                            try:
                                smoke = run_pytest(space.root, env, timeout=10)
                                status = (
                                    "survived"
                                    if smoke.passed
                                    else ("killed" if smoke.behavioral_failure else "invalid")
                                )
                                if smoke.passed:
                                    reference_path.write_text(reference)
                                    result = run_pytest(
                                        space.root, env, ["tests/test_reference.py"], timeout=10
                                    )
                                    reference_path.unlink()
                                    if result.behavioral_failure:
                                        evidence["survivors"].append(asdict(mutant))
                                    else:
                                        status = "no calibrated behavioral distinction"
                                evidence["screened"].append({"id": mutant.id, "outcome": status})
                            finally:
                                reference_path.unlink(missing_ok=True)
                                resolve_inside(space.root, mutant.path).write_text(saved)
                    survivors = evidence["survivors"]
                    if not survivors:
                        raise ValueError("no calibrated covered survivor in first eight candidates")
                    shown = survivors[0]
                    withheld = next(
                        (
                            m
                            for m in survivors[1:]
                            if (m["line"], m["col"]) != (shown["line"], shown["col"])
                        ),
                        None,
                    )
                    if withheld is None:
                        raise ValueError("no distinct-site withheld survivor in candidate cap")
                    identifier = "utility_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
                    eligible.append(
                        {
                            "id": identifier,
                            "identity": identity,
                            "source": source,
                            "function": node.name,
                            "smoke_count": smoke_count,
                            "shown": shown,
                            "withheld": withheld,
                            "provenance": {
                                "repository": name,
                                "module": relative,
                                "start": node.lineno,
                                "end": node.end_lineno,
                                "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                                **inventories[name],
                            },
                        }
                    )
                except (ValueError, RuntimeError) as error:
                    exclusions.append({"id": identity, "reason": str(error)[:500]})
                save(directory / f"evidence-{executed:03d}.json", evidence)
                save(
                    directory / "selection.partial.json",
                    {
                        "inspected": inspected,
                        "executed": executed,
                        "eligible": len(eligible),
                        "exclusions": exclusions,
                    },
                )
            if stop:
                break
        if stop:
            break
    selected = sorted(eligible, key=lambda c: digest([73, c["identity"]]))
    dev_count = max(1, math.ceil(len(selected) / 4))
    for index, case in enumerate(selected):
        case["split"] = "development" if index < dev_count else "held_out"
    record = {
        "schema_version": 1,
        "scope": "extracted upstream doctest-bearing utilities",
        "sandbox_image": IMAGE.get(),
        "seed": 73,
        "inspected": inspected,
        "executed": executed,
        "exclusions": exclusions,
        "cases": selected,
        "ready": len(selected) >= 4 and len(selected) - dev_count >= 3,
    }
    record["sha256"] = digest(record)
    save(directory / "selection.json", record)
    return directory, record
