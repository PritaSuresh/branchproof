"""Works out which tests touch which lines.

Running the whole suite for every mutant is the thing that makes mutation testing
too slow to use. A line to tests map means we only run the handful of tests that
actually reach the mutated line.

Built once per module and cached. If contexts are not available we fall back to
matching test files by name, which is cruder but still much cheaper than the
whole suite.
"""

from __future__ import annotations

import json
from pathlib import Path

from .runner import RunResult, run_pytest, tests_touching
from .workspace import is_test_path


def build(
    root: Path, env: Path, module: str, timeout: int = 900
) -> tuple[dict[int, list[str]], RunResult]:
    """Map line numbers in `module` to the tests that execute them."""
    # --cov takes a package or directory, not a path to a single file.
    parent = str(Path(module).parent) or "."
    result = run_pytest(
        root, env, timeout=timeout,
        extra=["--cov", parent, "--cov-context=test", "--cov-report="],
        # From python 3.12 the default coverage core is sysmon, which quietly
        # collects no per test contexts at all.
        env_extra={"COVERAGE_CORE": "ctrace"},
    )
    mapping: dict[int, list[str]] = {}
    data_file = root / ".coverage"
    if not data_file.exists():
        return mapping, result
    try:
        from coverage import CoverageData
    except ImportError:
        return mapping, result

    data = CoverageData(basename=str(data_file))
    data.read()
    target = str((root / module).resolve())
    for measured in data.measured_files():
        if Path(measured).resolve() != Path(target):
            continue
        for line, contexts in data.contexts_by_lineno(measured).items():
            # A context is a pytest node id with a phase suffix. Doctests produce
            # contexts pointing at the source file itself, which are not tests we
            # can run or show as examples.
            tests = sorted({
                node for node in (_test_context(c) for c in contexts if c)
                if node and is_test_path(node.split("::")[0])
            })
            if tests:
                mapping[int(line)] = tests
    return mapping, result


def covered_mutants(mutants: list, mapping: dict[int, list[str]]) -> list:
    """Mutants on lines the tests reach.

    With no usable map we keep everything and fall back to picking test files by
    name, which is slower but does not silently leave the module untested.
    """
    if not mapping:
        return list(mutants)
    return [m for m in mutants if m.line in mapping]


def tests_for_line(
    root: Path, module: str, line: int, mapping: dict[int, list[str]]
) -> list[str]:
    """Tests to run for a mutant on this line."""
    hits = mapping.get(line)
    if hits:
        return hits
    return tests_touching(root, module)


def save(path: Path, module: str, mapping: dict[int, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[module] = {str(line): tests for line, tests in mapping.items()}
    path.write_text(json.dumps(existing, indent=2))


def context_files(node_ids: list[str]) -> list[str]:
    """File paths behind a list of pytest node ids, in order and without repeats."""
    seen: list[str] = []
    for node in node_ids:
        path = _test_context(node).split("::")[0]
        if path and path not in seen:
            seen.append(path)
    return seen


def _test_context(context: str) -> str:
    """Remove only coverage's final phase suffix, preserving parametrized ids."""
    node, separator, phase = context.rpartition("|")
    return node if separator and phase in {"setup", "run", "teardown"} else context
