"""Builds the small amount of code the model needs to see.

Sending the whole repository is expensive and makes the model worse, not better.
It gets the mutated line, the function around it, and the tests that already
touch that file.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .coverage_map import context_files
from .models import Mutant
from .workspace import OutsideWorkspace, resolve_inside


def enclosing_block(source: str, line: int) -> tuple[str, str]:
    """The function or class containing a line, with its name."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "", ""
    lines = source.splitlines()
    best: tuple[int, int, str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if node.lineno <= line <= (node.end_lineno or node.lineno):
            size = (node.end_lineno or node.lineno) - node.lineno
            if best is None or size < best[1] - best[0]:
                best = (node.lineno, node.end_lineno or node.lineno, node.name)
    if best is None:
        low = max(1, line - 15)
        high = min(len(lines), line + 15)
        return "\n".join(lines[low - 1 : high]), ""
    start, end, name = best
    return "\n".join(lines[start - 1 : end]), name


def existing_tests(root: Path, test_files: list[str], budget: int = 4000) -> str:
    """A sample of the tests that already exist for this area."""
    chunks: list[str] = []
    used = 0
    for relative in context_files(test_files)[:3]:
        try:
            path = resolve_inside(root, relative)
        except OutsideWorkspace:
            continue
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        room = max(0, budget - used)
        if room == 0:
            break
        snippet = text[:room]
        used += len(snippet)
        chunks.append(f"# {relative}\n{snippet}")
    return "\n\n".join(chunks)


def build(root: Path, mutant: Mutant, test_files: list[str], previous: str = "") -> str:
    source = resolve_inside(root, mutant.path).read_text()
    block, name = enclosing_block(source, mutant.line)
    lines = source.splitlines()
    target_line = lines[mutant.line - 1] if mutant.line <= len(lines) else ""

    parts = [
        "A change was made to one line of this file and the existing tests did not notice.",
        "",
        f"File: {mutant.path}",
        f"Line {mutant.line}: {target_line.strip()}",
        f"The change: {mutant.before}  ->  {mutant.after}",
        "",
        f"The code around it{f' (in {name})' if name else ''}:",
        "```python",
        block,
        "```",
        "",
        "Tests that already exist for this file:",
        "```python",
        existing_tests(root, test_files) or "# none found",
        "```",
    ]
    try:
        tree = ast.parse(source)
        owners = [n for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                  and n.lineno <= mutant.line <= (n.end_lineno or n.lineno)]
        owners.sort(key=lambda n: (n.lineno, -(n.end_lineno or n.lineno)))
        if owners:
            parts += ["", "Defining scope: " + ".".join(n.name for n in owners)]
        default_target = any(
            default is not None and default.lineno <= mutant.line <= default.end_lineno
            and (default.lineno != mutant.line or default.col_offset <=
                 len(lines[mutant.line - 1][:mutant.col].encode("utf-8")))
            and (default.end_lineno != mutant.line or default.end_col_offset >=
                 len(lines[mutant.line - 1][:mutant.end_col].encode("utf-8")))
            for n in owners if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            for default in [*n.args.defaults, *n.args.kw_defaults])
        if default_target:
            parts += ["", "This mutation changes a function parameter default. "
                      "Passing that argument explicitly bypasses its default. For methods, "
                      "inspect the concrete receiver's overrides: an overriding method may "
                      "forward its own default explicitly to the base implementation. "
                      "Exercise the defining implementation's omitted-argument behavior "
                      "through a meaningful behavioral call, not signature inspection."]
    except SyntaxError:
        pass
    if previous:
        parts += ["", "What happened last time you tried:", previous]
    return "\n".join(parts)
