"""Finds small changes to source code and applies one at a time.

Each change is a mutant. If the tests still pass with a mutant applied, the tests
are not checking that bit of behaviour.

Changes are spliced straight into the source text using the column offsets the
parser gives us, so the rest of the file stays byte for byte identical. Anything
that spans more than one line is skipped, which keeps the splice simple and safe.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .models import Mutant
from .workspace import resolve_inside

COMPARE_TOKENS = {
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
    "Eq": "==",
    "NotEq": "!=",
}

# Swapping < for <= moves a boundary. Swapping < for > flips the direction.
COMPARE_SWAPS = {
    "Lt": ["<=", ">"],
    "LtE": ["<", ">="],
    "Gt": [">=", "<"],
    "GtE": [">", "<="],
    "Eq": ["!="],
    "NotEq": ["=="],
}

BINOP_TOKENS = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "FloorDiv": "//", "Mod": "%"}
BINOP_SWAPS = {
    "Add": ["-"],
    "Sub": ["+"],
    "Mult": ["/"],
    "Div": ["*"],
    "FloorDiv": ["/"],
    "Mod": ["*"],
}

BOOLOP_TOKENS = {"And": "and", "Or": "or"}
BOOLOP_SWAPS = {"And": ["or"], "Or": ["and"]}


def _column(line: str, byte_offset: int) -> int:
    """AST offsets count UTF-8 bytes; source splices count Python characters."""
    return len(line.encode("utf-8")[:byte_offset].decode("utf-8"))


def _span_between(
    lines: list[str], line_no: int, start: int, end: int, token: str
) -> tuple[int, int] | None:
    """Column range of `token` in the gap between two sub-expressions."""
    if line_no - 1 >= len(lines):
        return None
    line = lines[line_no - 1]
    start, end = _column(line, start), _column(line, end)
    gap = line[start:end]
    found = gap.find(token)
    if found == -1:
        return None
    return start + found, start + found + len(token)


def _mutant(
    path: str, line: int, col: int, end_col: int, kind: str, before: str, after: str
) -> Mutant:
    return Mutant(
        id=f"{path}:{line}:{col}:{kind}:{before}->{after}",
        path=path,
        line=line,
        col=col,
        end_col=end_col,
        kind=kind,
        before=before,
        after=after,
    )


def discover(source: str, path: str) -> list[Mutant]:
    """Every single-line change we know how to make in this file."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    found: list[Mutant] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op_name = type(node.ops[0]).__name__
            token = COMPARE_TOKENS.get(op_name)
            left, right = node.left, node.comparators[0]
            if token is None or left.end_lineno != right.lineno:
                continue
            span = _span_between(
                lines, left.end_lineno, left.end_col_offset, right.col_offset, token
            )
            if span is None:
                continue
            for replacement in COMPARE_SWAPS[op_name]:
                found.append(
                    _mutant(
                        path, left.end_lineno, span[0], span[1], "comparison", token, replacement
                    )
                )

        elif isinstance(node, ast.BinOp):
            op_name = type(node.op).__name__
            token = BINOP_TOKENS.get(op_name)
            if token is None or node.left.end_lineno != node.right.lineno:
                continue
            span = _span_between(
                lines, node.left.end_lineno, node.left.end_col_offset, node.right.col_offset, token
            )
            if span is None:
                continue
            for replacement in BINOP_SWAPS[op_name]:
                found.append(
                    _mutant(
                        path, node.left.end_lineno, span[0], span[1],
                        "arithmetic", token, replacement,
                    )
                )

        elif isinstance(node, ast.BoolOp) and len(node.values) >= 2:
            op_name = type(node.op).__name__
            token = BOOLOP_TOKENS[op_name]
            left, right = node.values[0], node.values[1]
            if left.end_lineno != right.lineno:
                continue
            span = _span_between(
                lines, left.end_lineno, left.end_col_offset, right.col_offset, token
            )
            if span is None:
                continue
            for replacement in BOOLOP_SWAPS[op_name]:
                found.append(
                    _mutant(path, left.end_lineno, span[0], span[1], "boolean", token, replacement)
                )

        elif isinstance(node, ast.Constant) and node.lineno == node.end_lineno:
            line = lines[node.lineno - 1]
            start = _column(line, node.col_offset)
            end = _column(line, node.end_col_offset)
            before = line[start:end]
            if isinstance(node.value, bool):
                after = "False" if node.value else "True"
                found.append(
                    _mutant(
                        path, node.lineno, start, end,
                        "constant", before, after,
                    )
                )
            elif isinstance(node.value, int) and not isinstance(node.value, bool):
                after = str(node.value + 1)
                found.append(
                    _mutant(
                        path, node.lineno, start, end,
                        "constant", before, after,
                    )
                )

        elif isinstance(node, ast.Return) and node.value is not None:
            value = node.value
            if value.lineno != value.end_lineno:
                continue
            if isinstance(value, ast.Constant) and value.value is None:
                continue
            line = lines[value.lineno - 1]
            start = _column(line, value.col_offset)
            end = _column(line, value.end_col_offset)
            text = line[start:end]
            found.append(
                _mutant(
                    path, value.lineno, start, end,
                    "return", text, "None",
                )
            )

    # Two identical spans can be produced by nested nodes. Keep the first of each.
    unique: dict[str, Mutant] = {}
    for mutant in found:
        unique.setdefault(mutant.id, mutant)
    return sorted(unique.values(), key=lambda m: (m.line, m.col, m.id))


def discover_in_file(root: Path, relative: str) -> list[Mutant]:
    return discover(resolve_inside(root, relative).read_text(), relative)


def apply_mutant(source: str, mutant: Mutant) -> str:
    """Put one mutant into the source text."""
    lines = source.splitlines(keepends=True)
    index = mutant.line - 1
    if index < 0 or index >= len(lines):
        raise ValueError(f"line {mutant.line} is past the end of {mutant.path}")
    line = lines[index]
    if not 0 <= mutant.col < mutant.end_col <= len(line.rstrip("\r\n")):
        raise ValueError(f"invalid source span at {mutant.path}:{mutant.line}")
    actual = line[mutant.col : mutant.end_col]
    if actual != mutant.before:
        raise ValueError(
            f"expected {mutant.before!r} at {mutant.path}:{mutant.line}:"
            f"{mutant.col}, found {actual!r}"
        )
    lines[index] = line[: mutant.col] + mutant.after + line[mutant.end_col :]
    return "".join(lines)


def write_mutant(root: Path, mutant: Mutant) -> str:
    """Apply a mutant on disk. Returns the original text so the caller can restore it."""
    target = resolve_inside(root, mutant.path)
    original = target.read_text()
    target.write_text(apply_mutant(original, mutant))
    return original
