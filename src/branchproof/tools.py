"""The things the model is allowed to do.

Every action has a fixed signature and edits follow the current stage policy.
Running Python tests executes repository code; these guards are not an OS sandbox.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from pathlib import Path

from .runner import normalize_test_target, run_pytest
from .workspace import OutsideWorkspace, Workspace, repository_files, resolve_inside

MAX_CHARS = 12000


def _breaks_syntax(path: str, before: str, after: str) -> str | None:
    """Report an edit that would leave a valid Python file unparseable.

    A file that already fails to parse is left alone; the edit is not the cause.
    """
    if not path.endswith(".py"):
        return None
    try:
        ast.parse(before)
    except SyntaxError:
        return None
    try:
        ast.parse(after)
    except SyntaxError as error:
        return f"{error.msg} at line {error.lineno}"
    return None


def _flatten(text: str) -> str:
    """Ignore differences that carry no meaning for locating a block of code."""
    return text.replace("\r\n", "\n").expandtabs()


def _shift(lines: list[str], found: str, wanted: str) -> list[str]:
    """Move replacement lines to the indentation the file actually uses."""
    if found == wanted:
        return lines
    shifted = []
    for line in lines:
        if not line.strip():
            shifted.append(line)
        elif line.startswith(wanted):
            shifted.append(found + line[len(wanted):])
        else:
            shifted.append(found + line.lstrip())
    return shifted


def _indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _windows(lines: list[str], old: list[str]) -> list[int]:
    """Line offsets where old matches once per-line whitespace is ignored.

    Models copy code out of a numbered read and lose a space or land on a
    different indentation level. The block is still the one they meant.
    """
    wanted = [line.strip() for line in old]
    hits = []
    for start in range(len(lines) - len(old) + 1):
        if [line.strip() for line in lines[start:start + len(old)]] == wanted:
            hits.append(start)
    return hits


def _anchored(lines: list[str], old: list[str]) -> list[int]:
    """Line offsets where the first and last line match and the middle is close.

    A last resort for a block whose interior the model paraphrased slightly.
    Only used for blocks long enough that the two anchors mean something.
    """
    if len(old) < 3:
        return []
    head, tail = old[0].strip(), old[-1].strip()
    middle = "\n".join(line.strip() for line in old[1:-1])
    hits = []
    for start in range(len(lines) - len(old) + 1):
        block = lines[start:start + len(old)]
        if block[0].strip() != head or block[-1].strip() != tail:
            continue
        here = "\n".join(line.strip() for line in block[1:-1])
        if difflib.SequenceMatcher(None, middle, here).ratio() >= 0.7:
            hits.append(start)
    return hits


def _closest(lines: list[str], old: list[str]) -> str:
    """The real file text nearest to what the model asked for, with line numbers."""
    wanted = "\n".join(line.strip() for line in old)
    best, score = None, 0.0
    span = max(1, len(old))
    for start in range(max(1, len(lines) - span + 1)):
        block = lines[start:start + span]
        here = "\n".join(line.strip() for line in block)
        ratio = difflib.SequenceMatcher(None, wanted, here).ratio()
        if ratio > score:
            best, score = start, ratio
    if best is None or score < 0.4:
        return ""
    numbered = [f"{n:5d}  {lines[n - 1]}" for n in range(best + 1, best + span + 1)]
    return "\n".join(numbered)[:1500]

TOOL_SCHEMA = {
    "list_files": {"prefix": "str, optional"},
    "read_file": {"path": "str", "start": "int, optional", "end": "int, optional"},
    "search_code": {"query": "str", "path": "str, optional"},
    "replace_lines": {"path": "str", "start": "int", "end": "int", "content": "str"},
    "edit_file": {"path": "str", "content": "str (whole file)"},
    "run_tests": {"path": "str, optional"},
    "run_mutant": {},
    "finish": {"reason": "str"},
}


@dataclass
class ToolBox:
    space: Workspace
    env: Path
    test_timeout: int = 120
    protect_existing: bool = False
    refuse_new_files: bool = False

    def list_files(self, prefix: str = "") -> str:
        files = []
        for path in sorted(repository_files(self.space.root)):
            name = path.relative_to(self.space.root).as_posix()
            if path.is_symlink() or any(p.startswith(".") for p in Path(name).parts):
                continue
            if name.startswith(prefix):
                files.append(name)
        return "\n".join(files)[:MAX_CHARS] or "No matching files."

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        if any(value is not None and not isinstance(value, int) for value in (start, end)):
            return "Refused. Line numbers must be integers."
        try:
            target = resolve_inside(self.space.root, path)
        except OutsideWorkspace:
            return f"Refused. {path} is outside the repository."
        relative = target.relative_to(self.space.root.resolve())
        if ".git" in relative.parts or any(part.startswith(".env") for part in relative.parts):
            return "Refused. Repository metadata and environment files are not model inputs."
        if not target.is_file():
            return f"{path} does not exist."
        try:
            lines = target.read_text(errors="replace").splitlines()
        except OSError as reason:
            return f"Refused. Could not read {path}: {reason}"
        first = max(1, start or 1)
        last = min(len(lines), end or len(lines))
        numbered = [f"{n:5d}  {lines[n - 1]}" for n in range(first, last + 1)]
        text = "\n".join(numbered)
        if len(text) > MAX_CHARS:
            shown = text[:MAX_CHARS].rsplit("\n", 1)[0]
            cut = first + shown.count("\n")
            return (
                f"{shown}\n... output stops here. {path} has {len(lines)} lines and you "
                f"have seen up to line {cut}. Call read_file again with start and end "
                "to see the rest."
            )
        return text

    def search_code(self, query: str, path: str | None = None) -> str:
        if not isinstance(query, str):
            return "Refused. The search query must be a string."
        hits: list[str] = []
        for filepath in sorted(repository_files(self.space.root)):
            if filepath.suffix != ".py" or filepath.is_symlink():
                continue
            relative = filepath.relative_to(self.space.root)
            if any(part == ".git" or part.startswith(".env") for part in relative.parts):
                continue
            if path and not relative.as_posix().startswith(path):
                continue
            try:
                lines = filepath.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                if query in line:
                    hits.append(f"{relative}:{number}: {line.strip()}")
                    if len(hits) >= 40:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else f"No match for {query!r}."

    def _apply(self, path: str, current: str, updated: str, note: str) -> str:
        """Check an edit for syntax damage, write it, and show the result."""
        broken = _breaks_syntax(path, current, updated)
        if broken:
            return (
                f"Not applied. {path} was left unchanged because that edit makes it "
                f"unparseable: {broken}. Check the indentation of your replacement "
                "against the surrounding block and try again."
            )
        try:
            self.space.write(path, updated)
        except OutsideWorkspace:
            return f"Refused. {path} is outside the repository."
        except (ValueError, OSError) as reason:
            return f"Refused. {reason}"
        return f"{note}\n{self._around(path, current, updated)}"

    def _around(self, path: str, before: str, after: str) -> str:
        """Numbered lines around the change, so the next edit has real positions."""
        old_lines, new_lines = before.split("\n"), after.split("\n")
        first = 0
        while first < min(len(old_lines), len(new_lines)) and old_lines[first] == new_lines[first]:
            first += 1
        tail = 0
        while (tail < min(len(old_lines), len(new_lines)) - first
               and old_lines[-1 - tail] == new_lines[-1 - tail]):
            tail += 1
        low = max(1, first + 1 - 4)
        high = min(len(new_lines), len(new_lines) - tail + 4)
        window = [f"{n:5d}  {new_lines[n - 1]}" for n in range(low, high + 1)]
        return f"{path} now reads:\n" + "\n".join(window)[:2000]

    def replace_in_file(self, path: str, old: str, new: str) -> str:
        """Swap one piece of text for another.

        Rewriting a whole file to change two lines does not work. A model asked to
        reproduce two thousand lines verbatim will not, and the file ends up
        corrupted. Existing files are edited by naming the part to replace.

        The text is matched exactly first. When that fails, the same block is
        looked for while ignoring per-line whitespace, since a copied block that
        lost a space or landed at the wrong indentation is still the right block.
        """
        if not isinstance(old, str) or not isinstance(new, str):
            return "Refused. Replacement text must be a string."
        try:
            target = resolve_inside(self.space.root, path)
        except OutsideWorkspace:
            return f"Refused. {path} is outside the repository."
        if not target.is_file():
            return f"{path} does not exist. Use edit_file to create it."
        if self.protect_existing:
            return f"Refused. {path} already exists. Put your test in a new file."
        if not old.strip():
            return "Refused. Say which text to replace."

        try:
            current = _flatten(target.read_text())
        except (OSError, UnicodeError) as reason:
            return f"Refused. Could not read {path}: {reason}"
        wanted = _flatten(old).strip("\n")
        replacement = _flatten(new).strip("\n")

        found = current.count(wanted)
        if found == 1:
            updated = current.replace(wanted, replacement, 1)
            count = len(wanted.split("\n"))
            return self._apply(path, current, updated, f"Replaced {count} line(s) in {path}.")
        if found > 1:
            places = []
            offset = current.find(wanted)
            while offset != -1 and len(places) < 8:
                places.append(current.count("\n", 0, offset) + 1)
                offset = current.find(wanted, offset + 1)
            spots = ", ".join(str(n) for n in places)
            return (
                f"That text appears {found} times in {path}, at lines {spots}. "
                "Include more surrounding lines so it matches only once, or use "
                "replace_lines with the line numbers you want."
            )

        lines = current.split("\n")
        old_lines = wanted.split("\n")
        hits = _windows(lines, old_lines) or _anchored(lines, old_lines)
        if len(hits) == 1:
            at = hits[0]
            shifted = _shift(replacement.split("\n"),
                             _indent(lines[at]), _indent(old_lines[0]))
            updated = "\n".join(lines[:at] + shifted + lines[at + len(old_lines):])
            note = (f"Replaced {len(old_lines)} line(s) at lines "
                    f"{at + 1}-{at + len(old_lines)} in {path}. The text you gave did "
                    "not match character for character; the block was matched ignoring "
                    "whitespace and your replacement was reindented to fit.")
            return self._apply(path, current, updated, note)
        if len(hits) > 1:
            spots = ", ".join(str(at + 1) for at in hits[:8])
            return (
                f"That block matches {len(hits)} places in {path} (lines {spots}) once "
                "whitespace is ignored. Use replace_lines with the line numbers you want."
            )

        near = _closest(lines, old_lines)
        if near:
            return (
                f"That text is not in {path}. The closest lines in the file are:\n{near}\n"
                "Copy those lines exactly, or call replace_lines with their numbers."
            )
        return (
            f"That text is not in {path}. Read the file first and copy the lines "
            "exactly, including indentation."
        )

    def replace_lines(self, path: str, start: int, end: int, content: str) -> str:
        """Replace lines start..end (1-based, inclusive) with new content."""
        try:
            target = resolve_inside(self.space.root, path)
        except OutsideWorkspace:
            return f"Refused. {path} is outside the repository."
        if not target.is_file():
            return f"{path} does not exist."
        if self.protect_existing:
            return f"Refused. {path} already exists. Put your test in a new file."
        try:
            lines = target.read_text().splitlines(keepends=True)
        except (OSError, UnicodeError) as reason:
            return f"Refused. Could not read {path}: {reason}"
        if start < 1 or end < start or start > len(lines):
            return f"Invalid range. {path} has {len(lines)} lines."
        end = min(end, len(lines))
        new_lines = content.splitlines(keepends=True)
        if content and not content.endswith("\n"):
            new_lines[-1] += "\n"
        updated = "".join(lines[:start - 1] + new_lines + lines[end:])
        removed = end - start + 1
        added = len(new_lines)
        note = (f"Replaced lines {start}-{end} ({removed} lines) with {added} lines "
                f"in {path}.")
        return self._apply(path, "".join(lines), updated, note)

    def edit_file(self, path: str, content: str) -> str:
        """Write a whole file. Meant for new files."""
        if self.protect_existing or self.refuse_new_files:
            try:
                target = resolve_inside(self.space.root, path)
            except OutsideWorkspace:
                return f"Refused. {path} is outside the repository."
            if self.protect_existing and target.exists():
                return (
                    f"Refused. {path} already exists and this would replace all of it. "
                    "Put your test in a new file."
                )
            if self.refuse_new_files and not target.exists():
                return (
                    f"Refused. {path} does not exist. Fixing a bug means changing code "
                    "that is already there, not adding a new module. Use replace_in_file "
                    "on the file that has the problem."
                )
        try:
            self.space.write(path, content)
        except OutsideWorkspace:
            return f"Refused. {path} is outside the repository."
        except (ValueError, OSError) as reason:
            return f"Refused. {reason}"
        return f"Wrote {len(content.splitlines())} lines to {path}."

    # The harden loop still calls the old name.
    def edit_test(self, path: str, content: str) -> str:
        return self.edit_file(path, content)

    def run_tests(self, path: str | None = None) -> str:
        if path is not None:
            try:
                path = normalize_test_target(self.space.root, path)
            except ValueError as reason:
                return f"Refused. {reason}"
        targets = [path] if path else None
        result = run_pytest(self.space.root, self.env, targets, timeout=self.test_timeout)
        head = "passed" if result.passed else "failed"
        return f"{head}\n{result.feedback()}"
