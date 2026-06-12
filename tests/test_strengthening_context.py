"""Context for inherited methods must not hide the implementation under test."""

from branchproof.context import build
from branchproof.mutation import discover


def test_default_mutation_names_defining_class_and_dispatch_risk(tmp_path):
    source = ("class Base:\n"
              "    def describe(self, value=0):\n"
              "        return str(value)\n\n"
              "class Child(Base):\n"
              "    def describe(self, value=0):\n"
              "        return super().describe(value)\n")
    (tmp_path / "example.py").write_text(source)
    mutant = next(m for m in discover(source, "example.py") if m.line == 2)
    prompt = build(tmp_path, mutant, [])
    assert "Defining scope: Base.describe" in prompt
    assert "Passing that argument explicitly bypasses its default" in prompt
    assert "not signature inspection" in prompt


def test_body_constant_does_not_get_default_argument_advice(tmp_path):
    source = "def positive(value):\n    return value > 0\n"
    (tmp_path / "example.py").write_text(source)
    mutant = next(m for m in discover(source, "example.py") if m.kind == "constant")
    prompt = build(tmp_path, mutant, [])
    assert "Defining scope: positive" in prompt
    assert "This mutation changes a function parameter default" not in prompt


def test_keyword_only_default_is_identified(tmp_path):
    source = "def describe(value, *, relative=False):\n    return value, relative\n"
    (tmp_path / "example.py").write_text(source)
    mutant = next(m for m in discover(source, "example.py") if m.before == "False")
    assert "This mutation changes a function parameter default" in build(tmp_path, mutant, [])
