"""Scripted model decisions, real mutation screening and test execution. No API calls."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from branchproof.agent import AgentAction, ScriptedProvider
from branchproof.sandbox import execution_image, resolve_image
from branchproof.strengthen import strengthen

SOURCE = '''def allowed(balance, amount):
    if amount <= 0:
        return False
    return balance >= amount
'''
EXISTING = '''from account import allowed

def test_ordinary_inputs():
    assert allowed(100, 50)
    assert not allowed(100, 500)
'''
WEAK = '''from account import allowed

def test_another_ordinary_input():
    assert allowed(100, 20)
'''
STRONG = '''from account import allowed

def test_boundaries():
    assert not allowed(100, 0)
    assert not allowed(100, -1)
    assert allowed(100, 100)
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/strengthen-demo"))
    parser.add_argument("--sandbox-image", default="hardener-sandbox:tests-v1")
    parser.add_argument("--unsafe-local", action="store_true")
    args = parser.parse_args(argv)
    image = None if args.unsafe_local else resolve_image(args.sandbox_image)
    print("SCRIPTED DEMO: real execution, no model calls; not model-performance evidence.")
    provider = ScriptedProvider([
        AgentAction(tool="edit_test", args={"path": "tests/test_added.py", "content": WEAK}),
        AgentAction(tool="edit_test", args={"path": "tests/test_added.py", "content": STRONG}),
    ])
    with tempfile.TemporaryDirectory(prefix="strengthen-demo-") as temporary:
        repo = Path(temporary)
        (repo / "tests").mkdir()
        (repo / "account.py").write_text(SOURCE)
        (repo / "tests/test_account.py").write_text(EXISTING)
        for command in (["git", "init", "-q"], ["git", "add", "."],
                        ["git", "-c", "user.name=Demo", "-c", "user.email=demo@example.invalid",
                         "commit", "-qm", "Scripted fixture"]):
            subprocess.run(command, cwd=repo, check=True, capture_output=True)
        with execution_image(image):
            directory, result = strengthen(
                repo, Path(sys.executable).parent.parent, "account.py", provider, args.output,
                limit=1, max_calls=2, stability_runs=3)
        assert result["accepted"] == 1, result
        assert len(result["targets"][0]["attempts"]) == 2
        patch = directory / result["targets"][0]["patch"]
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=repo, check=True)
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)
        assert (repo / "account.py").read_text() == SOURCE
        print("PASS: weak test rejected, replacement accepted, patch applies, checkout unchanged.")
        print(f"Saved evidence: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
