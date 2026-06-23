"""Scripted blind-generation preflight; no model calls or model success claim."""

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from strengthen_demo import EXISTING, SOURCE, STRONG, WEAK

from branchproof.agent import AgentAction, ScriptedProvider
from branchproof.artifacts import new_run_directory
from branchproof.blind_eval import generate, grade, initial_context
from branchproof.mutation import discover
from branchproof.real_cases import initialise
from branchproof.sandbox import execution_image, resolve_image
from branchproof.test_study import save
from branchproof.workspace import worktree

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sha256:1724b62eeb9fe490dfae47ec569db35b4a4777a4e139cad1a370b79e8f9ae417"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox-image", default=IMAGE)
    args = parser.parse_args()
    directory = new_run_directory(ROOT / "runs/blind-evaluation-preflight")
    env = Path(sys.prefix)
    with tempfile.TemporaryDirectory(prefix="branchproof-blind-preflight-") as temporary:
        repo = Path(temporary)
        (repo / "tests").mkdir()
        (repo / "account.py").write_text(SOURCE)
        (repo / "tests/test_account.py").write_text(EXISTING)
        initialise(repo)
        hidden = next(
            m for m in discover(SOURCE, "account.py") if m.before == ">=" and m.after == ">"
        )
        public = {
            "id": "demo",
            "context": initial_context(repo, "account.py", "allowed", ["tests/test_account.py"]),
        }
        private = {"id": "demo", "mutants": [asdict(hidden)]}
        results = {}
        image = resolve_image(args.sandbox_image)
        with execution_image(image):
            for label, submissions in {
                "wrong_then_strong": [
                    STRONG.replace("assert allowed(100, 100)", "assert not allowed(100, 100)"),
                    STRONG,
                ],
                "valid_but_weak": [WEAK],
            }.items():
                provider = ScriptedProvider(
                    [
                        AgentAction(
                            tool="edit_test",
                            args={"path": "tests/test_branchproof_demo.py", "content": text},
                        )
                        for text in submissions
                    ]
                )
                with worktree(repo) as space:
                    generated = generate(space, env, public, provider, "agent")
                assert generated["status"] == "valid_candidate"
                result = grade(repo, env, private, generated["content"])
                assert result["valid"]
                results[label] = {"submissions": generated["submissions"], "grade": result}
                save(directory / f"{label}.json", generated)
        assert results["wrong_then_strong"]["submissions"] == 2
        assert results["wrong_then_strong"]["grade"]["detected"] == 1
        assert results["valid_but_weak"]["submissions"] == 1
        assert results["valid_but_weak"]["grade"]["detected"] == 0
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)
        assert (repo / "account.py").read_text() == SOURCE
    record = {
        "status": "passed",
        "model_calls": 0,
        "purpose": "scripted harness preflight, NOT model-performance evaluation",
        "results": results,
        "sandbox_image": image,
    }
    save(directory / "preflight.json", record)
    print(json.dumps({"directory": str(directory), "status": "passed", "model_calls": 0}))


if __name__ == "__main__":
    main()
