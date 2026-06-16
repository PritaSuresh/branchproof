"""The attempt loop.

The model looks at the code, writes a test, and the verifier decides. If the
verifier says no, the reason goes back to the model and it tries again, up to a
fixed number of attempts.

Writing the same test twice is caught here. Repeating a rejected attempt burns
the budget and never works.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from . import context
from .agent import AgentAction, Provider, ProviderError
from .coverage_map import context_files
from .feedback import describe
from .models import Attempt, Mutant, TaskRun
from .tools import ToolBox
from .verify import verify_candidate
from .workspace import Workspace, resolve_inside

MAX_LOOKUPS = 5


def _default_test_path(mutant: Mutant) -> str:
    stem = Path(mutant.path).stem
    suffix = hashlib.sha256(mutant.id.encode()).hexdigest()[:10]
    return f"tests/test_{stem}_hardened_{suffix}.py"


def _summarise(action: AgentAction) -> str:
    if action.tool == "edit_test":
        return str(action.args.get("path", ""))
    if action.tool == "read_file":
        return str(action.args.get("path", ""))
    if action.tool == "search_code":
        return str(action.args.get("query", ""))
    return action.note[:60]


def harden_mutant(
    space: Workspace,
    env: Path,
    mutant: Mutant,
    provider: Provider,
    scoped_tests: list[str],
    max_attempts: int = 3,
    stability_runs: int = 3,
    test_timeout: int = 120,
    suite_timeout: int = 300,
) -> TaskRun:
    started = time.time()
    run = TaskRun(task_id=mutant.id, repo=space.repo, mutant=mutant)
    tools = ToolBox(space=space, env=env, test_timeout=test_timeout, protect_existing=True)
    # Scoped tests are node ids, which pytest understands but a reader does not.
    nearby_files = context_files(scoped_tests)
    seen: set[str] = set()
    messages = [{"role": "user", "content": context.build(space.root, mutant, nearby_files)}]

    for attempt_number in range(1, max_attempts + 1):
        wrote_test = False

        for _ in range(MAX_LOOKUPS):
            try:
                action, prompt_tokens, completion_tokens = provider.next_action(messages)
            except ProviderError as error:
                run.prompt_tokens += error.prompt_tokens
                run.completion_tokens += error.completion_tokens
                run.attempts.append(
                    Attempt(attempt_number, "provider", "", False, str(error))
                )
                if error.recoverable and error.usage_known:
                    if error.response:
                        messages.append({"role": "assistant", "content": error.response})
                    messages.append({"role": "user", "content":
                        "Your response was not an executable action: " + str(error) +
                        '\nReply with a JSON object containing "tool", "args", and "note". '
                        'For a new test use {"tool":"edit_test","args":{"path":'
                        '"tests/test_new_regression.py","content":"complete Python test"},'
                        '"note":"reason"}. Use a permitted tool name, not a final answer.'})
                    continue
                run.status = "error"
                run.failure_class = {
                    "token_budget": "token reservation exhausted",
                    "call_budget": "model-call budget exhausted",
                    "usage_unknown": "unknown provider usage",
                }.get(error.kind, "provider error")
                run.seconds = round(time.time() - started, 1)
                return run
            run.tool_calls += 1
            run.prompt_tokens += prompt_tokens
            run.completion_tokens += completion_tokens

            if action.tool == "finish":
                run.attempts.append(
                    Attempt(attempt_number, "finish", action.note[:60], False, "agent stopped")
                )
                run.status = "failed"
                run.failure_class = "agent gave up"
                run.seconds = round(time.time() - started, 1)
                return run

            if action.tool == "edit_test":
                path = str(action.args.get("path") or _default_test_path(mutant))
                content = str(action.args.get("content", ""))
                digest = hashlib.sha256(content.encode()).hexdigest()
                if digest in seen:
                    messages.append({"role": "assistant", "content": action.model_dump_json()})
                    messages.append(
                        {
                            "role": "user",
                            "content": "That is the same test you already tried. It was "
                            "rejected once. Write something different.",
                        }
                    )
                    run.attempts.append(
                        Attempt(attempt_number, "edit_test", path, False, "repeated the same test")
                    )
                    continue
                observation = tools.edit_test(path, content)
                if not observation.startswith("Wrote "):
                    messages.append({"role": "assistant", "content": action.model_dump_json()})
                    messages.append({"role": "user", "content": observation})
                    run.attempts.append(
                        Attempt(attempt_number, "edit_test", path, False, observation)
                    )
                    continue
                seen.add(digest)
                messages.append({"role": "assistant", "content": action.model_dump_json()})
                wrote_test = True
                break

            if action.tool == "read_file":
                observation = tools.read_file(
                    str(action.args.get("path", "")),
                    action.args.get("start"),
                    action.args.get("end"),
                )
            elif action.tool == "search_code":
                observation = tools.search_code(str(action.args.get("query", "")))
            elif action.tool == "run_tests":
                observation = tools.run_tests(action.args.get("path"))
            else:
                observation = f"There is no tool called {action.tool!r}."

            messages.append({"role": "assistant", "content": action.model_dump_json()})
            messages.append({"role": "user", "content": observation})

        if not wrote_test:
            run.attempts.append(
                Attempt(attempt_number, "none", "", False, "no test was written")
            )
            messages.append({"role": "user", "content":
                             "You did not write a test. Use edit_test this time."})
            continue

        verdict = verify_candidate(
            space, env, path, mutant, scoped_tests,
            stability_runs=stability_runs,
            test_timeout=test_timeout,
            suite_timeout=suite_timeout,
        )
        feedback = describe(verdict, mutant.label)
        run.attempts.append(
            Attempt(attempt_number, "edit_test", path, verdict.accepted, verdict.reason, feedback)
        )

        if verdict.accepted:
            run.status = "success"
            run.accepted_test_path = path
            run.accepted_test_content = space.read(path)
            run.seconds = round(time.time() - started, 1)
            return run

        if verdict.first_failure and verdict.first_failure.name == "stayed_in_bounds":
            # A test changed the workspace while executing. Do not reuse that state.
            run.status = "failed"
            run.failure_class = "workspace integrity violation"
            run.seconds = round(time.time() - started, 1)
            return run

        # Only newly created candidates can reach here; existing files are protected.
        candidate = resolve_inside(space.root, path)
        if candidate.exists():
            candidate.unlink()
        messages.append({"role": "user", "content": feedback +
                         "\nThe rejected candidate file was removed. Production code and "
                         "existing tests are unchanged. Write a complete replacement test."})

    run.status = "failed"
    run.failure_class = _classify(run)
    run.seconds = round(time.time() - started, 1)
    return run


def _classify(run: TaskRun) -> str:
    """Best guess at why this task failed, refined by hand later."""
    reasons = [a.reason for a in run.attempts]
    if any("repeated the same test" in r for r in reasons):
        return "repeated the same strategy"
    if any("candidate_collects" in r for r in reasons):
        return "invalid test"
    if any("passes_on_original" in r for r in reasons):
        return "wrong expected behaviour"
    if any("mutant_execution_valid" in r for r in reasons):
        return "invalid mutant execution"
    if any("kills_mutant" in r for r in reasons):
        return "mutant survived"
    if any("suite_still_green" in r for r in reasons):
        return "broke the existing suite"
    if any("stable_across_runs" in r for r in reasons):
        return "unstable test"
    if any("no test was written" in r for r in reasons):
        return "no test produced"
    return "attempts exhausted"
