"""Paired checkpoints: one initial submission, optional original-feedback continuation.

Generation accepts only public task context. Hidden grading runs separately.
"""

from __future__ import annotations

import hashlib
import json

from .agent import ProviderError, action_error
from .blind_eval import SYSTEM, original_checks
from .runner import with_upstream_collection
from .tools import ToolBox
from .workspace import resolve_inside


def schedule(manifest):
    jobs = [
        {"id": c["id"], "repeat": repeat, "status": "pending"}
        for c in manifest["cases"]
        for repeat in range(3)
    ]
    return sorted(
        jobs,
        key=lambda j: hashlib.sha256(
            json.dumps([20261020, j["id"], j["repeat"]]).encode()
        ).hexdigest(),
    )


def request(task):
    if set(task) != {"id", "context"}:
        raise ValueError("Only public task id and context are allowed")
    path = f"tests/test_branchproof_{task['id']}.py"
    contract = f"\nCreate only {path}. Use edit_test with the complete Python file."
    system = (
        SYSTEM
        + contract
        + (
            "\nThe first response must submit edit_test. After rejection, permitted tools "
            "and original-code execution feedback may be used to revise the candidate."
        )
    )
    return path, system, [{"role": "user", "content": task["context"] + contract}]


@with_upstream_collection
def generate_pair(space, env, task, provider):
    path, system, messages = request(task)
    if resolve_inside(space.root, path).exists():
        raise ValueError("Designated test path already exists")
    tools = ToolBox(space, env, test_timeout=30, protect_existing=True)
    result = {
        "status": "no_valid_candidate",
        "path": path,
        "actions": [],
        "submissions": 0,
        "initial": None,
        "final": None,
    }
    for decision in range(5):
        try:
            action, _, _ = provider.next_action(messages, system=system)
        except ProviderError as error:
            result["status"] = error.kind + "_error"
            break
        messages.append({"role": "assistant", "content": action.model_dump_json()})
        entry = {"action": action.model_dump()}
        result["actions"].append(entry)
        error = action_error(action)
        if decision == 0 and (error or action.tool != "edit_test"):
            result["status"] = "invalid_initial_action"
            entry["feedback"] = error or "First response must submit edit_test"
            break
        if error:
            feedback = "Rejected action: " + error
        elif action.tool == "finish":
            break
        elif action.tool == "edit_test":
            if action.args["path"] != path:
                feedback = "Refused: only designated new test file allowed"
                if decision == 0:
                    entry["feedback"] = feedback
                    result["status"] = "invalid_initial_action"
                    break
            else:
                result["submissions"] += 1
                content = action.args["content"]
                space.write(path, content)
                valid, feedback = original_checks(space, env, path)
                checkpoint = {"content": content, "original_valid": valid}
                if decision == 0:
                    result["initial"] = checkpoint
                result["final"] = checkpoint
                entry["feedback"] = feedback
                if valid:
                    result["status"] = "valid_candidate"
                    break
                if result["submissions"] == 3:
                    break
        elif action.tool in {"read_file", "search_code", "run_tests"}:
            feedback = getattr(tools, action.tool)(**action.args)
        else:
            feedback = "Refused: unsupported tool"
        entry["feedback"] = feedback
        messages.append({"role": "user", "content": feedback})
    result["messages"] = messages
    return result


def summarize(manifest, record):
    expected = {(j["id"], j["repeat"]) for j in schedule(manifest)}
    jobs = record["jobs"]
    if len(jobs) != len(expected) or {(j["id"], j["repeat"]) for j in jobs} != expected:
        raise ValueError("Incomplete or duplicate schedule")
    mutants = {m["id"]: m for c in manifest["cases"] for m in c["mutants"]}
    result = {}
    for checkpoint in ("initial", "final"):
        grades = [j.get("grades", {}).get(checkpoint, {}) for j in jobs]
        detected = [m for g in grades for m in g.get("mutations", []) if m["detected"]]
        contract = sum(
            mutants[m["id"]]["before"] == "False" and mutants[m["id"]]["after"] == "None"
            for m in detected
        )
        result[checkpoint] = {
            "scheduled": len(expected),
            "valid": sum(bool(g.get("valid")) for g in grades),
            "successful": sum(bool(g.get("valid") and g.get("detected", 0)) for g in grades),
            "hidden_detected": len(detected),
            "hidden_scheduled": 3 * len(mutants),
            "false_to_none_detected": contract,
            "false_to_none_scheduled": 3
            * sum(m["before"] == "False" and m["after"] == "None" for m in mutants.values()),
            "other_detected": len(detected) - contract,
            "per_target_successes": {
                c["id"]: sum(
                    bool(
                        j.get("grades", {}).get(checkpoint, {}).get("valid")
                        and j["grades"][checkpoint].get("detected", 0)
                    )
                    for j in jobs
                    if j["id"] == c["id"]
                )
                for c in manifest["cases"]
            },
        }
    result["invalid_initial_submissions"] = sum(
        bool(j.get("generation", {}).get("initial"))
        and not j["generation"]["initial"]["original_valid"]
        for j in jobs
    )
    result["rescued_valid"] = sum(
        bool(j.get("generation", {}).get("initial"))
        and not j["generation"]["initial"]["original_valid"]
        and bool(j.get("grades", {}).get("final", {}).get("valid"))
        for j in jobs
    )
    result["rescued_successful"] = sum(
        bool(j.get("generation", {}).get("initial"))
        and not j["generation"]["initial"]["original_valid"]
        and bool(j.get("grades", {}).get("final", {}).get("valid"))
        and j["grades"]["final"].get("detected", 0) > 0
        for j in jobs
    )
    return result
