"""Publish complete shared-candidate results without changing frozen metrics."""

import argparse
import json
from collections import Counter
from pathlib import Path

from branchproof.paired_eval import summarize

ROOT = Path(__file__).resolve().parents[1]


def report(manifest, study):
    if study["status"] != "complete" or any(
        set(j.get("grades", {})) != {"initial", "final"} for j in study["jobs"]
    ):
        raise ValueError("All scheduled generation and independent grades must be complete")
    if study["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Manifest mismatch")
    summary = summarize(manifest, study)
    if summary != study["summary"]:
        raise ValueError("Stored summary does not recompute")
    lookup = {m["id"]: m for c in manifest["cases"] for m in c["mutants"]}
    operators = {}
    for checkpoint in ("initial", "final"):
        detected = Counter()
        for j in study["jobs"]:
            grade = j["grades"][checkpoint]
            rows = grade.get("mutations", [])
            if len({m["id"] for m in rows}) != len(rows):
                raise ValueError("Duplicate mutation grade")
            for mutation in rows:
                if mutation["detected"]:
                    detected[lookup[mutation["id"]]["kind"]] += 1
        pool = Counter(m["kind"] for m in lookup.values())
        operators[checkpoint] = {
            kind: {"detected": detected[kind], "scheduled": 3 * pool[kind]}
            for kind in ("return", "constant", "boolean", "comparison", "arithmetic")
        }
    rates = study["profile"]["pricing_reference"]["meters"]
    input_rate = max(m["retailPrice"] for m in rates if " Inp " in m["meterName"])
    output_rate = max(m["retailPrice"] for m in rates if " Opt " in m["meterName"])
    usage = {}
    for phase in ("initial", "continuation"):
        traces = [
            t
            for j in study["jobs"]
            for t in (j["trace"][:1] if phase == "initial" else j["trace"][1:])
        ]
        prompt = sum(t.get("prompt_tokens", 0) for t in traces)
        completion = sum(t.get("completion_tokens", 0) for t in traces)
        usage[phase] = {
            "calls": len(traces),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usage_complete": all(t["usage_known"] for t in traces),
            "uncached_public_reference_usd": (prompt * input_rate + completion * output_rate) / 1e6,
        }
    return {
        "summary": summary,
        "by_operator": operators,
        "usage": usage,
        "manifest_sha256": manifest["sha256"],
        "jobs": study["jobs"],
        "profile": study["profile"],
        "actual_invoice_verified": False,
        "generation_seconds": sum(j["generation_seconds"] for j in study["jobs"]),
        "status": "complete",
        "pushed": False,
    }


def render(result, manifest):
    s = result["summary"]
    lines = [
        "# Test-generation evaluation",
        "",
        "Ten existing development targets, three runs each. One initial candidate per pair;",
        "the feedback loop may revise it only after original-code rejection. Hidden",
        "mutations and their execution results remain unavailable during generation.",
        "",
        "| Checkpoint | Successful / 30 | Valid / 30 | Hidden detections / 51 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for checkpoint in ("initial", "final"):
        c = s[checkpoint]
        lines.append(
            f"| {checkpoint} | {c['successful']}/30 | {c['valid']}/30 | {c['hidden_detected']}/51 |"
        )
    lines += [
        "",
        f"Invalid initial submissions: {s['invalid_initial_submissions']}; "
        f"rescued to valid: {s['rescued_valid']}; rescued to successful: "
        f"{s['rescued_successful']}.",
        "",
        "The final checkpoint is not an independently sampled arm. If the first candidate",
        "was valid, it was retained unchanged even when hidden detection was zero.",
        "",
        "## Return-contract concentration",
        "",
        "| Checkpoint | False → None / 30 | Other mutations / 21 |",
        "| --- | ---: | ---: |",
    ]
    for checkpoint in ("initial", "final"):
        c = s[checkpoint]
        lines.append(
            f"| {checkpoint} | {c['false_to_none_detected']}/30 | {c['other_detected']}/21 |"
        )
    lines += [
        "",
        "## Operator breakdown",
        "",
        "| Operator | Initial | Final |",
        "| --- | ---: | ---: |",
    ]
    for kind in ("return", "constant", "boolean", "comparison", "arithmetic"):
        cells = []
        for checkpoint in ("initial", "final"):
            c = result["by_operator"][checkpoint][kind]
            cells.append(f"{c['detected']}/{c['scheduled']}" if c["scheduled"] else "Not measured")
        lines.append(f"| {kind} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "## Repetition outcomes",
        "",
        "| Function | Initial successes / 3 | Final successes / 3 |",
        "| --- | ---: | ---: |",
    ]
    for case in manifest["cases"]:
        lines.append(
            f"| {case['identity']} | "
            f"{s['initial']['per_target_successes'][case['id']]}/3 | "
            f"{s['final']['per_target_successes'][case['id']]}/3 |"
        )
    lines += [
        "",
        "## Usage",
        "",
        "| Phase | Calls | Prompt tokens | Completion tokens | Reference USD |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for phase, u in result["usage"].items():
        lines.append(
            f"| {phase} | {u['calls']} | {u['prompt_tokens']} | "
            f"{u['completion_tokens']} | ${u['uncached_public_reference_usd']:.4f} |"
        )
    lines += [
        "",
        "Reference costs use public Azure rates without cache discounts, not an invoice.",
        "",
        "## Scope and interpretation",
        "",
        "These are 30 target-repetitions, not 30 independent functions. Seven functions",
        "share one module; ten of seventeen distinct mutations are False-to-None.",
        "This is a narrow development-set mechanism check, not broad test-quality accuracy.",
        "It measures bounded continuation after failure, not feedback versus equal-budget",
        "resampling, and cannot attribute gains to feedback separately from added compute.",
        "No statistical significance, production reliability, or real-bug detection claim.",
        "Public-code pretraining exposure and semantic equivalence remain unmeasured.",
        "",
        "[Protocol](METHOD.md) · "
        "[Full records](results.json)",
        "",
    ]
    return "\n".join(lines)



def main():
    parser = argparse.ArgumentParser(
        description="Recompute saved evaluation metrics; no model calls."
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/dataset.json")
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results.json")
    args = parser.parse_args()
    manifest = json.loads(args.dataset.read_text())
    study = json.loads(args.results.read_text())
    result = report(manifest, study)
    for key in ("summary", "by_operator", "usage"):
        if result[key] != study[key]:
            raise ValueError(f"Saved {key} does not recompute")
    print(json.dumps({key: result[key] for key in ("summary", "by_operator", "usage")}, indent=2))


if __name__ == "__main__":
    main()
