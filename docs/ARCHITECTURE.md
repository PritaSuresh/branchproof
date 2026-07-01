# Architecture

BranchProof writes tests, not production fixes. The model proposes inputs and assertions;
Python code controls file access, execution, retries and acceptance.

## Test strengthening

1. Create a temporary Git worktree from a clean repository.
2. Run the existing suite and find covered mutations that it misses.
3. Give the model the source, relevant tests and the selected mutation.
4. Accept only test-file edits. Run the candidate on original and mutated code.
5. Return execution errors for a bounded retry, retaining the conversation.
6. Repeat the checks and replay accepted patches in a fresh worktree.
7. Export each patch separately with its trace and usage.

Available model actions include reading files, searching code, writing tests and
running tests. The model has no shell tool. A mutation that survives does not
necessarily represent a real bug or a useful test gap.

## Blind evaluation

The evaluation path gives the model a function, module and existing tests, but
not the mutation. It finishes all generation before checking hidden mutations.
Only failures on the original implementation are returned during revision.

The shared-candidate comparison retains the first submission as its baseline.
It continues from that same submission only when original-code checks reject it.
This measures the effect of bounded continuation, not superiority over an
equally expensive independent retry.

## Execution boundaries

The host owns credentials, Git operations and tool validation. Target code runs
in a disposable Docker container with copied inputs, no network or host mounts,
a read-only root filesystem, a non-root user and resource limits.

Docker reduces host exposure; it does not make arbitrary repository code safe.
Tests can interfere with pytest inside the container. Passing checks is evidence,
not proof of correctness. See [security boundaries](../SECURITY.md).

## Source map

| Files | Responsibility |
| --- | --- |
| `agent.py`, `config.py` | Azure client, model settings and action parsing |
| `tools.py`, `workspace.py` | File access, edit policies and isolated worktrees |
| `runner.py`, `sandbox.py` | Test execution and container controls |
| `mutation.py`, `coverage_map.py`, `context.py` | Mutation discovery and source context |
| `loop.py`, `feedback.py`, `verify.py` | Retry loop and acceptance checks |
| `strengthen.py`, `cli.py`, `artifacts.py` | Product workflow and exported results |
| `test_study.py`, `real_cases.py`, `blind_eval.py` | Dataset preparation and independent grading |
| `paired_eval.py` | Shared-candidate evaluation and outcome accounting |
