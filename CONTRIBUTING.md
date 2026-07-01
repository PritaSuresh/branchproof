# Development

Use Python 3.11+ and a dedicated virtual environment. The exact tested host
dependencies are pinned in `requirements-dev.lock`; the target execution image
has its own independent lock.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-build-isolation --no-deps -e .
ruff check src tests examples
pytest -q
```

For the real container checks:

```bash
docker build -t hardener-sandbox:py312 -t hardener-sandbox:tests-v1 docker
HARDENER_DOCKER_TESTS=1 pytest -q tests/test_sandbox.py
HARDENER_DOCKER_TESTS=1 pytest -q tests/test_strengthen.py
HARDENER_DOCKER_TESTS=1 pytest -q tests/test_test_study.py
python examples/strengthen_demo.py
```

The tests use tiny synthetic repositories and scripted model replies. Those are
implementation tests, not evidence of LLM performance. Never put provider keys
in fixtures or enable paid model experiments in ordinary pull-request CI.

## Changing agent behavior

1. Identify the failure in a saved trace.
2. Add a regression test for any deterministic defect.
3. Write the experimental hypothesis before changing prompts or workflow.
4. Keep the same frozen development cases and report actual token use, errors,
   eligibility exclusions and independent correctness.
5. Save new runs separately. Do not overwrite a failed experiment or edit its
   manifest. Do not inspect held-out answers during development.

Passing generated tests is not ground truth. Keep hidden mutations out of model
context, and do not use hidden grading results to select a candidate.

Before publishing: inspect the staged file list and scan for secrets, pick a
license, run CI on the destination repository, and review any upstream code you
intend to distribute with the appropriate license notices.
