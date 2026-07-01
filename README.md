# BranchProof

BranchProof generates pytest patches for **a surviving mutation shown to the
model**. It targets small deterministic Python utilities, verifies candidate
tests by execution and exports patches; it does not discover arbitrary bugs,
edit production code, or prove correctness.

## Workflow

1. Run the existing suite and screen bounded single-line mutations.
2. Let the model inspect source, choose inputs and write a new test file.
3. Require original-code passes, behavioral mutant failures, suite preservation,
   test-only writes and repeated passes **and repeated mutant failures**.
4. Independently replay accepted patches in a fresh workspace and export them
   with execution traces and token accounting.

Execution uses disposable Docker containers with no network or host mounts.
Containment is not proof that adversarial tests cannot manipulate evidence.
Patches are verified independently, not as a combined batch. Bounded retries
are supported; the blind comparison below measures them separately.

## Install and run

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-build-isolation --no-deps -e .
docker build -t hardener-sandbox:tests-v1 docker
branchproof strengthen --repo /path/to/clean-repo --module package/utility.py --screen-only
```

The existing Docker image tag is retained for compatibility; the product,
distribution, Python import package and command are named **branchproof**.
For generation, export the Azure variables in [.env.example](.env.example) and
omit `--screen-only`. The CLI does not load `.env` implicitly. Only share code
you are authorized to send to the provider. Never commit credentials.

Supported dependencies are those in the pinned sandbox image; there is no
automatic installation or fallback to host execution. Input must be a clean
committed Git repository. An unkillable/equivalent mutation may yield no test.

## Evaluation

The [shared-candidate evaluation](benchmarks/RESULTS.md) uses GPT-5 mini
on ten real functions across three repositories, three runs each, with seventeen
hidden screened mutations. The one-shot baseline is the agent's exact first
candidate; only original-code rejection permits feedback-driven revision.
The model receives no mutation or hidden grading feedback.

| Metric | Initial candidate | After bounded feedback revisions |
| --- | ---: | ---: |
| Patches detecting at least one hidden mutation | 16/30 | 20/30 |
| Hidden detections across target-repetitions | 23/51 | 30/51 |
| Valid patches | 21/30 | 27/30 |
| Model decisions, cumulative | 30 | 36 |

Six invalid first candidates became valid; four then detected hidden changes.
Three initial-action failures remain failures. Five functions succeeded in all
three final runs; two never succeeded. Thirty repetitions are not thirty independent
functions. Seven functions share one module, and ten of seventeen mutations test
the False-versus-None return contract. This is not general accuracy or proof that
feedback beats equal-cost resampling. Total uncached public Azure reference cost
was about $0.297, not a verified invoice.

This uses a separate blind-generation path; the public `strengthen` command above
still explicitly shows its target mutation. Neither mode's results are transferred
to the other. The primary blind metric is patches that detect at least one hidden
survivor, not merely test files that pass.

See [evaluation details](docs/EVALUATION.md) for the grading rules and how to
recompute the published numbers without model calls.

## Development

```sh
pytest -q
ruff check src tests examples scripts
python examples/strengthen_demo.py
```

See [architecture](docs/ARCHITECTURE.md), [security boundaries](SECURITY.md)
and [release checks](docs/RELEASE.md).
