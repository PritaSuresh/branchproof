# Evaluation

The completed study uses ten functions from python-utils, python-string-utils and
python-slugify, with three runs each and seventeen hidden mutations.

- [Results and limitations](../benchmarks/RESULTS.md)
- [Method](../benchmarks/METHOD.md)
- [Dataset and complete contexts](../benchmarks/dataset.json)
- [Complete run record](../benchmarks/results.json)
- [Upstream revisions, licenses and baseline checks](../benchmarks/provenance.json)

The result is 16/30 successful first candidates and 20/30 after continuation.
Six invalid submissions became valid; four of those then detected a hidden
mutation. Three invalid initial actions remain failures. All scheduled attempts
stay in the denominator.

Thirty repetitions are not thirty independent functions. Seven functions share
one module. Ten of seventeen mutations replace False with None. The comparison
does not separate the value of feedback from additional compute. Public-code
pretraining exposure and semantic equivalence are unknown.

## Check the numbers

```sh
python scripts/summarize_evaluation.py
pytest -q tests/test_paired_published_results.py
```

These commands recompute the published metrics from saved job records. They need
no credentials or model calls and do not rerun target repositories. The complete
record contains generated candidates and execution traces, so separate duplicate
proposal files are unnecessary.

Unit tests and the scripted examples check the implementation using small fixture
repositories. Those checks are separate from the model-performance evaluation.
Rerunning target execution requires the pinned upstream checkouts and image from
the dataset; the summary command does not install those dependencies.
