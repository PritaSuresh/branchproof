# Evaluation method

The evaluation measures whether generated pytest patches catch screened mutations
that the existing upstream suites miss. It uses ten functions from three pinned
open-source projects, with three generations per function and seventeen mutations.

## Task and baseline

Each task supplies the function name, complete module and relevant existing tests.
The mutation, its location and its grading results are hidden during generation.
The baseline is the exact first candidate, retained before any revision. This is a
comparison within the same pipeline, not an external test-generation tool.

Only original-code rejection permits continuation. The model sees the execution
feedback and may revise its candidate with the same conversation. At most three
submissions and five decisions are allowed. Generation stops at the first valid
candidate. An invalid first action remains a failure.

All generation finishes before hidden grading. A hidden miss never triggers a
revision. Identical candidates within a pair may reuse their grade.

## Execution and success

Validation and grading use fresh worktrees and the pinned Docker image.

- A valid patch edits only its designated test file, passes on the original
  implementation and preserves the existing suite.
- A successful patch is valid and also fails behaviorally on at least one hidden
  mutation during repeated execution.
- Collection errors, timeouts and provider failures do not count as detections.
- Every scheduled attempt stays in the denominator, including failed generation.

The primary metric is successful patches over all thirty scheduled attempts.
Secondary metrics are validity, hidden detections, per-function repetition
outcomes, mutation-kind breakdown and model usage. Repetitions are not independent
functions. False-to-None mutations are also reported separately.

## Configuration and accounting

The completed run used Azure GPT-5 mini, low reasoning, a 4,096-token completion
limit, a 120-second request timeout and no SDK retries. Task order uses the fixed
seed in `paired_eval.schedule`. The dataset records the immutable image ID,
upstream revisions and full contexts.

The reservation was 80,000 tokens per pair. Usage and cost are reported from
saved request traces. Costs use public Azure reference rates and are not a
verified invoice. Source and license records are in [provenance.json](provenance.json).

The [complete results](results.json) retain the original model outputs, feedback,
grading evidence and metadata. Repackaging filenames does not alter those records.
The [report](RESULTS.md) states the sample concentration and limits on attribution.
