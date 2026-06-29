# Test-generation evaluation

Ten existing development targets, three runs each. One initial candidate per pair;
the feedback loop may revise it only after original-code rejection. Hidden
mutations and their execution results remain unavailable during generation.

| Checkpoint | Successful / 30 | Valid / 30 | Hidden detections / 51 |
| --- | ---: | ---: | ---: |
| initial | 16/30 | 21/30 | 23/51 |
| final | 20/30 | 27/30 | 30/51 |

Invalid initial submissions: 6; rescued to valid: 6; rescued to successful: 4.

The final checkpoint is not an independently sampled arm. If the first candidate
was valid, it was retained unchanged even when hidden detection was zero.

## Return-contract concentration

| Checkpoint | False → None / 30 | Other mutations / 21 |
| --- | ---: | ---: |
| initial | 11/30 | 12/21 |
| final | 16/30 | 14/21 |

## Operator breakdown

| Operator | Initial | Final |
| --- | ---: | ---: |
| return | 14/33 | 19/33 |
| constant | 6/15 | 8/15 |
| boolean | 3/3 | 3/3 |
| comparison | Not measured | Not measured |
| arithmetic | Not measured | Not measured |

## Repetition outcomes

| Function | Initial successes / 3 | Final successes / 3 |
| --- | ---: | ---: |
| python-utils:python_utils/converters.py:scale_1024 | 3/3 | 3/3 |
| python-utils:python_utils/formatters.py:camel_to_underscore | 0/3 | 0/3 |
| python-string-utils:string_utils/validation.py:is_credit_card | 2/3 | 2/3 |
| python-string-utils:string_utils/validation.py:is_ip_v4 | 1/3 | 3/3 |
| python-string-utils:string_utils/validation.py:is_json | 1/3 | 3/3 |
| python-string-utils:string_utils/validation.py:is_palindrome | 2/3 | 2/3 |
| python-string-utils:string_utils/validation.py:is_pangram | 0/3 | 0/3 |
| python-string-utils:string_utils/validation.py:is_slug | 1/3 | 1/3 |
| python-string-utils:string_utils/validation.py:is_url | 3/3 | 3/3 |
| python-slugify:slugify/special.py:add_uppercase_char | 3/3 | 3/3 |

## Usage

| Phase | Calls | Prompt tokens | Completion tokens | Reference USD |
| --- | ---: | ---: | ---: | ---: |
| initial | 30 | 207258 | 36719 | $0.2480 |
| continuation | 6 | 54793 | 5579 | $0.0492 |

Reference costs use public Azure rates without cache discounts, not an invoice.

## Scope and interpretation

These are 30 target-repetitions, not 30 independent functions. Seven functions
share one module; ten of seventeen distinct mutations are False-to-None.
This is a narrow development-set mechanism check, not broad test-quality accuracy.
It measures bounded continuation after failure, not feedback versus equal-budget
resampling, and cannot attribute gains to feedback separately from added compute.
No statistical significance, production reliability, or real-bug detection claim.
Public-code pretraining exposure and semantic equivalence remain unmeasured.

[Protocol](METHOD.md) · [Full records](results.json)
