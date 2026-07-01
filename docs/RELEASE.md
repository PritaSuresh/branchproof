# Release checks

Before publishing a release:

- Install from the dependency lock and build a wheel.
- Run the unit tests, lint and explicit Docker checks.
- Check that the wheel contains only the BranchProof package.
- Recompute the published evaluation metrics from the saved records.
- Review the diff for credentials, private source and upstream license notices.
- Confirm the destination repository and run its CI.

The public commands are `strengthen`, `test-prepare`, `test-study` and
`test-score`. Production source is read-only in the strengthening workflow.
Accepted patches are independent proposals, not changes applied automatically.

The command shows its target mutation to the model. The completed
[paired evaluation](../benchmarks/RESULTS.md) uses a separate blind path.
Its results do not establish whole-repository bug discovery or superiority over
equal-budget retries. See [evaluation details](EVALUATION.md).
