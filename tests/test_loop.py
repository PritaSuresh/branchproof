"""Runs the whole loop with fixed replies instead of a model."""

from branchproof.agent import AgentAction, ProviderError, ScriptedProvider
from branchproof.loop import harden_mutant
from branchproof.mutation import discover_in_file
from branchproof.workspace import worktree

SCOPED = ["tests/test_bank.py"]

MISSES = """from bank import can_withdraw


def test_below_balance():
    assert can_withdraw(100, 1) is True
"""

CATCHES = """from bank import can_withdraw


def test_exact_balance_is_allowed():
    assert can_withdraw(100, 100) is True
"""


def boundary_mutant(root):
    return next(
        m for m in discover_in_file(root, "bank.py") if m.before == ">=" and m.after == ">"
    )


def edit(content):
    return AgentAction(
        tool="edit_test", args={"path": "tests/test_hardened.py", "content": content}
    )


def test_a_good_test_is_accepted_first_time(sample_repo, env):
    provider = ScriptedProvider([edit(CATCHES)])
    with worktree(sample_repo) as space:
        run = harden_mutant(
            space, env, boundary_mutant(space.root), provider, SCOPED, stability_runs=2
        )
    assert run.status == "success"
    assert len(run.attempts) == 1
    assert run.attempts[0].accepted


def test_known_usage_bad_action_is_repaired_within_same_call_budget(sample_repo, env):
    class MalformedThenValid:
        calls = 0

        def next_action(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("The model returned no tool name.", response='{"test":"x"}',
                                    usage_known=True, prompt_tokens=10, completion_tokens=5,
                                    recoverable=True)
            assert any('"test":"x"' in m["content"] for m in messages)
            assert "executable action" in messages[-1]["content"]
            return edit(CATCHES), 20, 10

    provider = MalformedThenValid()
    with worktree(sample_repo) as space:
        run = harden_mutant(space, env, boundary_mutant(space.root), provider, SCOPED,
                            max_attempts=1, stability_runs=1)
    assert run.status == "success"
    assert provider.calls == 2
    assert run.prompt_tokens == 30
    assert run.completion_tokens == 15


def test_unknown_usage_errors_are_never_retried(sample_repo, env):
    class Lost:
        calls = 0

        def next_action(self, messages):
            self.calls += 1
            raise ProviderError("lost response", recoverable=True, usage_known=False)

    provider = Lost()
    with worktree(sample_repo) as space:
        run = harden_mutant(space, env, boundary_mutant(space.root), provider, SCOPED,
                            max_attempts=3)
    assert provider.calls == 1
    assert run.status == "error"


def test_a_weak_test_is_rejected_then_repaired(sample_repo, env):
    provider = ScriptedProvider([edit(MISSES), edit(CATCHES)])
    with worktree(sample_repo) as space:
        run = harden_mutant(
            space, env, boundary_mutant(space.root), provider, SCOPED, stability_runs=2
        )
    assert run.status == "success"
    assert len(run.attempts) == 2
    assert not run.attempts[0].accepted
    assert "also passes with the change applied" in run.attempts[0].feedback
    assert run.attempts[1].accepted


def test_retry_retains_reads_candidate_and_explicit_removal_notice(sample_repo, env):
    class InspectingProvider:
        def __init__(self):
            self.calls = 0

        def next_action(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AgentAction(tool="read_file", args={"path": "bank.py"}), 1, 1
            if self.calls == 2:
                return edit(MISSES), 1, 1
            history = "\n".join(m["content"] for m in messages)
            assert "def fee" in history  # observation from the first attempt
            assert "test_below_balance" in history  # rejected candidate
            assert "candidate file was removed" in history
            assert "also passes with the change applied" in history
            return edit(CATCHES), 1, 1

    provider = InspectingProvider()
    with worktree(sample_repo) as space:
        run = harden_mutant(space, env, boundary_mutant(space.root), provider, SCOPED,
                            stability_runs=1)
    assert run.status == "success"
    assert provider.calls == 3


def test_giving_up_after_the_attempt_budget(sample_repo, env):
    provider = ScriptedProvider([edit(MISSES), edit(MISSES + "\n"), edit(MISSES + "\n\n")])
    with worktree(sample_repo) as space:
        run = harden_mutant(
            space, env, boundary_mutant(space.root), provider, SCOPED,
            max_attempts=3, stability_runs=2,
        )
    assert run.status == "failed"
    assert len(run.attempts) == 3
    assert run.failure_class == "mutant survived"


def test_the_same_test_twice_is_not_run_twice(sample_repo, env):
    provider = ScriptedProvider([edit(MISSES), edit(MISSES), edit(CATCHES)])
    with worktree(sample_repo) as space:
        run = harden_mutant(
            space, env, boundary_mutant(space.root), provider, SCOPED, stability_runs=2
        )
    repeated = [a for a in run.attempts if a.reason == "repeated the same test"]
    assert repeated, "sending the same file again should be caught"
    assert run.status == "success"


def test_a_rejected_test_file_is_removed(sample_repo, env):
    provider = ScriptedProvider([edit(MISSES)])
    with worktree(sample_repo, keep=False) as space:
        harden_mutant(
            space, env, boundary_mutant(space.root), provider, SCOPED,
            max_attempts=1, stability_runs=2,
        )
        assert not (space.root / "tests" / "test_hardened.py").exists()


def test_the_agent_can_stop(sample_repo, env):
    stop = AgentAction(tool="finish", args={"reason": "no way to test this"})
    provider = ScriptedProvider([stop])
    with worktree(sample_repo) as space:
        run = harden_mutant(space, env, boundary_mutant(space.root), provider, SCOPED)
    assert run.status == "failed"
    assert run.failure_class == "agent gave up"
