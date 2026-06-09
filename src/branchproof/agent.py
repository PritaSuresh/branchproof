"""Talks to the model and turns its reply into one action."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, ValidationError

SYSTEM_PROMPT = """You write pytest tests for python code.

You are shown one line that was changed and a note that the existing tests still
pass with that change. This may indicate a missing assertion, but the change may
also be behaviorally equivalent or unreachable. Your job is to write a meaningful
test that would fail if someone made an observable faulty change for real.

Rules:
- Only write test files. Never change the code under test.
- The test must pass on the real code. If you are unsure what a function returns,
  read it or run something first rather than guessing.
- Pick inputs where the real version and the changed version must give different
  answers. A test that passes on both is no use.
- Assert behavior by calling the code under test. Do not detect the mutation by
  reading implementation text, bytecode, mutation markers or coverage artifacts.
- Keep it small and readable. Plain asserts, no mocking unless there is no way around it.
- Follow the style of the tests already in the file.

Reply with JSON only, in this shape:
{"tool": "read_file", "args": {"path": "..."}, "note": "why"}

Tools:
  read_file    args: path, and optionally start and end line numbers
  search_code  args: query
  edit_test    args: path (a test file), content (the whole file)
  run_tests    args: path (optional)
  finish       args: reason

Look at what you need, then use edit_test. Do not use finish unless the behaviour
genuinely cannot be reached from a test."""


class AgentAction(BaseModel):
    tool: str = Field(default="finish")
    args: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


def action_error(action: AgentAction) -> str | None:
    """Validate arguments before calling a tool, without coercing arbitrary JSON."""
    specs = {
        "list_files": ({}, {"prefix": str}),
        "read_file": ({"path": str}, {"start": int, "end": int}),
        "search_code": ({"query": str}, {"path": str}),
        "replace_in_file": ({"path": str, "old": str, "new": str}, {}),
        "replace_lines": ({"path": str, "start": int, "end": int, "content": str}, {}),
        "edit_file": ({"path": str, "content": str}, {}),
        "edit_test": ({"path": str, "content": str}, {}),
        "run_tests": ({}, {"path": str}),
        "finish": ({}, {"reason": str, "approved": bool}),
    }
    if action.tool not in specs:
        return f"Unknown tool: {action.tool}"
    required, optional = specs[action.tool]
    for key in required:
        if key not in action.args:
            return f"{action.tool} requires {key}."
    valid_keys = sorted(set(required) | set(optional))
    for key, value in action.args.items():
        expected = required.get(key, optional.get(key))
        if expected is None:
            return f"Unknown argument {key} for {action.tool}. Valid: {', '.join(valid_keys)}."
        if value is None and key in optional:
            continue
        if type(value) is not expected:
            return f"{key} must be {expected.__name__}."
    return None


class Provider(Protocol):
    def next_action(
        self, messages: list[dict[str, str]], system: str | None = None
    ) -> tuple[AgentAction, int, int]: ...


def parse_action_response(raw: str) -> tuple[AgentAction, str]:
    """Accept canonical actions or one unambiguous, validated named-tool object.

    No fuzzy key guessing, multiple calls or relaxed write policies. Some models
    encode {"edit_test": {"path": ..., "content": ...}} despite the canonical
    prompt. This is a wire encoding difference, not a different permission.
    """
    payload = json.loads(raw)
    names = {"read_file", "search_code", "edit_test", "run_tests", "finish"}
    if isinstance(payload, dict) and len(payload) == 1:
        name = next(iter(payload))
        if name in names and isinstance(payload[name], dict):
            action = AgentAction(tool=name, args=payload[name])
            error = action_error(action)
            if error:
                raise ValueError(error)
            return action, "named-tool-key"
    return AgentAction.model_validate(payload), "canonical"


class ProviderError(RuntimeError):
    """The model did not return a usable action; this is not an agent conclusion."""

    def __init__(self, message: str, *, prompt_tokens: int = 0, completion_tokens: int = 0,
                 usage_known: bool = False, response: str = "", recoverable: bool = False,
                 kind: str = "provider"):
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.usage_known = usage_known
        # Only model output belongs here, never SDK exception/request bodies.
        self.response = response
        # The model answered, it just answered in a shape the loop cannot execute.
        # Asking again is worth one call. A refusal or missing usage is not this.
        self.recoverable = recoverable
        self.kind = kind


def model_configuration(deployment: str, model: str | None = None,
                        reasoning_effort: str | None = None,
                        max_completion_tokens: int | None = None) -> dict:
    """Resolve request settings locally; an Azure deployment name may be an alias.

    Only GPT-5 mini is migrated here. Other GPT-5 variants must not accidentally
    inherit this profile. Unspecified legacy aliases retain the original behavior.
    """
    identity = model or deployment
    reasoning = identity in {"gpt-5-mini", "gpt-5-mini-2025-08-07"}
    if identity.startswith("gpt-5") and not reasoning:
        raise ValueError("Only gpt-5-mini has a tested GPT-5 request profile.")
    if model and not reasoning and model not in {
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14",
    }:
        raise ValueError("Unsupported underlying model; choose a tested request profile.")
    cap = max_completion_tokens if max_completion_tokens is not None else (
        16384 if reasoning else 4096)
    if type(cap) is not int or not 1 <= cap <= (128000 if reasoning else 32768):
        raise ValueError("Completion token cap is outside the supported model range.")
    effort = (reasoning_effort or "low") if reasoning else None
    if reasoning and effort not in {"minimal", "low", "medium", "high"}:
        raise ValueError("GPT-5 mini reasoning effort must be minimal, low, medium or high.")
    if not reasoning and reasoning_effort is not None:
        raise ValueError("Reasoning effort requires the GPT-5 mini profile.")
    return {
        "deployment": deployment, "configured_model": model or (identity if reasoning else None),
        "api_style": "azure-v1" if reasoning else "azure-versioned",
        "reasoning_effort": effort, "max_completion_tokens": cap,
        "temperature": None if reasoning else 0,
        "request_timeout_seconds": 120 if reasoning else 60,
        "sdk_max_retries": 2,
    }


class AzureProvider:
    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str, *,
                 model: str | None = None, reasoning_effort: str | None = None,
                 max_completion_tokens: int | None = None):
        from openai import AzureOpenAI, OpenAI

        self.configuration = model_configuration(
            deployment, model, reasoning_effort, max_completion_tokens,
        )
        self.deployment = deployment
        self.max_completion_tokens = self.configuration["max_completion_tokens"]
        self.last_response_metadata: dict = {}
        if self.configuration["api_style"] == "azure-v1":
            parsed = urlsplit(endpoint)
            if (parsed.scheme != "https" or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment
                    or parsed.path.rstrip("/") not in {"", "/openai/v1"}):
                raise ValueError("Use the Azure resource endpoint, not a Foundry project endpoint.")
            base_url = f"https://{parsed.netloc}/openai/v1/"
            self.client = OpenAI(base_url=base_url, api_key=api_key,
                                 timeout=120.0, max_retries=2)
        else:
            self.client = AzureOpenAI(
                azure_endpoint=endpoint, api_key=api_key, api_version=api_version,
                timeout=60.0, max_retries=2,
            )
            self.configuration["api_version"] = api_version

    def request_system(self, system: str) -> str:
        return system

    def next_action(
        self, messages: list[dict[str, str]], system: str | None = None
    ) -> tuple[AgentAction, int, int]:
        return self._complete(messages, system, {"type": "json_object"})

    def _complete(self, messages, system, response_format):
        from openai import OpenAIError

        self.last_response_metadata = {}
        parameters = ({"max_completion_tokens": self.max_completion_tokens,
                       "reasoning_effort": self.configuration["reasoning_effort"]}
                      if self.configuration["api_style"] == "azure-v1" else
                      {"max_tokens": self.max_completion_tokens, "temperature": 0})
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                **parameters,
                response_format=response_format,
                messages=[{"role": "system", "content": system or SYSTEM_PROMPT}, *messages],
            )
        except OpenAIError as error:
            # Provider responses may contain private request or account details.
            raise ProviderError(
                f"Model request failed ({type(error).__name__}); usage is unknown.",
                usage_known=False,
            ) from None
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        choice = response.choices[0] if response.choices else None
        self.last_response_metadata = {
            "returned_model": getattr(response, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            # Reasoning is a subset of completion_tokens, never added a second time.
            "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None),
                                        "reasoning_tokens", None),
            "cached_input_tokens": getattr(getattr(usage, "prompt_tokens_details", None),
                                           "cached_tokens", None),
        }
        evidence = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "usage_known": usage is not None}
        raw = (choice.message.content or "") if choice else ""
        evidence["response"] = raw
        if self.last_response_metadata["finish_reason"] == "length":
            raise ProviderError("Response hit its completion cap (including reasoning tokens).",
                                recoverable=True, **evidence)
        if (self.last_response_metadata["finish_reason"] == "content_filter"
                or (choice and getattr(choice.message, "refusal", None))):
            raise ProviderError("The model refused or content filtering blocked the action.",
                                **evidence)
        if not raw:
            evidence["response"] = ""
            raise ProviderError("The model returned no action.", recoverable=True, **evidence)
        try:
            action, encoding = parse_action_response(raw)
            self.last_response_metadata["action_encoding"] = encoding
        except (ValidationError, ValueError, TypeError) as error:
            raise ProviderError("The model returned an invalid action.",
                                recoverable=True, **evidence) from error
        if "tool" not in action.model_fields_set or not action.tool:
            raise ProviderError("The model returned no tool name.",
                                recoverable=True, **evidence)
        if usage is None:
            raise ProviderError("The model response omitted token usage.", **evidence)
        return action, prompt_tokens, completion_tokens


class ScriptedProvider:
    """Fixed replies, for tests and for trying the loop without spending money."""

    def __init__(self, actions: list[AgentAction]):
        self.actions = list(actions)
        self.index = 0

    def next_action(
        self, messages: list[dict[str, str]], system: str | None = None
    ) -> tuple[AgentAction, int, int]:
        del messages, system
        if self.index >= len(self.actions):
            return AgentAction(tool="finish", note="nothing left to do"), 0, 0
        action = self.actions[self.index]
        self.index += 1
        return action, 0, 0
