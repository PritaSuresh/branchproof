"""Real SDK serialization and error parsing, with all HTTP handled in memory."""

import json

import httpx2 as httpx
import openai
import pytest

from branchproof.agent import AzureProvider, ProviderError
from branchproof.config import Settings


def response(content='{"tool":"finish","args":{"reason":"done"}}', finish="stop",
             usage=True):
    payload = {
        "id": "offline-fixture", "object": "chat.completion", "created": 0,
        "model": "gpt-5-mini-2025-08-07",
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": content}}],
    }
    if usage:
        payload["usage"] = {
            "prompt_tokens": 123, "completion_tokens": 100, "total_tokens": 223,
            "completion_tokens_details": {"reasoning_tokens": 80},
            "prompt_tokens_details": {"cached_tokens": 64},
        }
    return payload


@pytest.fixture
def transport(monkeypatch):
    calls = []
    reply = {"payload": response(), "status": 200}

    def handle(request):
        calls.append(request)
        payload = reply["queue"].pop(0) if "queue" in reply else reply["payload"]
        return httpx.Response(reply["status"], json=payload)

    original_openai, original_azure = openai.OpenAI, openai.AzureOpenAI
    clients = []

    def factory(cls, **kwargs):
        client = cls(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: factory(original_openai, **kwargs))
    monkeypatch.setattr(openai, "AzureOpenAI", lambda **kwargs: factory(original_azure, **kwargs))
    yield calls, reply
    for client in clients:
        client.close()


def provider(**kwargs):
    return AzureProvider("https://fixture.openai.azure.com/", "dummy-not-a-real-key",
                         "gpt-5-mini", "2024-10-21", **kwargs)


def test_gpt5_mini_serializes_supported_request_and_counts_reasoning_once(transport):
    calls, _ = transport
    agent = provider()
    action, input_tokens, output_tokens = agent.next_action([
        {"role": "user", "content": "Investigate this bug."}], system="Return JSON only.")
    request = calls[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://fixture.openai.azure.com/openai/v1/chat/completions"
    assert body["model"] == "gpt-5-mini"
    assert body["max_completion_tokens"] == 16384
    assert body["reasoning_effort"] == "low"
    assert body["response_format"] == {"type": "json_object"}
    assert "temperature" not in body and "max_tokens" not in body
    assert body["messages"][0]["content"] == "Return JSON only."
    assert action.tool == "finish"
    # Reasoning tokens are included in completion tokens.
    assert (input_tokens, output_tokens) == (123, 100)
    assert agent.last_response_metadata == {
        "returned_model": "gpt-5-mini-2025-08-07", "finish_reason": "stop",
        "reasoning_tokens": 80, "cached_input_tokens": 64,
        "action_encoding": "canonical",
    }
    assert agent.client.timeout == 120
    assert agent.client.max_retries == 2


def test_deployment_alias_uses_explicit_underlying_model(transport):
    calls, _ = transport
    agent = AzureProvider("https://fixture.services.ai.azure.com/openai/v1/", "dummy",
                          "my-custom-name", "unused", model="gpt-5-mini",
                          reasoning_effort="medium", max_completion_tokens=20000)
    agent.next_action([])
    body = json.loads(calls[0].content)
    assert body["model"] == "my-custom-name"
    assert body["reasoning_effort"] == "medium"
    assert body["max_completion_tokens"] == 20000
    assert str(calls[0].url).count("/openai/v1/") == 1


def test_gpt41_legacy_request_is_preserved(transport):
    calls, _ = transport
    agent = AzureProvider("https://fixture.openai.azure.com/", "dummy", "gpt-4.1-mini",
                          "2024-10-21")
    agent.next_action([])
    body = json.loads(calls[0].content)
    assert body["temperature"] == 0 and body["max_tokens"] == 4096
    assert "max_completion_tokens" not in body and "reasoning_effort" not in body
    assert "deployments/gpt-4.1-mini/chat/completions" in str(calls[0].url)
    assert calls[0].url.params["api-version"] == "2024-10-21"
    assert agent.client.timeout == 60


@pytest.mark.parametrize("content,finish,match", [
    (None, "length", "completion cap"),
    ('{"tool":"finish"}', "length", "completion cap"),
    (None, "content_filter", "filtering"),
    (None, "stop", "no action"),
    ('{"tool":', "stop", "invalid action"),
])
def test_unusable_response_keeps_all_billed_tokens(transport, content, finish, match):
    _, reply = transport
    reply["payload"] = response(content, finish)
    agent = provider()
    with pytest.raises(ProviderError, match=match) as caught:
        agent.next_action([])
    assert caught.value.usage_known
    assert (caught.value.prompt_tokens, caught.value.completion_tokens) == (123, 100)
    assert caught.value.response == (content or "")
    assert agent.last_response_metadata["reasoning_tokens"] == 80


def test_missing_usage_stays_unknown(transport):
    _, reply = transport
    reply["payload"] = response(usage=False)
    with pytest.raises(ProviderError, match="omitted token usage") as caught:
        provider().next_action([])
    assert not caught.value.usage_known


@pytest.mark.parametrize("kwargs", [
    {"reasoning_effort": "none"}, {"reasoning_effort": "xhigh"},
    {"max_completion_tokens": 0}, {"max_completion_tokens": True},
    {"max_completion_tokens": 128001}, {"model": "gpt-5.1"},
    {"model": "gpt-4.1-mini", "reasoning_effort": "low"},
])
def test_invalid_profiles_fail_before_client_creation(transport, kwargs):
    with pytest.raises(ValueError):
        provider(**kwargs)
    assert not transport[0]


def test_project_endpoint_rejected_before_any_request(transport):
    with pytest.raises(ValueError, match="project endpoint"):
        AzureProvider("https://fixture.services.ai.azure.com/api/projects/project", "dummy",
                      "gpt-5-mini", "unused")
    assert not transport[0]


def test_environment_settings_reach_cli_provider(transport, monkeypatch):
    from branchproof.cli import _provider

    values = {"AZURE_OPENAI_ENDPOINT": "https://fixture.openai.azure.com/",
              "AZURE_OPENAI_API_KEY": "dummy", "AZURE_OPENAI_DEPLOYMENT": "pilot-alias",
              "AZURE_OPENAI_MODEL": "gpt-5-mini", "HARDENER_REASONING_EFFORT": "medium",
              "HARDENER_MAX_COMPLETION_TOKENS": "20000"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    agent = _provider(Settings.from_env())
    assert agent.configuration["configured_model"] == "gpt-5-mini"
    assert agent.configuration["deployment"] == "pilot-alias"
    assert agent.configuration["reasoning_effort"] == "medium"
    assert agent.max_completion_tokens == 20000
    assert not transport[0]


def test_provider_error_does_not_expose_response_body(transport):
    _, reply = transport
    agent = provider()
    agent.next_action([])
    reply.update(status=401, payload={"error": {"message": "DUMMY_SECRET_DO_NOT_LOG"}})
    with pytest.raises(ProviderError) as caught:
        agent.next_action([])
    assert "DUMMY_SECRET_DO_NOT_LOG" not in str(caught.value)
    assert not caught.value.usage_known
    assert agent.last_response_metadata == {}
