from types import SimpleNamespace

import pytest

from branchproof.agent import AzureProvider, ProviderError, model_configuration


@pytest.mark.parametrize("content", [None, '{"tool":', '{"args": {}}'])
def test_unusable_azure_response_keeps_usage_and_model_output(content):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=123, completion_tokens=45),
    )
    provider = AzureProvider.__new__(AzureProvider)
    provider.deployment = "test"
    provider.configuration = model_configuration("test")
    provider.max_completion_tokens = 4096
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: response,
    )))
    with pytest.raises(ProviderError) as caught:
        provider.next_action([])
    assert caught.value.prompt_tokens == 123
    assert caught.value.completion_tokens == 45
    assert caught.value.usage_known
    assert caught.value.response == (content or "")
