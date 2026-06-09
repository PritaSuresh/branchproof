import pytest

from branchproof.agent import parse_action_response


def test_unambiguous_named_tool_is_normalized_without_changing_permissions():
    action, encoding = parse_action_response(
        '{"edit_test":{"path":"tests/test_new.py","content":"def test_ok(): pass"}}')
    assert action.tool == "edit_test"
    assert action.args["path"] == "tests/test_new.py"
    assert encoding == "named-tool-key"


@pytest.mark.parametrize("raw", [
    '{"edit_test":{"file_path":"x.py","content":"x"}}',
    '{"edit_test":{"path":12,"content":"x"}}',
])
def test_named_tool_args_are_not_guessed_or_coerced(raw):
    with pytest.raises(ValueError):
        parse_action_response(raw)


def test_multiple_tool_keys_are_not_executed_as_a_batch():
    action, encoding = parse_action_response('{"read_file":{},"edit_test":{}}')
    assert "tool" not in action.model_fields_set
    assert encoding == "canonical"  # provider rejects missing explicit tool, as before
