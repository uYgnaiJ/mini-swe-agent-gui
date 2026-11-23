from unittest.mock import MagicMock, patch

import pytest

import minisweagent.models.litellm_model as litellm_model_module
from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


@pytest.fixture(autouse=True)
def isolated_pricing_cache(tmp_path, monkeypatch):
    """Keep the unpriced-model cache in the test's tmp dir, not the user config."""
    monkeypatch.setattr(litellm_model_module, "_UNPRICED_CACHE_PATH", tmp_path / "unpriced_models.json")
    monkeypatch.setattr(litellm_model_module, "_unpriced_cache", None)


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


class TestLitellmModel:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_direct_answer_without_tool_call(self, mock_cost, mock_completion):
        response = _mock_litellm_response(None)
        response.choices[0].finish_reason = "stop"
        response.choices[0].message.content = "The answer is 4."
        mock_completion.return_value = response
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "What is 2 + 2?"}])
        assert result["extra"]["actions"] == []

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_finish_reason_threaded_into_format_error_template(self, mock_cost, mock_completion):
        """The response finish_reason is exposed to format_error_template via template_kwargs, so a
        config can report a max_tokens truncation instead of the misleading "no tool call" error."""
        response = _mock_litellm_response(None)
        response.choices[0].finish_reason = "length"
        mock_completion.return_value = response
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            format_error_template="{% if finish_reason == 'length' %}cut off{% else %}{{ error }}{% endif %}",
        )
        with pytest.raises(FormatError) as exc:
            model.query([{"role": "user", "content": "test"}])
        assert exc.value.messages[0]["content"] == "cut off"

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_unknown_model_records_call_without_cost(self, mock_cost, mock_completion, reset_global_stats):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.side_effect = Exception("This model isn't mapped yet")
        model = LitellmModel(model_name="openai/unknown-model", cost_tracking="ignore_errors")

        result = model.query([{"role": "user", "content": "test"}])

        assert result["extra"]["cost"] == 0.0
        assert result["extra"]["cost_known"] is False
        assert GLOBAL_MODEL_STATS.n_calls == 1
        assert GLOBAL_MODEL_STATS.cost == 0.0

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_known_model_reports_and_counts_cost(self, mock_cost, mock_completion, reset_global_stats):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001
        model = LitellmModel(model_name="openai/known-model", cost_tracking="ignore_errors")

        result = model.query([{"role": "user", "content": "test"}])

        assert result["extra"]["cost"] == 0.001
        assert result["extra"]["cost_known"] is True
        assert GLOBAL_MODEL_STATS.cost == 0.001

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_zero_cost_is_treated_as_unknown(self, mock_cost, mock_completion, reset_global_stats):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.0
        model = LitellmModel(model_name="openai/zero-cost", cost_tracking="ignore_errors")

        result = model.query([{"role": "user", "content": "test"}])

        assert result["extra"]["cost_known"] is False
        assert GLOBAL_MODEL_STATS.cost == 0.0

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_missing_cost_still_raises_in_strict_mode(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.side_effect = Exception("This model isn't mapped yet")
        model = LitellmModel(model_name="openai/unknown-model")

        with pytest.raises(RuntimeError, match="Error calculating cost"):
            model.query([{"role": "user", "content": "test"}])

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_unpriced_lookup_is_cached_across_model_instances(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.side_effect = Exception("This model isn't mapped yet")

        first = LitellmModel(
            model_name="openai/unpriced", cost_tracking="ignore_errors", model_kwargs={"api_base": "https://a"}
        )
        first.query([{"role": "user", "content": "test"}])
        second = LitellmModel(
            model_name="openai/unpriced", cost_tracking="ignore_errors", model_kwargs={"api_base": "https://a"}
        )
        result = second.query([{"role": "user", "content": "test"}])

        assert mock_cost.call_count == 1
        assert result["extra"]["cost_known"] is False

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_provider_change_invalidates_the_pricing_cache(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.side_effect = Exception("This model isn't mapped yet")

        first = LitellmModel(
            model_name="openai/unpriced", cost_tracking="ignore_errors", model_kwargs={"api_base": "https://a"}
        )
        first.query([{"role": "user", "content": "test"}])
        other_provider = LitellmModel(
            model_name="openai/unpriced", cost_tracking="ignore_errors", model_kwargs={"api_base": "https://b"}
        )
        other_provider.query([{"role": "user", "content": "test"}])

        assert mock_cost.call_count == 2

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_unrelated_cost_errors_are_not_cached(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.side_effect = Exception("response has no usage information")

        for _ in range(2):
            model = LitellmModel(
                model_name="openai/usage-missing", cost_tracking="ignore_errors", model_kwargs={"api_base": "https://a"}
            )
            model.query([{"role": "user", "content": "test"}])

        assert mock_cost.call_count == 2
