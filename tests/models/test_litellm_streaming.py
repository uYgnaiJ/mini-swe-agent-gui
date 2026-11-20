from unittest.mock import patch

from litellm.types.utils import ModelResponse, StreamingChoices

from minisweagent.models.litellm_model import LitellmModel


def _make_chunk(text: str | None = None, tool_arguments: str | None = None) -> ModelResponse:
    delta = {"role": "assistant", "content": text}
    if tool_arguments is not None:
        delta["tool_calls"] = [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": tool_arguments},
            }
        ]
    return ModelResponse(
        id="chatcmpl-test",
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=None)],
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )


def test_stream_query_emits_deltas_and_returns_parsed_message():
    events: list[dict] = []
    chunks = [
        _make_chunk("I will "),
        _make_chunk("run it."),
        _make_chunk(tool_arguments='{"command": "echo hi"}'),
    ]
    model = LitellmModel(model_name="gpt-4o-mini")

    with (
        patch("litellm.completion", return_value=iter(chunks)),
        patch("litellm.cost_calculator.completion_cost", return_value=0.001),
    ):
        message = model.stream_query([{"role": "user", "content": "test"}], events.append)

    assert [event["type"] for event in events] == [
        "assistant_start",
        "assistant_delta",
        "assistant_delta",
        "tool_call_delta",
        "assistant_end",
    ]
    assert "".join(event["text"] for event in events if event["type"] == "assistant_delta") == "I will run it."
    assert message["extra"]["actions"] == [{"command": "echo hi", "tool_call_id": "call_1"}]


def test_stream_query_emits_end_even_when_provider_fails():
    events: list[dict] = []
    model = LitellmModel(model_name="gpt-4o-mini")
    model.abort_exceptions = [RuntimeError]

    with patch("litellm.completion", side_effect=RuntimeError("provider failed")):
        try:
            model.stream_query([{"role": "user", "content": "test"}], events.append)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected provider error to propagate")

    assert events == [{"type": "assistant_start"}, {"type": "assistant_end"}]


def test_stream_query_falls_back_for_non_streaming_models():
    events: list[dict] = []
    model = LitellmModel(model_name="gpt-4o-mini")
    model.supports_streaming = False
    message = {"role": "assistant", "content": "Final answer", "extra": {"actions": [], "cost": 0.0}}

    with patch.object(model, "query", return_value=message):
        result = model.stream_query([{"role": "user", "content": "test"}], events.append)

    assert result is message
    assert events == [
        {"type": "assistant_start"},
        {"type": "assistant_delta", "text": "Final answer"},
        {"type": "assistant_end"},
    ]
