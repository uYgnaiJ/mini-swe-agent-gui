from minisweagent.models.utils.token_usage import extract_token_usage


def test_extracts_token_usage_from_completion_response():
    message = {
        "role": "assistant",
        "extra": {
            "response": {
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                }
            }
        },
    }
    assert extract_token_usage(message) == {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}


def test_extracts_input_output_tokens_from_responses_api_message():
    message = {
        "object": "response",
        "usage": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
    }
    assert extract_token_usage(message) == {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}


def test_extracts_direct_usage_and_missing_usage_is_zero():
    assert extract_token_usage({"extra": {"usage": {"prompt_tokens": 4, "completion_tokens": 6}}}) == {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
    }
    assert extract_token_usage({"role": "assistant", "content": "no usage"}) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
