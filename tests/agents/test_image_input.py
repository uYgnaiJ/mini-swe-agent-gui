import json

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output

PNG_BYTES = b"\x89PNG\r\n\x1a\nminimal"


def _agent(outputs, *, multimodal_regex=""):
    return DefaultAgent(
        model=DeterministicToolcallModel(outputs=outputs, multimodal_regex=multimodal_regex),
        env=LocalEnvironment(),
        system_template="system",
        instance_template="Task: {{task}}",
        step_limit=1,
        cost_limit=5.0,
    )


def _submit_output():
    command = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    return make_toolcall_output(
        "Done",
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
        [{"command": command, "tool_call_id": "call_1"}],
    )


def test_agent_attaches_image_to_first_user_message(tmp_path):
    image = tmp_path / "pixel.png"
    image.write_bytes(PNG_BYTES)
    events = []
    agent = _agent([_submit_output()])
    agent.set_event_callback(events.append)

    agent.run("describe this image", images=[str(image)])

    user_message = agent.messages[1]
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0] == {"type": "text", "text": "Task: describe this image"}
    assert user_message["content"][1]["type"] == "image_url"
    assert user_message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert events[0]["type"] == "run_start"
    assert events[0]["images"] == [user_message["content"][1]["image_url"]["url"]]


def test_agent_without_images_keeps_plain_text_content():
    agent = _agent([_submit_output()])

    agent.run("plain task")

    assert agent.messages[1]["content"] == "Task: plain task"


def test_agent_accepts_remote_image_url():
    agent = _agent([_submit_output()])

    agent.run("look", images=["https://example.com/screenshot.png"])

    assert agent.messages[1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/screenshot.png"},
    }


def test_agent_accepts_singular_image_alias():
    agent = _agent([_submit_output()])

    agent.run("look", image="https://example.com/one.png")

    assert agent.messages[1]["content"][1]["image_url"]["url"] == "https://example.com/one.png"
