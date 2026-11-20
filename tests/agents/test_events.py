import json
from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output


def _config() -> dict:
    config = yaml.safe_load((Path("src/minisweagent/config/mini.yaml")).read_text())["agent"]
    return {**config, "cost_limit": 5.0}


def test_agent_emits_streaming_and_action_events(tmp_path):
    submit_command = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho result"
    outputs = [
        make_toolcall_output(
            "First step",
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "echo hello"}'},
                }
            ],
            [{"command": "echo hello", "tool_call_id": "call_1"}],
        ),
        make_toolcall_output(
            "Done",
            [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": submit_command}),
                    },
                }
            ],
            [{"command": submit_command, "tool_call_id": "call_2"}],
        ),
    ]
    events: list[dict] = []
    agent = DefaultAgent(
        model=DeterministicToolcallModel(outputs=outputs),
        env=LocalEnvironment(cwd=str(tmp_path)),
        **_config(),
    )
    agent.set_event_callback(events.append)

    agent.run("test task")

    event_types = [event["type"] for event in events]
    assert "run_start" in event_types
    assert "assistant_message" in event_types
    assert "action_start" in event_types
    assert "action_output" in event_types
    assert "observation" in event_types
    assert event_types[-1] == "done"
    assert events[-1]["exit_status"] == "Submitted"
    assert events[-1]["submission"] == "result\n"
