"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import copy
import json
import logging
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.agents.utils.conversation import conversational_reply
from minisweagent.exceptions import (
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    RepeatedAction,
    Submitted,
    TimeExceeded,
    UserInterruption,
)
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.models.utils.token_usage import extract_token_usage
from minisweagent.utils.images import build_multimodal_content, normalize_images
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    max_repeated_actions: int = 3
    """Exit after the same action sequence repeats this many times (0 = no limit)."""
    max_empty_submission_retries: int = 1
    """Ask the model for a final answer when it submits an empty result."""
    conversational_response: str = ""
    """Override the automatic reply for greetings and other simple chat messages."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        self.event_callback: Callable[[dict[str, Any]], None] | None = None
        self._stop_requested = threading.Event()
        self._last_action_signature: tuple[str, ...] | None = None
        self._n_repeated_actions = 0
        self._n_empty_submission_retries = 0

    def set_event_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        """Register a callback for live run events (used by the GUI).

        The callback receives JSON-serializable dictionaries such as
        ``{"type": "assistant_delta", "text": "..."}``. It is safe to call this
        from a background thread.
        """
        self.event_callback = callback

    def request_stop(self) -> None:
        """Request that the agent stop after its current operation."""
        self._stop_requested.set()

    def stop(self) -> None:
        """Alias for :meth:`request_stop`."""
        self.request_stop()

    def _raise_if_stop_requested(self) -> None:
        if self._stop_requested.is_set():
            raise UserInterruption(
                {
                    "role": "exit",
                    "content": "Stopped by user",
                    "extra": {"exit_status": "Stopped", "submission": ""},
                }
            )

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event)

    @property
    def tokens(self) -> int:
        return self.token_usage["total_tokens"]

    def _record_token_usage(self, message: dict) -> dict[str, int]:
        usage = extract_token_usage(message)
        for key, value in usage.items():
            self.token_usage[key] += value
        return usage

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def _done_event(self, result: dict) -> dict:
        """Build the final event emitted for a completed run."""
        return {
            "type": "done",
            "exit_status": result.get("exit_status", ""),
            "submission": result.get("submission", ""),
            "cost": self.cost,
            "n_calls": self.n_calls,
            "prompt_tokens": self.token_usage["prompt_tokens"],
            "completion_tokens": self.token_usage["completion_tokens"],
            "total_tokens": self.token_usage["total_tokens"],
            "tokens": self.token_usage["total_tokens"],
            "token_usage": dict(self.token_usage),
        }

    def _complete_conversational_task(self, task: str) -> dict | None:
        """Answer a greeting/social message without entering the bash loop.

        The normal agent protocol requires a bash tool call on every turn. For
        messages such as "hi" that is the wrong behaviour: it makes the model
        inspect/search the repository even though there is no engineering task.
        Return ``None`` for real tasks so the normal loop runs unchanged.
        """
        reply = conversational_reply(task)
        if reply is None:
            return None
        reply = self.config.conversational_response.strip() or reply
        assistant_message = self.model.format_message(role="assistant", content=reply)
        self.add_messages(assistant_message)
        self._emit(
            {
                "type": "assistant_message",
                "content": reply,
                "actions": [],
                "cost": 0.0,
                "step": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "tokens": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens_used": 0,
                "cumulative_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "conversational": True,
            }
        )
        result = {
            "exit_status": "Submitted",
            "submission": reply,
        }
        self.add_messages(
            self.model.format_message(
                role="exit",
                content=reply,
                extra={"exit_status": "Submitted", "submission": reply, "conversational": True},
            )
        )
        self.save(self.config.output_path)
        self._emit(self._done_event(result))
        return result

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    @staticmethod
    def _normalize_run_images(images: list | None, image: object = None) -> list[str]:
        """Return model-ready URLs for the supported image argument aliases."""
        if image is not None:
            if images is None:
                images = [image]
            elif isinstance(images, list):
                images = [*images, image]
            else:
                images = [images, image]
        return normalize_images(images)

    def _reset_turn_state(self) -> None:
        """Reset per-turn bookkeeping without clearing the conversation."""
        self._last_action_signature = None
        self._n_repeated_actions = 0
        self._n_empty_submission_retries = 0
        self.n_consecutive_format_errors = 0

    def _emit_run_start(self, task: str, image_urls: list[str]) -> None:
        self._emit(
            {
                "type": "run_start",
                "task": task,
                "images": image_urls,
                "step_limit": self.config.step_limit,
                "cost_limit": self.config.cost_limit,
            }
        )

    def _run_until_finished(self) -> dict:
        """Run the agent loop until the last message is an exit message."""
        while True:
            try:
                self._raise_if_stop_requested()
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                # The call was billed before parsing failed, so query() never got to charge it.
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                usage = self._record_token_usage(e.messages[0])
                self.n_consecutive_format_errors += 1
                self._emit(
                    {
                        "type": "format_error",
                        "message": get_content_string(e.messages[0]) or str(e),
                        "step": self.n_calls,
                        "consecutive_errors": self.n_consecutive_format_errors,
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "tokens": usage["total_tokens"],
                        "usage": usage,
                        "total_prompt_tokens": self.token_usage["prompt_tokens"],
                        "total_completion_tokens": self.token_usage["completion_tokens"],
                        "total_tokens_used": self.token_usage["total_tokens"],
                        "cumulative_usage": dict(self.token_usage),
                    }
                )
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                if isinstance(e, Submitted) and self._should_retry_empty_submission(e):
                    # The exit message uses the internal "exit" role, which must not be sent back to the model.
                    self._n_empty_submission_retries += 1
                    if self.messages[-1].get("role") == "assistant":
                        self.add_messages(
                            *self.model.format_observation_messages(
                                self.messages[-1],
                                [{"output": "The submission was empty.", "returncode": 0, "exception_info": ""}],
                                self.get_template_vars(),
                            )
                        )
                    self.add_messages(
                        self.model.format_message(
                            role="user",
                            content=(
                                "You finished the run but submitted an empty result, so the user did not receive an "
                                "answer. Reply directly with a complete final answer to the original request, using the "
                                "information you gathered. Do not run the submission command until your answer is ready."
                            ),
                        )
                    )
                else:
                    self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                self._emit({"type": "error", "message": str(e), "traceback": traceback.format_exc()})
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        result = self.messages[-1].get("extra", {})
        self._emit(self._done_event(result))
        return result

    def _drop_trailing_exit_messages(self) -> None:
        """Remove internal exit messages before a follow-up turn.

        ``exit`` is an internal role that records why a run ended.  It is not
        part of the chat transcript and must not be sent to the model when the
        user adds another instruction to the same conversation.
        """
        while self.messages and self.messages[-1].get("role") == "exit":
            self.messages.pop()

    def run(self, task: str = "", images: list | None = None, image: object = None, **kwargs) -> dict:
        """Run step() until agent is finished.

        ``images`` (or the singular ``image`` alias) may contain local paths,
        http(s) URLs, data URLs, or raw bytes.  They are attached to the first
        user message so multimodal models can inspect them alongside the task
        text.
        """
        image_urls = self._normalize_run_images(images, image)
        self.extra_template_vars |= {"task": task, "images": image_urls, **kwargs}
        self.messages = []
        self._reset_turn_state()
        self._emit_run_start(task, image_urls)
        task_content = build_multimodal_content(self._render_template(self.config.instance_template), image_urls)
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=task_content),
        )
        # Short greetings should not skip over an attached image.
        conversational_result = self._complete_conversational_task(task) if not image_urls else None
        if conversational_result is not None:
            return conversational_result
        return self._run_until_finished()

    def continue_run(
        self,
        task: str = "",
        images: list | None = None,
        image: object = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        """Append a follow-up turn and keep the existing conversation context.

        Unlike :meth:`run`, this method deliberately does not clear
        ``self.messages`` or render ``instance_template`` again.  The model sees
        the previous user instructions, assistant tool calls, and observations,
        followed by the new user instruction.  This is what makes a GUI
        follow-up a continuation of the same conversation rather than a fresh
        agent run.

        ``messages`` allows a freshly constructed agent (for example after a
        GUI server restart) to be seeded with the persisted conversation
        history before the new turn starts.
        """
        image_urls = self._normalize_run_images(images, image)
        if messages is not None:
            self.messages = copy.deepcopy(messages)
        if not self.messages:
            # Nothing to continue; behave like a normal first run.  The image
            # URLs are already normalized, which normalize_images accepts.
            return self.run(task, images=image_urls, **kwargs)
        self.extra_template_vars |= {"task": task, "images": image_urls, **kwargs}
        self._reset_turn_state()
        self._drop_trailing_exit_messages()
        follow_up_text = task or "Please analyze the attached image."
        self.add_messages(
            self.model.format_message(role="user", content=build_multimodal_content(follow_up_text, image_urls))
        )
        self._emit_run_start(task, image_urls)
        # Short greetings should not skip over an attached image.
        conversational_result = self._complete_conversational_task(task) if not image_urls else None
        if conversational_result is not None:
            return conversational_result
        return self._run_until_finished()

    def _should_retry_empty_submission(self, error: Submitted) -> bool:
        if self._n_empty_submission_retries >= self.config.max_empty_submission_retries:
            return False
        if not error.messages:
            return False
        message = error.messages[-1]
        submission = message.get("extra", {}).get("submission") or get_content_string(message)
        return not str(submission or "").strip()

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        self._raise_if_stop_requested()
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        self._emit({"type": "step_start", "step": self.n_calls})
        if self.event_callback is not None and hasattr(self.model, "stream_query"):

            def forward_stream_event(event: dict[str, Any]) -> None:
                self._raise_if_stop_requested()
                event.setdefault("step", self.n_calls)
                self._emit(event)

            message = self.model.stream_query(self.messages, forward_stream_event)
        else:
            message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        usage = self._record_token_usage(message)
        self.add_messages(message)
        self._emit(
            {
                "type": "assistant_message",
                "content": get_content_string(message),
                "actions": message.get("extra", {}).get("actions", []),
                "cost": message.get("extra", {}).get("cost", 0.0),
                "step": self.n_calls,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "tokens": usage["total_tokens"],
                "usage": usage,
                "total_prompt_tokens": self.token_usage["prompt_tokens"],
                "total_completion_tokens": self.token_usage["completion_tokens"],
                "total_tokens_used": self.token_usage["total_tokens"],
                "cumulative_usage": dict(self.token_usage),
            }
        )
        self._raise_if_stop_requested()
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        actions = message.get("extra", {}).get("actions", [])
        if not actions and (content := get_content_string(message).strip()):
            self.add_messages(
                self.model.format_message(
                    role="exit",
                    content=content,
                    extra={"exit_status": "Submitted", "submission": content, "direct_response": True},
                )
            )
            return self.messages[-1:]
        signature = tuple(str(action.get("command", "")).strip() for action in actions)
        if self._last_action_signature == signature:
            self._n_repeated_actions += 1
        else:
            self._last_action_signature = signature
            self._n_repeated_actions = 1
        if 0 < self.config.max_repeated_actions <= self._n_repeated_actions:
            description = "no tool call" if not signature else repr(signature)
            self._emit(
                {
                    "type": "stalled",
                    "message": f"Stopping after {self._n_repeated_actions} identical turns: {description}",
                    "step": self.n_calls,
                }
            )
            raise RepeatedAction(
                {
                    "role": "exit",
                    "content": "RepeatedAction",
                    "extra": {
                        "exit_status": "RepeatedAction",
                        "submission": "",
                        "reason": f"Repeated the same action {self._n_repeated_actions} times",
                    },
                }
            )
        outputs = []
        for action in actions:
            self._raise_if_stop_requested()
            command = action.get("command", "")
            self._emit({"type": "action_start", "command": command, "tool_call_id": action.get("tool_call_id")})
            output = self.env.execute(action)
            outputs.append(output)
            self._emit(
                {
                    "type": "action_output",
                    "command": command,
                    "tool_call_id": action.get("tool_call_id"),
                    "output": output.get("output", ""),
                    "returncode": output.get("returncode"),
                    "exception_info": output.get("exception_info"),
                }
            )
            self._raise_if_stop_requested()
        messages = self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )
        for observation in messages:
            self._emit(
                {
                    "type": "observation",
                    "content": get_content_string(observation),
                    "returncode": observation.get("extra", {}).get("returncode"),
                }
            )
        return messages

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                    "tokens": self.token_usage["total_tokens"],
                    "token_usage": dict(self.token_usage),
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
