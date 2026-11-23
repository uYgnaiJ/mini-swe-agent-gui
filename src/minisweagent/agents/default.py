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

_PLAN_PROMPT = """You are the planning stage of a coding agent that has a single bash tool.
Decide what the latest user request needs BEFORE any tool is used.

Reply with exactly one JSON object, no markdown:
{
  "intent": "question" | "exploration" | "small_edit" | "large_edit",
  "use_tools": true | false,
  "write_files": true | false,
  "reply_directly": true | false,
  "steps": <integer between 1 and {max_steps}>,
  "plan": "<one sentence describing the next actions>"
}

Rules:
- Questions, explanations, or opinions: use_tools=false, write_files=false, reply_directly=true, steps <= 2.
- Read-only investigation of the repository: use_tools=true, write_files=false, steps 2-6.
- Code changes: write_files=true; small_edit 4-10 steps, large_edit 10-{max_steps}.
- Never plan an unbounded run. Prefer the smallest number of steps that can finish the job.
"""

_CHECKPOINT_PROMPT = """You are the checkpoint stage of a coding agent.
The agent has used its step budget. Review the work so far and decide what happens next.

Reply with exactly one JSON object, no markdown:
{
  "status": "report" | "continue",
  "summary": "<what was done and what remains, in 2-3 sentences>",
  "answer": "<final answer for the user when status=report; otherwise empty>",
  "next_steps": <integer between 1 and {max_steps} when status=continue; otherwise 0>,
  "plan": "<how to continue when status=continue; otherwise empty>"
}

Report when the task is complete, blocked, or more exploration would not help.
Continue only when concrete progress is still possible, and keep next_steps tight.
"""


def _extract_json_object(text: str) -> dict | None:
    """Return the first JSON object embedded in model output, if any."""
    value = str(text or "")
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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
    auto_budget: bool = False
    """Classify each request and set a step budget before running (the GUI enables this)."""
    min_step_budget: int = 3
    """Smallest step budget the planner may assign."""
    max_step_budget: int = 25
    """Largest step budget the planner may assign."""
    max_budget_extensions: int = 2
    """How often a checkpoint may extend the budget before the turn has to end."""
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
        self.n_calls_without_cost = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        self.event_callback: Callable[[dict[str, Any]], None] | None = None
        self._stop_requested = threading.Event()
        self._last_action_signature: tuple[str, ...] | None = None
        self._n_repeated_actions = 0
        self._n_empty_submission_retries = 0
        self._step_budget = 0
        self._budget_extensions = 0

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

    def _conversation_excerpt(self, *, limit: int = 6, chars: int = 500) -> str:
        """Compact transcript for the planner and checkpoint prompts."""
        lines = []
        for message in self.messages[-limit:]:
            role = str(message.get("role") or "")
            text = get_content_string(message).strip()
            if role in ("system", "exit") or not text:
                continue
            lines.append(f"{role}: {text[:chars]}")
        return "\n".join(lines)

    def _query_json(self, system_prompt: str, user_prompt: str) -> dict | None:
        """Best-effort JSON query used by the planner and checkpoint stages."""
        try:
            message = self.model.query(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
        except Exception:  # noqa: BLE001 - planning must never abort a run
            return None
        extra = message.get("extra", {})
        self.cost += extra.get("cost", 0.0)
        self._record_token_usage(message)
        return _extract_json_object(get_content_string(message))

    def _budget_from_plan(self, plan: dict | None) -> int:
        """Clamp a planner/checkpoint step estimate to the configured bounds."""
        try:
            steps = int(plan.get("steps", plan.get("next_steps")))
        except (AttributeError, TypeError, ValueError):
            steps = self.config.max_step_budget
        steps = max(self.config.min_step_budget, min(self.config.max_step_budget, steps))
        if plan and plan.get("reply_directly"):
            steps = min(steps, 2)
        if self.config.step_limit > 0:
            steps = min(steps, self.config.step_limit)
        return max(1, steps)

    def _append_system_note(self, note: str) -> None:
        if not self.messages or self.messages[0].get("role") != "system":
            return
        content = self.messages[0].get("content")
        if not isinstance(content, str):
            return
        # Replace the previous turn's note instead of stacking them.
        content = content.split("\n<work_plan>", 1)[0]
        self.messages[0]["content"] = content + note

    def _plan_turn(self, task: str) -> None:
        """Classify the request and set the step budget before the loop starts."""
        if not self.config.auto_budget:
            return
        plan = self._query_json(
            _PLAN_PROMPT.replace("{max_steps}", str(self.config.max_step_budget)),
            f"Recent conversation:\n{self._conversation_excerpt() or '(none)'}\n\nLatest user request:\n{task}",
        )
        budget = self._budget_from_plan(plan)
        self._step_budget = budget
        plan_text = str(plan.get("plan") or "") if plan else ""
        write_files = bool(plan and plan.get("write_files"))
        reply_directly = bool(plan and plan.get("reply_directly"))
        self._emit(
            {
                "type": "plan",
                "intent": str(plan.get("intent") or "") if plan else "",
                "plan": plan_text,
                "steps": budget,
                "write_files": write_files,
                "reply_directly": reply_directly,
                "source": "planner" if plan else "fallback",
            }
        )
        notes = [
            "Internal plan from the planning stage (not a user instruction; the user request takes precedence):",
            f"- Approach: {plan_text or 'planner unavailable'}",
            f"- Step budget: about {budget} steps.",
        ]
        if reply_directly:
            notes.append("- Answer directly without calling tools.")
        elif not write_files:
            notes.append("- Read-only: do not modify files unless the user explicitly asks for changes.")
        self._append_system_note("\n<work_plan>\n" + "\n".join(notes) + "\n</work_plan>")

    def _budget_exhausted(self) -> bool:
        return self.config.auto_budget and 0 < self._step_budget <= self.n_calls

    def _finish_from_checkpoint(self, decision: dict | None) -> None:
        answer = ""
        if decision:
            answer = str(decision.get("answer") or "").strip() or str(decision.get("summary") or "").strip()
        status = "Submitted" if answer else "LimitsExceeded"
        raise Submitted(
            self.model.format_message(
                role="exit",
                content=answer or "Stopped after reaching the step budget.",
                extra={
                    "exit_status": status,
                    "submission": answer if status == "Submitted" else "",
                    "budget_summary": True,
                },
            )
        )

    def _handle_budget_checkpoint(self) -> None:
        """Summarize the work at the budget boundary and either report or replan."""
        self._emit({"type": "budget_reached", "step": self.n_calls, "budget": self._step_budget})
        decision = self._query_json(
            _CHECKPOINT_PROMPT.replace("{max_steps}", str(self.config.max_step_budget)),
            f"Task: {self.extra_template_vars.get('task', '')}\n"
            f"Steps used: {self.n_calls} of {self._step_budget}\n"
            f"Work so far:\n{self._conversation_excerpt() or '(no messages)'}",
        )
        steps = self._budget_from_plan(decision) if decision and decision.get("status") == "continue" else 0
        if self.config.step_limit > 0:
            steps = min(steps, self.config.step_limit - self.n_calls)
        if steps <= 0 or self._budget_extensions >= self.config.max_budget_extensions:
            self._finish_from_checkpoint(decision)
        self._budget_extensions += 1
        self._step_budget = self.n_calls + steps
        self._emit(
            {
                "type": "budget_extended",
                "step": self.n_calls,
                "steps": steps,
                "budget": self._step_budget,
                "summary": str(decision.get("summary") or ""),
                "plan": str(decision.get("plan") or ""),
            }
        )

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
            "calls_without_cost": self.n_calls_without_cost,
            "prompt_tokens": self.token_usage["prompt_tokens"],
            "completion_tokens": self.token_usage["completion_tokens"],
            "total_tokens": self.token_usage["total_tokens"],
            "tokens": self.token_usage["total_tokens"],
            "token_usage": dict(self.token_usage),
        }

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
        self._step_budget = 0
        self._budget_extensions = 0
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
                if e.messages[0].get("extra", {}).get("cost_known", True) is False:
                    self.n_calls_without_cost += 1
                usage = self._record_token_usage(e.messages[0])
                self.n_consecutive_format_errors += 1
                self._emit(
                    {
                        "type": "format_error",
                        "message": get_content_string(e.messages[0]) or str(e),
                        "step": self.n_calls,
                        "cost": e.messages[0].get("extra", {}).get("cost", 0.0),
                        "cost_known": e.messages[0].get("extra", {}).get("cost_known", True),
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
        self._plan_turn(task)
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
        self._plan_turn(task)
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
        if 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if self._budget_exhausted():
            self._handle_budget_checkpoint()
        if 0 < self.config.step_limit <= self.n_calls:
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
        if message.get("extra", {}).get("cost_known", True) is False:
            self.n_calls_without_cost += 1
        usage = self._record_token_usage(message)
        self.add_messages(message)
        self._emit(
            {
                "type": "assistant_message",
                "content": get_content_string(message),
                "actions": message.get("extra", {}).get("actions", []),
                "cost": message.get("extra", {}).get("cost", 0.0),
                "cost_known": message.get("extra", {}).get("cost_known", True),
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
